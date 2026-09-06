"""Whale Watcher — Polymarket WebSocket large trade tracker.

Connects to Polymarket's live WebSocket feed and tracks trades ≥ WHALE_MIN_TRADE_USD.
When a market receives whale activity, its api_signal gets a boost in trade selection,
increasing our conviction to copy that direction.

WebSocket:
    URL: wss://ws-live-data.polymarket.com
    Topic: activity (orders_matched events)

Bootstrap on startup: fetch last 2h of trades via REST to prime the cache.

Public API:
    start_whale_ws()             — start background WebSocket thread
    get_whale_signal(market_title) — returns 0.5-1.0 (>0.7 = whale activity)
    get_recent_whales(n=10)      — last N large trades for dashboard display
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests

import config

log = logging.getLogger("polybot.whale_watcher")

# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

# Rolling window of whale trades: {market_title: deque of trade dicts}
_whale_trades: dict[str, deque] = {}
_whale_lock = threading.Lock()
_ws_running = False

# Recent whales flat list for dashboard
_recent_whales: deque = deque(maxlen=200)


def _now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def _trim_old(market_title: str):
    """Remove trades outside the WHALE_LOOKBACK_HOURS window.

    Trims by slug key (exact match) or by normalized title substring.
    """
    cutoff = _now_ts() - config.WHALE_LOOKBACK_HOURS * 3600
    needle = market_title.strip().lower()
    with _whale_lock:
        keys_to_check = []
        for slug_key in list(_whale_trades.keys()):
            slug_lower = slug_key.lower()
            if (slug_lower == needle
                    or needle in slug_lower
                    or slug_lower in needle
                    or slug_key == market_title):
                keys_to_check.append(slug_key)
        for key in keys_to_check:
            q = _whale_trades[key]
            while q and q[0].get("_ts", 0) < cutoff:
                q.popleft()
            if not q:
                del _whale_trades[key]


def _record_trade(trade: dict):
    """Add a trade to the in-memory store if it's large enough."""
    try:
        size = float(trade.get("size", 0) or trade.get("amount", 0) or 0)
        if size < config.WHALE_MIN_TRADE_USD:
            return

        market_slug = (
            trade.get("market_slug", "")
            or trade.get("marketId", "")
            or trade.get("conditionId", "")
            or "unknown"
        )
        # Store the human-readable title too (from trade payload or slug fallback)
        raw_title = trade.get("title", "") or trade.get("market_title", "") or market_slug
        entry = {
            "_ts":         _now_ts(),
            "size_usd":    size,
            "market":      market_slug,
            "normalized_title": raw_title.strip().lower(),
            "outcome":     trade.get("outcome", ""),
            "side":        trade.get("side", ""),
            "price":       float(trade.get("price", 0) or 0),
            "timestamp":   datetime.now(timezone.utc).isoformat(),
        }

        with _whale_lock:
            if market_slug not in _whale_trades:
                _whale_trades[market_slug] = deque(maxlen=100)
            _whale_trades[market_slug].append(entry)
            _recent_whales.appendleft(entry)

        log.debug(
            "Whale trade: $%.0f on '%s' %s @ %.3f",
            size, market_slug[:40], entry["outcome"], entry["price"]
        )
    except Exception as e:
        log.debug("Error recording trade: %s", e)


# ---------------------------------------------------------------------------
# Bootstrap: fetch recent large trades via REST
# ---------------------------------------------------------------------------

# Public data-api endpoint: no auth, returns most-recent trades with size filter.
# Replaces old CLOB /trades bootstrap which needed auth and kept 401'ing.
_DATA_API = "https://data-api.polymarket.com"


def _bootstrap_recent_trades():
    """Fetch the last WHALE_LOOKBACK_HOURS of large trades from public data-api.

    Uses https://data-api.polymarket.com/trades (public, no auth) — the same
    endpoint tracker.py already hits successfully. Filters server-side via
    `filterAmount` so we only pull whale-sized trades.
    """
    try:
        lookback = config.WHALE_LOOKBACK_HOURS
        cutoff = _now_ts() - lookback * 3600
        min_usd = int(config.WHALE_MIN_TRADE_USD)
        r = requests.get(
            f"{_DATA_API}/trades",
            params={"limit": 500, "filterAmount": min_usd},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        trades = data if isinstance(data, list) else data.get("data", [])

        count = 0
        for t in trades:
            # data-api timestamps are unix seconds
            try:
                ts = float(t.get("timestamp", 0) or 0)
            except (TypeError, ValueError):
                ts = 0
            if ts and ts < cutoff:
                continue
            size = float(t.get("size", 0) or 0)
            if size >= config.WHALE_MIN_TRADE_USD:
                # data-api uses "slug" for market_slug; normalize for _record_trade
                enriched = dict(t)
                if "slug" in t and "market_slug" not in t:
                    enriched["market_slug"] = t.get("slug") or t.get("eventSlug") or ""
                _record_trade(enriched)
                count += 1

        log.info("Whale bootstrap: %d whale trades loaded from last %dh (data-api public)",
                 count, lookback)
    except Exception as e:
        log.warning("Whale bootstrap failed: %s — WS will fill cache", e)


def _run_http_poll_fallback():
    """Background thread: poll public data-api for trades every 60s.

    Graceful-degrade path when WS is unreliable. Same signal, slower cadence.
    Runs only when WHALE_HTTP_POLL_ENABLED is True.
    """
    log.info("Whale HTTP-poll fallback started (60s interval)")
    min_usd = int(config.WHALE_MIN_TRADE_USD)
    seen_ids = deque(maxlen=2000)   # dedup trades across polls
    while True:
        try:
            r = requests.get(
                f"{_DATA_API}/trades",
                params={"limit": 100, "filterAmount": min_usd},
                timeout=15,
            )
            r.raise_for_status()
            trades = r.json() if isinstance(r.json(), list) else r.json().get("data", [])
            new_count = 0
            for t in trades:
                # Dedup: timestamp+price+slug+size ~= unique-enough key
                tid = f"{t.get('timestamp','')}:{t.get('slug','')}:{t.get('price','')}:{t.get('size','')}"
                if tid in seen_ids:
                    continue
                seen_ids.append(tid)
                size = float(t.get("size", 0) or 0)
                if size >= config.WHALE_MIN_TRADE_USD:
                    enriched = dict(t)
                    if "slug" in t and "market_slug" not in t:
                        enriched["market_slug"] = t.get("slug") or t.get("eventSlug") or ""
                    _record_trade(enriched)
                    new_count += 1
            if new_count:
                log.info("Whale HTTP-poll: %d new whale trades", new_count)
        except Exception as e:
            log.warning("Whale HTTP-poll error: %s", e)
        time.sleep(60)


# ---------------------------------------------------------------------------
# WebSocket connection
# ---------------------------------------------------------------------------

def _run_ws():
    """Background thread: connect to Polymarket WebSocket and listen for trades."""
    global _ws_running

    try:
        import websocket  # websocket-client package
    except ImportError:
        log.error("websocket-client not installed. Run: pip3 install websocket-client")
        _ws_running = False
        return

    ws_url = "wss://ws-live-data.polymarket.com"

    def on_open(ws):
        log.info("Whale WS: connected to %s", ws_url)
        # Polymarket real-time-data-client format (verified 2026-04-17):
        #   { "action": "subscribe",
        #     "subscriptions": [{"topic": "activity", "type": "trades"}] }
        # See: github.com/Polymarket/real-time-data-client/blob/main/src/client.ts
        subscription_payload = {
            "action": "subscribe",
            "subscriptions": [{"topic": "activity", "type": "trades"}],
        }
        ws.send(json.dumps(subscription_payload))
        log.info("Whale WS: subscription sent: %s", subscription_payload)

    def on_message(ws, message):
        try:
            data = json.loads(message)
            events = data if isinstance(data, list) else [data]
            for event in events:
                event_type = event.get("type", "") or event.get("topic", "")
                log.debug("Whale WS message: type=%s keys=%s",
                          event_type, list(event.keys())[:8])
                # Accept broad set of trade-ish types (schema may change)
                if event_type in ("trade", "trades", "orders_matched",
                                  "TRADE", "TRADES", "activity"):
                    # Some shapes wrap trades in `payload` / `data`
                    payload = event.get("payload") or event.get("data") or event
                    if isinstance(payload, list):
                        for t in payload:
                            _record_trade(t)
                    elif isinstance(payload, dict):
                        _record_trade(payload)
                elif "trades" in event and isinstance(event["trades"], list):
                    for t in event["trades"]:
                        _record_trade(t)
        except Exception as e:
            log.debug("WS message parse error: %s", e)

    def on_error(ws, error):
        log.warning("Whale WS error: %s", error)

    def on_close(ws, close_status_code, close_msg):
        log.info("Whale WS closed (status=%s). Will reconnect in 30s.", close_status_code)
        _ws_running = False

    while True:
        _ws_running = True
        # WAVE 7 FIX 2026-04-26 (Round-8 council finding #5): re-bootstrap on
        # every (re)connect to recover trades during the disconnect window.
        # Apr-20 incident "Whale WebSocket Confirmed Disconnected — Zero Trades
        # Captured" was 30+ minutes of silent data loss.
        try:
            _bootstrap_recent_trades()
        except Exception as _bse:
            log.warning("Whale WS bootstrap-on-reconnect failed: %s", _bse)
        try:
            ws = websocket.WebSocketApp(
                ws_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            ws.run_forever(ping_interval=30, ping_timeout=10)
        except Exception as e:
            log.warning("Whale WS exception: %s", e)

        _ws_running = False
        time.sleep(30)  # wait before reconnecting


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_whale_ws_thread: Optional[threading.Thread] = None


def start_whale_ws():
    """Start the whale watcher WebSocket in a background daemon thread.

    Also bootstraps from REST API so the cache has data immediately.
    Safe to call multiple times — won't launch duplicate threads.
    """
    global _whale_ws_thread

    if not getattr(config, "WHALE_WS_ENABLED", False):
        log.info("Whale watcher disabled (WHALE_WS_ENABLED=False)")
        return

    # Idempotency guard: don't launch if thread is already alive
    if _whale_ws_thread is not None and _whale_ws_thread.is_alive():
        return

    # Bootstrap from REST first
    _bootstrap_recent_trades()

    _whale_ws_thread = threading.Thread(target=_run_ws, daemon=True, name="whale-ws")
    _whale_ws_thread.start()
    log.info("Whale watcher thread started")

    # Optional HTTP-poll fallback — runs in parallel to WS as a belt-and-suspenders
    # graceful-degrade path. Enable via config.WHALE_HTTP_POLL_ENABLED=True.
    if getattr(config, "WHALE_HTTP_POLL_ENABLED", False):
        poll_thread = threading.Thread(
            target=_run_http_poll_fallback, daemon=True, name="whale-http-poll"
        )
        poll_thread.start()
        log.info("Whale HTTP-poll fallback thread started")


def get_whale_signal(market_title: str) -> float:
    """Return whale activity signal for a market (0.5–1.0).

    > WHALE_SIGNAL_THRESHOLD = significant large-trader activity in last LOOKBACK hours.
    Returns 0.5 (neutral) if no whale activity detected.
    """
    _trim_old(market_title)

    # Match against both slug keys and normalized titles stored in each trade
    needle = market_title.strip().lower()
    candidates = []
    with _whale_lock:
        for slug_key, trades in _whale_trades.items():
            slug_lower = slug_key.lower()
            # Check slug key: exact or substring match
            slug_match = (slug_lower == needle
                          or needle in slug_lower
                          or slug_lower in needle)
            if slug_match:
                candidates.extend(list(trades))
            else:
                # Check normalized_title stored on each trade entry
                for t in trades:
                    norm = t.get("normalized_title", "")
                    if (norm == needle
                            or needle in norm
                            or norm in needle):
                        candidates.append(t)

    if not candidates:
        return 0.5

    total_usd = sum(t["size_usd"] for t in candidates)

    # Scale: $500 = barely notable, $10k+ = very strong signal
    # Signal approaches WHALE_SIGNAL_THRESHOLD at $5k, 1.0 at $50k+
    threshold = getattr(config, "WHALE_SIGNAL_THRESHOLD", 0.7)
    if total_usd >= 50_000:
        return 1.0
    elif total_usd >= 5_000:
        return threshold + (total_usd - 5_000) / 45_000 * (1.0 - threshold)
    elif total_usd >= 500:
        return 0.5 + (total_usd - 500) / 4_500 * (threshold - 0.5)
    else:
        return 0.5


def get_recent_whales(n: int = 10) -> list[dict]:
    """Return the last N large trades across all markets, for dashboard display."""
    with _whale_lock:
        return list(_recent_whales)[:n]
