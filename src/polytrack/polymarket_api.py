"""Polymarket CLOB/Gamma API signal module.

Fetches real-time orderbook data and market metadata to produce a 0-1
opportunity signal for any market. Integrated into trade_selector confidence
scoring as the 'api_signal' weight.

APIs used (both public, no auth required):
  Gamma: https://gamma-api.polymarket.com  — market metadata, prices, volume
  CLOB:  https://clob.polymarket.com       — live orderbook depth, token prices

Signal logic (adapted from MoonDev + Polymarket ecosystem research):
  - Low spread (<2%) + high bid imbalance (>60%) + price 0.2-0.8 = strong buy
  - Very thin orderbook or wide spread = skip (signal → 0)
  - Returns 0-1 float: 0.5 = neutral/no data, >0.7 = strong opportunity
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

import config

log = logging.getLogger("polybot.polymarket_api")

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[float, dict]] = {}   # key → (timestamp, result)


def _gamma_get(url: str, params: dict = None, timeout: int = 8, retries: int = 2):
    """WAVE 8 FIX 2026-04-26 (Round-8 council #3 deferred, now shipped):
    retry helper with exponential backoff and 429-aware sleep.

    Was: each requests.get failed → returned None → callers silently fell
    back, INCLUDING copy_manager's MAX_RESOLVE_DAYS deadline gate which
    fails-OPEN on None (skips the deadline check entirely → bot copies
    trades that don't resolve for 6 months during a Gamma outage).

    Now: 2 retries with backoff (1s, 2s, 4s) and respect 429 Retry-After.
    Returns parsed JSON or None on permanent failure.
    """
    import time as _time
    backoff = 1.0
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params or {}, timeout=timeout)
            if r.status_code == 429:
                # Rate limited — respect Retry-After header if present
                wait = r.headers.get("Retry-After")
                try:
                    wait_s = float(wait) if wait else backoff
                except (ValueError, TypeError):
                    wait_s = backoff
                _time.sleep(min(wait_s, 30.0))
                backoff *= 2
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries:
                log.warning("Gamma %s failed after %d retries: %s",
                            url.split("/")[-1], retries, e)
                return None
            _time.sleep(backoff)
            backoff *= 2
    return None


def _cached(key: str, ttl: int = config.API_SIGNAL_CACHE_TTL) -> Optional[dict]:
    entry = _cache.get(key)
    if entry and (time.time() - entry[0]) < ttl:
        return entry[1]
    return None


def _store(key: str, result: dict) -> dict:
    _cache[key] = (time.time(), result)
    return result


# ---------------------------------------------------------------------------
# Gamma API (market metadata)
# ---------------------------------------------------------------------------

def search_markets(query: str, limit: int = 20) -> list[dict]:
    """Search active Gamma markets by keyword. Returns raw market dicts."""
    key = f"search:{query}:{limit}"
    cached = _cached(key)
    if cached is not None:
        return cached.get("markets", [])

    try:
        r = requests.get(
            f"{config.POLYMARKET_GAMMA_URL}/markets",
            params={"active": "true", "closed": "false", "search": query, "limit": limit},
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        markets = data if isinstance(data, list) else data.get("data", [])
        result = {"markets": markets}
        _store(key, result)
        return markets
    except Exception as e:
        log.debug("Gamma search '%s' error: %s", query, e)
        return []


def get_market_by_condition_id(condition_id: str) -> Optional[dict]:
    """Fetch a specific market from Gamma by conditionId."""
    key = f"market:{condition_id}"
    cached = _cached(key)
    if cached is not None:
        return cached

    try:
        r = requests.get(
            f"{config.POLYMARKET_GAMMA_URL}/markets",
            params={"conditionId": condition_id},
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        markets = data if isinstance(data, list) else data.get("data", [])
        if markets:
            result = markets[0]
            _store(key, result)
            return result
    except Exception as e:
        log.debug("Gamma market fetch error: %s", e)
    return None


def find_market_by_title(query: str) -> Optional[dict]:
    """Find best-matching market for a given title/question string."""
    markets = search_markets(query, limit=10)
    if not markets:
        return None
    query_lower = query.lower()
    # Exact match first
    for m in markets:
        if query_lower in m.get("question", "").lower():
            return m
    return markets[0] if markets else None


# ---------------------------------------------------------------------------
# CLOB API (orderbook)
# ---------------------------------------------------------------------------

def get_orderbook(token_id: str) -> Optional[dict]:
    """Fetch live orderbook for a token from CLOB API.

    Returns dict with keys: bids, asks (each list of {price, size} dicts).
    Endpoint: GET /book?token_id={token_id}
    """
    key = f"ob:{token_id}"
    cached = _cached(key, ttl=30)   # orderbook changes fast — 30s TTL
    if cached is not None:
        return cached

    try:
        r = requests.get(
            f"{config.POLYMARKET_CLOB_URL}/book",
            params={"token_id": token_id},
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        _store(key, data)
        return data
    except Exception as e:
        log.debug("CLOB orderbook error for token %s: %s", token_id[:16], e)
    return None


def _parse_orderbook_stats(ob: dict) -> dict:
    """Compute spread, depth, and imbalance from raw orderbook.

    Note: bids/asks are NOT sorted best-first in CLOB API — must sort.
    When asks is empty (market near ITM resolution), spread is 99.0 (skip signal).
    """
    bids = ob.get("bids", [])
    asks = ob.get("asks", [])

    # Depth always computable from whichever side has data
    bid_depth = sum(float(b.get("price", 0)) * float(b.get("size", 0)) for b in bids[:5])
    ask_depth = sum(float(a.get("price", 0)) * float(a.get("size", 0)) for a in asks[:5])

    if not bids or not asks:
        # One side empty = market near resolution, not a good entry signal
        total = bid_depth + ask_depth
        imbalance = bid_depth / total if total > 0 else 0.5
        return {
            "spread_pct": 99.0,
            "bid_depth": round(bid_depth, 2),
            "ask_depth": round(ask_depth, 2),
            "imbalance": round(imbalance, 4),
        }

    # Best bid = highest bid price; best ask = lowest ask price
    try:
        best_bid = max(float(b["price"]) for b in bids)
        best_ask = min(float(a["price"]) for a in asks)
    except (KeyError, ValueError):
        return {"spread_pct": 99.0, "bid_depth": round(bid_depth, 2),
                "ask_depth": round(ask_depth, 2), "imbalance": 0.5}

    mid = (best_bid + best_ask) / 2 if (best_bid + best_ask) > 0 else 0.5
    spread_pct = ((best_ask - best_bid) / mid * 100) if mid > 0 else 99.0

    total = bid_depth + ask_depth
    imbalance = bid_depth / total if total > 0 else 0.5

    return {
        "spread_pct": round(spread_pct, 3),
        "bid_depth": round(bid_depth, 2),
        "ask_depth": round(ask_depth, 2),
        "imbalance": round(imbalance, 4),
        "best_bid": best_bid,
        "best_ask": best_ask,
    }


# ---------------------------------------------------------------------------
# Main signal function
# ---------------------------------------------------------------------------

def score_market_opportunity(
    market_title: str,
    current_price: float,
    token_id: Optional[str] = None,
    condition_id: Optional[str] = None,
) -> dict:
    """Score a market for trading opportunity. Returns 0-1 signal.

    Algorithm:
      1. Fetch orderbook (if token_id available) or market metadata
      2. Score spread quality   (low spread → good)
      3. Score price zone       (0.2-0.8 → best entry range)
      4. Score order imbalance  (one-sided orderbook → signal)
      5. Score liquidity depth  (deeper = safer)
      6. Composite 0-1 score

    Returns:
        {
          "signal": float 0-1,
          "direction": "buy" | "skip",
          "spread_pct": float,
          "imbalance": float,
          "reason": str,
        }
    """
    key = f"opp:{market_title[:40]}:{current_price:.3f}"
    cached = _cached(key)
    if cached is not None:
        return cached

    result = {
        "signal": 0.5,
        "direction": "skip",
        "spread_pct": 99.0,
        "imbalance": 0.5,
        "reason": "no data",
    }

    # --- Try to get orderbook ---
    ob_stats = None
    if token_id:
        ob = get_orderbook(token_id)
        if ob:
            ob_stats = _parse_orderbook_stats(ob)

    # --- Fallback: use market volume + price from Gamma search ---
    # Also fetch market_data to use native spread/bestBid/bestAsk fields
    market_data = find_market_by_title(market_title) if not condition_id else \
                  get_market_by_condition_id(condition_id)

    # --- Score components ---
    # 1. Spread quality (0 = bad, 1 = perfect)
    # Prefer Gamma's native spread/bestBid/bestAsk fields — authoritative and always present
    if ob_stats and ob_stats["spread_pct"] < 50:
        spread_pct = ob_stats["spread_pct"]
        imbalance = ob_stats["imbalance"]
        bid_depth = ob_stats["bid_depth"]
    elif market_data and market_data.get("spread") is not None:
        # Gamma market: spread field is already in decimal (e.g. 0.01 = 1 cent)
        native_spread = float(market_data.get("spread", 0.05))
        best_bid = float(market_data.get("bestBid", current_price))
        best_ask = float(market_data.get("bestAsk", current_price + native_spread))
        mid = (best_bid + best_ask) / 2 if (best_bid + best_ask) > 0 else 0.5
        spread_pct = (native_spread / mid * 100) if mid > 0 else 99.0
        imbalance = 0.5   # no depth data from Gamma
        vol = float(market_data.get("liquidity", market_data.get("volume", 0)) or 0)
        bid_depth = vol / 10_000
    else:
        # Unknown spread — use moderate default
        spread_pct = 5.0
        imbalance = 0.5
        bid_depth = 0.0
        if market_data:
            try:
                vol = float(market_data.get("liquidity", market_data.get("volume", 0)) or 0)
                bid_depth = vol / 10_000
            except Exception:
                pass

    # Score spread (0-1): < 1% = 1.0, > 5% = 0.0
    spread_score = max(0.0, 1.0 - spread_pct / 5.0)

    # Score price zone (0-1): sweet spot 0.25-0.75
    if 0.25 <= current_price <= 0.75:
        price_score = 1.0
    elif current_price < 0.25:
        price_score = current_price / 0.25   # scales 0→0.25 = 0→1
    else:  # > 0.75
        price_score = (1.0 - current_price) / 0.25   # scales 0.75→1.0 = 1→0
    price_score = max(0.0, price_score)

    # Score imbalance (signal strength): deviation from 0.5
    imbalance_score = abs(imbalance - 0.5) * 2.0   # 0 = balanced, 1 = all one side

    # Score depth (liquidity): deeper is better, cap at $5000 notional
    depth_score = min(1.0, bid_depth / 500.0)

    # --- Composite ---
    # Weights: spread matters most, then price zone, then imbalance signal, then depth
    composite = (
        0.35 * spread_score
        + 0.25 * price_score
        + 0.30 * imbalance_score
        + 0.10 * depth_score
    )
    composite = round(min(1.0, max(0.0, composite)), 4)

    # Direction hint from imbalance (if we have real orderbook)
    direction = "buy" if (ob_stats and imbalance > 0.55) else "skip"

    reason_parts = []
    if spread_pct < 2:
        reason_parts.append(f"tight spread {spread_pct:.1f}%")
    elif spread_pct > 4:
        reason_parts.append(f"wide spread {spread_pct:.1f}%")
    if imbalance > 0.60:
        reason_parts.append(f"bid-heavy {imbalance:.0%}")
    elif imbalance < 0.40:
        reason_parts.append(f"ask-heavy {1-imbalance:.0%}")
    if not reason_parts:
        reason_parts.append("balanced market")

    result.update({
        "signal": composite,
        "direction": direction,
        "spread_pct": spread_pct,
        "imbalance": imbalance,
        "reason": ", ".join(reason_parts),
    })

    _store(key, result)
    log.debug(
        "API signal '%s' @ %.3f: signal=%.3f spread=%.1f%% imbalance=%.2f (%s)",
        market_title[:40], current_price, composite, spread_pct, imbalance,
        result["reason"],
    )
    return result


# ---------------------------------------------------------------------------
# Token ID resolution
# ---------------------------------------------------------------------------

def get_yes_token_id(condition_id: str) -> Optional[str]:
    """Get the YES/Up outcome token_id for a market.

    First tries CLOB /markets/{condition_id}, falls back to Gamma market clobTokenIds.
    Note: clobTokenIds in Gamma API is a JSON-encoded string inside JSON — needs double parse.
    """
    key = f"token:{condition_id}"
    cached = _cached(key, ttl=3600)
    if cached is not None:
        return cached.get("token_id")

    # Try CLOB API first
    try:
        r = requests.get(
            f"{config.POLYMARKET_CLOB_URL}/markets/{condition_id}",
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        tokens = data.get("tokens", [])
        for t in tokens:
            if t.get("outcome", "").lower() in ("yes", "up", "higher"):
                token_id = str(t.get("token_id", ""))
                _store(key, {"token_id": token_id})
                return token_id
        if tokens:
            token_id = str(tokens[0].get("token_id", ""))
            _store(key, {"token_id": token_id})
            return token_id
    except Exception as e:
        log.debug("CLOB token lookup error for %s: %s", condition_id[:16], e)

    # Fall back: parse clobTokenIds from Gamma market
    mkt = get_market_by_condition_id(condition_id)
    if mkt:
        try:
            import json as _json
            clob_ids_raw = mkt.get("clobTokenIds", "[]")
            clob_ids = _json.loads(clob_ids_raw) if isinstance(clob_ids_raw, str) else clob_ids_raw
            outcomes_raw = mkt.get("outcomes", '["Yes","No"]')
            outcomes = _json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
            for i, outcome in enumerate(outcomes):
                if outcome.lower() in ("yes", "up", "higher") and i < len(clob_ids):
                    token_id = str(clob_ids[i])
                    _store(key, {"token_id": token_id})
                    return token_id
        except Exception as e:
            log.debug("Gamma clobTokenIds parse error: %s", e)
    return None


def get_token_id_for_outcome(market: dict, outcome: str) -> Optional[str]:
    """Resolve CLOB token_id for a specific outcome within a market dict.

    WAVE 12 SHIP-2 (2026-05-16): generic version of get_yes_token_id covering
    sports markets (outcomes are team names) + binary markets (Yes/No). Used
    by callers to pass `token_id` to `db.log_simulated_trade` so the wired
    `fill_simulator` walks the real orderbook for the actual side traded
    (was using mid-price → dry-run slippage = $0 silently — audit verdict).

    `market`: dict returned by find_market_by_title / get_market_by_condition_id.
        Polymarket markets carry:
          - `clobTokenIds`: JSON-encoded list of token_id strings, ordered
                            same as `outcomes`.
          - `outcomes`: JSON-encoded list of outcome name strings.
    `outcome`: case-insensitive name (e.g. "Yes", "Cavaliers", "Tundra Esports").

    Returns the token_id string for that outcome, or None if not found.
    Caller falls back to mid-price model when None (current behavior).

    Failure-tolerant: any parse error returns None (never raises), so a
    malformed market dict cannot block trade recording.
    """
    if not market or not outcome:
        return None
    try:
        import json as _json
        clob_raw = market.get("clobTokenIds", "[]")
        clob_ids = _json.loads(clob_raw) if isinstance(clob_raw, str) else clob_raw
        out_raw = market.get("outcomes", "[]")
        outcomes = _json.loads(out_raw) if isinstance(out_raw, str) else out_raw
        norm = outcome.strip().lower()
        for i, o in enumerate(outcomes or []):
            if str(o).strip().lower() == norm and i < len(clob_ids):
                return str(clob_ids[i])
        log.debug(
            "get_token_id_for_outcome: outcome %r not in %s",
            outcome, [str(o) for o in (outcomes or [])][:4],
        )
        return None
    except Exception as e:
        log.debug("get_token_id_for_outcome error: %s", e)
        return None


# ---------------------------------------------------------------------------
# Convenience: score by title only (used in trade_selector)
# ---------------------------------------------------------------------------

def get_api_signal(market_title: str, price: float) -> float:
    """Quick helper: returns just the 0-1 signal float for confidence scoring.

    Caches aggressively (API_SIGNAL_CACHE_TTL seconds from config).
    Returns 0.5 (neutral) on any failure so confidence formula is unaffected.
    """
    try:
        result = score_market_opportunity(market_title, price)
        return result.get("signal", 0.5)
    except Exception as e:
        log.debug("get_api_signal error for '%s': %s", market_title[:40], e)
        return 0.5
