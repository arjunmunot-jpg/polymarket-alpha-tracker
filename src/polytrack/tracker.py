"""Discover and score Polymarket traders using public API data.

Data sources (no auth required):
  Leaderboard:  https://lb-api.polymarket.com   — profit + volume rankings
  Activity:     https://data-api.polymarket.com  — per-wallet trade history
  Positions:    https://data-api.polymarket.com  — per-wallet open positions

All data is fetched via public HTTP APIs (no CLI tools or auth required).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests

import config
import db

log = logging.getLogger("polybot.tracker")

# ---------------------------------------------------------------------------
# API base URLs
# ---------------------------------------------------------------------------
_LB_API = "https://lb-api.polymarket.com"
_DATA_API = "https://data-api.polymarket.com"

# ---------------------------------------------------------------------------
# Cache: key -> (timestamp, data)
# ---------------------------------------------------------------------------
_cache: Dict[str, Tuple[float, object]] = {}
_CACHE_TTL = 3600  # 1 hour default

# Minimum settled trades to use settled win rate in scoring (else fall back to raw)
_MIN_SETTLED_FOR_SCORING = 3

# Retry config
_MAX_RETRIES = 1
_RETRY_DELAY = 2.0
_REQUEST_TIMEOUT = 10


def _cached_get(key: str, ttl: int = _CACHE_TTL):
    """Return cached value if still fresh, else None."""
    entry = _cache.get(key)
    if entry and (time.time() - entry[0]) < ttl:
        return entry[1]
    return None


def _store_cache(key: str, value):
    """Store a value in the cache."""
    _cache[key] = (time.time(), value)
    return value


def _api_get(url: str, params: Optional[dict] = None, timeout: int = _REQUEST_TIMEOUT):
    """HTTP GET with retry logic. Returns parsed JSON or None."""
    for attempt in range(_MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < _MAX_RETRIES:
                log.debug("API retry %s (attempt %d): %s", url, attempt + 1, e)
                time.sleep(_RETRY_DELAY)
            else:
                log.warning("API failed %s: %s", url, e)
                return None


# ---------------------------------------------------------------------------
# Leaderboard: top traders by profit and volume
# ---------------------------------------------------------------------------

def fetch_leaderboard() -> List[dict]:
    """Fetch top traders from the Polymarket leaderboard API.

    Merges profit and volume leaderboards to get a richer candidate set.
    Returns a list of dicts with: address, pnl, volume, username.
    """
    cache_key = f"leaderboard:{config.LEADERBOARD_PERIOD}"
    cached = _cached_get(cache_key)
    if cached is not None:
        return cached

    fetch_limit = config.TOP_N_TRADERS * 3  # fetch extra to filter from

    # Fetch profit leaderboard
    profit_data = _api_get(
        f"{_LB_API}/profit",
        params={"period": config.LEADERBOARD_PERIOD, "limit": fetch_limit},
    )
    profit_list = profit_data if isinstance(profit_data, list) else []

    # Fetch volume leaderboard
    volume_data = _api_get(
        f"{_LB_API}/volume",
        params={"period": config.LEADERBOARD_PERIOD, "limit": fetch_limit},
    )
    volume_list = volume_data if isinstance(volume_data, list) else []

    # Build lookup: address -> {pnl, volume, username}
    traders: Dict[str, dict] = {}

    for entry in profit_list:
        addr = entry.get("proxyWallet", "").lower()
        if not addr:
            continue
        traders[addr] = {
            "address": addr,
            "pnl": float(entry.get("amount", 0)),
            "volume": 0.0,
            "username": entry.get("name") or entry.get("pseudonym") or "",
        }

    for entry in volume_list:
        addr = entry.get("proxyWallet", "").lower()
        if not addr:
            continue
        vol = float(entry.get("amount", 0))
        if addr in traders:
            traders[addr]["volume"] = vol
        else:
            traders[addr] = {
                "address": addr,
                "pnl": 0.0,
                "volume": vol,
                "username": entry.get("name") or entry.get("pseudonym") or "",
            }

    result = list(traders.values())
    log.info("Leaderboard: %d traders from profit(%d) + volume(%d)",
             len(result), len(profit_list), len(volume_list))
    return _store_cache(cache_key, result)


# ---------------------------------------------------------------------------
# Trader profile: trade history from data-api
# ---------------------------------------------------------------------------

def fetch_profile(address: str) -> Optional[dict]:
    """Fetch trader profile data: recent trades, trade count, username.

    Returns a dict with trader profile data:
      {
        "address": str,
        "username": str,
        "trades_count": int,
        "recent_trades": [
          {"timestamp": str, "market_title": str, "outcome": str,
           "side": str, "price": float, "cost_usd": float, "amount": float},
          ...
        ]
      }
    or None on failure.
    """
    cache_key = f"profile:{address.lower()}"
    cached = _cached_get(cache_key)
    if cached is not None:
        return cached

    # Fetch recent trade activity
    activity = _api_get(
        f"{_DATA_API}/activity",
        params={"user": address, "type": "TRADE", "limit": 50},
    )
    if activity is None:
        return None

    trades_raw = activity if isinstance(activity, list) else []

    # Parse into the format copy_manager.py expects
    recent_trades = []
    username = ""
    for t in trades_raw:
        if not username:
            username = t.get("name") or t.get("pseudonym") or ""

        ts = t.get("timestamp", 0)
        # Convert unix timestamp to ISO format
        try:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            ts_str = dt.isoformat()
        except (OSError, ValueError):
            ts_str = ""

        recent_trades.append({
            "timestamp": ts_str,
            "market_title": t.get("title", "Unknown"),
            "outcome": t.get("outcome", ""),
            "side": t.get("side", "BUY"),
            "price": float(t.get("price", 0)),
            "cost_usd": float(t.get("usdcSize", 0)),
            "amount": float(t.get("size", 0)),
            "slug": t.get("slug", ""),
            "conditionId": t.get("conditionId", ""),
        })

    profile = {
        "address": address.lower(),
        "username": username,
        "trades_count": len(recent_trades),
        "recent_trades": recent_trades,
    }
    return _store_cache(cache_key, profile)


# ---------------------------------------------------------------------------
# Market volume check via Gamma API
# ---------------------------------------------------------------------------

def check_market_volume(slug: str) -> float:
    """Check 24h volume for a market. Returns volume in USD or 0 on failure."""
    cache_key = f"market_vol:{slug}"
    cached = _cached_get(cache_key, ttl=300)  # 5 min cache for volume
    if cached is not None:
        return cached

    data = _api_get(
        f"{config.POLYMARKET_GAMMA_URL}/markets",
        params={"slug": slug, "limit": 1},
    )
    if isinstance(data, list) and data:
        vol = data[0].get("volume") or data[0].get("volume24hr") or 0
        result = float(vol)
        _store_cache(cache_key, result)
        return result
    return 0.0


# ---------------------------------------------------------------------------
# Settled win rate from DB
# ---------------------------------------------------------------------------

def _calc_settled_win_rate(trader_address: str) -> float:
    """Calculate win rate counting only markets that have actually resolved.

    Queries simulated_trades for this trader where resolved=1
    (market outcome known -- not expired, abandoned, or still open).
    Returns a float in [0.0, 1.0], or -1.0 if insufficient settled data
    (fewer than _MIN_SETTLED_FOR_SCORING resolved trades).
    """
    conn = db._connect()
    try:
        row = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN sim_pnl > 0 THEN 1 ELSE 0 END) as wins
            FROM simulated_trades
            WHERE trader_address = ?
              AND resolved = 1
        """, (trader_address,)).fetchone()
    finally:
        conn.close()

    if row is None:
        return -1.0

    total = int(row["total"] or 0)
    wins = int(row["wins"] or 0)

    if total < _MIN_SETTLED_FOR_SCORING:
        return -1.0

    return round(wins / total, 4)


# ---------------------------------------------------------------------------
# Normalization + scoring
# ---------------------------------------------------------------------------

def _normalize(values: List[float]) -> List[float]:
    """Min-max normalize a list of values to 0-1 range."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    span = hi - lo
    if span == 0:
        return [1.0] * len(values)
    return [(v - lo) / span for v in values]


def score_traders(candidates: List[dict]) -> List[dict]:
    """Score and rank trader candidates.

    Each candidate is a dict with at minimum: address, pnl, volume.
    We enrich with profile data (trade count + recency) via the data API.
    """
    enriched = []
    for c in candidates:
        pnl = float(c.get("pnl") or 0)
        volume = float(c.get("volume") or 0)

        # Fetch profile for trade count + recency
        profile = fetch_profile(c["address"])
        trade_count = 0
        recency_hours = 999
        if profile:
            trade_count = profile.get("trades_count") or 0
            trades = profile.get("recent_trades") or []
            if trades:
                try:
                    latest = datetime.fromisoformat(
                        trades[0]["timestamp"].replace(" UTC", "+00:00")
                    )
                    recency_hours = (datetime.now(timezone.utc) - latest).total_seconds() / 3600
                except (KeyError, ValueError):
                    pass

        # Apply minimum filters
        if trade_count < config.MIN_TRADES_COUNT:
            log.debug("Skip %s: only %d trades", c["address"][:10], trade_count)
            continue
        if pnl < config.MIN_PNL_USD:
            log.debug("Skip %s: PnL $%.0f below threshold", c["address"][:10], pnl)
            continue
        if recency_hours > config.MAX_TRADER_INACTIVITY_HOURS:
            log.info("Skip %s: inactive %.1fh (max %dh)",
                     c["address"][:10], recency_hours, config.MAX_TRADER_INACTIVITY_HOURS)
            continue

        # Settled win rate: only count trades where market actually resolved
        settled_wr = _calc_settled_win_rate(c["address"])
        raw_win_rate = float(c.get("win_rate") or c.get("winRate") or 0)
        # Use settled win rate when available; fall back to raw
        effective_wr = settled_wr if settled_wr >= 0 else raw_win_rate
        wr_source = "settled" if settled_wr >= 0 else "raw"
        if settled_wr >= 0 and raw_win_rate > 0:
            log.debug("Trader %s win rate: settled=%.2f raw=%.2f (using %s)",
                      c["address"][:10], settled_wr, raw_win_rate, wr_source)

        enriched.append({
            "address": c["address"],
            "username": c.get("username", ""),
            "pnl": pnl,
            "volume": volume,
            "trade_count": trade_count,
            "recency_hours": recency_hours,
            "win_rate": effective_wr,
            "win_rate_source": wr_source,
        })

    if not enriched:
        return []

    # Normalize each dimension to 0-1
    pnls_norm = _normalize([t["pnl"] for t in enriched])
    vols_norm = _normalize([t["volume"] for t in enriched])
    counts_norm = _normalize([float(t["trade_count"]) for t in enriched])
    wr_norm = _normalize([t["win_rate"] for t in enriched])
    # Recency: lower is better -> invert
    recency_raw = [t["recency_hours"] for t in enriched]
    recency_inv = [max(recency_raw) - r for r in recency_raw] if recency_raw else []
    recency_norm = _normalize(recency_inv)

    w = config.SCORE_WEIGHTS
    # Win rate weight: borrow 0.10 from PnL weight when win_rate data exists
    wr_weight = w.get("win_rate", 0.10)
    pnl_weight = w["pnl"]
    # Only reduce pnl weight if win_rate is not already in the configured weights
    if "win_rate" not in w:
        pnl_weight = max(w["pnl"] - wr_weight, 0.10)

    for i, t in enumerate(enriched):
        t["score"] = (
            pnl_weight * pnls_norm[i]
            + w["volume"] * vols_norm[i]
            + w["trade_count"] * counts_norm[i]
            + w["recency"] * recency_norm[i]
            + wr_weight * wr_norm[i]
        )

    enriched.sort(key=lambda t: t["score"], reverse=True)
    return enriched


# ---------------------------------------------------------------------------
# Full discovery pipeline
# ---------------------------------------------------------------------------

def discover_and_store() -> List[dict]:
    """Full discovery pipeline: fetch leaderboard + seed wallets -> score -> persist to DB."""
    log.info("Discovering top traders (period=%s)...", config.LEADERBOARD_PERIOD)

    candidates = fetch_leaderboard()

    # Merge in seed insider wallets (always included even if not on leaderboard)
    seed_addrs = set(a.lower() for a in getattr(config, "SEED_INSIDER_WALLETS", []))
    candidate_addrs = set(c["address"].lower() for c in candidates)
    for seed_addr in seed_addrs:
        if seed_addr.lower() not in candidate_addrs:
            # Fetch profile to get username
            profile = fetch_profile(seed_addr)
            username = ""
            if profile:
                username = profile.get("username", "")
            candidates.append({
                "address": seed_addr,
                "pnl": config.MIN_PNL_USD,  # floor — will be scored on activity
                "volume": 0.0,
                "username": username,
            })

    if not candidates:
        log.warning("No trader candidates found (leaderboard + seeds)")
        return []

    scored = score_traders(candidates)

    # --- Quality floor: skip traders below minimum absolute score ---
    qualified = [t for t in scored if t["score"] >= config.MIN_TRADER_SCORE]
    dropped = len(scored) - len(qualified)
    if dropped:
        log.info("Score floor %.2f dropped %d/%d traders",
                 config.MIN_TRADER_SCORE, dropped, len(scored))
    top = qualified[: config.TOP_N_TRADERS]

    for t in top:
        db.upsert_trader(
            address=t["address"],
            username=t["username"],
            pnl=t["pnl"],
            volume=t["volume"],
            trade_count=t["trade_count"],
            score=t["score"],
        )
        log.info(
            "Trader %s (%s) — score=%.3f pnl=$%.0f trades=%d",
            t["username"] or t["address"][:10],
            t["address"][:10],
            t["score"],
            t["pnl"],
            t["trade_count"],
        )

    return top
