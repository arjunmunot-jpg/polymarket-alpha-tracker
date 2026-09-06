"""Insider trade detector.

Scans the monthly Polymarket leaderboard for wallets exhibiting statistically
unusual win patterns — consistent early bets on outcomes that resolve correctly.

Five signals (expanded from 3, incorporating NickNaskida polymarket-insider-bot framework):

  1. Win rate / binomial (weight 0.35)
     Binomial test: if a trader wins W out of N resolved trades, compute the
     probability of that happening by chance at 50/50 odds. High win rate with
     high trade count is a very strong signal. Need >= INSIDER_MIN_RESOLVED_TRADES.

  2. Early entry (weight 0.25)
     Average price of WINNING Yes bets. If they consistently buy "Yes" at < 0.30
     (i.e. they bet when the outcome looked unlikely) and those bets resolve
     correctly, they had information the market didn't.

  3. Consistency (weight 0.15)
     Low variance in win rate across three equal time-buckets of recent trades.
     A truly lucky trader will have variable performance; an informed one is
     consistently right.

  4. Unusual sizing (weight 0.15)
     Bets significantly larger than their own average trade size — smart money
     concentrates on high-conviction plays. Bet >= 3x personal average = signal.

  5. Market concentration (weight 0.10)
     Wallet concentrating > 60% of recent trade count in a single market.
     Unusual focus on one outcome = potential insider knowledge.

Confidence score = weighted sum of normalised signal scores (0-1).
Wallets >= INSIDER_CONFIDENCE_THRESHOLD are flagged in the DB.
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests

import config
import db
import tracker as tracker_mod

log = logging.getLogger("polybot.insider_detector")

# Gamma API base for market resolution lookups
_GAMMA_API = "https://gamma-api.polymarket.com"


# ---------------------------------------------------------------------------
# Trade enrichment — add resolution + current price from Gamma API
# ---------------------------------------------------------------------------

def _enrich_trades_with_resolution(trades):
    # type: (list) -> list
    """Add ``resolved`` and ``current_price`` fields to trades via Gamma API.

    The Polymarket Data API (activity endpoint) returns trades without
    resolution status or current market prices.  This function looks up each
    trade's market on the Gamma API (cached per slug) and attaches:
      - resolved (bool): whether the market has resolved or closed
      - current_price (float): outcome price (1.0 = won, 0.0 = lost, or live price)

    Strategy: try /events?slug= first (works for ephemeral binary markets like
    BTC Up/Down), then fall back to /markets?slug=.  Both are cached per slug.
    """
    import json as _json

    enriched = []
    # Cache market lookups by slug to avoid duplicate API calls.
    # Value is a dict with at least {closed, outcomePrices, bestAsk}.
    market_cache = {}  # type: Dict[str, dict]

    def _lookup_via_events(slug):
        # type: (str) -> Optional[dict]
        """Try /events?slug= — returns event with nested markets list."""
        try:
            resp = requests.get(
                "{}/events".format(_GAMMA_API),
                params={"slug": slug},
                timeout=5,
            )
            if not resp.ok:
                return None
            data = resp.json()
            if not isinstance(data, list) or not data:
                return None
            event = data[0]
            # The event itself has closed/active flags
            markets = event.get("markets") or []
            if markets:
                # Return the first market within the event (most events
                # have a single market; multi-outcome events are handled
                # by matching the trade's slug to the market's slug).
                return markets[0]
            # No nested markets — use event-level fields as fallback
            return {
                "closed": event.get("closed", False),
                "outcomePrices": event.get("outcomePrices", ""),
                "bestAsk": event.get("bestAsk", 0),
            }
        except Exception:
            return None

    def _lookup_via_markets(slug):
        # type: (str) -> Optional[dict]
        """Try /markets?slug= — works for standalone markets."""
        try:
            resp = requests.get(
                "{}/markets".format(_GAMMA_API),
                params={"slug": slug},
                timeout=5,
            )
            if not resp.ok:
                return None
            data = resp.json()
            if isinstance(data, list) and data:
                return data[0]
            if isinstance(data, dict) and data:
                return data
            return None
        except Exception:
            return None

    # --- Parallel slug lookup ---
    max_workers = getattr(config, "INSIDER_MAX_ENRICH_WORKERS", 8)
    unique_slugs = list({t.get("slug", "") for t in trades if t.get("slug")})

    def _fetch_slug(slug):
        # type: (str) -> tuple
        market_data = _lookup_via_events(slug)
        if market_data is None:
            market_data = _lookup_via_markets(slug)
        return slug, market_data or {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_slug, s): s for s in unique_slugs}
        for fut in as_completed(futures):
            try:
                slug, data = fut.result()
                market_cache[slug] = data
            except Exception as exc:
                log.warning("Slug lookup error for %s: %s", futures[fut], exc)

    for trade in trades:
        slug = trade.get("slug", "")
        market_data = market_cache.get(slug, {})

        # Resolution: check 'closed' flag (Gamma uses 'closed', not 'resolved')
        trade["resolved"] = bool(market_data.get("closed", False))

        # Current price from outcomePrices.
        # For resolved markets this is ["1","0"] or ["0","1"].
        # For live markets it's the current probability, e.g. ["0.65","0.35"].
        current_price = 0.0
        outcome_prices = market_data.get("outcomePrices", "")
        if outcome_prices and isinstance(outcome_prices, str):
            try:
                prices = _json.loads(outcome_prices)
                if isinstance(prices, list) and prices:
                    # First element = Yes price, second = No price.
                    # Match to the trade's outcome.
                    outcome = (trade.get("outcome") or "").lower()
                    if outcome == "no" and len(prices) > 1:
                        current_price = float(prices[1])
                    else:
                        current_price = float(prices[0])
            except (ValueError, TypeError, IndexError):
                pass
        elif isinstance(outcome_prices, list) and outcome_prices:
            try:
                outcome = (trade.get("outcome") or "").lower()
                if outcome == "no" and len(outcome_prices) > 1:
                    current_price = float(outcome_prices[1])
                else:
                    current_price = float(outcome_prices[0])
            except (ValueError, TypeError, IndexError):
                pass

        if current_price == 0.0:
            current_price = float(market_data.get("bestAsk", 0) or 0)

        trade["current_price"] = current_price
        enriched.append(trade)

    cached_count = sum(1 for v in market_cache.values() if v)
    resolved_count = sum(1 for t in enriched if t.get("resolved"))
    log.debug(
        "Enriched %d trades (%d markets cached, %d resolved)",
        len(enriched), cached_count, resolved_count,
    )
    return enriched


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def _binomial_win_probability(wins: int, total: int) -> float:
    """Probability of winning `wins` or more out of `total` fair coin flips.

    Uses the regularised incomplete beta function approximation.
    Returns a value 0-1 where higher = more statistically improbable (= stronger signal).
    """
    if total == 0:
        return 0.0
    p = wins / total
    # Convert win rate to signal: how many std-devs above 0.5?
    # σ of binomial = sqrt(0.25/n), so z = (p - 0.5) / σ
    sigma = math.sqrt(0.25 / total)
    if sigma == 0:
        return 0.0
    z = (p - 0.5) / sigma
    # Clip z to 0-5 range and normalise to 0-1
    return max(0.0, min(1.0, z / 5.0))


def _early_entry_signal(trades: list[dict]) -> float:
    """Score based on how early (low price) their winning Yes bets were placed.

    Returns 0-1: higher means they consistently buy underpriced Yes outcomes
    that later resolve correctly (classic insider pattern).
    """
    winning_yes_prices = []
    for t in trades:
        outcome = (t.get("outcome") or "").lower()
        side = (t.get("side") or "").lower()
        resolved = t.get("resolved")
        price = float(t.get("price") or t.get("avgPrice") or 0)

        # Match "yes" or "up" outcomes — both are the affirmative side
        if outcome in ("yes", "up") and side == "buy" and price > 0:
            # We consider a trade "winning" if resolved=True or if price has risen significantly.
            # Price movement alone is a heuristic proxy — not a confirmed win — so we
            # weight it lower (0.5x) compared to actually resolved trades (1.0x).
            current = float(t.get("current_price") or price)
            if resolved:
                winning_yes_prices.append(price)
            elif current > price * 1.15:  # 15% price increase — stricter heuristic
                # Discount price-only signals: count at half weight
                winning_yes_prices.append(price * 0.5 + 0.5 * current)

    if not winning_yes_prices:
        return 0.0

    avg_entry = sum(winning_yes_prices) / len(winning_yes_prices)
    # Lower avg entry price = stronger early entry signal
    # Price < 0.20 → signal 1.0; price > 0.70 → signal 0.0
    signal = max(0.0, min(1.0, (0.70 - avg_entry) / 0.50))
    return signal


def _unusual_sizing_signal(trades: list[dict]) -> float:
    """Score based on whether bets are significantly larger than trader's own average.

    High-conviction insiders concentrate capital on their best information.
    Consistent sizing (no big outliers) = no special signal.
    Returns 0-1.
    """
    costs = []
    for t in trades:
        # Try multiple field names: tracker.py normalises to cost_usd,
        # but raw API data uses usdcSize; size is share count (not USD).
        cost = float(
            t.get("cost_usd", 0)
            or t.get("usdcSize", 0)
            or t.get("amount", 0)
            or t.get("size", 0)
            or 0
        )
        if cost > 0:
            costs.append(cost)

    if len(costs) < 3:
        return 0.0

    avg_cost = sum(costs) / len(costs)
    if avg_cost <= 0:
        return 0.0

    # Score = fraction of trades that are 3x+ the personal average
    large_bets = sum(1 for c in costs if c >= avg_cost * 3.0)
    fraction = large_bets / len(costs)

    # 0 large bets = 0.0 signal, >= 30% large bets = 1.0 signal
    return min(1.0, fraction / 0.30)


def _market_concentration_signal(trades: list[dict]) -> float:
    """Score based on how concentrated the wallet is in a single market.

    > 60% of recent trades in one market = unusual focus = potential insider.
    Returns 0-1.
    """
    if not trades:
        return 0.0

    market_counts: dict[str, int] = {}
    for t in trades:
        title = t.get("market_title", "") or t.get("question", "") or "unknown"
        market_counts[title] = market_counts.get(title, 0) + 1

    if not market_counts:
        return 0.0

    max_count = max(market_counts.values())
    concentration = max_count / len(trades)

    # < 30% concentration = no signal; > 60% = full signal
    return max(0.0, min(1.0, (concentration - 0.30) / 0.30))


def _consistency_signal(trades: list[dict]) -> float:
    """Score based on consistency of wins across three time buckets.

    Low std-dev across buckets = consistently right = stronger insider signal.
    Returns 0-1.
    """
    if len(trades) < 6:
        return 0.0

    bucket_size = len(trades) // 3
    buckets = [
        trades[:bucket_size],
        trades[bucket_size: 2 * bucket_size],
        trades[2 * bucket_size:],
    ]

    win_rates = []
    for bucket in buckets:
        resolved = [t for t in bucket if t.get("resolved") or
                    float(t.get("current_price") or 0) > float(t.get("price") or t.get("avgPrice") or 0) * 1.05]
        if bucket:
            win_rates.append(len(resolved) / len(bucket))

    if len(win_rates) < 2:
        return 0.0

    mean_wr = sum(win_rates) / len(win_rates)
    variance = sum((r - mean_wr) ** 2 for r in win_rates) / len(win_rates)
    std_dev = math.sqrt(variance)

    # Low std-dev = consistent = high signal
    # std_dev < 0.05 → signal 1.0; std_dev > 0.30 → signal 0.0
    signal = max(0.0, min(1.0, (0.30 - std_dev) / 0.25))
    return signal


# ---------------------------------------------------------------------------
# Main scoring
# ---------------------------------------------------------------------------

def score_wallet(address: str) -> Optional[dict]:
    """Fetch profile and compute insider confidence score.

    Returns dict with confidence and signals, or None if not enough data.
    """
    profile = tracker_mod.fetch_profile(address)
    if not profile:
        return None

    trades = profile.get("recent_trades") or []
    total = len(trades)

    if total < config.INSIDER_MIN_RESOLVED_TRADES:
        log.debug("Skip %s: only %d trades (need %d)",
                  address[:10], total, config.INSIDER_MIN_RESOLVED_TRADES)
        return None

    # --- Enrich trades with resolution status + current prices ---
    trades = _enrich_trades_with_resolution(trades)

    # Count wins (price rose meaningfully or marked resolved)
    wins = sum(
        1 for t in trades
        if t.get("resolved") or
        float(t.get("current_price") or 0) > float(t.get("price") or t.get("avgPrice") or 0) * 1.05
    )

    # --- Signal 1: Win rate (binomial) ---
    win_rate_signal = _binomial_win_probability(wins, total)

    # --- Signal 2: Early entry ---
    early_entry_signal = _early_entry_signal(trades)

    # --- Signal 3: Consistency ---
    consistency_signal = _consistency_signal(trades)

    # --- Signal 4: Unusual sizing (NickNaskida framework) ---
    sizing_signal = _unusual_sizing_signal(trades)

    # --- Signal 5: Market concentration ---
    concentration_signal = _market_concentration_signal(trades)

    confidence = (
        # Weights per signal_weights_research.md (2026-04-17). Research-backed:
        # - arXiv 2411.13564 / 2212.05912 / 2502.08728: size & concentration > win_rate
        # - arXiv 2603.03136: trader concentration is strongest Polymarket-specific signal
        # Previous: 0.35/0.25/0.15/0.15/0.10 (practitioner heuristic, unvalidated)
        # Live-editable via config.INSIDER_WEIGHTS if set; falls back to these defaults.
        config.INSIDER_WEIGHTS.get("win_rate", 0.25) * win_rate_signal
        + config.INSIDER_WEIGHTS.get("early_entry", 0.25) * early_entry_signal
        + config.INSIDER_WEIGHTS.get("consistency", 0.15) * consistency_signal
        + config.INSIDER_WEIGHTS.get("unusual_sizing", 0.20) * sizing_signal
        + config.INSIDER_WEIGHTS.get("concentration", 0.15) * concentration_signal
    )

    win_rate = wins / total if total > 0 else 0.0
    signals = {
        "win_rate": round(win_rate, 3),
        "win_rate_signal": round(win_rate_signal, 3),
        "early_entry_signal": round(early_entry_signal, 3),
        "consistency_signal": round(consistency_signal, 3),
        "sizing_signal": round(sizing_signal, 3),
        "concentration_signal": round(concentration_signal, 3),
        "total_trades": total,
        "wins": wins,
    }

    log.debug(
        "Wallet %s — win_rate=%.1f%% wr=%.2f ee=%.2f cons=%.2f size=%.2f conc=%.2f → conf=%.2f",
        address[:10],
        win_rate * 100,
        win_rate_signal,
        early_entry_signal,
        consistency_signal,
        sizing_signal,
        concentration_signal,
        confidence,
    )

    return {"address": address, "confidence": round(confidence, 4), "signals": signals}


def _fetch_polymarketscan_wallets() -> List[str]:
    """Fetch top-performer wallet addresses from PolymarketScan leaderboard.

    PolymarketScan exposes a public JSON endpoint used by their leaderboard page.
    We filter for wallets with win_rate > INSIDER_MIN_WIN_RATE to get only
    genuinely consistent performers (not just high-volume bettors who got lucky).

    Returns a list of wallet address strings.
    """
    min_win_rate = getattr(config, "INSIDER_MIN_WIN_RATE", 0.55)
    # PolymarketScan leaderboard API — returns top 500 traders with stats
    urls = [
        "https://polymarketscan.io/api/leaderboard?limit=500&sort=profit",
        "https://polymarketscan.io/api/traders?limit=500&sortBy=profit&order=desc",
    ]
    for url in urls:
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if not resp.ok:
                continue
            data = resp.json()
            # Handle both list and {data: [...]} response formats
            if isinstance(data, dict):
                data = data.get("data") or data.get("traders") or data.get("results") or []
            if not isinstance(data, list) or not data:
                continue
            wallets = []
            for entry in data:
                addr = entry.get("address") or entry.get("proxy_address") or entry.get("wallet")
                if not addr:
                    continue
                wr = float(entry.get("win_rate") or entry.get("winRate") or 0)
                if wr >= min_win_rate:
                    wallets.append(addr)
            if wallets:
                log.debug("PolymarketScan returned %d wallets (win_rate >= %.0f%%)",
                          len(wallets), min_win_rate * 100)
                return wallets
        except Exception as e:
            log.debug("PolymarketScan URL %s failed: %s", url, e)
            continue
    log.debug("PolymarketScan: no valid response from any endpoint")
    return []


def run_detection() -> list[dict]:
    """Full insider detection pipeline.

    1. Fetch monthly leaderboard (broader window for better sample size)
    2. Score each wallet
    3. Flag those >= INSIDER_CONFIDENCE_THRESHOLD in the DB
    4. Return list of flagged wallets

    Returns list of dicts: {address, confidence, signals, is_copying}
    """
    log.info("=== Insider detection scan (period=%s) ===",
             config.INSIDER_LEADERBOARD_PERIOD)

    # Source 1: Polymarket leaderboard via tracker (replaces Bullpen CLI)
    candidates = []  # type: List[dict]
    try:
        lb = tracker_mod.fetch_leaderboard()
        candidates = [{"address": e.get("address", e.get("name", ""))} for e in lb if e.get("address") or e.get("name")]
        log.info("Leaderboard source: %d wallets from Polymarket API", len(candidates))
    except Exception as e:
        log.error("Polymarket leaderboard fetch failed: %s", e)

    # Source 2: PolymarketScan leaderboard (independent top-performers list)
    if getattr(config, "INSIDER_POLYMARKETSCAN_ENABLED", True):
        try:
            pms_wallets = _fetch_polymarketscan_wallets()
            existing = {c["address"] for c in candidates if c.get("address")}
            added = 0
            for addr in pms_wallets:
                if addr not in existing:
                    candidates.append({"address": addr})
                    existing.add(addr)
                    added += 1
            log.info("PolymarketScan source: +%d new wallets (%d total)", added, len(candidates))
        except Exception as e:
            log.warning("PolymarketScan fetch failed (non-fatal): %s", e)

    if not isinstance(candidates, list):
        candidates = []

    # Always include seed wallets (known high-value insiders not on monthly leaderboard)
    seed_wallets = getattr(config, "SEED_INSIDER_WALLETS", [])
    existing_addrs = {c.get("address") for c in candidates}
    for addr in seed_wallets:
        if addr not in existing_addrs:
            candidates.append({"address": addr})

    log.info("Scanning %d wallets for insider signals (%d seeds)...",
             len(candidates), len(seed_wallets))

    flagged = []
    for c in candidates:
        address = c.get("address")
        if not address:
            continue

        result = score_wallet(address)
        if result is None:
            continue

        if result["confidence"] >= config.INSIDER_CONFIDENCE_THRESHOLD:
            log.info(
                "INSIDER FLAGGED: %s — confidence=%.2f win_rate=%.1f%%",
                address[:10],
                result["confidence"],
                result["signals"]["win_rate"] * 100,
            )
            db.flag_insider_wallet(address, result["confidence"], result["signals"])

            # Pull existing DB entry to check is_copying status
            all_insiders = {w["address"]: w for w in db.get_insider_wallets()}
            entry = all_insiders.get(address, {})
            result["is_copying"] = entry.get("is_copying", 0)
            flagged.append(result)
        else:
            log.debug("Not flagged: %s — confidence=%.2f", address[:10], result["confidence"])

    log.info("Insider scan complete: %d/%d wallets flagged", len(flagged), len(candidates))
    return flagged
