"""Trade selection engine: confidence scoring, deduplication, and priority ranking.

Every trade candidate (from copy-trading or insider detection) passes through here
before being logged to the DB. Only high-confidence, deduplicated trades are kept.

Confidence formula (Phase 4):
    signal = trader_score * 0.55 + insider_flag * 0.15 + volume_factor * 0.10 + api_signal * 0.20
    min_required = CONF_BASE + market_price * CONF_PRICE_SENSITIVITY

Sliding scale examples (CONF_BASE=0.35, CONF_PRICE_SENSITIVITY=0.20):
    price=0.30 → min_required=0.41
    price=0.50 → min_required=0.45
    price=0.70 → min_required=0.49
    price=0.90 → min_required=0.53
"""

from __future__ import annotations

import logging
from typing import Optional

import config
import db
import polymarket_api

log = logging.getLogger("polybot.trade_selector")


def compute_confidence(
    trader_score: float,
    market_price: float,
    market_volume: float,
    is_insider: bool = False,
    api_signal: float = 0.0,
    outcome: str = "",
    side: str = "Buy",
    conf_base: float = None,
    news_signal: float = 0.5,
) -> tuple:
    """Compute signal confidence and minimum required threshold for a trade.

    Args:
        trader_score  — 0-1 score of the source trader
        market_price  — market odds (0-1)
        market_volume — USD volume in the market
        is_insider    — True if signal comes from insider detection
        api_signal    — 0-1 score from Polymarket CLOB/Gamma API (0 if unavailable)
        outcome       — "Yes" or "No" (used for NO prior adjustment)
        side          — "Buy" or "Sell"
        news_signal   — 0-1 sentiment from news_agent. 0.5=neutral. Treated as
                        signed deviation aligned with outcome direction.

    Returns:
        (signal, min_required, passes)
        signal       — our composite confidence in this trade (0-1)
        min_required — sliding-scale threshold based on market price
        passes       — True if signal >= min_required
    """
    w = config.TRADE_CONFIDENCE_WEIGHTS
    volume_factor = min(1.0, market_volume / 10_000)

    # WAVE 8 FIX 2026-04-26 (news pipeline integration): align news_signal
    # with the trade's outcome direction. news_signal is P(event happens).
    # For YES outcome: high news_signal → news supports the trade.
    # For NO outcome: low news_signal → news supports the trade.
    # Convert to signed deviation [-1, +1]; multiply by weight to get the
    # additive contribution. Neutral news (0.5) contributes 0 — no free boost.
    outcome_lower = (outcome or "").lower()
    if outcome_lower in ("yes", "up", "true"):
        aligned_news = news_signal
    elif outcome_lower in ("no", "down", "false"):
        aligned_news = 1.0 - news_signal
    else:
        aligned_news = 0.5  # non-binary outcome → neutral
    news_boost = (aligned_news - 0.5) * 2.0  # range [-1, +1]

    signal = (
        trader_score * w["trader_score"]
        + (w["insider_signal"] if is_insider else 0.0)
        + volume_factor * w["volume_factor"]
        + api_signal * w.get("api_signal", 0.0)
        + news_boost * w.get("news_signal", 0.0)
    )
    signal = max(0.0, min(1.0, signal))

    # Statistical NO prior: 75% of Polymarket markets resolve NO (86M+ transactions).
    # Adjust confidence to reflect this base rate.
    if side.lower() == "buy":
        outcome_lower = outcome.lower()
        if outcome_lower == "yes":
            signal = signal - getattr(config, "NO_PRIOR_PENALTY", 0.0)
        elif outcome_lower == "no":
            signal = signal + getattr(config, "NO_PRIOR_BONUS", 0.0)
    signal = max(0.0, min(1.0, signal))

    # Calibration correction: adjust raw confidence based on historical accuracy.
    # If we consistently say "60% confident" but only win 45%, the correction
    # pulls signal DOWN so EV and Kelly size correctly. If we underpredict,
    # it nudges signal UP so we don't miss profitable trades.
    signal = _apply_calibration_correction(signal)

    effective_conf_base = conf_base if conf_base is not None else config.CONF_BASE
    min_required = effective_conf_base + market_price * config.CONF_PRICE_SENSITIVITY

    passes = signal >= min_required
    return signal, min_required, passes


# ── Calibration feedback ────────────────────────────────────────────────────

_calibration_cache: dict = {}
_calibration_ts: float = 0.0
_researcher_mod = None  # lazy import cache — heavy module with optuna


def _apply_calibration_correction(raw_confidence: float) -> float:
    """Adjust confidence using the calibration curve (predicted vs actual win rate).

    Refreshes the calibration data every 30 minutes from the DB.
    Returns the corrected confidence, clamped to [0.05, 0.95].
    """
    import time
    global _calibration_cache, _calibration_ts, _researcher_mod

    # Refresh calibration data every 30 minutes
    now = time.time()
    if now - _calibration_ts > 1800 or not _calibration_cache:
        try:
            if _researcher_mod is None:
                import researcher as _rm
                _researcher_mod = _rm
            curve = _researcher_mod.compute_calibration_curve(min_trades_per_bucket=5)
            _calibration_cache = {}
            for bucket in curve:
                mid = bucket["midpoint"]
                error = bucket["calibration_error"]  # actual - predicted
                _calibration_cache[mid] = error
            _calibration_ts = now
        except Exception:
            return raw_confidence  # can't calibrate yet, return raw

    if not _calibration_cache:
        return raw_confidence  # no calibration data yet

    # Find the nearest calibration bucket
    midpoints = sorted(_calibration_cache.keys())
    nearest = min(midpoints, key=lambda m: abs(m - raw_confidence))

    # Only correct if the bucket is reasonably close (within 0.10)
    if abs(nearest - raw_confidence) > 0.10:
        return raw_confidence

    error = _calibration_cache[nearest]
    # Apply correction: if actual > predicted (error > 0), boost confidence
    # If actual < predicted (error < 0), reduce confidence
    # Use 50% of the error as correction (conservative, avoids oscillation)
    corrected = raw_confidence + error * 0.50
    return max(0.05, min(0.95, corrected))


def rank_candidates(candidates: list, conf_base: float = None) -> list:
    """Score, filter, deduplicate, and prioritise trade candidates.

    Each candidate dict must have:
        market_title   str
        outcome        str   ("Yes" / "No")
        side           str   ("Buy" / "Sell")
        price          float  market price of the outcome
        trader_score   float  0-1 score of the source trader
        market_volume  float  USD volume in this market
        is_insider     bool   True if signal comes from insider detection
        source         str    "copy" or "insider"
        trader_address str
        shares         float
        cost           float

    Returns ranked list of candidates that passed all filters,
    sorted: insiders first, then by confidence descending.
    """
    # Algorithmic strategies (scanner, weather, bond, arb, btc_5min) compute their own
    # directional signal and set trader_score/confidence as a proxy. The copy-trader
    # formula doesn't apply to them — and the NO_PRIOR_PENALTY would double-count since
    # their signal already encodes direction. Gate on CONF_BASE only, then pass through.
    # WAVE 5 FIX 2026-04-25: corr_arb added (council CRITICAL-1, both Contrarian +
    # Executor flagged). Was being routed through copy-trader NO_PRIOR_PENALTY path,
    # mis-scoring all 79K corr_arb candidates. Confirmed via grep: corr_arb is the
    # DB `source` label for me_arb/ss_arb/cv_arb/ri_arb sub-strategies.
    ALGO_SOURCES = {"scanner", "bond", "arb", "corr_arb", "weather", "btc_5min"}
    effective_conf_base = conf_base if conf_base is not None else config.CONF_BASE

    scored = []
    for c in candidates:
        source = c.get("source", "copy")

        if source in ALGO_SOURCES:
            raw_conf = float(c.get("confidence") or c.get("trader_score", 0.5))
            if raw_conf < effective_conf_base:
                # ULTRA-REVIEW LOGGING AUDIT 2026-04-25: demoted to DEBUG (was 300/4h spam)
                log.debug(
                    "Reject algo '%s' source=%s conf=%.2f < base=%.2f",
                    c.get("market_title", "?")[:40], source, raw_conf, effective_conf_base,
                )
                continue
            c["confidence"] = round(raw_conf, 4)
            c["min_required"] = round(effective_conf_base, 4)
            scored.append(c)
            continue

        # Auto-fetch api_signal from Polymarket CLOB/Gamma if not already provided
        if "api_signal" not in c:
            c["api_signal"] = polymarket_api.get_api_signal(
                c.get("market_title", ""),
                float(c.get("price", 0.5)),
            )

        # Blend in whale signal — take the higher of api_signal and whale activity
        try:
            import whale_watcher
            whale_sig = whale_watcher.get_whale_signal(c.get("market_title", ""))
            current_api = float(c.get("api_signal", 0.0))
            if whale_sig > current_api:
                c["api_signal"] = whale_sig
                c["whale_boosted"] = True
        except Exception as e:
            log.debug("Whale signal lookup failed: %s", e)

        # WAVE 8 FIX 2026-04-26 (news pipeline integration): pull cached news
        # signal for this market+outcome. Non-blocking (allow_fetch=False) —
        # uses cache only, returns 0.5 neutral on miss. Background news worker
        # keeps cache fresh every NEWS_REFRESH_INTERVAL.
        news_sig = 0.5
        try:
            import news_agent
            news_sig = news_agent.get_news_signal(
                market_title=c.get("market_title", ""),
                outcome=c.get("outcome", ""),
                allow_fetch=False,
            )
        except Exception as _ne:
            log.debug("news signal lookup failed: %s", _ne)

        signal, min_req, passes = compute_confidence(
            trader_score=float(c.get("trader_score", 0)),
            market_price=float(c.get("price", 0.5)),
            market_volume=float(c.get("market_volume", 0)),
            is_insider=bool(c.get("is_insider", False)),
            api_signal=float(c.get("api_signal", 0.0)),
            outcome=str(c.get("outcome", "")),
            side=str(c.get("side", "Buy")),
            conf_base=effective_conf_base,
            news_signal=news_sig,
        )

        if not passes:
            log.info(
                "Reject trade '%s' %s: signal=%.2f < min=%.2f",
                c.get("market_title", "?")[:40],
                c.get("outcome", "?"),
                signal,
                min_req,
            )
            continue

        c["confidence"] = round(signal, 4)
        c["min_required"] = round(min_req, 4)
        scored.append(c)

    # Deduplicate: keep highest-confidence candidate per (market_title, outcome)
    best: dict[tuple, dict] = {}
    for c in scored:
        key = (c.get("market_title", ""), c.get("outcome", ""))
        if key not in best or c["confidence"] > best[key]["confidence"]:
            best[key] = c

    deduped = list(best.values())

    # YES+NO conflict check: if both sides of the same binary market are in the
    # candidate list, keep only the higher-confidence side. Betting both sides of
    # a binary market is financially incoherent.
    market_sides: dict[str, list[dict]] = {}
    for c in deduped:
        market_sides.setdefault(c.get("market_title", ""), []).append(c)
    conflict_free = []
    for title, sides in market_sides.items():
        if len(sides) > 1:
            winner = max(sides, key=lambda x: x["confidence"])
            log.info(
                "Conflict YES+NO '%s' — keeping %s (conf=%.2f)",
                title[:50], winner.get("outcome"), winner["confidence"],
            )
            conflict_free.append(winner)
        else:
            conflict_free.extend(sides)
    deduped = conflict_free

    # Also skip any market+outcome we already have an open position on
    final = []
    for c in deduped:
        if db.is_already_simulated(c.get("market_title", ""), c.get("outcome", "")):
            # ULTRA-REVIEW LOGGING AUDIT 2026-04-25: demoted to DEBUG (was per-cycle spam)
            log.debug(
                "Dedup skip '%s' %s — already have open position",
                c.get("market_title", "?")[:40],
                c.get("outcome", "?"),
            )
            continue
        final.append(c)

    # Sort: insider-sourced first, then by confidence descending
    final.sort(key=lambda c: (0 if c.get("is_insider") else 1, -c["confidence"]))

    if final:
        log.info(
            "Trade selector: %d candidates → %d passed (%.0f%% filtered out)",
            len(candidates),
            len(final),
            100 * (1 - len(final) / max(len(candidates), 1)),
        )

    return final
