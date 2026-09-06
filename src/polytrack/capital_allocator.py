"""Capital Allocator — Phase 3 meta-bot.

Combines signals from all trade sources (copy, insider, future strategies) and
deploys capital optimally using:
  - Expected Value (EV) gating: only trade when the market offers positive edge
  - Fractional Kelly criterion (quarter-Kelly) for position sizing
  - Per-strategy capital budgets (copy=60%, insider=40% of TOTAL_BUDGET_USD)
  - Conviction detection: both copy AND insider agree → strongest signal (+15% confidence)

This sits between copy_manager/insider_detector and the DB log.
The trade_selector module still runs first (confidence gate + dedup).

Usage:
    approved = capital_allocator.allocate(candidates)
    for c in approved:
        db.log_simulated_trade(..., kelly_fraction=c["kelly_fraction"], ev=c["ev"])
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import config
import db
import trade_selector
import regime_detector

log = logging.getLogger("polybot.capital_allocator")


def _is_live_eligible(candidate: dict) -> bool:
    """Returns True if the candidate trade would pass live-mode gates.

    Live-mode only fires trades where ev>0 AND kelly_fraction>0, so calibration
    should train on exactly that same slice — otherwise the training-time
    distribution differs from inference-time distribution.
    """
    return bool(candidate.get("ev", 0) > 0 and candidate.get("kelly_fraction", 0) > 0)


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Polymarket taker fee rates by market category (official docs.polymarket.com/trading/fees)
# fee_amount = feeRate × price × (1 - price)  [per share, highest at p=0.50]
# Geopolitics is free (0.0). This significantly affects EV at p near 0.50.
# ---------------------------------------------------------------------------
_CATEGORY_FEE_RATES: dict[str, float] = {
    "crypto":       0.072,   # 7.2% — highest
    "sports":       0.030,
    "finance":      0.040,
    "politics":     0.040,
    "economics":    0.050,
    "culture":      0.050,
    "weather":      0.050,
    "tech":         0.040,
    "geopolitics":  0.000,   # FREE — best EV markets
    "other":        0.050,
    "default":      0.040,   # conservative default
}


def get_fee_rate(market_title: str = "", source: str = "") -> float:
    """Estimate Polymarket taker fee rate for a market.

    Uses title keyword matching to detect category; falls back to 4% default.
    Geopolitics markets have 0% fees — a free edge advantage.
    """
    title_lower = (market_title or "").lower()

    if any(kw in title_lower for kw in ["bitcoin", "btc", "ethereum", "eth", "crypto", "token", "defi", "nft"]):
        return _CATEGORY_FEE_RATES["crypto"]
    if any(kw in title_lower for kw in ["nba", "nfl", "mlb", "nhl", "soccer", "tennis", "golf",
                                         "championship", "league", "world cup", "olympic", "match"]):
        return _CATEGORY_FEE_RATES["sports"]
    if any(kw in title_lower for kw in ["war", "nato", "military", "invasion", "conflict",
                                         "geopolit", "ukraine", "russia", "china", "taiwan",
                                         "sanctions", "treaty", "un resolution"]):
        return _CATEGORY_FEE_RATES["geopolitics"]
    if any(kw in title_lower for kw in ["election", "president", "senate", "congress",
                                         "vote", "governor", "mayor", "parliament", "poll"]):
        return _CATEGORY_FEE_RATES["politics"]
    if any(kw in title_lower for kw in ["gdp", "inflation", "fed", "interest rate", "cpi",
                                         "unemployment", "recession", "stock", "market cap"]):
        return _CATEGORY_FEE_RATES["finance"]
    if any(kw in title_lower for kw in ["temperature", "rain", "snow", "hurricane",
                                         "weather", "forecast", "degrees"]):
        return _CATEGORY_FEE_RATES["weather"]

    return _CATEGORY_FEE_RATES["default"]


def compute_ev(confidence: float, price: float, market_title: str = "", source: str = "") -> float:
    """Fee-adjusted expected value for a taker Yes/Buy bet.

    Correct formula (Polymarket official fee structure):
        fee_amount = feeRate × price × (1 − price)
        EV = q × (1 − price − fee) − (1−q) × (price + fee)

    This is the true taker EV — including the fee significantly affects the threshold.
    At price=0.50, 4% rate: fee = 0.04 × 0.25 = 0.01 per share (1¢)
    At price=0.50, 7.2% crypto rate: fee = 0.018 per share (1.8¢)
    Geopolitics: fee = 0 → all other things equal, these have the best true EV.

    Returns EV per dollar risked (positive = we have edge after fees).

    Examples:
        confidence=0.70, price=0.50, default rate → fee=0.01 → EV≈+0.38
        confidence=0.70, price=0.50, crypto rate → fee=0.018 → EV≈+0.364
        confidence=0.50, price=0.70, default → EV ≈ −0.305 (no edge)
    """
    # WAVE 7 FIX 2026-04-26 (Round-8 council finding #1): same NaN guard
    # as compute_kelly. NaN propagation through EV → Kelly → bankroll wipeout.
    import math as _math
    if not _math.isfinite(confidence) or not _math.isfinite(price):
        return -1.0  # treat as no-edge so EV gate filters out

    if price < 0 or price > 1:   # boundary fix: allow 0.01 and 0.99 (valid near-certain markets)
        return -1.0
    q = float(confidence)
    p = float(price)
    fee_rate = get_fee_rate(market_title, source)
    fee = fee_rate * p * (1.0 - p)   # per-share fee cost (quadratic, max at p=0.50)
    # Taker pays fee on buy; net proceeds if win = (1 - p - fee), loss = (p + fee)
    return q * (1.0 - p - fee) - (1.0 - q) * (p + fee)


def _dampen_confidence(confidence: float) -> float:
    """Pull extreme confidence toward 0.5 to reduce overbetting (BUG H3).

    KNOWN LIMITATION: Our 'confidence' is a composite score (trader_score * weights
    + insider + volume + api_signal), NOT a calibrated probability of winning.
    Kelly criterion assumes P(win), so feeding an uncalibrated composite score
    causes systematic overbetting at high confidence and underbetting at low.

    This dampening applies a conservative linear shrinkage toward 0.5:
        adjusted = 0.5 + (raw - 0.5) * 0.5
    Effect: 0.80 -> 0.65, 0.60 -> 0.55, 0.30 -> 0.40
    This halves the distance from 0.5, reducing Kelly's sensitivity to
    confidence miscalibration.

    Only applied when KELLY_CONFIDENCE_DAMPEN is True and DRY_RUN is False.
    In DRY_RUN mode we skip dampening to observe raw (undampened) performance
    for calibration data collection.
    """
    return 0.5 + (confidence - 0.5) * 0.5


# ---------------------------------------------------------------------------
# Calibration: map raw composite confidence → calibrated P(win)
# ---------------------------------------------------------------------------

_calibration_cache: dict = {}        # bucket_center → actual_win_rate
_calibration_cache_ts: float = 0.0
_calibration_by_source: dict = {}    # source → {bucket_center → actual_win_rate}
_CALIBRATION_TTL = 1800              # 30-minute cache


def _calibrate_confidence(raw_confidence: float, source: str = None) -> float:
    """Map raw confidence score to calibrated win probability using resolved trade data.

    Queries resolved trades from the DB, groups by confidence bucket (0.0-0.2,
    0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0), and returns the actual win rate for
    the bucket containing raw_confidence.

    Interpolates between bucket centers for smoother estimates.

    Falls back to a conservative mapping when data is sparse (<5 trades in a bucket):
        calibrated = 0.3 + raw * 0.3   (maps 0→0.3, 1→0.6 — compresses toward uncertainty)

    Per-source calibration is used when enough data exists for the specific strategy.

    Args:
        raw_confidence: The composite confidence score (0-1), NOT a true probability.
        source: Strategy name (e.g. 'copy', 'insider', 'scanner') for per-source curves.

    Returns:
        Calibrated win probability (0-1).
    """
    global _calibration_cache, _calibration_cache_ts, _calibration_by_source

    now = time.time()
    if now - _calibration_cache_ts > _CALIBRATION_TTL or not _calibration_cache:
        # WAVE 6 FIX 2026-04-26 (Round-5 council finding #5): only stamp the
        # TTL on SUCCESSFUL refresh. Was unconditional → if refresh raised,
        # ts updated anyway → next 30 min returned stale/empty cache → silent
        # Kelly dampening across all sources. Failure mode is invisible since
        # _conservative_fallback returns plausible values.
        try:
            _refresh_calibration_cache()
            _calibration_cache_ts = now
        except Exception as _e:
            log.warning(
                "calibration refresh failed, retaining prior cache: %s", _e,
            )

    # Try per-source calibration first (if enough data)
    curve = None
    if source and source in _calibration_by_source:
        curve = _calibration_by_source[source]
    if not curve:
        curve = _calibration_cache

    if not curve:
        # No calibration data at all — use conservative fallback
        return _conservative_fallback(raw_confidence)

    return _interpolate_curve(raw_confidence, curve)


def _refresh_calibration_cache():
    """Query resolved trades from DB and build calibration curves.

    WAVE 6 FIX 2026-04-26 (Round-6 council finding #3): atomic swap.
    Was wiping `_calibration_by_source = {}` mid-rebuild — readers in other
    threads saw an empty dict during the ~100ms-1s rebuild window, fell back
    to global cache (which might also be empty if the rebuild raised), and
    got `_conservative_fallback` returning silent uniform calibration. Now
    we build into local dicts and swap atomically at the end so readers
    always see a coherent snapshot (old complete OR new complete, never
    half-built).
    """
    global _calibration_cache, _calibration_by_source

    try:
        conn = db._connect()

        # --- Global calibration (all sources) — build local first ---
        new_global = _build_curve_from_query(conn, source_filter=None)

        # --- Calibration-data health: how much did the live-eligible filter drop? ---
        # We train calibration only on live-eligible rows (ev>0 AND kelly>0). Track how
        # many rows that filter excludes so we can catch regressions (e.g. filter too
        # tight → sparse curve → conservative fallback takes over silently).
        try:
            totals = conn.execute(
                "SELECT COUNT(*) as total, "
                "SUM(CASE WHEN is_live_eligible=1 THEN 1 ELSE 0 END) as eligible "
                "FROM simulated_trades WHERE resolved=1"
            ).fetchone()
            total = int(totals["total"]) if totals and totals["total"] else 0
            eligible = int(totals["eligible"]) if totals and totals["eligible"] else 0
            if total > 0:
                dropped = total - eligible
                drop_pct = 100.0 * dropped / total
                if drop_pct >= 10.0:
                    log.info(
                        "calibration: %d/%d resolved trades live-eligible "
                        "(%.1f%% dropped as non-live-eligible)",
                        eligible, total, drop_pct,
                    )
        except Exception as e:
            log.debug("calibration health check failed: %s", e)

        # --- Per-source calibration — build local dict, swap at end ---
        new_by_source = {}
        sources_rows = conn.execute(
            "SELECT DISTINCT source FROM simulated_trades WHERE resolved=1 AND source IS NOT NULL"
        ).fetchall()
        for row in sources_rows:
            src = row["source"]
            if src:
                src_curve = _build_curve_from_query(conn, source_filter=src)
                if src_curve:
                    new_by_source[src] = src_curve

        conn.close()
        # Atomic swap — readers see either old complete or new complete, never half.
        _calibration_cache = new_global
        _calibration_by_source = new_by_source
    except Exception as e:
        log.warning("Failed to refresh calibration cache: %s", e)


def _build_curve_from_query(conn, source_filter: str = None) -> dict:
    """Build a {bucket_center: actual_win_rate} dict from resolved trades.

    Buckets: 0.0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
    Only includes buckets with >= 5 resolved trades.
    """
    bucket_edges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    curve = {}

    for lo, hi in bucket_edges:
        center = (lo + hi) / 2.0
        # Phase 8: only train calibration on live-eligible trades
        # (ev>0 AND kelly_fraction>0). Trades below that bar never get placed
        # live, so they shouldn't shape the curve used at inference time.
        if source_filter:
            row = conn.execute("""
                SELECT COUNT(*) as n,
                       SUM(CASE WHEN sim_pnl > 0 THEN 1 ELSE 0 END) as wins
                FROM simulated_trades
                WHERE resolved=1 AND confidence >= ? AND confidence < ?
                  AND source = ?
                  AND is_live_eligible = 1
            """, (lo, hi, source_filter)).fetchone()
        else:
            row = conn.execute("""
                SELECT COUNT(*) as n,
                       SUM(CASE WHEN sim_pnl > 0 THEN 1 ELSE 0 END) as wins
                FROM simulated_trades
                WHERE resolved=1 AND confidence >= ? AND confidence < ?
                  AND is_live_eligible = 1
            """, (lo, hi)).fetchone()

        n = int(row["n"]) if row and row["n"] else 0
        # WAVE 6 FIX 2026-04-26 (Round-4 council finding #1): lowered from n>=5
        # to n>=3. Combined with single-bucket fallback in _interpolate_curve,
        # this lets sparse-data sources still get usable buckets without
        # collapsing to constant 0.667 across all confidences.
        if n >= 3:
            wins = int(row["wins"]) if row["wins"] else 0
            curve[center] = wins / n

    return curve


def _conservative_fallback(raw: float) -> float:
    """Conservative mapping when calibration data is sparse.

    Maps 0→0.3, 1→0.6 — compresses all confidence toward uncertainty.
    """
    return 0.3 + raw * 0.3


def _interpolate_curve(raw_confidence: float, curve: dict) -> float:
    """Interpolate between bucket centers for a smoother calibrated probability.

    If raw_confidence falls between two bucket centers with data, linearly interpolate.
    If only one bucket is near, use its value directly.
    If no bucket is close, fall back to conservative mapping.
    """
    centers = sorted(curve.keys())
    if not centers:
        return _conservative_fallback(raw_confidence)

    # WAVE 6 FIX 2026-04-26 (Round-4 council finding #1): single-bucket =
    # NO calibration curve. Was returning curve[centers[0]] for EVERY raw
    # confidence — verified empirically: btc_5min raw=0.10, 0.50, 0.95 all
    # mapped to calibrated=0.667 (the bucket's WR). With low-priced btc
    # strikes Q=2.0, P=0.053, Kelly≈0.65 → max-position bet on every signal
    # regardless of merit. Direct capital-burn at go-live.
    # Now: fall back to conservative shrinkage so size tracks raw signal again.
    if len(centers) == 1:
        return _conservative_fallback(raw_confidence)

    # Clamp to range of available data
    if raw_confidence <= centers[0]:
        return curve[centers[0]]
    if raw_confidence >= centers[-1]:
        return curve[centers[-1]]

    # Find surrounding buckets and interpolate
    for i in range(len(centers) - 1):
        lo_c, hi_c = centers[i], centers[i + 1]
        if lo_c <= raw_confidence <= hi_c:
            span = hi_c - lo_c
            if span == 0:
                return curve[lo_c]
            t = (raw_confidence - lo_c) / span
            return curve[lo_c] * (1.0 - t) + curve[hi_c] * t

    # Should not reach here, but fallback
    return _conservative_fallback(raw_confidence)


# ---------------------------------------------------------------------------
# Price-scaled allocation
# ---------------------------------------------------------------------------

def _price_allocation_scale(price: float) -> float:
    """Scale allocation inversely with risk. Low-price = smaller bet, high-price = bigger bet.

    10c market -> 20% of normal Kelly, 50c -> 100%, 90c -> 100% (capped at 1.0).
    This keeps longshots in the portfolio but sizes them proportionally to their price,
    preventing oversized bets on low-probability events.
    """
    return min(1.0, price * 2.0)


def compute_kelly(confidence: float, price: float, source: str = None) -> float:
    """Fractional Kelly criterion for binary prediction markets.

    Uses the academically correct formula for bounded binary markets (arXiv:2412.14144):

        f* = (Q - P) / (1 + Q)

    where Q = q/(1-q) = our belief odds, P = p/(1-p) = market odds.

    This is more conservative than the standard Kelly at moderate confidence (e.g.
    conf=0.60, price=0.40: standard=0.333, correct=0.20), which reduces risk of
    overbetting. Position-sizing errors hurt quadratically; probability errors hurt
    linearly -- getting this right matters.

    Pipeline:
      1. Calibrate raw confidence → true P(win) using resolved trade data
      2. (Optional) Dampen toward 0.5 if KELLY_CONFIDENCE_DAMPEN=True and live mode
      3. Compute Kelly fraction
      4. (Optional) Scale by market price if KELLY_PRICE_SCALE_ENABLED=True

    Args:
        confidence: Raw composite confidence score (0-1).
        price: Market price / implied probability (0-1).
        source: Strategy name for per-source calibration (e.g. 'copy', 'insider').

    Returns the Kelly fraction (0-1) after calibration and price scaling.
    Multiply by KELLY_FRACTION (0.25) and the strategy budget to get position size.
    Returns 0.0 when Kelly is negative (no edge -- don't bet).
    """
    # WAVE 7 FIX 2026-04-26 (Round-8 council finding #1): NaN/inf guard.
    # Empirically verified: Python's min/max on NaN return the non-NaN side,
    # so the existing `q = max(1e-6, min(1-1e-6, q_raw))` clamp DOESN'T
    # rescue NaN — it converts NaN into ~0.999999, which then yields
    # Kelly ≈ 1.0 (full bankroll bet). This is THE most dangerous bug
    # surfaced this session: a single NaN at confidence input wipes out
    # the bankroll in one trade. Guard at function entry instead.
    import math as _math
    if not _math.isfinite(confidence) or not _math.isfinite(price):
        log.error("compute_kelly got non-finite input: conf=%r price=%r src=%s",
                  confidence, price, source)
        return 0.0

    if price <= 0 or price >= 1:
        return 0.0

    raw_conf = float(confidence)

    # WAVE 6 FIX 2026-04-26 (Round-4 council finding #3): Structural-arb
    # sources (corr_arb, arb, bond) compute confidence as a deterministic
    # edge measure, NOT as a P(win) composite. Routing them through the
    # resolved-trade calibration curve compresses 0.95 → 0.5 (because that
    # curve is built from copy/scanner P(win) data), neutering Kelly and
    # giving 1342 corr_arb candidates → 1 approved per cycle. Bypass
    # calibration for structural sources — their confidence IS the truth.
    STRUCTURAL_SOURCES = {"corr_arb", "arb", "bond"}
    if source in STRUCTURAL_SOURCES:
        calibrated_q = raw_conf
    else:
        # Step 1: Calibrate raw confidence → true P(win) using historical data.
        calibrated_q = _calibrate_confidence(raw_conf, source=source)

    # ULTRA-REVIEW POSITION-SIZING AUDIT 2026-04-25: hard price floor.
    # KELLY_PRICE_SCALE damps but never to zero — at price=$0.025 even scaled
    # Kelly produces $8.55 on $25 bankroll = 34% of capital on a single 2.5¢
    # contract. Refuse anything below MIN_PRICE_FOR_BET (default 0.05 = 5¢).
    min_price = getattr(config, "MIN_PRICE_FOR_BET", 0.05)
    if float(price) < min_price:
        return 0.0

    # Step 2: Apply confidence dampening ON TOP of calibration.
    # ULTRA-REVIEW POSITION-SIZING AUDIT 2026-04-25: removed `not dry_run`
    # guard — was creating a sizing discontinuity at go-live (dampening
    # turns on with no warm-up). Now dampens consistently across modes
    # when KELLY_CONFIDENCE_DAMPEN=True. DRY_RUN data still gets clean raw
    # confidence in the brier_score column (unaffected — that's pre-dampen).
    dampen = getattr(config, "KELLY_CONFIDENCE_DAMPEN", False)
    if dampen:
        q_raw = _dampen_confidence(calibrated_q)
    else:
        q_raw = calibrated_q

    # Clamp confidence to open interval -- prevents div/0 and ensures Q is finite
    # e.g. conviction-boosted confidence can hit exactly 1.0 and would return 0 without clamp
    q = max(1e-6, min(1.0 - 1e-6, q_raw))
    p = float(price)
    P = p / (1.0 - p)       # market odds
    Q = q / (1.0 - q)       # our belief odds
    k = (Q - P) / (1.0 + Q)
    kelly_fraction = max(0.0, k)

    # Step 4: Price-scaled allocation — scale down longshots, keep fair-odds full-sized
    price_scale = 1.0
    if getattr(config, "KELLY_PRICE_SCALE_ENABLED", False):
        price_scale = _price_allocation_scale(price)
        kelly_fraction = kelly_fraction * price_scale

    log.debug(
        "Kelly calibration: raw=%.3f -> calibrated=%.3f, price_scale=%.2f, kelly=%.4f",
        raw_conf, calibrated_q, price_scale, kelly_fraction,
    )

    return kelly_fraction


# ---------------------------------------------------------------------------
# Conviction detection
# ---------------------------------------------------------------------------

def detect_conviction_trades(candidates: list[dict]) -> list[dict]:
    """Find trades where BOTH copy AND insider agree on the same market+outcome.

    When both strategies agree, it's a conviction signal:
      - confidence boosted by 15% (capped at 1.0)
      - source set to 'conviction'
      - keeps the candidate with the higher base confidence as the merged record

    Non-paired candidates pass through unchanged with their original source.

    Args:
        candidates: list of dicts, each with at least: market_title, outcome,
                    source ('copy' or 'insider'), confidence

    Returns:
        New list — conviction pairs merged, unpaired trades unchanged.
    """
    # WAVE 6 FIX 2026-04-26 (Round-5 council finding #2): outcome
    # normalization. Group key was raw outcome string, so 'Yes' vs 'yes'
    # vs 'YES' produced 3 different groups → conviction merge never fired.
    # DB confirmed 0 conviction rows EVER even after Round 4 broadened
    # ALGO_AGREE — the broadening was silently dead because of this bug.
    def _norm_outcome(o):
        s = (o or "").strip().lower()
        if s in ("yes", "y", "true", "1"): return "yes"
        if s in ("no", "n", "false", "0"): return "no"
        return s

    # Group by (market_title-lower, outcome-normalized)
    by_market: dict[tuple, list[dict]] = {}
    for c in candidates:
        key = (
            (c.get("market_title", "") or "").strip().lower(),
            _norm_outcome(c.get("outcome", "")),
        )
        by_market.setdefault(key, []).append(c)

    # WAVE 6 FIX 2026-04-26 (Round-4 council finding #2): broadened gate.
    # Was hardcoded to {copy, insider}. Verified 0 conviction trades EVER
    # written. corr_arb + scanner + arb all share the "copy bucket" but
    # were silently treated as duplicates instead of independent agreement.
    # Now: any 2+ from ALGO_AGREE = conviction (the strongest signal-
    # confirmation lever was disabled for the strategies that need it most).
    ALGO_AGREE = {"copy", "insider", "corr_arb", "scanner", "arb"}
    result = []
    for key, group in by_market.items():
        sources = {c.get("source", "copy") for c in group}
        agreeing = sources & ALGO_AGREE
        if len(agreeing) >= 2:
            # Conviction: pick highest-confidence base, boost it
            best = max(group, key=lambda c: c.get("confidence", 0))
            merged = dict(best)
            merged["confidence"] = min(1.0, best["confidence"] * 1.15)
            # WAVE 8 FIX 2026-04-26 (Round-10 council #4): preserve original
            # source so calibration + budget bucket lookups still work
            # correctly. Was overwriting source="conviction" which:
            # (a) routed through empty per-source calibration → fell back
            #     to global → discarded insider's tighter calibration curve,
            # (b) sourced budget from crowded copy pool instead of insider's
            #     dedicated 20% → starved highest-conviction signals.
            # Track conviction sideband via `is_conviction` + `conviction_sources`.
            merged["is_conviction"] = True
            merged["conviction_sources"] = sorted(agreeing)
            # `source` stays = best.get("source") — preserved via dict(best)
            result.append(merged)
            log.info(
                "Conviction signal: %s %s — sources=%s base=%s conf %.3f → %.3f",
                key[0][:40], key[1], sorted(agreeing),
                merged.get("source", "?"),
                best["confidence"], merged["confidence"],
            )
        else:
            # No pair — all pass through unchanged
            result.extend(group)

    return result


# ---------------------------------------------------------------------------
# Main allocation pipeline
# ---------------------------------------------------------------------------

def allocate(candidates: list[dict]) -> list[dict]:
    """Full capital allocation pipeline.

    Steps:
      1. trade_selector.rank_candidates() — confidence gate + deduplication
      2. EV filter — reject trades below EV_MIN_THRESHOLD
      3. detect_conviction_trades() — merge cross-strategy agreements
      4. Kelly sizing — quarter-Kelly × strategy budget remaining
      5. Per-strategy budget cap — enforce STRATEGY_BUDGETS fractions
      6. MAX_COST_PER_TRADE hard cap

    Args:
        candidates: raw trade candidates from any source (copy, insider).
            Each must have: market_title, outcome, side, price, trader_score,
            market_volume, is_insider (bool), source ('copy'|'insider'),
            trader_address.

    Returns:
        Approved trades with these fields added/updated:
            ev              — expected value per dollar risked
            kelly_fraction  — raw Kelly fraction (before quarter-Kelly mult)
            cost_simulated  — Kelly-sized position in USD
            shares_simulated — cost_simulated / price
            source          — may be upgraded to 'conviction'
    """
    if not candidates:
        return []

    # Fetch regime — adjusts confidence thresholds and max-open limits dynamically
    regime_conf_base = config.CONF_BASE
    try:
        regime = regime_detector.get_regime()
        regime_max_open = regime["max_open"]
        regime_conf_base = regime.get("conf_base", config.CONF_BASE)
        log.debug("Regime: %s | max_open=%d conf_base=%.3f", regime["state"], regime_max_open, regime_conf_base)
    except Exception as e:
        log.warning("Regime detector failed, using config defaults: %s", e)
        regime_max_open = config.MAX_OPEN_POSITIONS

    # Guard: enforce max concurrent open positions
    # In DRY_RUN + DRY_RUN_UNLIMITED mode, skip cap to collect maximum learning data
    open_count = db.get_open_position_count()
    unlimited_mode = config.DRY_RUN and getattr(config, "DRY_RUN_UNLIMITED", False)
    effective_max = regime_max_open if not unlimited_mode else 9999
    if not unlimited_mode and open_count >= effective_max:
        log.info(
            "Max open positions reached (%d/%d regime=%s) — skipping %d candidates",
            open_count, effective_max, regime.get("state", "?"), len(candidates)
        )
        return []

    # Step 0: Log ALL incoming candidates to candidate_signals for backtest replay
    # This captures every signal the bot evaluates, including rejected ones.
    ranked_titles = set()  # track which pass confidence gate

    # Step 1: confidence gate + dedup (existing logic preserved)
    # Apply regime-adjusted conf_base for this allocation round
    ranked = trade_selector.rank_candidates(candidates, conf_base=regime_conf_base)

    ranked_titles = {(c.get("market_title", ""), c.get("outcome", "")) for c in ranked}

    # Log rejected candidates (failed confidence gate)
    for c in candidates:
        key = (c.get("market_title", ""), c.get("outcome", ""))
        if key not in ranked_titles:
            try:
                price = c.get("price", 0.5)
                ev_val = compute_ev(c.get("confidence", 0), price, c.get("market_title", ""), c.get("source", ""))
                db.log_candidate_signal(
                    market_title=c.get("market_title", ""),
                    outcome=c.get("outcome", ""),
                    price=price,
                    confidence=c.get("confidence", 0),
                    ev=round(ev_val, 4),
                    kelly=0.0,
                    source=c.get("source", "copy"),
                    trader_address=c.get("trader_address", ""),
                    market_volume=c.get("market_volume", 0),
                    end_date=c.get("end_date", ""),
                    was_approved=False,
                    rejection_reason="confidence_gate",
                )
            except Exception:
                pass  # never let logging break trading

    if not ranked:
        log.debug("No candidates passed confidence gate")
        return []

    # Step 2: EV filter
    # Bond and arb sources bypass EV gate — they have special risk profiles:
    # bonds: near-certain outcomes where standard EV formula undervalues tiny gains
    # arb: guaranteed profit regardless of outcome (risk-free)
    EV_BYPASS_SOURCES = {"bond", "arb", "corr_arb"}
    ev_passed = []
    for c in ranked:
        source = c.get("source", "copy")
        title = c.get("market_title", "")
        if source in EV_BYPASS_SOURCES:
            c["ev"] = round(compute_ev(c["confidence"], c.get("price", 0.5), title, source), 4)
            ev_passed.append(c)
            continue
        ev = compute_ev(c["confidence"], c.get("price", 0.5), title, source)
        if ev < config.EV_MIN_THRESHOLD:
            log.debug(
                "EV filter drop: '%s' EV=%.3f (min=%.3f)",
                c.get("market_title", "")[:40], ev, config.EV_MIN_THRESHOLD
            )
            # Log EV-rejected candidate
            try:
                db.log_candidate_signal(
                    market_title=c.get("market_title", ""),
                    outcome=c.get("outcome", ""),
                    price=c.get("price", 0.5),
                    confidence=c.get("confidence", 0),
                    ev=round(ev, 4),
                    kelly=0.0,
                    source=source,
                    trader_address=c.get("trader_address", ""),
                    market_volume=c.get("market_volume", 0),
                    end_date=c.get("end_date", ""),
                    was_approved=False,
                    rejection_reason="ev_filter",
                )
            except Exception:
                pass
            continue
        c["ev"] = round(ev, 4)
        ev_passed.append(c)

    if not ev_passed:
        log.info("All candidates filtered out by EV threshold (%.2f)", config.EV_MIN_THRESHOLD)
        return []

    # Step 2.5: LLM swarm gate for high-confidence trades
    swarm_threshold = getattr(config, "SWARM_CONFIDENCE_THRESHOLD", 0.62)
    swarm_enabled = getattr(config, "SWARM_ENABLED", False)
    swarm_passed = []
    if swarm_enabled:
        import llm_swarm
        for c in ev_passed:
            if c.get("confidence", 0) >= swarm_threshold:
                try:
                    result = llm_swarm.evaluate(c)
                except Exception as e:
                    log.warning("Swarm eval failed for '%s': %s — proceeding without swarm",
                                c.get("market_title", "")[:40], e)
                    swarm_passed.append(c)
                    continue
                if not result["proceed"]:
                    log.info(
                        "Swarm rejected: '%s' — %s",
                        c.get("market_title", "")[:40], result["reason"]
                    )
                    continue
                c["confidence"] = result["confidence"]  # may be lowered on split vote
                c["swarm_votes"] = result["votes"]
            swarm_passed.append(c)
    else:
        swarm_passed = ev_passed

    if not swarm_passed:
        log.info("All candidates filtered by LLM swarm")
        return []

    # Step 3: conviction detection
    merged = detect_conviction_trades(swarm_passed)

    # Step 4-6: Kelly sizing with per-strategy budget enforcement
    open_costs = db.get_open_cost_by_strategy()
    approved = []
    # In unlimited learning mode there's no slot cap; use large number
    slots_available = len(merged) if unlimited_mode else (effective_max - open_count)

    for c in merged:
        if len(approved) >= slots_available:
            log.info("Position slots filled (%d new this batch) — stopping allocation", len(approved))
            break
        source = c.get("source", "copy")
        price = float(c.get("price", 0.5))

        # Determine budget bucket: conviction, scanner, arb draw from the copy pool.
        # Bond has its own pool now (BUG M6: was overloading "copy" with 5 strategies).
        COPY_BUCKET_SOURCES = {"conviction", "scanner", "arb", "corr_arb"}
        budget_key = "copy" if source in COPY_BUCKET_SOURCES else source
        budget_fraction = config.STRATEGY_BUDGETS.get(budget_key, 0.10)  # low fallback if unknown
        # MARATHON SHIP-10 (2026-04-26 audit Dry-run Phase E): drainable bankroll.
        # Was `strategy_budget = TOTAL_BUDGET_USD * budget_fraction` (static).
        # Drawdown didn't reduce sizing — Kelly never saw the loss-feedback signal
        # that LIVE mode would see (live wallet drains on losses).
        # Now: subtract realized cumulative loss from the bankroll so Kelly auto-
        # de-risks during drawdown periods. Wins replenish. Live behavior parity.
        try:
            realized_pnl = float(db.get_total_resolved_pnl()) if hasattr(db, "get_total_resolved_pnl") else 0.0
        except Exception:
            realized_pnl = 0.0
        effective_bankroll = max(
            float(getattr(config, "MIN_TRADE_SIZE_USD", 0.5)),
            float(config.TOTAL_BUDGET_USD) + realized_pnl,
        )
        strategy_budget = effective_bankroll * budget_fraction

        # Current spend in this bucket (open positions only)
        # For shared-pool sources, sum all sources that draw from the same pool
        if source in COPY_BUCKET_SOURCES or source == "copy":
            strategy_spent = sum(
                open_costs.get(s, 0.0)
                for s in list(COPY_BUCKET_SOURCES) + ["copy"]
            )
        else:
            strategy_spent = open_costs.get(source, 0.0)

        budget_remaining = max(0.0, strategy_budget - strategy_spent)
        if budget_remaining <= 0 and not unlimited_mode:
            log.info(
                "Budget exhausted for strategy '%s' ($%.2f/$%.2f) — skipping '%s'",
                source, strategy_spent, strategy_budget,
                c.get("market_title", "")[:35]
            )
            continue
        # In unlimited learning mode, always have budget available
        if unlimited_mode and budget_remaining <= 0:
            budget_remaining = config.MAX_COST_PER_TRADE * 2  # synthetic budget for sizing

        # Kelly sizing (with calibration + price scaling)
        k = compute_kelly(c["confidence"], price, source=source)
        raw_size = config.KELLY_FRACTION * k * budget_remaining

        # WAVE 6 FIX 2026-04-26 (council finding #2): KELLY_MAX_MULTIPLIER was
        # documented as a Thorp-style safety cap (effective Kelly ≤ 0.3125)
        # and referenced by c04_sizing_contract probe + LIVE_CONSTRAINTS
        # preflight, but was NEVER actually enforced here. With high
        # confidence + price-scale boosts, raw Kelly could exceed 1.0 (full
        # bankroll bet). Now: cap raw_size at KELLY_MAX_MULTIPLIER × base
        # quarter-Kelly × budget. Defensive even though MAX_COST_PER_TRADE
        # also clips downstream — keeps the invariant the comment claimed.
        kmax = float(getattr(config, "KELLY_MAX_MULTIPLIER", 1.0))
        kelly_cap = config.KELLY_FRACTION * kmax * budget_remaining
        raw_size = min(raw_size, kelly_cap)

        # WAVE 6 FIX 2026-04-26 (Round-5 council finding #4): structural-arb
        # max-cost split. Round 4 unlocked Kelly for corr_arb/arb/bond (was
        # forced to ~0 by calibration) — now Kelly correctly says e.g. $124
        # for a guaranteed-profit ME_ARB. But MAX_COST_PER_TRADE=$50 silently
        # caps every approval back to flat $50, leaving 60% of structural
        # edge on the table. New: separate cap for structural sources
        # (default 5x normal; can be tuned via config). Risk-bounded because
        # these are guaranteed-edge signals (per EV_BYPASS_SOURCES).
        STRUCTURAL_FOR_CAP = {"corr_arb", "arb", "bond"}
        if source in STRUCTURAL_FOR_CAP:
            structural_cap = float(getattr(
                config, "MAX_COST_PER_TRADE_STRUCTURAL",
                config.MAX_COST_PER_TRADE * 5,
            ))
            max_cost_eff = structural_cap
        else:
            max_cost_eff = config.MAX_COST_PER_TRADE
        # Apply caps and floor — never exceed budget remaining
        size = min(raw_size, max_cost_eff, budget_remaining)
        size = max(size, config.MIN_TRADE_SIZE_USD)
        if size > budget_remaining:
            # Floor would exceed budget; skip this trade rather than overspend
            log.debug("Size floor $%.2f > budget $%.2f — skipping '%s'",
                      size, budget_remaining, c.get("market_title", "")[:35])
            continue

        c["kelly_fraction"] = round(k, 4)
        c["cost_simulated"] = round(size, 4)
        c["shares_simulated"] = round(size / price, 4) if price > 0 else 0.0

        # Serialize swarm votes for DB persistence (if present)
        if "swarm_votes" in c and isinstance(c["swarm_votes"], dict):
            import json as _json
            c["swarm_votes_json"] = _json.dumps(c["swarm_votes"])
        else:
            c["swarm_votes_json"] = None

        log.info(
            "[ALLOC] %s '%s' %s @ $%.3f | EV=%.3f conf=%.2f kelly=%.3f size=$%.2f",
            source.upper(), c.get("market_title", "")[:35], c.get("outcome", ""),
            price, c.get("ev", 0), c["confidence"], k, size
        )
        approved.append(c)

    return approved
