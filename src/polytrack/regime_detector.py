"""Market regime detector — 5-state rule-based classifier.

Fetches aggregate Polymarket data from Gamma API every REGIME_INTERVAL seconds,
computes 3 features across top-100 active markets, and classifies the current
trading environment into one of 5 states.

States and their effect on capital allocation:
    CRASH    — high spread / very low volume. Very cautious.
    BEAR     — wide spread and low yes-prices. Conservative.
    NEUTRAL  — default state.
    BULL     — tight spread, normal yes-prices. Slightly aggressive.
    EUPHORIA — market-wide overconfidence. Cautious (counter-trend).

Used by capital_allocator.allocate() to dynamically adjust confidence thresholds
and max-open-positions limits without touching config.py at runtime.

Usage:
    regime = regime_detector.get_regime()
    # {"state": "BULL", "conf_base": 0.33, "max_open": 5, "features": {...}}
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

import config

log = logging.getLogger("polybot.regime_detector")

# ---------------------------------------------------------------------------
# State definitions
# ---------------------------------------------------------------------------

# Each state: (conf_base_delta, max_open_override)
_STATES = {
    "CRASH":    (+0.10, 1),
    "BEAR":     (+0.05, 3),
    "NEUTRAL":  (0.00,  5),
    "BULL":     (-0.02, 5),
    "EUPHORIA": (+0.08, 2),
}

# Cache: refresh at most every REGIME_INTERVAL seconds
_cache: dict = {}
_cache_ts: float = 0.0


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _fetch_regime_markets(n: int = 100) -> list[dict]:
    """Fetch top-N active markets from Gamma API for regime feature computation."""
    try:
        r = requests.get(
            f"{config.POLYMARKET_GAMMA_URL}/markets",
            params={"active": "true", "closed": "false", "limit": n, "_order": "volume"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else data.get("data", [])
    except Exception as e:
        log.warning("Regime market fetch failed: %s", e)
        return []


def _compute_features(markets: list[dict]) -> dict:
    """Compute aggregate regime features from a list of Gamma market dicts."""
    if not markets:
        return {"avg_yes_price": 0.5, "avg_spread_pct": 3.0, "avg_volume": 5000.0}

    yes_prices = []
    spreads = []
    volumes = []

    for mkt in markets:
        # Yes price
        try:
            import json
            prices_raw = mkt.get("outcomePrices", "[]")
            prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
            if prices:
                yes_prices.append(float(prices[0]))
        except (ValueError, TypeError, Exception):
            pass

        # Spread
        try:
            best_bid = float(mkt.get("bestBid", 0) or 0)
            best_ask = float(mkt.get("bestAsk", 0) or 0)
            if best_bid > 0 and best_ask > 0:
                mid = (best_bid + best_ask) / 2.0
                if mid > 0:
                    spreads.append((best_ask - best_bid) / mid * 100.0)
        except (ValueError, TypeError):
            pass

        # Volume
        try:
            vol = float(mkt.get("volume", 0) or mkt.get("liquidity", 0) or 0)
            if vol > 0:
                volumes.append(vol)
        except (ValueError, TypeError):
            pass

    # Filter out near-certain markets (price > 0.90) for EUPHORIA detection
    # These are resolved/near-certain events that inflate avg_yes artificially
    yes_prices_filtered = [p for p in yes_prices if p <= 0.90]

    return {
        "avg_yes_price":  round(sum(yes_prices) / len(yes_prices), 4) if yes_prices else 0.5,
        "avg_yes_price_filtered": round(sum(yes_prices_filtered) / len(yes_prices_filtered), 4) if yes_prices_filtered else 0.5,
        "avg_spread_pct": round(sum(spreads) / len(spreads), 4) if spreads else 3.0,
        "avg_volume":     round(sum(volumes) / len(volumes), 0) if volumes else 5000.0,
        "market_count":   len(markets),
    }


def _classify(features: dict) -> str:
    """Apply rule-based classifier to features → regime state name."""
    avg_spread = features["avg_spread_pct"]
    avg_yes    = features["avg_yes_price"]
    avg_vol    = features["avg_volume"]

    # CRASH: extreme spread or very low volume — market is broken
    if avg_spread > 5.0 or avg_vol < 1000:
        return "CRASH"

    # EUPHORIA: widespread overconfidence — high yes prices or razor-thin spreads with big volume
    # Use filtered avg_yes that excludes near-certain markets (price > 0.90)
    # to avoid false EUPHORIA triggers from resolved/near-certain events
    filtered_avg_yes = features.get("avg_yes_price_filtered", avg_yes)
    if filtered_avg_yes > 0.75 or (avg_spread < 1.0 and avg_vol > 50_000):
        return "EUPHORIA"

    # BEAR: wide spread + low yes prices — market fearful/illiquid
    if avg_spread > 3.0 and avg_yes < 0.30:
        return "BEAR"

    # BULL: tight spread + yes prices in healthy range — healthy, active market
    if avg_spread < 2.0 and 0.30 <= avg_yes <= 0.60:
        return "BULL"

    # Default
    return "NEUTRAL"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_regime(force_refresh: bool = False) -> dict:
    """Compute or return cached current market regime.

    Returns:
        {
            "state":     "BULL",       # one of CRASH/BEAR/NEUTRAL/BULL/EUPHORIA
            "conf_base": 0.33,         # adjusted confidence base for this regime
            "max_open":  5,            # adjusted max open positions for this regime
            "features":  {...},        # raw features used for classification
            "cached":    True/False,
        }
    """
    global _cache, _cache_ts

    now = time.time()
    interval = getattr(config, "REGIME_INTERVAL", 1800)

    if not force_refresh and _cache and (now - _cache_ts) < interval:
        return dict(_cache, cached=True)

    sample_size = getattr(config, "REGIME_SAMPLE_SIZE", 100)
    markets = _fetch_regime_markets(n=sample_size)
    features = _compute_features(markets)
    state = _classify(features)

    # WAVE 10 SHIP-5 (2026-04-26 post-Marathon audit Dry-run #6): removed
    # the "DRY_RUN forces NEUTRAL" override. Was a parity violation —
    # the CRASH/EUPHORIA codepath (tightened conf_base + max_open) had
    # ZERO test coverage in production data. We would flip DRY_RUN=False
    # and discover its bugs in real money. Now: dry-run sees the same
    # regime the live bot would see — the protective tightening is
    # exercised continuously instead of being a black box on go-live day.
    # If "data collection continues" was the original concern, the
    # widened LEARNING_MODE thresholds elsewhere already preserve volume;
    # losing the regime parity test cost more than it saved.

    conf_delta, max_open = _STATES[state]
    conf_base = round(config.CONF_BASE + conf_delta, 4)

    result = {
        "state":    state,
        "conf_base": conf_base,
        "max_open":  max_open,
        "features":  features,
        "cached":    False,
    }

    _cache = result
    _cache_ts = now

    log.info(
        "Regime: %s | conf_base=%.2f max_open=%d | avg_yes=%.2f avg_spread=%.1f%% avg_vol=%.0f",
        state, conf_base, max_open,
        features["avg_yes_price"], features["avg_spread_pct"], features["avg_volume"],
    )

    return result
