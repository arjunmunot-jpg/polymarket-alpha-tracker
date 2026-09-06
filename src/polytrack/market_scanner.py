"""Market Scanner — proactive underpriced market discovery.

Inspired by AI Pathways / MoonDev research:
  "Prediction market edge lives in underpriced Yes bets ($0.10–$0.45)
   where asymmetric upside is highest and crowd sentiment hasn't caught up."

Scans Polymarket Gamma API for active markets that meet all of:
  - Price in SCANNER_PRICE_MIN–SCANNER_PRICE_MAX (0.10–0.45)
  - Volume >= SCANNER_MIN_VOLUME ($10k daily)
  - Resolves within MAX_RESOLVE_DAYS (7 days)
  - Spread < MAX_SPREAD_PCT (2%)
  - Orderbook shows bid imbalance > 55% (smart money accumulating)

Passes qualifying markets through the full capital_allocator pipeline
(confidence gate → EV filter → Kelly sizing) before logging.

Source label: "scanner"
Strategy budget: shared with "copy" bucket (both are market opportunity signals)
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Optional

import requests

import config
import capital_allocator
import db
import polymarket_api  # WAVE 12 SHIP-2: token_id resolution for fill_simulator wire

log = logging.getLogger("polybot.market_scanner")


# ---------------------------------------------------------------------------
# Finance / crypto category filter (word-boundary regex)
# ---------------------------------------------------------------------------
# Uses \b word boundaries to prevent false positives:
#   "eth" won't match "Netherlands", "avalanche" won't match NHL teams,
#   "sol" won't match "solution", "ada" won't match "Canada"
# Removed "price" (too generic — matches any price mention in sports markets)

_FINANCE_PATTERN = re.compile(
    r"(?i)\b(?:"
    # Crypto / digital assets
    r"bitcoin|btc|ethereum|eth|crypto(?:currency)?|solana|xrp|dogecoin|doge"
    r"|cardano|polkadot|bnb|stablecoin|defi|airdrop|token|nft"
    r"|megaeth|chainlink|uniswap|coinbase|binance|kraken|microstrategy"
    # Macro / monetary
    r"|fed(?:eral reserve)?|interest rate|cpi|inflation|gdp|tariff"
    r"|s&p|nasdaq|dow jones|stock market|dollar|treasury|bond yield"
    r"|fomc|powell|ecb|recession|ipo|market cap"
    # COUNCIL-PROFIT-RESEARCH-2026-04-25: politics + geopolitics tokens.
    # Research note 08 identifies "longshot-bias fade" (10¢-90¢ asymmetry)
    # as the missing pattern. Geopolitics has 0% fees per capital_allocator.py:61
    # — the best EV slot. These tokens are scanner-domain (verified, not sports).
    r"|election|congress|senate|war|nato|ukraine|russia|iran|israel"
    r"|ceasefire|treaty|sanctions|impeach\w*|geopolit\w*"
    # ULTRA-REVIEW BOND+SCANNER AUDIT 2026-04-25: expansion pack adds ~1000 politics
    # markets that were previously stranded (president+nominee+primary+democrat+
    # republican+governor + key geo names). Coverage: 30.7% → ~50% of Polymarket.
    r"|nominee|nomination|primary|president(?:ial)?|democrat\w*|republican\w*"
    r"|trump|biden|harris|vance|governor|mayor"
    r"|china|taiwan|korea|gaza|venezuela|cuba"
    r")\b"
)


def _is_finance_market(title: str) -> bool:
    """True if the market title contains crypto/finance/macro keywords.

    Uses word-boundary regex to avoid false positives like 'Netherlands' matching 'eth'.
    """
    return bool(_FINANCE_PATTERN.search(title))


# ---------------------------------------------------------------------------
# Claude-based probability estimation (replaces price-derived confidence)
# ---------------------------------------------------------------------------

# Module-level cache: {market_title: {"probability": float, "confidence": float,
#                                      "reasoning": str, "ts": float}}
_claude_estimate_cache: dict = {}

# Counter for rate limiting within a single scan cycle
_claude_calls_this_cycle: int = 0


def _reset_claude_cycle_counter() -> None:
    """Reset the per-cycle Claude call counter. Called at the start of each scan."""
    global _claude_calls_this_cycle
    _claude_calls_this_cycle = 0


def _estimate_probability_with_claude(market: dict) -> dict:
    """Ask Claude to estimate the true probability of a market outcome.

    Uses the same anthropic SDK pattern as llm_swarm._call_claude().

    Args:
        market: dict with title, description, current price, volume, end date

    Returns:
        {"probability": float, "confidence": float, "reasoning": str}
        On failure, returns a fallback dict with probability=None.
    """
    global _claude_calls_this_cycle

    title = market.get("market_title", "") or market.get("question", "") or market.get("title", "")
    price = float(market.get("price", 0.5))
    volume = float(market.get("market_volume", 0) or market.get("volume", 0) or 0)
    end_date = market.get("end_date", "") or market.get("endDate", "") or market.get("endDateIso", "")
    description = market.get("description", "") or market.get("question", title)
    outcome = market.get("outcome", "Yes")

    cache_ttl = getattr(config, "SCANNER_CLAUDE_CACHE_TTL", 1800)
    max_calls = getattr(config, "SCANNER_CLAUDE_MAX_CALLS", 10)

    # --- Cache check ---
    cache_key = f"{title}|{outcome}"
    now = time.time()
    cached = _claude_estimate_cache.get(cache_key)
    if cached and (now - cached["ts"]) < cache_ttl:
        log.debug("Claude estimate cache HIT: '%s'", title[:40])
        return {
            "probability": cached["probability"],
            "confidence": cached["confidence"],
            "reasoning": cached["reasoning"],
        }

    # --- Rate limit check ---
    if _claude_calls_this_cycle >= max_calls:
        log.debug("Claude rate limit reached (%d/%d), skipping '%s'", _claude_calls_this_cycle, max_calls, title[:40])
        return {"probability": None, "confidence": None, "reasoning": "rate_limit_exceeded"}

    # --- Build prompt ---
    prompt = f"""You are a prediction market analyst estimating the TRUE probability of an event.

Market question: "{title}"
Description: {description[:500] if description != title else "Not available"}
Current YES price: ${price:.2f} ({price * 100:.0f}% implied probability)
Market outcome being evaluated: {outcome}
Market volume: ${volume:,.0f}
End date: {end_date if end_date else "Unknown"}

Your task:
1. Estimate the TRUE probability that this market resolves {outcome} (0.00 to 1.00)
2. Rate your confidence in this estimate (0.00 = pure guess, 1.00 = certain)
3. Consider: base rates, current events, historical patterns, and whether the current price is too high or too low

IMPORTANT: Be independent from the market price. The market may be wrong.

Respond in EXACTLY this JSON format (no markdown, no extra text):
{{"probability": 0.XX, "confidence": 0.XX, "reasoning": "one sentence explanation"}}"""

    # --- Call Claude ---
    try:
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = response.content[0].text.strip()
        _claude_calls_this_cycle += 1

        # Parse JSON response — handle possible markdown wrapping
        json_text = raw_text
        if "```" in json_text:
            # Strip markdown code fences
            json_text = re.sub(r"```(?:json)?\s*", "", json_text)
            json_text = json_text.strip()
        result = json.loads(json_text)

        prob = float(result.get("probability", 0.5))
        conf = float(result.get("confidence", 0.3))
        reasoning = str(result.get("reasoning", ""))

        # Clamp values
        prob = max(0.01, min(0.99, prob))
        conf = max(0.05, min(0.95, conf))

        # Store in cache
        _claude_estimate_cache[cache_key] = {
            "probability": prob,
            "confidence": conf,
            "reasoning": reasoning,
            "ts": now,
        }

        log.info(
            "Claude estimate for '%s' %s: prob=%.2f conf=%.2f (market=%.2f, divergence=%.2f) — %s",
            title[:40], outcome, prob, conf, price, abs(prob - price), reasoning[:80],
        )

        return {"probability": prob, "confidence": conf, "reasoning": reasoning}

    except json.JSONDecodeError as e:
        _claude_calls_this_cycle += 1
        log.warning("Claude estimate JSON parse failed for '%s': %s (raw: %s)", title[:40], e, raw_text[:100])
        return {"probability": None, "confidence": None, "reasoning": f"json_parse_error: {e}"}
    except Exception as e:
        log.warning("Claude estimate failed for '%s': %s", title[:40], e)
        return {"probability": None, "confidence": None, "reasoning": f"api_error: {e}"}


def _compute_claude_confidence(
    claude_prob: float,
    claude_conf: float,
    market_price: float,
    outcome: str,
) -> float:
    """Compute trade confidence from Claude's probability estimate.

    The signal is the DIVERGENCE between Claude's estimated probability and
    the market price, weighted by Claude's confidence in its own estimate.

    Only generates a trade signal when Claude thinks the market is meaningfully
    wrong (divergence > fee + minimum edge).

    Args:
        claude_prob:  Claude's estimated probability (0-1)
        claude_conf:  Claude's confidence in its estimate (0-1)
        market_price: current market price for this outcome (0-1)
        outcome:      "Yes" or "No"

    Returns:
        Confidence score for the trade (0-1). Returns 0.0 if no edge detected.
    """
    # For YES bets: edge = claude_prob - market_price (Claude thinks it's worth more)
    # For NO bets: edge = (1 - claude_prob) - (1 - market_price) = market_price - claude_prob
    if outcome.lower() in ("yes", "up"):
        edge = claude_prob - market_price
    else:
        edge = market_price - claude_prob

    # Compute Polymarket fee for this price point
    fee_rate = getattr(config, "POLYMARKET_BASE_FEE_RATE", 0.022)
    fee = fee_rate * market_price * (1.0 - market_price)
    min_edge = fee + 0.02  # fee + 2% minimum edge

    if edge < min_edge:
        return 0.0  # no tradeable edge

    # Scale confidence: larger edge + higher Claude confidence = higher signal
    # Normalize edge to 0-1 range (max realistic edge ~0.40)
    edge_factor = min(edge / 0.40, 1.0)
    signal = 0.40 + edge_factor * 0.40 + claude_conf * 0.20
    return round(min(0.95, max(0.05, signal)), 4)


# ---------------------------------------------------------------------------
# Gamma API fetch
# ---------------------------------------------------------------------------


def _fetch_active_markets(limit: int = 100) -> list[dict]:
    """Fetch active, non-closed markets from Gamma API sorted by volume descending."""
    try:
        r = requests.get(
            f"{config.POLYMARKET_GAMMA_URL}/markets",
            params={
                "active": "true",
                "closed": "false",
                "limit": limit,
                "_order": "volume",
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else data.get("data", [])
    except Exception as e:
        log.warning("Gamma markets fetch failed: %s", e)
        return []


def _fetch_all_markets_via_events(max_events: int = 100) -> list[dict]:
    """Fetch markets via the /events endpoint which returns nested markets.

    The events endpoint is much richer than /markets: one call with 50 events
    yields ~800+ nested markets vs 200 from /markets. We paginate to get
    enough coverage. The /markets q= search is broken (returns random trending
    results regardless of query), so we fetch broadly and filter locally.
    """
    seen_ids: set = set()
    all_markets: list[dict] = []

    for offset in range(0, max_events, 50):
        try:
            r = requests.get(
                f"{config.POLYMARKET_GAMMA_URL}/events",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": 50,
                    "offset": offset,
                },
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
            events = data if isinstance(data, list) else data.get("data", [])
            if not events:
                break

            for event in events:
                for m in event.get("markets", []):
                    mid = m.get("id") or m.get("conditionId") or ""
                    if mid and mid not in seen_ids:
                        seen_ids.add(mid)
                        all_markets.append(m)
        except Exception as e:
            log.debug("Events fetch offset=%d failed: %s", offset, e)
            break

    log.info("Events fetch: %d unique markets from %d API pages", len(all_markets), (offset // 50) + 1)
    return all_markets


def _parse_yes_outcome(market: dict):
    """WAVE 6 FIX 2026-04-26 (Round-7 council finding #5): explicit Yes-finder.

    Polymarket does NOT guarantee outcomes[0] is "Yes". Directional markets
    like ["Down","Up"] silently mis-classify as bond-eligible when they're
    actually directional. Returns (outcome_name, price) for the Yes side
    only, or None if no "Yes"/"true"/"y" outcome exists.
    """
    try:
        prices_raw = market.get("outcomePrices", "[]")
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
        outcomes_raw = market.get("outcomes", "[]")
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
        if not prices or not outcomes or len(prices) != len(outcomes):
            return None
        for name, p in zip(outcomes, prices):
            if str(name).strip().lower() in ("yes", "true", "y"):
                return (str(name), float(p))
        return None
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def _parse_outcome_price(market: dict) -> Optional[float]:
    """Extract the Yes/Up outcome price from a Gamma market dict.

    Gamma returns outcomes as a JSON-encoded string and outcomePrices similarly.
    Returns the first (Yes) outcome price, or None if parsing fails.
    """
    try:
        prices_raw = market.get("outcomePrices", "[]")
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
        if prices:
            return float(prices[0])
    except (ValueError, TypeError, json.JSONDecodeError):
        pass

    # Fallback: use bestBid/bestAsk midpoint
    best_bid = market.get("bestBid")
    best_ask = market.get("bestAsk")
    if best_bid is not None and best_ask is not None:
        try:
            return (float(best_bid) + float(best_ask)) / 2.0
        except (ValueError, TypeError):
            pass

    return None


def _compute_spread_pct(market: dict, price: float) -> float:
    """Compute bid-ask spread % from Gamma market fields."""
    try:
        spread = float(market.get("spread", 0) or 0)
        best_bid = float(market.get("bestBid", price) or price)
        best_ask = float(market.get("bestAsk", price + spread) or (price + spread))
        mid = (best_bid + best_ask) / 2.0
        if mid > 0:
            return (best_ask - best_bid) / mid * 100.0
    except (ValueError, TypeError):
        pass
    return 99.0


def _days_to_resolve(market: dict) -> Optional[int]:
    """Return days until market resolution, or None if unknown.

    Returns negative values for already-expired markets — callers must
    check for this (negative = already expired, should be skipped).
    """
    end_str = market.get("endDate", "") or market.get("endDateIso", "")
    if not end_str:
        return None
    try:
        end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
        return (end_dt - datetime.now(timezone.utc)).days
    except (ValueError, TypeError):
        return None


def _get_outcome_name(market: dict) -> str:
    """Extract the first outcome name (Yes, Up, etc.)."""
    try:
        outcomes_raw = market.get("outcomes", '["Yes","No"]')
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
        if outcomes:
            return str(outcomes[0])
    except (ValueError, TypeError, json.JSONDecodeError):
        pass
    return "Yes"


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------

def _get_price_bucket(price: float) -> str:
    """Classify a bond price into a human-readable bucket for tracking.

    Buckets:
        85-89c  = "deep discount" (highest risk, highest return if it resolves)
        90-92c  = "discount"
        93-95c  = "standard"
        96-98c  = "premium" (safest, lowest return)
        99c     = "near-par"
    """
    if price < 0.90:
        return "85-89c (deep discount)"
    if price < 0.93:
        return "90-92c (discount)"
    if price < 0.96:
        return "93-95c (standard)"
    if price < 0.99:
        return "96-98c (premium)"
    return "99c (near-par)"


def _log_bond_price_bucket(price: float, market_title: str) -> None:
    """Log the price bucket when a bond trade is placed.

    Data collection for future entry-point optimization: tracks which price
    buckets produce the most bond trades so we can correlate with resolution
    rates over time.
    """
    bucket = _get_price_bucket(price)
    log.info(
        "Bond PLACED bucket=%s price=$%.3f market='%s'",
        bucket, price, market_title[:60],
    )


def scan_high_probability_bonds() -> list[dict]:
    """Scan for near-certain YES outcomes priced $0.92–$0.99.

    Near-certain markets that haven't resolved yet trade at a slight discount.
    Documented annualized return: ~520% at scale (ChainCatcher 95M tx analysis).
    Risk is near-zero; gain is small but highly predictable.

    Bypasses EV filter in capital_allocator (source="bond") because standard EV
    formula undervalues these tiny-gain, near-zero-risk positions.

    Diversification: No single market may exceed BOND_MAX_POSITION_PCT (20%) of
    the total bond budget. This limits black swan exposure — if one "near-certain"
    market resolves NO, the loss is capped at 20% of bond capital.
    """
    if not getattr(config, "BOND_SCANNER_ENABLED", False):
        return []

    import risk_guard
    safe, reason = risk_guard.check_safe_to_trade()
    if not safe:
        log.warning("Bond scanner blocked by risk guard: %s", reason)
        return []

    log.info("=== Bond scanner: scanning for near-certain YES outcomes ===")

    # --- Diversification setup: query existing bond positions ---
    bond_budget_fraction = config.STRATEGY_BUDGETS.get("bond", 0.10)
    bond_budget_total = config.TOTAL_BUDGET_USD * bond_budget_fraction
    max_per_market = bond_budget_total * getattr(config, "BOND_MAX_POSITION_PCT", 0.20)

    open_bond_positions = db.get_open_bond_positions_by_market()
    total_bond_deployed = sum(open_bond_positions.values())

    log.info(
        "Bond diversification: %d open positions, $%.2f/$%.2f deployed (%.1f%% of bond budget), "
        "max $%.2f per market",
        len(open_bond_positions), total_bond_deployed, bond_budget_total,
        (total_bond_deployed / bond_budget_total * 100.0) if bond_budget_total > 0 else 0.0,
        max_per_market,
    )
    for mkt_title, cost in sorted(open_bond_positions.items(), key=lambda x: -x[1]):
        pct = (cost / bond_budget_total * 100.0) if bond_budget_total > 0 else 0.0
        log.info(
            "  Bond position: '%s' = $%.2f (%.1f%% of budget)",
            mkt_title[:50], cost, pct,
        )

    # Use events endpoint for broader coverage. Bond scanner has NO category filter
    # (any near-certain market is a valid bond), so cast a wide net.
    markets = _fetch_all_markets_via_events(max_events=100)
    if not markets:
        return []

    candidates = []
    skipped_diversification = 0
    skipped_finance = 0
    skipped_dedup = 0
    price_bucket_counts = {}  # type: dict[str, int]

    for mkt in markets:
        title = mkt.get("question", "") or mkt.get("title", "")
        if not title:
            continue

        # ULTRA-REVIEW BOND+SCANNER AUDIT 2026-04-25: bond bypasses finance filter.
        # Bond thesis is "near-certain YES at $0.92–$0.99" — a 95% sports favorite
        # resolves just as predictably as a 95% crypto favorite. The C1 finance
        # over-correction caused 10-day darkness. Bond now uses BOND_FINANCE_ONLY
        # config flag (default False) — opt-in to re-enable.
        if getattr(config, "BOND_FINANCE_ONLY", False) and not _is_finance_market(title):
            skipped_finance += 1
            continue

        # WAVE 6 FIX 2026-04-26 (Round-7 council finding #5): bond explicitly
        # validates the Yes-side. _parse_outcome_price returned outcomes[0]
        # blindly — for `["Down","Up"]`-style directional markets, that
        # would buy "Down" at $0.95 thinking it was the favored side. Bond
        # bypasses EV gates → wrong-side bet at 0.95 is EV ≈ -0.85/dollar.
        # Now: use _parse_yes_outcome which only returns markets with an
        # explicit "Yes" outcome name; directional markets are skipped.
        yes_pair = _parse_yes_outcome(mkt)
        if yes_pair is None:
            continue
        outcome_yes_name, price = yes_pair

        if price < config.BOND_PRICE_MIN or price > config.BOND_PRICE_MAX:
            continue

        try:
            volume = float(mkt.get("volume", 0) or mkt.get("liquidity", 0) or 0)
        except (ValueError, TypeError):
            volume = 0.0
        if volume < config.BOND_MIN_VOLUME:
            continue

        # Must resolve within BOND_MAX_HOURS hours
        days = _days_to_resolve(mkt)
        max_days = config.BOND_MAX_HOURS / 24.0
        if days is not None and (days < 0 or days > max_days):
            continue

        # Spread must be tight (near-certain = tight spread)
        spread_pct = _compute_spread_pct(mkt, price)
        if spread_pct > 1.5:  # tighter than general scanner
            continue

        # --- Dedup check: skip markets we already hold an open bond position on ---
        outcome_name_for_dedup = _get_outcome_name(mkt)
        if db.is_already_simulated(title, outcome_name_for_dedup):
            log.debug(
                "Bond dedup skip '%s' %s — already have open position",
                title[:40], outcome_name_for_dedup,
            )
            skipped_dedup += 1
            continue

        # --- Resolution rate tracking: log price bucket ---
        bucket = _get_price_bucket(price)
        price_bucket_counts[bucket] = price_bucket_counts.get(bucket, 0) + 1
        log.debug(
            "Bond candidate '%s' @ $%.3f — bucket: %s",
            title[:40], price, bucket,
        )

        # --- Diversification check: skip if this market already has too much capital ---
        existing_cost = open_bond_positions.get(title, 0.0)
        proposed_cost = config.MAX_COST_PER_TRADE  # conservative estimate (actual size set by Kelly)
        if existing_cost + proposed_cost > max_per_market:
            log.info(
                "Bond diversification LIMIT: '%s' — existing=$%.2f + proposed=$%.2f > max=$%.2f (%.0f%% cap). Skipping.",
                title[:40], existing_cost, proposed_cost, max_per_market,
                getattr(config, "BOND_MAX_POSITION_PCT", 0.20) * 100,
            )
            skipped_diversification += 1
            continue

        # WAVE 6 FIX 2026-04-26 (Round-7 #5 follow-on): use the validated
        # Yes-side outcome name (from _parse_yes_outcome above), not
        # _get_outcome_name which blindly returns outcomes[0]. Otherwise
        # bond candidate would record `outcome="Down"` even though `price`
        # is the Yes-side price → state inconsistency.
        end_date = mkt.get("endDate", "") or mkt.get("endDateIso", "")

        # WAVE 12 SHIP-2 (2026-05-16): resolve CLOB token_id for fill_simulator
        # wire. `mkt` is the gamma market dict — clobTokenIds + outcomes both
        # available. None → log_simulated_trade falls back to mid-price.
        _bond_token = polymarket_api.get_token_id_for_outcome(mkt, outcome_yes_name)
        candidates.append({
            "market_title":   title,
            "outcome":        outcome_yes_name,
            "side":           "Buy",
            "price":          price,
            "trader_score":   0.95,           # very high — near-certain outcome
            "market_volume":  min(volume, 100_000.0),
            "is_insider":     False,
            "source":         "bond",
            "trader_address": "scanner",
            "shares":         0,
            "cost":           config.MAX_COST_PER_TRADE,
            "api_signal":     _score_imbalance(mkt, price),
            "end_date":       end_date,
            "price_bucket":   bucket,
            "token_id":       _bond_token,    # WAVE 12 SHIP-2
            "expected_price": price,
        })

    # --- Price bucket summary ---
    if price_bucket_counts:
        bucket_summary = ", ".join(
            "%s=%d" % (b, n) for b, n in sorted(price_bucket_counts.items())
        )
        log.info("Bond price buckets: %s", bucket_summary)

    log.info(
        "Bond scanner: %d candidates from %d markets "
        "(skipped: finance=%d, dedup=%d, diversification=%d)",
        len(candidates), len(markets),
        skipped_finance, skipped_dedup, skipped_diversification,
    )
    if not candidates:
        return []

    approved = capital_allocator.allocate(candidates)
    log.info("Bond scanner: %d -> %d approved", len(candidates), len(approved))

    for c in approved:
        db.log_simulated_trade(
            trader_address="scanner",
            market_title=c["market_title"],
            outcome=c["outcome"],
            side=c["side"],
            price=c["price"],
            shares=c["shares_simulated"],
            cost=c["cost_simulated"],
            confidence=c["confidence"],
            source="bond",
            kelly_fraction=c.get("kelly_fraction"),
            ev=c.get("ev"),
            end_date=c.get("end_date"),
            swarm_votes=c.get("swarm_votes_json"),
            # WAVE 12 SHIP-2 (2026-05-16): pass token_id for orderbook-walk slippage
            token_id=c.get("token_id"),
            expected_price=c.get("expected_price"),
        )
        # Resolution rate tracking: log price bucket at placement time
        _log_bond_price_bucket(c["price"], c["market_title"])
        log.info(
            "[BOND] %s '%s' @ $%.3f | bucket=%s conf=%.2f cost=$%.2f",
            c["outcome"], c["market_title"][:40], c["price"],
            c.get("price_bucket", "?"), c["confidence"], c["cost_simulated"],
        )

    return approved


def scan_dutch_book() -> list[dict]:
    """Scan for Dutch book arbitrage: YES+NO bestBid sum < (1 - ARB_MIN_PROFIT_PCT).

    When the sum of both sides' best bids is below $1.00 (minus slippage buffer),
    buying both sides guarantees profit regardless of outcome.

    This is risk-free arbitrage — bypasses confidence gate and EV filter.
    $40M extracted from Polymarket 2024-2025 (arXiv:2508.03474).
    """
    import risk_guard
    safe, reason = risk_guard.check_safe_to_trade()
    if not safe:
        log.warning("Dutch book arb blocked by risk guard: %s", reason)
        return []

    arb_min = getattr(config, "ARB_MIN_PROFIT_PCT", 0.02)
    log.info("=== Dutch book arb scanner ===")
    markets = _fetch_active_markets(limit=200)
    if not markets:
        return []

    arb_candidates = []
    for mkt in markets:
        title = mkt.get("question", "") or mkt.get("title", "")
        if not title:
            continue

        # Need both YES and NO bids
        try:
            prices_raw = mkt.get("outcomePrices", "[]")
            prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
            if len(prices) < 2:
                continue
            yes_price = float(prices[0])
            no_price = float(prices[1])
        except (ValueError, TypeError, json.JSONDecodeError, IndexError):
            continue

        # Dutch book condition: YES + NO < 1 - arb_min_profit
        total = yes_price + no_price
        if total >= (1.0 - arb_min):
            continue

        guaranteed_profit_pct = round(1.0 - total, 4)

        try:
            volume = float(mkt.get("volume", 0) or mkt.get("liquidity", 0) or 0)
        except (ValueError, TypeError):
            volume = 0.0
        if volume < 1000:  # low bar — arb is risk-free
            continue

        end_date = mkt.get("endDate", "") or mkt.get("endDateIso", "")
        avg_price = total / 2.0

        arb_candidates.append({
            "market_title":           title,
            "outcome":                "YES+NO (arb)",
            "side":                   "Buy",
            "price":                  avg_price,
            "trader_score":           1.0,         # guaranteed profit
            "market_volume":          min(volume, 100_000.0),
            "is_insider":             False,
            "source":                 "arb",
            "trader_address":         "scanner",
            "is_arb":                 True,
            "guaranteed_profit_pct":  guaranteed_profit_pct,
            "shares":                 0,
            "cost":                   config.MAX_COST_PER_TRADE,
            "api_signal":             0.9,
            "end_date":               end_date,
            "yes_price":              yes_price,
            "no_price":               no_price,
        })

        log.info(
            "[ARB] '%s' — YES=%.3f + NO=%.3f = %.3f → profit=%.1f%%",
            title[:40], yes_price, no_price, total, guaranteed_profit_pct * 100
        )

    log.info("Dutch book: %d arb opportunities found", len(arb_candidates))
    if not arb_candidates:
        return []

    approved = capital_allocator.allocate(arb_candidates)
    for c in approved:
        db.log_simulated_trade(
            trader_address="scanner",
            market_title=c["market_title"],
            outcome=c["outcome"],
            side=c["side"],
            price=c["price"],
            shares=c["shares_simulated"],
            cost=c["cost_simulated"],
            confidence=c["confidence"],
            source="arb",
            kelly_fraction=c.get("kelly_fraction"),
            ev=c.get("ev"),
            end_date=c.get("end_date"),
            swarm_votes=c.get("swarm_votes_json"),
        )
        log.info(
            "[ARB ✓] '%s' @ guaranteed=%.1f%% cost=$%.2f",
            c["market_title"][:40], c.get("guaranteed_profit_pct", 0) * 100, c["cost_simulated"]
        )

    return approved


def _scan_no_side(markets: list[dict]) -> list[dict]:
    """Scan for overpriced YES markets where the NO side has value.

    YES at $0.55-$0.90 means NO is at $0.10-$0.45 — same asymmetric upside
    as the YES scanner, but on the other side. 75% of Polymarket markets
    resolve NO (statistical prior), so these are systematically underpriced.
    """
    no_price_min = config.SCANNER_PRICE_MIN   # NO price floor (from config)
    no_price_max = config.SCANNER_PRICE_MAX   # NO price ceiling (from config)

    candidates = []
    for mkt in markets:
        title = mkt.get("question", "") or mkt.get("title", "")
        if not title:
            continue

        # Category filter for NO side too
        if config.SCANNER_FINANCE_ONLY and not _is_finance_market(title):
            continue

        yes_price = _parse_outcome_price(mkt)
        if yes_price is None:
            continue

        no_price = 1.0 - yes_price
        if no_price < no_price_min or no_price > no_price_max:
            continue

        try:
            volume = float(mkt.get("volume", 0) or mkt.get("liquidity", 0) or 0)
        except (ValueError, TypeError):
            volume = 0.0
        if volume < config.SCANNER_MIN_VOLUME:
            continue

        days = _days_to_resolve(mkt)
        if days is not None and (days < 0 or days > config.MAX_RESOLVE_DAYS):
            continue

        spread_pct = _compute_spread_pct(mkt, yes_price)
        if spread_pct > config.MAX_SPREAD_PCT:
            continue

        end_date = mkt.get("endDate", "") or mkt.get("endDateIso", "")

        # --- Confidence computation ---
        use_claude = getattr(config, "SCANNER_USE_CLAUDE_PROBABILITY", False)
        conf_no = None
        claude_reasoning = ""

        if use_claude:
            claude_result = _estimate_probability_with_claude({
                "market_title": title,
                "question": title,
                "description": mkt.get("description", ""),
                "price": no_price,
                "market_volume": min(volume, 100_000.0),
                "end_date": end_date,
                "outcome": "No",
            })
            claude_prob = claude_result.get("probability")
            claude_conf = claude_result.get("confidence")

            if claude_prob is not None and claude_conf is not None:
                conf_no = _compute_claude_confidence(
                    claude_prob, claude_conf, no_price, "No",
                )
                claude_reasoning = claude_result.get("reasoning", "")
                if conf_no == 0.0:
                    log.debug(
                        "Claude sees no NO edge: '%s' — claude_prob=%.2f vs market=%.2f",
                        title[:40], claude_prob, no_price,
                    )
                    continue  # skip — no tradeable edge per Claude

        if conf_no is None:
            # Fallback: price-derived NO confidence (original logic)
            # NO side confidence: 75% markets resolve NO -> encode that prior
            conf_no = round(min(no_price + 0.15, 0.75), 3)
            if use_claude:
                log.warning("Using price-derived NO confidence for '%s' (Claude unavailable)", title[:40])

        candidate_dict = {
            "market_title":   title,
            "outcome":        "No",
            "side":           "Buy",
            "price":          no_price,
            "confidence":     conf_no,
            "market_volume":  min(volume, 100_000.0),
            "is_insider":     False,
            "source":         "scanner",
            "trader_address": "scanner",
            "shares":         0,
            "cost":           config.MAX_COST_PER_TRADE,
            "api_signal":     0.5,           # neutral — no specific bid imbalance for NO
            "end_date":       end_date,
        }
        if claude_reasoning:
            candidate_dict["claude_reasoning"] = claude_reasoning
        candidates.append(candidate_dict)

    return candidates


def scan_underpriced_markets() -> list[dict]:
    """Scan Gamma API for underpriced markets and run through allocator.

    Returns list of approved candidates (same format as capital_allocator output).
    """
    if not config.SCANNER_ENABLED:
        return []

    import risk_guard
    safe, reason = risk_guard.check_safe_to_trade()
    if not safe:
        log.warning("Scanner blocked by risk guard: %s", reason)
        return []

    log.info("=== Market scanner: scanning for underpriced opportunities ===")

    # Reset Claude call counter for this scan cycle
    _reset_claude_cycle_counter()

    # Use events endpoint (800+ markets per page) instead of /markets (200 max).
    # /markets q= search is broken. Events gives nested markets from all categories;
    # the _is_finance_market() filter then keeps only crypto/finance.
    markets = _fetch_all_markets_via_events(max_events=100)
    if not markets:
        log.info("Scanner: no markets returned from Gamma API")
        return []

    candidates = []
    skipped_price = skipped_volume = skipped_date = skipped_spread = 0

    for mkt in markets:
        title = mkt.get("question", "") or mkt.get("title", "")
        if not title:
            continue

        # Parse yes outcome price
        price = _parse_outcome_price(mkt)
        if price is None or price <= 0:
            continue

        # --- Category filter: crypto/finance only ---
        if config.SCANNER_FINANCE_ONLY and not _is_finance_market(title):
            continue

        # --- Price range filter: target underpriced asymmetric upside ---
        if price < config.SCANNER_PRICE_MIN or price > config.SCANNER_PRICE_MAX:
            skipped_price += 1
            continue

        # --- Volume filter: ensure exit liquidity ---
        try:
            volume = float(mkt.get("volume", 0) or mkt.get("liquidity", 0) or 0)
        except (ValueError, TypeError):
            volume = 0.0
        if volume < config.SCANNER_MIN_VOLUME:
            skipped_volume += 1
            continue

        # --- Resolve deadline: only short-term markets ---
        days = _days_to_resolve(mkt)
        if days is not None and (days < 0 or days > config.MAX_RESOLVE_DAYS):
            skipped_date += 1
            continue

        # --- Spread filter ---
        spread_pct = _compute_spread_pct(mkt, price)
        if spread_pct > config.MAX_SPREAD_PCT:
            skipped_spread += 1
            continue

        outcome_name = _get_outcome_name(mkt)
        end_date = mkt.get("endDate", "") or mkt.get("endDateIso", "")

        # Use volume as market_volume for confidence scoring
        market_volume = min(volume, 100_000.0)  # cap for normalization

        # --- Confidence computation ---
        use_claude = getattr(config, "SCANNER_USE_CLAUDE_PROBABILITY", False)
        conf_yes = None
        claude_reasoning = ""

        if use_claude:
            # Claude-based probability estimation: independent signal, not price-derived
            claude_result = _estimate_probability_with_claude({
                "market_title": title,
                "question": title,
                "description": mkt.get("description", ""),
                "price": price,
                "market_volume": market_volume,
                "end_date": end_date,
                "outcome": outcome_name,
            })
            claude_prob = claude_result.get("probability")
            claude_conf = claude_result.get("confidence")

            if claude_prob is not None and claude_conf is not None:
                conf_yes = _compute_claude_confidence(
                    claude_prob, claude_conf, price, outcome_name,
                )
                claude_reasoning = claude_result.get("reasoning", "")
                if conf_yes == 0.0:
                    log.debug(
                        "Claude sees no edge: '%s' %s — claude_prob=%.2f vs market=%.2f",
                        title[:40], outcome_name, claude_prob, price,
                    )
                    continue  # skip — no tradeable edge per Claude
            else:
                # Claude failed — fall back to price-derived confidence
                log.debug("Claude fallback for '%s': %s", title[:40], claude_result.get("reasoning", ""))
                conf_yes = None  # will be computed below

        if conf_yes is None:
            # Fallback: price-derived confidence (original logic)
            # ULTRA-REVIEW BOND+SCANNER AUDIT 2026-04-25: ceiling discount lifted
            # 0.70 → 0.85 (smaller penalty). Old combo (raw cap 0.70 × discount 0.70)
            # = effective 0.49 ceiling, colliding with CONF_BASE 0.25 floor and
            # killing 80%+ of low-price candidates. EV_MIN_THRESHOLD remains the
            # real filter once it flips back to 0.02 in live mode.
            raw_conf = min(price + max(0.08, 0.20 - price * 0.20), 0.85)
            imbalance = _score_imbalance(mkt, price)
            imbalance_boost = max(0.0, (imbalance - 0.5) * 0.10)  # up to +5%
            conf_yes = round(min(raw_conf * 0.85 + imbalance_boost, 0.80), 3)
            if use_claude:
                log.warning("Using price-derived confidence for '%s' (Claude unavailable)", title[:40])

        candidate_dict = {
            "market_title":   title,
            "outcome":        outcome_name,
            "side":           "Buy",
            "price":          price,
            "confidence":     conf_yes,
            "market_volume":  market_volume,
            "is_insider":     False,
            "source":         "scanner",
            "trader_address": "scanner",
            "shares":         0,             # Kelly will compute this
            "cost":           config.MAX_COST_PER_TRADE,
            "api_signal":     _score_imbalance(mkt, price),
            "end_date":       end_date,
        }
        if claude_reasoning:
            candidate_dict["claude_reasoning"] = claude_reasoning
        candidates.append(candidate_dict)

    # Also scan for value on the NO side (YES overpriced = NO underpriced)
    no_side = _scan_no_side(markets)
    candidates.extend(no_side)

    log.info(
        "Scanner: %d raw markets → %d YES + %d NO candidates "
        "(skipped: price=%d vol=%d date=%d spread=%d)",
        len(markets), len(candidates) - len(no_side), len(no_side),
        skipped_price, skipped_volume, skipped_date, skipped_spread,
    )

    if not candidates:
        return []

    approved = capital_allocator.allocate(candidates)
    log.info("Scanner: %d candidates → %d approved by allocator", len(candidates), len(approved))

    for c in approved:
        db.log_simulated_trade(
            trader_address="scanner",
            market_title=c["market_title"],
            outcome=c["outcome"],
            side=c["side"],
            price=c["price"],
            shares=c["shares_simulated"],
            cost=c["cost_simulated"],
            confidence=c["confidence"],
            source="scanner",
            kelly_fraction=c.get("kelly_fraction"),
            ev=c.get("ev"),
            end_date=c.get("end_date"),
            swarm_votes=c.get("swarm_votes_json"),
        )
        log.info(
            "[SCAN ✓] %s '%s' @ $%.3f | EV=%.3f conf=%.2f kelly=%.3f cost=$%.2f",
            c["outcome"],
            c["market_title"][:40],
            c["price"],
            c.get("ev", 0),
            c["confidence"],
            c.get("kelly_fraction", 0),
            c["cost_simulated"],
        )

    return approved


def _score_imbalance(market: dict, price: float) -> float:
    """Score bid imbalance from Gamma market fields as api_signal (0-1).

    Low price + high bid imbalance = smart money accumulating Yes → strong signal.
    Returns 0.5 (neutral) if data unavailable.
    """
    try:
        best_bid = float(market.get("bestBid", 0) or 0)
        best_ask = float(market.get("bestAsk", 0) or 0)
        if best_bid <= 0 or best_ask <= 0:
            return 0.5

        # Imbalance: how much closer is the best_bid to the price vs best_ask?
        # High bid relative to ask = buyers more aggressive = imbalance signal
        mid = (best_bid + best_ask) / 2.0
        if mid <= 0:
            return 0.5

        # Normalize: bid imbalance 0-1 where 0.5 = perfectly balanced
        imbalance = best_bid / (best_bid + best_ask)  # 0-1, > 0.5 = bid-heavy

        # Scale: 0.5 imbalance → 0.5 signal, 0.7 imbalance → 0.9 signal
        signal = 0.5 + (imbalance - 0.5) * 2.0
        return max(0.0, min(1.0, round(signal, 4)))
    except (ValueError, TypeError):
        return 0.5
