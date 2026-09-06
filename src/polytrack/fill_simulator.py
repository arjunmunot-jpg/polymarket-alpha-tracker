"""Realistic fill simulator using live CLOB orderbook depth.

MARATHON SHIP-11 (2026-04-26 audit Dry-run Phase A): closes the lowest-
scoring dimension of the dry-run audit (3/10). Was: `db.log_simulated_trade`
recorded fills at the observed mid price → optimistic. Now: walk the actual
bid/ask side of the orderbook for our notional, compute size-weighted fill
price + slippage_bps + partial-fill flag.

Modeled on QuantConnect's FillModel pattern + Alpaca's documented gap
("paper trading has infinite liquidity, no slippage"). Phase-B Probe c25
(slippage_tracking) consumes the new columns.

Usage:
    from fill_simulator import simulate_fill
    fill = simulate_fill(token_id="0xABC...", side="BUY", notional_usd=10.0)
    if fill.is_filled:
        record(price=fill.fill_price, slippage_bps=fill.slippage_bps, ...)
    else:
        # Insufficient liquidity at requested size — partial only
        record(price=fill.fill_price, partial_fill_pct=fill.fill_pct, ...)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("polybot.fill_simulator")


@dataclass
class FillResult:
    """Outcome of a fill simulation against orderbook depth."""
    is_filled: bool
    fill_price: float           # size-weighted average fill price
    expected_price: float       # mid (what immediate-fill model would record)
    notional_filled_usd: float  # how much of the request was filled
    notional_requested_usd: float
    slippage_bps: float         # (fill - expected) / expected, in bps, signed (negative for adverse)
    levels_consumed: int        # how many orderbook levels we walked
    fill_pct: float             # 0.0-1.0
    reason: str = ""            # diagnostic / failure reason
    book_top_bid: Optional[float] = None
    book_top_ask: Optional[float] = None


def simulate_fill(
    token_id: str,
    side: str,
    notional_usd: float,
    *,
    fetch_book=None,
    fallback_price: Optional[float] = None,
    max_levels: int = 20,
) -> FillResult:
    """Walk live orderbook for `token_id` to estimate realistic fill.

    Args:
        token_id: Polymarket CLOB token id (e.g. yesTokenId).
        side: "BUY" or "SELL".
        notional_usd: USD amount we want to spend (BUY) or receive (SELL).
        fetch_book: optional callable that returns
            {"bids": [[price, size], ...], "asks": [[price, size], ...]}.
            Defaults to polymarket_api.get_orderbook(token_id).
        fallback_price: used when book unavailable (returns immediate-fill
            model assuming no slippage; sets reason="book_unavailable").
        max_levels: max orderbook levels to walk before flagging partial.

    Returns:
        FillResult with size-weighted execution price, slippage_bps, etc.

    Failure-tolerant: any exception fetching the book → fall back to
    immediate-fill at fallback_price (or 0.5 if neither given). NEVER raises.
    """
    side_upper = (side or "").strip().upper()
    if side_upper not in ("BUY", "SELL"):
        return FillResult(
            is_filled=False, fill_price=0.0, expected_price=0.0,
            notional_filled_usd=0.0, notional_requested_usd=float(notional_usd),
            slippage_bps=0.0, levels_consumed=0, fill_pct=0.0,
            reason=f"invalid side: {side!r}",
        )
    notional = float(notional_usd or 0.0)
    if notional <= 0:
        return FillResult(
            is_filled=False, fill_price=0.0, expected_price=0.0,
            notional_filled_usd=0.0, notional_requested_usd=notional,
            slippage_bps=0.0, levels_consumed=0, fill_pct=0.0,
            reason="non-positive notional",
        )

    # Fetch the book (best-effort)
    book = None
    if fetch_book is None:
        try:
            from polymarket_api import get_orderbook
            book = get_orderbook(token_id)
        except Exception as e:
            log.debug("orderbook fetch failed: %s", e)
    else:
        try:
            book = fetch_book(token_id)
        except Exception as e:
            log.debug("custom fetch_book failed: %s", e)

    if not book:
        # Fall back to immediate-fill at fallback_price
        fp = float(fallback_price) if fallback_price is not None else 0.5
        return FillResult(
            is_filled=True, fill_price=fp, expected_price=fp,
            notional_filled_usd=notional, notional_requested_usd=notional,
            slippage_bps=0.0, levels_consumed=0, fill_pct=1.0,
            reason="book_unavailable_fallback",
        )

    bids = book.get("bids") or []
    asks = book.get("asks") or []

    # WAVE 13 SHIP-1 (2026-05-16): polymarket CLOB returns orderbook levels as
    # list-of-DICTS {"price":"0.54","size":"100"} with string-encoded numerics,
    # NOT list-of-LISTS [[0.54, 100]] as the original code assumed. That bug hid
    # behind a DEBUG-level catch for 214 production trades (SHIP-2 silent failure
    # in MISTAKES.md 2026-05-16). Helper handles BOTH formats defensively per
    # Rule 12 — never raise on a malformed level; skip and walk the next one.
    def _level_px(lvl):
        try:
            if isinstance(lvl, dict):
                p = lvl.get("price")
                return float(p) if p is not None else None
            return float(lvl[0])
        except (ValueError, TypeError, IndexError, KeyError):
            return None

    def _level_sz(lvl):
        try:
            if isinstance(lvl, dict):
                s = lvl.get("size")
                return float(s) if s is not None else None
            return float(lvl[1])
        except (ValueError, TypeError, IndexError, KeyError):
            return None

    # ORDERING FIX 2026-07-13 (proven by running against the live CLOB): /book returns
    # bids ASCENDING and asks DESCENDING — best-of-book is the LAST element on both
    # sides (polymarket_api._parse_orderbook_stats documented this; this module assumed
    # best-first). bids[0]/asks[0] read 0.01/0.99 placeholder-looking tops and the walk
    # crossed from the WORST ask → every simulated fill ≈ 0.99 (the uniform "+49c cost"
    # artifact; historical slippage_bps values from this path are suspect). Sort
    # DEFENSIVELY — idempotent whatever order the server uses; unparseable levels sink
    # to the tail and are skipped by the walk.
    bids = sorted(bids, key=lambda l: _level_px(l) if _level_px(l) is not None else -1.0,
                  reverse=True)   # best (highest) bid first
    asks = sorted(asks, key=lambda l: _level_px(l) if _level_px(l) is not None else 2.0)
    # best (lowest) ask first

    # Top of book → expected/mid
    top_bid = _level_px(bids[0]) if bids else None
    top_ask = _level_px(asks[0]) if asks else None
    if top_bid is not None and top_ask is not None and top_bid > 0 and top_ask > 0:
        mid = (top_bid + top_ask) / 2.0
    elif top_bid is not None:
        mid = top_bid
    elif top_ask is not None:
        mid = top_ask
    else:
        # Empty book on both sides — fall back
        fp = float(fallback_price) if fallback_price is not None else 0.5
        return FillResult(
            is_filled=True, fill_price=fp, expected_price=fp,
            notional_filled_usd=notional, notional_requested_usd=notional,
            slippage_bps=0.0, levels_consumed=0, fill_pct=1.0,
            reason="empty_book_fallback",
        )

    # Walk the relevant side
    if side_upper == "BUY":
        levels = asks[:max_levels]  # cross the ask
    else:
        levels = bids[:max_levels]  # hit the bid

    if not levels:
        # No liquidity on the side we want
        return FillResult(
            is_filled=False, fill_price=mid, expected_price=mid,
            notional_filled_usd=0.0, notional_requested_usd=notional,
            slippage_bps=0.0, levels_consumed=0, fill_pct=0.0,
            reason=f"no_liquidity_{side_upper.lower()}",
            book_top_bid=top_bid, book_top_ask=top_ask,
        )

    # Walk levels, accumulating notional + cost
    remaining_usd = notional
    total_cost = 0.0
    total_size = 0.0
    levels_consumed = 0
    for level in levels:
        # WAVE 13 SHIP-1: use the dict/list-tolerant extractor helpers defined
        # above. Was `for px, sz in levels` which raised "too many values to
        # unpack" the moment polymarket returned dicts instead of 2-tuples.
        px_f = _level_px(level)
        sz_f = _level_sz(level)
        if px_f is None or sz_f is None:
            continue
        if px_f <= 0 or sz_f <= 0:
            continue
        # Each level fills at price px_f, max size = sz_f
        # Notional consumed at this level = px_f * shares_taken (BUY) or px_f * shares_sold (SELL)
        # Shares we can take here = min(sz_f, remaining_usd / px_f)
        shares_at_level = min(sz_f, remaining_usd / px_f)
        if shares_at_level <= 0:
            break
        cost_at_level = shares_at_level * px_f
        total_cost += cost_at_level
        total_size += shares_at_level
        levels_consumed += 1
        remaining_usd -= cost_at_level
        if remaining_usd <= 1e-9:
            break

    if total_size <= 0:
        return FillResult(
            is_filled=False, fill_price=mid, expected_price=mid,
            notional_filled_usd=0.0, notional_requested_usd=notional,
            slippage_bps=0.0, levels_consumed=0, fill_pct=0.0,
            reason="zero_walked",
            book_top_bid=top_bid, book_top_ask=top_ask,
        )

    fill_price = total_cost / total_size
    fill_pct = total_cost / notional if notional > 0 else 0.0
    is_filled = remaining_usd < (notional * 0.005)  # ≥99.5% filled
    # Slippage in bps. For BUY adverse = fill > mid (positive bps).
    # For SELL adverse = fill < mid (negative bps in the client's frame, but we
    # report unsigned magnitude with sign convention "positive = bad fill").
    if mid > 0:
        if side_upper == "BUY":
            slippage_bps = (fill_price - mid) / mid * 10_000.0
        else:
            slippage_bps = (mid - fill_price) / mid * 10_000.0
    else:
        slippage_bps = 0.0

    return FillResult(
        is_filled=is_filled,
        fill_price=fill_price,
        expected_price=mid,
        notional_filled_usd=total_cost,
        notional_requested_usd=notional,
        slippage_bps=slippage_bps,
        levels_consumed=levels_consumed,
        fill_pct=fill_pct,
        reason="filled" if is_filled else f"partial_{fill_pct:.1%}",
        book_top_bid=top_bid, book_top_ask=top_ask,
    )


# ── Convenience wrapper for log_simulated_trade integration ────────────

def estimate_slippage_bps(token_id: str, side: str, notional_usd: float,
                          fallback_price: Optional[float] = None) -> float:
    """One-line helper: just return slippage_bps for a planned trade.

    For callers that aren't ready to record full fill metadata yet but want
    to add a slippage cost to their simulation. Returns 0.0 on any failure.
    """
    result = simulate_fill(
        token_id=token_id, side=side, notional_usd=notional_usd,
        fallback_price=fallback_price,
    )
    return float(result.slippage_bps)
