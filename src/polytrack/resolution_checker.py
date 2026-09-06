"""Trade Resolution Checker v3 — polls Polymarket for resolved markets.

Runs every monitor cycle (5 min). For each unresolved simulated trade:
  1. BTC 5min trades: uses deterministic slug to look up event directly
  2. Other trades: searches Gamma API /markets by title
  3. Checks if the market has resolved (outcomePrices -> 1.0/0.0)
  4. Marks the trade Won/Lost via db.resolve_trade() with P&L

v2 fixes:
  - BTC 5min markets now use slug-based lookup (title search never found them)
  - Failures logged at WARNING level (was DEBUG — invisible in bot.log)
  - Added source-aware resolution routing
  - Added stale trade detection (warns if trades are open > 2x expected duration)

v3 fixes (2026-04-18):
  - Stale endDate handling: when market can't be found by title AND end_date is
    past, re-query Polymarket by slug to check closed status (Harvey-Weinstein
    pattern: endDate stale placeholder but market still closed=false and active).
  - If market still open: NULL end_date + set end_date_stale=1 in DB, skip until
    market actually closes. Prevents 285-warning-per-cycle zombie chasing.
  - If market truly gone (404/closed=true/no outcome): keep fuzzy-match fallback.
  - Log spam reduction: one log per market per cycle, tagged [stale_endDate].
  - Cycle-level stats: "resolved X, stale_endDate Y skipped, truly_lost Z".
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import requests

import config
import db

log = logging.getLogger("polybot.resolution_checker")


# ── BTC 5min slug-based lookup ───────────────────────────────────────────────

def _btc_slot_from_title(title: str) -> Optional[int]:
    """Parse a BTC 15-min market title to extract the slot end time.

    Title format: 'bitcoin up or down - april 14, 6:45pm-7:00pm et'
    Returns the slot start timestamp (Unix), or None if parsing fails.
    """
    try:
        # Extract date and time range from title
        # Pattern: "month day, start_time-end_time et"
        m = re.search(
            r'(\w+)\s+(\d+),?\s+(\d+:\d+(?:am|pm))\s*-\s*(\d+:\d+(?:am|pm))\s*et',
            title.lower()
        )
        if not m:
            return None

        month_str, day_str, start_str, end_str = m.groups()

        # Parse month
        months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        month = months.get(month_str)
        if not month:
            return None

        day = int(day_str)
        # WAVE 8 FIX 2026-04-26 (Round-9 council #4): use UTC year to avoid
        # off-by-one on Dec 31 in eastward timezones. datetime.now() returns
        # local time; in IST/UTC+5:30 the .year flips ~5h before US/UTC.
        # BTC slot resolution must use UTC year for correct Polymarket slug.
        year = datetime.now(timezone.utc).year

        # Parse start time — attach America/New_York timezone so
        # EDT (UTC-4) vs EST (UTC-5) is handled automatically.
        eastern = ZoneInfo("America/New_York")
        start_time = datetime.strptime(start_str, "%I:%M%p")
        start_dt = datetime(year, month, day, start_time.hour, start_time.minute,
                            tzinfo=eastern)
        # Convert to UTC
        start_utc = start_dt.astimezone(timezone.utc)
        slot_ts = int(start_utc.timestamp())
        # Round to nearest 900s slot
        slot_ts = (slot_ts // 900) * 900
        return slot_ts

    except Exception as e:
        log.warning("Failed to parse BTC slot from title '%s': %s", title[:50], e)
        return None


def _fetch_btc_market(title: str) -> Optional[dict]:
    """Look up a BTC 5min market using the deterministic slug pattern.

    BTC markets use slugs like 'btc-updown-15m-{slot_timestamp}'.
    This is the same approach btc_5min.py uses to find active markets.
    """
    slot = _btc_slot_from_title(title)
    if slot is None:
        log.warning("Could not parse BTC slot from: '%s'", title[:60])
        return None

    slug = f"btc-updown-15m-{slot}"

    # Try multiple slot offsets in case of timing edge cases
    for offset in [0, -900, 900]:
        try_slug = f"btc-updown-15m-{slot + offset}"
        try:
            r = requests.get(
                f"{config.POLYMARKET_GAMMA_URL}/events",
                params={"slug": try_slug},
                timeout=8,
            )
            r.raise_for_status()
            events = r.json()
            if not events:
                continue

            event = events[0]
            markets = event.get("markets", [])
            if not markets:
                continue

            market = markets[0]
            # Verify title similarity to avoid wrong slot
            market_q = (market.get("question") or "").lower()
            if "bitcoin" in market_q or "btc" in market_q:
                log.debug("BTC resolution: found market via slug '%s'", try_slug)
                return market

        except Exception as e:
            log.warning("BTC slug lookup failed for '%s': %s", try_slug, e)
            continue

    log.warning("BTC resolution: no market found for '%s' (tried slug '%s' +/- 900)",
                title[:50], slug)
    return None


# ── Generic title-based lookup ───────────────────────────────────────────────

def _fetch_market(title: str) -> Optional[dict]:
    """Search Gamma API for a market matching the given question title.

    Uses exact matching first, then substring matching as fallback.
    """
    # Strip strategy prefixes like [cv_arb], [ss_arb], [me_arb] before API lookup.
    search_title = re.sub(r"^\[[^\]]+\]\s*", "", title)
    try:
        r = requests.get(
            f"{config.POLYMARKET_GAMMA_URL}/markets",
            params={"q": search_title[:80], "limit": 10},
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        markets = data if isinstance(data, list) else data.get("data", [])
        title_lower = search_title.lower().strip()

        # Pass 1: exact match (case-insensitive, trimmed)
        for m in markets:
            q = (m.get("question") or m.get("title") or "").lower().strip()
            if q == title_lower:
                return m

        # Pass 2: substring match with overlap threshold
        for m in markets:
            q = (m.get("question") or m.get("title") or "").lower().strip()
            min_len = min(len(title_lower), len(q))
            overlap_required = max(60, int(min_len * 0.80))
            if (title_lower[:overlap_required] in q or
                    q[:overlap_required] in title_lower):
                return m

        # If we got results but none matched, try slug-based fallback
        if markets:
            log.debug(
                "Resolution: text search found %d markets for '%s' but none matched. "
                "Trying slug fallback.",
                len(markets), title[:40],
            )
    except Exception as e:
        log.warning("Resolution fetch failed for '%s': %s", title[:40], e)

    # Slug-based fallback — more reliable for weather/bracket markets where
    # the Gamma text search returns trending/irrelevant markets.
    slug = _derive_slug(title)
    for _ep in ("events", "markets"):
        try:
            r2 = requests.get(
                f"{config.POLYMARKET_GAMMA_URL}/{_ep}",
                params={"slug": slug},
                timeout=8,
            )
            if not r2.ok:
                continue
            data2 = r2.json()
            items = data2 if isinstance(data2, list) else data2.get("data", [])
            candidates = []
            for item in items:
                if "markets" in item:
                    candidates.extend(item["markets"])
                else:
                    candidates.append(item)
            if candidates:
                return candidates[0]
        except Exception:
            pass
    return None


# ── Stale-endDate re-query by slug ───────────────────────────────────────────

def _derive_slug(title: str) -> str:
    """Convert a market title to a best-guess Polymarket slug.

    Polymarket slugs are lowercase, hyphenated titles with punctuation stripped.
    This is a heuristic — close enough to find the market even if not exact.
    """
    # Strip strategy prefixes like [cv_arb], [ss_arb], [me_arb] inserted by corr_arb scanner.
    slug = re.sub(r"^\[[^\]]+\]\s*", "", title).lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug.strip())
    slug = re.sub(r"-+", "-", slug)
    return slug[:100]


def _check_market_still_open(title: str) -> Optional[bool]:
    """Re-query Polymarket Gamma API to test whether a market with a past
    end_date is still active (closed=false).

    Returns:
        True  — market found AND closed=false (still trading; endDate is stale)
        False — market found AND (closed=true OR resolved=true with no winner)
        None  — could not determine (404, empty response, network error)
    """
    slug = _derive_slug(title)
    # Try /events?slug= first (slug-based, most reliable)
    for endpoint, params in [
        (f"{config.POLYMARKET_GAMMA_URL}/events", {"slug": slug}),
        (f"{config.POLYMARKET_GAMMA_URL}/markets", {"slug": slug}),
    ]:
        try:
            r = requests.get(endpoint, params=params, timeout=8)
            if r.status_code == 404:
                return False
            r.raise_for_status()
            data = r.json()
            items = data if isinstance(data, list) else data.get("data", [])
            if not items:
                continue
            # Events wrap markets; flatten
            candidates = []
            for item in items:
                if "markets" in item:
                    candidates.extend(item["markets"])
                else:
                    candidates.append(item)
            for mkt in candidates:
                if mkt.get("closed") is False and not mkt.get("resolved"):
                    return True   # still open
                if mkt.get("closed") or mkt.get("resolved"):
                    return False  # genuinely closed
        except Exception:
            continue
    return None  # inconclusive


# ── Winner determination ─────────────────────────────────────────────────────

def _determine_winner(market: dict) -> Optional[str]:
    """Return 'Yes', 'No', 'Up', 'Down', or None if not yet settled.

    IMPORTANT: Polymarket BTC markets set closed=True with definitive
    outcomePrices=['1','0'] but NEVER set resolved=True.
    So we check outcomePrices first, then fall back to resolved flag.
    """
    is_resolved = market.get("resolved")
    is_closed = market.get("closed")

    # Parse outcomePrices (works for both resolved and closed markets)
    prices_raw = market.get("outcomePrices", "[]")
    outcomes_raw = market.get("outcomes", "[]")
    try:
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
    except Exception:
        prices, outcomes = [], []

    # Check for definitive prices (1/0) — the most reliable signal
    if prices and (is_resolved or is_closed):
        try:
            float_prices = [float(p) for p in prices]
            # WAVE 8 FIX 2026-04-26 (Round-10 council #3): UMA dispute /
            # invalid markets resolve as 50/50 split (`['0.5','0.5']`),
            # never hitting the >=0.99 path → trade stuck open forever →
            # silent budget exhaustion at ~3-5%/year invalid rate. Now:
            # detect invalid + return sentinel for refund handling.
            if len(float_prices) == 2 and all(0.45 <= p <= 0.55 for p in float_prices):
                return "INVALID"
            # Find the winning outcome (price >= 0.99)
            for i, p in enumerate(float_prices):
                if p >= 0.99:
                    # Use named outcome if available (Up/Down/Yes/No)
                    if outcomes and i < len(outcomes):
                        return outcomes[i]  # e.g. "Up", "Down", "Yes", "No"
                    return "Yes" if i == 0 else "No"
        except (ValueError, TypeError):
            pass

    # If neither resolved nor closed, market is still open
    if not is_resolved and not is_closed:
        return None

    # Fallback: check resolutionSource field
    src = (market.get("resolutionSource") or "").strip()
    if src.lower() in ("yes", "no", "up", "down"):
        return src.capitalize()

    return None


# ── P&L calculation ──────────────────────────────────────────────────────────

def _calculate_pnl(trade: dict, won: bool, title: str) -> float:
    """Calculate P&L for a resolved trade, accounting for fees."""
    price = trade["price"]
    shares = trade["shares_simulated"]
    try:
        from capital_allocator import get_fee_rate
        fee_rate = get_fee_rate(title, trade.get("source", ""))
    except Exception:
        fee_rate = 0.04  # fallback default taker fee rate
    fee_per_share = fee_rate * price * (1.0 - price)
    total_fee = fee_per_share * shares
    if won:
        pnl = round((1.0 - price) * shares - total_fee, 4)
    else:
        pnl = round(-price * shares - total_fee, 4)
    return pnl


# ── Main resolution loop ────────────────────────────────────────────────────

def check_resolutions():
    """Check all open simulated trades against Polymarket for resolution.

    Called from main.py during each monitor cycle (~5 min).
    Routes BTC trades to slug-based lookup, others to title search.

    v3 stale-endDate handling:
      When a market can't be found by title AND its stored end_date is in the
      past, we re-query Polymarket by derived slug to check closed status.
      - Still open (closed=false): NULL the end_date in DB + set end_date_stale=1.
        Logged once per market per cycle as [stale_endDate]. Not retried until
        a later cycle detects the market has actually closed.
      - Genuinely gone: logged as [truly_lost], kept in truly_lost_count.
    End-of-cycle summary: "resolved X, stale_endDate Y skipped, truly_lost Z".
    """
    trades = db.get_unresolved_trades()
    if not trades:
        return

    now = datetime.now(timezone.utc)

    log.info("Resolution check: %d unresolved trades to check", len(trades))

    # Group by market_title to minimise API calls
    by_market: dict[str, list[dict]] = {}
    for t in trades:
        by_market.setdefault(t["market_title"], []).append(t)

    resolved_count = 0
    stale_end_date_count = 0  # markets confirmed still-open despite past endDate
    truly_lost_count = 0      # markets with past endDate AND not found on Polymarket

    for title, market_trades in by_market.items():
        # Route to appropriate fetcher based on source
        source = market_trades[0].get("source", "")

        # Strip correlation-arb prefixes so Gamma API search finds the real title.
        # Updated 2026-04-17: added [cv_arb] (conditional valid) — was slipping through.
        # Case-insensitive match because some sources lowercase the prefix.
        lookup_title = title
        for _pfx in ("[COMP_ARB] ", "[ME_ARB] ", "[SS_ARB] ", "[TIME_ARB] ",
                     "[CV_ARB] ", "[cv_arb] ", "[comp_arb] ", "[me_arb] ",
                     "[ss_arb] ", "[time_arb] "):
            if lookup_title.startswith(_pfx):
                lookup_title = lookup_title[len(_pfx):]
                break

        if source in ("btc_5min", "btc_arb"):
            market = _fetch_btc_market(lookup_title)
        else:
            market = _fetch_market(lookup_title)

        if not market:
            # ── Stale-endDate check (v3) ──────────────────────────────────
            # If every trade in this market group has a past end_date (or is
            # already flagged stale), attempt a slug-based closed-status check
            # before logging a warning and giving up.
            representative = market_trades[0]
            stored_end_date = representative.get("end_date")
            already_stale = bool(representative.get("end_date_stale"))

            end_date_past = False
            # ULTRA-REVIEW RESOLUTION AUDIT 2026-04-25: was `if stored_end_date:`.
            # Empty string `''` is falsy → 118 corr_arb backfill-orphans with
            # end_date='' silently fall through to "Silent skip" path → never
            # auto-flag stale, never re-resolve unless Gamma exact-matches.
            # Now treat empty/whitespace string identically to NULL.
            stored_end_date_real = stored_end_date and str(stored_end_date).strip()
            if stored_end_date_real:
                try:
                    ed = datetime.fromisoformat(stored_end_date.replace("Z", "+00:00"))
                    if ed.tzinfo is None:
                        ed = ed.replace(tzinfo=timezone.utc)
                    end_date_past = ed < now
                except (ValueError, AttributeError):
                    pass

            if end_date_past or already_stale:
                still_open = _check_market_still_open(lookup_title)
                if still_open is True:
                    # Market is still active — endDate was stale placeholder.
                    # NULL end_date + set stale flag so we stop chasing this
                    # every cycle. Will re-resolve once market actually closes.
                    for t in market_trades:
                        if not t.get("end_date_stale"):
                            db.mark_end_date_stale(t["id"])
                    stale_end_date_count += 1
                    log.info(
                        "[stale_endDate] '%s' — endDate was past but market "
                        "still closed=false on Polymarket. end_date NULLed for "
                        "%d trade(s); will retry when market closes.",
                        title[:50], len(market_trades),
                    )
                    continue
                elif still_open is False:
                    # Market is genuinely closed/gone but we still can't find
                    # a resolution. Keep trying via other signals (future cycles).
                    truly_lost_count += 1
                    log.warning(
                        "[truly_lost] '%s' — market closed on Polymarket but "
                        "no resolution outcome found. Will retry next cycle.",
                        title[:50],
                    )
                    continue
                else:
                    # Inconclusive (network error, 404, empty).
                    # If end_date is > 24h past, stop retrying by flagging stale —
                    # these are dead markets the API will never return cleanly.
                    # 24h (was 48h, 2026-04-20): daily markets (weather) resolve same-day;
                    # 48h left 300+ weather trades spamming truly_lost for an extra day.
                    # ULTRA-REVIEW 2026-04-25: also stale-flag if end_date is
                    # empty/whitespace (the 118 corr_arb backfill-orphans).
                    stored_end_date_real = stored_end_date and str(stored_end_date).strip()
                    if not stored_end_date_real:
                        # No usable end_date — these are zombies. Mark stale to stop polling.
                        for t in market_trades:
                            if not t.get("end_date_stale"):
                                db.mark_end_date_stale(t["id"])
                        stale_end_date_count += 1
                        continue
                    if stored_end_date_real:
                        try:
                            ed = datetime.fromisoformat(stored_end_date.replace("Z", "+00:00"))
                            if ed.tzinfo is None:
                                ed = ed.replace(tzinfo=timezone.utc)
                            if (now - ed) > timedelta(hours=24):
                                for t in market_trades:
                                    if not t.get("end_date_stale"):
                                        db.mark_end_date_stale(t["id"])
                                stale_end_date_count += 1
                                log.info(
                                    "[stale_endDate] '%s' — API inconclusive + "
                                    "end_date >48h past. Flagged stale to stop retry.",
                                    title[:50],
                                )
                                continue
                        except (ValueError, AttributeError):
                            pass
                    truly_lost_count += 1
                    log.warning(
                        "[truly_lost] '%s' — could not find market on Polymarket "
                        "(inconclusive slug check). Will retry next cycle.",
                        title[:50],
                    )
                    continue
            else:
                # end_date is still in the future (or absent) — normal not-found,
                # market just hasn't closed yet. Silent skip (not an error).
                continue

        winner = _determine_winner(market)
        if winner is None:
            continue  # still open on Polymarket

        for t in market_trades:
            # Normalize comparison: Yes/No vs Up/Down
            trade_outcome = t["outcome"].lower()
            winner_lower = winner.lower()

            # WAVE 8 FIX 2026-04-26 (Round-10 council #3): INVALID resolution
            # handler. UMA-disputed markets refund holders pro-rata at $0.50.
            # Without this branch, invalid trades stuck open forever, silently
            # exhausting per-strategy budget. Approximate refund: shares × 0.5
            # − cost (i.e. position closed at midpoint, no win/loss).
            if winner_lower == "invalid":
                shares = float(t.get("shares_simulated") or 0)
                cost = float(t.get("cost_simulated") or 0)
                # Refund at $0.50/share minus original cost — typically ~0
                refund_pnl = round(shares * 0.5 - cost, 4)
                db.resolve_trade(t["id"], False, refund_pnl, resolved_outcome="invalid")
                log.info(
                    "RESOLVED #%d [%s] '%s' INVALID/disputed -> refund $%.4f",
                    t["id"], source, title[:35], refund_pnl,
                )
                resolved_count += 1
                continue

            # IB-009 fix (2026-04-25 marathon): corr_arb places trades with
            # outcome="no (all outcomes)" representing a basket of NO bets across
            # ALL N outcomes of a multi-outcome market. At resolution, exactly 1
            # outcome wins (worth 0 to NO holder) and N-1 lose (worth $1 each
            # to NO holder). This is deterministic arb — "won" doesn't apply
            # at the leg level. Compute portfolio pnl directly.
            if trade_outcome.startswith("no (all"):
                outcomes_raw = market.get("outcomes", "[]")
                try:
                    outcomes_list = (
                        json.loads(outcomes_raw)
                        if isinstance(outcomes_raw, str) else outcomes_raw
                    )
                except Exception:
                    outcomes_list = []
                n = len(outcomes_list)
                if n >= 2:
                    shares = float(t.get("shares_simulated") or 0)
                    price = float(t.get("price") or 0)
                    if shares > 0 and price > 0:
                        payout_per_share = (n - 1) / n
                        try:
                            from capital_allocator import get_fee_rate
                            fee_rate = get_fee_rate(title, source)
                        except Exception:
                            fee_rate = 0.04
                        fee_per_share = fee_rate * price * (1.0 - price)
                        pnl = round(
                            shares * (payout_per_share - price)
                            - fee_per_share * shares,
                            4,
                        )
                        db.resolve_trade(t["id"], True, pnl, resolved_outcome=f"basket_n{n}")
                        log.info(
                            "RESOLVED #%d [%s] '%s' arb_basket(n=%d) -> $%.4f",
                            t["id"], source, title[:35], n, pnl,
                        )
                        resolved_count += 1
                        continue
                # Fallback if outcomes can't be parsed: skip this trade,
                # don't force-resolve. Will retry next cycle.
                log.warning(
                    "[corr_arb] '%s' has 'no (all)' outcome but market lacks "
                    "parseable outcomes list; skipping resolution this cycle.",
                    title[:50],
                )
                continue

            # WAVE 6 FIX 2026-04-26 (Round-7 council finding #3): cv_arb /
            # complement-arb resolution handler. Mirrors IB-009 ME_ARB pattern.
            # corr_arb COMP_ARB path writes outcome="YES+NO (both sides)" — a
            # basket of YES + NO bets at avg_price. With Sum(YES,NO) < $1 the
            # arb is locked. At resolution, holder receives $1 from the winning
            # side; total cost was 2 * avg_price * shares; pnl = shares * (1 - 2*avg_price).
            # Without this handler, complement arbs stayed unresolved forever.
            if trade_outcome.startswith("yes+no") or trade_outcome.startswith("yes + no"):
                shares = float(t.get("shares_simulated") or 0)
                price = float(t.get("price") or 0)  # this is avg_price (cost per LEG)
                if shares > 0 and price > 0:
                    try:
                        from capital_allocator import get_fee_rate
                        fee_rate = get_fee_rate(title, source)
                    except Exception:
                        fee_rate = 0.04
                    # fee paid on the winning leg (the losing leg → $0 redemption)
                    fee_per_share = fee_rate * price * (1.0 - price)
                    # Pay 2*price per share-pair, redeem $1 from winner. Net per share-pair:
                    pnl = round(
                        shares * (1.0 - 2.0 * price) - fee_per_share * shares,
                        4,
                    )
                    db.resolve_trade(t["id"], True, pnl, resolved_outcome="complement_basket")
                    log.info(
                        "RESOLVED #%d [%s] '%s' complement_arb -> $%.4f",
                        t["id"], source, title[:35], pnl,
                    )
                    resolved_count += 1
                    continue
                # Fallback: malformed row, skip
                log.warning(
                    "[corr_arb] '%s' has 'yes+no' outcome but shares/price invalid; "
                    "skipping resolution this cycle.", title[:50],
                )
                continue

            # BTC markets: "up"/"down" outcomes
            won = trade_outcome == winner_lower
            # Also handle Yes/No mapping to Up/Down
            if not won and trade_outcome in ("yes", "up") and winner_lower in ("yes", "up"):
                won = True
            if not won and trade_outcome in ("no", "down") and winner_lower in ("no", "down"):
                won = True

            pnl = _calculate_pnl(t, won, title)

            db.resolve_trade(t["id"], won, pnl, resolved_outcome=str(winner))
            log.info(
                "RESOLVED #%d [%s] '%s' %s -> %s pnl=$%.2f (winner=%s)",
                t["id"], source, title[:35], t["outcome"],
                "WON" if won else "LOST", pnl, winner,
            )
            resolved_count += 1

    # ── Cycle-level summary ───────────────────────────────────────────────────
    parts = []
    if resolved_count:
        parts.append(f"{resolved_count} resolved")
    if stale_end_date_count:
        parts.append(f"{stale_end_date_count} stale_endDate skipped")
    if truly_lost_count:
        parts.append(f"{truly_lost_count} truly_lost")
    if parts:
        log.info("resolution cycle: %s", ", ".join(parts))
    else:
        log.debug("resolution cycle: nothing to report (all markets still open)")
