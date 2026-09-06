"""Per-strategy performance tracking — Phase 3.

Maintains rolling Sharpe ratio, win rate, and PnL per strategy (copy, insider, conviction).
Called from main.py's monitor cycle whenever trades resolve.

The `strategy_performance` table stores one row per (strategy, date), updated by
full recomputation each time (simpler and always correct vs. incremental updates).
"""

from __future__ import annotations

import math
import logging
from datetime import datetime, timezone

import db

log = logging.getLogger("polybot.strategy_tracker")

STRATEGIES = ("copy", "insider", "conviction", "btc_5min", "btc_arb", "scanner", "bond", "arb", "weather")


# ---------------------------------------------------------------------------
# Internal math
# ---------------------------------------------------------------------------

def _compute_sharpe(pnls: list[float]) -> float:
    """Sharpe ratio from a list of per-trade PnL values."""
    if len(pnls) < 3:
        return 0.0
    mean = sum(pnls) / len(pnls)
    variance = sum((p - mean) ** 2 for p in pnls) / len(pnls)
    std = math.sqrt(variance)
    return round((mean / std) if std > 0 else 0.0, 4)


def _compute_max_drawdown(pnls: list[float]) -> float:
    """Maximum cumulative drawdown from a sequence of PnL values.

    Simulates a running equity curve starting at 0, tracks the worst
    peak-to-trough decline. Returns a positive number representing the
    magnitude of the worst drawdown (e.g., 3.5 = lost $3.50 peak to trough).
    """
    if not pnls:
        return 0.0
    peak = 0.0
    equity = 0.0
    max_dd = 0.0
    for pnl in pnls:
        equity += pnl
        if equity > peak:
            peak = equity
        drawdown = peak - equity
        if drawdown > max_dd:
            max_dd = drawdown
    return max_dd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def update_on_resolve(trade_row: dict):
    """Update strategy_performance for the trade's strategy and today's date.

    Recomputes full-day stats (not incremental) to stay correct under retries.

    Args:
        trade_row: dict from simulated_trades (needs: source, sim_pnl, confidence)
    """
    strategy = trade_row.get("source", "copy")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    day_trades = db.get_trades_by_strategy_and_date(strategy, today)
    if not day_trades:
        return

    total = len(day_trades)
    won = sum(1 for t in day_trades if float(t.get("sim_pnl") or 0) > 0)
    total_pnl = sum(float(t.get("sim_pnl") or 0) for t in day_trades)
    avg_conf = sum(float(t.get("confidence") or 0) for t in day_trades) / total
    pnls = [float(t.get("sim_pnl") or 0) for t in day_trades]
    sharpe = _compute_sharpe(pnls)

    # Brier score: average (confidence - outcome)² — 0=perfect, 0.25=random guesser
    resolved_trades = [t for t in day_trades if t.get("brier_score") is not None]
    avg_brier = (
        sum(float(t["brier_score"]) for t in resolved_trades) / len(resolved_trades)
        if resolved_trades else None
    )

    # Max drawdown: worst cumulative loss streak (ordered by rowid as proxy for time)
    max_drawdown = _compute_max_drawdown(pnls)

    db.upsert_strategy_day(
        strategy=strategy, date=today,
        trades_total=total, trades_won=won,
        total_pnl=round(total_pnl, 4),
        avg_confidence=round(avg_conf, 4),
        sharpe=sharpe,
        avg_brier=round(avg_brier, 4) if avg_brier is not None else None,
        max_drawdown=round(max_drawdown, 4),
    )
    log.debug(
        "Strategy '%s' day stats: %d trades, won=%d, PnL=%.2f, Sharpe=%.3f",
        strategy, total, won, total_pnl, sharpe
    )


def get_summary() -> dict:
    """Return per-strategy aggregate metrics for the dashboard.

    Returns:
        {
          "copy":       {"trades_total": N, "trades_won": N, "win_rate": 0.6,
                         "total_pnl": 5.2, "avg_confidence": 0.72, "sharpe": 0.4},
          "insider":    {...},
          "conviction": {...},
        }
    """
    rows = db.get_strategy_performance()
    result = {}

    for strategy in STRATEGIES:
        strategy_rows = [r for r in rows if r["strategy"] == strategy]
        if not strategy_rows:
            result[strategy] = {
                "trades_total": 0,
                "trades_won": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_confidence": 0.0,
                "sharpe": 0.0,
            }
            continue

        total = sum(r["trades_total"] for r in strategy_rows)
        won   = sum(r["trades_won"] for r in strategy_rows)
        pnl   = sum(r["total_pnl"] for r in strategy_rows)
        latest = max(strategy_rows, key=lambda r: r["date"])

        result[strategy] = {
            "trades_total":   total,
            "trades_won":     won,
            "win_rate":       round(won / total, 3) if total else 0.0,
            "total_pnl":      round(pnl, 2),
            "avg_confidence": round(latest["avg_confidence"], 3),
            "sharpe":         round(latest["sharpe"], 3),
            "avg_brier":      round(latest.get("avg_brier") or 0.0, 3),
            "max_drawdown":   round(latest.get("max_drawdown") or 0.0, 2),
        }

    return result
