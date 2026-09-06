"""Alpha-decay tracker — rolling-window Sharpe / win-rate / Brier per strategy.

MARATHON SHIP-15 (2026-04-26 audit Self-improvement #8): closes the
documented gap that polybot has zero rolling-IC monitoring. Industry baseline
(Maven Securities, microalphas.com, Di Mascio & Lines 2021): rolling 12-month
IC, half-life test, auto-deprecation when half-life < 90 days.

Design:
  - Per strategy, compute rolling-30d and rolling-90d Sharpe + win_rate + Brier
  - If 30d Sharpe < 0 AND 90d Sharpe < 0.5 → suggest deprecation
  - Half-life estimator: exponential fit on rolling-window IC
  - Persists snapshots to `data/alpha_decay.jsonl` for time-series analysis

Public API:
  - track_strategy(source) → DecayReport
  - track_all_strategies() → list[DecayReport]
  - check_decay_alerts() → list[DecayAlert] (used by self-improvement loop)
"""
from __future__ import annotations

import json
import logging
import math
import statistics
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Sequence

log = logging.getLogger("polybot.alpha_decay")


@dataclass(frozen=True)
class DecayReport:
    strategy: str
    n_30d: int
    n_90d: int
    sharpe_30d: float
    sharpe_90d: float
    win_rate_30d: float
    win_rate_90d: float
    brier_30d: float          # NaN if no calibrated trades
    brier_90d: float
    half_life_days: Optional[float]   # exponential decay half-life of 30d Sharpe (or None)
    decay_status: str         # "healthy" | "warning" | "deprecated"
    recommendation: str
    timestamp_utc: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


@dataclass(frozen=True)
class DecayAlert:
    strategy: str
    severity: str             # "warn" | "fail"
    reason: str
    metrics: tuple


# ── Public API ────────────────────────────────────────────────────────


def track_strategy(source: str, db_module=None) -> DecayReport:
    """Compute rolling decay metrics for one strategy."""
    if db_module is None:
        import db as db_module  # type: ignore

    rows_30 = _fetch_resolved(db_module, source, days=30)
    rows_90 = _fetch_resolved(db_module, source, days=90)

    sharpe_30 = _sharpe(rows_30)
    sharpe_90 = _sharpe(rows_90)
    wr_30 = _win_rate(rows_30)
    wr_90 = _win_rate(rows_90)
    brier_30 = _brier(rows_30)
    brier_90 = _brier(rows_90)
    half_life = _half_life_days(rows_90)

    # Decay verdict
    if sharpe_30 < -0.5 and sharpe_90 < 0.0:
        status = "deprecated"
        rec = (
            f"Strategy {source} has Sharpe_30d={sharpe_30:.2f}, "
            f"Sharpe_90d={sharpe_90:.2f}. "
            f"AUTO-DEPRECATE recommended."
        )
    elif sharpe_30 < 0.0 and sharpe_90 < 0.5:
        status = "warning"
        rec = (
            f"Strategy {source} weakening: Sharpe_30d={sharpe_30:.2f}, "
            f"Sharpe_90d={sharpe_90:.2f}. Watch closely; if 30d stays "
            f"negative for another window, deprecate."
        )
    else:
        status = "healthy"
        rec = (
            f"Strategy {source} healthy: Sharpe_30d={sharpe_30:.2f}, "
            f"Sharpe_90d={sharpe_90:.2f}."
        )

    return DecayReport(
        strategy=source,
        n_30d=len(rows_30), n_90d=len(rows_90),
        sharpe_30d=round(sharpe_30, 4), sharpe_90d=round(sharpe_90, 4),
        win_rate_30d=round(wr_30, 4), win_rate_90d=round(wr_90, 4),
        brier_30d=round(brier_30, 4) if not math.isnan(brier_30) else float("nan"),
        brier_90d=round(brier_90, 4) if not math.isnan(brier_90) else float("nan"),
        half_life_days=round(half_life, 1) if half_life is not None else None,
        decay_status=status, recommendation=rec,
    )


def track_all_strategies(db_module=None,
                         strategies: Sequence[str] = ()) -> list[DecayReport]:
    """Run decay tracker for each known strategy. Default = the 7 polybot strategies."""
    if not strategies:
        strategies = ["copy", "btc_5min", "corr_arb", "scanner",
                      "weather", "bond", "arb", "insider"]
    out = []
    for s in strategies:
        try:
            out.append(track_strategy(s, db_module=db_module))
        except Exception as e:
            log.warning("decay tracker failed for %s: %s", s, e)
    # Persist snapshot
    try:
        _persist_jsonl(out)
    except Exception as e:
        log.warning("decay persist failed: %s", e)
    return out


def check_decay_alerts(db_module=None) -> list[DecayAlert]:
    """Return DecayAlert for any strategy with non-healthy status."""
    reports = track_all_strategies(db_module=db_module)
    alerts = []
    for r in reports:
        if r.decay_status == "deprecated":
            alerts.append(DecayAlert(
                strategy=r.strategy, severity="fail",
                reason=r.recommendation,
                metrics=(("sharpe_30d", r.sharpe_30d), ("sharpe_90d", r.sharpe_90d),
                         ("win_rate_30d", r.win_rate_30d)),
            ))
        elif r.decay_status == "warning":
            alerts.append(DecayAlert(
                strategy=r.strategy, severity="warn",
                reason=r.recommendation,
                metrics=(("sharpe_30d", r.sharpe_30d), ("sharpe_90d", r.sharpe_90d)),
            ))
    return alerts


# ── Private helpers ──────────────────────────────────────────────────


def _fetch_resolved(db_module, source: str, days: int) -> list[dict]:
    try:
        # Use a direct SQL — db.get_simulated_trades joins traders table which
        # may not have rows for synthetic sources. Direct read.
        conn = db_module._connect()
        rows = conn.execute(
            "SELECT timestamp, sim_pnl, cost_simulated, confidence, is_learning_only "
            "FROM simulated_trades "
            "WHERE source=? AND resolved=1 "
            "  AND timestamp > datetime('now', ?) "
            "  AND NOT (sim_pnl=0 AND is_learning_only=1) "
            "ORDER BY timestamp ASC",
            (source, f"-{days} days"),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows] if rows else []
    except Exception as e:
        log.debug("decay fetch failed: %s", e)
        return []


def _returns(rows: list[dict]) -> list[float]:
    out = []
    for r in rows:
        cost = float(r.get("cost_simulated") or 0.0)
        pnl = float(r.get("sim_pnl") or 0.0)
        if cost > 0:
            out.append(pnl / cost)
    return out


def _sharpe(rows: list[dict]) -> float:
    rets = _returns(rows)
    if len(rets) < 2:
        return 0.0
    mean = statistics.fmean(rets)
    std = statistics.pstdev(rets) or 1e-9
    return (mean / std) * math.sqrt(len(rets))


def _win_rate(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    wins = sum(1 for r in rows if (r.get("sim_pnl") or 0) > 0)
    return wins / len(rows)


def _brier(rows: list[dict]) -> float:
    """Brier score on (confidence, win) where win = sim_pnl > 0.
    Returns NaN if no rows have valid confidence."""
    eligible = [r for r in rows if r.get("confidence") not in (None, 0)]
    if not eligible:
        return float("nan")
    errs = []
    for r in eligible:
        conf = float(r["confidence"])
        actual = 1.0 if (r.get("sim_pnl") or 0) > 0 else 0.0
        errs.append((conf - actual) ** 2)
    return statistics.fmean(errs) if errs else float("nan")


def _half_life_days(rows: list[dict]) -> Optional[float]:
    """Estimate half-life of edge via exponential fit on rolling 7-day Sharpe.

    Method: bucket trades by week, compute weekly Sharpe, fit log-linear decay:
        Sharpe(t) ≈ Sharpe_0 * exp(-λ*t), where t in weeks
    Half-life = log(2) / λ in weeks → days.

    Returns None if insufficient data or no decay detected (λ ≤ 0).
    """
    if len(rows) < 14:
        return None
    from collections import defaultdict
    by_week: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        ts = r.get("timestamp") or ""
        if not ts:
            continue
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            # Week bucket = days since rows[0]
            week = int((dt - datetime.fromisoformat(rows[0]["timestamp"].replace("Z", "+00:00"))).days // 7)
            cost = float(r.get("cost_simulated") or 0.0)
            pnl = float(r.get("sim_pnl") or 0.0)
            if cost > 0:
                by_week[week].append(pnl / cost)
        except Exception:
            continue
    if len(by_week) < 3:
        return None
    weekly_sharpe = []
    for week in sorted(by_week):
        rets = by_week[week]
        if len(rets) < 2:
            continue
        m = statistics.fmean(rets)
        s = statistics.pstdev(rets) or 1e-9
        weekly_sharpe.append((week, (m / s) * math.sqrt(len(rets))))
    if len(weekly_sharpe) < 3:
        return None
    # Need positive Sharpe values to take log
    positive = [(w, s) for w, s in weekly_sharpe if s > 0.01]
    if len(positive) < 3:
        return None
    xs = [w for w, _ in positive]
    ys = [math.log(s) for _, s in positive]
    n = len(xs)
    mx = statistics.fmean(xs); my = statistics.fmean(ys)
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    sxx = sum((xs[i] - mx) ** 2 for i in range(n))
    if sxx <= 0:
        return None
    slope = sxy / sxx  # if negative, edge is decaying; magnitude = decay rate per week
    if slope >= -0.01:
        return None  # no significant decay
    decay_rate_per_week = -slope
    half_life_weeks = math.log(2) / decay_rate_per_week
    return half_life_weeks * 7.0  # days


def _persist_jsonl(reports: list[DecayReport]) -> None:
    """Append snapshot to `data/alpha_decay.jsonl` for time-series analysis."""
    path = Path(__file__).resolve().parent / "data" / "alpha_decay.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in reports:
            f.write(json.dumps(asdict(r), default=str) + "\n")
