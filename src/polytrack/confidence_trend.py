#!/usr/bin/env python3
"""
Confidence trend tracker — writes data/reports/confidence_trend_YYYYMMDD.md

Shows rolling Brier scores, calibration curves, EV realisation, critic scores,
and a traffic-light panel for each go-live criterion.

Usage:
    python3 confidence_trend.py
    python3 confidence_trend.py --db data/trades.db --out data/reports
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPORTS_DIR = Path("data/reports")
DB_PATH     = str(Path(__file__).resolve().parents[2] / "data" / "trades.db")  # resolved from the package root, never the cwd (a relative path picks up a decoy DB)


# ── helpers ────────────────────────────────────────────────────────────────

def _q(conn: sqlite3.Connection, sql: str, params=()):
    return conn.execute(sql, params).fetchall()


def _brier(trades: list[tuple]) -> float:
    """Compute Brier score from (confidence, win) pairs."""
    if not trades:
        return float("nan")
    return sum((conf - win) ** 2 for conf, win in trades) / len(trades)


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """IB-003 (Council R3, Phase 4 research 04_calibration_sub100): Wilson score interval
    for a binomial proportion. More accurate than Wald CI especially at small N or
    proportions near 0 or 1.

    Returns (lower, upper) bounds. Defaults to 95% CI (z=1.96).

    For btc_5min at N=53, observed 22 wins (41.5%):
        Wilson 95% CI = (0.290, 0.547) — contains 0.5 → cannot reject "no edge" hypothesis
    """
    if n <= 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = (z * (((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def _bucket_win_rates(trades: list[tuple]) -> list[dict]:
    """Build 0.1-wide calibration buckets from (confidence, win) pairs.

    IB-003: includes Wilson 95% CI per bucket (more accurate than Wald at small N).
    """
    buckets: dict[float, list] = {}
    for conf, win in trades:
        b = round(int(conf * 10) * 0.1, 1)
        buckets.setdefault(b, []).append(win)
    out = []
    for b, v in sorted(buckets.items()):
        n = len(v)
        wins = int(sum(v))
        win_rate = wins / n if n else 0.0
        ci_lo, ci_hi = wilson_ci(wins, n)
        out.append({
            "bucket": b,
            "n": n,
            "win_rate": win_rate,
            "ci95_lo": round(ci_lo, 4),
            "ci95_hi": round(ci_hi, 4),
            "ci95_width": round(ci_hi - ci_lo, 4),
        })
    return out


def _is_monotonic(buckets: list[dict]) -> bool:
    rates = [b["win_rate"] for b in buckets if b["n"] >= 3]
    return all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))


def _rolling_brier(conn: sqlite3.Connection, source: str, days: int = 30) -> float:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    # WAVE 6 FIX 2026-04-26 (council finding #1): exclude force-stale-flagged
    # zombie rows where sim_pnl=0 AND is_learning_only=1. 362 weather rows
    # bulk-resolved on disable would have polluted Brier as fake "losses".
    # Same filter as dashboard.py win-rate (Wave 1 fix #1, both sites).
    rows = _q(conn, """
        SELECT confidence, CASE WHEN sim_pnl > 0 THEN 1.0 ELSE 0.0 END
        FROM simulated_trades
        WHERE source=? AND resolved=1 AND is_live_eligible=1
          AND timestamp >= ? AND confidence > 0
          AND NOT (sim_pnl = 0 AND is_learning_only = 1)
    """, (source, cutoff))
    return _brier(rows)


def _ev_realisation(conn: sqlite3.Connection, source: str) -> dict:
    """Compare predicted EV vs realised P&L fraction."""
    rows = _q(conn, """
        SELECT ev, sim_pnl, cost_simulated
        FROM simulated_trades
        WHERE source=? AND resolved=1 AND ev IS NOT NULL AND cost_simulated > 0
    """, (source,))
    if not rows:
        return {}
    pred_ev = sum(ev * cost for ev, pnl, cost in rows) / sum(cost for _, _, cost in rows)
    real_ev = sum(pnl for _, pnl, _ in rows) / sum(cost for _, _, cost in rows)
    return {"predicted_ev": round(pred_ev, 4), "realised_ev": round(real_ev, 4),
            "ratio": round(real_ev / pred_ev, 2) if pred_ev != 0 else None}


def _critic_avg(reports_dir: Path, prefix: str, days: int = 7) -> float | None:
    """Parse critic markdown reports and average the score lines."""
    import re, os, time
    cutoff = time.time() - days * 86400
    scores = []
    for p in sorted(reports_dir.glob(f"{prefix}_*.md")):
        if os.path.getmtime(p) < cutoff:
            continue
        text = p.read_text(errors="replace")
        m = re.search(r"(?:score|rating)[:\s]+(\d+(?:\.\d+)?)\s*/\s*5", text, re.I)
        if m:
            scores.append(float(m.group(1)))
    return round(sum(scores) / len(scores), 2) if scores else None


def _count_go_live_gates(criteria_path: Path) -> tuple[int, int]:
    """Count checked vs total hard-gate checkboxes in GO_LIVE_CRITERIA.md."""
    if not criteria_path.exists():
        return 0, 0
    text = criteria_path.read_text()
    import re
    checked = len(re.findall(r"- \[x\]", text, re.I))
    total   = len(re.findall(r"- \[[ x]\]", text, re.I))
    return checked, total


# ── main ────────────────────────────────────────────────────────────────────

def run(db_path: str = DB_PATH, out_dir: str = str(REPORTS_DIR)) -> Path:
    conn = sqlite3.connect(db_path)
    reports = Path(out_dir)
    reports.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = reports / f"confidence_trend_{today}.md"

    sources = [r[0] for r in _q(conn, "SELECT DISTINCT source FROM simulated_trades")]

    lines = [
        f"# Polybot Confidence Trend — {today}",
        f"_Generated {datetime.now(timezone.utc).isoformat()}_",
        "",
        "## Rolling 30-day Brier scores (live-eligible trades only)",
        "",
        "| Strategy | N | Brier (30d) | Null hyp Brier | Better? | Monotonic cal? |",
        "|----------|---|-------------|----------------|---------|----------------|",
    ]

    for src in sorted(sources):
        # WAVE 6 FIX 2026-04-26 (council finding #1): zombie row exclusion.
        rows = _q(conn, """
            SELECT confidence, CASE WHEN sim_pnl > 0 THEN 1.0 ELSE 0.0 END
            FROM simulated_trades
            WHERE source=? AND resolved=1 AND is_live_eligible=1 AND confidence > 0
              AND NOT (sim_pnl = 0 AND is_learning_only = 1)
        """, (src,))
        if not rows:
            continue
        n = len(rows)
        brier = _brier(rows)
        # Null hypothesis: use avg price as predictor (Brier of always-predicting base rate)
        avg_win = sum(w for _, w in rows) / n
        null_brier = avg_win * (1 - avg_win)  # Brier for always-predicting mean
        better = "✅" if brier < null_brier else "❌"
        buckets = _bucket_win_rates(rows)
        mono = "✅" if _is_monotonic(buckets) else "❌"
        lines.append(
            f"| {src} | {n} | {brier:.4f} | {null_brier:.4f} | {better} | {mono} |"
        )

    lines += ["", "## Calibration buckets (all-time, live-eligible)", ""]
    for src in sorted(sources):
        # WAVE 6 FIX 2026-04-26 (council finding #1): zombie row exclusion.
        rows = _q(conn, """
            SELECT confidence, CASE WHEN sim_pnl > 0 THEN 1.0 ELSE 0.0 END
            FROM simulated_trades
            WHERE source=? AND resolved=1 AND is_live_eligible=1 AND confidence > 0
              AND NOT (sim_pnl = 0 AND is_learning_only = 1)
        """, (src,))
        if not rows:
            continue
        buckets = _bucket_win_rates(rows)
        if not buckets:
            continue
        lines.append(f"### {src}")
        # ULTRA-REVIEW BRIER AUDIT 2026-04-25: now display Wilson 95% CI columns.
        # IB-003 was half-shipped — wilson_ci() function existed at line 37 + populated
        # ci95_lo/ci95_hi/ci95_width in the bucket dict at line 75-77, but the markdown
        # render here suppressed those fields. Now exposed for proper calibration audit
        # (esp. small-N buckets where point-estimate is misleading).
        lines.append("| Conf bucket | N | Win rate | 95% CI | CI width |")
        lines.append("|-------------|---|----------|--------|----------|")
        for b in buckets:
            ci_lo = b.get("ci95_lo", 0)
            ci_hi = b.get("ci95_hi", 0)
            ci_w = b.get("ci95_width", ci_hi - ci_lo)
            lines.append(
                f"| {b['bucket']:.1f} | {b['n']} | {b['win_rate']:.1%} "
                f"| [{ci_lo:.1%}, {ci_hi:.1%}] | {ci_w:.1%} |"
            )
        lines.append("")

    lines += ["## EV realisation (predicted vs actual)", ""]
    for src in sorted(sources):
        ev = _ev_realisation(conn, src)
        if not ev:
            continue
        ratio = ev.get("ratio")
        ratio_str = f"{ratio:.2f}×" if ratio else "n/a"
        flag = "✅" if ratio and ratio > 0 else "❌"
        lines.append(
            f"- **{src}**: predicted EV={ev['predicted_ev']:.1%}  "
            f"realised EV={ev['realised_ev']:.1%}  ratio={ratio_str} {flag}"
        )

    # Critic scores
    lines += ["", "## Critic scores (7-day rolling avg)", ""]
    proc = _critic_avg(reports, "critic_process")
    res  = _critic_avg(reports, "critic_results")
    lines.append(f"- Process critic: {'%.2f/5' % proc if proc else 'no data'}"
                 f"  {'✅' if proc and proc >= 3.5 else '🟡'}")
    lines.append(f"- Results critic: {'%.2f/5' % res if res else 'no data'}"
                 f"  {'✅' if res and res >= 3.5 else '🟡'}")

    # Go-live gate count
    criteria = Path("GO_LIVE_CRITERIA.md")
    checked, total = _count_go_live_gates(criteria)
    lines += [
        "",
        "## Go-live gate progress",
        "",
        f"**{checked}/{total} hard gates met.**",
        "",
        "_(Open GO_LIVE_CRITERIA.md and tick boxes manually as each gate clears)_",
    ]

    # Traffic lights summary
    # Get latest review probe run
    probe_fail = 0
    try:
        from review.registry import ProbeRegistry
        from review.context_impl import SqliteReadonlyView, BotLogReader, HttpApiReader
        from review.base import ProbeContext
        ctx = ProbeContext(
            db=SqliteReadonlyView(db_path),
            logs=BotLogReader(log_path="bot.log"),
            api=HttpApiReader(),
        )
        reg = ProbeRegistry()
        results = reg.run_tier("nightly", ctx)
        probe_fail = sum(1 for r in results if r.status == "FAIL")
    except Exception:
        probe_fail = -1  # unknown

    probe_str = f"{probe_fail} FAIL" if probe_fail >= 0 else "unknown"
    probe_icon = "✅" if probe_fail == 0 else ("🟡" if probe_fail <= 2 else "🔴")

    lines += [
        "",
        "## Traffic lights at a glance",
        "",
        "| Metric | Value | Status |",
        "|--------|-------|--------|",
        f"| Review probe FAILs | {probe_str} | {probe_icon} |",
        f"| Process critic (7d avg) | {'%.2f/5' % proc if proc else 'no data'} | {'✅' if proc and proc >= 3.5 else '🟡'} |",
        f"| Results critic (7d avg) | {'%.2f/5' % res if res else 'no data'} | {'✅' if res and res >= 3.5 else '🟡'} |",
        f"| Go-live hard gates | {checked}/{total} | {'✅' if checked == total else '🔴'} |",
    ]

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Confidence trend written: {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",  default=DB_PATH)
    parser.add_argument("--out", default=str(REPORTS_DIR))
    args = parser.parse_args()
    sys.exit(0 if run(args.db, args.out) else 1)
