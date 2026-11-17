"""Offline smoke test for polytrack's paper-trading write path.

Copies src/polytrack into a temporary folder (the code writes its SQLite files next to
itself), points every HTTP proxy at a dead port so no request can leave, then drives the
real functions: record a paper trade with an injected order book, check the slippage
columns, check the pre-trade firewall and the halt refusal, resolve the trade and read
the summary. Prints one line per check and "SMOKE PASS" at the end; exits 1 on failure.

Run from the repository root:  python .claude/skills/verify/smoke_offline.py
"""
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ[var] = "http://127.0.0.1:9"
os.environ.pop("NO_PROXY", None)
os.environ.pop("no_proxy", None)

repo = Path(__file__).resolve().parents[3]
work = Path(tempfile.mkdtemp(prefix="polytrack-smoke-"))
pkg = work / "polytrack"
shutil.copytree(repo / "src" / "polytrack", pkg,
                ignore=shutil.ignore_patterns("__pycache__", "data"))
sys.path.insert(0, str(pkg))
sys.dont_write_bytecode = True

import config  # noqa: E402
import db  # noqa: E402
from brain.halt_state import HaltState  # noqa: E402
from fill_simulator import simulate_fill  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


assert Path(config.DB_PATH).is_relative_to(work), config.DB_PATH
db.init()
check("db.init creates the trade database", Path(config.DB_PATH).exists(), config.DB_PATH)

book = {"bids": [{"price": "0.48", "size": "100"}],
        "asks": [{"price": "0.50", "size": "10"}, {"price": "0.52", "size": "100"}]}
fill = simulate_fill("tok", "BUY", 10.0, fetch_book=lambda _t: book)
check("simulate_fill walks the book past the first level",
      fill.is_filled and fill.levels_consumed == 2 and fill.fill_price > 0.50,
      f"fill_price={fill.fill_price:.4f} levels={fill.levels_consumed} slippage_bps={fill.slippage_bps:.1f}")

db.log_simulated_trade("0xsmoke", "Smoke market A", "YES", "BUY", price=0.49, shares=20.0,
                       cost=10.0, confidence=0.6, source="copy", token_id="tok",
                       expected_price=0.49, book_fetcher=lambda _t: book)
conn = sqlite3.connect(config.DB_PATH)
conn.row_factory = sqlite3.Row
row = conn.execute("SELECT * FROM simulated_trades WHERE market_title LIKE '%smoke market a%'"
                   " COLLATE NOCASE").fetchone()
check("paper trade recorded with fill_price and slippage_bps",
      row is not None and row["fill_price"] is not None and row["slippage_bps"] is not None,
      f"fill_price={row['fill_price'] if row else None} slippage_bps={row['slippage_bps'] if row else None}")

before = conn.execute("SELECT COUNT(*) FROM simulated_trades").fetchone()[0]
db.log_simulated_trade("0xsmoke", "Smoke market B", "YES", "BUY", price=1.5, shares=1.0,
                       cost=1.5, source="copy")
after = conn.execute("SELECT COUNT(*) FROM simulated_trades").fetchone()[0]
check("firewall refuses a price outside [0, 1]", after == before)

HaltState().apply_halt("smoke_test_halt", scope="system", details={"why": "smoke"})
db.log_simulated_trade("0xsmoke", "Smoke market C", "YES", "BUY", price=0.4, shares=5.0,
                       cost=2.0, source="copy")
after_halt = conn.execute("SELECT COUNT(*) FROM simulated_trades").fetchone()[0]
check("system-wide halt refuses new trade writes", after_halt == after)

try:
    db.resolve_trade(row["id"], won=True, pnl=10.4, resolved_outcome="Yes")
    res = conn.execute("SELECT resolved, brier_score FROM simulated_trades WHERE id=?",
                       (row["id"],)).fetchone()
    check("resolve_trade sets resolved and Brier score",
          res["resolved"] == 1 and abs(res["brier_score"] - 0.16) < 1e-9,
          f"brier={res['brier_score']}")
except sqlite3.Error as exc:
    check("resolve_trade sets resolved and Brier score", False, f"{type(exc).__name__}: {exc}")
summary = db.get_simulation_summary()
check("simulation summary counts the trade", summary.get("total_trades") == 1, str(summary))
conn.close()

shutil.rmtree(work, ignore_errors=True)
print("SMOKE FAIL: " + ", ".join(failures) if failures else "SMOKE PASS")
sys.exit(1 if failures else 0)
