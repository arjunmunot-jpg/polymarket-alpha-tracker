---
name: verify
description: Run polymarket-alpha-tracker's paper-trading path end to end, offline, and prove that a change works. Use before every commit that changes behaviour, and whenever you need evidence that a fix or feature works.
---
# Verify polymarket-alpha-tracker

The project has no automated tests. The proof is an offline smoke test that drives the real
paper-trading write path, plus an import check. Neither makes a network call or places an order.

## Before you start

Works the same on the laptop and in a cloud session. Run from the repository root.

1. `python3 -m venv /tmp/polytrack-venv && /tmp/polytrack-venv/bin/pip install -r requirements.txt`
   (about 30 seconds on first install). Use `/tmp/polytrack-venv/bin/python` as `python` below.
2. No environment variables are needed for these checks. API keys (names in `.env.example`) are
   only read by the parts that call outside services.

## Fast checks

- Import check, under a second; expect `imports ok`:
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/polytrack python -c "import config, db, alpha_decay, capital_allocator, confidence_trend, fill_simulator, insider_detector, market_scanner, polymarket_api, regime_detector, resolution_checker, risk_guard, strategy_tracker, tracker, trade_selector, whale_watcher, brain.critic, brain.event_bus, brain.halt_state; print('imports ok')"`
  `brain.sign_triggers` and `brain.trigger_matcher` are left out because they import
  `brain.toml_loader`, which is not in this repository.

## End-to-end run

1. Run: `python .claude/skills/verify/smoke_offline.py`
2. What it does: copies `src/polytrack` to a temporary folder (the code writes its SQLite files
   next to itself), points the HTTP proxies at a dead port, and calls the real functions:
   `db.init`, `fill_simulator.simulate_fill` with an injected order book,
   `db.log_simulated_trade` with that book, the firewall with a price above 1, a system-wide halt
   through `brain.halt_state.HaltState`, `db.resolve_trade`, and `db.get_simulation_summary`.
3. Expect one `ok` line per check and `SMOKE PASS`, exit code 0. The `[FIREWALL REJECT]`,
   `HALT APPLIED` and `REFUSING trade write` log lines are the expected behaviour, not errors.
4. Clean up: nothing to do; the script deletes its temporary folder. It does not write to the
   repository's own data folders.

Current state (2026-11-17): six checks pass and `resolve_trade` fails with
`OperationalError: no such column: resolved_outcome`, so the script ends with `SMOKE FAIL` and
exit code 1. That failure is in the project code, not in the script. When a change fixes it, the
script must end with `SMOKE PASS`; until then, any other failing line is a regression.

## User flows to check

| Flow | How to drive it | Expected result |
|---|---|---|
| Record a paper trade with slippage | Smoke test, "paper trade recorded" line | `fill_price` and `slippage_bps` are set; a 10 USD buy against the injected book fills at about 0.5098 |
| Pre-trade firewall | Smoke test, "firewall refuses" line | No row is written for a price outside 0 to 1 |
| Halt stops trading | Smoke test, "system-wide halt" line | No row is written while a system-scope halt is active |
| Resolve a trade | Smoke test, "resolve_trade" line | `resolved = 1` and Brier score 0.16 for confidence 0.6 that won |
| Market scanning, wallet discovery, resolution checks | Not verifiable offline | These call the Polymarket APIs; see Known problems |

When a change touches a flow the smoke test does not cover, add a check to
`smoke_offline.py` that calls the real function with injected or recorded data, and keep the
script offline.

## Evidence to report

Report the commands you ran and their output lines (the `ok` and `FAIL` lines and the final
`SMOKE` line). If a step could not run, say which and why, and treat the change as unverified.

## Known problems

- `db.resolve_trade` fails on a fresh database (see Current state above).
- Anything that fetches market data (`market_scanner`, `tracker`, `insider_detector`,
  `regime_detector`, `resolution_checker`, `whale_watcher`, `polymarket_api`) has no offline
  check. Running those modules calls public Polymarket endpoints; do that only when the owner asks,
  and only with `DRY_RUN = True`.
