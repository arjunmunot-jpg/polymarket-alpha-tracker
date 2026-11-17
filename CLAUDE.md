# polymarket-alpha-tracker

A personal research system that finds consistently profitable Polymarket wallets, scores them,
turns their new positions into signals, and records what it would have traded in a paper-trading
ledger (SQLite) with simulated slippage. Its output is research for the owner; nobody trades from
it directly.

## Non-goals

- It does not place real orders. It runs in paper mode (`DRY_RUN = True` in `config.py`); the live
  executor the config comments mention is not in this repository.
- It does not publish the curated wallet roster. `data/roster.json` is the project's research
  output and is gitignored; do not commit wallet lists, sample rosters or trade databases.

## Commands that prove work

Run from the repository root with the virtual environment's `python` (after set up, put
`/tmp/polytrack-venv/bin` first on `PATH`). The code uses flat imports (`import config`,
`import db`), so the import path is `src/polytrack`, not `src`.

- Set up: `python3 -m venv /tmp/polytrack-venv && /tmp/polytrack-venv/bin/pip install -r requirements.txt`
- Tests: there are none. `pytest` finds no tests and exits with code 5. Do not report a pytest run
  as evidence.
- Import check (offline, under a second):
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/polytrack python -c "import config, db, alpha_decay, capital_allocator, confidence_trend, fill_simulator, insider_detector, market_scanner, polymarket_api, regime_detector, resolution_checker, risk_guard, strategy_tracker, tracker, trade_selector, whale_watcher, brain.critic, brain.event_bus, brain.halt_state; print('imports ok')"`
- Paper-trading smoke test (offline, under a second): `python .claude/skills/verify/smoke_offline.py`.
  It prints one line per check and `SMOKE PASS` or `SMOKE FAIL`. On 2026-11-17 it fails one
  check, `resolve_trade`, because of the schema bug below; the other six pass.
- End to end: use the `verify` skill in `.claude/skills/verify/`.
- No type checker or linter is configured.

## Where the code misleads

- The README's `python -m polytrack.market_scanner` does not work: `polytrack` is not installed
  as a package, and `market_scanner.py` has no `__main__` block. There is no single entry point in
  this repository; only `src/polytrack/confidence_trend.py` and
  `src/polytrack/brain/sign_triggers.py` have one.
- The repository is an extract of a larger private codebase. Many imports point at modules that
  are not here (`llm_swarm`, `champion_overrides`, `brain.notify`, `brain.toml_loader`,
  `brain.panel`, `researcher`, `news_agent`, `review.*`, `execution/`). Most sit inside
  `try`/`except` and fail silently, so a feature can look wired while doing nothing.
  `src/polytrack/brain/sign_triggers.py` and `src/polytrack/brain/trigger_matcher.py` cannot be
  imported at all. Comments also cite `MISTAKES.md` and `GO_LIVE_CRITERIA.md`, which are not in this repository.
- `db.resolve_trade` writes a `resolved_outcome` column that `db.init` never creates, so resolving
  a trade fails on a fresh database (`no such column: resolved_outcome`). Older databases from
  the full codebase may have the column.
- The module-level thresholds in `config.py` are LEARNING MODE values, widened to collect data.
  The live values are in `config.LIVE_CONSTRAINTS`; code that models live behaviour must read
  those.
- `trades.db` and `halt_state.sqlite` are written to `src/polytrack/data/`, next to the code. The
  roster is read from `data/` at the repository root, and `.gitignore` only covers root `data/`,
  so the databases under `src/polytrack/data/` are not ignored. Check `git status` before
  committing after a run.
- `python-dotenv` is in `requirements.txt` but nothing loads `.env`. API keys reach the code only
  as exported environment variables.
- `db.log_simulated_trade` returns `None` both when it records a trade and when the firewall or a
  halt refuses it. Check the `simulated_trades` table, not the return value.

## Decisions that are not in the code

- Turning off `DRY_RUN` or turning on `LIVE_COPY_ARMED` needs the owner's explicit approval in
  chat each time, because together they arm real orders. Do not change either flag as part of
  other work or from a script.
- Safety checks fail closed: when the halt database or the firewall cannot be checked, the trade
  write is refused. Keep new guards that way, and make them refuse on any halt that is not
  strategy-scoped: a check that looked for one scope name missed system-wide halts and let six
  strategies trade through one (see the comment in `db.log_simulated_trade`).
