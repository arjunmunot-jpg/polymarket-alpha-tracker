# Review guide for polymarket-alpha-tracker

## What failure looks like here

- A paper position, fill price, P&L or Brier score is computed wrong, so the research conclusions
  drawn from the ledger are wrong without anyone noticing.
- A safety guard (the pre-trade firewall in `db._in_line_firewall`, the halt check in
  `db.log_simulated_trade`, `risk_guard`) fails open or swallows an error, so trades are recorded
  that should have been refused, or a stale or default value is used as if it were real data.

## Check when the diff touches these

| Trigger in the diff | Check | Why (incident) |
|---|---|---|
| `except` blocks, fallbacks, default values in any module | The failure is logged at warning level or above, and a safety check still refuses on error | A fill-simulator fallback logged at debug hid that 0 of 214 trades got slippage (comment in `db.log_simulated_trade`, WAVE 12 SHIP-4) |
| Halt scopes or `src/polytrack/brain/halt_state.py` | Every reader handles the scope names the writers use; non-strategy halts stop every write | The reader checked `scope="global"` after the writer moved to `"system"`; six strategies traded under an active halt (R4, 2026-08-04) |
| Parsing of API responses (`polymarket_api`, `fill_simulator`, `tracker`) | The code handles the real response shape, such as order book levels as dicts with string numbers | Levels were assumed to be lists, so no fill was ever simulated (WAVE 13 SHIP-1, 2026-05-16) |
| Thresholds in `config.py` | Live behaviour reads `LIVE_CONSTRAINTS`, not the LEARNING MODE module values | The module values are deliberately loosened for data collection |
| `DRY_RUN` or `LIVE_COPY_ARMED` | The owner approved the change in chat | Together they arm real orders |
| SQL writes in `db.py` or `halt_state.py` | Columns exist in the schema created by `init` or `_init_schema`, and upserts do not reset columns owned by another function | `resolve_trade` writes a column that `init` does not create; an `INSERT OR REPLACE` once reset halt acknowledgments (2026-07-06) |
| `.gitignore`, `data/`, roster or database paths | No roster, wallet list or database is committed | The roster is the unpublished research output |

## Skip

- `data/` and any `*.db` or `*.sqlite*` file (local research output).
- `requirements.txt` version pins, unless the change breaks the setup command.

## Findings

- Tag each finding Important (must fix before merge), Nit (optional), or Pre-existing (already
  wrong before this change; do not hold the author to it).
- Give the file and line for each finding, say why it is wrong, and say how to show it fails
  (a command, an input, a check in `.claude/skills/verify/smoke_offline.py`). A guess from a
  function name is not evidence.
- Report at most 3 nits; if there are more, give their count in one line.
- Skip style and formatting. Flag the class of a bug so every instance gets fixed, not only the
  line you saw.

## Summary

Start the summary with a count, for example "2 Important, 1 Nit, 0 Pre-existing". When there
is nothing to fix, say so in one line.
