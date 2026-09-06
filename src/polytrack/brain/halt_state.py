"""SQLite-backed halt state.

Tables:
  halt_events              — append-only audit log of apply/clear events
  halt_strategies          — CURRENT halt state, one row per (trigger, scope, strategy)
  block_pending_review     — current block-pending items (pipeline gates)

Operations:
  apply_halt(trigger, scope, details)
  clear_halt(trigger_name, user_confirmation, scope=..., strategy=...)
  is_halted(scope, strategy)
  active_halts() -> list[dict]
  apply_block(trigger, details)
  clear_block(trigger, user_confirmation)
  active_blocks() -> list[dict]

Per runtime_constraints in TOML:
  "halt state persists to disk (SQLite), survives restart"
  "unblock = explicit DB write with user+timestamp+typed confirmation"

Confirmation token: user MUST pass the literal string "I-UNBLOCK-<trigger_name>"
to clear. Typo = rejected. Prevents accidental unblock.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


log = logging.getLogger("polybot.brain.halt_state")


# R3 (2026-08-04, VERIFIED BY RUNNING): this default was RELATIVE, so it resolved
# against the process cwd. `python3 -m brain.cli status` from any other directory
# silently CREATED a fresh empty store and reported "HALTS: 0 active" — output
# indistinguishable from a genuinely clear system, in the authoritative SAFETY
# store. test_no_relative_db_paths pins relative sqlite3.connect LITERALS and
# exempts variables "by convention"; this default broke that very convention.
# Anchored to the repo root, matching copy_paper_trader.py.
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data/brain/halt_state.sqlite"


_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS halt_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_utc TEXT NOT NULL,
        action TEXT NOT NULL,
        trigger_name TEXT NOT NULL,
        scope TEXT NOT NULL,
        strategy TEXT,
        actor TEXT,
        confirmation TEXT,
        details_json TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS halt_strategies (
        trigger_name TEXT NOT NULL,
        scope TEXT NOT NULL,
        strategy TEXT NOT NULL DEFAULT '',
        applied_at_utc TEXT NOT NULL,
        details_json TEXT,
        acknowledged INTEGER NOT NULL DEFAULT 0,
        acknowledged_at_utc TEXT,
        acknowledged_by TEXT,
        PRIMARY KEY (trigger_name, scope, strategy)
    )""",
    # Migrations for pre-acknowledgment DBs (safe to run on existing tables)
    """ALTER TABLE halt_strategies ADD COLUMN acknowledged INTEGER NOT NULL DEFAULT 0""",
    """ALTER TABLE halt_strategies ADD COLUMN acknowledged_at_utc TEXT""",
    """ALTER TABLE halt_strategies ADD COLUMN acknowledged_by TEXT""",
    """CREATE TABLE IF NOT EXISTS block_pending_review (
        trigger_name TEXT NOT NULL,
        correlation_id TEXT,
        applied_at_utc TEXT NOT NULL,
        details_json TEXT,
        PRIMARY KEY (trigger_name, correlation_id)
    )""",
    """CREATE INDEX IF NOT EXISTS idx_halt_events_ts ON halt_events(ts_utc)""",
    """CREATE INDEX IF NOT EXISTS idx_halt_strategies_scope ON halt_strategies(scope, strategy)""",
]


def confirmation_token(trigger_name: str) -> str:
    """Required user confirmation string to clear a halt/block."""
    return f"I-UNBLOCK-{trigger_name}"


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class HaltState:
    """Thread-safe wrapper over a SQLite file for halt state."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ---- Connection helpers ----

    def _connect(self) -> sqlite3.Connection:
        # WAVE 6 FIX 2026-04-26 (council finding #4): removed isolation_level=None
        # (autocommit mode). Was making apply_halt and clear_halt non-atomic —
        # DELETE+INSERT to halt_strategies+halt_events ran as 3 separate auto-
        # commit txns, so a process kill or DB lock between them could leave
        # the halt cleared with NO audit trail. Default isolation (deferred)
        # plus `with conn:` blocks now wrap everything in a single transaction.
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @contextmanager
    def _cursor(self):
        # WAVE 6 FIX 2026-04-26: `with conn:` block wraps in a transaction —
        # all writes inside the context are atomic (committed on exit success,
        # rolled back on exception). Pairs with the autocommit removal above.
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    yield conn
            finally:
                conn.close()

    def _init_schema(self) -> None:
        with self._cursor() as conn:
            for stmt in _SCHEMA:
                try:
                    conn.execute(stmt)
                except Exception as e:
                    # ALTER TABLE ... ADD COLUMN fails with "duplicate column name"
                    # on existing databases — this is expected and safe to ignore.
                    if "duplicate column name" in str(e).lower():
                        pass
                    else:
                        raise

    # ---- Halt ----

    def apply_halt(self, trigger_name: str, scope: str,
                   details: Optional[dict[str, Any]] = None,
                   strategy: str = "", actor: str = "brain") -> None:
        """Record a halt. Safe to re-apply (upsert by composite key)."""
        ts = _now_utc_iso()
        details_json = json.dumps(details or {}, default=str, sort_keys=True)
        with self._cursor() as conn:
            # HARDENING AUDIT FIX 2026-07-06: was INSERT OR REPLACE — REPLACE
            # deletes-and-reinserts the row, so re-applying an active halt
            # silently reset acknowledged/acknowledged_at_utc/acknowledged_by
            # (owned by acknowledge_halt). Upsert only the columns apply_halt
            # owns; acknowledgment state must survive a re-fire.
            conn.execute(
                """INSERT OR IGNORE INTO halt_strategies
                   (trigger_name, scope, strategy, applied_at_utc, details_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (trigger_name, scope, strategy or "", ts, details_json),
            )
            conn.execute(
                """UPDATE halt_strategies
                   SET applied_at_utc = ?, details_json = ?
                   WHERE trigger_name = ? AND scope = ? AND strategy = ?""",
                (ts, details_json, trigger_name, scope, strategy or ""),
            )
            conn.execute(
                """INSERT INTO halt_events
                   (ts_utc, action, trigger_name, scope, strategy, actor, confirmation, details_json)
                   VALUES (?, 'apply_halt', ?, ?, ?, ?, NULL, ?)""",
                (ts, trigger_name, scope, strategy or "", actor, details_json),
            )
        log.warning(
            "HALT APPLIED: trigger=%s scope=%s strategy=%r actor=%s",
            trigger_name, scope, strategy, actor,
        )
        # WAVE 6 FIX 2026-04-26 (council finding #5): operator notification.
        # Previously halt events were silent — only watchdog notified, and
        # only on process death. Knight Capital "bot alive but not trading"
        # was operationally invisible. Now: every halt fires a macOS notif.
        try:
            from brain.notify import notify
            scope_label = f"{scope}" + (f"/{strategy}" if strategy else "")
            notify(
                title="Polybot HALT applied",
                message=f"{trigger_name} ({scope_label}) by {actor}",
                sound="Basso",
                dedup_sec=180,  # don't spam repeated re-applies
                dedup_key=f"halt|{trigger_name}|{scope}|{strategy}",
            )
        except Exception as _e:
            log.debug("notify on apply_halt failed: %s", _e)

    def clear_halt(self, trigger_name: str, user_confirmation: str,
                   scope: str = "", strategy: str = "",
                   actor: str = "user") -> bool:
        """Clear a halt. Requires typed confirmation token.

        ULTRA-REVIEW WAVE 4 HARDENING 2026-04-25: confirmation token comparison
        is now whitespace-tolerant (strip) and case-insensitive (upper). Operators
        paste tokens from logs/Slack/notifications — strict comparison was
        rejecting valid tokens with trailing newlines or case variants.

        Returns True if cleared, False if no matching row or wrong token.
        """
        expected = confirmation_token(trigger_name)
        # Tolerant comparison: strip whitespace + case-fold
        if (user_confirmation or "").strip().upper() != expected.upper():
            log.warning(
                "clear_halt REJECTED: wrong confirmation for %s (got %r, expected %r)",
                trigger_name, user_confirmation, expected,
            )
            return False

        ts = _now_utc_iso()
        with self._cursor() as conn:
            cur = conn.execute(
                "SELECT * FROM halt_strategies WHERE trigger_name = ?"
                + (" AND scope = ?" if scope else "")
                + (" AND strategy = ?" if strategy else ""),
                tuple(
                    [trigger_name]
                    + ([scope] if scope else [])
                    + ([strategy] if strategy else [])
                ),
            )
            rows = cur.fetchall()
            if not rows:
                log.info("clear_halt: no matching halt for %s", trigger_name)
                return False
            for row in rows:
                conn.execute(
                    """DELETE FROM halt_strategies
                       WHERE trigger_name = ? AND scope = ? AND strategy = ?""",
                    (row["trigger_name"], row["scope"], row["strategy"]),
                )
                conn.execute(
                    """INSERT INTO halt_events
                       (ts_utc, action, trigger_name, scope, strategy, actor, confirmation, details_json)
                       VALUES (?, 'clear_halt', ?, ?, ?, ?, ?, NULL)""",
                    (ts, row["trigger_name"], row["scope"], row["strategy"],
                     actor, user_confirmation),
                )
        log.warning("HALT CLEARED: trigger=%s actor=%s", trigger_name, actor)
        return True

    def acknowledge_halt(self, trigger_name: str, actor: str = "operator",
                         scope: str = "", strategy: str = "") -> int:
        """Mark halt(s) as acknowledged by a human operator.

        Returns the number of rows acknowledged.
        Acknowledgment is informational — it does NOT clear the halt.
        Use clear_halt() to actually remove the halt.
        """
        ts = _now_utc_iso()
        with self._cursor() as conn:
            sql = (
                "UPDATE halt_strategies SET acknowledged=1, "
                "acknowledged_at_utc=?, acknowledged_by=? "
                "WHERE trigger_name=?"
            )
            params: list = [ts, actor, trigger_name]
            if scope:
                sql += " AND scope=?"
                params.append(scope)
            if strategy:
                sql += " AND strategy=?"
                params.append(strategy)
            cur = conn.execute(sql, tuple(params))
            changed = cur.rowcount
            if changed:
                conn.execute(
                    """INSERT INTO halt_events
                       (ts_utc, action, trigger_name, scope, strategy, actor, confirmation, details_json)
                       VALUES (?, 'acknowledge', ?, ?, ?, ?, NULL, NULL)""",
                    (ts, trigger_name, scope or "", strategy or "", actor),
                )
        log.info("HALT ACKNOWLEDGED: trigger=%s actor=%s rows=%d", trigger_name, actor, changed)
        return changed

    def unacknowledged_halts(self, older_than_minutes: float = 30.0) -> list[dict]:
        """Return active halts that have NOT been acknowledged and are older than N minutes.

        Used by the alert acknowledgment probe and the brain runner to flag
        operators about halts that haven't been seen.
        """
        from datetime import datetime, timedelta, timezone
        cutoff = (
            datetime.now(timezone.utc) - timedelta(minutes=older_than_minutes)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self._cursor() as conn:
            cur = conn.execute(
                """SELECT trigger_name, scope, strategy, applied_at_utc, details_json,
                          acknowledged, acknowledged_at_utc, acknowledged_by
                   FROM halt_strategies
                   WHERE acknowledged = 0 AND applied_at_utc <= ?
                   ORDER BY applied_at_utc ASC""",
                (cutoff,),
            )
            out = []
            for r in cur.fetchall():
                entry = _row_to_halt(r)
                entry["acknowledged"] = bool(r["acknowledged"])
                entry["acknowledged_at_utc"] = r["acknowledged_at_utc"]
                entry["acknowledged_by"] = r["acknowledged_by"]
                # Compute age in minutes
                try:
                    applied = datetime.fromisoformat(r["applied_at_utc"].replace("Z", "+00:00"))
                    age_min = (datetime.now(timezone.utc) - applied).total_seconds() / 60
                    entry["age_minutes"] = round(age_min, 1)
                except Exception:
                    entry["age_minutes"] = None
                out.append(entry)
            return out

    def is_halted(self, scope: str = "", strategy: str = "") -> bool:
        """Is anything halted matching scope+strategy? If both empty: global."""
        with self._cursor() as conn:
            sql = "SELECT 1 FROM halt_strategies WHERE 1=1"
            params: list[str] = []
            if scope:
                sql += " AND scope = ?"
                params.append(scope)
            if strategy:
                sql += " AND strategy = ?"
                params.append(strategy)
            sql += " LIMIT 1"
            cur = conn.execute(sql, tuple(params))
            return cur.fetchone() is not None

    def active_halts(self) -> list[dict]:
        """List all current halts (sorted by applied_at_utc).

        Each dict includes 'acknowledged' bool and 'acknowledged_by' for dashboard display.
        """
        with self._cursor() as conn:
            cur = conn.execute(
                """SELECT trigger_name, scope, strategy, applied_at_utc, details_json,
                          acknowledged, acknowledged_at_utc, acknowledged_by
                   FROM halt_strategies
                   ORDER BY applied_at_utc ASC"""
            )
            out = []
            for r in cur.fetchall():
                entry = _row_to_halt(r)
                entry["acknowledged"] = bool(r["acknowledged"])
                entry["acknowledged_at_utc"] = r["acknowledged_at_utc"]
                entry["acknowledged_by"] = r["acknowledged_by"]
                out.append(entry)
            return out

    def is_halted_by_trigger(self, trigger_name: str, scope: str = "",
                             strategy: str = "") -> bool:
        """Is there an active halt for this specific trigger_name?

        Used by BrainRunner to avoid duplicate halt entries.
        """
        with self._cursor() as conn:
            sql = "SELECT 1 FROM halt_strategies WHERE trigger_name = ?"
            params: list[str] = [trigger_name]
            if scope:
                sql += " AND scope = ?"
                params.append(scope)
            if strategy is not None and strategy != "":
                sql += " AND strategy = ?"
                params.append(strategy)
            sql += " LIMIT 1"
            cur = conn.execute(sql, tuple(params))
            return cur.fetchone() is not None

    def auto_clear_by_trigger(self, trigger_name: str, actor: str = "brain_runner") -> int:
        """Clear ALL halt_strategies rows for this trigger without a typed token.

        This is for bot-internal auto-recovery only (probe now passes → clear its
        previous halt automatically). Human-initiated unblocks always require the
        typed confirmation token via clear_halt().

        Returns number of rows cleared.
        """
        ts = _now_utc_iso()
        with self._cursor() as conn:
            cur = conn.execute(
                "SELECT trigger_name, scope, strategy FROM halt_strategies WHERE trigger_name = ?",
                (trigger_name,),
            )
            rows = cur.fetchall()
            if not rows:
                return 0
            conn.execute(
                "DELETE FROM halt_strategies WHERE trigger_name = ?",
                (trigger_name,),
            )
            for row in rows:
                conn.execute(
                    """INSERT INTO halt_events
                       (ts_utc, action, trigger_name, scope, strategy, actor, confirmation, details_json)
                       VALUES (?, 'auto_clear', ?, ?, ?, ?, NULL, NULL)""",
                    (ts, row["trigger_name"], row["scope"], row["strategy"], actor),
                )
        log.info("HALT auto-cleared: trigger=%s rows=%d actor=%s", trigger_name, len(rows), actor)
        return len(rows)

    # ---- Block pending review ----

    def apply_block(self, trigger_name: str,
                    correlation_id: str = "",
                    details: Optional[dict[str, Any]] = None,
                    actor: str = "brain") -> None:
        ts = _now_utc_iso()
        details_json = json.dumps(details or {}, default=str, sort_keys=True)
        with self._cursor() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO block_pending_review
                   (trigger_name, correlation_id, applied_at_utc, details_json)
                   VALUES (?, ?, ?, ?)""",
                (trigger_name, correlation_id, ts, details_json),
            )
            conn.execute(
                """INSERT INTO halt_events
                   (ts_utc, action, trigger_name, scope, strategy, actor, confirmation, details_json)
                   VALUES (?, 'apply_block', ?, 'block_pending_review', ?, ?, NULL, ?)""",
                (ts, trigger_name, correlation_id, actor, details_json),
            )

    def clear_block(self, trigger_name: str, user_confirmation: str,
                    correlation_id: str = "", actor: str = "user") -> bool:
        # ULTRA-REVIEW WAVE 4 HARDENING 2026-04-25: tolerant token comparison.
        expected = confirmation_token(trigger_name)
        if (user_confirmation or "").strip().upper() != expected.upper():
            return False
        ts = _now_utc_iso()
        with self._cursor() as conn:
            sql = "DELETE FROM block_pending_review WHERE trigger_name = ?"
            params: list[str] = [trigger_name]
            if correlation_id:
                sql += " AND correlation_id = ?"
                params.append(correlation_id)
            cur = conn.execute(sql, tuple(params))
            changed = cur.rowcount
            if changed:
                conn.execute(
                    """INSERT INTO halt_events
                       (ts_utc, action, trigger_name, scope, strategy, actor, confirmation, details_json)
                       VALUES (?, 'clear_block', ?, 'block_pending_review', ?, ?, ?, NULL)""",
                    (ts, trigger_name, correlation_id or "", actor, user_confirmation),
                )
            return changed > 0

    def active_blocks(self) -> list[dict]:
        with self._cursor() as conn:
            cur = conn.execute(
                """SELECT trigger_name, correlation_id, applied_at_utc, details_json
                   FROM block_pending_review
                   ORDER BY applied_at_utc ASC"""
            )
            out = []
            for r in cur.fetchall():
                out.append({
                    "trigger_name": r["trigger_name"],
                    "correlation_id": r["correlation_id"],
                    "applied_at_utc": r["applied_at_utc"],
                    "details": _loads(r["details_json"]),
                })
            return out

    # ---- Audit log ----

    def event_log(self, limit: int = 100) -> list[dict]:
        with self._cursor() as conn:
            cur = conn.execute(
                """SELECT ts_utc, action, trigger_name, scope, strategy, actor,
                          confirmation, details_json
                   FROM halt_events
                   ORDER BY id DESC LIMIT ?""",
                (limit,),
            )
            out = []
            for r in cur.fetchall():
                out.append({
                    "ts_utc": r["ts_utc"],
                    "action": r["action"],
                    "trigger_name": r["trigger_name"],
                    "scope": r["scope"],
                    "strategy": r["strategy"],
                    "actor": r["actor"],
                    "confirmation": r["confirmation"],
                    "details": _loads(r["details_json"]),
                })
            return out


# ---- Private helpers ----


def _row_to_halt(row: sqlite3.Row) -> dict:
    return {
        "trigger_name": row["trigger_name"],
        "scope": row["scope"],
        "strategy": row["strategy"],
        "applied_at_utc": row["applied_at_utc"],
        "details": _loads(row["details_json"]),
    }


def _loads(s: Optional[str]) -> dict:
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {"_parse_error": True, "raw": s}
