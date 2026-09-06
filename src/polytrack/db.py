"""SQLite database layer for Polybot."""

import sqlite3
import json
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional  # WAVE 12 SHIP-3 (2026-05-16): for log_shadow_trade signature

import config


def _connect() -> sqlite3.Connection:
    Path(config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    # WAVE 6 FIX 2026-04-26 (Round-6 council finding #4): default 5s timeout
    # made dedup-check exception path silently fire under news_agent INSERT
    # contention. Increased to 30s + busy_timeout PRAGMA. Combined with #5
    # fail-CLOSED on dedup exception, this turns stochastic duplicates into
    # one-cycle skips.
    conn = sqlite3.connect(config.DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init():
    """Create tables if they don't exist."""
    conn = _connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS traders (
            address         TEXT PRIMARY KEY,
            username        TEXT,
            first_seen      TEXT NOT NULL,
            last_scored     TEXT,
            score           REAL DEFAULT 0,
            pnl             REAL DEFAULT 0,
            volume          REAL DEFAULT 0,
            trade_count     INTEGER DEFAULT 0,
            is_copying      INTEGER DEFAULT 0,
            copy_sub_id     TEXT,
            dropped         INTEGER DEFAULT 0,
            drop_reason     TEXT,
            extra           TEXT  -- JSON blob for flexible data
        );

        CREATE TABLE IF NOT EXISTS copy_events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            trader_address  TEXT NOT NULL,
            event_type      TEXT NOT NULL,  -- started, stopped, paused, resumed
            timestamp       TEXT NOT NULL,
            reason          TEXT,
            FOREIGN KEY (trader_address) REFERENCES traders(address)
        );

        CREATE TABLE IF NOT EXISTS executions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            trader_address  TEXT NOT NULL,
            market_title    TEXT,
            outcome         TEXT,
            side            TEXT,
            price           REAL,
            shares          REAL,
            cost_usd        REAL,
            timestamp       TEXT NOT NULL,
            pnl             REAL,
            resolved        INTEGER DEFAULT 0,
            FOREIGN KEY (trader_address) REFERENCES traders(address)
        );

        CREATE TABLE IF NOT EXISTS learning_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            analysis        TEXT NOT NULL,  -- Claude's full analysis
            actions_taken   TEXT,           -- JSON: score changes, drops, etc.
            token_usage     INTEGER
        );

        CREATE TABLE IF NOT EXISTS insider_wallets (
            address         TEXT PRIMARY KEY,
            confidence      REAL NOT NULL,
            signals         TEXT,           -- JSON: what triggered detection
            first_flagged   TEXT NOT NULL,
            is_copying      INTEGER DEFAULT 0,
            copy_sub_id     TEXT
        );

        CREATE TABLE IF NOT EXISTS simulated_trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            trader_address  TEXT NOT NULL,
            market_title    TEXT,
            outcome         TEXT,
            side            TEXT,
            price           REAL,
            shares_simulated REAL,
            cost_simulated  REAL,
            timestamp       TEXT NOT NULL,
            current_price   REAL,
            sim_pnl         REAL DEFAULT 0,
            resolved        INTEGER DEFAULT 0,
            confidence      REAL DEFAULT 0,
            source          TEXT DEFAULT 'copy',
            kelly_fraction  REAL,
            ev              REAL,
            FOREIGN KEY (trader_address) REFERENCES traders(address)
        );

        CREATE TABLE IF NOT EXISTS strategy_performance (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy        TEXT NOT NULL,
            date            TEXT NOT NULL,
            trades_total    INTEGER DEFAULT 0,
            trades_won      INTEGER DEFAULT 0,
            total_pnl       REAL DEFAULT 0,
            avg_confidence  REAL DEFAULT 0,
            sharpe          REAL DEFAULT 0,
            UNIQUE(strategy, date)
        );

        CREATE TABLE IF NOT EXISTS candidate_signals (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            market_title    TEXT,
            outcome         TEXT,
            price           REAL,
            confidence      REAL,
            ev              REAL,
            kelly_fraction  REAL,
            source          TEXT,
            trader_address  TEXT,
            market_volume   REAL,
            end_date        TEXT,
            was_approved    INTEGER DEFAULT 0,
            rejection_reason TEXT
        );

        -- WAVE 12 SHIP-3 (2026-05-16): shadow_trades for live-vs-dry divergence audit.
        -- Pre-go-live invariant: when DRY_RUN flips False, every live trade gets a
        -- synthetic shadow leg (same logic against same data) and both go in this
        -- table. Probe c38_shadow_ab_drift reads it nightly and flags >20% median
        -- divergence (signals fill-model bug or real slippage delta).
        -- Currently empty (DRY_RUN=True) — writer log_shadow_trade exists but
        -- has no caller until execution_gateway lands (Phase 5).
        CREATE TABLE IF NOT EXISTS shadow_trades (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            market_title        TEXT NOT NULL,
            outcome             TEXT NOT NULL,
            side                TEXT,
            source              TEXT,
            -- shadow (dry-run simulation) leg
            shadow_price        REAL,
            shadow_shares       REAL,
            shadow_cost         REAL,
            shadow_pnl          REAL,
            shadow_slippage_bps REAL,
            shadow_fill_price   REAL,
            -- live (real-money) leg
            live_price          REAL,
            live_shares         REAL,
            live_cost           REAL,
            live_pnl            REAL,
            live_slippage_bps   REAL,
            live_fill_price     REAL,
            -- meta
            created_at          TEXT DEFAULT CURRENT_TIMESTAMP,
            notes               TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_shadow_trades_created ON shadow_trades(created_at);
        CREATE INDEX IF NOT EXISTS idx_shadow_trades_market ON shadow_trades(market_title, outcome);
    """)
    # Schema migrations — add columns to existing tables if missing
    _migrate(conn)
    # Ensure synthetic trader rows exist for algorithmic strategies.
    # simulated_trades.trader_address has a FK → traders.address, so algo sources
    # (scanner, weather, btc_5min, bond, arb) must have a row here or logging crashes.
    _ensure_synthetic_traders(conn)
    # WAVE 6 FIX 2026-04-26 (Round-6 council finding #2): partial unique index
    # on open positions. Closes the TOCTOU race in is_already_simulated → INSERT
    # path. Two strategies racing on same market+outcome will now get
    # IntegrityError on the second INSERT (caught downstream); DB enforces
    # what application-level dedup couldn't.
    try:
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS uniq_open_position
            ON simulated_trades (
                LOWER(TRIM(market_title)),
                LOWER(TRIM(outcome))
            ) WHERE resolved=0
        """)
    except Exception as _e:
        # Existing dups would block index creation. Log + continue (existing
        # 4-layer dedup still active; index is belt-and-suspenders).
        import logging as _logging
        _logging.getLogger("polybot.db").warning(
            "uniq_open_position index creation skipped: %s", _e,
        )
    conn.commit()
    conn.close()


def _ensure_synthetic_traders(conn: sqlite3.Connection):
    """Insert placeholder rows in traders for algorithmic strategy sources.

    simulated_trades.trader_address has a FOREIGN KEY → traders.address.
    Algo strategies use source names as trader_address ("scanner", "btc_signal", etc.)
    which don't exist as real traders. Without these rows every algo log call raises
    sqlite3.IntegrityError: FOREIGN KEY constraint failed.
    INSERT OR IGNORE is safe — won't touch existing rows.
    """
    now = datetime.now(timezone.utc).isoformat()
    # scanner = used by scanner, bond_scanner, arb_scanner (all in market_scanner.py)
    # weather = used by weather_bot.py
    # btc_signal = used by btc_5min.py
    # correlation_arb = used by correlation_arb.py
    # arb_detector = used by arb_detector.py
    for addr in ("scanner", "weather", "btc_signal", "btc_arb", "arb", "bond", "news_agent", "conviction", "correlation_arb", "arb_detector"):
        conn.execute(
            "INSERT OR IGNORE INTO traders (address, username, first_seen) VALUES (?, ?, ?)",
            (addr, addr, now),
        )


def _migrate(conn: sqlite3.Connection):
    """Add new columns to existing tables without breaking old data."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(simulated_trades)")}
    if "confidence" not in existing:
        conn.execute("ALTER TABLE simulated_trades ADD COLUMN confidence REAL DEFAULT 0")
    if "source" not in existing:
        conn.execute("ALTER TABLE simulated_trades ADD COLUMN source TEXT DEFAULT 'copy'")
    # Phase 3 columns — NULL on existing rows, shown as "—" in dashboard
    if "kelly_fraction" not in existing:
        conn.execute("ALTER TABLE simulated_trades ADD COLUMN kelly_fraction REAL")
    if "ev" not in existing:
        conn.execute("ALTER TABLE simulated_trades ADD COLUMN ev REAL")
    # Phase 4.5 — market end date for resolve deadline tracking
    if "end_date" not in existing:
        conn.execute("ALTER TABLE simulated_trades ADD COLUMN end_date TEXT")
    # Phase 5 — calibration, learning mode, resolution timestamp
    if "brier_score" not in existing:
        conn.execute("ALTER TABLE simulated_trades ADD COLUMN brier_score REAL")
    if "is_learning_only" not in existing:
        conn.execute("ALTER TABLE simulated_trades ADD COLUMN is_learning_only INTEGER DEFAULT 0")
    if "resolved_at" not in existing:
        conn.execute("ALTER TABLE simulated_trades ADD COLUMN resolved_at TEXT")
    if "swarm_votes" not in existing:
        conn.execute("ALTER TABLE simulated_trades ADD COLUMN swarm_votes TEXT")
    # Phase 8 — calibration scope filter: only trades with ev>0 AND kelly>0 are
    # "live-eligible" and used to build the calibration curve. Trained on all but
    # inference happens on live-eligible only — this column closes that gap.
    if "is_live_eligible" not in existing:
        conn.execute("ALTER TABLE simulated_trades ADD COLUMN is_live_eligible INTEGER DEFAULT 0")
        # Backfill existing rows: mark as eligible when ev>0 AND kelly_fraction>0
        conn.execute(
            "UPDATE simulated_trades SET is_live_eligible=1 "
            "WHERE ev > 0 AND kelly_fraction > 0 AND is_live_eligible=0"
        )
    # Phase 9 — stale end_date tracking: when Polymarket endDate is stale
    # placeholder data, resolution_checker NULLs end_date and sets this flag
    # so we stop chasing zombies until the market actually closes.
    if "end_date_stale" not in existing:
        conn.execute("ALTER TABLE simulated_trades ADD COLUMN end_date_stale INTEGER DEFAULT 0")
    # WAVE 10 SHIP-9 (2026-04-26 post-Marathon audit Dry-run P0 #1): wire
    # fill_simulator into the trade-write path. Was: trades recorded at
    # observed mid `price`, slippage = $0 silently (audit: "the dry-run
    # number Arjun stares at on dashboard is a lie of omission"). Now:
    # callers pass `token_id` to log_simulated_trade; orderbook walked;
    # actual fill_price + slippage_bps + fill_pct persisted alongside
    # the canonical mid (`expected_price`). Backtest c25 + calibration +
    # EV inputs all corrected. Old call sites (no token_id) leave columns
    # NULL and behavior is unchanged — additive opt-in path.
    if "fill_price" not in existing:
        conn.execute("ALTER TABLE simulated_trades ADD COLUMN fill_price REAL")
    if "expected_price" not in existing:
        conn.execute("ALTER TABLE simulated_trades ADD COLUMN expected_price REAL")
    if "slippage_bps" not in existing:
        conn.execute("ALTER TABLE simulated_trades ADD COLUMN slippage_bps REAL")
    if "fill_pct" not in existing:
        conn.execute("ALTER TABLE simulated_trades ADD COLUMN fill_pct REAL")

    # strategy_performance — calibration + drawdown metrics
    sp_existing = {row[1] for row in conn.execute("PRAGMA table_info(strategy_performance)")}
    if "avg_brier" not in sp_existing:
        conn.execute("ALTER TABLE strategy_performance ADD COLUMN avg_brier REAL DEFAULT 0")
    if "max_drawdown" not in sp_existing:
        conn.execute("ALTER TABLE strategy_performance ADD COLUMN max_drawdown REAL DEFAULT 0")

    # learning_log — suggestion quality tracking
    ll_existing = {row[1] for row in conn.execute("PRAGMA table_info(learning_log)")}
    if "suggestion_id" not in ll_existing:
        conn.execute("ALTER TABLE learning_log ADD COLUMN suggestion_id TEXT")
    if "verified" not in ll_existing:
        conn.execute("ALTER TABLE learning_log ADD COLUMN verified INTEGER DEFAULT 0")
    if "verified_at" not in ll_existing:
        conn.execute("ALTER TABLE learning_log ADD COLUMN verified_at TEXT")
    if "predicted_improvement" not in ll_existing:
        conn.execute("ALTER TABLE learning_log ADD COLUMN predicted_improvement TEXT")
    if "actual_improvement" not in ll_existing:
        conn.execute("ALTER TABLE learning_log ADD COLUMN actual_improvement TEXT")


# --- Trader helpers ---

def upsert_trader(address: str, username: str, pnl: float, volume: float,
                   trade_count: int, score: float):
    conn = _connect()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO traders (address, username, first_seen, last_scored, score, pnl, volume, trade_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(address) DO UPDATE SET
            username=excluded.username, last_scored=excluded.last_scored,
            score=excluded.score, pnl=excluded.pnl, volume=excluded.volume,
            trade_count=excluded.trade_count
    """, (address, username, now, now, score, pnl, volume, trade_count))
    conn.commit()
    conn.close()


def set_copying(address: str, sub_id: str):
    conn = _connect()
    now = datetime.now(timezone.utc).isoformat()
    # Ensure trader exists before copy_events FK insert (insider wallets may not be in traders yet).
    conn.execute(
        "INSERT OR IGNORE INTO traders (address, username, first_seen) VALUES (?, ?, ?)",
        (address, address, now),
    )
    conn.execute("UPDATE traders SET is_copying=1, copy_sub_id=? WHERE address=?",
                 (sub_id, address))
    conn.execute("INSERT INTO copy_events (trader_address, event_type, timestamp) VALUES (?, 'started', ?)",
                 (address, now))
    conn.commit()
    conn.close()


def set_stopped(address: str, reason: str = ""):
    conn = _connect()
    now = datetime.now(timezone.utc).isoformat()
    # Ensure trader exists before copy_events FK insert (mirrors set_copying upsert).
    conn.execute(
        "INSERT OR IGNORE INTO traders (address, username, first_seen) VALUES (?, ?, ?)",
        (address, address, now),
    )
    conn.execute("UPDATE traders SET is_copying=0, copy_sub_id=NULL, dropped=1, drop_reason=? WHERE address=?",
                 (reason, address))
    # ULTRA-REVIEW INSIDER AUDIT 2026-04-25: also clear insider_wallets.is_copying to
    # keep the dual columns in sync. Was causing 31 insiders to show is_copying=1 in
    # insider_wallets while traders had is_copying=0 (split-brain). Pipeline that reads
    # `traders` (run_copy_cycle) couldn't see them. Now both tables agree.
    conn.execute("UPDATE insider_wallets SET is_copying=0, copy_sub_id=NULL WHERE address=?",
                 (address,))
    conn.execute("INSERT INTO copy_events (trader_address, event_type, timestamp, reason) VALUES (?, 'stopped', ?, ?)",
                 (address, now, reason))
    conn.commit()
    conn.close()


def get_active_copies() -> list[dict]:
    conn = _connect()
    rows = conn.execute("SELECT * FROM traders WHERE is_copying=1").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_traders() -> list[dict]:
    conn = _connect()
    rows = conn.execute("SELECT * FROM traders ORDER BY score DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def log_execution(trader_address: str, market_title: str, outcome: str,
                  side: str, price: float, shares: float, cost_usd: float):
    conn = _connect()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO executions (trader_address, market_title, outcome, side, price, shares, cost_usd, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (trader_address, market_title, outcome, side, price, shares, cost_usd, now))
    conn.commit()
    conn.close()


def log_learning(analysis: str, actions: dict, token_usage: int,
                  suggestion_id: str = None, predicted_improvement: str = None):
    conn = _connect()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO learning_log
            (timestamp, analysis, actions_taken, token_usage, suggestion_id, predicted_improvement)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (now, analysis, json.dumps(actions), token_usage, suggestion_id, predicted_improvement))
    conn.commit()
    conn.close()


def get_unverified_suggestions(max_age_days: int = 7) -> list[dict]:
    """Return learning_log rows that are unverified and older than max_age_days."""
    conn = _connect()
    rows = conn.execute("""
        SELECT id, suggestion_id, predicted_improvement, timestamp
        FROM learning_log
        WHERE (verified IS NULL OR verified=0)
          AND suggestion_id IS NOT NULL
          AND timestamp <= datetime('now', ?)
    """, (f"-{max_age_days} days",)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def verify_suggestion(suggestion_id: str, actual_improvement: str):
    """Mark a learning suggestion as verified with the observed outcome."""
    conn = _connect()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        UPDATE learning_log
        SET verified=1, verified_at=?, actual_improvement=?
        WHERE suggestion_id=?
    """, (now, actual_improvement, suggestion_id))
    conn.commit()
    conn.close()


def log_candidate_signal(market_title: str, outcome: str, price: float,
                         confidence: float, ev: float, kelly: float,
                         source: str, trader_address: str,
                         market_volume: float, end_date: str,
                         was_approved: bool, rejection_reason: str = ""):
    """Log every candidate signal (approved or rejected) for backtest replay."""
    conn = _connect()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO candidate_signals
        (timestamp, market_title, outcome, price, confidence, ev, kelly_fraction,
         source, trader_address, market_volume, end_date, was_approved, rejection_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (now, market_title, outcome, price, confidence, ev, kelly,
          source, trader_address, market_volume, end_date,
          1 if was_approved else 0, rejection_reason))
    conn.commit()
    conn.close()


def update_candidate_signal_status(market_title: str, outcome: str,
                                   was_approved: bool,
                                   rejection_reason: str = "") -> int:
    """Resolve a `pending_allocation` candidate row by id-DESC most-recent match.

    WAVE 6 FIX 2026-04-25 (council Phase 3 corr_arb black hole): correlation_arb
    writes pre-allocation rows with rejection_reason='pending_allocation' BEFORE
    calling capital_allocator.allocate(). The allocator has 6 silent-drop sites
    that don't write rejection rows, leaving 672/1340 corr_arb candidates as
    orphans in 'pending_allocation' limbo per 24h. This function patches the
    most-recent pending row to a terminal status post-allocate.

    Returns: number of rows updated (0 = no pending row found, 1 = updated).
    """
    conn = _connect()
    cur = conn.execute(
        """
        UPDATE candidate_signals
        SET was_approved = ?, rejection_reason = ?
        WHERE id = (
            SELECT id FROM candidate_signals
            WHERE market_title = ? AND outcome = ?
              AND rejection_reason = 'pending_allocation'
            ORDER BY id DESC LIMIT 1
        )
        """,
        (1 if was_approved else 0, rejection_reason, market_title, outcome),
    )
    n = cur.rowcount
    conn.commit()
    conn.close()
    return int(n or 0)


def get_candidate_signals(days: int = 30) -> list[dict]:
    """Fetch candidate signals for backtest replay."""
    conn = _connect()
    rows = conn.execute("""
        SELECT * FROM candidate_signals
        WHERE timestamp >= datetime('now', ?)
        ORDER BY timestamp
    """, (f"-{days} days",)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_executions_since(days: int) -> list[dict]:
    conn = _connect()
    rows = conn.execute("""
        SELECT e.*, t.username FROM executions e
        JOIN traders t ON e.trader_address = t.address
        WHERE e.timestamp >= datetime('now', ?)
        ORDER BY e.timestamp DESC
    """, (f"-{days} days",)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_copy_events_since(days: int) -> list[dict]:
    conn = _connect()
    rows = conn.execute("""
        SELECT * FROM copy_events
        WHERE timestamp >= datetime('now', ?)
        ORDER BY timestamp DESC
    """, (f"-{days} days",)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# --- Simulation helpers ---

def _normalize_title(title: str) -> str:
    """Normalize market title for consistent dedup.

    Steps: NFKC unicode normalization, replace em-dashes/smart quotes,
    lowercase, strip, collapse whitespace, cap at 200 chars.
    """
    t = str(title)
    # Unicode NFKC normalization (handles ligatures, compatibility chars, etc.)
    t = unicodedata.normalize("NFKC", t)
    # Replace em-dashes and en-dashes with regular hyphens
    t = t.replace("\u2014", "-").replace("\u2013", "-")
    # Replace smart quotes with regular quotes
    t = t.replace("\u2018", "'").replace("\u2019", "'")   # single smart quotes
    t = t.replace("\u201c", '"').replace("\u201d", '"')    # double smart quotes
    return " ".join(t.lower().strip().split())[:200]


def _normalize_outcome(outcome: str) -> str:
    """Normalize outcome to lowercase stripped string for consistent matching."""
    return str(outcome).lower().strip()


_RECENT_TRADE_TIMES: dict = {}  # IB-031: per-strategy rate-cap tracking (in-process)
# WAVE 6 FIX 2026-04-26 (Round-6 council finding #1): the Knight Capital
# rate-cap firewall had a race. Read-modify-write of `times` list happened
# without a lock — two threads could both pass `len(times) >= 30` when count
# was 29, allowing 31 trades through. The LAST line of defense had a hole.
import threading as _firewall_threading
_RECENT_TRADE_TIMES_LOCK = _firewall_threading.Lock()
_FIREWALL_RATE_WINDOW_SEC = 60   # rolling window for rate-cap
_FIREWALL_RATE_MAX_PER_WINDOW = 30  # per-strategy max trades per 60s (Knight Capital prevention)


def _in_line_firewall(market_title: str, outcome: str, side: str, price: float,
                      shares: float, cost: float, source: str) -> tuple:
    """IB-031 (Council R2 / Council Operational R1) — in-line pre-trade firewall.

    Refuses obviously-bad trades AT the trade-write boundary, even if upstream
    filters (capital_allocator, trade_selector, strategy logic) have failed.

    Returns: (allowed: bool, reason: str). Caller must respect a False return.

    Knight Capital lesson: out-of-band probes (like c23_order_rate_throttle)
    are insufficient — the firewall must run synchronously at the order-write
    boundary so a runaway strategy cannot bypass it.

    Checks (fail-closed on any violation):
    1. Sanity: price ∈ [0, 1], shares > 0, cost > 0
    2. Notional cap: cost <= MAX_COST_PER_TRADE
    3. Per-market cap: cost <= MAX_PER_MARKET_USD (defensive — already enforced
       elsewhere, but caught here too)
    4. Rate cap: per-strategy rate <= _FIREWALL_RATE_MAX_PER_WINDOW per 60s
    5. Open-position cap: total open trades < MAX_OPEN_POSITIONS unless
       DRY_RUN_UNLIMITED=True
    """
    import logging as _logging
    log = _logging.getLogger("polybot.db.firewall")

    # WAVE 5 HARDENING 2026-04-25 (council CRITICAL-2):
    # Wrap the ENTIRE firewall body in fail-CLOSED outer try/except. Previously only
    # check 5 (open-position cap) failed-closed; checks 1-4 had no exception handling
    # and the config-import fallback was fail-OPEN ("firewall_skipped_config_unavailable").
    # Knight Capital lesson: ANY infra-level failure in the firewall must reject the
    # trade. Better to silently zero-trade and trip the c30 firewall-error alarm than
    # to silently let trades through.
    try:
        try:
            import config as _config
        except Exception as e:
            # Config-import failure is fail-CLOSED (was fail-OPEN before Wave 5).
            return False, f"firewall_infra_error: config_import_failed: {e}"

        # 1. Sanity
        if not (0.0 <= price <= 1.0):
            return False, f"price out of range [0,1]: {price}"
        if shares <= 0:
            return False, f"non-positive shares: {shares}"
        if cost <= 0:
            return False, f"non-positive cost: {cost}"

        # 2. Notional cap
        max_cost = float(getattr(_config, "MAX_COST_PER_TRADE", 50.0))
        if cost > max_cost:
            return False, f"cost ${cost:.2f} exceeds MAX_COST_PER_TRADE=${max_cost}"

        # 3. Per-market cap
        max_per_market = float(getattr(_config, "MAX_PER_MARKET_USD", 5.0))
        if cost > max_per_market * 10:  # firewall is the hard outer-bound (10x soft cap)
            return False, f"cost ${cost:.2f} exceeds 10x MAX_PER_MARKET_USD"

        # 4. Rate cap (Wave 6 Round-6 #1: now lock-guarded for thread safety)
        import time as _time
        now_ts = _time.time()
        key = source or "_unknown"
        with _RECENT_TRADE_TIMES_LOCK:
            times = _RECENT_TRADE_TIMES.setdefault(key, [])
            # Trim window
            cutoff = now_ts - _FIREWALL_RATE_WINDOW_SEC
            times[:] = [t for t in times if t >= cutoff]
            if len(times) >= _FIREWALL_RATE_MAX_PER_WINDOW:
                return False, (
                    f"rate cap: {len(times)} trades from '{key}' in last "
                    f"{_FIREWALL_RATE_WINDOW_SEC}s (max {_FIREWALL_RATE_MAX_PER_WINDOW})"
                )
            times.append(now_ts)

        # 4b. WAVE 5 FIX (council First-Principles Obs 7): same-market opposing-sides
        # pre-trade check. With $25 bankroll, two opposing $25 positions in the same
        # market = ~50% drawdown overnight. Reject unless explicitly arb-paired.
        # Cheap query backed by idx_simtrades_resolved_ts.
        try:
            conn_chk = _connect()
            opposite = "NO" if (outcome or "").upper() == "YES" else "YES"
            n_opposing = conn_chk.execute(
                "SELECT COUNT(*) FROM simulated_trades "
                "WHERE resolved=0 AND market_title=? AND UPPER(outcome)=?",
                (market_title, opposite),
            ).fetchone()[0]
            if n_opposing > 0 and (source or "") not in ("corr_arb", "arb"):
                return False, (
                    f"opposing-side cap: {n_opposing} open {opposite} positions in "
                    f"same market (source='{source}' not in arb whitelist)"
                )
        except Exception as e:
            # Let this fall through to outer except (fail-CLOSED).
            raise

        # 5. Open-position cap (skip if DRY_RUN_UNLIMITED)
        if not getattr(_config, "DRY_RUN_UNLIMITED", False):
            max_open = int(getattr(_config, "MAX_OPEN_POSITIONS", 5))
            try:
                conn_chk = _connect()
                n_open = conn_chk.execute(
                    "SELECT COUNT(*) FROM simulated_trades WHERE resolved=0"
                ).fetchone()[0]
                if n_open >= max_open:
                    return False, f"open-position cap: {n_open} >= MAX_OPEN_POSITIONS={max_open}"
            except Exception as e:
                # ULTRA-REVIEW FIREWALL AUDIT 2026-04-25: was `except: pass` (fail-OPEN
                # silently on DB lock / corruption). Knight Capital lesson: firewall is
                # the LAST line of defense — must fail-CLOSED on infra error.
                return False, f"firewall_infra_error: open_pos_query: {e}"

        return True, "ok"
    except Exception as e:
        # OUTER fail-CLOSED guard. Any unexpected throw → reject trade + log.
        log.error("[FIREWALL INFRA ERROR] outer-catch: %s (strategy=%s market='%s')",
                  e, source, market_title[:60])
        # WAVE 6 FIX 2026-04-26 (council finding #5): notify on firewall infra
        # error so operator knows trades are being rejected on plumbing, not
        # business rules. Dedupe-keyed by error class (not full message) so
        # 100 errors of the same class fire only one notification per 10min.
        try:
            from brain.notify import notify
            notify(
                title="Polybot FIREWALL INFRA ERROR",
                message=f"{type(e).__name__}: {str(e)[:80]} (strategy={source})",
                sound="Basso",
                dedup_sec=600,
                dedup_key=f"firewall_infra|{type(e).__name__}|{source}",
            )
        except Exception:
            pass
        return False, f"firewall_infra_error: outer: {e}"


def log_simulated_trade(trader_address: str, market_title: str, outcome: str,
                        side: str, price: float, shares: float, cost: float,
                        confidence: float = 0.0, source: str = "copy",
                        kelly_fraction: float = None, ev: float = None,
                        end_date: str = None, is_learning_only: bool = False,
                        swarm_votes: str = None, is_live_eligible: int = None,
                        one_per_market: bool = False,
                        token_id: str = None, expected_price: float = None,
                        book_fetcher=None):
    """WAVE 10 SHIP-9 (2026-04-26 post-Marathon audit Dry-run P0): new optional
    kwargs `token_id`, `expected_price`, `book_fetcher` enable orderbook-walk
    fill simulation. When `token_id` is provided, fill_simulator.simulate_fill
    is invoked AFTER the INSERT to populate fill_price / slippage_bps / fill_pct
    columns. `expected_price` is the mid the caller used for sizing (defaults
    to `price` if absent). `book_fetcher` lets callers inject a mock book in
    tests. ALL kwargs are optional → existing call sites unchanged. ANY error
    in the simulator falls back to mid-price (NULL fill columns). Knight
    Capital invariant: a fill-simulator fault cannot block trade recording.
    """
    # ---- IB-031: in-line pre-trade firewall (Knight Capital prevention) ----
    _allowed, _reason = _in_line_firewall(
        market_title, outcome, side, price, shares, cost, source
    )
    if not _allowed:
        # ULTRA-REVIEW LOGGING AUDIT 2026-04-25: canonical [FIREWALL REJECT] event.
        # WARNING level so it surfaces alongside real issues; structured for grep.
        import logging as _logging
        _logging.getLogger("polybot.db").warning(
            "[FIREWALL REJECT] reason='%s' strategy=%s market='%s' outcome=%s "
            "price=%s shares=%s cost=$%s",
            _reason, source, market_title[:60], outcome, price, shares, cost,
        )
        return None

    # ---- Phase 1 enforcement: refuse trade writes when halted ----
    # Late import to avoid circular deps (brain -> ... -> db should not cycle).
    # Any halt (strategy-scoped for `source`, or any global halt) aborts the write.
    # Returns None to match the normal success path (function does not return a rowid);
    # callers already tolerate no-op / exception paths here.
    try:
        import logging as _logging
        from brain.halt_state import HaltState
        _halt = HaltState()
        strategy_id = source or ""
        # R4 (2026-08-04, CONFIRMED by running): this read `scope="global"`, which NOTHING
        # in production writes — the 2026-04-25 BUG-1 fix renamed the WRITER "global" ->
        # "system" (review/cli.py) and this READER was never migrated, silently killing the
        # firewall for exactly the halts it exists to stop (c29 daily-loss kill, c14 clock
        # drift, c27 deployment inconsistency — all halt-severity with no affected_strategies,
        # therefore scope="system"). Measured: 6 of 6 strategies wrote trades under an active
        # system-wide halt. db.py was the ONLY remaining reader of "global" in the repo.
        # Now SCOPE-AGNOSTIC for non-strategy halts, matching execution/health_check.py's
        # active_halts() convention: any halt that is not strategy-scoped stops every write.
        # Naming a specific scope is what broke this once already.
        _wide = [h for h in _halt.active_halts()
                 if (h.get("scope") or "") not in ("strategy",)]
        if _halt.is_halted(scope="strategy", strategy=strategy_id) or _wide:
            _logging.getLogger("polybot.db").warning(
                "REFUSING trade write — halted (strategy=%s): %s / %s",
                strategy_id, market_title, outcome,
            )
            return None
    except Exception as _halt_exc:
        # WAVE 5 HARDENING 2026-04-25 (council Knight Capital Rank 4 + Op/Risk):
        # Was fail-OPEN. Knight Capital lesson: halt-DB unreachable means we cannot
        # verify a strategy is allowed to trade — refuse the write. Better to silently
        # zero-trade and trip the operator's "why are no trades firing?" alarm than
        # to let trades through during halt-state infrastructure failure.
        import logging as _logging
        _logging.getLogger("polybot.db").error(
            "halt_state check failed, fail-CLOSED (refusing trade): %s", _halt_exc,
        )
        return None
    # Normalize for consistent dedup — prevents whitespace/case mismatches causing doubles
    market_title = _normalize_title(market_title)
    outcome = _normalize_outcome(outcome)

    # One-per-market dedup guard: skip insertion if we already have an unresolved
    # open trade for this market+outcome+source. Used by corr_arb and btc strategies
    # to prevent re-entering the same position on every cycle.
    if one_per_market:
        conn_chk = _connect()
        existing = conn_chk.execute(
            "SELECT COUNT(*) FROM simulated_trades "
            "WHERE market_title=? AND outcome=? AND source=? AND resolved=0",
            (market_title, outcome, source),
        ).fetchone()[0]
        if existing > 0:
            import logging as _log2
            _log2.getLogger("polybot.db").debug(
                "one_per_market skip: already have open %s/%s (%s)", source, market_title[:40], outcome
            )
            return None

    # Auto-compute is_live_eligible if caller didn't pass it explicitly.
    # Rule: ev>0 AND kelly_fraction>0 => eligible for calibration curve.
    if is_live_eligible is None:
        try:
            is_live_eligible = 1 if (ev is not None and ev > 0 and
                                     kelly_fraction is not None and kelly_fraction > 0) else 0
        except Exception:
            is_live_eligible = 0
    conn = _connect()
    now = datetime.now(timezone.utc).isoformat()
    # Ensure trader_address exists in traders table (FK constraint).
    # Real wallet addresses (insider copies) may not have been upserted yet.
    conn.execute(
        "INSERT OR IGNORE INTO traders (address, username, first_seen) VALUES (?, ?, ?)",
        (trader_address, trader_address, now),
    )
    # WAVE 6 FIX 2026-04-26 (Round-6 council finding #2): catch IntegrityError
    # from the new uniq_open_position partial unique index. If a concurrent
    # writer beat us to inserting the same (market, outcome) open position,
    # the second INSERT raises IntegrityError → log + return None gracefully
    # instead of propagating to caller. DB-level dedup covers what app-level
    # 4-layer dedup couldn't (TOCTOU between is_already_simulated and INSERT).
    try:
        cur = conn.execute("""
            INSERT INTO simulated_trades
                (trader_address, market_title, outcome, side, price, shares_simulated,
                 cost_simulated, timestamp, confidence, source, kelly_fraction, ev, end_date,
                 is_learning_only, swarm_votes, is_live_eligible)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (trader_address, market_title, outcome, side, price, shares, cost, now,
              confidence, source, kelly_fraction, ev, end_date, int(is_learning_only),
              swarm_votes, int(is_live_eligible)))
    except sqlite3.IntegrityError as _ie:
        import logging as _logging
        _logging.getLogger("polybot.db").info(
            "[DEDUP-DB] race-loser on (%s, %s) — concurrent open position exists: %s",
            market_title[:50], outcome, _ie,
        )
        conn.close()
        return None

    # WAVE 10 SHIP-9 (2026-04-26 post-Marathon audit Dry-run P0 #1): wire
    # fill_simulator. Was orphan code per audit ("zero production callers").
    # When caller passed token_id, walk live orderbook for the requested
    # notional → write actual fill_price, slippage_bps, fill_pct. NEVER
    # raises — any error logs at debug and falls back to mid-price.
    _rowid = cur.lastrowid
    if token_id and _rowid:
        try:
            from fill_simulator import simulate_fill
            _fill = simulate_fill(
                token_id=token_id,
                side=side,
                notional_usd=float(cost),
                fetch_book=book_fetcher,
                fallback_price=float(expected_price if expected_price is not None else price),
            )
            conn.execute(
                "UPDATE simulated_trades SET fill_price=?, expected_price=?, "
                "slippage_bps=?, fill_pct=? WHERE id=?",
                (
                    float(_fill.fill_price) if _fill.is_filled else None,
                    float(_fill.expected_price),
                    float(_fill.slippage_bps),
                    float(_fill.fill_pct),
                    _rowid,
                ),
            )
        except Exception as _fs_exc:
            # WAVE 12 SHIP-4 (2026-05-16 council finding): promoted from DEBUG to
            # WARNING. This catch hides the entire SHIP-2 value (real orderbook
            # slippage). Before this fix, 0 of 214 production trades populated
            # slippage_bps despite the wire being in place — we had no
            # visibility into WHY. Same Rule-12 lesson that Wave 10 SHIP-6
            # already taught (silent fail = days of broken expensive code).
            # Now: every fill_simulator fallback is operator-visible + greppable.
            import logging as _logfs
            _logfs.getLogger("polybot.db").warning(
                "[FILL_SIM FALLBACK] strategy=%s market=%r outcome=%r "
                "token_id=%r — sim failed, recorded at mid-price: %s",
                source, (market_title or "")[:60], outcome, token_id, _fs_exc,
            )

    # Also log as approved candidate signal for backtest replay
    conn.execute("""
        INSERT INTO candidate_signals
        (timestamp, market_title, outcome, price, confidence, ev, kelly_fraction,
         source, trader_address, market_volume, end_date, was_approved, rejection_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, '')
    """, (now, market_title, outcome, price, confidence, ev, kelly_fraction,
          source, trader_address, 0, end_date))
    conn.commit()
    conn.close()

    # ULTRA-REVIEW LOGGING AUDIT 2026-04-25: canonical [TRADE FIRED] event.
    # Single grep target for probes (esp. C03 silent_data_failure) + post-trade
    # analytics. One line per simulated_trades insert, structured for parsing.
    import logging as _log_fired
    _log_fired.getLogger("polybot").info(
        "[TRADE FIRED] strategy=%s market='%s' outcome=%s side=%s "
        "price=%.4f shares=%.2f cost=$%.2f conf=%.3f ev=%s kelly=%s",
        source, market_title[:60], outcome, side,
        float(price), float(shares), float(cost),
        float(confidence) if confidence is not None else 0.0,
        f"{ev:.4f}" if ev is not None else "n/a",
        f"{kelly_fraction:.4f}" if kelly_fraction is not None else "n/a",
    )


def log_shadow_trade(
    market_title: str,
    outcome: str,
    side: str,
    source: str,
    *,
    shadow_price: Optional[float] = None,
    shadow_shares: Optional[float] = None,
    shadow_cost: Optional[float] = None,
    shadow_pnl: Optional[float] = None,
    shadow_slippage_bps: Optional[float] = None,
    shadow_fill_price: Optional[float] = None,
    live_price: Optional[float] = None,
    live_shares: Optional[float] = None,
    live_cost: Optional[float] = None,
    live_pnl: Optional[float] = None,
    live_slippage_bps: Optional[float] = None,
    live_fill_price: Optional[float] = None,
    notes: str = "",
) -> Optional[int]:
    """WAVE 12 SHIP-3 (2026-05-16): persist a (shadow, live) trade pair for c38.

    Pre-go-live invariant. When DRY_RUN flips False (after explicit Arjun
    sign-off), every live trade should ALSO run the same logic synthetically
    against the same market data — the shadow leg. Both legs land here so
    `c38_shadow_ab_drift` can compute median |live_pnl - shadow_pnl| / |live_pnl|
    nightly and flag >20% divergence (signals fill-model bug or real
    slippage/latency delta).

    Currently UNCALLED — execution_gateway (Phase 5) is the planned caller.
    Until then this is pre-go-live infrastructure: schema exists, writer exists,
    c38 probe exits its "table not present" path on next probe sweep.

    Returns the inserted rowid, or None on failure. Failure-tolerant: catches
    all exceptions and logs at ERROR (Rule 12 fail visibly) — never raises.

    Args expose paired sim/live fields so a single call records both legs of
    one trade decision; either side may be None during partial pipelines
    (e.g. live-only post-mortem reconstruction, shadow-only A/B research).
    """
    try:
        conn = _connect()
        cur = conn.execute(
            """
            INSERT INTO shadow_trades
                (market_title, outcome, side, source,
                 shadow_price, shadow_shares, shadow_cost, shadow_pnl,
                 shadow_slippage_bps, shadow_fill_price,
                 live_price, live_shares, live_cost, live_pnl,
                 live_slippage_bps, live_fill_price,
                 notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                market_title, outcome, side, source,
                shadow_price, shadow_shares, shadow_cost, shadow_pnl,
                shadow_slippage_bps, shadow_fill_price,
                live_price, live_shares, live_cost, live_pnl,
                live_slippage_bps, live_fill_price,
                notes,
            ),
        )
        rowid = cur.lastrowid
        conn.commit()
        conn.close()
        return int(rowid) if rowid else None
    except Exception as e:
        import logging as _logging
        _logging.getLogger("polybot.db").error(
            "log_shadow_trade failed for %s/%s: %s", market_title[:50], outcome, e,
        )
        return None


def get_unresolved_trades(limit: int = 500) -> list[dict]:
    """Return open simulated trades not yet marked resolved.

    WAVE 8 FIX 2026-04-26 (Round-9 council #1): split into oldest-half +
    newest-half UNION. Was `ORDER BY timestamp DESC LIMIT 500` — with 670
    open trades, the OLDEST 170 (likeliest already settled on Polymarket)
    were permanently invisible to the resolver. Open-position cap fills
    with zombie trades, strategies stop firing after ~6 weeks. Round-robin
    ensures the long tail gets a chance to resolve.
    """
    conn = _connect()
    half = limit // 2
    rows_old = conn.execute("""
        SELECT id, market_title, outcome, price, shares_simulated, cost_simulated,
               source, end_date,
               COALESCE(end_date_stale, 0) AS end_date_stale
        FROM simulated_trades
        WHERE resolved = 0
        ORDER BY timestamp ASC
        LIMIT ?
    """, (half,)).fetchall()
    rows_new = conn.execute("""
        SELECT id, market_title, outcome, price, shares_simulated, cost_simulated,
               source, end_date,
               COALESCE(end_date_stale, 0) AS end_date_stale
        FROM simulated_trades
        WHERE resolved = 0
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit - half,)).fetchall()
    conn.close()
    # Dedup: same trade may appear in both halves if total < limit
    seen = set()
    out = []
    for r in list(rows_old) + list(rows_new):
        d = dict(r)
        if d["id"] in seen:
            continue
        seen.add(d["id"])
        out.append(d)
    return out


def is_already_simulated(market_title: str, outcome: str) -> bool:
    """True if we already have an open simulated position on this market+outcome."""
    market_title = _normalize_title(market_title)
    outcome = _normalize_outcome(outcome)
    conn = _connect()
    row = conn.execute("""
        SELECT 1 FROM simulated_trades
        WHERE LOWER(TRIM(market_title))=? AND LOWER(TRIM(outcome))=? AND resolved=0
        LIMIT 1
    """, (market_title, outcome)).fetchone()
    conn.close()
    return row is not None


def resolve_trade(trade_id: int, won: bool, pnl: float, resolved_outcome=None):
    """Mark a simulated trade as resolved. Computes Brier score for calibration tracking.

    Brier score = (confidence - outcome)²
    outcome = 1.0 if won else 0.0
    Lower Brier = better calibration. Random guesser = 0.25.

    ULTRA-REVIEW BRIER+RESOLUTION AUDIT 2026-04-25: now writes resolved_outcome
    (was 100% NULL across all 416 resolved trades — broke any analytic that
    needs Yes/No truth, and forced confidence_trend.py to use sim_pnl>0 as
    a fragile win-proxy). Caller passes the truth label string ("Yes"/"No"/etc.).
    Backward compatible: existing callers without the kwarg still work; trade
    is resolved with NULL outcome (matches legacy state, future-fixable).
    """
    conn = _connect()
    now = datetime.now(timezone.utc).isoformat()
    trade = conn.execute(
        "SELECT confidence FROM simulated_trades WHERE id=?", (trade_id,)
    ).fetchone()

    brier = None
    if trade and trade["confidence"] is not None:
        outcome_val = 1.0 if won else 0.0
        brier = round((float(trade["confidence"]) - outcome_val) ** 2, 6)

    conn.execute("""
        UPDATE simulated_trades
        SET resolved=1, sim_pnl=?, resolved_at=?, brier_score=?, resolved_outcome=?
        WHERE id=?
    """, (pnl, now, brier, resolved_outcome, trade_id))
    conn.commit()
    conn.close()


def mark_end_date_stale(trade_id: int):
    """NULL out end_date and set end_date_stale=1 for a trade whose Polymarket
    endDate was stale placeholder data but the market is still open (closed=false).

    Called by resolution_checker when it confirms the market is still live.
    Resolution will resume normally once the market actually closes.
    """
    conn = _connect()
    conn.execute(
        "UPDATE simulated_trades SET end_date=NULL, end_date_stale=1 WHERE id=?",
        (trade_id,)
    )
    conn.commit()
    conn.close()


def get_calibration_data(min_resolved: int = 10) -> list[dict]:
    """Return calibration bucket data: confidence range → actual win rate.

    Used by researcher.py to detect if confidence scores are accurate.
    Only includes is_learning_only=0 trades (live-quality trades only).

    Returns list of dicts with keys: confidence_bucket, count, win_rate, avg_brier
    confidence_bucket format: "0.40-0.45" (5% buckets)
    """
    conn = _connect()
    rows = conn.execute("""
        SELECT
            PRINTF('%.2f-%.2f',
                ROUND(confidence / 0.05) * 0.05,
                ROUND(confidence / 0.05) * 0.05 + 0.05) as confidence_bucket,
            COUNT(*) as count,
            ROUND(1.0 * SUM(CASE WHEN sim_pnl > 0 THEN 1 ELSE 0 END) / COUNT(*), 4) as win_rate,
            AVG(brier_score) as avg_brier
        FROM simulated_trades
        WHERE resolved=1 AND confidence > 0 AND is_learning_only=0
        GROUP BY ROUND(confidence / 0.05)
        HAVING count >= ?
        ORDER BY confidence_bucket
    """, (min_resolved,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_resolved_trade_count(learning_only: bool = False) -> int:
    """Count resolved trades. Used to gate learning/research runs.

    Args:
        learning_only: if False (default), count only live-quality trades (is_learning_only=0).
                       if True, count ALL resolved trades including learning-only ones.
    """
    conn = _connect()
    if learning_only:
        # Count ALL resolved trades (both live and learning)
        row = conn.execute(
            "SELECT COUNT(*) as n FROM simulated_trades WHERE resolved=1"
        ).fetchone()
    else:
        # Count only live-quality trades (exclude learning-only)
        row = conn.execute(
            "SELECT COUNT(*) as n FROM simulated_trades WHERE resolved=1 AND is_learning_only=0"
        ).fetchone()
    conn.close()
    return int(row["n"]) if row else 0


def get_today_sim_loss() -> float:
    """Return total sim losses today (negative pnl sum, as a positive number)."""
    conn = _connect()
    row = conn.execute("""
        SELECT COALESCE(SUM(sim_pnl), 0) as total_pnl
        FROM simulated_trades
        WHERE date(timestamp) = date('now') AND sim_pnl < 0
    """).fetchone()
    conn.close()
    return abs(row["total_pnl"]) if row else 0.0


def get_total_resolved_pnl() -> float:
    """MARATHON SHIP-10 (2026-04-26 audit Dry-run Phase E): cumulative
    realized P&L across ALL resolved trades. Used by capital_allocator to
    compute effective_bankroll = TOTAL_BUDGET_USD + realized_pnl, so the
    Kelly path receives a drawdown signal in DRY_RUN that mirrors live
    (where wallet balance reflects realized P&L automatically).

    EXCLUDES Wave 6 zombie rows (sim_pnl=0 AND is_learning_only=1, the
    weather force-resolve-on-disable pattern) so paused-strategy zombies
    don't pollute the bankroll signal.
    """
    conn = _connect()
    row = conn.execute("""
        SELECT COALESCE(SUM(sim_pnl), 0) as total_pnl
        FROM simulated_trades
        WHERE resolved = 1
          AND NOT (sim_pnl = 0 AND is_learning_only = 1)
    """).fetchone()
    conn.close()
    return float(row["total_pnl"]) if row else 0.0


def get_simulated_trades(days: int = 30) -> list[dict]:
    conn = _connect()
    rows = conn.execute("""
        SELECT s.*, t.username FROM simulated_trades s
        JOIN traders t ON s.trader_address = t.address
        WHERE s.timestamp >= datetime('now', ?)
        ORDER BY s.timestamp DESC
    """, (f"-{days} days",)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_simulation_summary() -> dict:
    """Get high-level sim stats."""
    conn = _connect()
    row = conn.execute("""
        SELECT
            COUNT(*) as total_trades,
            SUM(cost_simulated) as total_cost,
            SUM(sim_pnl) as total_pnl,
            COUNT(DISTINCT trader_address) as unique_traders
        FROM simulated_trades
    """).fetchone()
    conn.close()
    return dict(row) if row else {}


# --- Insider wallet helpers ---

def flag_insider_wallet(address: str, confidence: float, signals: dict):
    """Insert or update an insider wallet flag."""
    conn = _connect()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO insider_wallets (address, confidence, signals, first_flagged)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(address) DO UPDATE SET
            confidence=excluded.confidence,
            signals=excluded.signals
    """, (address, confidence, json.dumps(signals), now))
    conn.commit()
    conn.close()


def get_insider_wallets() -> list[dict]:
    """Return all flagged insider wallets, highest confidence first."""
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM insider_wallets ORDER BY confidence DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def set_insider_copying(address: str, sub_id: str):
    """Mark an insider wallet as actively being copied."""
    conn = _connect()
    conn.execute(
        "UPDATE insider_wallets SET is_copying=1, copy_sub_id=? WHERE address=?",
        (sub_id, address)
    )
    conn.commit()
    conn.close()


def set_insider_stopped(address: str):
    """Mark an insider wallet as no longer copied."""
    conn = _connect()
    conn.execute(
        "UPDATE insider_wallets SET is_copying=0, copy_sub_id=NULL WHERE address=?",
        (address,)
    )
    conn.commit()
    conn.close()


# --- Phase 3: Capital allocator + strategy tracker helpers ---

def get_open_cost_by_strategy() -> dict:
    """Sum of cost_simulated for unresolved positions, grouped by source/strategy."""
    conn = _connect()
    rows = conn.execute("""
        SELECT source, COALESCE(SUM(cost_simulated), 0) as total
        FROM simulated_trades WHERE resolved=0
        GROUP BY source
    """).fetchall()
    conn.close()
    return {r["source"]: float(r["total"]) for r in rows}


def get_open_position_count() -> int:
    """Count of currently open (unresolved) simulated positions."""
    conn = _connect()
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM simulated_trades WHERE resolved=0 AND confidence > 0"
    ).fetchone()
    conn.close()
    return int(row["cnt"]) if row else 0


def get_open_bond_positions_by_market() -> dict:
    """Sum of cost_simulated for unresolved bond positions, grouped by market_title.

    Used by bond scanner diversification check to enforce BOND_MAX_POSITION_PCT
    (no more than 20% of bond budget in a single market).

    Returns:
        dict mapping market_title -> total cost deployed in that market.
    """
    conn = _connect()
    rows = conn.execute("""
        SELECT market_title, COALESCE(SUM(cost_simulated), 0) as total_cost
        FROM simulated_trades
        WHERE resolved=0 AND source='bond'
        GROUP BY market_title
    """).fetchall()
    conn.close()
    return {r["market_title"]: float(r["total_cost"]) for r in rows}


def get_trades_by_strategy_and_date(strategy: str, date: str) -> list:
    """Fetch all resolved trades for a strategy on a specific date (YYYY-MM-DD)."""
    conn = _connect()
    rows = conn.execute("""
        SELECT * FROM simulated_trades
        WHERE source=? AND date(timestamp)=? AND resolved=1
    """, (strategy, date)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_today_resolved_trades() -> list:
    """Return all trades that resolved today (by resolution date, not placement date).

    Falls back to timestamp if resolved_at is NULL (pre-migration trades).
    """
    conn = _connect()
    rows = conn.execute("""
        SELECT * FROM simulated_trades
        WHERE resolved=1
          AND date(COALESCE(resolved_at, timestamp))=date('now')
        ORDER BY COALESCE(resolved_at, timestamp) DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def upsert_strategy_day(strategy: str, date: str, trades_total: int,
                        trades_won: int, total_pnl: float,
                        avg_confidence: float, sharpe: float,
                        avg_brier: float = 0.0, max_drawdown: float = 0.0):
    """Insert or replace one strategy/day performance row."""
    conn = _connect()
    conn.execute("""
        INSERT INTO strategy_performance
            (strategy, date, trades_total, trades_won, total_pnl, avg_confidence, sharpe,
             avg_brier, max_drawdown)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(strategy, date) DO UPDATE SET
            trades_total=excluded.trades_total,
            trades_won=excluded.trades_won,
            total_pnl=excluded.total_pnl,
            avg_confidence=excluded.avg_confidence,
            sharpe=excluded.sharpe,
            avg_brier=excluded.avg_brier,
            max_drawdown=excluded.max_drawdown
    """, (strategy, date, trades_total, trades_won, total_pnl, avg_confidence, sharpe,
          avg_brier, max_drawdown))
    conn.commit()
    conn.close()


def get_strategy_performance() -> list:
    """Return all rows from strategy_performance ordered by strategy and date desc."""
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM strategy_performance ORDER BY strategy, date DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
