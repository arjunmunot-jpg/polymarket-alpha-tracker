"""MoonDev-style Risk Agent: hard safety checks before every trade.

Rules (from MoonDev spec):
  1. Max daily loss — halt all new trades if sim losses exceed MAX_DAILY_LOSS_USD today
  2. Min wallet balance — never trade if balance is below MIN_WALLET_BALANCE_USD
  3. Total budget cap — never exceed TOTAL_BUDGET_USD in total open positions

These run in both DRY_RUN and live mode.
"""

from __future__ import annotations

import logging
from typing import Optional

import config
import db

log = logging.getLogger("polybot.risk_guard")


def check_safe_to_trade() -> tuple[bool, str]:
    """Run all safety checks. Returns (safe, reason_if_not_safe).

    Returns:
        (True, "OK") if all checks pass
        (False, "<reason>") if any check fails — caller should skip the trade
    """
    # --- 1. Daily loss limit ---
    today_loss = db.get_today_sim_loss()
    if today_loss >= config.MAX_DAILY_LOSS_USD:
        msg = f"Daily loss limit hit: ${today_loss:.2f} >= ${config.MAX_DAILY_LOSS_USD:.2f}"
        log.warning("RISK GUARD: %s — halting new trades", msg)
        # WAVE 6 FIX 2026-04-26 (council finding #5): operator notification.
        # Daily-loss trip is c29-equivalent capital-preservation event —
        # operator MUST know. Was previously silent except for log line.
        try:
            from brain.notify import notify
            notify(
                title="Polybot DAILY LOSS LIMIT",
                message=msg,
                sound="Basso",
                dedup_sec=600,
                dedup_key="risk_guard|daily_loss_trip",
            )
        except Exception:
            pass
        # MARATHON SHIP-5 (2026-04-26 audit): publish event so the signed
        # `daily_loss_halt` trigger fires off the bus. Was bypassed before
        # (BrainEvent.publish had zero call sites; 55/57 triggers dead).
        try:
            from brain.event_bus import emit_daily_loss
            budget = float(getattr(config, "TOTAL_BUDGET_USD", 1.0)) or 1.0
            emit_daily_loss(
                measured_pct=today_loss / budget,
                threshold_pct=config.MAX_DAILY_LOSS_USD / budget,
                measured_usd=today_loss,
                threshold_usd=config.MAX_DAILY_LOSS_USD,
            )
        except Exception:
            pass
        return False, msg

    # --- 2. Total budget cap (open positions only, not all-time spend) ---
    total_open_cost = _get_open_position_cost()
    if total_open_cost >= config.TOTAL_BUDGET_USD:
        msg = f"Budget cap reached: ${total_open_cost:.2f} open >= ${config.TOTAL_BUDGET_USD:.2f}"
        log.warning("RISK GUARD: %s — halting new trades", msg)
        return False, msg

    # --- 3. Balance check (live mode only) ---
    if not config.DRY_RUN:
        balance = _get_live_balance()
        if balance is not None and balance < config.MIN_WALLET_BALANCE_USD:
            msg = f"Wallet balance too low: ${balance:.2f} < ${config.MIN_WALLET_BALANCE_USD:.2f}"
            log.warning("RISK GUARD: %s — halting new trades", msg)
            return False, msg

    return True, "OK"


def _get_open_position_cost() -> float:
    """Sum of cost_simulated for unresolved (open) simulated positions."""
    try:
        conn = db._connect()
        row = conn.execute(
            "SELECT COALESCE(SUM(cost_simulated), 0) as total FROM simulated_trades WHERE resolved=0"
        ).fetchone()
        conn.close()
        return float(row["total"]) if row else 0.0
    except Exception:
        return 0.0


def _get_live_balance() -> Optional[float]:
    """Fetch USDC balance from Bullpen CLI. Returns None on failure."""
    import subprocess, json
    try:
        result = subprocess.run(
            [config.BULLPEN_BIN, "portfolio", "balances", "--output", "json"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            # Look for USDC entry in balances list
            if isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict):
                        symbol = (entry.get("symbol") or entry.get("asset") or "").upper()
                        if "USDC" in symbol:
                            return float(entry.get("balance") or entry.get("amount") or 0)
            elif isinstance(data, dict):
                return float(data.get("usdc") or data.get("balance") or 0)
    except Exception as e:
        log.debug("Balance check failed: %s", e)
    return None


def get_status() -> dict:
    """Return current risk metrics for dashboard display."""
    today_loss = db.get_today_sim_loss()
    total_open = _get_open_position_cost()
    safe, reason = check_safe_to_trade()
    return {
        "safe": safe,
        "reason": reason,
        "today_loss_usd": round(today_loss, 2),
        "total_open_cost_usd": round(total_open, 2),
        "max_daily_loss_usd": config.MAX_DAILY_LOSS_USD,
        "total_budget_usd": config.TOTAL_BUDGET_USD,
    }
