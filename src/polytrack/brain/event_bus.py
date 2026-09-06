"""Event type definitions + dispatcher interface.

Events produced by the rest of Polybot (review probes, risk module, config
watcher, external watchdog, schedulers) are delivered to the brain via
subclass instances of BrainEvent. TriggerMatcher consumes these.

Emitter integrations are STUBBED. This module defines:
  - BrainEvent dataclass (top-level envelope)
  - Per-event payload shapes referenced by the TOML
  - AbstractEventBus (subscribe + publish)
  - InMemoryEventBus (concrete stub for tests + local dev)

TODO(integration): actual emitters live in review/risk/config-watcher. Wire
them in once brain is running. See event_correlation_id schema in the TOML
[integration.event_emitters] section.
"""
from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional


log = logging.getLogger("polybot.brain.event_bus")


# ---- Known event types (derived from triggers.proposed.v5.toml) ----
# Kept as a simple module-level tuple so tests can assert coverage.

KNOWN_EVENT_TYPES: tuple[str, ...] = (
    # Config + code change
    "config.change",
    "code.change",
    # Risk
    "risk.drawdown",
    "risk.daily_loss",
    "risk.consecutive_losing_days",
    "risk.exposure",
    "risk.correlated_exposure",
    "risk.warmup_cap",
    # Bankroll
    "bankroll.source_check",
    # Execution
    "execution.failure",
    # Strategy
    "strategy.flat_pnl",
    "strategy.kill_criteria_met",
    "strategy.budget_zero_with_trades",
    "strategy.pipeline.gate",
    # Probes
    "review.probe.failed",
    "review.probe.passed",
    # Market / metrics
    "market.volatility",
    "metric.anomaly",
    "metric.anchored_deviation",
    "metric.rapid_change_burst",
    # Scheduled
    "scheduled.daily",
    "scheduled.morning_summary.failed",
    # Clock
    "clock.drift",
    # Brain-internal
    "brain.queue_overloaded",
    "brain.cost_budget_exceeded",
    "brain.pending_queue_overflow",
    "brain.startup",
    # Emitter health
    "emitter.silent",
    # External watchdog
    "external_watchdog.warmup_cap_emitter_silent",
    "external_watchdog.probe_runner_silent",
    "external_watchdog.brain_silent",
    # User
    "user.request_review",
    # Proposal / deploy
    "proposal.promotion.dry_run_to_live",
    "proposal.deploy_attempted",
    # Claude-main activity
    "claude_main.commit",
    # Signature health
    "signature.changed",
)


# ---- Base envelope ----


@dataclass
class BrainEvent:
    """Top-level event envelope delivered to the brain.

    Required fields per [integration.event_emitters] in TOML:
      event_type, event_correlation_id, timestamp_utc, source_emitter.
    The payload dict carries event-type-specific fields.
    """
    event_type: str
    source_emitter: str
    payload: dict[str, Any] = field(default_factory=dict)
    event_correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.event_type:
            raise ValueError("BrainEvent.event_type must be non-empty")
        if not self.source_emitter:
            raise ValueError("BrainEvent.source_emitter must be non-empty")

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "event_correlation_id": self.event_correlation_id,
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "source_emitter": self.source_emitter,
            "payload": dict(self.payload),
        }

    @classmethod
    def make(cls, event_type: str, source_emitter: str,
             payload: Optional[dict[str, Any]] = None,
             correlation_id: Optional[str] = None) -> "BrainEvent":
        """Helper for tests + local construction."""
        ev = cls(
            event_type=event_type,
            source_emitter=source_emitter,
            payload=dict(payload or {}),
        )
        if correlation_id is not None:
            ev.event_correlation_id = correlation_id
        return ev


# ---- Abstract bus ----


EventHandler = Callable[[BrainEvent], None]


class AbstractEventBus(ABC):
    """Dispatcher interface.

    Subscribers register a handler per event_type (or "*" for all).
    Publish is synchronous; handlers running off-thread are the handler's
    responsibility.
    """

    @abstractmethod
    def subscribe(self, event_type: str, handler: EventHandler) -> None: ...

    @abstractmethod
    def publish(self, event: BrainEvent) -> None: ...


class InMemoryEventBus(AbstractEventBus):
    """Synchronous in-process bus. Adequate for tests and single-process
    brain. Production may swap in a queue-backed version.
    """

    def __init__(self) -> None:
        self._subs: dict[str, list[EventHandler]] = {}

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self._subs.setdefault(event_type, []).append(handler)

    def publish(self, event: BrainEvent) -> None:
        for et in (event.event_type, "*"):
            for h in list(self._subs.get(et, [])):
                try:
                    h(event)
                except Exception:
                    log.exception(
                        "Handler for %s raised on event %s",
                        et, event.event_correlation_id,
                    )


# ---- Event-type specific payload helpers (optional, typed shortcuts) ----
# These mirror the `match = { ... }` fields in the TOML so emitters have a
# canonical place to look. Keeping them as thin wrappers lets tests be
# descriptive while letting the matcher still accept plain dicts.


@dataclass
class ProbeFailedPayload:
    probe_severity: str  # "halt" | "warn" | "info"
    probe_scope: str  # "system_wide" | "single_strategy"
    probe_name: str = ""
    strategy: str = ""

    def to_dict(self) -> dict:
        return {
            "probe_severity": self.probe_severity,
            "probe_scope": self.probe_scope,
            "probe_name": self.probe_name,
            "strategy": self.strategy,
        }


@dataclass
class DrawdownPayload:
    scope: str  # "daily" | "weekly" | "strategy" | "global"
    strategy: str = ""
    measured_pct: float = 0.0
    threshold_pct: float = 0.0

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "strategy": self.strategy,
            "measured_pct": self.measured_pct,
            "threshold_pct": self.threshold_pct,
        }


@dataclass
class ConfigChangePayload:
    keys: list[str] = field(default_factory=list)
    is_new_key: bool = False
    sanity_bound_violated: bool = False
    value_magnitude_change_ratio: float = 1.0
    sum_violation_key: str = ""
    committer_email: str = ""
    diff_text: str = ""

    def to_dict(self) -> dict:
        return {
            "keys": list(self.keys),
            "is_new_key": self.is_new_key,
            "sanity_bound_violated": self.sanity_bound_violated,
            "value_magnitude_change_ratio": self.value_magnitude_change_ratio,
            "sum_violation_key": self.sum_violation_key,
            "committer_email": self.committer_email,
            "diff_text": self.diff_text,
        }


# ---- MARATHON SHIP-5 (2026-04-26 audit Patch C): Singleton event bus + helpers
# Closes the false-safety gap where 55 of 57 signed triggers were "wireless"
# because BrainEvent.publish() had zero call sites. Per Round-10 council finding
# and the Wave 8 startup banner, this was the single highest-leverage gap in the
# review system. Now: a process-wide singleton bus + thin emit helpers callable
# from anywhere in polybot. Each emitter site stays a single line.

_GLOBAL_BUS: Optional[AbstractEventBus] = None


def get_bus() -> AbstractEventBus:
    """Return the process-wide singleton event bus. Lazily constructed."""
    global _GLOBAL_BUS
    if _GLOBAL_BUS is None:
        _GLOBAL_BUS = InMemoryEventBus()
    return _GLOBAL_BUS


def set_bus(bus: AbstractEventBus) -> None:
    """Replace the singleton (e.g. for tests). Use sparingly."""
    global _GLOBAL_BUS
    _GLOBAL_BUS = bus


def publish_event(event_type: str, source_emitter: str,
                  payload: Optional[dict] = None) -> None:
    """One-line emit helper — every emitter site uses this.

    Failure-tolerant: if no subscribers, no handlers, or any handler raises,
    we DON'T propagate exceptions back to trading code. The bus is purely
    additive — never block trade decisions on bus failures.
    """
    try:
        ev = BrainEvent.make(event_type=event_type,
                             source_emitter=source_emitter,
                             payload=payload or {})
        get_bus().publish(ev)
    except Exception as e:
        log.warning("publish_event(%s) failed: %s — dropped", event_type, e)


# Specialized emit helpers for the 5 highest-priority trigger paths.
# Each maps cleanly to a payload schema and a documented trigger in
# data/brain/triggers.signed.toml.

def emit_probe_failed(probe_name: str, probe_severity: str,
                      probe_scope: str, strategy: str = "") -> None:
    """Emit `review.probe.failed`. Source: BrainRunner._apply_action."""
    publish_event(
        event_type="review.probe.failed",
        source_emitter="review.brain_runner",
        payload=ProbeFailedPayload(
            probe_severity=probe_severity,
            probe_scope=probe_scope,
            probe_name=probe_name,
            strategy=strategy,
        ).to_dict(),
    )


def emit_probe_passed(probe_name: str, strategy: str = "") -> None:
    """Emit `review.probe.passed`. Source: BrainRunner._apply_action."""
    publish_event(
        event_type="review.probe.passed",
        source_emitter="review.brain_runner",
        payload={"probe_name": probe_name, "strategy": strategy},
    )


def emit_daily_loss(measured_pct: float, threshold_pct: float,
                    measured_usd: float = 0.0,
                    threshold_usd: float = 0.0) -> None:
    """Emit `risk.daily_loss`. Source: risk_guard.check_safe_to_trade()."""
    publish_event(
        event_type="risk.daily_loss",
        source_emitter="risk_guard",
        payload={
            "scope": "daily",
            "measured_pct": float(measured_pct),
            "threshold_pct": float(threshold_pct),
            "measured_usd": float(measured_usd),
            "threshold_usd": float(threshold_usd),
        },
    )


def emit_drawdown(scope: str, measured_pct: float, threshold_pct: float,
                  strategy: str = "") -> None:
    """Emit `risk.drawdown`. Source: risk_guard / drawdown computation."""
    publish_event(
        event_type="risk.drawdown",
        source_emitter="risk_guard",
        payload=DrawdownPayload(
            scope=scope, strategy=strategy,
            measured_pct=measured_pct, threshold_pct=threshold_pct,
        ).to_dict(),
    )


def emit_consecutive_losing_days(strategy: str, days: int) -> None:
    """Emit `risk.consecutive_losing_days`. Source: nightly risk roll-up."""
    publish_event(
        event_type="risk.consecutive_losing_days",
        source_emitter="risk_guard",
        payload={"strategy": strategy, "days": int(days)},
    )


def emit_config_change(keys: list, is_new_key: bool = False,
                       sanity_bound_violated: bool = False,
                       value_magnitude_change_ratio: float = 1.0,
                       sum_violation_key: str = "",
                       committer_email: str = "",
                       diff_text: str = "") -> None:
    """Emit `config.change`. Source: champion_overrides.save_overrides_atomic
    + main.py preflight on DRY_RUN flip.
    """
    publish_event(
        event_type="config.change",
        source_emitter="config_watcher",
        payload=ConfigChangePayload(
            keys=list(keys),
            is_new_key=is_new_key,
            sanity_bound_violated=sanity_bound_violated,
            value_magnitude_change_ratio=value_magnitude_change_ratio,
            sum_violation_key=sum_violation_key,
            committer_email=committer_email,
            diff_text=diff_text,
        ).to_dict(),
    )
