"""Match events against loaded trigger specs.

Given a BrainEvent + TriggersConfig, returns the set of matching triggers
with their severities. Implements:
  - event_type exact match
  - `match.keys` membership for config.change events
  - pattern match (glob-style) for protected_patterns
  - regex match with word boundaries (probability_math_catchall)
  - conditions list matching (any_of)
  - numeric thresholds (min_submissions, drift_seconds_min, etc.)
  - exclude_matched_by chain with "highest_severity_wins" composition

Per brain_correction_layer_v2.md section 3; per TOML match semantics.

NO LLM calls here. This is pure rule evaluation.
"""
from __future__ import annotations

import fnmatch
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from brain.event_bus import BrainEvent
from brain.toml_loader import TriggerSpec, TriggersConfig


log = logging.getLogger("polybot.brain.trigger_matcher")


_SEVERITY_RANK = {
    "block_pending_review": 1,
    "flag": 2,
    "halt": 3,
}


@dataclass
class MatchResult:
    """One trigger matched against one event."""
    trigger: TriggerSpec
    event: BrainEvent
    # matched_by_reason: human-readable why
    reason: str = ""


def match_event(event: BrainEvent, config: TriggersConfig) -> list[MatchResult]:
    """Return all trigger matches for a single event.

    Resolves exclude_matched_by chains per exclude_composition rule. When
    composition is "highest_severity_wins" (default), the highest-severity
    matching trigger is kept and any lower-severity triggers listing the
    higher-severity one in exclude_matched_by are suppressed.
    """
    # Pass 1: literal matches by event_type rules
    raw_matches: list[MatchResult] = []
    for trig in config.triggers:
        reason = _evaluate_match(trig, event)
        if reason is not None:
            raw_matches.append(MatchResult(trigger=trig, event=event, reason=reason))

    if not raw_matches:
        return []

    matched_names = {m.trigger.name for m in raw_matches}

    # Pass 2: apply exclude_matched_by chain (highest-severity-wins default).
    kept: list[MatchResult] = []
    for m in raw_matches:
        excludes = m.trigger.match.get("exclude_matched_by") or []
        comp = m.trigger.match.get("exclude_composition", "highest_severity_wins")
        if not excludes:
            kept.append(m)
            continue
        overridden = False
        for other_name in excludes:
            if other_name not in matched_names:
                continue
            other = config.trigger_by_name(other_name)
            if other is None:
                continue
            if comp == "highest_severity_wins":
                if _SEVERITY_RANK.get(other.severity, 0) >= _SEVERITY_RANK.get(m.trigger.severity, 0):
                    overridden = True
                    break
            else:
                # default-unknown composition: suppress if ANY named match present
                overridden = True
                break
        if not overridden:
            kept.append(m)

    return kept


# ---- Per-trigger match logic ----


def _evaluate_match(trig: TriggerSpec, event: BrainEvent) -> Optional[str]:
    """Return a reason-string if trigger matches event, else None."""
    m = trig.match
    expected_type = m.get("event_type")
    if expected_type and expected_type != event.event_type:
        return None

    # All subsequent checks use event.payload OR top-level event fields.
    payload = dict(event.payload)

    # --- scope / band / breach_type / severity (payload equality) ---
    for field in ("scope", "band", "breach_type", "probe_severity",
                  "probe_scope", "warmup_scope", "stage", "backtest_status",
                  "emitter", "sum_violation_key", "deliberation_severity"):
        if field in m:
            expected = m[field]
            actual = payload.get(field)
            if isinstance(expected, list):
                if actual not in expected:
                    return None
            else:
                if actual != expected:
                    return None

    # --- boolean equality ---
    for field in ("is_new_key", "sanity_bound_violated", "schema_mismatch",
                  "tiered_anchor", "whitelisted_user"):
        if field in m:
            expected = m[field]
            actual = payload.get(field)
            if bool(actual) != bool(expected):
                return None

    # --- numeric thresholds (min / max / strict_lt) ---
    numeric_checks = [
        # (match-key, payload-key, op) where op ∈ {"ge","gt","le","lt"}
        ("count_min", "count", "ge"),
        ("silence_minutes_min", "silence_minutes", "ge"),
        ("silence_hours_min", "silence_hours", "ge"),
        ("silent_minutes_min", "silent_minutes", "ge"),
        ("git_activity_min", "git_activity", "ge"),
        ("drift_seconds_min", "drift_seconds", "ge"),
        ("drift_seconds_max_strict_lt", "drift_seconds", "lt"),
        ("sustained_minutes_min", "sustained_minutes", "ge"),
        ("signature_age_days_min", "signature_age_days", "ge"),
        ("sigma_threshold", "sigma", "ge"),
        ("window_days", "window_days", "eq"),
        ("trades_min", "trades_observed", "ge"),
        ("ratio_to_baseline_min", "ratio_to_baseline", "ge"),
        ("min_sustained_minutes", "sustained_minutes", "ge"),
        ("changes_min", "changes", "ge"),
        ("window_hours", "window_hours", "eq"),
        ("points_worse_min", "points_worse", "ge"),
        ("pending_count_min", "pending_count", "ge"),
        ("value_magnitude_change_ratio_min", "value_magnitude_change_ratio", "ge"),
        ("min_activity", "activity_count", "ge"),
    ]
    for m_key, ev_key, op in numeric_checks:
        if m_key not in m:
            continue
        if ev_key not in payload:
            # event didn't provide the measurement — fail closed: no match
            return None
        try:
            expected = float(m[m_key])
            actual = float(payload[ev_key])
        except (TypeError, ValueError):
            return None
        if op == "ge" and not (actual >= expected):
            return None
        if op == "gt" and not (actual > expected):
            return None
        if op == "lt" and not (actual < expected):
            return None
        if op == "le" and not (actual <= expected):
            return None
        if op == "eq" and actual != expected:
            return None

    # --- keys membership (config.change + sizing_threshold_change) ---
    if "keys" in m:
        wanted = set(m["keys"])
        provided = set(payload.get("keys", []))
        if not (wanted & provided):
            return None

    # --- conditions list with match_logic = "any_of" ---
    if "conditions" in m:
        wanted = set(m["conditions"])
        provided = set(payload.get("conditions", []))
        match_logic = m.get("match_logic", "any_of")
        if match_logic == "any_of":
            if not (wanted & provided):
                return None
        elif match_logic == "all_of":
            if not wanted.issubset(provided):
                return None

    # --- protected_patterns (glob match on changed_paths) ---
    if "protected_patterns" in m:
        patterns = list(m["protected_patterns"])
        paths = list(payload.get("changed_paths", []))
        if not any(fnmatch.fnmatch(p, pat) for p in paths for pat in patterns):
            return None

    # --- diff_regex (word-boundary regex; re.IGNORECASE | re.MULTILINE) ---
    if "diff_regex" in m:
        pat = re.compile(m["diff_regex"], re.IGNORECASE | re.MULTILINE)
        diff = payload.get("diff_text", "")
        if not pat.search(diff):
            return None

    # --- scheduled.daily action + check_time ---
    if "action" in m and event.event_type == "scheduled.daily":
        if payload.get("action") != m["action"]:
            return None
    if "check_time_ist" in m and event.event_type == "scheduled.daily":
        ct = payload.get("check_time_ist")
        if ct and ct != m["check_time_ist"]:
            return None

    # --- sampling_rate (only fires when payload says so) ---
    if "sampling_rate" in m:
        if not payload.get("sampled_in", False):
            return None

    # --- combined_with (rapid_change_with_loss) ---
    if "combined_with" in m:
        if not payload.get("combined_trigger_active", False):
            return None
        if "combined_drawdown_pct_min" in m:
            try:
                actual = float(payload.get("combined_drawdown_pct", 0.0))
                expected = float(m["combined_drawdown_pct_min"])
            except (TypeError, ValueError):
                return None
            if not (actual >= expected):
                return None

    # --- baseline-sample-size guard (anomalous_metric_deviation) ---
    if "min_baseline_samples" in m:
        try:
            actual = int(payload.get("baseline_samples", 0))
            expected = int(m["min_baseline_samples"])
        except (TypeError, ValueError):
            return None
        if actual < expected:
            return None

    return f"event_type={event.event_type} matched {trig.name}"


# ---- Helpers ----


def highest_severity(matches: list[MatchResult]) -> Optional[str]:
    """Return the highest severity label across matches, or None if empty."""
    if not matches:
        return None
    return max((m.trigger.severity for m in matches),
               key=lambda s: _SEVERITY_RANK.get(s, 0))


def group_by_correlation_id(matches: list[MatchResult]) -> dict[str, list[MatchResult]]:
    """Group matches by their event's correlation_id for dedup_with handling."""
    out: dict[str, list[MatchResult]] = {}
    for m in matches:
        out.setdefault(m.event.event_correlation_id, []).append(m)
    return out


def deduplicate_with(matches: list[MatchResult]) -> list[MatchResult]:
    """Apply dedup_with: when two triggers in the same correlation group list
    each other in dedup_with, keep only the highest-severity one.

    Returns a pruned list (original objects, no copies).
    """
    kept: list[MatchResult] = []
    groups = group_by_correlation_id(matches)
    for _cid, group in groups.items():
        names_in_group = {m.trigger.name for m in group}
        suppressed: set[str] = set()
        for m in group:
            dedup_names = set(m.trigger.dedup_with)
            if not dedup_names & names_in_group:
                continue
            # Collect severities among peers in dedup cluster (including self)
            cluster_names = (dedup_names & names_in_group) | {m.trigger.name}
            cluster_matches = [x for x in group if x.trigger.name in cluster_names]
            winner = max(
                cluster_matches,
                key=lambda x: _SEVERITY_RANK.get(x.trigger.severity, 0),
            )
            for x in cluster_matches:
                if x.trigger.name != winner.trigger.name:
                    suppressed.add(x.trigger.name)
        for m in group:
            if m.trigger.name not in suppressed:
                kept.append(m)
    return kept
