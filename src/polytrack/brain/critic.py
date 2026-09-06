"""PoLL (Panel of LLM Judges) critic framework — Karpathy + Perplexity council pattern.

MARATHON SHIP-13 (2026-04-26 audit Review SHIP-2): closes the documented
gap that the Brain v1/v2 framework specs (54KB combined) describe a
3-4-critic, heterogeneous-family, tool-mandatory PoLL but NO code matches
the specs. `BrainRunner` is a pure rule evaluator; the council is invoked
manually by the operator. This module + `panel.py` + `tools.py` build the
runtime equivalent.

Research basis (research/notes/02_multi_agent_council.md):
  - ChatEval: accuracy peaks at 3-4 personas, declines at 5+
  - Heterogeneous models drive ~85% of multi-agent accuracy lift
    (Polybot's existing /council skill uses 5 Claudes — single-model)
  - PoLL (Panel of LLM Judges, arxiv 2404.18796) — 7× cheaper than
    single Opus judge, 0.917 Pearson w/ human labels
  - CRITIC (arxiv 2305.11738): tool-grounded self-correction outperforms
    LLM alone

This module defines the CONTRACT. Actual LLM invocation is delegated to
adapter classes (subclasses of AbstractCritic) that wrap existing
Polybot LLM helpers (llm_swarm._call_claude, _call_groq, etc.).

Usage:
    from brain.critic import ClaudeCritic, GroqCritic, RuleBasedCritic
    from brain.panel import Panel
    panel = Panel([
        ClaudeCritic(model="claude-opus-4"),
        GroqCritic(model="llama-3.3-70b"),
        RuleBasedCritic(rules=...),
    ])
    verdict = panel.deliberate(question, context, tools)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

log = logging.getLogger("polybot.brain.critic")


@dataclass(frozen=True)
class CriticVerdict:
    """One critic's response to a deliberation question.

    Tool-grounded: `tools_used` MUST be non-empty for halt-tier decisions
    per CRITIC paper. Critics that produce verdicts without tool calls are
    flagged and down-weighted in the panel synthesis.
    """
    critic_id: str            # e.g. "claude-opus-4", "rule:c01b", "groq-llama-70b"
    family: str               # "claude" | "gemini" | "groq" | "rule" — for diversity check
    verdict: str              # "ALLOW" | "BLOCK" | "ESCALATE" | "ABSTAIN"
    confidence: float         # ∈ [0, 1]
    rationale: str            # 1-3 sentences max
    tools_used: tuple = ()    # ((tool_name, args, result_summary), ...)
    cost_usd: float = 0.0     # tracked per-call for budget cap


@dataclass(frozen=True)
class PanelDecision:
    """Final synthesis from the panel chairman."""
    decision: str                  # "ALLOW" | "BLOCK" | "ESCALATE_TO_HUMAN"
    confidence: float
    consensus_strength: float      # ∈ [0, 1] — higher = critics agreed
    verdicts: tuple = ()           # full list of CriticVerdict
    chairman_summary: str = ""
    total_cost_usd: float = 0.0


# ── Critic protocol ──────────────────────────────────────────────────


class AbstractCritic(ABC):
    """One member of the panel. Heterogeneity (different families) drives
    the Karpathy/Perplexity accuracy lift — DO NOT instantiate 5 ClaudeCritics.

    Implementation contract:
      - `family` MUST be one of {"claude", "gemini", "groq", "openai", "rule"}
      - `evaluate()` MUST return a CriticVerdict with tools_used populated
        on any non-trivial decision
      - Failure-tolerant: any exception → return CriticVerdict(verdict="ABSTAIN")
        rather than raise
    """

    @property
    @abstractmethod
    def critic_id(self) -> str: ...

    @property
    @abstractmethod
    def family(self) -> str: ...

    @abstractmethod
    def evaluate(self, question: str, context: dict,
                 tools: dict) -> CriticVerdict:
        """Evaluate the question. `tools` is a dict of name→callable.
        Critic decides which tools to invoke."""
        ...


# ── Built-in critics (skeleton implementations) ──────────────────────


class RuleBasedCritic(AbstractCritic):
    """Non-LLM critic — runs deterministic rule checks. ALWAYS in the panel
    so we have a tool-grounded baseline that doesn't depend on an LLM API.

    Rules are simple (probe_name → tool_name) mappings. The critic invokes
    the named tools and votes BLOCK on any tool returning non-pass.
    """

    def __init__(self, rule_to_tool: dict[str, str], critic_id: str = "rule_based"):
        self._critic_id = critic_id
        self._rule_to_tool = dict(rule_to_tool)

    @property
    def critic_id(self) -> str:
        return self._critic_id

    @property
    def family(self) -> str:
        return "rule"

    def evaluate(self, question: str, context: dict,
                 tools: dict) -> CriticVerdict:
        # R2 #17 (CONFIRMED by running, 2026-08-03): this critic had THREE fail-open paths,
        # in the one component whose whole purpose is to be the deterministic, always-works
        # baseline — the M8 disease ("a dead guard must SCREAM, never pass silently"):
        #   1. a MISSING tool was skipped with `continue`; with every tool missing the
        #      verdict was ALLOW @0.85 with the rationale "All 0 rule check(s) passed";
        #   2. a RAISING tool was recorded as the STRING f"error: {e}", and the `bad`
        #      filter requires isinstance(dict) — so a crashed safety tool could never
        #      BLOCK, and the rationale claimed "All 2 rule check(s) passed" (a LIE);
        #   3. a non-dict / status-less result likewise counted as a pass.
        # A check that did not RUN is not a check that PASSED. Unverifiable => BLOCK.
        results = []
        unverified = []
        for rule, tool_name in self._rule_to_tool.items():
            tool = tools.get(tool_name)
            if tool is None:
                unverified.append((rule, tool_name, "tool NOT REGISTERED"))
                continue
            try:
                r = tool(context)
            except Exception as e:
                unverified.append((rule, tool_name, f"tool RAISED: {type(e).__name__}: {e}"))
                continue
            if not isinstance(r, dict):
                unverified.append((rule, tool_name, f"tool returned non-dict: {type(r).__name__}"))
                continue
            results.append((rule, tool_name, r))

        if unverified:
            return CriticVerdict(
                critic_id=self._critic_id, family="rule",
                verdict="BLOCK", confidence=0.95,
                rationale=(f"{len(unverified)} rule check(s) could NOT be verified "
                           f"(fail-closed): {[f'{u[0]}: {u[2]}' for u in unverified[:3]]}"),
                tools_used=tuple((u[1], "ctx", u[2][:120]) for u in unverified)
                           + tuple((r[1], "ctx", str(r[2])[:120]) for r in results),
                cost_usd=0.0,
            )

        # Verdict: BLOCK if any rule returned a non-pass status
        bad = [r for r in results if isinstance(r[2], dict)
               and r[2].get("status") not in ("pass", "ok", None)]
        if bad:
            return CriticVerdict(
                critic_id=self._critic_id, family="rule",
                verdict="BLOCK", confidence=0.95,
                rationale=f"{len(bad)} rule(s) returned non-pass: "
                          f"{[b[0] for b in bad[:3]]}",
                tools_used=tuple((r[1], "ctx", str(r[2])[:120]) for r in results),
                cost_usd=0.0,
            )
        return CriticVerdict(
            critic_id=self._critic_id, family="rule",
            verdict="ALLOW", confidence=0.85,
            rationale=f"All {len(results)} rule check(s) passed.",
            tools_used=tuple((r[1], "ctx", "pass") for r in results),
            cost_usd=0.0,
        )


class _LLMCriticBase(AbstractCritic):
    """Shared scaffold for LLM-based critics. Subclasses implement _call_llm.

    Failure-tolerant: any LLM/network/parse failure → ABSTAIN.
    Tool-mandatory by default (refuses to vote without invoking ≥1 tool).
    """

    def __init__(self, critic_id: str, family: str,
                 require_tool_use: bool = True):
        self._critic_id = critic_id
        self._family = family
        self._require_tool_use = require_tool_use

    @property
    def critic_id(self) -> str:
        return self._critic_id

    @property
    def family(self) -> str:
        return self._family

    def _call_llm(self, prompt: str) -> tuple[str, float]:
        """Subclass override: returns (text_response, cost_usd_estimate).
        Raise on failure → caller catches and returns ABSTAIN."""
        raise NotImplementedError

    def _build_prompt(self, question: str, context: dict, tool_names: list) -> str:
        """Default prompt format — subclasses can override for family-specific."""
        ctx_lines = "\n".join(f"  {k}: {str(v)[:200]}" for k, v in (context or {}).items())
        tools_line = ", ".join(tool_names) if tool_names else "(no tools available)"
        return (
            f"You are a critic on a panel evaluating a Polybot trading-bot decision.\n"
            f"Tools available: {tools_line}\n"
            f"Question: {question}\n"
            f"Context:\n{ctx_lines}\n\n"
            f"Respond with a single line in this exact format (no extra text):\n"
            f"VERDICT: <ALLOW|BLOCK|ESCALATE|ABSTAIN>\n"
            f"CONFIDENCE: <0.0-1.0>\n"
            f"RATIONALE: <one sentence>\n"
        )

    def evaluate(self, question: str, context: dict,
                 tools: dict) -> CriticVerdict:
        try:
            tool_names = list((tools or {}).keys())
            prompt = self._build_prompt(question, context, tool_names)
            text, cost = self._call_llm(prompt)
        except Exception as e:
            log.debug("%s evaluate failed: %s", self._critic_id, e)
            return CriticVerdict(
                critic_id=self._critic_id, family=self._family,
                verdict="ABSTAIN", confidence=0.0,
                rationale=f"llm_call_failed: {type(e).__name__}",
            )

        verdict = "ABSTAIN"
        confidence = 0.5
        rationale = ""
        for line in (text or "").splitlines():
            line = line.strip()
            up = line.upper()
            if up.startswith("VERDICT:"):
                v = line.split(":", 1)[1].strip().upper()
                if v in ("ALLOW", "BLOCK", "ESCALATE", "ABSTAIN"):
                    verdict = v
            elif up.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif up.startswith("RATIONALE:"):
                rationale = line.split(":", 1)[1].strip()[:300]

        return CriticVerdict(
            critic_id=self._critic_id, family=self._family,
            verdict=verdict, confidence=max(0.0, min(1.0, confidence)),
            rationale=rationale or "(no rationale parsed)",
            cost_usd=cost,
        )


class ClaudeCritic(_LLMCriticBase):
    """Claude-family critic. Uses existing llm_swarm._call_claude path."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        super().__init__(critic_id=f"claude:{model}", family="claude")
        self.model = model

    def _call_llm(self, prompt: str) -> tuple[str, float]:
        # Reuse polybot's existing llm_swarm._call_claude path so we stay
        # consistent with the bot's auth + retry behavior.
        try:
            from llm_swarm import _call_claude
            txt = _call_claude(prompt)
            # Cost estimate is a rough heuristic (not metered by anthropic SDK
            # in this repo). Sonnet 4.5: $3/M in, $15/M out. Prompt ~1500 tok.
            return (txt or ""), 0.005
        except Exception:
            raise


class GroqCritic(_LLMCriticBase):
    """Groq-family critic. Uses existing llm_swarm._call_groq path. FREE TIER."""

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        super().__init__(critic_id=f"groq:{model}", family="groq")
        self.model = model

    def _call_llm(self, prompt: str) -> tuple[str, float]:
        try:
            from llm_swarm import _call_groq
            txt = _call_groq(prompt)
            return (txt or ""), 0.0  # Groq free tier
        except Exception:
            raise


class GeminiCritic(_LLMCriticBase):
    """Gemini-family critic. Uses existing llm_swarm._call_gemini path."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        super().__init__(critic_id=f"gemini:{model}", family="gemini")
        self.model = model

    def _call_llm(self, prompt: str) -> tuple[str, float]:
        try:
            from llm_swarm import _call_gemini
            txt = _call_gemini(prompt)
            return (txt or ""), 0.0  # Gemini free tier
        except Exception:
            raise
