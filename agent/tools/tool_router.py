"""
Advanced Tool Router for the Autonomous Research Agent.

This module provides a high level router that decides:
- which tools to call
- in what order
- with what intensity (aggressive vs conservative)
- and how to fuse or cascade results

Design goals:
- Respect Reparodynamics and TGRM concepts
- Use RYE, stability, volatility and Option C style diagnostics when available
- Learn from replay history to prioritize high yield tools
- Support multi step cascades and fusion of overlapping tool outputs
- Degrade gracefully when diagnostics or advanced metadata are missing

Usage pattern (high level):

    from tools.tool_router import ToolRouter, ToolContext

    router = ToolRouter(TOOL_REGISTRY)

    context = ToolContext(
        goal="Longevity and senescent cell clearance",
        subgoal="Map mechanisms for SASP suppression",
        domain="longevity",
        role="researcher",
        hallmark="senescence",
        needs_web=True,
        needs_pdf=True,
    )

    plan = router.plan_tools(
        payload={"query": "senescent cell SASP suppression interventions"},
        context=context,
        diagnostics=optional_diagnostics_dict,
        max_tools=3,
    )

    outputs = router.run_planned_tools(plan, payload, context)
    fused = router.fuse_outputs(outputs)

This router does not depend on Streamlit. It can be used by CoreAgent,
TGRMLoop, or any worker process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import math
import time
import statistics

# Optional imports from Reparodynamics metrics
try:
    from agent.rye_metrics import (
        build_run_diagnostics,
        autonomy_safety_envelope,
        early_failure_warning_score,
        estimate_breakthrough_probability,
        breakthrough_likelihood_90d,
        classify_run_tier,
    )
except Exception:  # pragma: no cover
    # Fallback stubs so the router still works if metrics are not available
    def build_run_diagnostics(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {}

    def autonomy_safety_envelope(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {}

    def early_failure_warning_score(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {}

    def estimate_breakthrough_probability(
        diagnostics: Dict[str, Any],
        domain: Optional[str] = None,
        horizon_hours: Optional[float] = None,
    ) -> Dict[str, Any]:
        return {"probability": None, "reason": "no metrics available"}

    def breakthrough_likelihood_90d(
        diagnostics: Dict[str, Any],
        domain: Optional[str] = None,
        hours_run_so_far: Optional[float] = None,
    ) -> Dict[str, Any]:
        return {"probability": None, "reason": "no metrics available"}

    def classify_run_tier(
        diagnostics: Dict[str, Any],
        breakthrough_prob: Optional[float] = None,
    ) -> Dict[str, Any]:
        return {"tier": "unknown", "reason": "no metrics available"}


# -------------------------------------------------------------------
# Core data structures
# -------------------------------------------------------------------


@dataclass
class ToolMetadata:
    """Normalized metadata for a single tool.

    The router tolerates minimal metadata and fills in defaults.
    """

    name: str
    kind: str = "generic"  # web, pdf, biomarker, sandbox, vectorizer, critic, etc
    description: str = ""
    domains: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)
    hallmarks: List[str] = field(default_factory=list)
    latency_class: str = "medium"  # fast, medium, slow
    reliability_hint: float = 0.5  # 0 to 1 hint, before replay data
    aggression_level: str = "balanced"  # conservative, balanced, aggressive
    callable: Optional[Callable[..., Any]] = None

    raw: Dict[str, Any] = field(default_factory=dict)
    fingerprint: Dict[str, float] = field(default_factory=dict)

    @staticmethod
    def from_registry_entry(name: str, entry: Any) -> "ToolMetadata":
        """Create ToolMetadata from a registry entry.

        Accepts either:
        - bare callable
        - dict with fields and callable under "fn" or "callable"
        """
        if callable(entry):
            return ToolMetadata(
                name=name,
                kind="generic",
                description="no description",
                callable=entry,
                raw={"type": "callable"},
            )

        meta: Dict[str, Any] = {}
        if isinstance(entry, dict):
            meta = dict(entry)
        else:
            meta = {"raw_entry": entry}

        fn = meta.get("callable") or meta.get("fn")
        kind = str(meta.get("kind", "generic"))
        description = str(meta.get("description", "") or "")
        domains = list(meta.get("domains", []) or [])
        roles = list(meta.get("roles", []) or [])
        hallmarks = list(meta.get("hallmarks", []) or [])
        latency_class = str(meta.get("latency_class", "medium"))
        reliability_hint = float(meta.get("reliability_hint", 0.5))
        aggression_level = str(meta.get("aggression_level", "balanced"))

        return ToolMetadata(
            name=name,
            kind=kind,
            description=description,
            domains=domains,
            roles=roles,
            hallmarks=hallmarks,
            latency_class=latency_class,
            reliability_hint=reliability_hint,
            aggression_level=aggression_level,
            callable=fn if callable(fn) else None,
            raw=meta,
        )


@dataclass
class ToolContext:
    """Context for a routing decision.

    This is passed in from CoreAgent or TGRMLoop.
    """

    goal: str
    subgoal: Optional[str] = None
    domain: Optional[str] = None
    role: Optional[str] = None
    hallmark: Optional[str] = None  # example: senescence, genomics, mitochondria
    run_tier: Optional[str] = None  # from classify_run_tier
    stability_index: Optional[float] = None
    volatility_score: Optional[float] = None
    hours_run_so_far: Optional[float] = None

    # Requirements
    needs_web: bool = False
    needs_pdf: bool = False
    needs_biomarkers: bool = False
    needs_sandbox: bool = False

    # Latency budget: fast, normal, long
    latency_budget: str = "normal"

    # If true, router should favor conservative, high precision tools
    verification_phase: bool = False

    # If true, router is allowed to be creative and exploratory
    exploration_phase: bool = False

    # Optional dictionary of extra hints, such as math_mode, critical_mode
    hints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolReplayStats:
    """Aggregate replay stats for a tool across runs."""

    calls: int = 0
    successes: int = 0
    failures: int = 0
    rye_gains: List[float] = field(default_factory=list)
    latencies_ms: List[float] = field(default_factory=list)

    @property
    def success_rate(self) -> Optional[float]:
        total = self.calls
        if total <= 0:
            return None
        return self.successes / float(total)

    @property
    def avg_rye_gain(self) -> Optional[float]:
        if not self.rye_gains:
            return None
        return statistics.mean(self.rye_gains)

    @property
    def avg_latency_ms(self) -> Optional[float]:
        if not self.latencies_ms:
            return None
        return statistics.mean(self.latencies_ms)


class ToolReplayBuffer:
    """Simple in memory replay buffer for tool statistics.

    This can be seeded from cycle history or updated live in TGRMLoop.
    """

    def __init__(self) -> None:
        self._stats: Dict[str, ToolReplayStats] = {}

    def record(
        self,
        tool_name: str,
        rye_gain: Optional[float] = None,
        success: Optional[bool] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        stats = self._stats.setdefault(tool_name, ToolReplayStats())
        stats.calls += 1
        if success is True:
            stats.successes += 1
        elif success is False:
            stats.failures += 1
        if isinstance(rye_gain, (int, float)):
            stats.rye_gains.append(float(rye_gain))
        if isinstance(latency_ms, (int, float)):
            stats.latencies_ms.append(float(latency_ms))

    def get(self, tool_name: str) -> Optional[ToolReplayStats]:
        return self._stats.get(tool_name)

    def snapshot(self) -> Dict[str, ToolReplayStats]:
        return dict(self._stats)


@dataclass
class ToolScore:
    """Score for a single tool under a given context."""

    tool: ToolMetadata
    score: float
    reasons: List[str] = field(default_factory=list)


@dataclass
class PlannedToolCall:
    """Represents a single planned tool execution."""

    tool: ToolMetadata
    score: float
    stage: str  # primary, cascade, verify, sanity
    reason: str


# -------------------------------------------------------------------
# Router
# -------------------------------------------------------------------


class ToolRouter:
    """High level tool router with fusion, cascade, and replay learning."""

    def __init__(
        self,
        tool_registry: Dict[str, Any],
        replay_buffer: Optional[ToolReplayBuffer] = None,
    ) -> None:
        self.replay_buffer = replay_buffer or ToolReplayBuffer()
        self.tools: Dict[str, ToolMetadata] = {}
        self._load_registry(tool_registry)

    # ------------------------------------------------------------------
    # Registry and fingerprints
    # ------------------------------------------------------------------
    def _load_registry(self, registry: Dict[str, Any]) -> None:
        """Normalize registry entries into ToolMetadata objects."""
        self.tools.clear()
        for name, entry in registry.items():
            meta = ToolMetadata.from_registry_entry(name, entry)
            meta.fingerprint = self._build_semantic_fingerprint(meta)
            self.tools[name] = meta

    def refresh_registry(self, registry: Dict[str, Any]) -> None:
        """Reload all tools from a new registry mapping."""
        self._load_registry(registry)

    def _build_semantic_fingerprint(self, meta: ToolMetadata) -> Dict[str, float]:
        """Construct a very simple semantic fingerprint for a tool.

        This is used to measure similarity between tools when fusing or
        when exploring tools with similar capabilities.
        """
        fp: Dict[str, float] = {}

        def bump(tag: str, value: float = 1.0) -> None:
            fp[tag] = fp.get(tag, 0.0) + value

        bump(f"kind:{meta.kind}", 2.0)

        for d in meta.domains:
            bump(f"domain:{d}", 1.5)
        for r in meta.roles:
            bump(f"role:{r}", 1.0)
        for h in meta.hallmarks:
            bump(f"hallmark:{h}", 2.0)

        text = (meta.description or "").lower()
        if "web" in text or "internet" in text or "search" in text:
            bump("cap:web", 1.5)
        if "pdf" in text or "paper" in text:
            bump("cap:pdf", 1.5)
        if "biomarker" in text or "omics" in text:
            bump("cap:biomarker", 1.5)
        if "sandbox" in text or "code" in text:
            bump("cap:sandbox", 1.0)
        if "critic" in text or "verify" in text or "validator" in text:
            bump("cap:critic", 2.0)

        if meta.aggression_level == "aggressive":
            bump("style:aggressive", 1.5)
        elif meta.aggression_level == "conservative":
            bump("style:conservative", 1.5)
        else:
            bump("style:balanced", 1.0)

        return fp

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _cosine_similarity(
        self, fp1: Dict[str, float], fp2: Dict[str, float]
    ) -> float:
        """Compute cosine similarity between two small sparse fingerprints."""
        if not fp1 or not fp2:
            return 0.0
        common = set(fp1.keys()) & set(fp2.keys())
        num = sum(fp1[k] * fp2[k] for k in common)
        den1 = math.sqrt(sum(v * v for v in fp1.values()))
        den2 = math.sqrt(sum(v * v for v in fp2.values()))
        if den1 <= 0 or den2 <= 0:
            return 0.0
        return num / (den1 * den2)

    def _context_fingerprint(self, context: ToolContext) -> Dict[str, float]:
        """Build a fingerprint from the context, to compare with tools."""
        fp: Dict[str, float] = {}

        def bump(tag: str, value: float = 1.0) -> None:
            fp[tag] = fp.get(tag, 0.0) + value

        if context.domain:
            bump(f"domain:{context.domain}", 2.0)
        if context.role:
            bump(f"role:{context.role}", 1.5)
        if context.hallmark:
            bump(f"hallmark:{context.hallmark}", 2.0)

        if context.needs_web:
            bump("cap:web", 2.0)
        if context.needs_pdf:
            bump("cap:pdf", 2.0)
        if context.needs_biomarkers:
            bump("cap:biomarker", 2.0)
        if context.needs_sandbox:
            bump("cap:sandbox", 1.5)

        if context.verification_phase:
            bump("cap:critic", 2.0)
            bump("style:conservative", 1.5)
        if context.exploration_phase:
            bump("style:aggressive", 1.5)

        return fp

    def _score_tool(
        self,
        meta: ToolMetadata,
        context: ToolContext,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> ToolScore:
        """Assign a score to a tool using metadata, replay, and diagnostics."""
        score = 0.0
        reasons: List[str] = []

        # 1. Basic semantic similarity
        ctx_fp = self._context_fingerprint(context)
        sim = self._cosine_similarity(meta.fingerprint, ctx_fp)
        score += sim * 4.0
        if sim > 0.0:
            reasons.append(f"semantic match {sim:.2f}")

        # 2. Domain and role hints
        if context.domain and context.domain in meta.domains:
            score += 2.0
            reasons.append("domain match")
        if context.role and context.role in meta.roles:
            score += 1.5
            reasons.append("role match")
        if context.hallmark and context.hallmark in meta.hallmarks:
            score += 2.0
            reasons.append("hallmark match")

        # 3. Base reliability
        base_rel = max(0.0, min(1.0, meta.reliability_hint))
        score += base_rel * 2.0
        reasons.append(f"base reliability {base_rel:.2f}")

        # 4. Latency budget
        if context.latency_budget == "fast":
            if meta.latency_class == "fast":
                score += 2.0
                reasons.append("fast tool fits fast budget")
            elif meta.latency_class == "slow":
                score -= 2.0
                reasons.append("slow tool under fast budget")
        elif context.latency_budget == "long":
            if meta.latency_class == "slow":
                score += 1.0
                reasons.append("slow but deeper tool allowed")

        # 5. Phase dependent style adjustments
        if context.verification_phase:
            if meta.aggression_level == "conservative":
                score += 1.5
                reasons.append("conservative tool in verification phase")
            if "critic" in meta.kind or "verify" in meta.description.lower():
                score += 2.0
                reasons.append("verification oriented tool")
        if context.exploration_phase:
            if meta.aggression_level == "aggressive":
                score += 1.5
                reasons.append("aggressive tool in exploration phase")

        # 6. Replay stats
        stats = self.replay_buffer.get(meta.name)
        if stats:
            if stats.success_rate is not None:
                sr = stats.success_rate
                score += sr * 3.0
                reasons.append(f"success rate {sr:.2f}")
            if stats.avg_rye_gain is not None:
                gain = stats.avg_rye_gain
                score += gain * 4.0
                reasons.append(f"avg RYE gain {gain:.3f}")
            if context.latency_budget == "fast" and stats.avg_latency_ms is not None:
                if stats.avg_latency_ms > 4000.0:
                    score -= 1.5
                    reasons.append("penalized for slow historical latency")

        # 7. Diagnostics (Option C style)
        diagnostics = diagnostics or {}
        stab = diagnostics.get("stability_index")
        volatility = diagnostics.get("volatility_score")
        early_fail = diagnostics.get("early_failure_warning") or {}
        ef_score = early_fail.get("score")

        if isinstance(stab, (int, float)):
            if stab < 0.3:
                # Environment is unstable. Favor conservative tools.
                if meta.aggression_level == "conservative":
                    score += 1.5
                    reasons.append("favored in low stability regime")
                if meta.aggression_level == "aggressive":
                    score -= 1.0
                    reasons.append("penalized in low stability regime")
            elif stab > 0.7:
                # Environment is stable. Aggressive tools get slight boost.
                if meta.aggression_level == "aggressive":
                    score += 1.0
                    reasons.append("stable regime allows aggressive tool")

        if isinstance(volatility, (int, float)):
            if volatility < 0.3 and meta.aggression_level == "aggressive":
                score += 0.5
                reasons.append("low volatility supports exploration")
            if volatility > 0.7 and meta.aggression_level == "aggressive":
                score -= 0.5
                reasons.append("high volatility dampens exploration")

        if isinstance(ef_score, (int, float)):
            if ef_score > 0.7 and meta.aggression_level == "aggressive":
                score -= 1.5
                reasons.append("early failure risk, aggressive tool penalized")
            if ef_score > 0.7 and meta.aggression_level == "conservative":
                score += 1.0
                reasons.append("early failure risk, conservative tool favored")

        return ToolScore(tool=meta, score=score, reasons=reasons)

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    def plan_tools(
        self,
        payload: Dict[str, Any],
        context: ToolContext,
        diagnostics: Optional[Dict[str, Any]] = None,
        max_tools: int = 3,
        allow_cascade: bool = True,
        allow_fusion: bool = True,
    ) -> Dict[str, Any]:
        """Plan which tools to run and in which order.

        Returns a dict with:
        - primary_tools: List[PlannedToolCall]
        - cascade_tools: List[PlannedToolCall]
        - fusion_enabled: bool
        """
        diagnostics = diagnostics or {}

        # Score all tools
        scores: List[ToolScore] = []
        for meta in self.tools.values():
            scores.append(self._score_tool(meta, context, diagnostics))

        # Sort descending by score
        scores.sort(key=lambda s: s.score, reverse=True)

        # Primary tools: top K with positive scores
        primary: List[PlannedToolCall] = []
        for s in scores:
            if len(primary) >= max_tools:
                break
            if s.score <= 0.0:
                continue
            primary.append(
                PlannedToolCall(
                    tool=s.tool,
                    score=s.score,
                    stage="primary",
                    reason="; ".join(s.reasons),
                )
            )

        # Cascade tools: next few candidates, for follow up if gaps remain
        cascade: List[PlannedToolCall] = []
        if allow_cascade:
            for s in scores[len(primary) : len(primary) + max_tools * 2]:
                if s.score <= 0.0:
                    continue
                cascade.append(
                    PlannedToolCall(
                        tool=s.tool,
                        score=s.score,
                        stage="cascade",
                        reason="backup or refinement tool",
                    )
                )

        return {
            "primary_tools": primary,
            "cascade_tools": cascade,
            "fusion_enabled": bool(allow_fusion and len(primary) >= 2),
            "context": context,
        }

    # ------------------------------------------------------------------
    # Execution and cascade
    # ------------------------------------------------------------------
    def run_planned_tools(
        self,
        plan: Dict[str, Any],
        payload: Dict[str, Any],
        context: ToolContext,
        max_cascade_tools: int = 2,
        gap_detector: Optional[
            Callable[[List[Dict[str, Any]]], Dict[str, Any]]
        ] = None,
    ) -> List[Dict[str, Any]]:
        """Execute primary tools and optional cascade tools.

        gap_detector, if provided, receives all primary results and returns:
            {
                "has_gaps": bool,
                "missing_fields": [...],
                "confidence": float in [0, 1]
            }

        The router uses that to decide if cascade tools should run.
        """
        primary_tools: List[PlannedToolCall] = plan.get("primary_tools", [])
        cascade_tools: List[PlannedToolCall] = plan.get("cascade_tools", [])

        all_outputs: List[Dict[str, Any]] = []

        # Run primary tools
        for pt in primary_tools:
            result = self._run_single_tool(pt.tool, payload, context)
            all_outputs.append(
                {
                    "tool": pt.tool.name,
                    "stage": pt.stage,
                    "score": pt.score,
                    "reason": pt.reason,
                    "output": result,
                }
            )

        # Decide whether cascades are needed
        run_cascade = False
        if gap_detector is not None and all_outputs:
            gap_info = gap_detector(all_outputs)
            if gap_info.get("has_gaps"):
                run_cascade = True
        else:
            # Simple heuristic: if only one tool ran, allow a small cascade
            if len(primary_tools) <= 1 and cascade_tools:
                run_cascade = True

        if not run_cascade:
            return all_outputs

        used = 0
        for ct in cascade_tools:
            if used >= max_cascade_tools:
                break
            result = self._run_single_tool(ct.tool, payload, context)
            all_outputs.append(
                {
                    "tool": ct.tool.name,
                    "stage": ct.stage,
                    "score": ct.score,
                    "reason": ct.reason,
                    "output": result,
                }
            )
            used += 1

        return all_outputs

    def _run_single_tool(
        self,
        meta: ToolMetadata,
        payload: Dict[str, Any],
        context: ToolContext,
    ) -> Any:
        """Internal helper to execute one tool and record basic replay stats."""
        fn = meta.callable
        if fn is None:
            return {"error": f"tool {meta.name} has no callable registered"}

        start = time.time()
        try:
            result = fn(payload=payload, context=context)
        except TypeError:
            # Fallback to simple signature if the tool does not accept named args
            try:
                result = fn(payload, context)
            except Exception as e:
                latency = (time.time() - start) * 1000.0
                self.replay_buffer.record(
                    meta.name, rye_gain=None, success=False, latency_ms=latency
                )
                return {"error": str(e), "tool": meta.name}
        except Exception as e:
            latency = (time.time() - start) * 1000.0
            self.replay_buffer.record(
                meta.name, rye_gain=None, success=False, latency_ms=latency
            )
            return {"error": str(e), "tool": meta.name}

        latency = (time.time() - start) * 1000.0

        # Try to infer success and RYE gain from output fields if present
        success = None
        rye_gain = None
        if isinstance(result, dict):
            # Many tools will not provide these fields, so this is optional
            success_val = result.get("success")
            if isinstance(success_val, bool):
                success = success_val
            rye_val = result.get("rye_gain") or result.get("delta_rye")
            if isinstance(rye_val, (int, float)):
                rye_gain = float(rye_val)

        self.replay_buffer.record(
            meta.name, rye_gain=rye_gain, success=success, latency_ms=latency
        )
        return result

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------
    def fuse_outputs(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fuse multiple tool outputs into a single consolidated result.

        This is a generic fusion strategy:
        - collect mechanism chains and hypotheses
        - merge citations
        - compute consensus confidence using simple statistics
        - preserve per tool contributions in metadata

        Expected fields in outputs (if tools support them):
        - output["mechanisms"] list of dict or strings
        - output["hypotheses"] list
        - output["citations"] list of dict
        - output["summary"] string
        """
        if not results:
            return {"fused": False, "reason": "no tool outputs"}

        fused: Dict[str, Any] = {
            "fused": True,
            "mechanisms": [],
            "hypotheses": [],
            "citations": [],
            "summaries": [],
            "tools": [],
        }

        citation_keys = set()
        mech_seen = set()
        hyp_seen = set()

        for entry in results:
            tool_name = entry.get("tool")
            out = entry.get("output", {})
            fused["tools"].append(
                {
                    "tool": tool_name,
                    "stage": entry.get("stage"),
                    "score": entry.get("score"),
                    "reason": entry.get("reason"),
                }
            )

            if not isinstance(out, dict):
                continue

            # Summaries
            summary = out.get("summary") or out.get("text") or out.get("description")
            if isinstance(summary, str) and summary.strip():
                fused["summaries"].append({"tool": tool_name, "text": summary.strip()})

            # Mechanisms
            mechanisms = out.get("mechanisms") or out.get("chains") or []
            if isinstance(mechanisms, dict):
                mechanisms = [mechanisms]
            if isinstance(mechanisms, list):
                for m in mechanisms:
                    if isinstance(m, dict):
                        sig = str(m.get("id") or m.get("label") or m)
                    else:
                        sig = str(m)
                    sig_norm = sig.strip().lower()
                    if not sig_norm or sig_norm in mech_seen:
                        continue
                    mech_seen.add(sig_norm)
                    fused["mechanisms"].append({"tool": tool_name, "data": m})

            # Hypotheses
            hyps = out.get("hypotheses") or []
            if isinstance(hyps, list):
                for h in hyps:
                    if isinstance(h, dict):
                        text = h.get("text", "")
                        conf = h.get("confidence")
                    else:
                        text = str(h)
                        conf = None
                    sig_norm = text.strip().lower()
                    if not sig_norm or sig_norm in hyp_seen:
                        continue
                    hyp_seen.add(sig_norm)
                    fused["hypotheses"].append(
                        {
                            "tool": tool_name,
                            "text": text,
                            "confidence": conf,
                        }
                    )

            # Citations
            cits = out.get("citations") or []
            if isinstance(cits, list):
                for ct in cits:
                    if not isinstance(ct, dict):
                        continue
                    key = (
                        ct.get("source"),
                        ct.get("title"),
                        ct.get("url"),
                    )
                    if key in citation_keys:
                        continue
                    citation_keys.add(key)
                    fused["citations"].append(
                        {
                            "tool": tool_name,
                            "source": ct.get("source"),
                            "title": ct.get("title"),
                            "url": ct.get("url"),
                        }
                    )

        # Consensus metrics
        conf_vals = []
        for h in fused["hypotheses"]:
            c = h.get("confidence")
            if isinstance(c, (int, float)):
                conf_vals.append(float(c))
        if conf_vals:
            fused["hypothesis_confidence"] = {
                "min": min(conf_vals),
                "max": max(conf_vals),
                "avg": statistics.mean(conf_vals),
            }

        fused["summary"] = self._build_fused_summary(fused)
        return fused

    def _build_fused_summary(self, fused: Dict[str, Any]) -> str:
        """Create a human readable summary of fused tool outputs."""
        lines: List[str] = []
        tools = fused.get("tools") or []
        mechs = fused.get("mechanisms") or []
        hyps = fused.get("hypotheses") or []
        cits = fused.get("citations") or []

        lines.append("Fused tool outputs:")
        if tools:
            lines.append("Tools involved:")
            for t in tools:
                lines.append(f"- {t.get('tool')} (stage {t.get('stage')})")

        if mechs:
            lines.append("")
            lines.append("Representative mechanisms:")
            for m in mechs[:5]:
                data = m.get("data")
                text = ""
                if isinstance(data, dict):
                    text = data.get("label") or data.get("description") or str(data)
                else:
                    text = str(data)
                if len(text) > 220:
                    text = text[:220] + "..."
                lines.append(f"- {text}")

        if hyps:
            lines.append("")
            lines.append("Representative hypotheses:")
            for h in hyps[:5]:
                text = str(h.get("text", "")).strip()
                if len(text) > 220:
                    text = text[:220] + "..."
                conf = h.get("confidence")
                if isinstance(conf, (int, float)):
                    lines.append(f"- {text} (confidence {conf:.2f})")
                else:
                    lines.append(f"- {text}")

        if cits:
            lines.append("")
            lines.append("Key citations (subset):")
            for ct in cits[:5]:
                title = ct.get("title") or "Untitled"
                src = ct.get("source") or "source"
                url = ct.get("url") or ""
                if url:
                    lines.append(f"- [{src}] {title} {url}")
                else:
                    lines.append(f"- [{src}] {title}")

        return "\n".join(lines)


__all__ = [
    "ToolRouter",
    "ToolContext",
    "ToolReplayBuffer",
    "ToolReplayStats",
    "ToolMetadata",
    "ToolScore",
    "PlannedToolCall",
]
