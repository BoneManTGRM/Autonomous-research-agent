"""
snapshot_generator.py

Weekly snapshot report generator for the Autonomous Research Agent.

This module produces a structured Markdown summary of long run activity, including:
    - RYE trends and statistics
    - High value discoveries and hypotheses status
    - Tool usage statistics and mode balance
    - Cycle activity counts and stability hints
    - Contradictions resolved and verification health
    - Memory growth metrics and pruning diagnostics
    - Swarm and intelligence profile summaries

Output folder:
    logs/snapshots/week_<number>.md

Trigger this from:
    - engine_worker.py (weekly schedule)
    - core_agent.py (periodic call)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

LOG_DIR = Path("logs")
SNAPSHOT_DIR = LOG_DIR / "snapshots"

SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOT_VERSION: str = "2025-11-23"


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_json(obj: Any) -> str:
    try:
        return "```json\n" + json.dumps(obj, indent=2, ensure_ascii=False) + "\n```"
    except Exception:
        return str(obj)


def _as_float(x: Any) -> Optional[float]:
    try:
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            return float(x.strip())
        return None
    except Exception:
        return None


class SnapshotGenerator:
    """
    Build and write weekly snapshot Markdown files.

    Expected usage:

        sg = SnapshotGenerator(run_id="run_001")
        sg.generate(
            week_number=3,
            cycle_stats={"cycles_total": 2048, "cycles_good": 1800},
            rye_stats={"avg": 0.42, "max": 0.88},
            hypotheses={
                "pending": [...],
                "validated": [...],
                "rejected": [...]
            },
            discoveries=[ ... ],
            tool_usage={ ... },
            contradictions=[ ... ],
            memory_stats={ ... },
            extra={ ... },
            run_meta={
                "domain": "longevity",
                "preset": "longevity",
                "runtime_profile": "24_hours",
                "intelligence_profile_name": "longevity_clinical",
            },
            swarm_stats={
                "enabled": True,
                "avg_agents": 5,
                "max_agents": 8,
                "roles_used": {"researcher": 180, "critic": 120, "integrator": 75},
            },
            pruning_summary={ ... },
            intelligence_profile={ ... },
            auto_from_logs=False,
        )

    New convenience fields on the instance:
        - last_snapshot_path: Path of the most recent snapshot file
        - last_payload: dict of the last arguments passed to generate()
    """

    def __init__(self, run_id: Optional[str] = None) -> None:
        self.run_id = run_id
        self.last_snapshot_path: Optional[Path] = None
        self.last_payload: Optional[Dict[str, Any]] = None

    # --------------------------
    # Markdown helpers
    # --------------------------
    def _md_header(
        self,
        week_number: int,
        run_meta: Optional[Dict[str, Any]],
    ) -> List[str]:
        lines: List[str] = []

        # Machine readable header block
        header_meta: Dict[str, Any] = {
            "snapshot_version": SNAPSHOT_VERSION,
            "week_number": week_number,
            "timestamp": utc_iso(),
            "run_id": self.run_id,
        }
        if run_meta:
            header_meta["run_meta"] = run_meta

        lines.append("```json")
        lines.append(json.dumps(header_meta, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")

        # Human readable header
        lines.append(f"# Weekly Snapshot Report - Week {week_number}")
        lines.append("")
        lines.append(f"**Timestamp:** {header_meta['timestamp']}")
        if self.run_id:
            lines.append(f"**Run ID:** `{self.run_id}`")
        if run_meta:
            domain = run_meta.get("domain")
            preset = run_meta.get("preset")
            runtime_profile = run_meta.get("runtime_profile")
            intel_name = run_meta.get("intelligence_profile_name")
            if domain:
                lines.append(f"**Domain:** `{domain}`")
            if preset:
                lines.append(f"**Preset:** `{preset}`")
            if runtime_profile:
                lines.append(f"**Runtime profile:** `{runtime_profile}`")
            if intel_name:
                lines.append(f"**Intelligence profile:** `{intel_name}`")
        lines.append("")
        lines.append(f"_Snapshot engine version: {SNAPSHOT_VERSION}_")
        lines.append("")
        lines.append("---")
        lines.append("")
        return lines

    def _md_section(self, title: str, content: str) -> List[str]:
        return [f"## {title}", "", content, "", "---", ""]

    def _format_list_block(self, items: List[Any]) -> str:
        if not items:
            return "_None recorded._"
        return "\n".join(f"- {item}" for item in items)

    def _format_dict_pretty(self, d: Dict[str, Any]) -> str:
        if not d:
            return "_None recorded._"
        return _safe_json(d)

    # --------------------------
    # Optional auto log readers
    # --------------------------
    def _tail_file(self, path: Path, max_lines: int = 40) -> Optional[str]:
        if not path.exists():
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return None
        lines = text.splitlines()
        tail = lines[-max_lines:]
        return "\n".join(tail)

    def _auto_discovery_summary(self, limit_lines: int = 40) -> Optional[str]:
        from_discovery = self._tail_file(LOG_DIR / "discovery_log.md", max_lines=limit_lines)
        if not from_discovery:
            return None
        return (
            "Recent discovery log tail (see `logs/discovery_log.md` for full history):\n\n"
            "```markdown\n"
            + from_discovery
            + "\n```"
        )

    def _auto_pruning_summary(self, limit_lines: int = 40) -> Optional[str]:
        from_prune = self._tail_file(LOG_DIR / "memory_pruning_log.md", max_lines=limit_lines)
        if not from_prune:
            return None
        return (
            "Recent memory pruning log tail (see `logs/memory_pruning_log.md` for full history):\n\n"
            "```markdown\n"
            + from_prune
            + "\n```"
        )

    # --------------------------
    # Interpretation helpers
    # --------------------------
    def _interpret_cycles(self, cycle_stats: Dict[str, Any]) -> List[str]:
        hints: List[str] = []
        total = _as_float(cycle_stats.get("cycles_total"))
        good = _as_float(cycle_stats.get("cycles_good"))
        stable = _as_float(cycle_stats.get("cycles_stable"))
        failed = _as_float(cycle_stats.get("cycles_failed"))

        if total is not None and total > 0:
            if good is not None:
                ratio = good / total
                hints.append(f"- Good cycle ratio this week: {ratio:.2%}.")
                if ratio >= 0.8:
                    hints.append("- Cycle health looks strong and stable.")
                elif ratio >= 0.6:
                    hints.append("- Cycle health is acceptable but can be improved.")
                else:
                    hints.append("- Cycle health is weak; check tools, timeouts, and prompts.")
            if failed is not None and failed > 0:
                hints.append(f"- Reported failed cycles: {int(failed)}. Consider checking logs.")
            if stable is not None:
                hints.append(f"- Cycles flagged as stable: {int(stable)}.")
        return hints

    def _interpret_rye(self, rye_block: Dict[str, Any]) -> None:
        hints: List[str] = []
        avg_val = rye_block.get("avg") or rye_block.get("mean")
        max_val = rye_block.get("max")
        trend = rye_block.get("trend")
        stability_index = rye_block.get("stability_index")

        avg_f = _as_float(avg_val)
        max_f = _as_float(max_val)
        trend_f = _as_float(trend)
        stab_f = _as_float(stability_index)

        if avg_f is not None:
            if avg_f >= 0.10:
                hints.append(f"- Average RYE is in the strong zone at about {avg_f:.3f}.")
            elif avg_f >= 0.05:
                hints.append(f"- Average RYE is in a maintenance zone at about {avg_f:.3f}.")
            else:
                hints.append(f"- Average RYE is low at about {avg_f:.3f}; agent may be exploring or struggling.")
        if max_f is not None:
            hints.append(f"- Peak RYE this week reached about {max_f:.3f}.")
        if trend_f is not None:
            if trend_f > 0.0:
                hints.append("- RYE trend is improving week over week.")
            elif trend_f < 0.0:
                hints.append("- RYE trend is declining; consider revisiting goals or tools.")
            else:
                hints.append("- RYE trend is flat.")
        if stab_f is not None:
            if stab_f >= 0.7:
                hints.append("- RYE stability index suggests a stable equilibrium zone.")
            elif stab_f <= 0.3:
                hints.append("- RYE stability index is low; system may be in a volatile regime.")

        if hints:
            rye_block["interpretation"] = hints

    def _format_hypothesis_item(self, item: Union[str, Dict[str, Any]]) -> str:
        if isinstance(item, str):
            return item
        title = item.get("title") or item.get("text") or "<untitled hypothesis>"
        score = _as_float(item.get("score"))
        domain = item.get("domain") or item.get("domain_tag")
        tier = item.get("tier_label")
        parts: List[str] = [str(title)]
        meta_bits: List[str] = []
        if score is not None:
            meta_bits.append(f"score {score:.3f}")
        if domain:
            meta_bits.append(f"domain {domain}")
        if tier:
            meta_bits.append(str(tier))
        if meta_bits:
            parts.append("[" + ", ".join(meta_bits) + "]")
        return " ".join(parts)

    def _format_hypothesis_list(self, items: Optional[List[Any]]) -> str:
        if not items:
            return "_None recorded._"
        lines: List[str] = []
        for item in items:
            lines.append(f"- {self._format_hypothesis_item(item)}")
        return "\n".join(lines)

    def _format_discovery_item(self, item: Union[str, Dict[str, Any]]) -> str:
        if isinstance(item, str):
            return item
        title = item.get("title") or "<untitled discovery>"
        tier = item.get("tier_label") or item.get("tier")
        rye = _as_float(item.get("rye") or item.get("rye_after"))
        tags = item.get("tags") or []
        tag_str = ""
        if tags:
            if isinstance(tags, list):
                tag_str = ", ".join(str(t) for t in tags)
            else:
                tag_str = str(tags)
        parts: List[str] = [str(title)]
        meta_bits: List[str] = []
        if tier:
            meta_bits.append(str(tier))
        if rye is not None:
            meta_bits.append(f"rye {rye:.3f}")
        if tag_str:
            meta_bits.append(tag_str)
        if meta_bits:
            parts.append("[" + "; ".join(meta_bits) + "]")
        return " ".join(parts)

    def _format_discovery_list(self, items: Optional[List[Any]]) -> str:
        if not items:
            return "_None recorded or not yet summarized._"
        lines: List[str] = []
        for item in items:
            lines.append(f"- {self._format_discovery_item(item)}")
        return "\n".join(lines)

    def _interpret_swarm_stats(self, swarm_stats: Dict[str, Any]) -> List[str]:
        hints: List[str] = []
        enabled = swarm_stats.get("enabled")
        avg_agents = _as_float(swarm_stats.get("avg_agents"))
        max_agents = _as_float(swarm_stats.get("max_agents"))
        roles_used = swarm_stats.get("roles_used") or {}
        mode = swarm_stats.get("mode")

        if enabled:
            hints.append("- Swarm mode was active this week.")
        else:
            hints.append("- Swarm mode was disabled or rarely used this week.")

        if avg_agents is not None and max_agents is not None:
            hints.append(f"- Swarm size: average agents {avg_agents:.1f}, max agents {int(max_agents)}.")
        elif avg_agents is not None:
            hints.append(f"- Average swarm size: about {avg_agents:.1f} agents.")

        if roles_used:
            try:
                total_roles = sum(int(v) for v in roles_used.values())
            except Exception:
                total_roles = 0
            if total_roles > 0:
                top_roles = sorted(
                    roles_used.items(),
                    key=lambda kv: kv[1],
                    reverse=True,
                )[:5]
                role_bits = []
                for r, c in top_roles:
                    share = (float(c) / float(total_roles)) * 100.0 if total_roles > 0 else 0.0
                    role_bits.append(f"{r} {share:.1f}%")
                hints.append("- Swarm role mix: " + ", ".join(role_bits) + ".")

        if mode:
            hints.append(f"- Swarm coordination mode: `{mode}`.")

        return hints

    def _interpret_pruning_summary(self, pruning_summary: Dict[str, Any]) -> List[str]:
        hints: List[str] = []
        before = _as_float(pruning_summary.get("total_before"))
        after = _as_float(pruning_summary.get("total_after"))
        dropped = _as_float(pruning_summary.get("dropped"))
        max_drop_fraction = _as_float(pruning_summary.get("max_drop_fraction"))
        diag = pruning_summary.get("diagnostics") or {}
        avg_rye_kept = _as_float(diag.get("avg_rye_kept"))
        avg_rye_dropped = _as_float(diag.get("avg_rye_dropped"))

        if before is not None and after is not None and dropped is not None:
            hints.append(
                f"- Memory size changed from {int(before)} entries to {int(after)} "
                f"with {int(dropped)} entries dropped."
            )
        if max_drop_fraction is not None:
            hints.append(f"- Maximum drop fraction configured at about {max_drop_fraction:.2f}.")

        if avg_rye_kept is not None and avg_rye_dropped is not None:
            if avg_rye_kept >= avg_rye_dropped:
                hints.append(
                    "- Average RYE of kept entries is higher than or equal to dropped entries, "
                    "which is consistent with repair aware pruning."
                )
            else:
                hints.append(
                    "- Average RYE of dropped entries exceeded kept entries; consider reviewing pruning settings."
                )

        return hints

    # --------------------------
    # Main generator
    # --------------------------
    def generate(
        self,
        week_number: int,
        cycle_stats: Optional[Dict[str, Any]] = None,
        rye_stats: Optional[Dict[str, Any]] = None,
        hypotheses: Optional[Dict[str, List[Any]]] = None,
        discoveries: Optional[List[Any]] = None,
        tool_usage: Optional[Dict[str, Any]] = None,
        contradictions: Optional[List[str]] = None,
        memory_stats: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        *,
        run_meta: Optional[Dict[str, Any]] = None,
        swarm_stats: Optional[Dict[str, Any]] = None,
        pruning_summary: Optional[Dict[str, Any]] = None,
        intelligence_profile: Optional[Dict[str, Any]] = None,
        auto_from_logs: bool = False,
    ) -> Path:
        """
        Build and write a snapshot Markdown report.
        Returns the path of the created file.

        All original arguments remain supported. New fields:

            run_meta:
                High level context for the run, for example:
                    {
                        "domain": "longevity",
                        "preset": "longevity",
                        "runtime_profile": "24_hours",
                        "intelligence_profile_name": "longevity_clinical",
                    }

            swarm_stats:
                Swarm operation summary:
                    {
                        "enabled": True,
                        "avg_agents": 5,
                        "max_agents": 8,
                        "roles_used": {"researcher": 180, "critic": 120, ...},
                        "mode": "round_robin",
                    }

            pruning_summary:
                Latest memory_pruner summary dict.

            intelligence_profile:
                Resolved intelligence profile dict from intelligence_profiles.py.

            auto_from_logs:
                If True and some sections are missing, the generator will try to
                pull recent tails from:
                    logs/discovery_log.md
                    logs/memory_pruning_log.md
        """

        file_path = SNAPSHOT_DIR / f"week_{week_number}.md"

        # Store last payload for debugging and meta use
        self.last_payload = {
            "week_number": week_number,
            "cycle_stats": cycle_stats,
            "rye_stats": rye_stats,
            "hypotheses": hypotheses,
            "discoveries": discoveries,
            "tool_usage": tool_usage,
            "contradictions": contradictions,
            "memory_stats": memory_stats,
            "extra": extra,
            "run_meta": run_meta,
            "swarm_stats": swarm_stats,
            "pruning_summary": pruning_summary,
            "intelligence_profile": intelligence_profile,
            "auto_from_logs": auto_from_logs,
        }

        # Optionally enrich from logs
        auto_discovery_block = None
        auto_prune_block = None
        if auto_from_logs:
            if not discoveries:
                auto_discovery_block = self._auto_discovery_summary()
            if not pruning_summary:
                auto_prune_block = self._auto_pruning_summary()

        lines: List[str] = []

        # Header
        lines.extend(self._md_header(week_number, run_meta=run_meta))

        # Run meta section (more detailed)
        if run_meta or intelligence_profile or swarm_stats:
            meta_block: Dict[str, Any] = {}
            if run_meta:
                meta_block["run_meta"] = run_meta
            if intelligence_profile:
                meta_block["intelligence_profile"] = intelligence_profile
            if swarm_stats:
                swarm_block = dict(swarm_stats)
                swarm_hints = self._interpret_swarm_stats(swarm_block)
                if swarm_hints:
                    swarm_block["interpretation"] = swarm_hints
                meta_block["swarm_stats"] = swarm_block
            lines.extend(self._md_section("Run Context", self._format_dict_pretty(meta_block)))

        # Cycle statistics
        cycle_block = cycle_stats or {}
        cycle_hints = self._interpret_cycles(cycle_block)
        if cycle_hints:
            cycle_block = dict(cycle_block)
            cycle_block["interpretation"] = cycle_hints
        lines.extend(
            self._md_section(
                "Cycle Activity",
                self._format_dict_pretty(cycle_block),
            )
        )

        # RYE statistics and hints
        rye_block = dict(rye_stats or {})
        if rye_block:
            self._interpret_rye(rye_block)
        lines.extend(
            self._md_section(
                "RYE Performance",
                self._format_dict_pretty(rye_block),
            )
        )

        # Hypotheses sections
        hyp = hypotheses or {}
        lines.extend(
            self._md_section(
                "Hypotheses - Pending",
                self._format_hypothesis_list(hyp.get("pending")),
            )
        )
        lines.extend(
            self._md_section(
                "Hypotheses - Validated",
                self._format_hypothesis_list(hyp.get("validated")),
            )
        )
        lines.extend(
            self._md_section(
                "Hypotheses - Rejected",
                self._format_hypothesis_list(hyp.get("rejected")),
            )
        )

        # Discoveries
        if discoveries:
            discoveries_block = self._format_discovery_list(discoveries)
        elif auto_discovery_block:
            discoveries_block = auto_discovery_block
        else:
            discoveries_block = "_None recorded or not yet summarized._"

        lines.extend(self._md_section("High Value Discoveries", discoveries_block))

        # Tool usage
        lines.extend(
            self._md_section(
                "Tool Usage Summary",
                self._format_dict_pretty(tool_usage or {}),
            )
        )

        # Contradictions resolved
        lines.extend(
            self._md_section(
                "Contradictions Resolved",
                self._format_list_block(contradictions or []),
            )
        )

        # Memory statistics
        mem_block = dict(memory_stats or {})
        if pruning_summary:
            prune_block = dict(pruning_summary)
            prune_hints = self._interpret_pruning_summary(prune_block)
            if prune_hints:
                prune_block["interpretation"] = prune_hints
            mem_block = {**mem_block, "latest_pruning_summary": prune_block}
        elif auto_prune_block:
            mem_block = {
                **mem_block,
                "latest_pruning_log_tail": auto_prune_block,
            }

        lines.extend(
            self._md_section(
                "Memory Metrics and Pruning",
                self._format_dict_pretty(mem_block),
            )
        )

        # Extra section if provided
        if extra:
            lines.extend(
                self._md_section(
                    "Additional Notes",
                    self._format_dict_pretty(extra),
                )
            )

        # Write file
        file_path.write_text("\n".join(lines), encoding="utf-8")
        self.last_snapshot_path = file_path

        return file_path
