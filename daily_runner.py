"""Daily runner for the Autonomous Research Agent.

This script is designed to be triggered once per day by an external scheduler
(cron, GitHub Actions, Render cron, etc.) to:

- Run a small number of cycles on a "daily update" goal
- Use your real memory file (same as Streamlit and long runs)
- Respect presets (general, longevity, math)
- Log results back into MemoryStore
- Print a human readable daily summary with RYE stats

Default:
    python daily_runner.py

Advanced examples:
    python daily_runner.py --preset longevity --cycles 5
    python daily_runner.py --preset math --cycles 2 --goal-prefix "Daily math scan"
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_store import MemoryStore
from agent.core_agent import CoreAgent
from agent.presets import get_preset
from settings import get_settings  # thin wrapper around AppConfig


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def resolve_memory_path() -> str:
    """Use the same memory file as the main app, with a safe fallback."""
    cfg = get_settings()
    path = getattr(cfg, "memory_file", None)
    if path is None:
        return "logs/sessions/default_memory.json"
    return str(path)


def ensure_directories() -> None:
    """Ensure logs and sessions directories exist."""
    logs = Path("logs")
    sessions = logs / "sessions"
    logs.mkdir(exist_ok=True, parents=True)
    sessions.mkdir(exist_ok=True, parents=True)


def build_daily_goal(prefix: str, domain_label: str) -> str:
    """Construct the daily goal string."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    return (
        f"{prefix} for {today}: summarize new or important findings in "
        f"{domain_label}, repair, and resilience, and update long run hypotheses."
    )


def summarize_day(summaries: List[Dict[str, Any]]) -> None:
    """Print a clear, lay interpretation of how the day went."""
    if not summaries:
        print("No cycles were run today.")
        return

    rye_vals: List[float] = []
    issues_total: int = 0
    issues_samples: List[str] = []

    for s in summaries:
        v = s.get("RYE")
        if isinstance(v, (int, float)):
            rye_vals.append(float(v))
        issues = s.get("issues_before") or []
        if isinstance(issues, list):
            issues_total += len(issues)
            for item in issues:
                if len(issues_samples) < 5:
                    issues_samples.append(str(item))

    print("\nDaily summary (human readable):")
    if rye_vals:
        avg_rye = sum(rye_vals) / len(rye_vals)
        print(f"- Cycles run: {len(summaries)}")
        print(f"- RYE range: {min(rye_vals):.3f} to {max(rye_vals):.3f}")
        print(f"- RYE average: {avg_rye:.3f}")

        if avg_rye > 0.5:
            print("- Interpretation: Very productive day. The agent is finding useful repairs.")
        elif avg_rye > 0.1:
            print("- Interpretation: Mixed but net positive. Some useful repairs and updates.")
        elif avg_rye > 0:
            print("- Interpretation: Small gains. The agent is working, but improvements are modest.")
        else:
            print("- Interpretation: No clear net repair today. Good candidate day to adjust goals or sources.")
    else:
        print("- RYE values were not available in these summaries.")

    print(f"- Total issues detected before repair across cycles: {issues_total}")
    if issues_samples:
        print("- Example issues the agent tried to repair today:")
        for ex in issues_samples:
            print(f"  • {ex}")


# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Daily runner for the Autonomous Research Agent")

    parser.add_argument(
        "--preset",
        type=str,
        default="longevity",
        choices=["general", "longevity", "math"],
        help="Domain preset to use for the daily scan.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=3,
        help="Number of cycles to run for the daily update.",
    )
    parser.add_argument(
        "--goal-prefix",
        type=str,
        default="Daily science update",
        help="Text prefix for the daily goal string.",
    )

    args = parser.parse_args()

    ensure_directories()

    # Shared settings and memory file
    memory_path = resolve_memory_path()
    memory = MemoryStore(memory_path)
    agent = CoreAgent(memory_store=memory, config={})

    # Preset and domain
    preset = get_preset(args.preset)
    domain_tag = preset.get("domain", args.preset)
    domain_label = preset.get("label", domain_tag)
    source_controls: Dict[str, bool] = preset.get("source_controls", {})

    # Build daily goal string
    goal = build_daily_goal(prefix=args.goal_prefix, domain_label=domain_label)

    # Starting index from existing history
    history = memory.get_cycle_history()
    start_idx = len(history)

    print("==============================================")
    print("Autonomous Research Agent - Daily Runner")
    print(f"Date (UTC): {datetime.utcnow().isoformat(timespec='seconds')}")
    print(f"Memory file: {memory_path}")
    print(f"Preset: {args.preset} (domain: {domain_tag})")
    print(f"Daily goal: {goal}")
    print(f"Planned cycles: {args.cycles}")
    print("==============================================\n")

    # Run a small number of cycles per day using the same interface as the UI
    summaries: List[Dict[str, Any]] = []
    for i in range(args.cycles):
        idx = start_idx + i
        result = agent.run_cycle(
            goal=goal,
            cycle_index=idx,
            role="agent",
            source_controls=source_controls,
            pdf_bytes=None,
            biomarker_snapshot=None,
            domain=domain_tag,
        )
        summary = result.get("summary", {})
        summaries.append(summary)
        rye = summary.get("RYE")
        print(f"Cycle {idx} completed. RYE: {rye}")

    print("\nDaily run complete. Raw summaries:")
    for s in summaries:
        print("-", s.get("RYE"), s.get("issues_before"))

    # Lay friendly interpretation at the end
    summarize_day(summaries)


if __name__ == "__main__":
    main()
