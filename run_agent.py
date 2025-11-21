# run_agent.py
"""
Headless runner for the Autonomous Research Agent.

Use this script for long runs (hours or days) outside of Streamlit.
It reuses:
- CoreAgent
- MemoryStore
- presets
- report_generator

Example:
    python run_agent.py \
        --goal "Multi-hour longevity research on rapamycin, metformin, and caloric restriction" \
        --preset longevity \
        --hours 8 \
        --max-cycles 5000
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import yaml

from agent.core_agent import CoreAgent
from agent.memory_store import MemoryStore
from agent.presets import get_preset
from agent.report_generator import generate_report

CONFIG_PATH_DEFAULT = "config/settings.yaml"
CYCLES_PER_HOUR_ESTIMATE = 120  # tweak based on real runs later


def load_settings(config_path: str = CONFIG_PATH_DEFAULT) -> Dict[str, Any]:
    """Load YAML settings file into a dictionary."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_directories() -> None:
    """Make sure logs and reports folders exist."""
    logs_path = Path("logs")
    sessions_path = logs_path / "sessions"
    reports_path = Path("reports")

    logs_path.mkdir(exist_ok=True, parents=True)
    sessions_path.mkdir(exist_ok=True, parents=True)
    reports_path.mkdir(exist_ok=True, parents=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless autonomous research agent runner")
    parser.add_argument(
        "--goal",
        type=str,
        required=True,
        help="Research goal for this long run.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="longevity",
        choices=["general", "longevity", "math"],
        help="Domain preset to use.",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=8.0,
        help="Target hours for the run (approximate, mapped to cycles).",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=5000,
        help="Hard cap on cycles. The smaller of this and hours estimate is used.",
    )
    parser.add_argument(
        "--stop-rye",
        type=float,
        default=0.0,
        help="Optional RYE stop threshold. Use 0 to disable.",
    )

    args = parser.parse_args()

    ensure_directories()

    # Load config and memory
    config = load_settings(CONFIG_PATH_DEFAULT)
    memory_file = config.get("memory_file", "logs/sessions/default_memory.json")
    memory = MemoryStore(memory_file)
    agent = CoreAgent(memory_store=memory, config=config)

    # Preset and domain
    preset = get_preset(args.preset)
    domain_tag = preset.get("domain", args.preset)
    source_controls = preset.get("source_controls", {})

    # Map hours to an estimated cycle budget
    estimated_cycles = int(args.hours * CYCLES_PER_HOUR_ESTIMATE)
    effective_max_cycles = min(estimated_cycles, args.max_cycles)

    stop_rye = args.stop_rye if args.stop_rye > 0 else None

    print("========================================")
    print("Autonomous Research Agent - Headless Run")
    print("Goal:", args.goal)
    print("Preset:", args.preset, "(domain:", domain_tag + ")")
    print("Target hours (approx):", args.hours)
    print("Cycle budget (estimated):", effective_max_cycles)
    if stop_rye is not None:
        print("RYE stop threshold:", stop_rye)
    else:
        print("RYE stop threshold: disabled")
    print("========================================")
    print("Starting continuous run...")
    print()

    summaries = agent.run_continuous(
        goal=args.goal,
        max_cycles=effective_max_cycles,
        stop_rye=stop_rye,
        role="agent",
        source_controls=source_controls,
        pdf_bytes=None,
        biomarker_snapshot=None,
        domain=domain_tag,
    )

    print()
    print("Run finished.")
    print("Cycles completed:", len(summaries))

    # Build a report from full history
    report_md = generate_report(memory_store=memory, goal=None)

    reports_path = Path("reports")
    reports_path.mkdir(exist_ok=True, parents=True)
    report_file = reports_path / "autonomous_research_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)

    print("Report written to:", report_file)


if __name__ == "__main__":
    main()
