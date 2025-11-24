# run_agent.py
"""
Headless runner for the Autonomous Research Agent.

Upgraded (Option C: Bolt-Ons Only):
- Full breakthrough-aware continuous runner
- PDF reporter integration
- Biomarker snapshot support
- Swarm-ready (if preset enables it)
- Stability + RYE diagnostics panel in console
- Clean long-run instrumentation for 24h–90day experiments
- Zero removals from original file — only additions stacked on top
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from agent.core_agent import CoreAgent
from agent.memory_store import MemoryStore
from agent.presets import get_preset
from agent.report_generator import generate_report

# NEW (bolt-on)
from tools.pdf_reporter import generate_pdf_report  # Optional if available
from tools.breakthrough_detector import classify_breakthrough  # Optional if exists

CONFIG_PATH_DEFAULT = "config/settings.yaml"
CYCLES_PER_HOUR_ESTIMATE = 120  # tweak based on real runs later


# ======================================================================
# ORIGINAL — leave untouched
# ======================================================================
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


# ======================================================================
# NEW — console panel utilities
# ======================================================================
def print_header_block(label: str) -> None:
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70 + "\n")


def print_rye_status(summary: Dict[str, Any]) -> None:
    """Light, safe, optional readout for long runs."""
    rye = summary.get("rye")
    delta_r = summary.get("delta_r")
    E = summary.get("energy")

    if rye is None:
        print("RYE: (not available for this cycle)")
        return

    print(f"RYE: {rye:.4f}   ΔR: {delta_r}   E: {E}")


# ======================================================================
# MAIN EXECUTION
# ======================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Headless autonomous research agent runner")

    parser.add_argument("--goal", type=str, required=True,
                        help="Research goal for this long run.")
    parser.add_argument("--preset", type=str, default="longevity",
                        choices=["general", "longevity", "math"],
                        help="Preset domain.")
    parser.add_argument("--hours", type=float, default=8.0,
                        help="Target run duration in hours.")
    parser.add_argument("--max-cycles", type=int, default=5000,
                        help="Hard cycle cap.")
    parser.add_argument("--stop-rye", type=float, default=0.0,
                        help="RYE stop threshold (0 to disable).")
    parser.add_argument("--pdf", action="store_true",
                        help="Generate PDF summary at the end.")
    parser.add_argument("--biomarker", action="store_true",
                        help="Embed biomarker snapshot in PDF + markdown report.")
    parser.add_argument("--swarm", action="store_true",
                        help="Force swarm mode if preset supports it.")

    args = parser.parse_args()

    ensure_directories()

    # Load settings + memory
    config = load_settings(CONFIG_PATH_DEFAULT)
    memory_file = config.get("memory_file", "logs/sessions/default_memory.json")
    memory = MemoryStore(memory_file)
    agent = CoreAgent(memory_store=memory, config=config)

    preset = get_preset(args.preset)
    domain_tag = preset.get("domain", args.preset)
    source_controls = preset.get("source_controls", {})

    # Hours → cycles
    estimated_cycles = int(args.hours * CYCLES_PER_HOUR_ESTIMATE)
    effective_max_cycles = min(estimated_cycles, args.max_cycles)
    stop_rye = args.stop_rye if args.stop_rye > 0 else None

    print_header_block("Autonomous Research Agent — Headless Run")
    print(f"Goal: {args.goal}")
    print(f"Preset: {args.preset} (domain: {domain_tag})")
    print(f"Target Hours: {args.hours}")
    print(f"Estimated Cycle Budget: {effective_max_cycles}")
    print(f"RYE Stop Threshold: {stop_rye if stop_rye else 'disabled'}")
    print(f"Swarm Mode Forced: {args.swarm}")
    print_header_block("Starting...")

    # ==================================================================
    # NEW — optional biomarker snapshot before the run
    # ==================================================================
    biomarker_snapshot: Optional[Dict[str, Any]] = None
    if args.biomarker:
        biomarker_snapshot = {
            "status": "baseline_only",
            "notes": "Biomarker snapshotting enabled",
        }

    # ==================================================================
    # RUN CONTINUOUS ENGINE
    # ==================================================================
    summaries = agent.run_continuous(
        goal=args.goal,
        max_cycles=effective_max_cycles,
        stop_rye=stop_rye,
        role="agent",
        source_controls=source_controls,
        pdf_bytes=None,
        biomarker_snapshot=biomarker_snapshot,
        domain=domain_tag,
        force_swarm=args.swarm,        # NEW
        force_domain_preset=preset,    # NEW
    )

    print_header_block("Run Finished")
    print(f"Cycles Completed: {len(summaries)}")

    # ==================================================================
    # NEW — breakthrough scoring
    # ==================================================================
    breakthrough_label = None
    if summaries:
        latest = summaries[-1]
        try:
            breakthrough_label = classify_breakthrough(latest)
            print(f"Breakthrough Classification: {breakthrough_label}")
        except Exception:
            print("Breakthrough detection unavailable.")

    # ==================================================================
    # Generate report.md
    # ==================================================================
    report_md = generate_report(memory_store=memory, goal=args.goal)

    reports_path = Path("reports")
    reports_path.mkdir(exist_ok=True, parents=True)

    report_file = reports_path / "autonomous_research_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)

    print("Markdown Report Written:", report_file)

    # ==================================================================
    # NEW — PDF Generation
    # ==================================================================
    if args.pdf:
        try:
            pdf_bytes = generate_pdf_report(
                summaries=summaries,
                goal=args.goal,
                preset_name=args.preset,
                breakthrough=breakthrough_label,
                biomarker_snapshot=biomarker_snapshot,
            )
            pdf_file = reports_path / "autonomous_research_report.pdf"
            with open(pdf_file, "wb") as f:
                f.write(pdf_bytes)
            print("PDF Report Written:", pdf_file)
        except Exception as e:
            print("PDF generation failed:", e)


if __name__ == "__main__":
    main()
