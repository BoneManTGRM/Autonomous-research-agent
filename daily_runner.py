"""Daily runner for the Autonomous Research Agent.

This is optional. It is meant to be triggered once per day by an
external scheduler (cron, GitHub Actions, etc.) to:

- Run a small number of cycles on a fixed "daily update" goal,
- Log results to the existing memory file.

Example usage:
    python daily_runner.py
"""

from __future__ import annotations

from datetime import datetime

from agent.memory_store import MemoryStore
from agent.core_agent import CoreAgent


def main() -> None:
    memory = MemoryStore("logs/sessions/default_memory.json")
    agent = CoreAgent(memory_store=memory, config={})

    today = datetime.utcnow().strftime("%Y-%m-%d")
    goal = f"Daily science update for {today}: summarize new findings in aging, repair, and resilience."
    history = memory.get_cycle_history()
    start_idx = len(history)

    # Run a small number of cycles per day
    summaries = []
    for i in range(3):
        result = agent.run_cycle(goal=goal, cycle_index=start_idx + i)
        summaries.append(result["summary"])

    print("Daily run complete. Summaries:")
    for s in summaries:
        print("-", s.get("RYE"), s.get("issues_before"))


if __name__ == "__main__":
    main()
