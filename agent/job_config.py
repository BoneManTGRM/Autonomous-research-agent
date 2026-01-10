"""
job_config.py
==============

This module defines a small dataclass for representing job
configurations for the Autonomous Research Agent.  A job defines how
many cycles to run, which domain or preset to use, optional swarm
parameters and any supplementary metadata.  Jobs are serialised to
JSON and dropped into a file‑based queue for the worker to consume.

The :class:`JobConfig` class provides convenience methods for dumping
itself to JSON and writing directly to the pending queue.  You can
extend the dataclass with additional fields as your engine requires.
"""

from __future__ import annotations

import os
import json
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional



def resolve_runs_root() -> Path:
    """Resolve the runs root directory.

    Uses ARA_RUNS_DIR when set, otherwise falls back to a local ./runs folder.
    """
    env = (
        os.environ.get("ARA_RUNS_DIR")
        or os.environ.get("ARA_RUN_ROOT")
        or os.environ.get("RUNS_DIR")
    )
    if env:
        try:
            return Path(env).expanduser().resolve()
        except Exception:
            return Path(env)
    return Path('runs').resolve()

PENDING_DIR = resolve_runs_root() / 'queue' / 'pending'


@dataclass
class JobConfig:
    cycles: int
    domain: str = "general"
    swarm_size: int = 1
    roles: Optional[List[tuple]] = None
    goal: str = ""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Extended configuration options for autonomous runs
    # Enable convergence check after a minimum number of cycles
    convergence_check: bool = True
    # Control critic verbosity (e.g. 'low', 'medium', 'high')
    critic_verbosity: str = "low"
    # Auto resolve TODO items during runs
    handle_unresolved_todos: bool = True
    # Method used to resolve TODOs
    todo_resolution_method: str = "recursive search + integrator handoff"
    # Explorer tool batching and retries
    explorer_batch_size: int = 5
    semantic_scholar_retries: int = 3
    # External validation toggles
    tavily_enabled: bool = True
    crossref_validation: bool = True
    # Consensus voting parameters
    consensus_check_interval: int = 3
    consensus_threshold: float = 0.65
    # Snapshotting and visualization
    autosave_snapshots: bool = True
    visualize_concept_graph: bool = True

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Flatten roles tuples to lists for JSON compatibility
        if self.roles is not None:
            data["roles"] = [[r, d] for (r, d) in self.roles]
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def write_to_queue(self, queue_dir: Optional[Path] = None) -> Path:
        """Write this job configuration into the pending queue.

        The filename will include the run ID and a ``_job.json`` suffix.

        Parameters
        ----------
        queue_dir:
            Override the default pending directory.  This is mainly
            provided for testing; in production the default of
            ``runs/queue/pending`` should be used.

        Returns
        -------
        pathlib.Path
            The path to the job file written.
        """
        target_dir = queue_dir or PENDING_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{self.run_id}_job.json"
        path = target_dir / fname
        with path.open("w", encoding="utf-8") as f:
            f.write(self.to_json())
        return path


__all__ = ["JobConfig"]
