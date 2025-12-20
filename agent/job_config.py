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

import json
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional


PENDING_DIR = Path("runs") / "queue" / "pending"


@dataclass
class JobConfig:
    cycles: int
    domain: str = "general"
    swarm_size: int = 1
    roles: Optional[List[tuple]] = None
    goal: str = ""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))

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