"""
Mechanism Hyperstack Manager
Ultra ceiling version

This module coordinates:
    • MechanismCompressionEngine
    • MechanismGraphEngine
    • MechanismEvolutionEngine

Goals:
    • Turn raw mechanisms into a compressed, graphed, evolving population
    • Feed the best mechanisms back into MemoryStore and ReplayBuffer
    • Expose a single entry point that CoreAgent, verification_pipeline,
      and multi_hallmark_pipeline can call to amplify learning speed
      and discovery rate over long runs.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .mechanism_compression_engine import MechanismCompressionEngine
from .mechanism_graph_engine import MechanismGraphEngine
from .mechanism_evolution_engine import MechanismEvolutionEngine


@dataclass
class HyperstackConfig:
    """Config knobs for the Hyperstack."""
    max_pool_size: int = 80
    min_pool_size: int = 10
    generations: int = 3
    top_write_back: int = 25
    attach_to_cycle_log: bool = True
    attach_to_run_reports: bool = True
    hallmark_weight_boost: float = 1.05
    rye_floor_for_promotion: float = 0.35
    stability_floor_for_promotion: float = 0.3


@dataclass
class HyperstackSummary:
    """Compact summary returned per hypercycle."""
    goal: str
    hallmark: Optional[str]
    total_input_mechanisms: int
    total_output_mechanisms: int
    generations: int
    avg_rye_before: Optional[float]
    avg_rye_after: Optional[float]
    avg_stability_before: Optional[float]
    avg_stability_after: Optional[float]
    promoted_count: int
    replay_items_written: int
    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )


class MechanismHyperstack:
    """
    High level orchestrator for mechanism compression, graphing, and evolution.

    Interactions:
        • Reads existing mechanisms from MemoryStore
        • Compresses them and builds a causal graph
        • Evolves the pool across several generations
        • Writes the best subset back to MemoryStore
        • Logs replay items to replay_buffer
        • Produces a HyperstackSummary suitable for cycle logs and reports
    """

    def __init__(
        self,
        memory_store: Any,
        hallmark_profiles: Any,
        replay_buffer: Any,
        config: Optional[HyperstackConfig] = None,
    ) -> None:
        self.memory_store = memory_store
        self.replay_buffer = replay_buffer
        self.config = config or HyperstackConfig()

        self.compressor = MechanismCompressionEngine(hallmark_profiles)
        self.graph = MechanismGraphEngine()
        self.evolver = MechanismEvolutionEngine(
            compression_engine=self.compressor,
            graph_engine=self.graph,
            replay_buffer=replay_buffer,
        )

    # ---------------------------------------------------------
    # Public entrypoint
    # ---------------------------------------------------------
    def run_hypercycle(
        self,
        goal: str,
        hallmark: Optional[str] = None,
        seed_mechanisms: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[HyperstackSummary, List[Dict[str, Any]]]:
        """
        Run one hypercycle:
            1. Build mechanism pool
            2. Compress and graph
            3. Evolve for N generations
            4. Select winners and write back
            5. Return summary and final pool
        """
        pool = self._build_mechanism_pool(
            goal=goal,
            hallmark=hallmark,
            seed_mechanisms=seed_mechanisms,
        )

        if len(pool) < self.config.min_pool_size:
            summary = HyperstackSummary(
                goal=goal,
                hallmark=hallmark,
                total_input_mechanisms=len(pool),
                total_output_mechanisms=len(pool),
                generations=0,
                avg_rye_before=self._avg(pool, "rye"),
                avg_rye_after=self._avg(pool, "rye"),
                avg_stability_before=self._avg(pool, "stability"),
                avg_stability_after=self._avg(pool, "stability"),
                promoted_count=0,
                replay_items_written=0,
            )
            return summary, pool

        avg_rye_before = self._avg(pool, "rye")
        avg_stab_before = self._avg(pool, "stability")

        # Build graph
        self.graph.add_mechanisms(pool)
        self.graph.infer_edges()

        # Evolve pool
        evolved = self.evolver.evolve(
            mechanisms=pool,
            generations=self.config.generations,
        )

        avg_rye_after = self._avg(evolved, "rye")
        avg_stab_after = self._avg(evolved, "stability")

        promoted, replay_count = self._write_back(
            goal=goal,
            hallmark=hallmark,
            mechanisms=evolved,
        )

        summary = HyperstackSummary(
            goal=goal,
            hallmark=hallmark,
            total_input_mechanisms=len(pool),
            total_output_mechanisms=len(evolved),
            generations=self.config.generations,
            avg_rye_before=avg_rye_before,
            avg_rye_after=avg_rye_after,
            avg_stability_before=avg_stab_before,
            avg_stability_after=avg_stab_after,
            promoted_count=promoted,
            replay_items_written=replay_count,
        )

        return summary, evolved

    # ---------------------------------------------------------
    # Pool builder
    # ---------------------------------------------------------
    def _build_mechanism_pool(
        self,
        goal: str,
        hallmark: Optional[str],
        seed_mechanisms: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        if seed_mechanisms:
            pool = list(seed_mechanisms)
        else:
            pool = self._pull_from_memory(goal=goal, hallmark=hallmark)

        # Clip and light cleanup
        unique = {}
        for m in pool:
            mid = m.get("id") or m.get("mechanism_id")
            if not mid:
                continue
            unique[mid] = m

        pool = list(unique.values())
        if len(pool) > self.config.max_pool_size:
            pool = sorted(
                pool,
                key=lambda m: (m.get("rye", 0), m.get("stability", 0)),
                reverse=True,
            )[: self.config.max_pool_size]

        return pool

    def _pull_from_memory(
        self,
        goal: str,
        hallmark: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Try several strategies to pull mechanisms from MemoryStore."""
        mechanisms: List[Dict[str, Any]] = []

        try:
            if hallmark and hasattr(self.memory_store, "get_best_mechanisms"):
                mechanisms = self.memory_store.get_best_mechanisms(
                    hallmark=hallmark,
                    top_k=self.config.max_pool_size,
                )
            elif hasattr(self.memory_store, "get_mechanisms_for_goal"):
                mechanisms = self.memory_store.get_mechanisms_for_goal(
                    goal,
                    limit=self.config.max_pool_size,
                )
            elif hasattr(self.memory_store, "get_mechanisms"):
                mechanisms = self.memory_store.get_mechanisms(
                    limit=self.config.max_pool_size
                )
        except Exception:
            mechanisms = []

        return mechanisms or []

    # ---------------------------------------------------------
    # Write back
    # ---------------------------------------------------------
    def _write_back(
        self,
        goal: str,
        hallmark: Optional[str],
        mechanisms: List[Dict[str, Any]],
    ) -> Tuple[int, int]:
        """
        Feed the best mechanisms back into:
            • memory_store.gold_mechanisms or equivalent
            • replay_buffer as replay items
        """
        if not mechanisms:
            return 0, 0

        sorted_mechs = sorted(
            mechanisms,
            key=lambda m: (m.get("rye", 0), m.get("stability", 0)),
            reverse=True,
        )

        top = sorted_mechs[: self.config.top_write_back]
        promoted_count = 0
        replay_count = 0

        for m in top:
            rye = float(m.get("rye", 0) or 0.0)
            stability = float(m.get("stability", 0) or 0.0)

            if hallmark:
                rye *= self.config.hallmark_weight_boost

            if rye >= self.config.rye_floor_for_promotion and stability >= self.config.stability_floor_for_promotion:
                promoted = self._promote_to_memory(
                    goal=goal,
                    hallmark=hallmark,
                    mechanism=m,
                )
                if promoted:
                    promoted_count += 1

            replay_written = self._write_replay_item(
                goal=goal,
                hallmark=hallmark,
                mechanism=m,
            )
            if replay_written:
                replay_count += 1

        return promoted_count, replay_count

    def _promote_to_memory(
        self,
        goal: str,
        hallmark: Optional[str],
        mechanism: Dict[str, Any],
    ) -> bool:
        """Try several memory_store hooks if available."""
        try:
            if hasattr(self.memory_store, "promote_to_gold"):
                self.memory_store.promote_to_gold(
                    hallmark or "general",
                    mechanism,
                )
                return True
            if hasattr(self.memory_store, "add_mechanism"):
                self.memory_store.add_mechanism(goal, mechanism)
                return True
            if hasattr(self.memory_store, "save_mechanism"):
                self.memory_store.save_mechanism(goal, mechanism)
                return True
        except Exception:
            return False
        return False

    def _write_replay_item(
        self,
        goal: str,
        hallmark: Optional[str],
        mechanism: Dict[str, Any],
    ) -> bool:
        if self.replay_buffer is None:
            return False

        try:
            text = mechanism.get("text", "")
            rye = float(mechanism.get("rye", 0) or 0.0)
            stability = float(mechanism.get("stability", 0) or 0.0)
            biomarkers = mechanism.get("biomarkers", [])

            item = {
                "item_id": mechanism.get("id"),
                "hallmark": hallmark or mechanism.get("hallmark") or "general",
                "stage": "hyperstack",
                "mechanism_chain": [text],
                "biomarker_pattern": biomarkers,
                "hypothesis_text": text,
                "rye_score": rye,
                "energy_cost": mechanism.get("energy_E", 0.0),
                "decision": "accepted" if rye >= self.config.rye_floor_for_promotion else "pending",
                "reason": "Hyperstack write back",
                "source_citations": mechanism.get("citations", []),
                "tags": list(set(mechanism.get("tags", []) + ["hyperstack"])),
                "run_id": mechanism.get("run_id"),
                "cycle_index": mechanism.get("cycle_index"),
            }
            self.replay_buffer.add_item(item)
            return True
        except Exception:
            return False

    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------
    def _avg(self, mechanisms: List[Dict[str, Any]], key: str) -> Optional[float]:
        vals: List[float] = []
        for m in mechanisms:
            v = m.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if not vals:
            return None
        return sum(vals) / len(vals)

    # ---------------------------------------------------------
    # Optional integration helpers
    # ---------------------------------------------------------
    def attach_summary_to_cycle(
        self,
        cycle_log: Dict[str, Any],
        summary: HyperstackSummary,
    ) -> None:
        """Attach hyperstack summary onto a TGRM cycle log if configured."""
        if not self.config.attach_to_cycle_log:
            return

        try:
            hs_block = {
                "goal": summary.goal,
                "hallmark": summary.hallmark,
                "total_input_mechanisms": summary.total_input_mechanisms,
                "total_output_mechanisms": summary.total_output_mechanisms,
                "generations": summary.generations,
                "avg_rye_before": summary.avg_rye_before,
                "avg_rye_after": summary.avg_rye_after,
                "avg_stability_before": summary.avg_stability_before,
                "avg_stability_after": summary.avg_stability_after,
                "promoted_count": summary.promoted_count,
                "replay_items_written": summary.replay_items_written,
                "timestamp": summary.timestamp,
            }
            cycle_log.setdefault("hyperstack", hs_block)
        except Exception:
            # Never allow reporting to break the run
            pass
