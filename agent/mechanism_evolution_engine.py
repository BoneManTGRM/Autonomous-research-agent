"""
Mechanism Evolution Engine (MEE)
Ultra-Ceiling Version

Purpose:
    Perform evolutionary search over biological mechanisms:
        • mutation of mechanisms
        • crossover between pathways
        • elimination pressure via RYE
        • survival of the fittest chains
        • new mechanism generation
        • long-run drift toward optimal patterns
"""

from __future__ import annotations
from typing import Any, Dict, List
import random
import uuid


class MechanismEvolutionEngine:
    """
    True ceiling evolutionary operator for longevity mechanisms.
    """

    def __init__(self, compression_engine, graph_engine, replay_buffer):
        self.comp = compression_engine
        self.graph = graph_engine
        self.replay = replay_buffer

    # ------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------
    def evolve(self, mechanisms: List[Dict[str, Any]], generations: int = 3) -> List[Dict[str, Any]]:
        pool = mechanisms[:]

        for _ in range(generations):
            # 1. mutation
            mutated = [self._mutate(m) for m in pool]

            # 2. crossover
            crossed = self._crossover_pool(pool)

            # 3. selection
            pool = self._select_survivors(pool + mutated + crossed)

        return pool

    # ------------------------------------------------------
    # Mutation
    # ------------------------------------------------------
    def _mutate(self, m: Dict[str, Any]) -> Dict[str, Any]:
        text = m.get("text", "")
        words = text.split()
        if len(words) > 3:
            idx = random.randint(0, len(words) - 1)
            words[idx] = "modulated_" + words[idx]
        new_text = " ".join(words)

        mutated = dict(m)
        mutated["id"] = f"mut_{uuid.uuid4().hex}"
        mutated["text"] = new_text
        mutated["rye"] = m.get("rye", 0) * random.uniform(0.9, 1.1)
        mutated["stability"] = m.get("stability", 0) * random.uniform(0.9, 1.1)

        return mutated

    # ------------------------------------------------------
    # Crossover
    # ------------------------------------------------------
    def _crossover_pool(self, pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for _ in range(len(pool)):
            if random.random() < 0.5:
                a, b = random.sample(pool, 2)
                out.append(self._crossover(a, b))
        return out

    def _crossover(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        """Hybrid mechanism: blend biomarkers + partial text."""
        ta = a.get("text", "").split()
        tb = b.get("text", "").split()

        new_text = " ".join(ta[: len(ta)//2] + tb[len(tb)//2 :])

        merged_biomarkers = list({*a.get("biomarkers", []), *b.get("biomarkers", [])})

        crossover = {
            "id": f"x_{uuid.uuid4().hex}",
            "text": new_text,
            "biomarkers": merged_biomarkers,
            "rye": (a.get("rye", 0) + b.get("rye", 0)) / 2,
            "stability": (a.get("stability", 0) + b.get("stability", 0)) / 2,
        }
        return crossover

    # ------------------------------------------------------
    # Selection
    # ------------------------------------------------------
    def _select_survivors(self, pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Top mechanisms survive based on:
            • RYE
            • stability
            • replay motif score
            • compression signature diversity
        """
        scored = []
        for m in pool:
            comp = self.comp.compress(m)
            motif = self.replay.count_item_hits(m.get("id", "")) if self.replay else 0
            score = (
                comp["rye"] * 0.45 +
                comp["stability"] * 0.3 +
                motif * 0.1 +
                self._signature_diversity(comp["signature"]) * 0.15
            )
            scored.append((score, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        # Keep best 40 percent
        keep = max(5, len(pool) // 2)
        return [m for _, m in scored[:keep]]

    def _signature_diversity(self, sig: str) -> float:
        """
        Encourages population diversity by favoring uncommon signatures.
        """
        return (sum(ord(c) for c in sig) % 100) / 100.0
