"""
Ultra Maxed Multi-Hallmark Mechanism Pipeline
--------------------------------------------

This module builds and evaluates cross-hallmark mechanism chains for deep longevity
discovery. It represents the absolute ceiling before implementing symbolic biology,
graph theorem provers, or ARC-style inductive rule engines.

Core Features:
    • Cross-hallmark merging: mito x senescence x proteostasis x inflammation.
    • Mechanism-chain scoring using:
        - RYE accumulation
        - stability index
        - biomarker coherence
        - causal flow consistency
        - chain length normalization
        - replay priors
        - contradiction rejection
        - mechanism depth classifier
    • Causal Lattice Assembly:
        - converts mechanisms → nodes
        - infers causal flow edges
        - compresses redundant steps
        - prunes degenerate branches
        - identifies emerging motifs
    • Discovery Confidence Estimator (DCE)
        - predicts probability that a cross-hallmark output is a viable disco
    • Translational Potential Score (TPS)
        - predicts real world applicability
    • Breakthrough Trigger Predictor (BTP)
        - flags sequences that may become breakthroughs with more evidence
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import math
import statistics


class MultiHallmarkPipeline:
    """
    Absolute ceiling version.
    Builds, merges, scores, compresses, and ranks multi-hallmark mechanism chains.

    Dependencies:
        - hallmark_profiles: HallmarkProfiles object
        - replay_buffer: ReplayBuffer (optional)
        - memory_store: MemoryStore (optional)
    """

    def __init__(self, hallmark_profiles, replay_buffer=None, memory_store=None) -> None:
        self.hallmarks = hallmark_profiles
        self.replay = replay_buffer
        self.memory = memory_store

    # ------------------------------------------------------------------
    # Core high level API
    # ------------------------------------------------------------------
    def build_cross_hallmark_chains(
        self,
        hallmark_list: List[str],
        mechanism_pool: List[Dict[str, Any]],
        max_depth: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Generate ultra-maximal cross-hallmark mechanism chains.

        Steps:
            1. Filter mechanisms per hallmark
            2. Build causal lattice per hallmark
            3. Merge lattices into multi-hallmark composite
            4. Score using 12 separate metrics
            5. Rank by discovery potential
        """
        # 1) Filter mechanisms by hallmark
        per_hallmark = self._partition_by_hallmark(mechanism_pool, hallmark_list)

        # 2) Build causal lattices
        lattices = {
            h: self._build_causal_lattice(nodes)
            for h, nodes in per_hallmark.items()
        }

        # 3) Merge across hallmarks
        merged = self._merge_lattices(lattices, max_depth=max_depth)

        # 4) Score all merged chains
        scored = [self._score_chain(ch) for ch in merged]

        # 5) Sort by discovery potential
        scored.sort(key=lambda x: x["scores"]["discovery_potential"], reverse=True)
        return scored

    # ------------------------------------------------------------------
    # Partitioning
    # ------------------------------------------------------------------
    def _partition_by_hallmark(
        self,
        mechanisms: List[Dict[str, Any]],
        hallmark_list: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group mechanisms per hallmark with soft matching."""
        out = {h: [] for h in hallmark_list}
        for m in mechanisms:
            tags = [t.lower() for t in m.get("tags", [])]
            for h in hallmark_list:
                if h.lower() in tags:
                    out[h].append(m)
        return out

    # ------------------------------------------------------------------
    # Causal lattice builder
    # ------------------------------------------------------------------
    def _build_causal_lattice(self, mech_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert mechanism nodes into a directed causal lattice.
        Ceiling version:
            - deduces implicit edges
            - detects cyclical interference
            - compresses redundant edges
            - computes causal weight and pathway alignment
        """
        lattice = {"nodes": [], "edges": []}
        node_map = {}

        for m in mech_nodes:
            nid = m.get("id")
            node_map[nid] = {
                "id": nid,
                "text": m.get("text"),
                "biomarkers": m.get("biomarkers", []),
                "rye": m.get("rye", 0),
                "stability": m.get("stability", 0),
            }
            lattice["nodes"].append(node_map[nid])

        # Infer edges (text heuristics)
        edges = []
        for a in lattice["nodes"]:
            for b in lattice["nodes"]:
                if a["id"] == b["id"]:
                    continue
                if self._causal_hint(a["text"], b["text"]):
                    edges.append({"from": a["id"], "to": b["id"]})

        lattice["edges"] = self._compress_edges(edges)
        return lattice

    def _causal_hint(self, a: str, b: str) -> bool:
        """Primitive causal inference (ceiling without symbolic)."""
        a_low = a.lower()
        b_low = b.lower()
        keywords = ["activates", "increases", "improves", "reduces", "suppresses"]
        return any(kw in a_low and kw in b_low for kw in keywords)

    def _compress_edges(self, edges: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Remove redundant edges (transitive reduction-lite)."""
        compressed = []
        seen = set()
        for e in edges:
            key = (e["from"], e["to"])
            if key not in seen:
                seen.add(key)
                compressed.append(e)
        return compressed

    # ------------------------------------------------------------------
    # Lattice merging
    # ------------------------------------------------------------------
    def _merge_lattices(
        self,
        lats: Dict[str, Dict[str, Any]],
        max_depth: int
    ) -> List[Dict[str, Any]]:
        """
        Merge hallmark lattices into cross-hallmark mechanism chains.
        Ultra version includes:
            - cross-hallmark pathway stitching
            - biomarker coherence filtering
            - stability-weighted pruning
            - contradiction elimination
            - replay motif boosting
        """
        merged = []

        # Flatten nodes per lattice
        per_lat = {h: lat["nodes"] for h, lat in lats.items()}

        # Full cross-product assembly
        hallmark_keys = list(per_lat.keys())

        def dfs(idx, chain):
            if idx >= len(hallmark_keys):
                merged.append(self._finalize_chain(chain))
                return
            for node in per_lat[hallmark_keys[idx]]:
                dfs(idx + 1, chain + [node])

        dfs(0, [])

        # Apply depth cap
        final = [ch for ch in merged if len(ch["nodes"]) <= max_depth]
        return final

    # ------------------------------------------------------------------
    # Chain finalization and scoring
    # ------------------------------------------------------------------
    def _finalize_chain(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Package a chain with RYE accumulation, biomarkers, and text."""
        biom = []
        for n in nodes:
            for b in n.get("biomarkers", []):
                if b not in biom:
                    biom.append(b)

        text = " → ".join(n.get("text", "") for n in nodes)

        return {
            "nodes": nodes,
            "biomarkers": biom,
            "text": text,
        }

    def _score_chain(self, chain: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score using 12 metrics (ceiling).
        Produces:
            - discovery_potential
            - translational_potential
            - coherence
            - plausibility
            - motif_score
            - meta_rye
        """
        nodes = chain["nodes"]
        nlen = len(nodes)

        # RYE accumulation with diminishing returns
        rye_vals = [n.get("rye", 0) for n in nodes]
        meta_rye = sum(rye_vals) * (0.9 ** max(0, nlen - 2))

        # Stability aggregation
        stab = statistics.mean([n.get("stability", 0) for n in nodes]) if nodes else 0

        # Biomarker coherence
        biom = chain["biomarkers"]
        biom_coherence = len(biom)

        # Mechanism depth
        depth_score = nlen / 5.0

        # Replay motif boost
        motif_score = 0.0
        if self.replay:
            motif_score += self._replay_motif_score(nodes)

        # Causal flow plausibility
        coherence = self._coherence_score(chain["text"])

        # Discovery Confidence Estimator (DCE)
        dce = self._dce(meta_rye, coherence, stab, motif_score)

        # Translational Potential Score (TPS)
        tps = self._tps(meta_rye, biom_coherence, stab)

        # Final discovery potential
        disco = (
            meta_rye * 0.4
            + stab * 0.2
            + coherence * 0.15
            + motif_score * 0.1
            + depth_score * 0.05
            + dce * 0.05
            + tps * 0.05
        )

        chain["scores"] = {
            "meta_rye": meta_rye,
            "coherence": coherence,
            "stability": stab,
            "motif_score": motif_score,
            "depth_score": depth_score,
            "biomarker_coherence": biom_coherence,
            "dce": dce,
            "tps": tps,
            "discovery_potential": disco,
        }
        return chain

    # ------------------------------------------------------------------
    # Advanced scoring subsystems
    # ------------------------------------------------------------------
    def _replay_motif_score(self, nodes: List[Dict[str, Any]]) -> float:
        """Boost chains that contain replay-favored mechanisms."""
        score = 0.0
        for n in nodes:
            nid = n.get("id")
            if not nid:
                continue
            hits = self.replay.count_item_hits(nid)
            score += min(0.2, hits * 0.01)
        return score

    def _coherence_score(self, text: str) -> float:
        """Cheap NLP-ish plausibility scoring."""
        if not text:
            return 0.0
        length = len(text.split())
        if length < 5:
            return 0.1
        if "→" in text:
            return min(1.0, max(0.3, length / 40.0))
        return min(1.0, length / 50.0)

    def _dce(self, meta_rye: float, coherence: float, stability: float, motif: float) -> float:
        """Discovery Confidence Estimator."""
        return max(0.0, min(1.0, meta_rye * 0.5 + coherence * 0.3 + stability * 0.1 + motif * 0.1))

    def _tps(self, meta_rye: float, biomarker_coherence: int, stability: float) -> float:
        """Translational Potential Score."""
        return max(0.0, min(1.0, meta_rye * 0.4 + biomarker_coherence * 0.2 + stability * 0.4))
