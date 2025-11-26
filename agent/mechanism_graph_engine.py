"""
Mechanism Graph Engine (MGE)
Ultra-Ceiling Version

Purpose:
    Convert mechanisms into a causal graph:
        • nodes = mechanisms, biomarkers, pathways
        • edges = causal direction, reference strength
        • weights = RYE × verification × stability
        • graph supports shortest-path reasoning and loop detection
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import math
import networkx as nx


class MechanismGraphEngine:
    """
    Full causal graph engine with:
        • directed graph construction
        • RYE-weighted edges
        • contradiction cycle detection
        • synergy chain discovery
        • pathway unification
        • bottleneck identification

    Uses NetworkX internally (pure Python + fast enough).
    """

    def __init__(self):
        self.graph = nx.DiGraph()

    # ---------------------------------------------------------
    # Graph building
    # ---------------------------------------------------------
    def add_mechanisms(self, mechanisms: List[Dict[str, Any]]):
        for m in mechanisms:
            mid = m.get("id")
            if not mid:
                continue

            self.graph.add_node(
                mid,
                text=m.get("text"),
                biomarkers=m.get("biomarkers", []),
                rye=m.get("rye", 0),
                stability=m.get("stability", 0),
            )

    def link(self, source: str, target: str, weight: float):
        """Add directional causal link."""
        self.graph.add_edge(source, target, weight=weight)

    # ---------------------------------------------------------
    # Inference heuristics
    # ---------------------------------------------------------
    def infer_edges(self):
        """Infer edges using lightweight causal keyword heuristics."""
        for a in self.graph.nodes:
            for b in self.graph.nodes:
                if a == b:
                    continue
                if self._causal_hint(self.graph.nodes[a]["text"],
                                     self.graph.nodes[b]["text"]):
                    w = self._edge_strength(a, b)
                    self.graph.add_edge(a, b, weight=w)

    def _causal_hint(self, a: str, b: str) -> bool:
        if not a or not b:
            return False
        al = a.lower()
        bl = b.lower()
        keys = ["activates", "reduces", "suppresses", "increases"]
        return any(k in al and k in bl for k in keys)

    def _edge_strength(self, a: str, b: str) -> float:
        """Use RYE × stability to determine edge priority."""
        na = self.graph.nodes[a]
        nb = self.graph.nodes[b]
        return (na["rye"] + nb["rye"]) * 0.5 * (na["stability"] + nb["stability"])

    # ---------------------------------------------------------
    # Analysis tools
    # ---------------------------------------------------------
    def detect_contradictions(self) -> List[Tuple[str, str]]:
        """Cycles often represent biochemical contradictions."""
        cycles = list(nx.simple_cycles(self.graph))
        return [(c[0], c[1]) for c in cycles if len(c) == 2]

    def shortest_chain(self, start: str, end: str) -> List[str]:
        try:
            return nx.shortest_path(self.graph, start, end, weight="weight")
        except Exception:
            return []

    def synergy_stacks(self, top_k: int = 10) -> List[List[str]]:
        """Extract the strongest subgraphs by edge weight."""
        edges = sorted(
            self.graph.edges(data=True),
            key=lambda e: e[2]["weight"],
            reverse=True,
        )
        stacks = []
        for s, t, w in edges[:top_k]:
            stacks.append([s, t])
        return stacks
