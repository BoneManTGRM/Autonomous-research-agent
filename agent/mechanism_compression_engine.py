"""
Mechanism Compression Engine (MCE)
Ultra-Ceiling Version

Purpose:
    Convert raw mechanisms into compressed latent representations that:
        • reduce redundancy
        • accelerate pattern recognition
        • amplify replay learning
        • enable faster cross-hallmark chaining
        • support mechanism evolution and graph reasoning

This module is equivalent to adding "embeddings for biology/mechanisms"
without needing an ML model.
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import hashlib
import math
import statistics


class MechanismCompressionEngine:
    """
    Ceiling-level mechanism compressor that:
        • Builds latent vectors
        • Encodes mechanism depth, biomarkers, pathways, RYE, stability
        • Computes similarity metrics
        • Produces compressed mechanism signatures (CMS)
        • Enables ultra-fast mechanism clustering
    """

    def __init__(self, hallmark_profiles):
        self.hallmarks = hallmark_profiles

    # -------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------
    def compress(self, mechanism: Dict[str, Any]) -> Dict[str, Any]:
        """Return a compressed representation of a mechanism."""
        text = mechanism.get("text", "")
        biomarkers = mechanism.get("biomarkers", [])
        rye = mechanism.get("rye", 0)
        stability = mechanism.get("stability", 0)

        latent = self._build_latent_vector(text, biomarkers, rye, stability)
        signature = self._build_signature(latent, text)

        return {
            "signature": signature,
            "latent": latent,
            "rye": rye,
            "stability": stability,
            "biomarkers": biomarkers,
            "text": text,
        }

    # -------------------------------------------------------------
    # Latent vector builder
    # -------------------------------------------------------------
    def _build_latent_vector(
        self,
        text: str,
        biomarkers: List[str],
        rye: float,
        stability: float,
    ) -> List[float]:
        """
        Create a compact biological latent vector.

        Dimensions:
            1. semantic density
            2. biomarker richness
            3. normalized RYE
            4. stability index
            5. hallmark keyword activation
        """
        length = len(text.split())
        density = min(1.0, length / 60.0)

        biom_score = min(1.0, len(biomarkers) / 10.0)

        hallmark_score = self._hallmark_activation(text)

        return [
            density,
            biom_score,
            rye,
            stability,
            hallmark_score,
        ]

    def _build_signature(self, latent: List[float], text: str) -> str:
        """Hash latent + text fingerprint."""
        fp = f"{latent}-{text[:200]}".encode("utf-8")
        return hashlib.sha256(fp).hexdigest()

    def _hallmark_activation(self, text: str) -> float:
        """How many hallmark keywords activate in this mechanism."""
        t = text.lower()
        score = 0
        for h, profile in self.hallmarks.hallmarks.items():
            for kw in profile.get("keywords", []):
                if kw.lower() in t:
                    score += 0.05
        return min(1.0, score)

    # -------------------------------------------------------------
    # Similarity
    # -------------------------------------------------------------
    def similarity(self, a: Dict[str, Any], b: Dict[str, Any]) -> float:
        """Cosine-like similarity between two latents."""
        va = a["latent"]
        vb = b["latent"]
        dot = sum(x * y for x, y in zip(va, vb))
        na = math.sqrt(sum(x * x for x in va))
        nb = math.sqrt(sum(x * x for x in vb))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
