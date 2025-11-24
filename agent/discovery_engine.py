# agent/discovery_engine.py

"""
DiscoveryEngine for the Autonomous Research Agent.

Goal:
    Scan cycle history and extract candidate discoveries that are:
    - high RYE
    - novel compared to prior notes or hypotheses
    - well supported by cycle context
    - aligned with domain specific discovery classes
      (mechanism, intervention, treatment, biomarker_shift,
       mathematical_structure, prediction, etc.)

Design:
    - Reads full cycle history from MemoryStore
    - Computes simple baselines for RYE and delta_R
    - Uses lightweight NLP heuristics and preset style hints
    - Avoids duplicates using text fingerprints
    - Writes unified JSON discovery log to:
        logs/discovery/discovery_log.json
        logs/discovery_log.json   (compat with existing UI loader)
    - Exposes run_discovery_pass for worker or CoreAgent

This module is intentionally self contained and does not depend
on the UI. It only needs a MemoryStore like object with:

    memory_store.get_cycle_history() -> List[Dict[str, Any]]

and optional presets like your existing PRESETS structure.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# Main canonical path plus a top level mirror so app_streamlit can read it
DISCOVERY_DIR = Path("logs/discovery")
DISCOVERY_LOG_MAIN = DISCOVERY_DIR / "discovery_log.json"
DISCOVERY_LOG_MIRROR = Path("logs/discovery_log.json")


# ------------------------------------------------------------
# Data structures
# ------------------------------------------------------------

@dataclass
class Discovery:
    """Single discovery record extracted from cycle history."""

    id: str
    created_at: str

    # Linkage
    cycle: Optional[int]
    role: str
    domain: str

    # Core content
    text: str
    kind: str          # mechanism, intervention, treatment, biomarker_shift, mathematical_structure, prediction, other
    source_type: str   # note, repair, hypothesis

    # Metrics
    rye: Optional[float]
    delta_R: Optional[float]
    energy_E: Optional[float]

    novelty_score: float
    support_score: float
    combined_score: float

    # Optional extras
    tags: List[str]
    extra: Dict[str, Any]


# ------------------------------------------------------------
# Low level helpers
# ------------------------------------------------------------

def _ensure_discovery_dirs() -> None:
    DISCOVERY_DIR.mkdir(parents=True, exist_ok=True)
    DISCOVERY_LOG_MAIN.parent.mkdir(parents=True, exist_ok=True)


def _load_json_file(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_discovery_log() -> List[Dict[str, Any]]:
    """
    Load existing discovery log entries.

    This is also compatible with the app_streamlit loader which
    looks for logs/discovery_log.json or logs/discovery/discovery_log.json.
    """
    # Prefer the canonical location first
    data = _load_json_file(DISCOVERY_LOG_MAIN)
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]

    # Fallback to older flat path
    data = _load_json_file(DISCOVERY_LOG_MIRROR)
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]

    return []


def _save_discovery_log(entries: List[Dict[str, Any]]) -> None:
    """Write discovery log to both main and mirror paths."""
    _ensure_discovery_dirs()
    safe_entries = [e for e in entries if isinstance(e, dict)]
    try:
        with DISCOVERY_LOG_MAIN.open("w", encoding="utf-8") as f:
            json.dump(safe_entries, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    try:
        with DISCOVERY_LOG_MIRROR.open("w", encoding="utf-8") as f:
            json.dump(safe_entries, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ------------------------------------------------------------
# Heuristics for novelty, support, and classification
# ------------------------------------------------------------

def _text_fingerprint(text: str) -> str:
    """Very cheap fingerprint to avoid duplicates."""
    t = " ".join(text.lower().split())
    if len(t) > 400:
        t = t[:400]
    return t


def _compute_stats(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    mean_v = sum(values) / len(values)
    var = sum((v - mean_v) ** 2 for v in values) / max(len(values) - 1, 1)
    std_v = math.sqrt(var)
    return mean_v, std_v


def _z_score(value: float, mean_v: Optional[float], std_v: Optional[float]) -> Optional[float]:
    if mean_v is None or std_v is None or std_v == 0.0:
        return None
    return (value - mean_v) / std_v


def _keyword_hits(text: str, keywords: Iterable[str]) -> int:
    t = text.lower()
    return sum(1 for kw in keywords if kw and kw.lower() in t)


def _classify_kind(text: str, domain: str) -> Tuple[str, List[str]]:
    """
    Very lightweight classifier to map text to discovery types plus tags.
    Tuned for longevity and math, with a fallback class for everything else.
    """
    t = text.lower()
    tags: List[str] = []

    # Longevity style cues
    lon_mech = ["pathway", "mechanism", "signaling", "axis", "cascade"]
    lon_int = ["intervention", "stack", "protocol", "treatment", "drug", "compound", "synergy"]
    lon_bio = ["biomarker", "marker", "blood", "lab", "plasma", "serum", "cholesterol"]
    lon_age = ["lifespan", "healthspan", "aging", "senescence", "autophagy", "mTOR", "rapamycin", "metformin", "NAD", "sirtuin"]

    # Math style cues
    math_formal = ["theorem", "lemma", "corollary", "axiom", "proof", "conjecture"]
    math_struct = ["structure", "operator", "measure", "functional", "stability", "equilibrium", "lyapunov", "martingale", "markov"]
    prediction_terms = ["predict", "prediction", "forecast", "estimate", "expected", "likely"]

    # Tag building
    if any(word in t for word in lon_age):
        tags.append("aging")
    if any(word in t for word in lon_mech):
        tags.append("mechanism")
    if any(word in t for word in lon_int):
        tags.append("intervention")
    if any(word in t for word in lon_bio):
        tags.append("biomarker")

    if any(word in t for word in math_formal):
        tags.append("math_formal")
    if any(word in t for word in math_struct):
        tags.append("math_structure")

    if any(word in t for word in prediction_terms):
        tags.append("prediction")

    # Primary kind selection
    if "math" in domain:
        if any(word in t for word in math_formal):
            return "mathematical_structure", tags or ["math"]
        if any(word in t for word in math_struct):
            return "mathematical_structure", tags or ["math"]
        if any(word in t for word in prediction_terms):
            return "prediction", tags or ["math_prediction"]
        return "framework", tags or ["math_candidate"]

    # Longevity / biology like domains
    if "longevity" in domain or "bio" in domain or "aging" in domain:
        if any(word in t for word in lon_bio):
            return "biomarker_shift", tags or ["biomarker"]
        if any(word in t for word in lon_int):
            return "intervention", tags or ["intervention"]
        if any(word in t for word in lon_mech):
            return "mechanism", tags or ["mechanism"]
        if any(word in t for word in prediction_terms):
            return "prediction", tags or ["prediction"]
        return "treatment", tags or ["longevity_candidate"]

    # Generic fallback
    if any(word in t for word in prediction_terms):
        return "prediction", tags or ["prediction"]
    if "mechanism" in t:
        return "mechanism", tags or ["mechanism"]
    if "intervention" in t or "treatment" in t or "protocol" in t:
        return "intervention", tags or ["intervention"]

    return "other", tags or ["generic"]


def _score_candidate(
    text: str,
    rye: Optional[float],
    delta_R: Optional[float],
    rye_mean: Optional[float],
    rye_std: Optional[float],
    delta_mean: Optional[float],
    delta_std: Optional[float],
    seen_fingerprints: Set[str],
) -> Tuple[float, float, float, float]:
    """
    Compute novelty_score, support_score, combined_score, and z_rye.

    Heuristics:
        - novelty comes from uniqueness and richness of text
        - support comes from high RYE and positive delta_R
        - combined_score is balanced and bounded between 0 and 1
    """
    t = text.strip()
    if not t:
        return 0.0, 0.0, 0.0, 0.0

    fingerprint = _text_fingerprint(t)
    already_seen = fingerprint in seen_fingerprints

    # Novelty score
    length_bonus = min(len(t) / 400.0, 1.0)  # long enough to be meaningful
    duplicate_penalty = 0.0 if not already_seen else 0.6

    novelty_score = max(0.0, min(1.0, 0.3 + 0.7 * length_bonus - duplicate_penalty))

    # Support from RYE and delta_R
    z_rye = 0.0
    if isinstance(rye, (int, float)):
        z_val = _z_score(float(rye), rye_mean, rye_std)
        z_rye = float(z_val) if z_val is not None else 0.0
    z_rye_clamped = max(-2.0, min(3.0, z_rye))

    z_delta = 0.0
    if isinstance(delta_R, (int, float)):
        z_val = _z_score(float(delta_R), delta_mean, delta_std)
        z_delta = float(z_val) if z_val is not None else 0.0
    z_delta_clamped = max(-2.0, min(3.0, z_delta))

    # Map both to [0,1]
    rye_support = 0.5 + 0.15 * z_rye_clamped
    delta_support = 0.5 + 0.10 * z_delta_clamped
    support_score = max(0.0, min(1.0, 0.4 * rye_support + 0.6 * delta_support))

    # Combined
    combined_score = max(0.0, min(1.0, 0.5 * novelty_score + 0.5 * support_score))

    return novelty_score, support_score, combined_score, z_rye


# ------------------------------------------------------------
# High level engine
# ------------------------------------------------------------

class DiscoveryEngine:
    """
    Offline discovery scanner.

    Typical usage from worker or CoreAgent:

        from agent.discovery_engine import DiscoveryEngine, run_discovery_pass

        de = DiscoveryEngine(memory_store=memory, presets=PRESETS)
        summary = de.run_pass()
    """

    def __init__(self, memory_store: Any, presets: Optional[Dict[str, Any]] = None) -> None:
        self.memory_store = memory_store
        self.presets = presets or {}

    def run_pass(
        self,
        domains: Optional[List[str]] = None,
        min_combined_score: float = 0.55,
        min_novelty_score: float = 0.45,
        min_support_score: float = 0.45,
        max_candidates_per_cycle: int = 5,
    ) -> Dict[str, Any]:
        """
        Scan full cycle history and append any new high quality discoveries.

        Returns a summary dict with counts and basic stats.
        """
        history: List[Dict[str, Any]] = self.memory_store.get_cycle_history() or []

        # Early exit
        if not history:
            return {
                "new_discoveries": 0,
                "total_discoveries": len(load_discovery_log()),
                "message": "No cycles in history. Nothing to analyze.",
            }

        # Compute baselines
        rye_vals = [float(e["RYE"]) for e in history if isinstance(e.get("RYE"), (int, float))]
        delta_vals = [float(e["delta_R"]) for e in history if isinstance(e.get("delta_R"), (int, float))]

        rye_mean, rye_std = _compute_stats(rye_vals)
        delta_mean, delta_std = _compute_stats(delta_vals)

        existing = load_discovery_log()
        existing_fp: Set[str] = set()
        for e in existing:
            txt = str(e.get("text", "") or "")
            if txt:
                existing_fp.add(_text_fingerprint(txt))

        new_entries: List[Discovery] = []

        # Domain filters
        allowed_domains = set(d.lower() for d in domains) if domains else None

        for entry in history:
            domain = str(entry.get("domain", "general"))
            if allowed_domains and domain.lower() not in allowed_domains:
                continue

            cycle_idx = entry.get("cycle")
            role = str(entry.get("role", "agent"))
            rye = float(entry["RYE"]) if isinstance(entry.get("RYE"), (int, float)) else None
            delta_R = float(entry["delta_R"]) if isinstance(entry.get("delta_R"), (int, float)) else None
            energy_E = float(entry["energy_E"]) if isinstance(entry.get("energy_E"), (int, float)) else None

            # Collect all candidate texts
            notes = entry.get("notes_added") or []
            repairs = entry.get("repairs") or []
            hyps = entry.get("hypotheses") or []

            candidates: List[Tuple[str, str]] = []

            for n in notes:
                txt = str(n).strip()
                if txt:
                    candidates.append(("note", txt))

            for r in repairs:
                txt = str(r).strip()
                if txt:
                    candidates.append(("repair", txt))

            for h in hyps:
                if isinstance(h, dict):
                    txt = str(h.get("text", "")).strip()
                else:
                    txt = str(h).strip()
                if txt:
                    candidates.append(("hypothesis", txt))

            if not candidates:
                continue

            cycle_new: List[Discovery] = []

            for source_type, text in candidates:
                novelty_score, support_score, combined_score, z_rye_val = _score_candidate(
                    text=text,
                    rye=rye,
                    delta_R=delta_R,
                    rye_mean=rye_mean,
                    rye_std=rye_std,
                    delta_mean=delta_mean,
                    delta_std=delta_std,
                    seen_fingerprints=existing_fp,
                )

                # Hard gate
                if combined_score < min_combined_score:
                    continue
                if novelty_score < min_novelty_score:
                    continue
                if support_score < min_support_score:
                    continue

                fingerprint = _text_fingerprint(text)
                if fingerprint in existing_fp:
                    continue

                kind, tags = _classify_kind(text, domain=domain)

                disc = Discovery(
                    id=f"{domain}-{cycle_idx}-{kind}-{len(cycle_new)}",
                    created_at=datetime.utcnow().isoformat(),
                    cycle=int(cycle_idx) if isinstance(cycle_idx, int) else None,
                    role=role,
                    domain=domain,
                    text=text,
                    kind=kind,
                    source_type=source_type,
                    rye=rye,
                    delta_R=delta_R,
                    energy_E=energy_E,
                    novelty_score=novelty_score,
                    support_score=support_score,
                    combined_score=combined_score,
                    tags=tags,
                    extra={
                        "z_rye": z_rye_val,
                        "length": len(text),
                    },
                )
                cycle_new.append(disc)
                existing_fp.add(fingerprint)

                if len(cycle_new) >= max_candidates_per_cycle:
                    break

            new_entries.extend(cycle_new)

        if not new_entries:
            return {
                "new_discoveries": 0,
                "total_discoveries": len(existing),
                "message": "No new discoveries above thresholds.",
            }

        # Merge and save
        merged = existing + [asdict(d) for d in new_entries]
        _save_discovery_log(merged)

        return {
            "new_discoveries": len(new_entries),
            "total_discoveries": len(merged),
            "mean_rye": rye_mean,
            "std_rye": rye_std,
            "mean_delta_R": delta_mean,
            "std_delta_R": delta_std,
            "message": f"Discovery pass complete. Added {len(new_entries)} new discoveries.",
        }


# ------------------------------------------------------------
# Convenience function
# ------------------------------------------------------------

def run_discovery_pass(
    memory_store: Any,
    presets: Optional[Dict[str, Any]] = None,
    domains: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Convenience entry point so worker or CoreAgent can do:

        from agent.discovery_engine import run_discovery_pass

        summary = run_discovery_pass(memory_store)
    """
    engine = DiscoveryEngine(memory_store=memory_store, presets=presets)
    return engine.run_pass(domains=domains, **kwargs)
