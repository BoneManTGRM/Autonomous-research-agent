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
       mathematical_structure, prediction, equilibrium_shift, etc.)
    - compatible with Tier 1 / Tier 2 / Tier 3 discovery hints

Design:
    - Reads full cycle history from MemoryStore
    - Computes baselines for RYE and delta_R
    - Uses lightweight NLP heuristics and preset style hints
    - Avoids duplicates using text fingerprints
    - Integrates optionally with:
        * HypothesisManager (auto create pending hypotheses)
        * DiscoveryLogger (Markdown log of discovery candidates)
        * intelligence_profiles (adaptive thresholds and scoring)
    - Writes unified JSON discovery log to:
        logs/discovery/discovery_log.json
        logs/discovery_log.json   (compat with existing UI loader)
    - Exposes run_discovery_pass for worker or CoreAgent

This module is intentionally self contained with optional hooks.
It only requires a MemoryStore like object with:

    memory_store.get_cycle_history() -> List[Dict[str, Any]]

and optional components:

    hypothesis_manager: HypothesisManager
    discovery_logger: DiscoveryLogger
    intelligence_profile: Dict[str, Any] from intelligence_profiles.py
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Placeholder and template detection
#
# DiscoveryEngine should never emit internal template entries, maintenance logs,
# or example placeholders.  BANNED_DISCOVERY_PATTERNS matches strings that
# indicate these artifacts.  _contains_banned_pattern is used to skip
# candidates that match any of these patterns.
# Expand the banned patterns list to catch more template leakage and
# promptâdirective artefacts.  This prevents system directives or
# initialization text from being treated as real discoveries.  The
# patterns below are all kept lowercase for simple substring checks.
BANNED_DISCOVERY_PATTERNS: List[str] = [
    "maintenance_mode",
    "maintenance mode",
    "discovery_log",
    "discovery log",
    "placeholder discovery",
    "template entry",
    "template",
    "example",
    "inconclusive verification",
    "rejected hypothesis",
    "performed targeted research",
    "initial discovery_log.json",
    "initial discovery log",
    "used only to show",
    "shows how an",
    # Additional patterns to filter run directive and prompt injection text
    "system directive",
    "autonomous research swarm",
    "coordinated cycle",
    "single coordinated cycle",
    "64-agent",
    "64 agent",
    "run summary",
    "agent autonomous",
]


def _contains_banned_pattern(text: str) -> bool:
    """Return True if the given text contains any banned placeholder pattern."""
    if not text:
        return False
    lower = text.lower()
    for pat in BANNED_DISCOVERY_PATTERNS:
        if pat in lower:
            return True
    return False

try:
    # Optional imports, only used if passed in via __init__
    from .hypothesis_manager import HypothesisManager  # type: ignore
    from .discovery_log import DiscoveryLogger  # type: ignore
except Exception:  # pragma: no cover - keep import optional
    HypothesisManager = Any  # type: ignore
    DiscoveryLogger = Any  # type: ignore

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
    kind: str          # mechanism, intervention, treatment, biomarker_shift, mathematical_structure, prediction, equilibrium_shift, framework, other
    source_type: str   # note, repair, hypothesis

    # Metrics
    rye: Optional[float]
    delta_R: Optional[float]
    energy_E: Optional[float]

    novelty_score: float
    support_score: float
    combined_score: float

    # Search and information metrics
    info_gain: Optional[float]
    search_energy: Optional[float]
    semantic_diversity: Optional[float]

    # Swarm context
    swarm_size: Optional[int]
    swarm_config: Optional[Dict[str, Any]]

    # Tier heuristics
    tier_label: Optional[str]

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
    Load existing discovery log entries, filtering out bootstrapping and
    example placeholders.

    This function reads the discovery log from either the primary or
    mirror locations and then removes entries that are known to be
    templates or bootstrap examples.  Entries with a ``run_id`` of
    "default" or any value containing "bootstrap"/"example" are
    discarded, as are entries whose ``tags`` contain "example",
    "template", or "bootstrap".  Filtering here prevents these
    artefacts from polluting discovery summaries and RYE statistics.
    """
    # Prefer the canonical location first
    data = _load_json_file(DISCOVERY_LOG_MAIN)
    if not isinstance(data, list):
        data = _load_json_file(DISCOVERY_LOG_MIRROR)
    entries: List[Dict[str, Any]] = []
    if isinstance(data, list):
        for d in data:
            if not isinstance(d, dict):
                continue
            run_id = str(d.get("run_id", ""))
            # Discard entries from bootstrapping or examples
            if run_id.lower() in {"default", ""}:
                continue
            if any(tok in run_id.lower() for tok in ["bootstrap", "example"]):
                continue
            tags = d.get("tags") or []
            bad_tag = False
            if isinstance(tags, list):
                for t in tags:
                    if isinstance(t, str) and any(x in t.lower() for x in ["example", "template", "bootstrap"]):
                        bad_tag = True
                        break
            if bad_tag:
                continue
            entries.append(d)
    return entries


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
    Lightweight classifier to map text to discovery types plus tags.
    Tuned for longevity and math, with a fallback for everything else.
    """
    t = text.lower()
    tags: List[str] = []

    # Longevity style cues
    lon_mech = ["pathway", "mechanism", "signaling", "axis", "cascade"]
    lon_int = ["intervention", "stack", "protocol", "treatment", "drug", "compound", "synergy"]
    lon_bio = ["biomarker", "marker", "blood", "lab", "plasma", "serum", "cholesterol", "inflammation"]
    lon_age = ["lifespan", "healthspan", "aging", "senescence", "autophagy", "mtor", "rapamycin", "metformin", "nad", "sirtuin", "telomere"]

    # Math style cues
    math_formal = ["theorem", "lemma", "corollary", "axiom", "proof", "conjecture"]
    math_struct = ["structure", "operator", "measure", "functional", "stability", "equilibrium", "lyapunov", "martingale", "markov", "invariant"]
    prediction_terms = ["predict", "prediction", "forecast", "estimate", "expected", "likely", "probability"]

    # Reparodynamics specific cues
    reparo_terms = ["rye", "repair yield", "delta_r", "energy", "equilibrium", "plateau", "coherence", "stability zone"]

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

    if any(word in t for word in reparo_terms):
        tags.append("reparodynamics")

    # Primary kind selection by domain

    # Math heavy
    if "math" in domain:
        if any(word in t for word in math_formal):
            return "mathematical_structure", tags or ["math"]
        if any(word in t for word in math_struct):
            return "mathematical_structure", tags or ["math"]
        if any(word in t for word in prediction_terms):
            return "prediction", tags or ["math_prediction"]
        return "framework", tags or ["math_candidate"]

    # Longevity or biology like domains
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

    # Reparodynamics equilibrium style
    if "equilibrium" in t or "plateau" in t or "stability zone" in t:
        tags.append("equilibrium")
        return "equilibrium_shift", tags or ["equilibrium"]

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
) -> Tuple[float, float, float, float, float]:
    """
    Compute novelty_score, support_score, combined_score, z_rye, z_delta.

    Heuristics:
        - novelty comes from uniqueness and richness of text
        - support comes from high RYE and positive delta_R
        - combined_score is balanced and bounded between 0 and 1
    """
    t = text.strip()
    if not t:
        return 0.0, 0.0, 0.0, 0.0, 0.0

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

    return novelty_score, support_score, combined_score, z_rye, z_delta


def _infer_tier_label(combined_score: float, z_rye: float) -> Optional[str]:
    """
    Rough Tier style label for discoveries.
    Uses combined_score and RYE z score to hint at Tier level.
    """
    # Strong RYE plus high combined score suggests higher Tier patterns
    if combined_score >= 0.9 and z_rye >= 1.0:
        return "tier3_candidate"
    if combined_score >= 0.8 and z_rye >= 0.3:
        return "tier2_candidate"
    if combined_score >= 0.65:
        return "tier1_candidate"
    return None


def _short_title_from_text(text: str, kind: str, domain: str) -> str:
    """
    Build a short title for auto created hypotheses.
    """
    t = " ".join(text.strip().split())
    if len(t) > 120:
        t = t[:117].rstrip() + "..."
    base = kind.replace("_", " ").title()
    dom = domain.split("/")[-1]
    return f"{base} candidate in {dom}: {t}"


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

    You can optionally pass:
        - hypothesis_manager for auto hypothesis creation
        - discovery_logger for Markdown discovery logging
        - intelligence_profile for adaptive thresholds
        - run_id for traceability
    """

    def __init__(
        self,
        memory_store: Any,
        presets: Optional[Dict[str, Any]] = None,
        hypothesis_manager: Optional[HypothesisManager] = None,
        discovery_logger: Optional[DiscoveryLogger] = None,
        intelligence_profile: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        domain: Optional[str] = None,
        runtime_profile: Optional[str] = None,
    ) -> None:
        self.memory_store = memory_store
        self.presets = presets or {}

        self.hypothesis_manager = hypothesis_manager
        self.discovery_logger = discovery_logger
        self.intelligence_profile = intelligence_profile or {}
        self.run_id = run_id
        self.domain_hint = domain
        self.runtime_profile = runtime_profile

    # --------------------------------------------------------
    # Threshold adaptation using intelligence profile
    # --------------------------------------------------------
    def _adapt_thresholds(
        self,
        min_combined_score: float,
        min_novelty_score: float,
        min_support_score: float,
    ) -> Tuple[float, float, float]:
        """
        Adapt thresholds based on intelligence_profile hints:
            - safety_bias
            - discovery_focus
            - tier3_bias
            - equilibrium_focus
        """
        profile = self.intelligence_profile or {}
        safety_bias = str(profile.get("safety_bias", "medium")).lower()
        discovery_focus = float(profile.get("discovery_focus", 0.5))
        tier3_bias = float(profile.get("tier3_bias", 0.0))
        equilibrium_focus = float(profile.get("equilibrium_focus", 0.5))

        mc = float(min_combined_score)
        mn = float(min_novelty_score)
        ms = float(min_support_score)

        # Safety high: require stronger evidence
        if safety_bias == "high":
            mc += 0.05
            mn += 0.03
            ms += 0.03
        elif safety_bias == "low":
            mc -= 0.04
            mn -= 0.02
            ms -= 0.02

        # Discovery focus: allow more borderline candidates
        if discovery_focus > 0.7:
            mc -= 0.03
            mn -= 0.02
        elif discovery_focus < 0.3:
            mc += 0.02

        # Tier3 bias: push for strongest discoveries, fewer but higher quality
        if tier3_bias > 0.6:
            mc += 0.02
            ms += 0.02

        # Equilibrium focus: slightly prefer well supported discoveries
        if equilibrium_focus > 0.6:
            ms += 0.02

        # Clamp
        mc = max(0.40, min(0.95, mc))
        mn = max(0.20, min(0.90, mn))
        ms = max(0.20, min(0.90, ms))

        return mc, mn, ms

    def _adapt_max_candidates(self, max_candidates_per_cycle: int) -> int:
        """
        Adapt per cycle candidate cap based on discovery_focus.
        """
        profile = self.intelligence_profile or {}
        discovery_focus = float(profile.get("discovery_focus", 0.5))

        cap = int(max_candidates_per_cycle)

        if discovery_focus > 0.8:
            cap = int(cap * 1.5)
        elif discovery_focus < 0.3:
            cap = max(1, int(cap * 0.6))

        # Keep sane bounds
        cap = max(1, min(20, cap))
        return cap

    # --------------------------------------------------------
    # Core pass
    # --------------------------------------------------------
    def run_pass(
        self,
        domains: Optional[List[str]] = None,
        min_combined_score: float = 0.55,
        min_novelty_score: float = 0.45,
        min_support_score: float = 0.45,
        max_candidates_per_cycle: int = 5,
        # New options
        create_hypotheses: bool = False,
        log_to_markdown: bool = True,
    ) -> Dict[str, Any]:
        """
        Scan full cycle history and append any new high quality discoveries.

        Returns a summary dict with counts and basic stats.

        Arguments:
            domains:
                Optional list of allowed domain labels to include.

            min_combined_score, min_novelty_score, min_support_score:
                Base thresholds that will be adapted using intelligence_profile
                if provided.

            max_candidates_per_cycle:
                Maximum number of discoveries per cycle before cutoff.
                Also adapted by discovery_focus.

            create_hypotheses:
                If True and a HypothesisManager is attached, spawn pending
                hypotheses for high tier discoveries.

            log_to_markdown:
                If True and a DiscoveryLogger is attached, mirror discoveries
                into the Markdown discovery log.
        """
        history: List[Dict[str, Any]] = []
        try:
            if hasattr(self.memory_store, "get_cycle_history"):
                history = self.memory_store.get_cycle_history() or []  # type: ignore[assignment]
        except Exception:
            history = []

        # Early exit
        if not history:
            return {
                "new_discoveries": 0,
                "total_discoveries": len(load_discovery_log()),
                "message": "No cycles in history. Nothing to analyze.",
            }

        # Compute baselines
        rye_vals = [
            float(e["RYE"])
            for e in history
            if isinstance(e.get("RYE"), (int, float))
        ]
        delta_vals = [
            float(e.get("delta_R", e.get("delta_r")))
            for e in history
            if isinstance(e.get("delta_R", e.get("delta_r")), (int, float))
        ]

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

        # Adapt thresholds and caps based on intelligence profile
        local_min_combined, local_min_novelty, local_min_support = self._adapt_thresholds(
            min_combined_score,
            min_novelty_score,
            min_support_score,
        )
        local_max_candidates = self._adapt_max_candidates(max_candidates_per_cycle)

        for entry in history:
            domain = str(entry.get("domain", self.domain_hint or "general"))
            if allowed_domains and domain.lower() not in allowed_domains:
                continue

            cycle_idx = entry.get("cycle")
            role = str(entry.get("role", "agent"))

            rye = float(entry["RYE"]) if isinstance(entry.get("RYE"), (int, float)) else None
            delta_raw = entry.get("delta_R", entry.get("delta_r"))
            delta_R = float(delta_raw) if isinstance(delta_raw, (int, float)) else None

            energy_raw = (
                entry.get("energy_E")
                or entry.get("energy_e")
                or entry.get("Energy")
            )
            energy_E = float(energy_raw) if isinstance(energy_raw, (int, float)) else None

            # Search and swarm metrics if present in cycle history
            info_gain = entry.get("info_gain")
            if info_gain is None:
                info_gain = entry.get("search_info_gain")

            search_energy = entry.get("search_energy")
            if search_energy is None:
                search_energy = entry.get("search_cost")

            semantic_diversity = entry.get("semantic_diversity")

            swarm_size = entry.get("swarm_size")
            swarm_config = entry.get("swarm_config")

            # Collect all candidate texts
            notes = entry.get("notes_added") or []
            repairs = entry.get("repairs") or []
            hyps = entry.get("hypotheses") or []

            candidates: List[Tuple[str, str]] = []

            for n in notes:
                txt = str(n).strip()
                # Skip internal/template artifacts
                if txt and not _contains_banned_pattern(txt):
                    candidates.append(("note", txt))

            for r in repairs:
                txt = str(r).strip()
                if txt and not _contains_banned_pattern(txt):
                    candidates.append(("repair", txt))

            for h in hyps:
                if isinstance(h, dict):
                    txt = str(h.get("text", "")).strip()
                else:
                    txt = str(h).strip()
                if txt and not _contains_banned_pattern(txt):
                    candidates.append(("hypothesis", txt))

            if not candidates:
                continue

            cycle_new: List[Discovery] = []

            for source_type, text in candidates:
                novelty_score, support_score, combined_score, z_rye_val, z_delta_val = _score_candidate(
                    text=text,
                    rye=rye,
                    delta_R=delta_R,
                    rye_mean=rye_mean,
                    rye_std=rye_std,
                    delta_mean=delta_mean,
                    delta_std=delta_std,
                    seen_fingerprints=existing_fp,
                )

                # Hard gates using adapted thresholds
                if combined_score < local_min_combined:
                    continue
                if novelty_score < local_min_novelty:
                    continue
                if support_score < local_min_support:
                    continue

                fingerprint = _text_fingerprint(text)
                if fingerprint in existing_fp:
                    continue

                kind, tags = _classify_kind(text, domain=domain)
                tier_label = _infer_tier_label(combined_score, z_rye_val)

                disc = Discovery(
                    id=f"{domain}-{cycle_idx}-{kind}-{len(cycle_new)}",
                    created_at=datetime.utcnow().isoformat() + "Z",
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
                    info_gain=info_gain if isinstance(info_gain, (int, float)) else None,
                    search_energy=search_energy if isinstance(search_energy, (int, float)) else None,
                    semantic_diversity=semantic_diversity if isinstance(semantic_diversity, (int, float)) else None,
                    swarm_size=int(swarm_size) if isinstance(swarm_size, int) else None,
                    swarm_config=swarm_config if isinstance(swarm_config, dict) else None,
                    tier_label=tier_label,
                    tags=tags,
                    extra={
                        "z_rye": z_rye_val,
                        "z_delta": z_delta_val,
                        "length": len(text),
                        "run_id": self.run_id,
                        "runtime_profile": self.runtime_profile,
                        "info_gain": info_gain,
                        "search_energy": search_energy,
                        "semantic_diversity": semantic_diversity,
                        "swarm_size": swarm_size,
                    },
                )
                cycle_new.append(disc)
                existing_fp.add(fingerprint)

                if len(cycle_new) >= local_max_candidates:
                    break

            new_entries.extend(cycle_new)

        if not new_entries:
            return {
                "new_discoveries": 0,
                "total_discoveries": len(existing),
                "message": "No new discoveries above thresholds.",
                "thresholds": {
                    "min_combined_score": local_min_combined,
                    "min_novelty_score": local_min_novelty,
                    "min_support_score": local_min_support,
                    "max_candidates_per_cycle": local_max_candidates,
                },
                "mean_rye": rye_mean,
                "std_rye": rye_std,
                "mean_delta_R": delta_mean,
                "std_delta_R": delta_std,
            }

        # Optional: auto create hypotheses for high tier discoveries
        created_hypotheses = 0
        if create_hypotheses and self.hypothesis_manager is not None:
            for disc in new_entries:
                if self._maybe_create_hypothesis_for_discovery(disc):
                    created_hypotheses += 1

        # Optional: mirror into Markdown discovery log
        if log_to_markdown and self.discovery_logger is not None:
            for disc in new_entries:
                self._log_discovery_markdown(disc)

        # Merge and save JSON discovery log
        merged = existing + [asdict(d) for d in new_entries]
        _save_discovery_log(merged)

        return {
            "new_discoveries": len(new_entries),
            "total_discoveries": len(merged),
            "created_hypotheses": created_hypotheses,
            "mean_rye": rye_mean,
            "std_rye": rye_std,
            "mean_delta_R": delta_mean,
            "std_delta_R": delta_std,
            "thresholds": {
                "min_combined_score": local_min_combined,
                "min_novelty_score": local_min_novelty,
                "min_support_score": local_min_support,
                "max_candidates_per_cycle": local_max_candidates,
            },
            "message": f"Discovery pass complete. Added {len(new_entries)} new discoveries.",
        }

    # --------------------------------------------------------
    # Optional integration hooks
    # --------------------------------------------------------
    def _maybe_create_hypothesis_for_discovery(self, disc: Discovery) -> bool:
        """
        Create a pending hypothesis for this discovery if it looks strong enough.
        Uses tier_label and combined_score to decide.
        """
        if self.hypothesis_manager is None:
            return False

        # Only create hypotheses for Tier 1 and above
        if disc.tier_label not in ("tier1_candidate", "tier2_candidate", "tier3_candidate"):
            return False

        # Basic description with metrics
        description_lines = [
            disc.text,
            "",
            "Metrics:",
            f"- domain: {disc.domain}",
            f"- kind: {disc.kind}",
            f"- source_type: {disc.source_type}",
            f"- RYE: {disc.rye}",
            f"- delta_R: {disc.delta_R}",
            f"- energy_E: {disc.energy_E}",
            f"- novelty_score: {disc.novelty_score:.3f}",
            f"- support_score: {disc.support_score:.3f}",
            f"- combined_score: {disc.combined_score:.3f}",
        ]
        if disc.tier_label:
            description_lines.append(f"- tier_label: {disc.tier_label}")
        if disc.info_gain is not None:
            description_lines.append(f"- info_gain: {disc.info_gain}")
        if disc.search_energy is not None:
            description_lines.append(f"- search_energy: {disc.search_energy}")
        if disc.semantic_diversity is not None:
            description_lines.append(f"- semantic_diversity: {disc.semantic_diversity}")
        if disc.swarm_size is not None:
            description_lines.append(f"- swarm_size: {disc.swarm_size}")

        description = "\n".join(description_lines)

        title = _short_title_from_text(
            text=disc.text,
            kind=disc.kind,
            domain=disc.domain,
        )

        tags = list(set(disc.tags + ["discovery_candidate", disc.tier_label or "discovery"]))

        try:
            self.hypothesis_manager.create_hypothesis(
                title=title,
                description=description,
                cycle_index=disc.cycle,
                agent_role=disc.role,
                rye_before=None,
                rye_after=disc.rye,
                delta_r=disc.delta_R,
                energy=disc.energy_E,
                tags=tags,
                info_gain=disc.info_gain,
                search_energy=disc.search_energy,
                semantic_diversity=disc.semantic_diversity,
                swarm_size=disc.swarm_size,
                swarm_config=disc.swarm_config,
            )
            return True
        except Exception:
            return False

    def _log_discovery_markdown(self, disc: Discovery) -> None:
        """
        Mirror discovery into the Markdown discovery log using DiscoveryLogger.
        """
        if self.discovery_logger is None:
            return

        tier_str = disc.tier_label or "unclassified"
        description_lines = [
            f"Discovery candidate with combined_score {disc.combined_score:.3f} and tier label {tier_str}.",
            "",
            "Text:",
            disc.text,
            "",
            "Metrics:",
            f"- RYE: {disc.rye}",
            f"- delta_R: {disc.delta_R}",
            f"- energy_E: {disc.energy_E}",
            f"- novelty_score: {disc.novelty_score:.3f}",
            f"- support_score: {disc.support_score:.3f}",
            f"- tier_label: {tier_str}",
        ]
        if disc.info_gain is not None:
            description_lines.append(f"- info_gain: {disc.info_gain}")
        if disc.search_energy is not None:
            description_lines.append(f"- search_energy: {disc.search_energy}")
        if disc.semantic_diversity is not None:
            description_lines.append(f"- semantic_diversity: {disc.semantic_diversity}")
        if disc.swarm_size is not None:
            description_lines.append(f"- swarm_size: {disc.swarm_size}")

        description = "\n".join(description_lines)

        tags = list(set(disc.tags + ["discovery", "discovery_candidate", tier_str]))

        try:
            self.discovery_logger.log_event(
                kind="discovery_candidate",
                title=f"[Cycle {disc.cycle}] {disc.kind} candidate in {disc.domain}",
                description=description,
                cycle_index=disc.cycle,
                agent_role=disc.role,
                rye_before=None,
                rye_after=disc.rye,
                delta_r=disc.delta_R,
                energy=disc.energy_E,
                tags=tags,
                extra={
                    "discovery_id": disc.id,
                    "tier_label": disc.tier_label,
                    "domain": disc.domain,
                    "role": disc.role,
                    "run_id": self.run_id,
                    "info_gain": disc.info_gain,
                    "search_energy": disc.search_energy,
                    "semantic_diversity": disc.semantic_diversity,
                    "swarm_size": disc.swarm_size,
                },
            )
        except Exception:
            # Do not let logging failures break discovery pass
            return


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

    Any extra kwargs are forwarded to DiscoveryEngine.run_pass, for example:
        - min_combined_score
        - min_novelty_score
        - min_support_score
        - max_candidates_per_cycle
        - create_hypotheses
        - log_to_markdown

    You can also pass constructor kwargs such as:
        - hypothesis_manager
        - discovery_logger
        - intelligence_profile
        - run_id
        - domain
        - runtime_profile
    """
    constructor_keys = {
        "hypothesis_manager",
        "discovery_logger",
        "intelligence_profile",
        "run_id",
        "domain",
        "runtime_profile",
    }

    ctor_kwargs: Dict[str, Any] = {}
    run_kwargs: Dict[str, Any] = {}

    for k, v in kwargs.items():
        if k in constructor_keys:
            ctor_kwargs[k] = v
        else:
            run_kwargs[k] = v

    engine = DiscoveryEngine(
        memory_store=memory_store,
        presets=presets,
        **ctor_kwargs,
    )
    return engine.run_pass(domains=domains, **run_kwargs)
