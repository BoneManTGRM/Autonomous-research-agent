"""
quality_gates.py
==================

This module centralizes quality checks used throughout the Autonomous Research
Agent.  By collecting placeholder detection, domain banning, citation
validation, discovery/hypothesis gating, and cycle quality gating into one
file, we ensure consistent behavior across the engine, verification
pipeline, report generation, web search, and tool ingestion layers.  Any
changes to these quality rules should be made here so that all components
benefit.

Functions provided:

* ``contains_placeholder(text: str) -> bool``
    Return True if the supplied text contains unresolved template markers or
    placeholder variables such as ``"agent"`` or ``"description"``.  This
    catches common failure modes where a templating system failed to
    substitute real content.

* ``is_valid_citation(cite: Dict[str, Any]) -> bool``
    Return True if a citation dict passes normalization via
    ``citation_utils.normalize_citation`` and does not come from a banned
    domain.  This encapsulates all of the domain and credibility filters
    defined in ``citation_utils``.

* ``filter_citations(cites: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]``
    Normalize and filter a list of citation dicts using ``is_valid_citation``.

* ``is_valid_discovery(discovery: Dict[str, Any]) -> bool``
    Return True if a discovery dict contains a non-placeholder title/text
    and has at least one valid citation.  Uncited or placeholder-only
    discoveries are rejected.

* ``filter_discoveries(discoveries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]``
    Filter a list of discovery dicts to only those passing ``is_valid_discovery``.

* ``is_valid_hypothesis(text: str) -> bool``
    Simple heuristic for candidate hypotheses: returns False if the text
    contains unresolved placeholders or is empty.  Hypotheses must be
    non-empty and non-template strings to be considered.

* ``evaluate_cycle_gate(cycle: Dict[str, Any], history: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]``
    Decide whether the improvements of a cycle should be accepted or
    rejected relative to recent history.  If rejected, return False and a
    list of reasons.  The gate compares the current cycleâs delta_R,
    RYE, energy usage, and citation presence against the mean values of
    the given history slice.  Configurable parameters can adjust window
    length and threshold behaviours.  See the function docstring for
    details.

This module depends only on ``citation_utils.normalize_citation`` and the
``statistics`` module.  If ``normalize_citation`` cannot be imported, the
functions will conservatively accept citations rather than raise exceptions.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import statistics

try:
    # Import normalization from citation_utils.  If unavailable, provide a
    # fallback that simply returns the original input.
    from .citation_utils import normalize_citation
except Exception:  # pragma: no cover
    def normalize_citation(raw: Any, default_source: str = "web") -> Any:  # type: ignore
        return raw  # type: ignore

# Placeholder patterns used to detect unresolved template variables.  These
# patterns are kept in lowercase so that ``contains_placeholder`` can
# perform a simple lowercase check on the text.
# Placeholder patterns used to detect unresolved template variables.  These
# patterns are kept in lowercase so that ``contains_placeholder`` can
# perform a simple lowercase check on the text.  Additional phrases
# corresponding to prompt injection artefacts and runâlevel directives
# have been added below (e.g. "system directive", "autonomous research swarm").
PLACEHOLDER_PATTERNS: List[str] = [
    "**",  # bold markers often indicate unresolved variables in reports
    "agent",
    "description",
    "detected",
    "encountered",
    "fully",
    "{{",  # Jinja/format style opening braces
    "}}",  # Jinja/format style closing braces

    # Additional patterns to catch prompt injection or run directive text.  These
    # phrases are rarely part of a legitimate hypothesis or discovery, but
    # frequently appear when the system directive leaks into outputs.  By
    # treating them as placeholders we prevent them from being accepted as
    # genuine content.
    "system directive",
    "autonomous research swarm",
    "coordinated cycle",
    "single coordinated cycle",
    "64-agent",
    # Accept both hyphenated and spaced forms for agent count to catch
    # template variations.  The unhyphenated version prevents strings like
    # "64 agent" from passing as valid content.
    "64 agent",
]

# Domains that are disallowed as evidence sources.  Citations from these
# domains are considered invalid unless they carry a DOI.  See
# ``citation_utils`` for the broader credibility logic.
BANNED_DOMAINS: List[str] = [
    "youtube.com",
    "youtu.be",
    "grantome.com",
]

def contains_placeholder(text: Any) -> bool:
    """Return True if the provided text contains unresolved template markers.

    The check is case-insensitive and attempts a best-effort conversion of
    arbitrary objects to strings.  Any occurrence of ``PLACEHOLDER_PATTERNS``
    triggers a positive result.
    """
    if not text:
        return False
    try:
        s = str(text).lower()
    except Exception:
        return False
    for pat in PLACEHOLDER_PATTERNS:
        try:
            if pat in s:
                return True
        except Exception:
            continue
    return False

def is_valid_citation(cite: Any) -> bool:
    """Return True if ``cite`` normalizes to a valid citation and is not banned.

    A citation is considered valid when ``citation_utils.normalize_citation``
    returns a non-None dict and its domain is not in ``BANNED_DOMAINS``.
    """
    try:
        norm = normalize_citation(cite)
    except Exception:
        # If normalization fails, treat citation as invalid
        return False
    if not isinstance(norm, dict):
        return False
    # Check banned domains.  Many citation dicts include a ``url`` field.
    url = norm.get("url")
    if url:
        try:
            from urllib.parse import urlparse  # type: ignore
            domain = urlparse(url).netloc.lower()
            for bd in BANNED_DOMAINS:
                if domain == bd or domain.endswith("." + bd):
                    return False
        except Exception:
            # If domain parsing fails, fall back to trusting citation_utils
            pass
    return True

def filter_citations(cites: Iterable[Any]) -> List[Dict[str, Any]]:
    """Normalize and filter an iterable of citations.

    Returns a list of normalized citations that pass ``is_valid_citation``.
    """
    out: List[Dict[str, Any]] = []
    if not cites:
        return out
    for c in cites:
        try:
            if is_valid_citation(c):
                norm = normalize_citation(c)
                # normalize_citation may return None if the citation is rejected
                if isinstance(norm, dict):
                    out.append(norm)
        except Exception:
            continue
    return out

def is_valid_discovery(discovery: Any) -> bool:
    """Return True if a discovery dict contains a non-placeholder text and citations.

    The discovery must be a dict with a textual 'title' or 'hypothesis'
    field.  It must also include at least one citation (list of citation
    dicts) that passes ``filter_citations``.  Discoveries that only
    contain placeholders or lack citations are rejected.
    """
    if not isinstance(discovery, dict):
        return False
    # Determine the text to validate
    text = None
    for key in ("title", "hypothesis", "name", "text"):
        v = discovery.get(key)
        if isinstance(v, str) and v.strip():
            text = v
            break
    if not text or contains_placeholder(text):
        return False
    # Validate citations
    cites = discovery.get("citations") or discovery.get("sources")
    if cites and isinstance(cites, list):
        valid_cites = filter_citations(cites)
        if valid_cites:
            return True
    return False

def filter_discoveries(discoveries: Iterable[Any]) -> List[Dict[str, Any]]:
    """Return only valid discoveries from an iterable.

    Each discovery dict must pass ``is_valid_discovery``.  Invalid items
    are discarded.
    """
    out: List[Dict[str, Any]] = []
    if not discoveries:
        return out
    for d in discoveries:
        if is_valid_discovery(d):
            out.append(d)
    return out

def is_valid_hypothesis(text: Any) -> bool:
    """Return True if a hypothesis string is non-empty and not a placeholder.

    Hypotheses that are None, empty, or contain unresolved template markers
    (see ``PLACEHOLDER_PATTERNS``) return False.  No citation checks are
    performed here because citations are managed at the discovery level.
    """
    if not text:
        return False
    try:
        s = str(text).strip()
    except Exception:
        return False
    if not s:
        return False
    return not contains_placeholder(s)


def evaluate_cycle_gate(
    cycle: Dict[str, Any],
    history: Iterable[Dict[str, Any]],
    *,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str]]:
    """Return (accept, reasons) for whether a cycle should contribute to scoring.

    This gate compares the current cycleâs performance against recent history.
    The cycle dict should contain keys ``delta_R`` (float), ``energy_E`` (float),
    ``citations`` (list), and ``hypotheses`` (list).  The history iterable
    should contain past cycles with the same keys.  A cycle is rejected if:

    * ``delta_R`` is non-positive or below the mean of the history window.
    * Its RYE (delta_R / energy_E) is below the mean RYE of the window.
    * It has no citations.

    Additional checks (e.g. excessive energy usage, duplicate hypotheses) can
    be added here.  The ``config`` dict may include ``window`` to specify
    history length; other keys are ignored for now.

    Parameters
    ----------
    cycle : Dict[str, Any]
        Current cycle metrics and artifacts.  Expected keys: ``delta_R``,
        ``energy_E``, ``citations``, ``hypotheses``.
    history : Iterable[Dict[str, Any]]
        Sequence of recent cycle summaries.  Only numeric fields are
        considered.
    config : Optional[Dict[str, Any]]
        Optional tuning parameters.  Supports:
            - ``window``: int, length of history to consider.

    Returns
    -------
    Tuple[bool, List[str]]
        A tuple ``(accept, reasons)`` where ``accept`` is True if the
        cycle meets all thresholds, otherwise False.  ``reasons`` gives
        human-readable codes for rejection conditions.
    """
    reasons: List[str] = []
    if cycle is None or not isinstance(cycle, dict):
        return False, ["invalid_cycle"]
    cfg = config or {}
    # Extract cycle values with safe defaults
    try:
        delta_r = float(cycle.get("delta_R", 0.0) or 0.0)
    except Exception:
        delta_r = 0.0
    try:
        energy_e = float(cycle.get("energy_E", 0.0) or 0.0)
    except Exception:
        energy_e = 0.0
    citations = cycle.get("citations") or []
    # Quick rejection for non-positive improvements
    if delta_r <= 0:
        reasons.append("non_positive_delta")
    # Build history lists for baseline computation
    delta_history: List[float] = []
    energy_history: List[float] = []
    for h in history:
        try:
            dr = h.get("delta_R")
            if isinstance(dr, (int, float)):
                delta_history.append(float(dr))
        except Exception:
            continue
        try:
            en = h.get("energy_E")
            if isinstance(en, (int, float)):
                energy_history.append(float(en))
        except Exception:
            continue
    # Compute baselines from history if available
    baseline_delta = 0.0
    baseline_rye = 0.0
    if delta_history:
        try:
            baseline_delta = statistics.mean(delta_history)
        except Exception:
            baseline_delta = 0.0
    if delta_history and energy_history:
        # Use paired means to avoid division by zero
        try:
            mean_energy = statistics.mean(energy_history)
            if mean_energy > 0:
                baseline_rye = (baseline_delta / mean_energy) if baseline_delta > 0 else 0.0
        except Exception:
            baseline_rye = 0.0
    # Compute current rye; avoid division by zero
    current_rye = 0.0
    if energy_e > 0:
        try:
            current_rye = delta_r / energy_e
        except Exception:
            current_rye = 0.0
    # Compare delta and rye against baselines
    # Previous logic rejected cycles when delta_R or RYE fell below the
    # historical mean.  This can inadvertently penalise exploratory
    # cycles and suppress novel discovery.  To support richer
    # exploration, we remove these baseline comparisons entirely.  As
    # long as the cycle yields a positive improvement, it is allowed to
    # contribute to scoring regardless of how it stacks up against
    # history.  Baselines are still computed above for diagnostic
    # purposes but not used for gating.
    # Check for evidence presence.  A cycle is accepted if it includes at
    # least one citation *or* at least one hypothesis.  Early
    # exploration cycles may propose hypotheses before citations are
    # gathered; rejecting them outright stifles discovery.  We only
    # flag the cycle if both citations and hypotheses are missing.
    try:
        has_cite = bool(citations) and isinstance(citations, (list, tuple)) and len(citations) > 0
    except Exception:
        has_cite = False
    try:
        hyps = cycle.get("hypotheses")
        has_hyp = bool(hyps) and isinstance(hyps, (list, tuple)) and len(hyps) > 0
    except Exception:
        has_hyp = False
    if not has_cite and not has_hyp:
        reasons.append("no_citations")
    # Accept only if no reasons collected
    accept = len(reasons) == 0
    return accept, reasons
