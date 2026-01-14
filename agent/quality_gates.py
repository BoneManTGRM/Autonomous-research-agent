"""
quality_gates.py
==================

This module centralizes quality checks used throughout the Autonomous Research
Agent.  By collecting placeholder detection, domain banning, citation
validation, and discovery/hypothesis gating into one file, we ensure
consistent behavior across the engine, verification pipeline, report
generation, web search, and tool ingestion layers.  Any changes to these
quality rules should be made here so that all components benefit.

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

This module depends only on ``citation_utils.normalize_citation``.  If
``normalize_citation`` cannot be imported, the functions will conservatively
accept citations rather than raise exceptions.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

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
PLACEHOLDER_PATTERNS: List[str] = [
    "**",  # bold markers often indicate unresolved variables in reports
    "agent",
    "description",
    "detected",
    "encountered",
    "fully",
    "{{",  # Jinja/format style opening braces
    "}}",  # Jinja/format style closing braces
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
