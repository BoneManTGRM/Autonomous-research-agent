"""Citation utilities for the Autonomous Research Agent.

This module centralizes all logic for:
- Normalizing raw citation objects from tools or memory.
- Extracting citations from cycles, tool_events, and memory stores.
- Merging, deduplicating, and trimming citation lists.
- Building human readable bibliographies for reports.

Design goals:
- Be extremely defensive about incoming data.
- Never crash if a tool returns unexpected shapes.
- Preserve as much provenance as possible (source, url, doi, etc).
- Be backwards compatible with any existing "citations" field that
  already uses simple dicts like:
      {"source": "web", "title": "...", "url": "..."}

Typical usage patterns:

1. From a tool handler or TGRM loop:

    from .citation_utils import extract_citations_from_tool_events, merge_citation_lists

    citations = extract_citations_from_tool_events(tool_events)
    cycle["citations"] = merge_citation_lists(cycle.get("citations") or [], citations)

2. From reporting code:

    from .citation_utils import build_bibliography_markdown

    bib_md = build_bibliography_markdown(all_citations, max_items=50)

3. From MemoryStore centric reporting:

    from .citation_utils import extract_citations_from_memory_store

    cites = extract_citations_from_memory_store(memory_store, goal="longevity")

All public functions are safe to call with None or unexpected types.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_str(value: Any) -> str:
    """Best effort string conversion that never raises."""
    try:
        if value is None:
            return ""
        s = str(value)
        # Repair common mojibake (UTF-8 bytes that were decoded as cp1252/latin1).
        # Safe to call on any string; will no-op if no marker characters exist.
        try:
            s = normalize_text(s)
        except Exception:
            pass
        return s.strip()
    except Exception:
        return ""


def _safe_float(value: Any) -> Optional[float]:
    """Best effort float conversion with None on failure."""
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return float(str(value))
    except Exception:
        return None


def _first_non_empty(*values: Any) -> Optional[str]:
    """Return first non empty string representation among provided values."""
    for v in values:
        s = _safe_str(v)
        if s:
            return s
    return None


def _normalize_author_list(value: Any) -> Optional[str]:
    """Normalize various author representations into a compact string.

    Accepts:
        - List of strings.
        - List of dicts with "name" or "full_name".
        - Comma separated string.

    Returns a compact author string, truncated if extremely long.
    """
    if value is None:
        return None

    # Already a simple string
    if isinstance(value, str):
        text = _safe_str(value)
        if not text:
            return None
        if len(text) > 300:
            text = text[:297] + "..."
        return text

    names: List[str] = []

    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, dict):
                name = _first_non_empty(
                    item.get("name"),
                    item.get("full_name"),
                    item.get("author"),
                )
                if name:
                    names.append(name)
            else:
                s = _safe_str(item)
                if s:
                    names.append(s)

    if not names:
        return None

    joined = "; ".join(names)
    if len(joined) > 300:
        joined = joined[:297] + "..."
    return joined


def _normalize_year(meta: Dict[str, Any]) -> Optional[str]:
    """Try to extract a publication year from any plausible field."""
    for key in ("year", "publication_year", "pub_year", "date", "published"):
        val = meta.get(key)
        if isinstance(val, int):
            if 1800 <= val <= 2100:
                return str(val)
        s = _safe_str(val)
        if not s:
            continue
        # Try simple year scan
        for token in s.replace("/", " ").replace("-", " ").split():
            if token.isdigit() and len(token) == 4:
                try:
                    num = int(token)
                except Exception:
                    continue
                if 1800 <= num <= 2100:
                    return str(num)
    return None


def _normalize_doi(meta: Dict[str, Any]) -> Optional[str]:
    """Extract a DOI-like string if present."""
    candidates = [
        meta.get("doi"),
        meta.get("DOI"),
        meta.get("identifier"),
        meta.get("id"),
    ]
    for cand in candidates:
        s = _safe_str(cand)
        if not s:
            continue
        if "10." in s:
            return s
    return None


def _normalize_url(meta: Dict[str, Any]) -> Optional[str]:
    """Extract url or link field from mixed metadata."""
    candidates = [
        meta.get("url"),
        meta.get("link"),
        meta.get("href"),
        meta.get("pdf_url"),
        meta.get("source_url"),
    ]
    for cand in candidates:
        s = _safe_str(cand)
        if s.startswith("http://") or s.startswith("https://"):
            return s
    return None


def _normalize_source(meta: Dict[str, Any], default: str = "web") -> str:
    """Normalize a provider/source label like 'tavily', 'pubmed', etc."""
    for key in ("source", "provider", "engine", "origin"):
        s = _safe_str(meta.get(key))
        if s:
            return s.lower()
    return default


def _normalize_title(meta: Dict[str, Any]) -> Optional[str]:
    """Extract a reasonable title string."""
    candidates = [
        meta.get("title"),
        meta.get("headline"),
        meta.get("name"),
        meta.get("paper_title"),
    ]
    for cand in candidates:
        s = _safe_str(cand)
        if s:
            if len(s) > 400:
                s = s[:397] + "..."
            return s
    return None


def _normalize_venue(meta: Dict[str, Any]) -> Optional[str]:
    """Extract journal, conference, or site host if present."""
    candidates = [
        meta.get("journal"),
        meta.get("venue"),
        meta.get("conference"),
        meta.get("host"),
        meta.get("publisher"),
        meta.get("site"),
    ]
    for cand in candidates:
        s = _safe_str(cand)
        if s:
            if len(s) > 200:
                s = s[:197] + "..."
            return s
    return None


def _normalize_snippet(meta: Dict[str, Any]) -> Optional[str]:
    """Extract a short descriptive snippet or abstract."""
    candidates = [
        meta.get("snippet"),
        meta.get("summary"),
        meta.get("description"),
        meta.get("abstract"),
        meta.get("content"),
    ]
    for cand in candidates:
        s = _safe_str(cand)
        if s:
            if len(s) > 500:
                s = s[:497] + "..."
            return s
    return None


def _normalize_score(meta: Dict[str, Any]) -> Optional[float]:
    """Normalize a relevance/score field if available."""
    for key in ("score", "relevance", "confidence", "weight"):
        if key in meta:
            val = _safe_float(meta.get(key))
            if val is not None:
                return val
    return None


# ---------------------------------------------------------------------------
# Public normalization API
# ---------------------------------------------------------------------------


def normalize_citation(raw: Any, default_source: str = "web") -> Optional[Dict[str, Any]]:
    """Normalize a raw citation-like object into a standard dict.

    Input can be:
        - Already normalized dict with "title" and "url".
        - Dict from Tavily, PubMed, Semantic Scholar, or custom tools.
        - Generic dict with "metadata" or "raw" nested blobs.
        - Any other type, which will be ignored (return None).

    Output schema (all fields optional except 'source'):

        {
            "source": str,         # tavily, pubmed, semantic_scholar, web, file, sandbox, etc
            "title": str | None,
            "url": str | None,
            "doi": str | None,
            "authors": str | None,
            "venue": str | None,
            "year": str | None,
            "snippet": str | None,
            "score": float | None,
            "raw": dict            # original dict (best effort)
        }
    """
    if not isinstance(raw, dict):
        return None

    # Flatten obvious nested fields so the normalizers can see them
    meta: Dict[str, Any] = dict(raw)
    for nested_key in ("metadata", "raw", "extra", "info", "paper"):
        nested = meta.get(nested_key)
        if isinstance(nested, dict):
            for k, v in nested.items():
                meta.setdefault(k, v)

    source = _normalize_source(meta, default=default_source)
    title = _normalize_title(meta)
    url = _normalize_url(meta)
    doi = _normalize_doi(meta)
    authors = _normalize_author_list(
        meta.get("authors")
        or meta.get("author")
        or meta.get("creators")
    )
    venue = _normalize_venue(meta)
    year = _normalize_year(meta)
    snippet = _normalize_snippet(meta)
    score = _normalize_score(meta)

    # -----------------------------------------------------------------
    # Stub and placeholder filtering
    #
    # Remove citations that are clearly non-substantive. These may arise
    # from errors (e.g. API failures), empty search results, or generic
    # placeholder messages like "No Semantic Scholar results found".  Such
    # entries dilute report quality and should not count toward evidence.
    try:
        t_low = (title or "").strip().lower()
    except Exception:
        t_low = ""
    try:
        s_low = (snippet or "").strip().lower()
    except Exception:
        s_low = ""
    # Drop titles that start with [STUB] or indicate no results
    if t_low:
        if t_low.startswith("[stub]"):
            return None
        if t_low.startswith("no title"):
            return None
        if "no results" in t_low:
            return None
        if "no semantic scholar" in t_low:
            return None
        if "error" in t_low and ("semantic scholar" in t_low or "pubmed" in t_low or "api" in t_low):
            return None
    if s_low:
        if "semantic scholar error" in s_low or "pubmed request failed" in s_low:
            return None

    # -----------------------------------------------------------------
    # Credibility filtering
    #
    # Only include citations from trusted sources or those with a DOI.  The
    # original implementation accepted anything with a title or URL which
    # resulted in mixedâquality sources leaking into reports.  To improve
    # citation quality and relevance we drop entries from unknown sources
    # unless a DOI is present.  Known credible sources include:
    #   - pubmed, semantic_scholar, tavily (scientific search)
    #   - papers (internal PDF ingestion)
    #   - file/sandbox/pdf (local files)
    #
    # Any citation coming from a generic web search without a DOI is
    # discarded.  This prevents lowâquality blogs or anonymous pages from
    # diluting the bibliography.

    def _is_credible_source(src: str) -> bool:
        if not src:
            return False
        src_l = src.lower()
        credible = {
            "pubmed",
            "semantic_scholar",
            "papers",
            "file",
            "sandbox",
            "pdf",
            "tavily",
        }
        # Accept Tavily only if the caller wants general search.  The RYE
        # pipeline filters these further during verification.  Without a
        # DOI we mark Tavily as less credible and will drop below.
        return src_l in credible

    # Drop tool/API error messages that can accidentally be recorded as citations.
    # (Common when a provider quota is exceeded, e.g. Tavily.)
    try:
        blob = f"{title or ''} {snippet or ''}".strip().lower()
    except Exception:
        blob = ""

    if blob and not (url or doi):
        src_l = _safe_str(source).lower()
        quota_err = (
            "tavily api error" in blob
            or "exceeds your plan" in blob
            or "set usage limit" in blob
            or "request exceeds" in blob
        )
        generic_api_err = ("api error" in blob or "rate limit" in blob) and len(blob) < 300

        # If it looks like a provider error and there's no URL/DOI, it's not a real citation.
        if quota_err:
            return None
        if src_l == "tavily" and "error" in blob:
            return None
        if generic_api_err:
            return None

    # Drop untrusted sources without DOI.  This must happen before
    # constructing the final dict to avoid including lowâquality citations
    # downstream.  Without this filter, references from random web pages
    # (source "web" with no DOI) clutter reports and reduce relevance.
    if not doi:
        # If the source is not in our credible list, consult domain whitelist.
        # This allows keeping high-quality publishers even when a DOI is missing.
        # Known credible domains include NIH/NCBI, major journals (Nature, Science,
        # Cell, NEJM, Lancet, BMJ), and large open-access publishers.
        credible_domains = {
            "nih.gov",
            "ncbi.nlm.nih.gov",
            "nature.com",
            "sciencemag.org",
            "science.org",
            "cell.com",
            "nejm.org",
            "thelancet.com",
            "lancet.com",
            "bmj.com",
            "plos.org",
            "frontiersin.org",
        }
        from urllib.parse import urlparse  # type: ignore
        domain = ""
        if url:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower()
            except Exception:
                domain = ""
        if not _is_credible_source(source):
            # Keep only if domain ends with one of the credible domains
            if not (domain and any(domain.endswith(d) for d in credible_domains)):
                return None

    # If we still have nothing meaningful, drop it unless there is at least URL or DOI
    if not any([title, url, doi, snippet]):
        if url or doi:
            return {
                "source": source,
                "title": title or url or doi or "Untitled",
                "url": url,
                "doi": doi,
                "authors": authors,
                "venue": venue,
                "year": year,
                "snippet": snippet,
                "score": score,
                "raw": raw,
            }
        return None

    return {
        "source": source,
        "title": title or "Untitled",
        "url": url,
        "doi": doi,
        "authors": authors,
        "venue": venue,
        "year": year,
        "snippet": snippet,
        "score": score,
        "raw": raw,
    }


def normalize_citation_list(
    items: Optional[Iterable[Any]],
    default_source: str = "web",
) -> List[Dict[str, Any]]:
    """Normalize many raw citation objects in one call."""
    if items is None:
        return []
    out: List[Dict[str, Any]] = []
    for item in items:
        norm = normalize_citation(item, default_source=default_source)
        if norm is not None:
            out.append(norm)
    return out

# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Repair common mojibake sequences.

    The most common failure mode we see in citations and report text is:
    UTF-8 bytes being decoded as cp1252 or latin1, producing marker characters like
    "Ã", "Ã¢", "Ã" (e.g. "ÃÂ·" instead of "Â·", or "Ã¢â¬Â¢" instead of "â¢").

    This function performs a best-effort reverse transform:
      1) Try cp1252 -> utf-8
      2) Try latin1 -> utf-8

    It only attempts repairs when marker characters are present, and only keeps a
    candidate repair when it reduces the number of marker characters.
    """
    if not isinstance(text, str):
        return text

    markers = ("Ã", "Ã¢", "Ã", "\uFFFD")
    if not any(m in text for m in markers):
        return text

    def _count_markers(s: str) -> int:
        return sum(s.count(m) for m in markers)

    best = text
    best_count = _count_markers(text)

    for enc in ("cp1252", "latin1"):
        try:
            fixed = text.encode(enc).decode("utf-8")
        except Exception:
            continue
        cnt = _count_markers(fixed)
        if fixed and cnt < best_count:
            best = fixed
            best_count = cnt

    # Extra micro-fixes for a few stubborn sequences we sometimes see after partial repairs.
    if "Ã¢â¬Â¢" in best:
        best = best.replace("Ã¢â¬Â¢", "â¢")
    # Normalize middle-dot mojibake: replace the split sequence "ÃÂ·" with a plain
    # middle dot, then replace all remaining middle dots with hyphens.  This
    # prevents re-encoding issues where UTF-8 middle dots become the two
    # characters "Ã" and "Â·" when decoded as latin1.
    if "ÃÂ·" in best:
        best = best.replace("ÃÂ·", "Â·")
    if "Â·" in best:
        best = best.replace("Â·", "-")

    # Additional micro-fixes for other two-byte mojibake patterns.  In some
    # citations we observe sequences like "ÃÂ¢" or "ÃÂ¤" which are typically
    # remnants of mis-decoded UTFâ8 punctuation (for example bullet or dot
    # characters).  Replace these and their single-character forms with
    # hyphens to improve readability.  These currency symbols rarely occur
    # legitimately in citation metadata, so this is a reasonable tradeoff.
    if "ÃÂ¢" in best:
        best = best.replace("ÃÂ¢", "-")
    if "Â¢" in best:
        best = best.replace("Â¢", "-")
    if "ÃÂ¤" in best:
        best = best.replace("ÃÂ¤", "-")
    if "Â¤" in best:
        best = best.replace("Â¤", "-")
    return best


# ---------------------------------------------------------------------------
# Extraction from cycles, tool events, and memory store
# ---------------------------------------------------------------------------


def extract_citations_from_tool_events(
    tool_events: Optional[Iterable[Dict[str, Any]]],
    default_source: str = "web",
) -> List[Dict[str, Any]]:
    """Extract citations from generic tool_events.

    This is intentionally forgiving and tries a few patterns that are common in
    autonomous agents, for example:

        event = {
            "tool": "tavily_search",
            "results": [...],        # list of search result dicts
        }

        event = {
            "tool": "pubmed_query",
            "papers": [...],
        }

        event = {
            "tool": "semantic_scholar",
            "data": {
                "papers": [...]
            }
        }

        event = {
            "tool": "file_reader",
            "payload": {
                "documents": [...]
            }
        }

    The function walks common containers and normalizes each candidate.
    """
    if not tool_events:
        return []

    collected: List[Dict[str, Any]] = []

    for ev in tool_events:
        if not isinstance(ev, dict):
            continue

        tool_name = _safe_str(ev.get("tool") or ev.get("name") or ev.get("kind"))
        # Guess a default source from tool name
        if tool_name:
            lname = tool_name.lower()
            if "pubmed" in lname:
                local_source = "pubmed"
            elif "semantic" in lname:
                local_source = "semantic_scholar"
            elif "tavily" in lname:
                local_source = "tavily"
            elif "file" in lname or "pdf" in lname:
                local_source = "file"
            else:
                local_source = default_source
        else:
            local_source = default_source

        # First, if event already has a "citations" field, trust that
        if isinstance(ev.get("citations"), (list, tuple)):
            collected.extend(
                normalize_citation_list(ev.get("citations"), default_source=local_source)
            )

        candidate_containers: List[Any] = []

        # Direct containers
        for key in ("results", "papers", "items", "data", "documents", "hits", "output", "response"):
            if key in ev:
                candidate_containers.append(ev[key])

        # Some tools tuck everything inside "payload"
        payload = ev.get("payload")
        if isinstance(payload, dict):
            for key in ("results", "papers", "items", "data", "documents", "hits", "output", "response"):
                if key in payload:
                    candidate_containers.append(payload[key])

        # In some tools the result list is directly in "data" as nested lists
        for container in candidate_containers:
            if isinstance(container, dict):
                # Maybe deeper nested list
                for v in container.values():
                    if isinstance(v, (list, tuple)):
                        for entry in v:
                            norm = normalize_citation(entry, default_source=local_source)
                            if norm is not None:
                                collected.append(norm)
            elif isinstance(container, (list, tuple)):
                for entry in container:
                    norm = normalize_citation(entry, default_source=local_source)
                    if norm is not None:
                        collected.append(norm)

    return collected


def extract_citations_from_cycle(
    cycle: Dict[str, Any],
    default_source: str = "web",
) -> List[Dict[str, Any]]:
    """Extract and normalize citations from a single cycle dict.

    Sources:
        - cycle["citations"] if already present.
        - cycle["tool_events"] if available (using extract_citations_from_tool_events).
    """
    if not isinstance(cycle, dict):
        return []

    out: List[Dict[str, Any]] = []

    # Pre-attached citations
    pre = cycle.get("citations")
    if pre:
        out.extend(normalize_citation_list(pre, default_source=default_source))

    # Tool events
    tool_events = cycle.get("tool_events")
    if tool_events:
        out.extend(extract_citations_from_tool_events(tool_events, default_source=default_source))

    return out


def extract_citations_from_cycles(
    cycles: Sequence[Dict[str, Any]],
    default_source: str = "web",
) -> List[Dict[str, Any]]:
    """Collect citations across many cycles and normalize them."""
    all_cites: List[Dict[str, Any]] = []
    for c in cycles:
        all_cites.extend(extract_citations_from_cycle(c, default_source=default_source))
    return all_cites


def extract_citations_from_memory_store(
    memory_store: Any,
    *,
    goal: Optional[str] = None,
    run_id: Optional[str] = None,
    limit: int = 1000,
    default_source: str = "web",
) -> List[Dict[str, Any]]:
    """High level helper to pull citations directly from a MemoryStore-like object.

    It is defensive and works with:
        - memory_store.get_citations(goal=...)
        - memory_store.get_all_citations()

    Args:
        memory_store: Memory store instance (or compatible object).
        goal: Optional goal filter if supported by get_citations.
        run_id: Optional run filter (applied after retrieval if present).
        limit: Maximum number of citations after normalization and dedup.
        default_source: Fallback source label.

    Returns:
        A normalized, deduplicated list of citation dicts.
    """
    try:
        base: List[Dict[str, Any]] = []

        if hasattr(memory_store, "get_citations"):
            # MemoryStore.get_citations(goal=None) returns all citations
            if goal is not None:
                raw = memory_store.get_citations(goal=goal)
            else:
                raw = memory_store.get_citations(goal=None)
            if isinstance(raw, list):
                base = raw
        elif hasattr(memory_store, "get_all_citations"):
            raw = memory_store.get_all_citations()
            if isinstance(raw, list):
                base = raw
        else:
            return []

        # Optional run_id filter
        if run_id is not None:
            base = [c for c in base if isinstance(c, dict) and c.get("run_id") == run_id]

        norm = normalize_citation_list(base, default_source=default_source)

        # Dedup and trim
        seen = set()
        unique: List[Dict[str, Any]] = []
        for c in norm:
            key = _citation_key(c)
            if key in seen:
                continue
            seen.add(key)
            unique.append(c)
            if len(unique) >= limit:
                break
        return unique
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Merging, deduplication, and limiting
# ---------------------------------------------------------------------------


def _citation_key(c: Dict[str, Any]) -> Tuple[str, str, str]:
    """Stable deduplication key (doi, url, title)."""
    doi = _safe_str(c.get("doi")).lower()
    url = _safe_str(c.get("url")).lower()
    title = _safe_str(c.get("title")).lower()
    return (doi, url, title)


def merge_citation_lists(
    primary: Optional[Iterable[Dict[str, Any]]],
    secondary: Optional[Iterable[Dict[str, Any]]],
    max_items: int = 200,
) -> List[Dict[str, Any]]:
    """Merge two citation lists, normalize them, and dedupe.

    Items from 'primary' are preferred over 'secondary' when duplicates occur.
    """
    norm_primary = normalize_citation_list(primary or [])
    norm_secondary = normalize_citation_list(secondary or [])

    seen = set()
    merged: List[Dict[str, Any]] = []

    # Preserve order: primary first, then secondary
    for src_list in (norm_primary, norm_secondary):
        for c in src_list:
            key = _citation_key(c)
            if key in seen:
                continue
            seen.add(key)
            merged.append(c)
            if len(merged) >= max_items:
                return merged

    return merged


# ---------------------------------------------------------------------------
# Bibliography and display helpers
# ---------------------------------------------------------------------------


def format_citation_markdown(c: Dict[str, Any]) -> str:
    """Format a single normalized citation as a short markdown line.

    Example:

        **[pubmed]** Title of the paper (Smith et al., 2023)
           - https://...

    If there is no url, only the text part is returned.
    """
    src = _safe_str(c.get("source") or "web")
    title = _safe_str(c.get("title") or "Untitled")
    authors = _safe_str(c.get("authors"))
    year = _safe_str(c.get("year"))
    venue = _safe_str(c.get("venue"))
    url = _safe_str(c.get("url"))
    doi = _safe_str(c.get("doi"))

    pieces: List[str] = []
    pieces.append(f"**[{src}]** {title}")

    meta_parts: List[str] = []
    if authors:
        meta_parts.append(authors)
    if year:
        meta_parts.append(year)
    if venue:
        meta_parts.append(venue)
    if doi:
        meta_parts.append(f"DOI: {doi}")

    if meta_parts:
        pieces.append(f"({'; '.join(meta_parts)})")

    line = " ".join(pieces)

    if url:
        return f"{line}\n   - {url}"
    return line


def build_bibliography_markdown(
    citations: Optional[Iterable[Dict[str, Any]]],
    max_items: int = 50,
) -> str:
    """Build a markdown section listing unique citations.

    This is primarily intended for use in reports. It performs normalization
    and deduplication again for safety.
    """
    if not citations:
        return "No citations recorded."

    norm = normalize_citation_list(citations)
    # Simple dedup for safety
    seen = set()
    unique: List[Dict[str, Any]] = []
    for c in norm:
        key = _citation_key(c)
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
        if len(unique) >= max_items:
            break

    lines: List[str] = []
    for i, c in enumerate(unique, start=1):
        lines.append(f"{i}. {format_citation_markdown(c)}")

    if len(norm) > max_items:
        lines.append(f"... and {len(norm) - max_items} more.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cycle and memory_store attachment helpers
# ---------------------------------------------------------------------------


def attach_citations_to_cycle(
    cycle: Dict[str, Any],
    new_citations: Iterable[Dict[str, Any]],
    max_items: int = 200,
) -> None:
    """Merge new citations into a cycle in place.

    The cycle dict is updated but nothing is returned for ergonomic use in
    the TGRM loop.
    """
    existing = cycle.get("citations") or []
    merged = merge_citation_lists(existing, new_citations, max_items=max_items)
    cycle["citations"] = merged


def attach_citations_to_memory_store(
    memory_store: Any,
    cycle_index: Optional[int],
    citations: Iterable[Dict[str, Any]],
    max_items: int = 200,
) -> None:
    """Attach citations to a cycle and optionally persist them in a MemoryStore.

    Behavior:
        - If memory_store exposes get_cycle_history, the target cycle is
          selected by index (or the last one if index is missing).
        - Citations are merged into that cycle in memory.
        - If memory_store exposes update_cycle(idx, cycle) or
          save_cycle_history(history), those are used to persist the
          updated cycle list.
        - If memory_store exposes add_citation(...), normalized citations
          are also written into the top level citation log using the
          cycle goal/domain/run metadata where available.

    This function is fully defensive: if anything is missing or unexpected,
    it simply returns without raising.
    """
    try:
        if not hasattr(memory_store, "get_cycle_history"):
            return
        history = list(memory_store.get_cycle_history())
        if not history:
            return

        if cycle_index is None or cycle_index < 0 or cycle_index >= len(history):
            idx = len(history) - 1
        else:
            idx = cycle_index

        cycle = history[idx]
        if not isinstance(cycle, dict):
            return

        # Merge into the in memory cycle object
        attach_citations_to_cycle(cycle, citations, max_items=max_items)

        # Persist back to the memory store if it supports cycle mutation
        if hasattr(memory_store, "update_cycle"):
            memory_store.update_cycle(idx, cycle)
        elif hasattr(memory_store, "save_cycle_history"):
            memory_store.save_cycle_history(history)

        # If the store has add_citation, also record normalized citations
        if hasattr(memory_store, "add_citation"):
            goal = cycle.get("goal") or "global"
            run_id = cycle.get("run_id")
            domain = cycle.get("domain")
            role = cycle.get("role")
            try:
                cyc_idx_num = int(cycle.get("cycle")) if "cycle" in cycle else None
            except Exception:
                cyc_idx_num = None

            # Use the merged list attached to the cycle to avoid divergence
            merged = cycle.get("citations") or []
            norm_merged = normalize_citation_list(merged)

            for c in norm_merged:
                try:
                    memory_store.add_citation(
                        goal=goal,
                        citation=c,
                        run_id=run_id,
                        role=role,
                        domain=domain,
                        cycle_index=cyc_idx_num,
                        tool_name=_safe_str(c.get("source")) or None,
                        extra=None,
                    )
                except Exception:
                    # Never let a single bad citation break the run
                    continue
    except Exception:
        # Fail silent. Reporting code will still work with whatever is present.
        return
