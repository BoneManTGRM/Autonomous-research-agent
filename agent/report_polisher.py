"""report_polisher.py
====================

Deterministic, evidence-respecting polishing for ARA markdown reports.

Why this exists
---------------
The ARA generators produce *accurate* structured reports, but raw output can be
noisy (template echoes, placeholder discoveries, repeated goal strings, log-like
phrasing). This module adds a final, deterministic pass that converts run
artifacts into a cleaner, executive-ready markdown report **without inventing
facts**.

Design principles
-----------------
* Never hallucinate: only summarize what is present in the provided context.
* Be explicit about uncertainty and data quality.
* Prefer human-readable synthesis over raw dumps.
* Preserve citations and evidence summaries.

Usage
-----
This module is intended to be called by ``report_generator.py``.

Typical call patterns:

    from .report_polisher import build_publishable_report

    md = build_publishable_report(context)

Where ``context`` is a dict containing (best-effort):
    - run_id, domain, goal, runtime, cycles
    - findings: list of discovery dicts (label, evidence_summary, score, ...)
    - citations: list of citation dicts (title, url/link, doi)

The functions are robust to missing keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import math
import os
import re
import textwrap
from collections import Counter
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def _safe_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return ""


def sanitize_goal_text(goal: Optional[str], *, max_chars: int = 700) -> str:
    """Remove prompt scaffolding and keep the user-intent portion.

    The ARA system sometimes passes the full run prompt/spec as the "goal".
    This function strips directive headers and bullet lists so reports don't
    echo the full control prompt.
    """
    if not goal:
        return ""

    directive_prefixes = (
        "title",
        "instruction handling rule",
        "primary objective",
        "primary goal",
        "secondary objectives",
        "evidence priority",
        "evidence priority order",
        "high-probability",
        "high probability",
        "phase",
        "cycle",
        "stop conditions",
        "you must",
        "you must not",
        "exclusions",
        "discovery standard",
        "falsification",
        "checkpoint",
        "null result",
        "truth",
    )

    cleaned: List[str] = []
    for ln in _safe_str(goal).splitlines():
        s = ln.strip()
        if not s:
            continue

        lower = s.lower().rstrip(":").strip()

        # Drop markdown headings
        if re.match(r"^#+\s", s):
            continue

        # Drop numbered list items and bullets (prompt scaffolding)
        if re.match(r"^\d+[\.|\)]\s", s):
            continue
        if s.startswith(("-", "•", "*")):
            continue

        if any(lower.startswith(p) for p in directive_prefixes):
            continue

        cleaned.append(s)
        if sum(len(x) for x in cleaned) >= max_chars:
            break

    out = "\n".join(cleaned).strip()
    # Final guard against leaked headers
    out = re.sub(r"(?im)^\s*(title|instruction handling rule)\s*:?.*$", "", out).strip()
    # Collapse excess blank lines
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


def _wrap(p: str, width: int = 92) -> str:
    p = _safe_str(p).strip()
    if not p:
        return ""
    return "\n".join(textwrap.fill(line, width=width) for line in p.splitlines())


def _first_sentence(text: str, *, max_chars: int = 260) -> str:
    s = _safe_str(text).strip()
    if not s:
        return ""
    # Split on sentence boundaries conservatively
    parts = re.split(r"(?<=[\.!\?])\s+", s)
    first = parts[0].strip() if parts else s
    if len(first) > max_chars:
        first = first[: max_chars - 1].rstrip() + "…"
    return first


# ---------------------------------------------------------------------------
# Evidence/source quality heuristics (transparent + conservative)
# ---------------------------------------------------------------------------


_HIGH_TRUST_DOMAINS = {
    "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "nih.gov",
    "clinicaltrials.gov",
    "nature.com",
    "science.org",
    "sciencemag.org",
    "cell.com",
    "nejm.org",
    "thelancet.com",
    "jamanetwork.com",
    "bmj.com",
    "plos.org",
}

_LOW_TRUST_HINTS = (
    "wikipedia.org",
    "merriam-webster.com",
    "thefreedictionary.com",
    "dictionary.cambridge.org",
    "oxfordlearnersdictionaries.com",
    "reddit.com",
)


def _domain_from_url(url: str) -> str:
    u = _safe_str(url).strip()
    if not u:
        return ""
    try:
        netloc = urlparse(u).netloc.lower()
        return netloc
    except Exception:
        return ""


def score_source(url: str) -> int:
    """Return a coarse 0–2 score for source trustworthiness.

    2 = high-trust scholarly / primary sources
    1 = unknown / mixed quality
    0 = clearly non-scholarly (dictionaries, wikis, forums)
    """
    dom = _domain_from_url(url)
    if not dom:
        return 1
    if any(h in dom for h in _LOW_TRUST_HINTS):
        return 0
    if dom in _HIGH_TRUST_DOMAINS or any(dom.endswith("." + d) for d in _HIGH_TRUST_DOMAINS):
        return 2
    # Heuristic: .gov and .edu are often higher quality, but not guaranteed
    if dom.endswith(".gov") or dom.endswith(".edu"):
        return 2
    return 1


def summarize_evidence_base(citations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    urls = [
        _safe_str(c.get("url") or c.get("link") or "").strip()
        for c in citations
        if isinstance(c, dict)
    ]
    scores = [score_source(u) for u in urls if u]
    high = sum(1 for s in scores if s == 2)
    low = sum(1 for s in scores if s == 0)
    mid = sum(1 for s in scores if s == 1)
    avg = (sum(scores) / len(scores)) if scores else 0.0
    if not scores:
        quality = "unknown"
    elif avg >= 1.5:
        quality = "high"
    elif avg >= 0.9:
        quality = "mixed"
    else:
        quality = "low"
    return {
        "n": len(citations),
        "high": high,
        "mid": mid,
        "low": low,
        "avg": avg,
        "quality": quality,
    }


# ---------------------------------------------------------------------------
# Finding cleanup + ranking
# ---------------------------------------------------------------------------


_PLACEHOLDER_PATTERNS = (
    "initial discovery_log.json",
    "discovery log initialized",
    "template entry",
    "placeholder discovery",
    "shows how",
    "used only to show",
    "example",
    "performed initial",
    "initial research summary",
)


def is_placeholder_label(label: str) -> bool:
    s = _safe_str(label).strip().lower()
    if not s:
        return True
    if len(s) < 6:
        return True
    if "**" in s:
        return True
    if any(p in s for p in _PLACEHOLDER_PATTERNS):
        return True
    return False


def _norm_label(label: str) -> str:
    return re.sub(r"\s+", " ", _safe_str(label).strip()).lower()


def dedupe_findings(findings: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for d in findings:
        if not isinstance(d, dict):
            continue
        label = _safe_str(d.get("label") or "").strip()
        key = _norm_label(label)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _finding_rank_key(d: Dict[str, Any]) -> Tuple[int, float, float, str]:
    """Sort key for findings.

    Priority order:
      1) verification_status (validated > pending/unknown > rejected)
      2) priority_rank (lower is better)
      3) score (higher is better)
      4) timestamp (newer last; but we treat as tie-break)
    """
    status = _safe_str(d.get("verification_status") or "").strip().lower()
    if status in {"validated", "verified", "confirmed", "supported"}:
        status_bucket = 2
    elif status in {"rejected", "failed", "invalid"}:
        status_bucket = 0
    else:
        status_bucket = 1

    pr = d.get("priority_rank")
    try:
        pr_val = float(pr)
    except Exception:
        pr_val = math.inf

    sc = d.get("score")
    try:
        sc_val = float(sc)
    except Exception:
        sc_val = 0.0

    ts = _safe_str(d.get("timestamp") or "")
    return (-status_bucket, pr_val, -sc_val, ts)


def filter_findings(
    findings: Sequence[Dict[str, Any]],
    *,
    goal: str = "",
    require_evidence_summary: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (kept, dropped) findings.

    *Never* invents data; just removes obvious placeholders and goal-echoes.
    """
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []

    goal_norm = _norm_label(goal)

    for d in dedupe_findings(findings):
        label = _safe_str(d.get("label") or "").strip()
        if is_placeholder_label(label):
            dropped.append(d)
            continue

        # Drop items that mostly restate the goal (common failure mode)
        if goal_norm and goal_norm[:20] and goal_norm[:20] in _norm_label(label):
            dropped.append(d)
            continue

        if require_evidence_summary:
            ev = _safe_str(d.get("evidence_summary") or "").strip()
            if not ev:
                dropped.append(d)
                continue

        kept.append(d)

    kept_sorted = sorted(kept, key=_finding_rank_key)
    return kept_sorted, dropped


def bucket_findings(findings: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "validated": [],
        "promising": [],
        "rejected": [],
    }
    for d in findings:
        status = _safe_str(d.get("verification_status") or "").strip().lower()
        if status in {"validated", "verified", "confirmed", "supported"}:
            buckets["validated"].append(d)
        elif status in {"rejected", "failed", "invalid"}:
            buckets["rejected"].append(d)
        else:
            buckets["promising"].append(d)
    return buckets


# ---------------------------------------------------------------------------
# Markdown builders
# ---------------------------------------------------------------------------


def _citation_key(c: Dict[str, Any]) -> str:
    url = _safe_str(c.get("url") or c.get("link") or "").strip().lower()
    doi = _safe_str(c.get("doi") or "").strip().lower()
    title = _safe_str(c.get("title") or c.get("raw") or "").strip().lower()
    return url or doi or title


def _merge_citations(
    base: Sequence[Dict[str, Any]],
    extra: Sequence[Dict[str, Any]],
    *,
    max_total: int = 60,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for seq in (base, extra):
        for c in seq:
            if not isinstance(c, dict):
                continue
            k = _citation_key(c)
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(c)
            if len(out) >= max_total:
                return out
    return out


def _format_source_line(idx: int, c: Dict[str, Any]) -> str:
    title = _safe_str(c.get("title") or c.get("raw") or "(untitled)").strip()
    url = _safe_str(c.get("url") or c.get("link") or "").strip()
    if url:
        return f"{idx}. {title} — {url}"
    return f"{idx}. {title}"


def _extract_tags(d: Dict[str, Any], limit: int = 6) -> List[str]:
    tags = d.get("tags")
    if isinstance(tags, list):
        out = [
            _safe_str(t).strip() for t in tags if _safe_str(t).strip()
        ]
        return out[:limit]
    return []


def _extract_inline_refs(d: Dict[str, Any], citation_index: Dict[str, int]) -> List[int]:
    refs: List[int] = []
    cites = d.get("citations")
    if not isinstance(cites, list):
        return refs
    for c in cites:
        if not isinstance(c, dict):
            continue
        k = _citation_key(c)
        if not k:
            continue
        idx = citation_index.get(k)
        if isinstance(idx, int) and idx not in refs:
            refs.append(idx)
    return refs[:4]


def build_publishable_report(context: Dict[str, Any]) -> str:
    """Build a polished, publishable markdown report.

    This function is deterministic and only uses the provided context.
    """
    run_id = _safe_str(context.get("run_id") or "").strip()
    domain = _safe_str(context.get("domain") or "general").strip() or "general"
    goal_raw = _safe_str(context.get("goal") or "").strip()
    goal = sanitize_goal_text(goal_raw)
    runtime = _safe_str(context.get("runtime") or "(runtime unavailable)").strip()
    cycles = context.get("cycles")
    try:
        cycles_n = int(cycles)
    except Exception:
        cycles_n = None

    findings_raw = context.get("findings") if isinstance(context.get("findings"), list) else []
    citations_raw = context.get("citations") if isinstance(context.get("citations"), list) else []
    try:
        max_sources = int(context.get("max_sources") or 60)
    except Exception:
        max_sources = 60
    max_sources = max(5, min(200, max_sources))
    try:
        max_findings = int(context.get("max_findings") or 12)
    except Exception:
        max_findings = 12
    max_findings = max(3, min(60, max_findings))

    # Merge citations from findings so the report can cite them.
    findings_cites: List[Dict[str, Any]] = []
    for d in findings_raw:
        if isinstance(d, dict) and isinstance(d.get("citations"), list):
            for c in d.get("citations"):
                if isinstance(c, dict):
                    findings_cites.append(c)

    citations = _merge_citations(citations_raw, findings_cites, max_total=max(80, max_sources))
    citation_index = {_citation_key(c): i for i, c in enumerate(citations, start=1) if _citation_key(c)}

    kept, dropped = filter_findings(findings_raw, goal=goal)
    evidence_summary = summarize_evidence_base(citations)

    # If the captured evidence base is dominated by dictionaries/glossaries,
    # tighten filtering (common failure mode when a query becomes "define X").
    low_domains = {
        _domain_from_url(_safe_str(c.get("url") or c.get("link") or ""))
        for c in citations
        if isinstance(c, dict)
    }
    has_dictionary_sources = any(any(h in d for h in _LOW_TRUST_HINTS) for d in low_domains if d)
    if has_dictionary_sources:
        extra_drop_terms = ("definition", "dictionary")
        kept2: List[Dict[str, Any]] = []
        for d in kept:
            lbl = _safe_str(d.get("label") or "").lower()
            if any(t in lbl for t in extra_drop_terms):
                dropped.append(d)
                continue
            kept2.append(d)
        kept = kept2

    buckets = bucket_findings(kept)

    # Quick quality assessment (transparent heuristics)
    quality_flags: List[str] = []
    if cycles_n is not None and cycles_n < 5:
        quality_flags.append("Very short run (few cycles); treat conclusions as preliminary.")
    if evidence_summary["n"] == 0:
        quality_flags.append("No citations captured; findings are not well-supported in the logs.")
    if evidence_summary["quality"] == "low":
        quality_flags.append("Evidence base quality appears low (many non-scholarly sources).")
    if has_dictionary_sources:
        quality_flags.append("Dictionary/glossary sources detected; likely query drift or low-signal retrieval.")

    # Top themes (keywords) from finding labels
    labels_text = " ".join([_safe_str(d.get("label") or "") for d in kept])
    tokens = [t.lower() for t in re.findall(r"[a-zA-Z][a-zA-Z\-]{4,}", labels_text)]
    stop = {
        "study", "studies", "effect", "effects", "using", "based", "shown",
        "result", "results", "across", "within", "system", "mechanism", "pathway",
        "intervention", "candidate", "analysis", "data", "model", "models",
        "improves", "improve", "increase", "increases", "decrease", "decreases",
        "longevity", "anti-aging", "aging", "ageing",
    }
    top_terms = [t for t, _ in Counter([t for t in tokens if t not in stop]).most_common(8)]

    # ------------------------------------------------------------------
    # Compose markdown
    # ------------------------------------------------------------------
    out: List[str] = []
    out.append("# Autonomous research report\n")

    out.append("## Run overview\n")
    if run_id:
        out.append(f"- Run ID: `{run_id}`")
    out.append(f"- Domain: **{domain}**")
    if cycles_n is not None:
        out.append(f"- Cycles captured: **{cycles_n}**")
    out.append(f"- Runtime: **{runtime}**")
    if evidence_summary["n"]:
        out.append(
            f"- Evidence base: **{evidence_summary['n']}** sources (quality: **{evidence_summary['quality']}**)"
        )
    out.append("")

    out.append("## Run goal\n")
    if goal:
        out.append(_wrap(goal))
    else:
        out.append("(Goal not available.)")
    out.append("")

    out.append("## Executive summary\n")

    # Outcome sentence
    if buckets["validated"]:
        out.append(
            _wrap(
                "The run produced at least one *validated* candidate finding. The items below are the highest-signal "
                "mechanisms/interventions extracted from the run artifacts, along with the evidence summaries recorded by the system."
            )
        )
    elif buckets["promising"]:
        out.append(
            _wrap(
                "No validated findings were recorded in the artifacts, but the run produced a small set of *promising* candidates "
                "worth follow-up. These are ranked by the run’s own scoring/priority fields when available."
            )
        )
    else:
        out.append(
            _wrap(
                "No credible mechanism or intervention candidates survived basic filtering. This should be treated as a *null/low-signal* "
                "run. The report focuses on what was captured and what to change to obtain higher-signal outputs."
            )
        )

    out.append("")
    bullets: List[str] = []
    if top_terms:
        bullets.append(f"Top themes observed: {', '.join(top_terms)}.")
    if kept:
        bullets.append(f"Candidate findings retained after filtering: **{len(kept)}** (dropped: **{len(dropped)}**).")
    else:
        bullets.append(f"Candidate findings retained after filtering: **0** (dropped: **{len(dropped)}**).")
    if evidence_summary["n"]:
        bullets.append(
            f"Sources captured: {evidence_summary['high']} high-trust, {evidence_summary['mid']} mixed, {evidence_summary['low']} low-trust."
        )
    for q in quality_flags[:4]:
        bullets.append(q)
    out.extend([f"- {b}" for b in bullets])
    out.append("")

    # Key findings
    out.append("## Key findings\n")

    def _emit_findings_block(title: str, items: Sequence[Dict[str, Any]], limit: int = 8) -> None:
        out.append(f"### {title}\n")
        if not items:
            out.append("No items in this category.\n")
            return
        for i, d in enumerate(items[:limit], start=1):
            label = _safe_str(d.get("label") or "").strip()
            kind = _safe_str(d.get("kind") or "").strip() or "finding"
            status = _safe_str(d.get("verification_status") or "").strip() or "(unverified)"
            score = d.get("score")
            score_txt = ""
            if isinstance(score, (int, float)):
                score_txt = f"score {float(score):.2f}"
            tags = _extract_tags(d)
            tags_txt = f" | tags: {', '.join(tags)}" if tags else ""

            ev = _safe_str(d.get("evidence_summary") or d.get("data", {}).get("evidence_summary") or "")
            ev1 = _first_sentence(ev)

            refs = _extract_inline_refs(d, citation_index)
            refs_txt = f" [{' ,'.join(str(r) for r in refs)}]".replace(" ,", ", ") if refs else ""

            meta_bits = ", ".join([b for b in [score_txt, status] if b])
            meta_txt = f"({meta_bits})" if meta_bits else ""
            out.append(f"{i}. **{label}** — *{kind}* {meta_txt}{refs_txt}")
            if ev1:
                out.append(f"   - Evidence: {_wrap(ev1)}")
            if tags_txt:
                out.append(f"   -{tags_txt}")
        if len(items) > limit:
            out.append(f"\n… and {len(items) - limit} more.\n")
        else:
            out.append("")

    # Respect max_findings by prioritizing validated+promising, and showing only a
    # small rejected slice.
    validated_limit = min(6, max(1, max_findings // 2))
    remaining = max_findings - validated_limit
    promising_limit = max(1, remaining)
    rejected_limit = min(6, len(buckets["rejected"]))

    _emit_findings_block("Validated / strongest", buckets["validated"], limit=validated_limit)
    _emit_findings_block("Promising but unverified", buckets["promising"], limit=promising_limit)
    _emit_findings_block("Rejected / failed verification", buckets["rejected"], limit=rejected_limit)

    # Filtering disclosure
    out.append("## Data quality and filtering\n")
    out.append(
        _wrap(
            "To prevent template echoes and log noise from polluting conclusions, the report generator removes placeholder-like entries "
            "(e.g., discovery log templates, example records) and obvious goal restatements. This section discloses what was removed."
        )
    )
    out.append("")
    out.append(f"- Dropped items: **{len(dropped)}**")
    if dropped:
        examples = [
            _safe_str(d.get("label") or "").strip() for d in dropped if _safe_str(d.get("label") or "").strip()
        ][:5]
        if examples:
            out.append("- Examples dropped: " + "; ".join(examples))
    out.append("")

    # Evidence base
    out.append("## Evidence base\n")
    if not citations:
        out.append("No citations were captured in the run artifacts.\n")
    else:
        out.append(f"Unique sources captured: **{len(citations)}**\n")
        for i, c in enumerate(citations[:max_sources], start=1):
            out.append(_format_source_line(i, c))
        if len(citations) > max_sources:
            out.append(f"… and {len(citations) - max_sources} more.")
        out.append("")

    # Recommendations
    out.append("## Recommendations\n")
    recs: List[str] = []
    if cycles_n is not None and cycles_n < 10:
        recs.append("Increase cycle count (or enable multi-agent runs) so hypotheses and verification have time to converge.")
    if evidence_summary["quality"] in {"low", "unknown"}:
        recs.append("Prefer scholarly sources (PubMed / clinical trials / major journals) and avoid dictionary or glossary hits.")
    recs.append("Ensure each candidate discovery includes an evidence_summary and citations so the report can attribute claims.")
    recs.append("If the run is immune–senescence focused, enforce domain keywords in search queries to avoid generic web results.")
    out.extend([f"- {r}" for r in recs])
    out.append("")

    return "\n".join(out).strip() + "\n"


def build_findings_report(context: Dict[str, Any]) -> str:
    """Build a polished targeted findings report."""
    run_id = _safe_str(context.get("run_id") or "").strip()
    goal_raw = _safe_str(context.get("goal") or "").strip()
    goal = sanitize_goal_text(goal_raw)
    runtime = _safe_str(context.get("runtime") or "").strip()

    findings_raw = context.get("findings") if isinstance(context.get("findings"), list) else []
    citations_raw = context.get("citations") if isinstance(context.get("citations"), list) else []
    unstructured = context.get("unstructured_findings") if isinstance(context.get("unstructured_findings"), list) else []
    try:
        max_sources = int(context.get("max_sources") or 60)
    except Exception:
        max_sources = 60
    max_sources = max(5, min(200, max_sources))
    kept, dropped = filter_findings(findings_raw, goal=goal)

    citations = _merge_citations(citations_raw, [], max_total=max(80, max_sources))

    # Bucket by kind
    by_kind: Dict[str, List[Dict[str, Any]]] = {}
    for d in kept:
        k = _safe_str(d.get("kind") or "other").strip().lower() or "other"
        by_kind.setdefault(k, []).append(d)

    out: List[str] = []
    out.append("# Targeted findings report\n")
    if run_id:
        out.append(f"- Run ID: `{run_id}`")
    if runtime:
        out.append(f"- Runtime: {runtime}")
    out.append("")
    if goal:
        out.append("## Goal\n")
        out.append(_wrap(goal))
        out.append("")

    out.append("## Top candidates\n")
    if not kept:
        out.append("No credible candidates survived filtering.\n")
    else:
        # Show a compact table-like list
        limit = 20
        for i, d in enumerate(kept[:limit], start=1):
            label = _safe_str(d.get("label") or "").strip()
            kind = _safe_str(d.get("kind") or "other").strip() or "other"
            status = _safe_str(d.get("verification_status") or "").strip() or "(unverified)"
            score = d.get("score")
            score_txt = f"{float(score):.2f}" if isinstance(score, (int, float)) else ""
            ev = _first_sentence(_safe_str(d.get("evidence_summary") or ""), max_chars=180)
            meta = ", ".join([x for x in [score_txt and f"score {score_txt}", status] if x])
            out.append(f"{i}. **{label}** — *{kind}* ({meta})")
            if ev:
                out.append(f"   - {_wrap(ev)}")
        if len(kept) > limit:
            out.append(f"\n… and {len(kept) - limit} more.\n")
        out.append("")

    # Unstructured signals (e.g., hypotheses text mining) – shown separately so
    # the reader can distinguish between structured memory entries and raw text.
    if unstructured:
        out.append("## Other signals (unstructured)\n")
        seen: set = set()
        picked: List[str] = []
        for t in unstructured:
            s = _safe_str(t).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            picked.append(s)
            if len(picked) >= 25:
                break
        if not picked:
            out.append("No unstructured signals kept after de-duplication.\n")
        else:
            for i, s in enumerate(picked, start=1):
                out.append(f"{i}. {s}")
            if len(unstructured) > len(picked):
                out.append(f"… and {len(unstructured) - len(picked)} more.")
        out.append("")

    out.append("## Filtering disclosure\n")
    out.append(f"- Dropped placeholder/low-signal items: **{len(dropped)}**\n")

    out.append("## Sources\n")
    if not citations:
        out.append("No citations captured.\n")
    else:
        for i, c in enumerate(citations[:max_sources], start=1):
            out.append(_format_source_line(i, c))
        if len(citations) > max_sources:
            out.append(f"… and {len(citations) - max_sources} more.")
        out.append("")

    return "\n".join(out).strip() + "\n"
