"""
citation_manager.py
===================

This module provides simple utilities for retrieving and formatting
citations without relying on external AI search APIs.  It implements
lightweight access to PubMed via the NCBI E‑utilities and basic local
PDF search.  These helpers can be used by the agent to populate the
source citation viewer in the Streamlit UI.

The functions here do not perform any heavy parsing of PDFs or
full‑text retrieval; they aim to provide bibliographic metadata only.
If you require deeper content extraction consider integrating a
specialised library such as GROBID or spaCy in a separate module.

Notes
-----
* The PubMed API has rate limits.  For production use you should
  register an email and API key and pass them into the search
  functions.  See https://pubmed.ncbi.nlm.nih.gov/ for details.
* If network access is unavailable or PubMed returns no results,
  an empty list is returned rather than raising an exception.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import requests


NCBI_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


def search_pubmed(query: str, max_results: int = 5, *, email: Optional[str] = None, api_key: Optional[str] = None) -> List[Dict[str, str]]:
    """Search PubMed for a query and return a list of citation dicts.

    The returned dicts include ``pmid``, ``title``, ``journal``, ``authors``
    and ``pubdate`` fields.  If the request fails or returns no IDs an
    empty list is returned.

    Parameters
    ----------
    query:
        Free text search query.  Use PubMed syntax (e.g. ``"cell senescence"[Title]``) for
        more precise searches.
    max_results:
        Maximum number of articles to return.
    email:
        Optional contact email for NCBI.  Passing an email is recommended
        when using the API in production; otherwise your requests may
        be throttled more aggressively.
    api_key:
        Optional NCBI API key for higher rate limits.

    Returns
    -------
    list of dict
        Each entry represents a PubMed article with bibliographic
        metadata.
    """
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": str(max_results),
        "retmode": "json",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    try:
        resp = requests.get(NCBI_ESEARCH_URL, params=params, timeout=10)
        resp.raise_for_status()
        esearch = resp.json()
        idlist = esearch.get("esearchresult", {}).get("idlist", [])
    except Exception:
        return []

    if not idlist:
        return []
    # Fetch summaries for IDs
    summary_params = {
        "db": "pubmed",
        "id": ",".join(idlist),
        "retmode": "json",
    }
    if email:
        summary_params["email"] = email
    if api_key:
        summary_params["api_key"] = api_key
    try:
        sresp = requests.get(NCBI_ESUMMARY_URL, params=summary_params, timeout=10)
        sresp.raise_for_status()
        summaries = sresp.json().get("result", {})
    except Exception:
        return []

    results: List[Dict[str, str]] = []
    for pid in idlist:
        rec = summaries.get(str(pid))
        if not rec:
            continue
        title = rec.get("title", "").rstrip(".")
        journal = rec.get("fulljournalname", "") or rec.get("source", "")
        pubdate = rec.get("pubdate", "")
        authors = rec.get("sortfirstauthor", rec.get("authors", [{}])[0].get("name", ""))
        results.append({
            "pmid": pid,
            "title": title,
            "journal": journal,
            "pubdate": pubdate,
            "authors": authors,
        })
    return results


def format_citation(entry: Dict[str, str]) -> str:
    """Return a human‑readable citation string from a PubMed summary dict."""
    parts = []
    authors = entry.get("authors")
    if authors:
        parts.append(authors)
    title = entry.get("title")
    if title:
        parts.append(title)
    journal = entry.get("journal")
    pubdate = entry.get("pubdate")
    journal_part = ", ".join([p for p in [journal, pubdate] if p])
    if journal_part:
        parts.append(journal_part)
    pmid = entry.get("pmid")
    if pmid:
        parts.append(f"PMID:{pmid}")
    return ". ".join(parts)


def search_local_pdfs(directory: Path, query: str, max_results: int = 5) -> List[str]:
    """Search local PDF filenames for a query.

    This helper performs a simple case‑insensitive substring match on
    filenames within ``directory``.  It returns up to ``max_results``
    matching filenames.  It does not parse the contents of the PDF.

    Parameters
    ----------
    directory:
        Root directory to search for PDF files.
    query:
        Substring to match against filenames.
    max_results:
        Maximum number of results to return.

    Returns
    -------
    list of str
        Filenames (not full paths) of matching PDFs.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    matches: List[str] = []
    for path in directory.rglob("*.pdf"):
        if pattern.search(path.name):
            matches.append(path.name)
            if len(matches) >= max_results:
                break
    return matches


def get_citations(query: str, *, local_dir: Optional[Path] = None, max_results: int = 5) -> Dict[str, List[str]]:
    """Retrieve citations from PubMed and optionally local PDFs.

    Returns a dictionary with keys ``"pubmed"`` and ``"local"`` where
    each value is a list of formatted citation strings or filenames.
    """
    pubs = search_pubmed(query, max_results=max_results)
    pub_citations = [format_citation(p) for p in pubs]
    local = []
    if local_dir is not None:
        local = search_local_pdfs(local_dir, query, max_results=max_results)
    return {"pubmed": pub_citations, "local": local}


__all__ = ["search_pubmed", "format_citation", "search_local_pdfs", "get_citations"]