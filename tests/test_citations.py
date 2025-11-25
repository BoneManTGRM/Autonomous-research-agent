"""Tests for agent.citation_utils.

These tests are written for pytest. They focus on:
- Robust normalization of messy citation objects.
- Extraction from tool_events and cycles.
- Merging and deduplication behavior.
- Bibliography formatting.
- Attachment helpers that update cycles and optional memory stores.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from agent.citation_utils import (
    normalize_citation,
    normalize_citation_list,
    extract_citations_from_tool_events,
    extract_citations_from_cycle,
    extract_citations_from_cycles,
    merge_citation_lists,
    format_citation_markdown,
    build_bibliography_markdown,
    attach_citations_to_cycle,
    attach_citations_to_memory_store,
)


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------


def test_normalize_basic_web_citation() -> None:
    raw = {
        "source": "tavily",
        "title": "Repair dynamics and complex systems",
        "url": "https://example.com/repair",
        "snippet": "A paper about reparodynamics.",
    }

    norm = normalize_citation(raw)

    assert norm is not None
    assert norm["source"] == "tavily"
    assert norm["title"] == "Repair dynamics and complex systems"
    assert norm["url"] == "https://example.com/repair"
    assert "raw" in norm


def test_normalize_nested_metadata_and_doi() -> None:
    raw = {
        "provider": "pubmed",
        "metadata": {
            "title": "Longevity pathways",
            "journal": "Aging Cell",
            "year": 2023,
            "doi": "10.1000/example.doi",
            "authors": [{"name": "Smith J"}, {"name": "Lee A"}],
        },
    }

    norm = normalize_citation(raw)

    assert norm is not None
    assert norm["source"] == "pubmed"
    assert norm["title"] == "Longevity pathways"
    assert norm["venue"] == "Aging Cell"
    assert norm["year"] == "2023"
    assert norm["doi"] == "10.1000/example.doi"
    assert "Smith" in (norm["authors"] or "")
    assert "Lee" in (norm["authors"] or "")


def test_normalize_ignores_empty_objects() -> None:
    assert normalize_citation({}) is None
    assert normalize_citation({"metadata": {}}) is None


def test_normalize_citation_list_filters_nones() -> None:
    items = [
        {"title": "Item 1", "url": "https://one"},
        {},  # should be ignored
        {"metadata": {"title": "Item 2", "url": "https://two"}},
    ]

    out = normalize_citation_list(items)
    titles = [c["title"] for c in out]

    assert len(out) == 2
    assert "Item 1" in titles
    assert "Item 2" in titles


# ---------------------------------------------------------------------------
# Extraction tests
# ---------------------------------------------------------------------------


def test_extract_from_tool_events_tavily_style() -> None:
    tool_events = [
        {
            "tool": "tavily_search",
            "results": [
                {"title": "A", "url": "https://a"},
                {"title": "B", "url": "https://b"},
            ],
        }
    ]

    cites = extract_citations_from_tool_events(tool_events)
    urls = {c["url"] for c in cites}

    assert len(cites) == 2
    assert "https://a" in urls
    assert "https://b" in urls
    assert all(c["source"] == "tavily" for c in cites)


def test_extract_from_tool_events_pubmed_style_nested() -> None:
    tool_events = [
        {
            "tool": "pubmed_query",
            "data": {
                "papers": [
                    {"metadata": {"title": "Paper 1", "doi": "10.1234/x"}},
                    {"metadata": {"title": "Paper 2", "doi": "10.5678/y"}},
                ]
            },
        }
    ]

    cites = extract_citations_from_tool_events(tool_events)
    titles = {c["title"] for c in cites}

    assert titles == {"Paper 1", "Paper 2"}
    assert all(c["source"] == "pubmed" for c in cites)


def test_extract_from_cycle_attached_and_tool_events() -> None:
    cycle = {
        "citations": [
            {"title": "Already there", "url": "https://existing"},
        ],
        "tool_events": [
            {
                "tool": "tavily_search",
                "results": [{"title": "New one", "url": "https://new"}],
            }
        ],
    }

    cites = extract_citations_from_cycle(cycle)
    titles = {c["title"] for c in cites}
    urls = {c["url"] for c in cites}

    assert "Already there" in titles
    assert "https://existing" in urls
    assert "New one" in titles
    assert "https://new" in urls


def test_extract_from_cycles_multi_cycle() -> None:
    history = [
        {
            "citations": [
                {"title": "A", "url": "https://a"},
            ]
        },
        {
            "tool_events": [
                {
                    "tool": "pubmed_query",
                    "papers": [{"title": "B", "url": "https://b"}],
                }
            ]
        },
    ]

    cites = extract_citations_from_cycles(history)
    titles = {c["title"] for c in cites}

    assert "A" in titles
    assert "B" in titles
    assert len(cites) == 2


# ---------------------------------------------------------------------------
# Merge, dedupe, and bibliography tests
# ---------------------------------------------------------------------------


def test_merge_citation_lists_prefers_primary_on_duplicates() -> None:
    primary = [
        {
            "source": "tavily",
            "title": "Same",
            "url": "https://same",
            "snippet": "Primary version",
        }
    ]
    secondary = [
        {
            "source": "pubmed",
            "title": "Same",
            "url": "https://same",
            "snippet": "Secondary version",
        }
    ]

    merged = merge_citation_lists(primary, secondary)
    assert len(merged) == 1
    assert merged[0]["source"] == "tavily"
    assert merged[0]["snippet"] == "Primary version"


def test_merge_citation_lists_respects_max_items() -> None:
    primary = [{"title": f"P{i}", "url": f"https://p{i}"} for i in range(5)]
    secondary = [{"title": f"S{i}", "url": f"https://s{i}"} for i in range(5)]

    merged = merge_citation_lists(primary, secondary, max_items=6)
    assert len(merged) == 6


def test_format_citation_markdown_includes_source_title_and_url() -> None:
    c = {
        "source": "pubmed",
        "title": "Interesting paper",
        "url": "https://pubmed.example/paper",
        "authors": "Smith J; Lee A",
        "year": "2024",
        "venue": "Nice Journal",
    }

    line = format_citation_markdown(c)

    assert "**[pubmed]**" in line
    assert "Interesting paper" in line
    assert "Smith" in line
    assert "2024" in line
    assert "https://pubmed.example/paper" in line


def test_build_bibliography_markdown_dedupes_and_limits() -> None:
    cites = [
        {"title": "A", "url": "https://a"},
        {"title": "B", "url": "https://b"},
        {"title": "A", "url": "https://a"},  # duplicate
    ]

    md = build_bibliography_markdown(cites, max_items=10)

    assert md.count("https://a") == 1
    assert "https://b" in md
    assert md.strip().startswith("1.")


# ---------------------------------------------------------------------------
# Attachment helper tests
# ---------------------------------------------------------------------------


def test_attach_citations_to_cycle_in_place_merges() -> None:
    cycle = {
        "citations": [
            {"title": "Old", "url": "https://old"},
        ]
    }
    new = [
        {"title": "New", "url": "https://new"},
        {"title": "Old", "url": "https://old"},  # duplicate
    ]

    attach_citations_to_cycle(cycle, new)

    titles = {c["title"] for c in cycle["citations"]}
    assert titles == {"Old", "New"}
    assert len(cycle["citations"]) == 2


class DummyMemoryStore:
    """Simple in memory stub used to test attachment helpers."""

    def __init__(self, history: Optional[List[Dict[str, Any]]] = None) -> None:
        self._history: List[Dict[str, Any]] = history or []
        self.update_calls: List[int] = []
        self.save_calls: int = 0

    def get_cycle_history(self) -> List[Dict[str, Any]]:
        return self._history

    # version 1 - per cycle update
    def update_cycle(self, index: int, cycle: Dict[str, Any]) -> None:
        self._history[index] = cycle
        self.update_calls.append(index)

    # version 2 - whole history save
    def save_cycle_history(self, history: List[Dict[str, Any]]) -> None:
        self._history = list(history)
        self.save_calls += 1


def test_attach_citations_to_memory_store_with_index() -> None:
    history = [
        {"citations": [{"title": "A", "url": "https://a"}]},
        {"citations": []},
    ]
    store = DummyMemoryStore(history=history)

    new = [{"title": "B", "url": "https://b"}]
    attach_citations_to_memory_store(store, cycle_index=1, citations=new)

    updated_history = store.get_cycle_history()
    titles = {c["title"] for c in updated_history[1]["citations"]}

    assert "A" not in titles
    assert "B" in titles
    assert 1 in store.update_calls or store.save_calls > 0


def test_attach_citations_to_memory_store_defaults_to_last_cycle_when_index_missing() -> None:
    history = [
        {"citations": [{"title": "First", "url": "https://first"}]},
        {"citations": []},
    ]
    store = DummyMemoryStore(history=history)

    new = [{"title": "Last", "url": "https://last"}]
    attach_citations_to_memory_store(store, cycle_index=None, citations=new)

    updated_history = store.get_cycle_history()
    titles_last = {c["title"] for c in updated_history[-1]["citations"]}

    assert "Last" in titles_last
    assert store.update_calls or store.save_calls
