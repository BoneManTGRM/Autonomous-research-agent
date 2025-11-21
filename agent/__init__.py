"""Autonomous research agent package.

This package contains modules implementing the core functionality of the agent,
including the main control loop (TGRM), RYE metrics, memory storage, and
research tools. Each module is designed to be self-contained and easy to
extend.

Key components:
- core_agent: high level orchestration of research cycles
- tgrm_loop: Test, Detect, Repair, Verify loop with Reparodynamics and RYE
- rye_metrics: computation of delta_R, energy E, and RYE
- memory_store: JSON backed memory plus optional vector memory hooks
- vector_memory: semantic, time-decayed long term memory
- tools_web: Tavily based web search and citation normalization
- tools_papers: PDF ingestion and summarisation helpers
- tools_files: local file reading and summarisation
- tools_pubmed: PubMed ingestion via NCBI E utilities
- tools_semantic_scholar: Semantic Scholar ingestion
- hypothesis_engine: simple structured hypothesis generation
- biomarker_analyzer: basic rule based biomarker scoring for longevity mode
- continuous_runner: helper for running continuous sessions from scripts
"""

__all__ = [
    "core_agent",
    "tgrm_loop",
    "rye_metrics",
    "memory_store",
    "vector_memory",
    "tools_web",
    "tools_papers",
    "tools_files",
    "tools_pubmed",
    "tools_semantic_scholar",
    "hypothesis_engine",
    "biomarker_analyzer",
    "continuous_runner",
]
