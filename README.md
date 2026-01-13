# ⭐ Autonomous Research Agent  
### Powered by Reparodynamics • TGRM • RYE • Swarm Intelligence • MSIL

**Live Deployment**  
https://autonomous-research-agent-6tqt.onrender.com

**Important**  
Add your Tavily API key in the sidebar. Without it, search runs in offline stub mode.

---

## 🔬 Scientific Attribution and Field Protection

**Reparodynamics**, **TGRM (Targeted Gradient Repair Mechanism)**, and  
**RYE (Repair Yield per Energy)** were created and defined by **Cody Ryan Jenkins**.

These scientific terms may not be renamed, repurposed, rebranded, or represented as originating from any other individual, institution, or entity.

**Required Citation**  
Jenkins, C. R. (2025). *Reparodynamics v2.2*.  
DOI: https://doi.org/10.5281/zenodo.17653046

Reparodynamics is a claimed scientific trademark of Cody Ryan Jenkins.

---

## 🚀 Overview

The **Autonomous Research Agent (ARA)** is a long-run scientific intelligence engine based on Reparodynamics, the universal science of self-repairing systems.

The architecture is **engineered for multi-hour to multi-month autonomous operation**, but the current public build runs in **finite-cycle mode only** with explicitly defined cycle budgets.

Long-run continuity logic, stability kernels, and recovery dynamics are already implemented and validated internally.

Powered by:

- **Reparodynamics** – self-repair physics for AI  
- **TGRM v3** – high-precision gradient repair cycles  
- **RYE v3** – efficiency-driven learning laws  
- **MSIL v2** – Meta Stability Intelligence Layer  
- **Stability Kernel v2** – equilibrium and oscillation analysis  
- **Ultra Swarm Engine** – 5 to 32 autonomous agents  
- **Discovery Engine v3** – breakthrough-tier detection  
- **MemoryStore v3** – persistent learning substrate  

---

## 🔁 TGRM Loop — The Repair Core

Each cycle executes a closed self-repair loop:

1. **TEST** – scan memory and reasoning traces  
2. **DETECT** – identify contradictions, noise, instability  
3. **REPAIR** – search, ingest PDFs, analyze data, refine hypotheses  
4. **VERIFY** – compute ΔR, energy cost, and RYE  

Each cycle logs:

- ΔR (repair gain)  
- E (energy cost)  
- **RYE (ΔR / E)**  
- Rolling RYE  
- Median RYE  
- RYE slope and curvature  
- Stability Index  
- Recovery Momentum  
- Oscillation Signature  

This forms a **self-repairing scientific intelligence loop** rather than a prompt-response model.

---

## ⚙ Key Components

### ⭐ CoreAgent v3
- Full TGRM integration  
- Advanced ΔR verification  
- Noise-resistant RYE v3  
- Hypothesis generation and scoring  
- PDF and PubMed ingestion  
- Biomarker processing support  
- Equilibrium-aware learning  
- Automatic continuity across finite runs  
- Ultra Fast Learning Mode (10× throughput)  

### 🧬 Longevity Domain Gate & Evidence Filtering

To ensure that the agent remains focused on the biology of ageing and avoids “domain drift” into irrelevant material (e.g. Roman law, unrelated syndromes or legal history), **all evidence ingestion tools now enforce a Longevity Domain Gate**.  Each citation retrieved from PubMed, Semantic Scholar or PDF searches is scored against a small set of biological keywords:

- `longevity`
- `aging`
- `metabolism`
- `senescence`
- `epigenetics`
- `inflammation`

The relevance score is simply the fraction of these keywords that appear in the title and snippet of a candidate citation.  Only citations with a relevance score ≥ 0.65 are admitted to the evidence base.  Items containing unrelated terms such as “law”, “legal”, “Roman” or “syndrome” without any of the keywords are hard‑rejected.  When no results pass the gate, the ingestion tool falls back to returning the raw results to avoid empty responses, but downstream scoring will down‑weight them accordingly.

### 🌐 Web‑Blind Runs and Tavily Failure Handling

If the Tavily API fails during a cycle (for example, because the user did not provide an API key or the service is temporarily unavailable), the agent automatically switches into **academic‑only, high‑rigor mode**.  Citations returned with messages like “Tavily API error” are detected and a `tavily_web_blind_run` flag is recorded in the verification cycle.  This flag alerts the verification and reporting layers that the run relied solely on peer‑reviewed sources (PubMed, Semantic Scholar, PDF ingestion) and that no general web search was available.  In this mode the system raises evidence grade thresholds and tightens relevance filters to maintain scientific rigour.

### 🛠 Maintenance Mode Enforcement

When a cycle enters maintenance mode (detected via the `maintenance_mode` flag in the discovery log), exploration agents are disabled, the critic ratio is increased, citation‑backed gap closure is required, and new hypothesis creation is frozen.  This encourages the swarm to converge on high‑confidence results rather than wandering.  Maintenance logs and template entries are automatically filtered from hypothesis lists and discovery reports to prevent placeholder content from influencing verification or narrative summaries.

---

## ⭐ Swarm Intelligence (5 to 64 Agents)

Role-based swarm architecture:

- Researcher  
- Critic  
- Explorer  
- Integrator  
- Planner  
- Synthesis Agent  
- agent_01 through agent_64  

Each swarm cycle produces:

- Consensus RYE  
- Cross-agent contradiction detection  
- Multi-role hypotheses  
- Repair-signature comparison  
- Alignment score  
- Stability-weighted aggregation  

Swarm mode is the highest-performance operational mode of ARA.

---

## ⭐ MSIL — Meta Stability Intelligence Layer

MSIL provides continuous metacognitive analysis:

- Rolling and median RYE  
- RYE slope and curvature  
- Noise signature detection  
- Recovery Momentum  
- TGRM Harmonic Index  
- Stability Index  
- Expansion versus saturation detection  
- Equilibrium classification  
- Tier-style discovery likelihood  
- Multi-agent alignment metrics  
- Safety envelope estimation  

MSIL functions as the system’s **self-awareness layer**, enabling proto-AGI behavior during extended finite runs.

---

## ⭐ Stability Kernel v2

Advanced stability analytics:

- Equilibrium detection  
- Oscillation detection  
- Fragility versus robustness classification  
- Noise-suppressed stability curves  
- Recovery tracking  
- Run-phase segmentation  
- UI-ready stability summaries  

---

## ⭐ Discovery Engine v3

Multi-stage scientific discovery pipeline:

- Hypothesis grid construction  
- ΔR-weighted novelty scoring  
- Multi-agent critique loops  
- Verification engine integration  
- Breakthrough-tier classification  
- Discovery IDs and structured logs  
- Cross-run longitudinal density analysis  

Fully integrated with MSIL and swarm intelligence.

---

## ⭐ MemoryStore v3

Persistent adaptive memory system:

- Cross-run memory continuity  
- RYE-aware pruning  
- Stability-weighted retention  
- Scratch versus gold notebook separation  
- Half-life forgetting models  
- Replay buffer integration  

ARA can accumulate intelligence **across many finite runs**, approximating long-run behavior safely.

---

## ⭐ Protocol Synthesizer v2

Automatically generates actionable scientific protocols:

- Stepwise experimental proposals  
- Mechanism-centered synthesis  
- Multi-agent consensus optimization  
- Stability-aware refinement  
- Domain-specific mapping  

---

## ⭐ Reporting System

### Markdown Reporter
Generates `.md` reports containing:

- Cycle logs  
- RYE graphs  
- Stability signatures  
- Discovery summaries  

### PDF Reporter
Publication-ready research reports:

- Full MSIL diagnostics  
- Stability visualizations  
- Hypothesis matrices  
- Cross-agent alignment tables  
- Breakthrough-tier classification  

---

## 🧩 Run Control and Presets

### `run_agent.py`

Supports **finite-cycle execution only**:

- Explicit cycle budgets  
- Deterministic stopping conditions  
- Stability-safe termination  

The system architecture is designed for future reactivation of true wall-clock long-run execution.

### Presets include:
- General  
- Longevity  
- Math  
- Deep Biomarkers  
- High Stability  
- Breakthrough Mode  
- Ultra Fast Learning  

---

## 🧱 Repository Structure

Autonomous-research-agent/  
│  
├── config/  
│   └── settings.yaml  
│  
├── agent/  
│   ├── core_agent.py  
│   ├── tgrm_loop.py  
│   ├── memory_store.py  
│   ├── presets.py  
│   ├── stability_kernel.py  
│   ├── msil.py  
│   ├── discovery_manager.py  
│   ├── protocol_synthesizer.py  
│   ├── swarm_orchestrator.py  
│   ├── rye_metrics.py  
│   ├── verification_engine.py  
│   ├── contradictions.py  
│   ├── hypothesis_engine.py  
│   ├── hypothesis_manager.py  
│   ├── snapshot_generator.py  
│   ├── meta_agent.py  
│   ├── replay_buffer.py  
│   ├── memory_pruner.py  
│   ├── intelligence_profiles.py  
│   ├── multi_hallmark_pipeline.py  
│   │  
│   ├── specialists/  
│   │   ├── biomarker_agent.py  
│   │   ├── critic_agent.py  
│   │   ├── pdf_summarizer_agent.py  
│   │   ├── synergy_agent.py  
│   │   └── meta_evaluator.py  
│   │  
│   └── tools/  
│       ├── browser_tool.py  
│       ├── code_sandbox.py  
│       ├── web_search.py  
│       ├── data_connectors.py  
│       ├── tool_router.py  
│       └── pdf_reporter.py  
│  
├── swarm/  
│   ├── swarm_coordinator.py  
│   └── swarm_profiles.yaml  
│  
├── tests/  
│   ├── test_stability_kernel.py  
│   ├── test_msil.py  
│   ├── test_discovery_manager.py  
│   ├── test_citations.py  
│   ├── test_curriculum.py  
│   └── test_replay_buffer.py  
│  
├── app_streamlit.py  
├── engine_worker.py  
├── run_agent.py  
├── daily_runner.py  
├── storage.py  
├── config.py  
├── main.py  
├── render.yaml  
├── .env.example  
├── requirements.txt  
└── README.md  

---

## 🛡 Licensing (Option C – Strong Legal Tone)

**This software is PROPRIETARY. All rights reserved.**

### ARA-FCCL (Full Control License)
- No redistribution  
- No sublicensing  
- No derivative systems  
- No hosting without permission  
- No competing commercial use  
- All derivatives belong to **Cody Ryan Jenkins**  
- License revocable upon breach  

### ARA-Commercial License
For approved commercial entities:

- Single-seat rights  
- Non-transferable  
- Non-sublicensable  
- Royalty structures may apply  
- Subject to audit  
- Revocable upon violation  

**Use of this software constitutes full acceptance of these licenses.**  
See `LICENSE` and `COMMERCIAL_LICENSE.md` for full terms.

---

## ✔ End of README
