# Autonomous Research Agent  
Reparodynamics • TGRM • RYE • Swarm Mode  
**Live Production Research Agent:** https://autonomous-research-agent-hqby.onrender.com  

⚠️ IMPORTANT: You must provide your own Tavily API key inside the app sidebar or the agent will not run.

---

## 🔥 Overview

The Autonomous Research Agent is a fully operational, long-run TGRM-powered system capable of:

- Running continuous scientific research using the Reparodynamics framework  
- Computing RYE (Repair Yield per Energy) every cycle  
- Detecting contradictions, gaps, issues, and repairing them  
- Running for extended periods with watchdog, checkpointed run_state, and crash-proof resume  
- Operating as a single agent, duo, or full swarm (2–32 agents)  
- Executing Test → Detect → Repair → Verify autonomously  
- Ingesting papers, datasets, PDFs, CSVs, biomarkers, and more  
- Scaling to millions of cycles with stability guards  
- Functioning as a true autonomous research environment, not a demo

---

## ⚙️ Major Features

### 1. Full TGRM Loop  
- Issue detection  
- Repair actions  
- Verification  
- RYE computation (ΔR / E)  
- Domain-weighted behavior  

### 2. Multi-Format Ingestion  
Reads: TXT, PDF, CSV, JSON, Markdown, HTML, DOCX, XLSX, ZIP (recursive)

### 3. Real Literature Search  
Supports PubMed, Semantic Scholar, Tavily web search, and PDF ingestion.

### 4. Swarm Mode (2–32 Agents)  
Includes roles: Researcher, Critic, Explorer, Theorist, Integrator.  
All share memory and run coordinated TGRM cycles.

### 5. Continuous Long-Run Operation  
- Checkpointed state  
- Watchdog heartbeat  
- Crash-proof resume  
- Wall-clock runtime control  
- 10M-cycle safety ceiling  

### 6. Domain Presets  
Includes:  
- General Research  
- Longevity / Anti-aging  
- Math / Theory  

Each preset defines goals, source rules, RYE weighting, role bias, output style, and runtime profiles.

---

## 🧠 How the Agent Works

Each TGRM cycle:

1. TEST — detect contradictions, missing info  
2. DETECT — identify targets for correction  
3. REPAIR — apply transformations, ingestion, citations, refinements  
4. VERIFY — compute ΔR, E, RYE  
5. LOG — save cycle history + run_state  
6. LOOP — continue until time or RYE threshold ends run  

---

## 📂 Project Structure

autonomous-research-agent/  
├── agent/  
│   ├── core_agent.py  
│   ├── memory_store.py  
│   ├── presets.py  
│   ├── hypothesis_engine.py  
│   ├── rye_metrics.py  
│   ├── tools_papers.py  
│   ├── tools_files.py  
│   ├── tools_semantic_scholar.py  
│   ├── vector_memory.py  
│   └── report_generator.py  
├── app.py  
├── requirements.txt  
├── config/  
│   └── settings.yaml  
├── logs/  
│   └── sessions/  
└── .env   (Tavily key for local use)

---

## 🔑 Required API Keys

### Tavily (Required)  
Add via:  
- Sidebar input  
- `.env` file  
- Render environment variables  

Example:  
TAVILY_API_KEY=your_key_here

---

## 📊 Reports

Generates structured markdown including:  
- Summary  
- RYE metrics  
- Hypotheses  
- Citations  
- Notes  
- Biomarkers (longevity mode)

Download directly from the app.

---

## 🧪 Local Installation

git clone https://github.com/your-repo/autonomous-research-agent.git  
cd autonomous-research-agent  
pip install -r requirements.txt  
streamlit run app.py

Add your Tavily key before launching.

---

## 🌐 Deployment

Supports:  
- Render (recommended)  
- Streamlit Cloud  

This is a production-grade autonomous agent, not a demo.

---

# 📜 Licensing

This project uses a dual-license model to maximize adoption while protecting commercial value.

## Apache 2.0 (Open Core)

The core system — TGRM loop, RYE computation, ingestion engine, and single-agent mode — is licensed under Apache License 2.0.  
This encourages collaboration, research, and unrestricted use of the open components.

## Commercial License (Silt Pro / Enterprise / Swarm)

The following ARE NOT covered under Apache 2.0 and require a paid commercial license from Reparodynamics:

- Multi-agent swarm mode  
- Long-run background worker engine  
- Advanced presets and domain packs  
- Equilibrium and survival analytics  
- Anti-aging and longevity premium packs  
- Hosted cloud versions  
- Team and enterprise deployments  
- Proprietary optimizations  
- Any feature labeled “Pro”, “Enterprise”, or “Commercial”  

See the included LICENSE (Apache 2.0) and COMMERCIAL_LICENSE files.

---

## 🎉 Credits

Created by Cody R. Jenkins  
Reparodynamics • RYE • TGRM
