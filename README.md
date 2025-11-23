# Autonomous Research Agent  
Reparodynamics • TGRM • RYE • Swarm Mode  
**Live Production Deployment:** https://autonomous-research-agent-hqby.onrender.com  

⚠️ IMPORTANT  
Paste your **Tavily API key** inside the sidebar or web search will use stub mode.

---

# 🔥 Overview

This is a **full production autonomous research engine**, powered by:

- **TGRM** (Test → Detect → Repair → Verify)
- **RYE** (Repair Yield per Energy)
- **Reparodynamic cycle analysis**
- **Swarm Mode (2–32 coordinated agents)**
- **Browser tool**
- **Code sandbox**
- **Data connectors**
- **Background worker with watchdog + crash-recovery**

This system is built for **real scientific research**, not demos.  
Runs for **1h, 8h, 24h, 90 days, or Forever**.

---

# ⚙️ Major Capabilities

### ✔ Full TGRM Loop  
- Issue scanning  
- Hypothesis generation  
- Targeted repair  
- Verification step  
- ΔR and Energy tracking  
- RYE computation every cycle  

### ✔ Web + Literature Tools  
- Tavily browser  
- PubMed  
- Semantic Scholar  
- PDF ingestion  
- Local file ingestion  
- Data connectors for CSV, JSON, XLSX, ZIP, GZ  

### ✔ Code Sandbox  
Safe, isolated execution of:  
- Python snippets  
- Data transformations  
- Mini-simulations  
- Micro-experiments  

### ✔ Swarm Mode (up to 32 mini-agents)  
Roles include:  
- Researcher  
- Critic  
- Explorer  
- Theorist  
- Integrator  

Each runs its own TGRM cycles and all write to shared memory.

### ✔ Long-Run Engine (Background Worker)  
- Runs without UI  
- Checkpointed state  
- Resume after crash or restart  
- Watchdog heartbeat  
- Meta-controller (exploration → stabilization → refinement loops)  

### ✔ Domain Presets  
- **General Research**  
- **Longevity / Biology / Aging**  
- **Math / Theory**  
- **Custom profiles**  
Each preset controls:
- Default goals  
- Source configuration  
- Domain tags  
- Runtime profiles  
- RYE weighting hints  
- Tool permissions  

---

# 📂 Project Structure (Updated)

```
autonomous-research-agent/
├── agent/
│   ├── core_agent.py
│   ├── tgrm_loop.py
│   ├── hypothesis_engine.py
│   ├── rye_metrics.py
│   ├── memory_store.py
│   ├── presets.py
│   ├── report_generator.py
│   ├── vector_memory.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── browser_tool.py
│   │   ├── code_sandbox.py
│   │   ├── data_connectors.py
│   │   ├── tools_web.py
│   │   ├── tools_files.py
│   │   ├── tools_papers.py
│   │   ├── tools_pubmed.py
│   │   ├── tools_semantic_scholar.py
│   │   └── ...
│   └── biomarker_analyzer.py
├── engine_worker.py
├── app_streamlit.py
├── config/
│   └── settings.yaml
├── logs/
│   └── sessions/
└── requirements.txt
```

---

# 🔑 Required Keys

### ✔ Tavily  
For real web search:  
```
TAVILY_API_KEY=your_key_here
```
Can be added:
- Inside the UI sidebar  
- In `.env`  
- As Render environment variable  

Without it, the agent uses **offline stub search**.

---

# 🧪 Running Locally

```bash
git clone https://github.com/your-repo/autonomous-research-agent.git
cd autonomous-research-agent
pip install -r requirements.txt
streamlit run app_streamlit.py
```

Paste your Tavily key in the sidebar.

---

# 🧠 How Continuous Mode Works

Every cycle:

1. **TEST** – scan for contradictions, missing data, gaps  
2. **DETECT** – identify actionable issues  
3. **REPAIR** – execute targeted repairs using tools  
4. **VERIFY** – compute ΔR, E, RYE  
5. **LOG** – write to disk with timestamp  
6. **LOOP** – continue until time budget or collapse  

Supports:
- 1 hour  
- 8 hours  
- 24 hours  
- 90 days  
- Forever  

---

# 📊 Reporting

The UI provides:
- Full RYE report  
- Findings (cures, treatments, mechanisms)  
- Outcome summary  
- Tool usage visibility  
- Cycle breakdown  

All downloadable as Markdown.

---

# 🌐 Deployment (Render / Cloud)

Ready for:
- Render Web Service (UI)  
- Render Background Worker (continuous engine)  
- Streamlit Cloud  

Background worker starts with:
```bash
python engine_worker.py
```

Supports environment variables:
- WORKER_GOAL  
- WORKER_RUNTIME_PROFILE  
- WORKER_SWARM_ROLES  
- WORKER_MAX_MINUTES  
- WORKER_STOP_RYE  
- WORKER_META  
- WORKER_FOREVER  

---

# 📜 Licensing

This project uses a **hybrid open core + commercial** model.

## Open Core — Apache 2.0  
Free for:
- Core agent  
- TGRM loop  
- RYE metrics  
- Basic ingestion  
- Single-agent mode  

## Commercial License (Reparodynamics Enterprise)
Required for:
- Swarm mode  
- Background worker  
- Meta-controller engine  
- Domain packs  
- Hosted cloud offering  
- Premium optimizations  
- Any “Pro” or “Enterprise” feature  

See LICENSE and COMMERCIAL_LICENSE.

---

# 🎉 Credits
Created by **Cody R. Jenkins**  
Reparodynamics • RYE • TGRM
