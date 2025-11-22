# Autonomous Research Agent  
_Reparodynamics • TGRM • RYE • Swarm Mode_  
**Live App:** https://autonomous-research-agent-hqby.onrender.com  
**⚠️ IMPORTANT:** You MUST provide **your own Tavily API key** inside the app sidebar or the agent will not run.

---

## 🔥 Overview

The **Autonomous Research Agent** is a fully self-contained TGRM-powered system capable of:

- Running **continuous scientific research** using the Reparodynamics framework  
- Computing **RYE (Repair Yield per Energy)** every cycle  
- Detecting contradictions, gaps, issues, and repairing them  
- Running for **90 days safely** with watchdog, checkpointed run_state, crash-proof resume  
- Operating as a **single agent**, **multi-agent duo**, or **full swarm (2–32 agents)**  
- Executing **Test → Detect → Repair → Verify** autonomously  
- Ingesting papers, datasets, PDFs, CSVs, semantic queries, biomarker datasets, and more  
- Scaling to millions of cycles with stability guards

---

## 🚀 Live Demo (Render + Streamlit)

The app is hosted using **Render + Streamlit**:

👉 **https://autonomous-research-agent-hqby.onrender.com**

⚠️ **The app will not run unless you provide your own Tavily API key.**  
You can add it in the left sidebar under **“Tavily API Key”**.

---

## ⚙️ Major Features

### **1. Full TGRM Loop Implementation**
- Automated issue detection  
- Repair actions  
- Verification passes  
- RYE computation (`ΔR / E`)  
- Supports domain weighting (math / longevity / general)

---

### **2. Multi-Format Ingestion Engine**
Your agent can read:

- TXT  
- PDF (real extraction)  
- CSV → DataFrame summaries  
- JSON  
- Markdown  
- HTML  
- DOCX  
- XLSX  
- ZIP (recursive extraction)

---

### **3. Real Literature Search**
Includes:

- PubMed ingestion  
- Semantic Scholar ingestion  
- Web search via Tavily (requires your key)  
- PDF ingestion (remote or uploaded)

All returned in normalized citation format for RYE scoring.

---

### **4. Swarm Mode (2–32 Agents)**  
Roles include:

- **Researcher**  
- **Critic**  
- **Explorer**  
- **Theorist**  
- **Integrator**

Each performs specialized TGRM cycles with shared memory.

---

### **5. 90-Day Safe Continuous Operation**
Includes:

- Checkpointed `run_state`  
- Watchdog heartbeat  
- Crash-proof resume  
- 10M-cycle safety cap  
- Real wall-clock control using `max_minutes`

---

### **6. Domain Presets**
Presets include:

- **General Research**  
- **Longevity / Anti-aging**  
- **Math / Theory**  

Each preset defines:

- Default goal  
- Source usage rules  
- Role bias  
- RYE weighting  
- Reporting style  
- Cycle tuning  
- Runtime profiles (1h, 8h, 24h, 90 days, Forever)

---

## 📂 Project Structure

```
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
│   ├── vector_memory.py   (optional)
│   └── report_generator.py
├── app.py  (or streamlit_app.py)
├── requirements.txt
├── config/
│   └── settings.yaml
├── logs/
│   └── sessions/
├── README.md
└── .env (your Tavily key if using locally)
```

---

## 🔑 Required API Keys

### **Tavily**
Required for real web research.

The app will not run without it.

Place it in either:

- Streamlit sidebar text box  
- `.env` file  
- Render environment variable panel  

Example:

```
TAVILY_API_KEY=your_key_here
```

---

## 🧠 How the Agent Works

Each cycle performs:

1. **TEST** → Detect issues, contradictions, missing info  
2. **DETECT** → Identify targets for repair  
3. **REPAIR** → Apply transformations, corrections, ingestion, citations, hypotheses  
4. **VERIFY** → Compute ΔR, Energy, and RYE  
5. **LOG** → Save cycle history + run_state + watchdog  
6. **LOOP** → Continue until minutes or RYE threshold stops it

---

## 📊 Reports

The agent generates structured markdown reports including:

- Summary  
- RYE statistics  
- Hypotheses  
- Citations  
- Notes  
- Biomarkers (longevity preset)

You can download the report directly from the app.

---

## 🧪 Local Installation

```
git clone https://github.com/your-repo/autonomous-research-agent.git
cd autonomous-research-agent
pip install -r requirements.txt
streamlit run app.py
```

Add your Tavily key before running.

---

## 🌐 Deployment

This app runs on:

- **Render** (hosted server)  
- **Streamlit Cloud** (UI runtime)

The version you are using live is deployed on **Render**.

---

## 📜 License
MIT or your custom license.

---

## 🎉 Credits
Created by Cody R. Jenkins  
Reparodynamics • RYE • TGRM
