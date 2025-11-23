# ⭐ Autonomous Research Agent  
### Powered by Reparodynamics • TGRM • RYE • Swarm Mode

Live Deployment  
https://autonomous-research-agent-hqby.onrender.com

Important  
Add your Tavily key in the sidebar or search will run in stub mode.

---

# 🔬 Scientific Attribution and Field Protection

**Reparodynamics**, **TGRM**, and **RYE** were created and defined by **Cody Ryan Jenkins**.  
These scientific terms may not be renamed, rebranded, removed, or represented as originating from any other creator.

Any scientific, academic, commercial, or technical use must include attribution and citation.

**Required citation**  
Jenkins, C. R. (2025). Reparodynamics v2.2  
DOI: https://doi.org/10.5281/zenodo.17653046

Reparodynamics is a claimed scientific trademark of Cody Ryan Jenkins.

---

# 🚀 Overview

This project is a full production autonomous research engine built on the Reparodynamics framework.

It includes:

• Full TGRM cycle  
• RYE computation per cycle  
• Multi agent Swarm Mode  
• Browser tools  
• Code sandbox  
• Data ingestion  
• Background worker with watchdog  
• Long run survivability modes  

Designed for real scientific discovery and high autonomy.  
Runs for one hour, eight hours, one day, ninety days, or indefinitely.

---

# ⚡ Major Capabilities

### TGRM Loop  
• Issue scanning  
• Gap detection  
• Targeted repair steps  
• Verification with delta R  
• RYE computation and logging  

### Web and Literature Tools  
• Tavily browser  
• PubMed  
• Semantic Scholar  
• PDF ingestion  
• CSV, JSON, XLSX, ZIP, GZ  

### Code Sandbox  
Isolated Python execution for transformations and simulations.

### Swarm Mode  
Up to thirty two coordinated mini agents:  
Researcher, Critic, Explorer, Theorist, Integrator  
All writing to shared memory.

### Long Run Engine  
• Background worker  
• Crash recovery  
• Heartbeat watchdog  
• Meta controller for exploration, stabilization, refinement  

### Domain Presets  
• General research  
• Longevity  
• Math  
• Custom profiles  

Each preset controls tools, goals, tags, and runtime behavior.

---

# 📁 Project Structure

```
autonomous-research-agent/
    agent/
        core_agent.py
        tgrm_loop.py
        hypothesis_engine.py
        rye_metrics.py
        memory_store.py
        presets.py
        report_generator.py
        vector_memory.py
        tools/
            browser_tool.py
            code_sandbox.py
            data_connectors.py
            tools_web.py
            tools_files.py
            tools_papers.py
            tools_pubmed.py
            tools_semantic_scholar.py
    engine_worker.py
    app_streamlit.py
    config/
        settings.yaml
    logs/
        sessions/
    requirements.txt
```

---

# 🔑 Required Keys

### Tavily Search  
```
TAVILY_API_KEY=your_key_here
```

Add it in the UI sidebar or as an environment variable.  
Without this key, the agent uses offline stub search.

---

# 🧪 Running Locally

```
git clone your_repo_url_here
cd autonomous-research-agent
pip install -r requirements.txt
streamlit run app_streamlit.py
```

Paste your Tavily key into the UI sidebar.

---

# 🔁 How Continuous Mode Works

Each cycle performs:  
1. Test  
2. Detect  
3. Repair  
4. Verify  
5. Log  
6. Repeat  

Run modes  
One hour  
Eight hours  
One day  
Ninety days  
Forever

---

# 📊 Reporting

• RYE report  
• Findings and insights  
• Tool usage metrics  
• Cycle breakdown  
• Full session summary  
• Downloadable reports  

---

# ☁ Deployment

Compatible with:

• Render Web Service  
• Render Background Worker  
• Streamlit Cloud  

Background worker command  
```
python engine_worker.py
```

Environment variables  
WORKER_GOAL  
WORKER_RUNTIME_PROFILE  
WORKER_SWARM_ROLES  
WORKER_MAX_MINUTES  
WORKER_STOP_RYE  
WORKER_META  
WORKER_FOREVER  

---

# 📜 License

This project uses the Apache 2.0 license.  
See LICENSE and NOTICE files in this repository.

Apache requires that your attribution and NOTICE content remain in all forks and redistributions.

---

# 📘 Required NOTICE File

Create a file in the project root named NOTICE and place this inside:

```
Reparodynamics, TGRM, and RYE are scientific terms created and defined by Cody Ryan Jenkins.
These terms may not be rebranded or represented as originating from another source.
Any scientific or academic use requires attribution to the original creator.
© 2025 Cody Ryan Jenkins
```

Apache requires that this NOTICE file be preserved in all forks and redistributions.

---

# 🧑‍🔬 Credits

Created by **Cody R. Jenkins**  
Reparodynamics • TGRM • RYE
