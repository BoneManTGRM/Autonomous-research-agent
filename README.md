# ⭐ Autonomous Research Agent  
### Powered by Reparodynamics • TGRM • RYE • Swarm Mode

Live Deployment  
https://autonomous-research-agent-hqby.onrender.com

Important  
Add your Tavily key inside the sidebar. Without it, search will run in offline stub mode.

---

# 🔬 Scientific Attribution and Field Protection

**Reparodynamics**, **TGRM**, and **RYE** were created and defined by **Cody Ryan Jenkins**.  
These scientific terms may not be renamed, repurposed, rebranded, or represented as originating from any other individual, organization, or entity.

Any scientific, academic, commercial, or technical work that uses, references, or builds upon Reparodynamics or its components must include proper attribution and citation.

**Required citation**  
Jenkins, C. R. (2025). Reparodynamics v2.2  
DOI: https://doi.org/10.5281/zenodo.17653046

Reparodynamics is a claimed scientific trademark of Cody Ryan Jenkins.

This protects the identity and origin of the field while allowing open scientific collaboration.

---

# 🚀 Overview

The Autonomous Research Agent is a high autonomy research engine designed to operate using the principles of Reparodynamics. Its core purpose is to carry out real scientific exploration, structured self correction, and measurable knowledge improvement over long durations.

The system is not a toy, demo, or simple chatbot wrapper. It is engineered for extended autonomous operation and uses a layered repair process to refine knowledge in real time.

Key features include:

• The complete TGRM cycle (Test, Detect, Repair, Verify)  
• RYE computation per cycle to measure improvement per energy used  
• Multi agent Swarm Mode for coordinated research  
• Browser and paper ingestion tools for real information gathering  
• A robust memory system with vector retrieval  
• An isolated code sandbox for safe experimentation  
• A background worker capable of long continuous runs  
• A meta controller that adapts behavior across phases  

The system can run in short analytical bursts or multi day and multi month autonomous windows.

---

# ⚡ Major Capabilities and Extended Details

### Full TGRM Loop  
TGRM is the foundation of the agent.  
Each cycle performs:

• Test: Evaluate current knowledge, detect inconsistencies  
• Detect: Identify gaps, contradictions, missing information, or weak claims  
• Repair: Use targeted tools to gather data, run code, or ingest literature  
• Verify: Re evaluate the updated state and compute delta R  
• RYE: Calculate Repair Yield per Energy as a quantitative score of improvement  
• Log: Save cycle context for transparency and reproducibility  

This cycle repeats across the entire runtime.

### Web and Literature Tools  
The agent integrates multiple real information pipelines:

• Tavily browser for web search  
• PubMed queries for biomedical and biological literature  
• Semantic Scholar for technical and scientific papers  
• PDF ingestion to extract text from uploaded papers  
• Structured data ingestion for CSV, JSON, XLSX, TSV, ZIP, and GZ  

This allows the agent to ground its reasoning in external sources.

### Code Sandbox  
A safe isolated Python environment used for:

• Data cleaning  
• Numeric simulations  
• Statistical testing  
• Model evaluation  
• Plotting and transformation tasks  
• Verification of claims through computation  

This allows the agent to validate hypotheses using real code, not language model guesses.

### Swarm Mode  
Swarm mode enables a coordinated set of agents working together. Each agent has a defined research personality such as Researcher, Critic, Explorer, Theorist, or Integrator.

Agents can:

• Explore different hypotheses  
• Criticize or refine other agents’ claims  
• Independently run TGRM cycles  
• Write contributions to a shared memory  
• Build consensus across multiple reasoning styles  

This dramatically increases research coverage and reduces blind spots.

### Long Run Engine  
The long run engine supports extended autonomous research sessions. It includes:

• A background worker  
• Crash recovery logic  
• Automatic checkpointing  
• A heartbeat watchdog  
• A meta controller with phase transitions  
• Adjustable runtime limits  
• Forever mode for continuous operation  

The system is engineered for hours, days, or even multi month research windows.

### Domain Presets  
Each preset configures research behavior, tool permissions, data sources, and TGRM weighting. Included presets:

• General Science  
• Longevity and Anti Aging Research  
• Mathematics and Theory Development  
• Custom user defined profiles  

Presets allow rapid switching between research domains without reconfiguring the core engine.

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

Insert the key in the sidebar or set it as an environment variable.  
Without the key, the agent still operates but search becomes offline.

---

# 🧪 Running Locally

```
git clone your_repo_url_here
cd autonomous-research-agent
pip install -r requirements.txt
streamlit run app_streamlit.py
```

Once running, paste your Tavily API key into the sidebar.

---

# 🔁 How Continuous Mode Works

Every cycle includes:

1. Test  
2. Detect  
3. Repair  
4. Verify  
5. Compute RYE  
6. Log  
7. Repeat  

Runtime modes include:

• One hour analytical burst  
• Eight hour research session  
• Twenty four hour deep investigation  
• Ninety day long haul  
• Forever mode for unrestricted autonomous operation  

The long run engine adapts across exploration, stabilization, and refinement phases.

---

# 📊 Reporting and Analytics

The agent provides detailed research visualization including:

• RYE performance trends  
• Cycle by cycle delta R  
• Repair actions and tool usage  
• Evidence sources collected  
• Hypothesis evolution  
• Verification outcomes  
• Memory updates  
• Exportable session summaries  

Reports can be downloaded in Markdown for scientific archiving.

---

# ☁ Deployment

The system is ready to deploy on:

• Render Web Service for the user interface  
• Render Background Worker for long runs  
• Streamlit Cloud for simple deployment  

To start the worker:

```
python engine_worker.py
```

Key environment variables:

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
See LICENSE and NOTICE files.

Apache requires that attribution, copyright, and the contents of the NOTICE file remain in all forks and redistributions.

---

# 📘 Required NOTICE File

Create a file named NOTICE with the following content:

```
Reparodynamics, TGRM, and RYE are scientific terms created and defined by Cody Ryan Jenkins.
These terms may not be rebranded or represented as originating from another source.
Any scientific or academic use requires attribution to the original creator.
© 2025 Cody Ryan Jenkins
```

This notice protects the scientific identity and prevents rebranding.

---

# 🧑‍🔬 Credits

Created by **Cody R. Jenkins**  
Reparodynamics • TGRM • RYE  

---
