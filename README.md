# Autonomous Research Agent  
### Powered by Reparodynamics • RYE • TGRM • Swarm Mode

This project is a full autonomous research engine designed to run safely for **1 hour, 8 hours, 24 hours, 90 days, or Forever**, using the Reparodynamics framework:

- **TGRM** — Targeted Gradient Repair Mechanism  
- **RYE** — Repair Yield per Energy  
- **ΔR / E** performance tracking  
- **Test → Detect → Repair → Verify** closed-loop autonomy  
- **Equilibrium detection**, **trend analysis**, and **continuous operation**

It supports:
- Full file ingestion (PDF, CSV, JSON, DOCX, XLSX, ZIP, MD, HTML)
- Web search (Tavily)
- PubMed search
- Semantic Scholar search
- PDF parsing
- Hypothesis generation
- Cycle logging
- Biomarker mode
- Multi-Agent (Researcher + Critic)
- Full **Swarm Mode** (2–32 agents)
- Domain presets (General, Longevity, Math)
- Real-time RYE charts, delta_R, energy, citations, and summaries
- Continuous 90-day autonomous operation (crash-resistant)

---

## 🚀 Features

### **1. TGRM Autonomous Loop**
Every cycle runs:
1. **Test**: detect missing info, contradictions, gaps  
2. **Detect**: identify issues using notes, citations, metrics  
3. **Repair**: apply fixes, fetch data, generate hypotheses  
4. **Verify**: compute ΔR, energy E, and RYE = ΔR / E  

### **2. True Multi-Format File Ingestion**
Supports real parsing for:
- `.txt`
- `.pdf`
- `.csv`
- `.json`
- `.md`
- `.html`
- `.docx`
- `.xlsx`
- `.zip`

Includes safe fallbacks, size limits, caching, and protective timeouts.

### **3. Domain-Aware Intelligence**
Three built-in presets:
- **General**  
- **Longevity / Biomedicine**  
- **Mathematics / Theory**  

Each preset defines:
- Keywords  
- Default goals  
- Source usage rules  
- RYE weighting  
- Role bias  
- Reporting style  

### **4. Swarm Mode (2–32 Agents)**
Roles include:
- Researcher
- Critic
- Explorer
- Theorist
- Integrator

Each one performs specialized TGRM cycles with shared memory.

### **5. 90-Day Safe Continuous Operation**
Includes:
- Checkpointed run_state
- Watchdog heartbeat
- Crash-proof resume
- 10M cycle safety cap
- Real wall-clock control using max_minutes

---

## 📂 Project Structure
