# Autonomous Research Agent  
Reparodynamics • RYE • TGRM • Swarm Agents • 90-Day Autonomous Runs

The Autonomous Research Agent is a next-generation self-repairing research engine designed using the principles of **Reparodynamics**, including  
**RYE (Repair Yield per Energy)** and **TGRM (Targeted Gradient Repair Mechanism)**.

It performs:
- Literature research (web, PDF, Semantic Scholar, PubMed)
- Automatic hypothesis generation
- Multi-agent reasoning (Researcher + Critic)
- Full Swarm Mode (up to 32 specialized agents)
- 24h, 8h, 1h, 90-day, and Forever autonomous runs
- Continuous RYE-based adaptive repair
- Long-term memory + semantic vector memory
- Full Streamlit UI

---

# 🚀 **Live App (Render + Streamlit)**

### 👉 **https://autonomous-research-agent-hqby.onrender.com**

The app runs fully in the cloud on **Render + Streamlit**.

## ❗ IMPORTANT — You MUST Supply Your Own Tavily API Key

Without a Tavily key:
- **The app will not run**
- The "Run Agent" button will not execute cycles
- Web search & ingestion will fail

Enter your key in the sidebar under **“Tavily API Key”**.

Get a free key here:  
https://app.tavily.com

---

# 🧬 Key Concepts

### **Reparodynamics**
A universal science of stability and self-repair.

### **RYE – Repair Yield per Energy**
RYE = ΔR / E  
How much improvement is produced per unit of effort.

### **TGRM – Targeted Gradient Repair Mechanism**
The 4-phase repair loop:
1. Test  
2. Detect  
3. Repair  
4. Verify  

### **Swarm Mode**
The agent can run dozens of coordinated roles:
- Researcher  
- Critic  
- Explorer  
- Theorist  
- Integrator  

Each writes repairs into shared memory.

---

# 🌐 Major Features

### ✔ Real PDF ingestion  
### ✔ Semantic Scholar search  
### ✔ PubMed ingestion  
### ✔ Web research  
### ✔ DOCX / XLSX / HTML / ZIP support  
### ✔ 90-day safe autonomous mode  
### ✔ Continuous memory with repair indexing  
### ✔ Rolling RYE metrics, regression slope, efficiency charts  
### ✔ Full Markdown report generator  
### ✔ Upload your own papers for analysis  
### ✔ Multi-role or Swarm execution  

---

# 📁 Project Structure

Below is the recommended folder structure for the repository:

```
autonomous-research-agent/
│
├── agent/
│   ├── core_agent.py
│   ├── memory_store.py
│   ├── rye_metrics.py
│   ├── hypothesis_engine.py
│   ├── tools_files.py
│   ├── tools_papers.py
│   ├── presets.py
│   └── vector_memory.py   (optional)
│
├── ui/
│   └── app_streamlit.py
│
├── logs/
│   └── sessions/
│
├── config/
│   └── settings.yaml
│
├── README.md
└── requirements.txt
```

---

# 🖼 Project Structure Diagram  
*(Your uploaded image will appear here on GitHub)*

![Project Structure](EC99A08B-1FBC-48A0-9F9D-B9EF643FC881.jpeg)

---

# 🔑 Environment Variables

You only need **one**:

```
TAVILY_API_KEY=your_key_here
```

You can set it in:

- `.env`
- Render dashboard → Environment Variables
- Directly in the Streamlit sidebar (per-user secret)

---

# 🏁 Running Locally

```
pip install -r requirements.txt
streamlit run ui/app_streamlit.py
```

Then open:

```
http://localhost:8501
```

---

# 📄 Generating Reports

The UI includes:
- Full-cycle history export  
- Markdown report generator  
- Download button  

Reports include:
- RYE statistics  
- Notes  
- Hypotheses  
- Citations  
- Trend lines  

---

# 🔮 Vision

The agent demonstrates the first working **software implementation of Reparodynamics**:

- RYE as a live metric  
- TGRM as a self-repair loop  
- Swarms as coordinated stability systems  
- Continuous 90-day autonomous operation  

This is the foundation of a fully general, self-repairing scientific engine.

---

# 📬 Contact

For collaboration, extensions, validation studies, or research partnerships:
**Cody R. Jenkins – Reparodynamics Open Science Initiative**

---
