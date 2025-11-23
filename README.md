Autonomous Research Agent

Reparodynamics • TGRM • RYE • Swarm Mode
Live Production Deployment: https://autonomous-research-agent-hqby.onrender.com

Important
Paste your Tavily API key inside the sidebar or web search will use stub mode.

Scientific Attribution and Field Protection

Reparodynamics, TGRM, and RYE are scientific terms created and defined by Cody Ryan Jenkins.
These terms may not be rebranded, renamed, removed, or represented as originating from another creator.
Any use of these terms in scientific, academic, commercial, or technical work requires proper attribution and citation.

If you use this agent, or if your work references Reparodynamics, TGRM, or RYE, please cite:

Jenkins, C. R. (2025). Reparodynamics v2.1.
DOI: https://doi.org/10.5281/zenodo.17653046

Reparodynamics is a claimed scientific trademark of Cody Ryan Jenkins.

Overview

This is a full production autonomous research engine.
It is built around the Reparodynamics framework and includes:

• The TGRM cycle: Test, Detect, Repair, Verify
• RYE computation for every cycle
• Swarm Mode for coordinated multi agent research
• Browser tools
• Code sandbox
• Data connectors
• Background engine with watchdog and recovery

The system is designed for real scientific discovery and long duration autonomy.
It can run for one hour, eight hours, one day, ninety days, or indefinitely.

Major Capabilities

Full TGRM Loop
• Issue scanning
• Gap detection
• Targeted repair actions
• Verification with delta R
• RYE computation and logging

Web and Literature Tools
• Tavily browser
• PubMed
• Semantic Scholar
• PDF ingestion
• Local file ingestion
• CSV, JSON, XLSX, ZIP, GZ

Code Sandbox
Safe isolated execution of Python code.
Supports data cleaning, simulations, and validation steps.

Swarm Mode
Up to thirty two mini agents with roles like Researcher, Critic, Explorer, Theorist, and Integrator.
All write to a shared memory store.

Long Run Engine
• Background worker
• Checkpoints
• Auto resume
• Heartbeat watchdog
• Meta controller for exploration, stabilization, refinement

Domain Presets
• General Research
• Longevity
• Math
• Custom presets

Each preset controls goals, tools, tags, and runtime behavior.

Project Structure

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

Required Keys

Tavily

For real web search
TAVILY_API_KEY=your_key_here

Add it in the sidebar or in environment variables.
Without it, the agent uses offline stub search.

Running Locally

git clone your_repo_url_here
cd autonomous-research-agent
pip install -r requirements.txt
streamlit run app_streamlit.py

Paste your Tavily key into the UI sidebar.

How Continuous Mode Works

Each cycle includes:
1. Test
2. Detect
3. Repair
4. Verify
5. Log
6. Repeat

Runs can be one hour, eight hours, one day, ninety days, or forever.

Reporting Features

• RYE report
• Findings
• Cycle breakdown
• Tool usage logs
• Full session summary
• Downloadable reports

Deployment

Supports:

• Render Web Service
• Render Background Worker
• Streamlit Cloud

Background worker start command

python engine_worker.py

Environment variables
• WORKER_GOAL
• WORKER_RUNTIME_PROFILE
• WORKER_SWARM_ROLES
• WORKER_MAX_MINUTES
• WORKER_STOP_RYE
• WORKER_META
• WORKER_FOREVER

License

This project uses the Apache 2.0 license.
See LICENSE and NOTICE files.

Apache requirements mean that your copyright and attribution remain in every fork or redistribution.

Required Attribution Notice

Place this exact text in a file named NOTICE:

Reparodynamics, TGRM, and RYE are scientific terms created and defined by Cody Ryan Jenkins.
These terms may not be rebranded or represented as originating from another source.
Any scientific or academic use requires attribution to the original creator.
© 2025 Cody Ryan Jenkins

Apache requires this NOTICE file to be preserved in all forks and redistributions.

Credits

Created by Cody R. Jenkins
Reparodynamics • TGRM • RYE

If you want, I can also generate:
• LICENSE file
• NOTICE file
• Badges for GitHub
• A small citation card for your repo
• A banner image that shows Reparodynamics is protected
