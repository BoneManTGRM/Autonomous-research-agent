#!/usr/bin/env bash
set -euo pipefail

echo "=== ARA Unified Service Start ==="

# -------------------------------------------------------------------
# Always run from the project root so agent.* imports work
# -------------------------------------------------------------------
# Render normally uses /opt/render/project/src as the working dir.
# We try that first, then fall back to the script directory.
cd /opt/render/project/src 2>/dev/null || cd "$(dirname "$0")"

echo "[start_unified] Current working directory: $(pwd)"

# -------------------------------------------------------------------
# Shared run directory
# Honor existing ARA_RUNS_DIR if Render or you set it, otherwise default
# -------------------------------------------------------------------
if [ -z "${ARA_RUNS_DIR:-}" ]; then
  export ARA_RUNS_DIR="/opt/render/project/src/runs"
  echo "[start_unified] ARA_RUNS_DIR not set, defaulting to $ARA_RUNS_DIR"
else
  echo "[start_unified] ARA_RUNS_DIR already set to $ARA_RUNS_DIR"
fi

echo "[start_unified] Preparing run directories under $ARA_RUNS_DIR"
mkdir -p \
  "$ARA_RUNS_DIR" \
  "$ARA_RUNS_DIR/pending" \
  "$ARA_RUNS_DIR/active" \
  "$ARA_RUNS_DIR/finished" \
  "$ARA_RUNS_DIR/error" \
  "$ARA_RUNS_DIR/queue" \
  "$ARA_RUNS_DIR/memory"

echo "[start_unified] Run directory tree:"
for d in \
  "$ARA_RUNS_DIR" \
  "$ARA_RUNS_DIR/active" \
  "$ARA_RUNS_DIR/finished" \
  "$ARA_RUNS_DIR/pending" \
  "$ARA_RUNS_DIR/queue" \
  "$ARA_RUNS_DIR/error" \
  "$ARA_RUNS_DIR/memory"
do
  if [ -d "$d" ]; then
    echo "  dir: $d"
  else
    echo "  missing_dir: $d"
  fi
done

# -------------------------------------------------------------------
# Tavily and web search env defaults
# These only set sane defaults if not already provided by Render env.
# BrowserTool will honor these flags.
# -------------------------------------------------------------------
if [ -z "${ENABLE_TAVILY:-}" ]; then
  export ENABLE_TAVILY=1
  echo "[start_unified] ENABLE_TAVILY not set, defaulting to 1 (enabled)"
else
  echo "[start_unified] ENABLE_TAVILY already set to ${ENABLE_TAVILY}"
fi

if [ -z "${TAVILY_STUB_MODE:-}" ]; then
  export TAVILY_STUB_MODE=0
  echo "[start_unified] TAVILY_STUB_MODE not set, defaulting to 0 (real search if key present)"
else
  echo "[start_unified] TAVILY_STUB_MODE already set to ${TAVILY_STUB_MODE}"
fi

if [ -z "${DISABLE_WEB_SEARCH:-}" ]; then
  export DISABLE_WEB_SEARCH=0
  echo "[start_unified] DISABLE_WEB_SEARCH not set, defaulting to 0"
else
  echo "[start_unified] DISABLE_WEB_SEARCH already set to ${DISABLE_WEB_SEARCH}"
fi

if [ -z "${TAVILY_RPS:-}" ]; then
  echo "[start_unified] TAVILY_RPS not set, BrowserTool will use its internal default"
else
  echo "[start_unified] TAVILY_RPS already set to ${TAVILY_RPS}"
fi

if [ -z "${BROWSER_CACHE_TTL_SECONDS:-}" ]; then
  export BROWSER_CACHE_TTL_SECONDS=600
  echo "[start_unified] BROWSER_CACHE_TTL_SECONDS not set, defaulting to 600"
else
  echo "[start_unified] BROWSER_CACHE_TTL_SECONDS already set to ${BROWSER_CACHE_TTL_SECONDS}"
fi

# Do not echo the full key for safety
if [ -n "${TAVILY_API_KEY:-}" ]; then
  echo "[start_unified] Tavily key detected (tail: ${TAVILY_API_KEY: -4})"
else
  echo "[start_unified] No Tavily API key detected, BrowserTool will run in stub mode"
fi

# -------------------------------------------------------------------
# Environment for worker
# -------------------------------------------------------------------
export PYTHONUNBUFFERED=1

# Force queue mode engine so engine_worker runs the queue worker
export WORKER_MODE="queue"
export WORKER_QUEUE_MODE="1"

echo "[start_unified] Python binary: $(which python || echo 'python not found')"
python - << 'EOF' || echo "[start_unified] Warning: sanity check failed"
import os, sys
print("[start_unified] Python version:", sys.version.replace("\n", " "))
print("[start_unified] Sanity check: ARA_RUNS_DIR env =", os.getenv("ARA_RUNS_DIR"))
try:
    from agent import run_jobs
    print("[start_unified] Sanity check: run_jobs.BASE_DIR =", run_jobs.BASE_DIR)
except Exception as e:
    print("[start_unified] Sanity check: could not import agent.run_jobs:", e)
EOF

# -------------------------------------------------------------------
# Cleanup trap so worker is killed when container exits
# -------------------------------------------------------------------
cleanup() {
  if [ "${WORKER_PID-}" != "" ]; then
    echo "[start_unified] Stopping engine worker PID $WORKER_PID"
    kill "$WORKER_PID" 2>/dev/null || true
    wait "$WORKER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# -------------------------------------------------------------------
# Start engine worker in background
# -------------------------------------------------------------------
echo "[start_unified] Starting engine worker in queue mode..."
python engine_worker.py &
WORKER_PID=$!
echo "[start_unified] Engine worker PID: $WORKER_PID"

# -------------------------------------------------------------------
# Start Streamlit UI in foreground
# Render provides \$PORT, but default to 8501 if missing
# -------------------------------------------------------------------
PORT="${PORT:-8501}"
echo "[start_unified] Starting Streamlit UI on port $PORT"

exec streamlit run app_streamlit.py \
  --server.port "$PORT" \
  --server.address 0.0.0.0
