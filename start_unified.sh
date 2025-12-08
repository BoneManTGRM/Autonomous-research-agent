#!/usr/bin/env bash
set -euo pipefail

echo "=== ARA Unified Service Start ==="

# -------------------------------------------------------------------
# Always run from the project root so agent.* imports work
# -------------------------------------------------------------------
# Render normally uses /opt/render/project/src as the working dir,
# but we force it to be safe both locally and on Render.
cd /opt/render/project/src 2>/dev/null || cd "$(dirname "$0")"

echo "Current working directory: $(pwd)"

# -------------------------------------------------------------------
# Shared run directory
# Honor existing ARA_RUNS_DIR if Render or you set it, otherwise default
# -------------------------------------------------------------------
if [ -z "${ARA_RUNS_DIR:-}" ]; then
  export ARA_RUNS_DIR="/opt/render/project/src/runs"
  echo "ARA_RUNS_DIR not set, defaulting to $ARA_RUNS_DIR"
else
  echo "ARA_RUNS_DIR already set to $ARA_RUNS_DIR"
fi

echo "Preparing run directories under $ARA_RUNS_DIR"
mkdir -p \
  "$ARA_RUNS_DIR/pending" \
  "$ARA_RUNS_DIR/active" \
  "$ARA_RUNS_DIR/finished" \
  "$ARA_RUNS_DIR/error" \
  "$ARA_RUNS_DIR/queue"

ls -R "$ARA_RUNS_DIR" || true

# -------------------------------------------------------------------
# Environment for worker
# -------------------------------------------------------------------
export PYTHONUNBUFFERED=1

# Force queue mode engine so engine_worker runs run_job_queue_worker
export WORKER_MODE="queue"
export WORKER_QUEUE_MODE="1"

echo "Starting engine worker in queue mode..."
python engine_worker.py &
WORKER_PID=$!
echo "Engine worker PID: $WORKER_PID"

# -------------------------------------------------------------------
# Start Streamlit UI in foreground
# Render provides $PORT, but default to 8501 if missing
# -------------------------------------------------------------------
PORT="${PORT:-8501}"
echo "Starting Streamlit UI on port $PORT"

exec streamlit run app_streamlit.py \
  --server.port "$PORT" \
  --server.address 0.0.0.0
