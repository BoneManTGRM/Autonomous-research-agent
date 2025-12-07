#!/usr/bin/env bash
set -e

echo "=== ARA Unified Service Start ==="

# -------------------------------------------------------------------
# Ensure ARA_RUNS_DIR is set (Render passes it, but we guard anyway)
# -------------------------------------------------------------------
if [ -z "$ARA_RUNS_DIR" ]; then
  export ARA_RUNS_DIR="/opt/render/project/src/runs"
  echo "ARA_RUNS_DIR not set. Defaulting to $ARA_RUNS_DIR"
fi

echo "ARA_RUNS_DIR = $ARA_RUNS_DIR"

# -------------------------------------------------------------------
# Create required queue folders
# -------------------------------------------------------------------
echo "Ensuring run directories exist..."
mkdir -p "$ARA_RUNS_DIR/pending"
mkdir -p "$ARA_RUNS_DIR/active"
mkdir -p "$ARA_RUNS_DIR/finished"
mkdir -p "$ARA_RUNS_DIR/error"
mkdir -p "$ARA_RUNS_DIR/queue"   # legacy shadow folder

ls -R "$ARA_RUNS_DIR"

# -------------------------------------------------------------------
# FORCE worker into queue mode
# -------------------------------------------------------------------
export WORKER_QUEUE_MODE="1"
export WORKER_MODE="queue"

echo "Starting engine worker with queue mode..."
python engine_worker.py &
WORKER_PID=$!
echo "Engine worker started with PID $WORKER_PID"

# -------------------------------------------------------------------
# STREAMLIT UI (foreground — keeps service alive)
# -------------------------------------------------------------------
echo "Starting Streamlit UI..."
exec streamlit run app_streamlit.py \
    --server.port "$PORT" \
    --server.address 0.0.0.0
