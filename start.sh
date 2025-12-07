#!/usr/bin/env bash
set -e

echo "=== ARA Unified Service Start ==="

# -------------------------------------------------------------------
# Hard-set canonical runs directory so UI and worker ALWAYS match.
# Ignore any mismatched Render env var.
# -------------------------------------------------------------------
APP_ROOT="/opt/render/project/src"
export ARA_RUNS_DIR="$APP_ROOT/runs"

echo "APP_ROOT     = $APP_ROOT"
echo "ARA_RUNS_DIR = $ARA_RUNS_DIR"

# -------------------------------------------------------------------
# Reset and create required queue folders
# -------------------------------------------------------------------
echo "Resetting run directories..."
rm -rf "$ARA_RUNS_DIR"
mkdir -p "$ARA_RUNS_DIR/pending"
mkdir -p "$ARA_RUNS_DIR/active"
mkdir -p "$ARA_RUNS_DIR/finished"
mkdir -p "$ARA_RUNS_DIR/error"
mkdir -p "$ARA_RUNS_DIR/queue"   # legacy shadow support

echo "Directory tree:"
ls -R "$ARA_RUNS_DIR"

# -------------------------------------------------------------------
# FORCE worker into queue mode
# -------------------------------------------------------------------
export WORKER_QUEUE_MODE="1"
export WORKER_MODE="queue"

echo "Starting engine worker in QUEUE MODE..."
python engine_worker.py &
WORKER_PID=$!
echo "Engine worker started with PID $WORKER_PID"

# -------------------------------------------------------------------
# STREAMLIT UI (foreground, keeps service alive)
# -------------------------------------------------------------------
echo "Starting Streamlit UI..."
exec streamlit run app_streamlit.py \
    --server.port "$PORT" \
    --server.address 0.0.0.0
