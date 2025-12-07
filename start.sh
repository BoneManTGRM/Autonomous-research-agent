#!/usr/bin/env bash
set -e

echo "=== ARA Unified Service Start ==="
echo "ARA_RUNS_DIR = $ARA_RUNS_DIR"
echo "Starting engine worker and Streamlit..."

# Ensure run folders exists (safety)
mkdir -p "$ARA_RUNS_DIR/pending"
mkdir -p "$ARA_RUNS_DIR/active"
mkdir -p "$ARA_RUNS_DIR/finished"
mkdir -p "$ARA_RUNS_DIR/error"
mkdir -p "$ARA_RUNS_DIR/queue"   # legacy shadow folder

# --- Start the engine worker in background ---
python engine_worker.py &
WORKER_PID=$!
echo "Engine worker started with PID $WORKER_PID"

# --- Start Streamlit UI in foreground (keeps service alive) ---
exec streamlit run app_streamlit.py \
    --server.port "$PORT" \
    --server.address 0.0.0.0
