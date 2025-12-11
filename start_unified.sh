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
# Environment for worker
# -------------------------------------------------------------------
export PYTHONUNBUFFERED=1

# Force queue mode engine so engine_worker runs the queue worker
export WORKER_MODE="queue"
export WORKER_QUEUE_MODE="1"

echo "[start_unified] Python binary: $(which python || echo 'python not found')"

# Optional sanity check: what does agent.run_jobs think BASE_DIR is
python - << 'EOF' || echo "[start_unified] Warning: sanity check failed"
import os
try:
    from agent import run_jobs
    print("[start_unified] Sanity check: run_jobs.BASE_DIR =", run_jobs.BASE_DIR)
    print("[start_unified] Sanity check: ARA_RUNS_DIR env  =", os.getenv("ARA_RUNS_DIR"))
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
