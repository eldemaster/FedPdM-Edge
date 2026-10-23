#!/bin/bash
set -euo pipefail

# 1. Verify that pcubuntu's IP is 192.168.1.144
TARGET_IP="192.168.1.144"
if ! ip -4 addr show | grep -q "${TARGET_IP}"; then
    echo "ERROR: Local host IP is not ${TARGET_IP}. Aborting." >&2
    exit 1
fi
echo "Verified: Host IP includes ${TARGET_IP}"

# Configuration
FEDPDM_DIR="$HOME/fedpdm"
VENV_PYTHON="$HOME/venv/bin/python"
RPI1_HOST="eldemaster@192.168.1.147"
RPI2_HOST="eldemaster@192.168.1.115"
FL_SERVER="${TARGET_IP}:8081"
SEEDS=(42 100 2024 7 777)
TIMEOUT_SECONDS=1200  # 20 minutes timeout per seed

# Ensure results directory exists
mkdir -p "${FEDPDM_DIR}/results/fd002"

cd "${FEDPDM_DIR}"

for seed in "${SEEDS[@]}"; do
    echo "=================================================="
    echo "Starting federated run for SEED=${seed} at $(date)"
    echo "=================================================="

    # Pre-clean any running client processes on both Pis
    ssh -o ConnectTimeout=5 "${RPI1_HOST}" "pkill -f '[c]lient.py' || true" 2>/dev/null || true
    ssh -o ConnectTimeout=5 "${RPI2_HOST}" "pkill -f '[c]lient.py' || true" 2>/dev/null || true

    # Start server in background with SEED set and the venv python
    SERVER_LOG="${FEDPDM_DIR}/results/fd002/server_${seed}.log"
    PYTHONUNBUFFERED=1 SEED=${seed} "${VENV_PYTHON}" -u server.py > "${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!
    echo "Server started (PID: ${SERVER_PID}), logging to ${SERVER_LOG}"
    ln -sf "server_${seed}.log" "${FEDPDM_DIR}/results/fd002/server_seed.log"

    # Wait 5 seconds
    echo "Waiting 5 seconds before launching clients..."
    sleep 5

    # Start client on 192.168.1.147 (CLIENT_ID=1)
    echo "Launching Client 1 on ${RPI1_HOST}..."
    ssh -f -n "${RPI1_HOST}" "cd ~/fedpdm_client && nohup env PYTHONUNBUFFERED=1 SEED=${seed} CLIENT_ID=1 FL_SERVER=${FL_SERVER} ~/venv/bin/python -u client.py > client_1.log 2>&1 < /dev/null &"

    # Start client on 192.168.1.115 (CLIENT_ID=2)
    echo "Launching Client 2 on ${RPI2_HOST}..."
    ssh -f -n "${RPI2_HOST}" "cd ~/fedpdm_client && nohup env PYTHONUNBUFFERED=1 SEED=${seed} CLIENT_ID=2 FL_SERVER=${FL_SERVER} ~/venv/bin/python -u client.py > client_2.log 2>&1 < /dev/null &"

    # Wait for server process to exit, with 20 minute timeout
    echo "Waiting for server (PID: ${SERVER_PID}) to complete (timeout: 20 min)..."
    elapsed=0
    timed_out=false
    while kill -0 "${SERVER_PID}" 2>/dev/null; do
        if [ "${elapsed}" -ge "${TIMEOUT_SECONDS}" ]; then
            echo "ERROR: Server timed out after 20 minutes for seed ${seed}!" >&2
            kill -9 "${SERVER_PID}" 2>/dev/null || true
            timed_out=true
            break
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done

    # Reap process
    wait "${SERVER_PID}" 2>/dev/null || true

    if [ "${timed_out}" = true ]; then
        echo "Seed ${seed} run timed out."
    else
        echo "Server for seed ${seed} completed in ~${elapsed}s."
    fi

    # Kill leftover clients on both Pis
    echo "Killing leftover clients on Pis..."
    ssh -o ConnectTimeout=5 "${RPI1_HOST}" "pkill -f '[c]lient.py' || true" 2>/dev/null || true
    ssh -o ConnectTimeout=5 "${RPI2_HOST}" "pkill -f '[c]lient.py' || true" 2>/dev/null || true

    echo "Completed seed ${seed} at $(date)"
    echo ""
done

echo "=================================================="
echo "All federated runs finished at $(date)"
echo "=================================================="
