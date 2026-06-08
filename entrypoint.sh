#!/usr/bin/env bash
#
# Launch the two processes that make up a mock model-server pod: the real vLLM Rust
# frontend (vllm-rs) and our mock engine-core backend (mock-engine-nixl), wired over
# the in-pod engine-core handshake. Role, identity, and ports come from the env so the
# same image serves as a prefill or a decode instance in the llm-d P/D path.
set -euo pipefail

MODEL="${MODEL:?MODEL env required (HF model id, used for the tokenizer)}"
ROLE="${MOCK_PD_ROLE:-both}"                 # prefill | decode | both
ENGINE_ID="${POD_NAME:-mock-engine-0}"       # advertised as remote_engine_id
SIDE_CHANNEL_HOST="${POD_IP:-0.0.0.0}"       # advertised as remote_host
SIDE_CHANNEL_PORT="${MOCK_SIDE_CHANNEL_PORT:-5600}"
HANDSHAKE_PORT="${MOCK_HANDSHAKE_PORT:-29550}"

# Decode sits behind the routing sidecar (sidecar :8000 -> vllm-rs :8200); prefill is
# hit directly on :8000. Override with VLLM_PORT if needed.
case "$ROLE" in
    decode) HTTP_PORT="${VLLM_PORT:-8200}" ;;
    *)      HTTP_PORT="${VLLM_PORT:-8000}" ;;
esac

echo "[entrypoint] role=$ROLE model=$MODEL engine_id=$ENGINE_ID http=:$HTTP_PORT nixl=$SIDE_CHANNEL_HOST:$SIDE_CHANNEL_PORT"

# Frontend: binds the engine-core handshake and waits for our engine to join.
vllm-rs serve "$MODEL" \
    --data-parallel-size 1 --data-parallel-size-local 0 \
    --handshake-port "$HANDSHAKE_PORT" \
    --host 0.0.0.0 --port "$HTTP_PORT" &
FRONTEND_PID=$!

# Engine: connects as the headless DP engine, plays the NixlConnector role.
mock-engine-nixl \
    --handshake-address "tcp://127.0.0.1:${HANDSHAKE_PORT}" \
    --pd-role "$ROLE" \
    --engine-id "$ENGINE_ID" \
    --side-channel-host "$SIDE_CHANNEL_HOST" \
    --side-channel-port "$SIDE_CHANNEL_PORT" \
    --log-requests &
ENGINE_PID=$!

terminate() { kill "$ENGINE_PID" "$FRONTEND_PID" 2>/dev/null || true; }
trap terminate TERM INT

# If either process exits, tear the pod down so k8s restarts it.
wait -n "$FRONTEND_PID" "$ENGINE_PID"
echo "[entrypoint] a process exited; shutting down"
terminate
wait || true
