#!/usr/bin/env bash
#
# End-to-end LoRA metric check: real vLLM Rust frontend <--> our mock engine, exercising the
# LoRA layer all the way to the vllm:lora_requests_info Prometheus gauge.
#
# Needs an upstream vllm-rs (vllm-project/vllm#45030 or later), built once:
#
#   git clone https://github.com/vllm-project/vllm
#   (cd vllm/rust && cargo build --bin vllm-rs)
#   FRONTEND_BIN="$PWD/vllm/rust/target/debug/vllm-rs" ./scripts/e2e_lora.sh
#
# Flow:
#   1. load a (fake) LoRA adapter via POST /v1/load_lora_adapter  -> engine add_lora
#   2. send a request targeting that adapter; a slow inter-token latency keeps it decoding
#      across the scrape window
#   3. scrape /metrics mid-flight and assert running_lora_adapters names our adapter
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
ADAPTER="${ADAPTER:-test-adapter}"
HANDSHAKE_PORT="${HANDSHAKE_PORT:-29550}"
HTTP_HOST="${HTTP_HOST:-127.0.0.1}"
HTTP_PORT="${HTTP_PORT:-8000}"
# Slow decode so the request stays running across the scrape window. ITL_MS per token, over
# DECODE_TOKENS tokens, so it runs ~ITL_MS*DECODE_TOKENS ms; keep that well over the scrape loop.
ITL_MS="${ITL_MS:-2000}"
DECODE_TOKENS="${DECODE_TOKENS:-12}"
# A 0.29+ frontend rejects load_lora_adapter unless the engine advertises supports_lora.
MAX_LORAS="${MAX_LORAS:-1}"
FRONTEND_BIN="${FRONTEND_BIN:-$HOME/git/vllm/rust/target/debug/vllm-rs}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_BIN="$REPO_ROOT/target/debug/vllm-vcr"
BASE_URL="http://${HTTP_HOST}:${HTTP_PORT}"
LOG_DIR="$(mktemp -d)"

frontend_pid=""
engine_pid=""
req_pid=""

cleanup() {
    [[ -n "$req_pid" ]] && kill "$req_pid" 2>/dev/null || true
    [[ -n "$engine_pid" ]] && kill "$engine_pid" 2>/dev/null || true
    [[ -n "$frontend_pid" ]] && kill "$frontend_pid" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

fail() {
    echo "FAIL: $*" >&2
    echo "--- frontend log ($LOG_DIR/frontend.log) ---" >&2
    tail -40 "$LOG_DIR/frontend.log" >&2 || true
    echo "--- engine log ($LOG_DIR/engine.log) ---" >&2
    tail -40 "$LOG_DIR/engine.log" >&2 || true
    exit 1
}

[[ -x "$FRONTEND_BIN" ]] || fail "frontend binary not found at $FRONTEND_BIN
  build it from upstream vLLM (see the top of this script), then re-run with
  FRONTEND_BIN=/path/to/vllm/rust/target/debug/vllm-rs"
[[ -x "$ENGINE_BIN" ]] || { echo "building engine..."; (cd "$REPO_ROOT" && cargo build); }

echo "logs: $LOG_DIR"

# 1. Frontend: binds the handshake and waits for our external engine. The
#    /v1/load_lora_adapter route is only mounted when runtime LoRA updating is enabled (it's
#    an off-by-default admin endpoint), so opt in for this process.
echo "starting frontend ($MODEL) ..."
VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 "$FRONTEND_BIN" serve "$MODEL" \
    --data-parallel-size 1 \
    --data-parallel-size-local 0 \
    --handshake-port "$HANDSHAKE_PORT" \
    --host "$HTTP_HOST" \
    --port "$HTTP_PORT" \
    >"$LOG_DIR/frontend.log" 2>&1 &
frontend_pid=$!

# 2. Engine: slow inter-token latency so a request stays in decode (emitting scheduler_stats
#    every step) across the scrape window.
echo "starting mock engine (itl=${ITL_MS}ms) ..."
"$ENGINE_BIN" play \
    --handshake-address "tcp://127.0.0.1:${HANDSHAKE_PORT}" \
    --inter-token-latency "$ITL_MS" \
    --max-loras "$MAX_LORAS" \
    --log-requests \
    >"$LOG_DIR/engine.log" 2>&1 &
engine_pid=$!

# 3. Wait for health (first run downloads the tokenizer from HF).
echo "waiting for $BASE_URL/health ..."
for i in $(seq 1 120); do
    kill -0 "$frontend_pid" 2>/dev/null || fail "frontend exited during startup"
    kill -0 "$engine_pid" 2>/dev/null || fail "engine exited during startup"
    curl -fsS "$BASE_URL/health" >/dev/null 2>&1 && { echo "server up after ${i}s"; break; }
    sleep 1
    [[ "$i" == "120" ]] && fail "server did not become healthy within 120s"
done

# 4. Load a fake adapter. A non-local, non-existent path skips the local-path prefix check,
#    so the frontend just relays add_lora to the engine and exposes the adapter as a model id.
echo "--- load_lora_adapter ($ADAPTER) ---"
LOAD=$(curl -fsS "$BASE_URL/v1/load_lora_adapter" \
    -H 'Content-Type: application/json' \
    -d "{\"lora_name\":\"$ADAPTER\",\"lora_path\":\"$ADAPTER\"}") \
    || fail "load_lora_adapter request failed (engine add_lora rejected?)"
echo "$LOAD"

# 5. Fire a request against the adapter; the slow ITL keeps it decoding for the scrape window.
echo "--- request against adapter (backgrounded, ${DECODE_TOKENS} tokens @ ${ITL_MS}ms) ---"
curl -fsS "$BASE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$ADAPTER\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":$DECODE_TOKENS}" \
    >"$LOG_DIR/req.log" 2>&1 &
req_pid=$!

# 6. Scrape /metrics mid-flight and assert the gauge names our adapter as RUNNING.

# Value of one named label on the first vllm:lora_requests_info sample.
label_value() {
    sed -n "s/^vllm:lora_requests_info{[^}]*$1=\"\([^\"]*\)\"[^}]*}.*/\1/p" | head -1
}

# Exact membership in a comma-separated adapter list.
list_contains() {
    case ",$1," in
        *",$2,"*) return 0 ;;
        *) return 1 ;;
    esac
}

echo "--- scraping /metrics for vllm:lora_requests_info ---"
found=""
saw_waiting=""
for _ in $(seq 1 30); do
    sleep 1
    METRICS=$(curl -fsS "$BASE_URL/metrics" 2>/dev/null) || continue
    GAUGE=$(echo "$METRICS" | grep '^vllm:lora_requests_info{' || true)
    [[ -n "$GAUGE" ]] || continue
    RUNNING=$(echo "$GAUGE" | label_value running_lora_adapters)
    WAITING=$(echo "$GAUGE" | label_value waiting_lora_adapters)
    if list_contains "$RUNNING" "$ADAPTER"; then found="$GAUGE"; break; fi
    if list_contains "$WAITING" "$ADAPTER"; then saw_waiting="$GAUGE"; fi
done

if [[ -z "$found" ]]; then
    echo "--- vllm:lora_requests_info lines seen ---" >&2
    curl -fsS "$BASE_URL/metrics" 2>/dev/null | grep 'lora' >&2 || echo "(no lora metric lines at all)" >&2
    if [[ -n "$saw_waiting" ]]; then
        fail "adapter '$ADAPTER' appeared only in waiting_lora_adapters, never in running_lora_adapters
  (the engine sent no SCHEDULED event for the request)"
    fi
    fail "vllm:lora_requests_info never named adapter '$ADAPTER' in running_lora_adapters (gauge absent, or the engine never reported the adapter at all)"
fi

echo ""
echo "$found"
echo "PASS: vllm:lora_requests_info reported running adapter '$ADAPTER' end-to-end."
