#!/usr/bin/env bash
# Per-line kind e2e: run a model-server image (frontend + engine, one pod) on a kind
# cluster and drive the real OpenAI HTTP surface through the shipped vllm-rs frontend,
# cross-checking every response against the engine's own counters on the control API.
#
#   IMAGE=localhost/vllm-vcr-slim:vllm0.33 ci/kind-e2e.sh
#
# Env:
#   IMAGE         image to test (required); loaded into kind, never pulled
#   CLUSTER       kind cluster name (default vllm-vcr-e2e); created if missing
#   KEEP_CLUSTER  1 = leave the cluster up afterwards (default: delete it if we made it)
#   ARTIFACTS     directory for pod logs and the failure dump (default: mktemp)
set -euo pipefail

IMAGE="${IMAGE:?IMAGE is required}"
CLUSTER="${CLUSTER:-vllm-vcr-e2e}"
KEEP_CLUSTER="${KEEP_CLUSTER:-0}"
ARTIFACTS="${ARTIFACTS:-$(mktemp -d)}"
MODEL="Qwen/Qwen3-0.6B"
POD="vllm-vcr-e2e"
HTTP_PORT=18000
CONTROL_PORT=18001
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CTX="kind-${CLUSTER}"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
fail() {
    log "FAIL: $*"
    dump
    exit 1
}

mkdir -p "${ARTIFACTS}"
created_cluster=0
pf_pid=""

dump() {
    kubectl --context "${CTX}" logs "${POD}" > "${ARTIFACTS}/model-server.log" 2>&1 || true
    kubectl --context "${CTX}" describe pod "${POD}" > "${ARTIFACTS}/pod.txt" 2>&1 || true
    log "pod logs: ${ARTIFACTS}/model-server.log"
    tail -n 60 "${ARTIFACTS}/model-server.log" || true
}

cleanup() {
    [[ -n "${pf_pid}" ]] && kill "${pf_pid}" 2>/dev/null || true
    if [[ "${created_cluster}" == 1 && "${KEEP_CLUSTER}" != 1 ]]; then
        kind delete cluster --name "${CLUSTER}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

if ! kind get clusters 2>/dev/null | grep -qx "${CLUSTER}"; then
    log "creating kind cluster ${CLUSTER}"
    kind create cluster --name "${CLUSTER}" --wait 120s
    created_cluster=1
fi

log "loading ${IMAGE} into ${CLUSTER}"
kind load docker-image "${IMAGE}" --name "${CLUSTER}"

kubectl --context "${CTX}" delete pod "${POD}" --ignore-not-found --wait=true >/dev/null
sed "s|@IMAGE@|${IMAGE}|" "${ROOT}/deploy/kind-e2e/model-server.yaml" \
    | kubectl --context "${CTX}" apply -f -

log "waiting for the frontend to report healthy (engine handshake + tokenizer fetch)"
deadline=$((SECONDS + 300))
while :; do
    phase="$(kubectl --context "${CTX}" get pod "${POD}" -o jsonpath='{.status.phase}')"
    ready="$(kubectl --context "${CTX}" get pod "${POD}" \
        -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}')"
    [[ "${ready}" == True ]] && break
    case "${phase}" in
        Failed | Succeeded) fail "pod exited before becoming ready (phase ${phase})" ;;
    esac
    ((SECONDS < deadline)) || fail "pod never became ready (phase ${phase})"
    sleep 2
done

kubectl --context "${CTX}" port-forward "pod/${POD}" "${HTTP_PORT}:8000" "${CONTROL_PORT}:8001" \
    > "${ARTIFACTS}/port-forward.log" 2>&1 &
pf_pid=$!
for _ in $(seq 1 50); do
    curl -sf "http://127.0.0.1:${HTTP_PORT}/health" >/dev/null 2>&1 && break
    sleep 0.2
done

API="http://127.0.0.1:${HTTP_PORT}"
CONTROL="http://127.0.0.1:${CONTROL_PORT}"

# POST JSON, fail on a non-2xx status, print the body.
post() {
    local url="$1" body="$2" out status
    out="$(mktemp)"
    status="$(curl -s -o "${out}" -w '%{http_code}' -H 'Content-Type: application/json' -d "${body}" "${url}")"
    if [[ "${status}" != 2* ]]; then
        log "POST ${url} -> ${status}: $(cat "${out}")" >&2
        rm -f "${out}"
        return 1
    fi
    cat "${out}"
    rm -f "${out}"
}

stats() { curl -sf "${CONTROL}/stats"; }
expect() {
    local what="$1" got="$2" want="$3"
    [[ "${got}" == "${want}" ]] || fail "${what}: got '${got}', want '${want}'"
    log "ok: ${what} = ${got}"
}

curl -sf -X POST "${CONTROL}/stats/reset" >/dev/null || fail "control API /stats/reset"

log "check: /v1/models serves ${MODEL}"
models="$(curl -sf "${API}/v1/models")" || fail "/v1/models"
expect "served model" "$(jq -r --arg m "${MODEL}" '[.data[].id] | index($m) != null' <<<"${models}")" true

log "check: non-streaming completion runs to max_tokens"
resp="$(post "${API}/v1/completions" \
    "{\"model\":\"${MODEL}\",\"prompt\":\"The quick brown fox\",\"max_tokens\":16,\"ignore_eos\":true}")" \
    || fail "completion request"
expect "completion finish_reason" "$(jq -r '.choices[0].finish_reason' <<<"${resp}")" length
expect "completion completion_tokens" "$(jq -r '.usage.completion_tokens' <<<"${resp}")" 16

log "check: streaming chat completion frames and terminates"
sse="$(curl -sfN -H 'Content-Type: application/json' "${API}/v1/chat/completions" -d \
    "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8,\"ignore_eos\":true,\"stream\":true,\"stream_options\":{\"include_usage\":true}}")" \
    || fail "streaming chat request"
expect "stream terminator" "$(grep -c '^data: \[DONE\]$' <<<"${sse}")" 1
frames="$(grep '^data: {' <<<"${sse}" | sed 's/^data: //')"
expect "stream finish_reason" \
    "$(jq -rs '[.[].choices[]?.finish_reason | select(. != null)] | last' <<<"${frames}")" length
expect "stream usage completion_tokens" \
    "$(jq -rs '[.[] | select(.usage != null)] | last | .usage.completion_tokens' <<<"${frames}")" 8
[[ "$(jq -rs '[.[].choices[]?.delta.content // empty] | join("") | length' <<<"${frames}")" -gt 0 ]] \
    || fail "stream carried no content deltas"
log "ok: stream carried content deltas"

log "check: a repeated prompt hits the engine's prefix cache"
prompt="$(printf 'vllm-vcr kind e2e prefix cache probe %.0s' $(seq 1 40))"
body="{\"model\":\"${MODEL}\",\"prompt\":\"${prompt}\",\"max_tokens\":4,\"ignore_eos\":true}"
post "${API}/v1/completions" "${body}" >/dev/null || fail "prefix cache warm-up"
resp="$(post "${API}/v1/completions" "${body}")" || fail "prefix cache probe"
cached="$(jq -r '.usage.prompt_tokens_details.cached_tokens // 0' <<<"${resp}")"
[[ "${cached}" -gt 0 ]] || fail "second identical prompt reported ${cached} cached tokens"
log "ok: cached_tokens = ${cached}"
post "${API}/reset_prefix_cache" '{}' >/dev/null || fail "reset_prefix_cache utility call"
log "ok: reset_prefix_cache"

log "check: runtime LoRA load, serve, unload"
post "${API}/v1/load_lora_adapter" '{"lora_name":"e2e-adapter","lora_path":"/adapters/e2e"}' >/dev/null \
    || fail "load_lora_adapter"
expect "adapter listed" \
    "$(curl -sf "${API}/v1/models" | jq -r '[.data[].id] | index("e2e-adapter") != null')" true
resp="$(post "${API}/v1/completions" \
    '{"model":"e2e-adapter","prompt":"adapter","max_tokens":4,"ignore_eos":true}')" \
    || fail "completion on the adapter"
expect "adapter completion_tokens" "$(jq -r '.usage.completion_tokens' <<<"${resp}")" 4
post "${API}/v1/unload_lora_adapter" '{"lora_name":"e2e-adapter"}' >/dev/null \
    || fail "unload_lora_adapter"
expect "adapter unlisted" \
    "$(curl -sf "${API}/v1/models" | jq -r '[.data[].id] | index("e2e-adapter") != null')" false

log "check: a client disconnect aborts the request in the engine"
before="$(stats | jq -r .requests_aborted)"
curl -sf -X PATCH -H 'Content-Type: application/json' "${CONTROL}/config" \
    -d '{"inter_token_latency":100}' >/dev/null || fail "control PATCH /config"
curl -sN --max-time 1 -H 'Content-Type: application/json' "${API}/v1/completions" \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"abort me\",\"max_tokens\":500,\"ignore_eos\":true,\"stream\":true}" \
    >/dev/null 2>&1 || true
aborted="${before}"
for _ in $(seq 1 50); do
    aborted="$(stats | jq -r .requests_aborted)"
    [[ "${aborted}" -gt "${before}" ]] && break
    sleep 0.2
done
[[ "${aborted}" -gt "${before}" ]] || fail "engine never saw the abort (requests_aborted stayed ${aborted})"
log "ok: requests_aborted ${before} -> ${aborted}"
curl -sf -X PATCH -H 'Content-Type: application/json' "${CONTROL}/config" \
    -d '{"inter_token_latency":0}' >/dev/null || fail "control PATCH /config reset"

log "check: engine counters agree with what the frontend served"
s="$(stats)"
expect "engine requests_failed" "$(jq -r .requests_failed <<<"${s}")" 0
expect "engine running" "$(jq -r .running <<<"${s}")" 0
expect "engine waiting" "$(jq -r .waiting <<<"${s}")" 0
# completion, chat stream, prefix warm-up, prefix probe, adapter completion, abort.
expect "engine requests_received" "$(jq -r .requests_received <<<"${s}")" 6
expect "engine requests_completed" "$(jq -r .requests_completed <<<"${s}")" 5

log "check: frontend metrics carry the served requests"
metrics="$(curl -sf "${API}/metrics")" || fail "/metrics"
grep -q '^vllm:request_success_total' <<<"${metrics}" || fail "/metrics has no vllm:request_success_total"
log "ok: /metrics"

kubectl --context "${CTX}" logs "${POD}" > "${ARTIFACTS}/model-server.log" 2>&1
if grep -qiE 'panicked|error decoding|failed to decode' "${ARTIFACTS}/model-server.log"; then
    fail "model-server log carries a decode error or panic"
fi
log "ok: model-server log is clean"
log "PASS ${IMAGE}"
