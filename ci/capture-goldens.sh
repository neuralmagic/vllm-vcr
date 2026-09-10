#!/usr/bin/env bash
# Submit conformance capture jobs for one compat.toml line, upload their traces,
# and replace that line's generated block in conformance/manifest.toml.
#
#   LINE     compat.toml line to capture (default: the default = true line)
#   TARGETS  models.toml capture targets (default: the [goldens].targets set)
set -Eeuo pipefail

: "${S3_BUCKET:?S3_BUCKET is required, e.g. llm-d-artifacts-783952637884}"

NAMESPACE="${NAMESPACE:-inference-sim}"
LINE="${LINE:-}"
MANIFEST="${MANIFEST:-conformance/manifest.toml}"
OUT_DIR="${OUT_DIR:-goldens}"
POLL_SECONDS="${POLL_SECONDS:-20}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-10800}"
QUEUE_NAME="conformance-queue"
GEN=(python3 deploy/trace-capture/gen-capture-jobs.py)
if [[ -n "${LINE}" ]]; then
    GEN+=(--line "${LINE}")
fi
TARGETS="${TARGETS:-$("${GEN[@]}" --list-goldens | tr '\n' ' ')}"

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
    printf '[%s] %s\n' "$(timestamp)" "$*" >&2
}

group_start() {
    if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
        printf '::group::%s\n' "$*" >&2
    else
        log "BEGIN: $*"
    fi
}

group_end() {
    if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
        printf '::endgroup::\n' >&2
    fi
}

file_bytes() {
    wc -c < "$1" | tr -d '[:space:]'
}

on_error() {
    local status=$?
    local line="${1:-unknown}"
    log "ERROR: golden capture failed at line ${line} with exit code ${status}"
    [[ "${GITHUB_ACTIONS:-}" == "true" ]] && printf '::endgroup::\n' >&2
    exit "${status}"
}

trap 'on_error ${LINENO}' ERR

mkdir -p "${OUT_DIR}"

if [[ ! "${NAMESPACE}" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]]; then
    log "ERROR: invalid Kubernetes namespace: ${NAMESPACE}"
    exit 1
fi

read -r line_id vllm_tag <<<"$("${GEN[@]}" --show-line)"

log "Starting golden capture for line ${line_id} (${vllm_tag})"
log "Configuration: namespace=${NAMESPACE} targets=${TARGETS} manifest=${MANIFEST} out_dir=${OUT_DIR} poll=${POLL_SECONDS}s timeout=${TIMEOUT_SECONDS}s"

group_start "Prepare namespace and queue"
log "Ensuring namespace ${NAMESPACE} exists"
kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

if ! kubectl get clusterqueue "${QUEUE_NAME}" >/dev/null 2>&1; then
    log "ERROR: missing ClusterQueue ${QUEUE_NAME}; apply deploy/trace-capture/base/conformance-queue.yaml first"
    exit 1
fi

log "Ensuring LocalQueue ${QUEUE_NAME} exists in namespace ${NAMESPACE}"
cat <<YAML | kubectl apply -f -
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata:
  name: ${QUEUE_NAME}
  namespace: ${NAMESPACE}
spec:
  clusterQueue: ${QUEUE_NAME}
YAML
group_end

group_start "Prepare validation scripts"
log "Ensuring validation-scripts ConfigMap exists in namespace ${NAMESPACE}"
kubectl create configmap validation-scripts -n "${NAMESPACE}" \
    --from-file=loadgen.py=deploy/trace-capture/loadgen.py \
    --from-file=runner.sh=deploy/trace-capture/validation-runner.sh \
    --dry-run=client -o yaml | kubectl apply -f -
group_end

# One generated Job per target: the Job's name and workload annotation are the
# only naming rule, and the generator owns it.
group_start "Submit Kueue capture jobs"
for target in ${TARGETS}; do
    job_file="${OUT_DIR}/${target}.job.json"
    "${GEN[@]}" "${target}" > "${job_file}"
    log "${target}: submitting $(jq -r .metadata.name "${job_file}") (engine $(jq -r '.metadata.annotations["llm-d.ai/engine-image"]' "${job_file}"))"
    kubectl apply -f "${job_file}"
done
group_end

wait_for_loadgen_done() {
    local job="$1"
    local started="${SECONDS}"
    local deadline=$((SECONDS + TIMEOUT_SECONDS))
    local next_report="${SECONDS}"
    local pod=""

    log "${job}: waiting for pod creation and loadgen completion"
    while (( SECONDS < deadline )); do
        pod="$(kubectl get pod -n "${NAMESPACE}" -l "job-name=${job}" \
            -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
        if [[ -n "${pod}" ]] && kubectl exec -n "${NAMESPACE}" "${pod}" -c loadgen -- \
            test -f /trace/loadgen-done >/dev/null 2>&1; then
            log "${job}: loadgen complete in pod ${pod} after $((SECONDS - started))s"
            return 0
        fi
        if (( SECONDS >= next_report )); then
            log "${job}: still waiting after $((SECONDS - started))s; pod=${pod:-not-created-yet}"
            group_start "${job} status snapshot"
            kubectl get job,pod -n "${NAMESPACE}" -l "job-name=${job}" || true
            [[ -n "${pod}" ]] && kubectl logs -n "${NAMESPACE}" "${pod}" -c loadgen --tail=8 || true
            group_end
            next_report=$((SECONDS + 120))
        fi
        sleep "${POLL_SECONDS}"
    done

    log "ERROR: timed out waiting for ${job} after ${TIMEOUT_SECONDS}s"
    group_start "${job} timeout diagnostics"
    kubectl get job,pod -n "${NAMESPACE}" -l "job-name=${job}" -o wide || true
    kubectl describe job -n "${NAMESPACE}" "${job}" || true
    [[ -n "${pod}" ]] && kubectl describe pod -n "${NAMESPACE}" "${pod}" || true
    [[ -n "${pod}" ]] && kubectl logs -n "${NAMESPACE}" "${pod}" -c loadgen --tail=80 || true
    group_end
    return 1
}

entries="${OUT_DIR}/goldens-${line_id}.toml"
: > "${entries}"

for target in ${TARGETS}; do
    job_file="${OUT_DIR}/${target}.job.json"
    job="$(jq -r .metadata.name "${job_file}")"
    workload="$(jq -r '.metadata.annotations["llm-d.ai/workload"]' "${job_file}")"
    trace="${OUT_DIR}/${target}.jsonl"
    stats="${OUT_DIR}/${target}.step-stats.jsonl"
    gz="${trace}.gz"

    group_start "Capture ${target}"
    log "${target}: using Kubernetes Job ${job}"
    wait_for_loadgen_done "${job}"

    log "${target}: fetching trace from loadgen container"
    kubectl exec -n "${NAMESPACE}" "job/${job}" -c loadgen -- cat /trace/trace.jsonl > "${trace}"
    if kubectl exec -n "${NAMESPACE}" "job/${job}" -c loadgen -- cat /trace/step-stats.jsonl > "${stats}"; then
        log "${target}: fetched step stats (${stats}, $(file_bytes "${stats}") bytes)"
    else
        log "WARN: ${target}: step stats were not available"
        rm -f "${stats}"
    fi

    if [[ ! -s "${trace}" ]]; then
        log "ERROR: ${trace} is empty"
        exit 1
    fi
    log "${target}: fetched trace (${trace}, $(file_bytes "${trace}") bytes, $(wc -l < "${trace}" | tr -d '[:space:]') lines)"

    gzip -c "${trace}" > "${gz}"
    log "${target}: compressed trace (${gz}, $(file_bytes "${gz}") bytes)"
    meta="$(head -n 1 "${trace}")"
    model="$(jq -r '.meta.model // empty' <<<"${meta}")"
    gpu="$(jq -r '.meta.gpu // empty' <<<"${meta}")"

    if [[ -z "${model}" || -z "${gpu}" ]]; then
        log "ERROR: ${trace} meta must include model and gpu"
        head -n 1 "${trace}" >&2
        exit 1
    fi
    log "${target}: trace metadata model=${model} gpu=${gpu}"

    key="conformance/${vllm_tag}/${gpu}/${model}/${workload}.jsonl.gz"
    dest="s3://${S3_BUCKET}/${key}"
    log "${target}: uploading ${gz} -> ${dest}"
    aws s3 cp "${gz}" "${dest}"
    log "${target}: upload complete"

    log "${target}: appending generated manifest entry (line ${line_id})"
    cargo xtask golden-entry \
        --trace "${trace}" \
        --archive "${gz}" \
        --bucket-path "${key}" \
        --workload "${workload}" \
        --line "${line_id}" >> "${entries}"

    kubectl exec -n "${NAMESPACE}" "job/${job}" -c loadgen -- touch /trace/fetched
    if kubectl wait -n "${NAMESPACE}" --for=condition=complete "job/${job}" --timeout=10m; then
        log "${target}: job completed after trace fetch"
    else
        log "WARN: ${target}: job did not report complete within 10m after fetch"
    fi
    group_end
done

group_start "Update conformance manifest"
log "Replacing ${MANIFEST} line ${line_id} block from ${entries}"
cargo xtask set-goldens \
    --entries-file "${entries}" \
    --manifest "${MANIFEST}" \
    --line "${line_id}"
cat "${entries}"
group_end
