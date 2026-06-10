#!/usr/bin/env bash
#
# Drive a tap-recorded latency capture against the trace-capture-h200 pod and pull
# both recordings: the tap trace (server-side, full per-token ITL arrays) and the
# guidellm reports (client-side, mean-only ITL) from the SAME run, so the two can
# be compared request-for-request.
#
# Prereqs: deployment scaled to 1 and READY (kubectl -n weaton-dev scale deploy
# trace-capture-h200 --replicas=1), kubecontext pointed at the right cluster.
#
# guidellm 0.6.0 gotchas (learned the hard way): concurrent rate > 1 with synthetic
# data stalls unless --data-samples is set, and sweeping multiple rates in one
# invocation stalls too, so each concurrency level is its own invocation.
set -euo pipefail

NS=weaton-dev
DEPLOY=trace-capture-h200
MODEL=Qwen/Qwen3-8B
OUT_DIR="${OUT_DIR:-/tmp/trace-capture-h200}"
DATA="prompt_tokens=512,output_tokens=128"

mkdir -p "$OUT_DIR"

echo "==> port-forwarding $DEPLOY :8000"
kubectl -n "$NS" port-forward "deploy/$DEPLOY" 8000:8000 &
PF_PID=$!
trap 'kill $PF_PID 2>/dev/null || true' EXIT
sleep 3

curl -sf http://127.0.0.1:8000/v1/models >/dev/null || {
    echo "frontend not responding on :8000" >&2
    exit 1
}

run_bench() {
    local conc=$1 secs=$2
    echo "==> guidellm concurrent rate=$conc (${secs}s)"
    mkdir -p "$OUT_DIR/c$conc"
    uvx --from guidellm==0.6.0 guidellm benchmark \
        --target http://127.0.0.1:8000 \
        --model "$MODEL" \
        --profile concurrent \
        --rate "$conc" \
        --max-seconds "$secs" \
        --data "$DATA" \
        --data-samples 256 \
        --output-path "$OUT_DIR/c$conc/benchmarks.json"
}

run_bench 1 60
run_bench 16 120

echo "==> fetching tap trace"
kubectl -n "$NS" exec "deploy/$DEPLOY" -c tap -- cat /trace/trace.jsonl \
    > "$OUT_DIR/tap-trace.jsonl"

echo "==> done"
echo "  tap trace:        $OUT_DIR/tap-trace.jsonl ($(wc -l < "$OUT_DIR/tap-trace.jsonl" | tr -d ' ') lines)"
echo "  guidellm reports: $OUT_DIR/c1/benchmarks.json $OUT_DIR/c16/benchmarks.json"
echo
echo "Remember to scale back down:"
echo "  kubectl -n $NS scale deploy $DEPLOY --replicas=0"
