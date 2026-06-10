#!/usr/bin/env bash
#
# Drive the counterfactual-validation captures against the trace-capture-h200 pod.
#
#   run-validation.sh sweep      prompt-length sweep (c1 + c8 per length): fills the
#                                latency model's long-prefill TTFT buckets and the
#                                stall pool with big chunked-prefill interference
#   run-validation.sh multiturn  the agentic scenario at 0.4 sessions/s (stable for
#                                the engine even with prefix caching disabled)
#
# Run `sweep` then `multiturn` against the normal rig (just capture-up), fetch the
# tap trace, then `multiturn` again against the no-cache rig (just capture-up-nocache).
#
# Prereqs: deployment scaled to 1 and READY, kubecontext on the right cluster, uv.
set -euo pipefail

PHASE="${1:?usage: run-validation.sh sweep|multiturn}"
NS=weaton-dev
DEPLOY=trace-capture-h200
MODEL=Qwen/Qwen3-8B
OUT_DIR="${OUT_DIR:-/tmp/trace-validation-h200}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

loadgen() {
    uv run --with httpx "$SCRIPT_DIR/loadgen.py" \
        --url http://127.0.0.1:8000 --model "$MODEL" "$@"
}

case "$PHASE" in
sweep)
    # Word counts; the synthetic vocab tokenizes at ~1.54 tokens/word, so this
    # spans ~0.8k to ~12.3k wire tokens across the model's prompt buckets.
    for words in 512 1000 1500 3000 5500 8000; do
        echo "==> sweep prompt=$words words, c1 (45s)"
        loadgen --pattern constant --concurrency 1 --duration 45 \
            --prompt-tokens "$words" --output-tokens 128 \
            --out "$OUT_DIR/sweep-p$words-c1.json"
        echo "==> sweep prompt=$words words, c8 (75s)"
        loadgen --pattern constant --concurrency 8 --duration 75 \
            --prompt-tokens "$words" --output-tokens 128 \
            --out "$OUT_DIR/sweep-p$words-c8.json"
    done
    ;;
multiturn)
    echo "==> multiturn 0.4 sessions/s x 5 turns, ~10k-token shared prefix (240s)"
    loadgen --pattern multiturn --rate 0.4 --turns 5 \
        --prefix-tokens 6500 --prompt-tokens 128 --output-tokens 128 \
        --duration 240 --seed 7 \
        --out "$OUT_DIR/multiturn-loadgen.json"
    ;;
*)
    echo "unknown phase: $PHASE" >&2
    exit 1
    ;;
esac

echo "==> done; fetch the tap trace with: just capture-fetch <out>"
