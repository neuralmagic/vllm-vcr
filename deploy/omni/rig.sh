#!/usr/bin/env bash
# Drive the vLLM-Omni rig (deploy/omni/pod.yaml). See `just --list | grep omni`.
set -euo pipefail

NAMESPACE=${NAMESPACE:-inference-sim}
POD=omni-rig
VLLM_LINE=${VLLM_LINE:-0.32}
VLLM_TAG=${VLLM_TAG:-v${VLLM_LINE}.0}
MASTER_PORT=21212
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

k() { kubectl -n "$NAMESPACE" "$@"; }
in_head() { k exec -i "$POD" -c head -- "$@"; }
in_sim() { k exec -i "$POD" -c sim -- "$@"; }

cmd=${1:?usage: rig.sh up|down|build|head|stage|real-stage|record|stop-records|fetch|stop-stages|status|logs}
shift

case "$cmd" in
  up)
    k apply -f "$HERE/pod.yaml"
    k wait --for=condition=Ready "pod/$POD" --timeout=600s
    ;;

  down)
    k delete pod "$POD" --ignore-not-found
    ;;

  # Copy the working tree in and build play for the head's vLLM line.
  build)
    (cd "$REPO" && git ls-files -co --exclude-standard \
      | COPYFILE_DISABLE=1 tar --no-xattrs -czf - -T -) \
      | in_sim sh -c 'mkdir -p /work/src && cd /work/src \
          && find . -mindepth 1 -maxdepth 1 ! -name target -exec rm -rf {} + \
          && tar -xzf -'
    in_sim sh -s -- "$VLLM_LINE" "$VLLM_TAG" <<'SH'
set -e
cd /work/src
command -v cmake >/dev/null || { apt-get update -qq && apt-get install -y -qq cmake clang >/dev/null; }
cargo xtask pin-vllm "$1"
VLLM_TARGET_VERSION="$2" cargo build --quiet --release -p vllm-vcr
echo "built play for vLLM $2"
SH
    ;;

  # (Re)start the head for a model with every stage remote. Extra args go to `vllm serve`.
  head)
    in_head sh -s -- "$MASTER_PORT" "$@" <<'SH'
port=$1; shift
pkill -f "vllm serve" || true
sleep 4
pkill -9 -f "vllm serve" || true
nohup vllm serve "$@" --omni --stage-id none \
  --omni-master-address 127.0.0.1 --omni-master-port "$port" \
  --host 0.0.0.0 --port 8000 > /tmp/head.log 2>&1 &
for _ in $(seq 1 400); do
  grep -q "OmniMasterServer. Listening" /tmp/head.log && { echo "head: master listening"; exit 0; }
  pgrep -f "vllm serve" >/dev/null || { tail -n 30 /tmp/head.log; echo "head: exited"; exit 1; }
  sleep 3
done
echo "head: timed out waiting for the master server"; exit 1
SH
    ;;

  # Attach play as stage <id>. Extra args go to `vllm-vcr play`.
  stage)
    in_sim sh -s -- "$MASTER_PORT" "$@" <<'SH'
port=$1; stage=$2; shift 2
cd /work
RUST_LOG=${RUST_LOG:-info} nohup /work/src/target/release/vllm-vcr play \
  --omni-master-address "tcp://127.0.0.1:$port" --omni-stage-id "$stage" \
  --log-requests "$@" > "/work/stage$stage.log" 2>&1 &
SH
    ;;

  # Run a real GPU stage <id> of <model> in the head container, registering with
  # the master on <port> (the head's 21212, or a tap's --omni-master-listen port).
  # Extra args go to `vllm serve`.
  real-stage)
    in_head sh -s -- "$@" <<'SH'
stage=$1; model=$2; port=$3; shift 3
nohup vllm serve "$model" --omni --headless --stage-id "$stage" \
  --omni-master-address 127.0.0.1 --omni-master-port "$port" "$@" \
  > "/tmp/real-stage$stage.log" 2>&1 &
SH
    ;;

  # Record stage traffic: the tap registers with the head and listens for the
  # real stage on port 2121<3+2*stage> (21213 for stage 0, 21215 for stage 1).
  # Extra args go to `vllm-vcr record`; a diffusion stage needs
  # `--omni-stage-kind diffusion`. The trace is /work/stage<id>.jsonl.
  record)
    in_sim sh -s -- "$MASTER_PORT" "$@" <<'SH'
port=$1; stage=$2; shift 2
base=$((21213 + 2 * stage))
cd /work
RUST_LOG=${RUST_LOG:-info} nohup /work/src/target/release/vllm-vcr record \
  --omni-master-address "tcp://127.0.0.1:$port" \
  --omni-master-listen "tcp://127.0.0.1:$base" \
  --omni-coordinator-listen "tcp://127.0.0.1:$((base + 1))" \
  --engine-handshake "tcp://127.0.0.1:$((5580 + stage))" \
  --input-address "tcp://127.0.0.1:$((29560 + 2 * stage))" \
  --output-address "tcp://127.0.0.1:$((29561 + 2 * stage))" \
  --trace-out "/work/stage$stage.jsonl" "$@" > "/work/record$stage.log" 2>&1 &
echo "tap for stage $stage: real stage registers on port $base"
SH
    ;;

  # Stop the taps (finalizing their traces) and print a trace to stdout.
  stop-records)
    in_sim sh -c 'pkill -TERM -f "[v]llm-vcr record" || true'
    ;;

  fetch)
    in_sim cat "${1:?fetch <path in the sim container>}"
    ;;

  stop-stages)
    in_sim sh -c 'pkill -f "[v]llm-vcr play" || true'
    in_head sh -c 'pkill -f "stage-id [0-9]" || true'
    ;;

  status)
    in_head sh -c 'curl -s -m 5 -o /dev/null -w "head /health: %{http_code}\n" localhost:8000/health || true
grep -E "Remote .* replica attached|detached replica|OmniTiming|ERROR" /tmp/head.log | tail -n 12 | cut -c1-300'
    ;;

  logs)
    case "${1:?logs head|<stage-id>}" in
      head) in_head tail -n "${2:-40}" /tmp/head.log ;;
      real*) in_head tail -n "${2:-40}" "/tmp/real-stage${1#real}.log" ;;
      record*) in_sim tail -n "${2:-40}" "/work/record${1#record}.log" ;;
      *) in_sim tail -n "${2:-40}" "/work/stage$1.log" ;;
    esac
    ;;

  *)
    echo "unknown command: $cmd" >&2
    exit 2
    ;;
esac
