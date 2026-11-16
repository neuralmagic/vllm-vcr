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

cmd=${1:?usage: rig.sh up|down|build|head|stage|stop-stages|status|logs}
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

  stop-stages)
    in_sim sh -c 'pkill vllm-vcr || true'
    ;;

  status)
    in_head sh -c 'curl -s -m 5 -o /dev/null -w "head /health: %{http_code}\n" localhost:8000/health || true
grep -E "Remote .* replica attached|detached replica|OmniTiming|ERROR" /tmp/head.log | tail -n 12 | cut -c1-300'
    ;;

  logs)
    case "${1:?logs head|<stage-id>}" in
      head) in_head tail -n "${2:-40}" /tmp/head.log ;;
      *) in_sim tail -n "${2:-40}" "/work/stage$1.log" ;;
    esac
    ;;

  *)
    echo "unknown command: $cmd" >&2
    exit 2
    ;;
esac
