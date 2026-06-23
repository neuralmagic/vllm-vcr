# Trace capture

Record real vLLM engine-core traces with `vllm-vcr record` sitting between the frontend
and a headless GPU engine.

**Layout reference:** [docs/deploy.md](../../docs/deploy.md#deploytrace-capture--record-real-engine-traces)

## Structure

```text
deploy/trace-capture/
├── base/                    # Cluster-agnostic Deployments, PVCs, conformance queue
│   ├── kustomization.yaml
│   ├── h200-capture.yaml    # Default: vllm-rs frontend + tap + engine
│   ├── h200-capture-agentic.yaml
│   ├── offline-replay.yaml  # vllm-vcr play, no GPU
│   ├── gemma*/diffusiongemma* captures
│   └── pvcs/                # Shared model-weight caches
├── overlays/
│   └── inference-sim/       # Sets namespace; includes ../../base
├── models.toml              # Conformance Job matrix → gen-capture-jobs.py
└── *.yaml (root)            # Copies of base manifests for direct kubectl apply / just
```

## Quick start

```bash
# Deploy all base captures into inference-sim namespace
kustomize build deploy/trace-capture/overlays/inference-sim | kubectl apply -f -

# Or use justfile
just capture-up
just capture-status
```

Drive a capture and pull the trace:

```bash
bash deploy/trace-capture/run-capture.sh
```

## Conformance captures

One Job at a time on the GPU (see `conformance-queue.yaml`):

```bash
just conformance-list
just conformance-run qwen3-8b
```

Upload and register goldens: [docs/conformance.md](../../docs/conformance.md).

## Ops

- If **any** container in a tap pod restarts, **delete the pod** and redeploy (half-wired ZMQ records garbage).
- Traces are on **emptyDir** — fetch before scaling down or the pod is reaped (~35 min idle on some clusters).
