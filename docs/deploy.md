# Deploy layouts

Everything under [`deploy/`](../deploy/) is Kubernetes configuration for three kinds of
workloads:

| Directory | What it runs | Needs GPU? |
| --- | --- | --- |
| [`llm-d-pd/`](../deploy/llm-d-pd/) | Mock **prefill/decode** servers behind the llm-d router | No |
| [`llm-d-prefix-cache/`](../deploy/llm-d-prefix-cache/) | Mock **prefix-cache routing** (KV events, no P/D split) | No |
| [`trace-capture/`](../deploy/trace-capture/) | **Record** real vLLM engine traces (tap in the middle) | Yes (capture pods) |

Each area has its own README with deploy/test commands:

- [llm-d-pd](../deploy/llm-d-pd/README.md)
- [llm-d-prefix-cache](../deploy/llm-d-prefix-cache/README.md)
- [trace-capture](../deploy/trace-capture/README.md) (see also [`justfile`](../justfile) targets)

---

## How the YAML types fit together

```text
helmfile.yaml          → installs the llm-d router (Helm) + model servers (kustomize)
kustomization.yaml     → lists YAML files to apply together; can set namespace / image tag
Deployment.yaml        → the pods (containers, env, probes)
*.values.yaml          → Helm values for the router chart (not the mock pods directly)
ServiceAccount.yaml    → identity the pods run as
PersistentVolumeClaim  → shared disk for cached model weights (trace-capture only)
```

**helmfile** orchestrates two releases: the llm-d **router** chart, then the **modelserver**
kustomize bundle. **Kustomize** overlays (`trace-capture/overlays/…`) add cluster-specific
settings (namespace) on top of a shared **base/**.

---

## `deploy/llm-d-pd/` — GPU-free prefill/decode

Tests [llm-d P/D disaggregation](https://github.com/llm-d/llm-d) with `vllm-vcr` mock
engines and real NIXL KV transfer on CPU.

```text
client → Router/EPP → decode sidecar :8000 → mock decode :8200
                                              ↓ NIXL :5600
                                         mock prefill x2 :8000
```

| File | Purpose |
| --- | --- |
| [`helmfile.yaml`](../deploy/llm-d-pd/helmfile.yaml) | `helmfile apply` installs router + mock pods into `llm-d-pd-mock` |
| [`router/base.values.yaml`](../deploy/llm-d-pd/router/base.values.yaml) | Base router settings: EPP scheduler image, Envoy proxy, CPU/RAM |
| [`router/pd-disaggregation.values.yaml`](../deploy/llm-d-pd/router/pd-disaggregation.values.yaml) | P/D plugins, scheduling profiles, `matchLabels` for mock pods |
| [`modelserver/kustomization.yaml`](../deploy/llm-d-pd/modelserver/kustomization.yaml) | Bundles modelserver manifests; **set `vllm-vcr` image tag here** |
| [`modelserver/serviceaccount.yaml`](../deploy/llm-d-pd/modelserver/serviceaccount.yaml) | Shared ServiceAccount (`mock-pd-sa`) |
| [`modelserver/prefill-deployment.yaml`](../deploy/llm-d-pd/modelserver/prefill-deployment.yaml) | **2 prefill pods** — `MOCK_PD_ROLE=prefill`, API :8000, NIXL :5600 |
| [`modelserver/decode-deployment.yaml`](../deploy/llm-d-pd/modelserver/decode-deployment.yaml) | **1 decode pod** — llm-d sidecar :8000 → mock engine :8200, pulls KV from prefill |

Deploy: `helmfile -f deploy/llm-d-pd/helmfile.yaml apply`

---

## `deploy/llm-d-prefix-cache/` — GPU-free prefix-cache routing

Tests llm-d **precise-prefix-cache-routing**: one pool of monolithic mock pods publishes
KV-cache events; the EPP routes by longest prefix match.

```text
client → Router/EPP ──(prefix-cache-scorer)──► mock pod
              ▲                                    │
              └──── KV events ZMQ :5556 ───────────┘
```

| File | Purpose |
| --- | --- |
| [`helmfile.yaml`](../deploy/llm-d-prefix-cache/helmfile.yaml) | Installs router + mock pods into `llm-d-pc-mock` |
| [`router/base.values.yaml`](../deploy/llm-d-prefix-cache/router/base.values.yaml) | Same router base as P/D |
| [`router/precise-prefix-cache-routing.values.yaml`](../deploy/llm-d-prefix-cache/router/precise-prefix-cache-routing.values.yaml) | EPP plugins (`precise-prefix-cache-producer`, scorers), `blockSize: 64` |
| [`modelserver/kustomization.yaml`](../deploy/llm-d-prefix-cache/modelserver/kustomization.yaml) | Bundles deployment + SA; image tag override |
| [`modelserver/serviceaccount.yaml`](../deploy/llm-d-prefix-cache/modelserver/serviceaccount.yaml) | Shared ServiceAccount (`mock-pc-sa`) |
| [`modelserver/deployment.yaml`](../deploy/llm-d-prefix-cache/modelserver/deployment.yaml) | **3 monolithic pods** — KV events on :5556, `MOCK_PD_ROLE=both` |

Deploy: `helmfile -f deploy/llm-d-prefix-cache/helmfile.yaml apply`

---

## `deploy/trace-capture/` — record real engine traces

Pods run a **real vLLM engine** with `vllm-vcr record` (the tap) between frontend and
engine. Output is JSONL under `/trace` (see [Conformance](./conformance.md) for upload).

Canonical manifests live in **`base/`**. Apply via kustomize:

```bash
kustomize build deploy/trace-capture/overlays/inference-sim | kubectl apply -f -
```

Some files are duplicated at the **trace-capture root** for direct `kubectl apply` or
`just` targets; headers match `base/`.

### Bundling and infra

| File | Purpose |
| --- | --- |
| [`base/kustomization.yaml`](../deploy/trace-capture/base/kustomization.yaml) | Lists all capture Deployments, PVCs, conformance queue |
| [`overlays/inference-sim/kustomization.yaml`](../deploy/trace-capture/overlays/inference-sim/kustomization.yaml) | Sets namespace `inference-sim`, includes `../../base` |
| [`base/conformance-queue.yaml`](../deploy/trace-capture/base/conformance-queue.yaml) | Kueue `ClusterQueue` — **one capture Job on the GPU at a time** |
| [`conformance-queue.yaml`](../deploy/trace-capture/conformance-queue.yaml) | Root copy of the queue manifest |
| [`tap-sidecar-patch.yaml`](../deploy/trace-capture/tap-sidecar-patch.yaml) | Kustomize patch template to inject the tap into an existing vLLM Deployment |

### Standard text capture (Qwen / H200)

| File | Purpose |
| --- | --- |
| [`base/h200-capture.yaml`](../deploy/trace-capture/base/h200-capture.yaml) | **Default capture rig:** loadgen → vllm-rs :5570 → tap → headless engine :5580 |
| [`base/h200-capture-agentic.yaml`](../deploy/trace-capture/base/h200-capture-agentic.yaml) | Python frontend (`/v1/messages`) for agent clients; tap with `--record-tokens` |
| [`h200-capture.yaml`](../deploy/trace-capture/h200-capture.yaml) | Root copy (may add cluster-specific namespace/SA) |
| [`h200-capture-agentic.yaml`](../deploy/trace-capture/h200-capture-agentic.yaml) | Root copy of agentic capture |

Typical pod layout (all tap captures):

```text
loadgen / curl :8000
    → frontend :5570 (ZMQ)
    → tap :5570↔:5580  (vllm-vcr record → /trace)
    → engine :5580     (real vLLM, GPU)
```

**Ops:** containers handshake once at startup — if any container restarts, **delete the pod**.
Traces live on `emptyDir`; fetch before scale-down.

### Offline replay (no GPU)

| File | Purpose |
| --- | --- |
| [`base/offline-replay.yaml`](../deploy/trace-capture/base/offline-replay.yaml) | Python frontend + `vllm-vcr play --replay-match prefix` (no tap, no engine) |
| [`offline-replay.yaml`](../deploy/trace-capture/offline-replay.yaml) | Root copy |

See [Agentic offline replay](./agentic-offline-replay.md).

### Multimodal and model-specific captures

Same tap topology; different model images and engine flags.

| File | Model / notes |
| --- | --- |
| [`base/gemma3-capture.yaml`](../deploy/trace-capture/base/gemma3-capture.yaml) | Gemma-3-27B multimodal |
| [`base/gemma4-capture.yaml`](../deploy/trace-capture/base/gemma4-capture.yaml) | Gemma-4-26B multimodal |
| [`base/gemma4-31b-capture.yaml`](../deploy/trace-capture/base/gemma4-31b-capture.yaml) | Gemma-4-31B FP8 |
| [`base/gemma4-31b-bf16-capture.yaml`](../deploy/trace-capture/base/gemma4-31b-bf16-capture.yaml) | Gemma-4-31B bf16 (quality baseline vs FP8) |
| [`base/gemma4-e4b-capture.yaml`](../deploy/trace-capture/base/gemma4-e4b-capture.yaml) | Gemma-4-E4B (smaller variant) |
| [`base/diffusiongemma-capture.yaml`](../deploy/trace-capture/base/diffusiongemma-capture.yaml) | DiffusionGemma (block-style decode) |
| [`base/diffusiongemma-notap.yaml`](../deploy/trace-capture/base/diffusiongemma-notap.yaml) | DiffusionGemma without tap (debug wedge) |
| [`base/diffusiongemma-single.yaml`](../deploy/trace-capture/base/diffusiongemma-single.yaml) | Single-container `vllm serve` counterfactual |

Each has a matching file at `deploy/trace-capture/<name>.yaml` when used outside kustomize.

### Weight caches (PVCs)

RWX volumes so redeploys skip re-downloading large weights:

| File | Model weights |
| --- | --- |
| [`base/pvcs/gemma3-model-cache-pvc.yaml`](../deploy/trace-capture/base/pvcs/gemma3-model-cache-pvc.yaml) | Gemma-3-27B |
| [`base/pvcs/gemma4-model-cache-pvc.yaml`](../deploy/trace-capture/base/pvcs/gemma4-model-cache-pvc.yaml) | Gemma-4-26B |
| [`base/pvcs/gemma4-31b-model-cache-pvc.yaml`](../deploy/trace-capture/base/pvcs/gemma4-31b-model-cache-pvc.yaml) | Gemma-4-31B FP8 |
| [`base/pvcs/gemma4-31b-bf16-model-cache-pvc.yaml`](../deploy/trace-capture/base/pvcs/gemma4-31b-bf16-model-cache-pvc.yaml) | Gemma-4-31B bf16 |
| [`base/pvcs/gemma4-e4b-model-cache-pvc.yaml`](../deploy/trace-capture/base/pvcs/gemma4-e4b-model-cache-pvc.yaml) | Gemma-4-E4B |

Root copies: `gemma*-model-cache-pvc.yaml` at the trace-capture top level.

### Related non-YAML files

| File | Purpose |
| --- | --- |
| [`models.toml`](../deploy/trace-capture/models.toml) | Conformance capture matrix; [`gen-capture-jobs.py`](../deploy/trace-capture/gen-capture-jobs.py) emits Kueue Jobs |
| [`run-capture.sh`](../deploy/trace-capture/run-capture.sh) | Drive loadgen against `h200-capture`, pull trace locally |
| [`validation-runner.sh`](../deploy/trace-capture/validation-runner.sh) | In-pod load driver for conformance Jobs |
| [`Dockerfile.frontend-cpu`](../deploy/trace-capture/Dockerfile.frontend-cpu) | CPU vLLM frontend wheel for agentic capture (no GPU in frontend container) |

---

## Quick picker

| Goal | Start here |
| --- | --- |
| Test P/D routing without GPU | [`deploy/llm-d-pd/helmfile.yaml`](../deploy/llm-d-pd/helmfile.yaml) |
| Test prefix-cache routing | [`deploy/llm-d-prefix-cache/helmfile.yaml`](../deploy/llm-d-prefix-cache/helmfile.yaml) |
| Capture H200 latency trace | `kustomize build deploy/trace-capture/overlays/inference-sim` + `h200-capture` |
| Capture agent / Messages API | `h200-capture-agentic.yaml` |
| Replay capture offline (no GPU) | `offline-replay.yaml` |
| Conformance goldens (serialized GPU) | `conformance-queue.yaml` + [`models.toml`](../deploy/trace-capture/models.toml) |
