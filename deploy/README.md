# Deploy

Kubernetes manifests for three testbeds:

| Directory | Purpose |
| --- | --- |
| [`llm-d-pd/`](llm-d-pd/) | GPU-free **prefill/decode** mock servers + llm-d router |
| [`llm-d-prefix-cache/`](llm-d-prefix-cache/) | GPU-free **prefix-cache routing** mock servers + router |
| [`trace-capture/`](trace-capture/) | **Record** real vLLM engine traces via `vllm-vcr record` |

**Full file-by-file reference:** [docs/deploy.md](../docs/deploy.md) (also on the [published docs site](https://neuralmagic.github.io/vllm-vcr/deploy.html)).

Per-area runbooks:

- [llm-d-pd/README.md](llm-d-pd/README.md)
- [llm-d-prefix-cache/README.md](llm-d-prefix-cache/README.md)

Trace capture uses kustomize overlays — see [`justfile`](../justfile) (`capture-up`, `conformance-*`) and [Conformance](../docs/conformance.md).
