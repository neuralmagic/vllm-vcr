# Real-world examples

Two worked examples demonstrating how to run the simulator locally and deploy it to Kubernetes.

## Example 1: Local development on macOS

Running the simulator on a Mac for testing and development. No GPU or model weights required; the vLLM frontend downloads the tokenizer on first use.

### Prerequisites

- Rust 1.85 or newer
- A vLLM checkout with the Rust frontend built

### 1. Build the simulator

From the repo root:

```bash
cargo install --path . --locked
```

This installs the `vllm-vcr` binary with `record`, `play`, `inspect`, and `completions` subcommands.

### 2. Start the simulator engine

Launch the mock engine-core backend:

```bash
vllm-vcr play --handshake-address tcp://127.0.0.1:29550 --log-requests
```

The simulator binds `tcp://127.0.0.1:29550` for the ZMQ handshake and waits for a vLLM frontend to connect. It generates random tokens by default.

### 3. Start the vLLM Rust frontend

In a separate shell, start the vLLM frontend pointed at the same handshake address:

```bash
# Assumes vllm-rs is at $HOME/git/vllm-main/rust/target/debug/vllm-rs
# Override with FRONTEND_BIN=/path/to/vllm-rs if needed
./scripts/e2e.sh
```

Or manually:

```bash
vllm-rs serve Qwen/Qwen3-0.6B \
  --data-parallel-size 1 \
  --data-parallel-size-local 0 \
  --handshake-port 29550 \
  --host 127.0.0.1 \
  --port 8000
```

The frontend connects to the simulator, completes the handshake, and serves OpenAI-compatible HTTP endpoints at `http://127.0.0.1:8000`.

### 4. Send a test request

Once the frontend reports ready (first run downloads the tokenizer from Hugging Face):

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 16,
    "stream": true
  }'
```

The frontend tokenizes the request, the simulator generates random tokens, and the frontend streams the SSE response.

### 5. Replay a captured trace (optional)

If you have a trace captured with `vllm-vcr record --record-tokens`, replay it for content-identical output:

```bash
# Restart the simulator with the trace
vllm-vcr play \
  --handshake-address tcp://127.0.0.1:29550 \
  --replay-tokens /path/to/trace.jsonl.gz \
  --log-requests
```

Traces can be local JSONL files (`.jsonl` or `.jsonl.gz`) or `s3://bucket/key` URIs if AWS credentials are configured.

### Where traces live

- **Repo golden traces:** not committed; live in the private conformance bucket referenced by `conformance/manifest.toml` (CI fetches them by sha256).
- **Local captures:** written to the path specified by `vllm-vcr record --trace-out <path>`.
- **S3 traces:** referenced by `s3://bucket/key` URI; the simulator reads them transparently via the `sim-s3` crate.

## Example 2: Kubernetes dual-container deployment

Deploying the simulator to Kubernetes using the dual-container approach: a vLLM frontend container and a simulator engine container in one pod, wired over the in-pod ZMQ handshake.

### Architecture

- **Frontend container:** runs `vllm-rs serve` (the Rust frontend), binds the handshake on port 29550, serves HTTP on port 8000.
- **Engine container:** runs `vllm-vcr play`, connects to the frontend's handshake.

Containers in a pod share the network namespace, so the two talk over `tcp://127.0.0.1:29550` loopback ZMQ; no shared volume is needed. The image (`ghcr.io/neuralmagic/vllm-vcr`) ships both binaries, so both containers use the same image with different commands.

### Minimal deployment manifest

Commands mirror what `entrypoint.sh` runs, split across two containers:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-vcr-demo
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-vcr-demo
  template:
    metadata:
      labels:
        app: vllm-vcr-demo
    spec:
      containers:
        # vLLM Rust frontend: binds the engine-core handshake, serves HTTP.
        - name: frontend
          image: ghcr.io/neuralmagic/vllm-vcr:latest
          command: ["vllm-rs"]
          args:
            - serve
            - Qwen/Qwen3-0.6B
            - --data-parallel-size=1
            - --data-parallel-size-local=0
            - --handshake-port=29550
            - --host=0.0.0.0
            - --port=8000
          ports:
            - name: http
              containerPort: 8000
              protocol: TCP
          readinessProbe:
            httpGet:
              path: /v1/models
              port: http
            periodSeconds: 5
            timeoutSeconds: 2
            failureThreshold: 3
          resources:
            requests:
              cpu: "1"
              memory: 2Gi
        # Simulator engine: joins the handshake as the headless DP engine.
        - name: engine
          image: ghcr.io/neuralmagic/vllm-vcr:latest
          command: ["vllm-vcr"]
          args:
            - play
            - --handshake-address=tcp://127.0.0.1:29550
            - --pd-role=both
            # Latency model knobs (milliseconds; omit for instant tokens)
            - --time-to-first-token=50
            - --inter-token-latency=10
            - --log-requests
          resources:
            requests:
              cpu: "1"
              memory: 1Gi
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-vcr-demo
  namespace: default
spec:
  selector:
    app: vllm-vcr-demo
  ports:
    - name: http
      port: 80
      targetPort: 8000
      protocol: TCP
```

Apply it:

```bash
kubectl apply -f vllm-vcr-demo.yaml
```

Startup order does not matter: the frontend binds the handshake socket and the engine retries until it connects. If either container dies, its restart re-runs the handshake.

### Single-container alternative (`entrypoint.sh`)

The image's default entrypoint (`/usr/local/bin/entrypoint.sh`) launches both processes in one container, configured via env instead of args. This is what the llm-d P/D mock deployments use; if either process exits, the entrypoint tears the pod down so Kubernetes restarts it.

```yaml
      containers:
        - name: vllm-vcr
          image: ghcr.io/neuralmagic/vllm-vcr:latest
          env:
            - name: MODEL
              value: "Qwen/Qwen3-0.6B"
            - name: MOCK_TTFT_MS
              value: "50"
            - name: MOCK_ITL_MS
              value: "10"
          ports:
            - name: http
              containerPort: 8000
```

Key env vars read by `entrypoint.sh`:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL` | *required* | Hugging Face model id (tokenizer) |
| `MOCK_PD_ROLE` | `both` | `both` (monolithic), `prefill`, or `decode` (P/D split) |
| `MOCK_HANDSHAKE_PORT` | `29550` | ZMQ handshake port |
| `VLLM_PORT` | `8000` (`8200` for decode) | HTTP port for the frontend |
| `MOCK_TTFT_MS` | `0` | Time to first token (ms, mean) |
| `MOCK_ITL_MS` | `0` | Inter-token latency (ms, mean) |
| `MOCK_MAX_NUM_SEQS` | `128` | Concurrent sequences (scheduler batch size) |
| `MOCK_MAX_NUM_BATCHED_TOKENS` | `2048` | Per-step token budget |

See `entrypoint.sh` for the full list (latency std-devs, KV-cache events, failure injection, NIXL side-channel).

### Accessing the service

Port-forward to test:

```bash
kubectl port-forward svc/vllm-vcr-demo 8000:80
```

Then send a request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 16
  }'
```

### Trace capture variant (separate containers)

The real GPU trace-capture deployment (`deploy/trace-capture/h200-capture.yaml`) uses **three** containers in one pod instead of the unified `entrypoint.sh` approach:

1. **engine:** the real headless vLLM (`vllm serve --headless --data-parallel-rpc-port 5580`), binds GPU.
2. **tap:** `vllm-vcr record`, a transparent ZMQ proxy that relays frames between frontend and engine and writes `trace.jsonl`.
3. **frontend:** `vllm-rs serve --handshake-port 5570`, connects to the tap.

The tap sits on the wire: frontend :5570 → tap → engine :5580. It records every msgpack frame verbatim and writes the trace to a shared `emptyDir` volume mounted at `/trace`.

This three-container approach is for **capture** only (requires GPU, real vLLM engine). For **replay** or **testing** (no GPU), use the dual-process `entrypoint.sh` approach shown in the minimal manifest above.

### Image tags

- `ghcr.io/neuralmagic/vllm-vcr:latest` — the `default = true` line from `compat.toml` (currently `v0.23.0`).
- `ghcr.io/neuralmagic/vllm-vcr:vllm0.23` — floating, latest sim for the 0.23 line.
- `ghcr.io/neuralmagic/vllm-vcr:0.1.3-vllm0.23` — immutable, sim version × vLLM line.

For multi-version support, see [Versioning](./versioning.md) and [Conformance](./conformance.md).
