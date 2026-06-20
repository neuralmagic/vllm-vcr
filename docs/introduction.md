# vllm-vcr

Record, play, and inspect vLLM **V1 engine-core** traces. One binary with three
subcommands, named for the VCR metaphor:

- **`record`** taps a live vLLM frontend ↔ engine-core link (a transparent ZMQ
  proxy) and writes a JSONL trace.
- **`play`** runs a mock engine-core backend that speaks the real ZMQ + msgpack
  protocol, replaying a recorded trace or simulating from a latency model. No model
  weights, no GPU.
- **`inspect`** converts benchmark reports, summarizes traces, renders Perfetto
  timelines, and runs calibration.

With the `nixl` feature and a working libnixl/UCX runtime, `play` also moves
simulated KV-cache bytes between prefill and decode instances over NIXL.

## Why

Testing the software *around* a vLLM engine — frontends, cache-aware routers,
schedulers, autoscalers, whole CI matrices — normally means standing up real GPUs
and model weights just to get realistic timing and protocol behavior. `vllm-vcr`
takes the GPU out of that loop:

1. **CPU-only replay of real engine behavior.** `record` captures a real vLLM
   engine's per-request timing (TTFT, per-token gaps) and, optionally, its exact
   output tokens, straight off the engine-core protocol. `play` serves that capture
   back — sampled from a fitted latency model, or verbatim and byte-for-byte
   identical — on CPU, with no GPU and no model weights. A trace captured once on an
   H200 can then drive a frontend, a router, or an entire CI run on commodity
   machines.
2. **Faithful frontend compatibility.** It speaks the real ZMQ + msgpack
   engine-core protocol and sits behind vLLM's Rust or Python frontend unchanged, so
   the frontend still handles tokenization, chat templates, tool calling, streaming,
   and OpenAI-compatible request handling. Only the model backend is simulated.

`llm-d-inference-sim` models prefill/decode in the control plane (it adjusts latency
and finish metadata) but does not move KV-cache bytes. As an optional third
capability, `vllm-vcr`'s `play` *can* move simulated KV-cache bytes between prefill
and decode instances over [NIXL](https://github.com/ai-dynamo/nixl) (UCX backend) for
P/D data-plane testing — behind the `nixl` feature, and still without CUDA or model
weights.

## Where to start

- New here? Read [Architecture](./architecture.md), then [Install](./install.md)
  and the [Quick start](./quick-start.md).
- Capturing and replaying real traces? Jump to
  [Trace replay and calibration](./trace-replay/index.md).
- Running it across vLLM versions in CI? See [Versioning](./versioning.md) and
  [Conformance](./conformance.md).
