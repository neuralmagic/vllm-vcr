# Test suite

Integration tests for the vLLM VCR engine simulator. All engine tests use real
ZMQ transport and a real `EngineCoreClient`; there are no mocks.

| Suite | What it covers |
|---|---|
| `engine_core_e2e.rs` | Token streaming, finish reasons, abort/shutdown, prefix-cache reset, LoRA lifecycle, P/D handoff |
| `engine_synthetic_e2e.rs` | Replay modes (token, latency, prefix, gzip) against programmatically generated traces: batch context, speculative, diffusion, edge cases, mixed concurrency |
| `dataset_replay_e2e.rs` | `--replay-tokens` with a HuggingFace dataset file; ignored by default (downloads a tokenizer), run with `-- --ignored` |
| `trace_validation_ci.rs` | Structural validation and replay of the fixtures in `fixtures/synthetic/` |
| `real_trace_replay.rs`, `spec_replay_fidelity.rs`, `diffusion_replay_demo.rs`, `gemma4_replay_demo.rs`, `closed_loop_prefix_replay.rs` | Byte-identical replay of real GPU captures |
| `calibrate.rs` | Timing model quantile fidelity and open-loop arrival replay |
| `conformance.rs` | Golden testing against real vLLM captures from S3 (needs AWS credentials) |
| `tap_e2e.rs`, `kv_events_pubsub.rs` | Recording tap and KV-event publishing |
| `nixl_loopback.rs` | NIXL KV transfer over loopback (`nixl` feature) |

Shared helpers live in `test_helpers.rs` (RAII sim guard, unique IPC endpoints,
temp trace files) and `synthetic_trace_generator.rs` (seeded trace generators;
every generator stamps `arrival_ms` and `output_token_ids`, which token replay requires).

Run everything with `cargo test --workspace`, or one suite with
`cargo test --test <name>`. The synthetic suite honors
`SYNTHETIC_E2E_FAST_MODE=1` (set in CI) to shrink generated latencies.
