# Testing

The main local gate is:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked --no-deps -- -D warnings
cargo test --workspace --locked
```

The full smoke scripts also boot a real vLLM frontend:

```bash
./scripts/e2e.sh        # boots vllm-rs + this engine, asserts streaming + non-streaming flows
./scripts/e2e_lora.sh   # loads a LoRA adapter, asserts vllm:lora_requests_info names it
./scripts/e2e_generate.sh # exercises /inference/v1/generate token-in/token-out
```

These scripts need `vllm-rs` built once (`cargo build --bin vllm-rs` in the vLLM
`rust/` workspace). By default the scripts look for the binary at
`$HOME/git/vllm-main/rust/target/debug/vllm-rs`; override this path with
`FRONTEND_BIN=/path/to/vllm-rs`. The first run fetches the tokenizer from
Hugging Face.

`e2e_lora.sh` needs a frontend that exports `vllm:lora_requests_info` from the
frontend metrics path. The image and current default protocol pin qualify; if you use
your own checkout, point `FRONTEND_BIN` at a compatible build.

## Per-line kind e2e

CI's `kind-e2e` job runs the same kind of smoke against the image instead of host
binaries, once per `compat.toml` line. It builds `Dockerfile.slim` (the line's vllm-rs
frontend plus vllm-vcr, no UCX/NIXL), starts it as one pod on a kind cluster through
the shipped `entrypoint.sh`, and drives the OpenAI surface with `ci/kind-e2e.sh`:

- `/v1/models`, a non-streaming completion and a streaming chat completion run to
  `max_tokens` with the right finish reason, usage, and SSE terminator.
- A repeated prompt reports `cached_tokens` from the engine's prefix cache, and
  `/reset_prefix_cache` round-trips as an engine utility call.
- A LoRA adapter loads, serves a completion, and unloads at runtime.
- A client disconnect mid-stream reaches the engine as an abort.
- The engine's own counters (`/stats` on the control API) match what the frontend
  served, with no failures and nothing left running, and the pod log has no decode
  errors or panics.

Stable lines gate merges through the `kind-e2e-gate` check; `nightly` and `rc` report
without blocking. This proves the frontend and engine interoperate on each line; it
does not prove timing fidelity, which is what the conformance goldens are for
([conformance.md](conformance.md)).

Locally (with podman, also `export KIND_EXPERIMENTAL_PROVIDER=podman`):

```bash
just image-build-slim 0.33
just kind-e2e 0.33
```

`image-build-slim` pins `Cargo.toml` to the line like `image-build-line`; run
`cargo xtask pin-vllm <default line>` afterwards to restore the committed pin.
