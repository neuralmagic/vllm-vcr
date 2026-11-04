# Multi-version vLLM support: the compatibility shim

How the simulator builds against more than one vLLM line from a single `main`
branch. Read `versioning.md` first for the strategy (build matrix, one image per
line, `compat.toml` as source of truth) and `conformance.md` for capture/replay.
This doc covers the *code* side: how we absorb the protocol crate's API drift.

## Contents

- [The shape of the problem](#the-shape-of-the-problem)
- [Per-line builds: pin, don't patch](#per-line-builds-pin-dont-patch)
- [Capability cfgs](#capability-cfgs)
- [What the shim owns](#what-the-shim-owns)
- [Testing across lines](#testing-across-lines)
- [Current window](#current-window)
- [Follow-ups](#follow-ups)

## The shape of the problem

The wire protocol comes from one git dependency, `vllm-engine-core-client`,
which lives in the vLLM repo (`rust/src/engine-core-client/`). Its API drifts
across releases. Cargo can hold only one rev of a git dep per build, so each line
is a separate build (the matrix). The job of the shim is to let the same source
compile against each line's crate, isolating the divergences in one place.

## Per-line builds: pin, don't patch

Cargo rejects a `[patch]` that redirects a git dependency to a different rev of
the **same** source ("patches must point to different sources"). So the per-line
rev is swapped in `[workspace.dependencies]`, not via `--config patch`. `build.rs`
cannot do it either: dependency resolution happens before any build script runs.

`cargo xtask pin-vllm <line>` reads `compat.toml` and rewrites `Cargo.toml`: it sets
the `vllm-engine-core-client` rev to the line's `protocol_rev`, and inserts,
rewrites, or removes the fork `[patch]` to match the line's `patch_repo`/`patch_rev`
(a fork is a *different* source, so it is allowed to `[patch]`). The committed
`Cargo.toml` carries **no** `[patch]` block (the default line builds upstream), so
a forked line's block is inserted, not rewritten; the script strips any existing
block first, so it's idempotent. After the rewrite the rev no longer matches
`Cargo.lock`, so per-line builds omit `--locked`.

**Gotcha:** the script changes the manifest but not the environment, and
`build.rs` reads the line from `VLLM_TARGET_VERSION` (falling back to the
`compat.toml` default). So a per-line build must set **both**: run the script
*and* export `VLLM_TARGET_VERSION=<tag>`. The CI matrix does both; a local
older-line build must too:

```sh
cargo xtask pin-vllm 0.31
VLLM_TARGET_VERSION=v0.31.0 cargo build --workspace   # no --locked
```

## Capability cfgs

Where the crate's API genuinely diverges in a way owning a type can't hide (a
field whose *type* differs per line), the engine gates on a discrete capability,
not a version number. `build.rs` maps the target line to cfgs and declares them
with `cargo::rustc-check-cfg`:

- `vllm_weight_version` — 0.32 added `RequestBatchOutputs.weight_version`.
  Gates `sim_protocol::vllm::request_batch`. On 0.32+.
- `vllm_stream_delivery` — 0.32's `EngineCoreOutputStream` yields
  `EngineCoreStreamDelivery` batches instead of single outputs. Gates
  `sim_protocol::vllm::output_stream`, which flattens them back to one item per
  output. On 0.32+.
- `vllm_lora_tensorizer` — `LoraRequest.tensorizer_config_dict`, a positional
  slot 0.33 removed. Gates `sim_protocol::vllm::lora_request`. On lines below
  0.33.

Retired cfgs: `vllm_outputs_enum` (the 0.25 protocol restructure: types moved
into `protocol::{request, output, sampling}` and `EngineCoreOutputs` became the
classified enum over a private flat struct) went when 0.24 left the window, and
`vllm_cache_creation_tokens` (`PrefillStats.num_cache_creation_tokens`, added
in 0.26) went when 0.25 did, and `vllm_engine_id_u16`
(`EngineId::from_engine_index` narrowed to u16 in 0.28) went when 0.27 did.
Every supported line shares all three, so the outputs shim is unconditional,
the field is assigned directly, and `engine_id_from_index` passes the u16 through.

The capability list itself lives in `sim_compat::capabilities`, not in a build
script: cfgs from `build.rs` only reach the crate that owns it, so both the root
crate and `sim-protocol` (which owns the outputs shim) run their own build script
and call the same `capabilities::emit`. A crate that grows cfg-gated code needs a
`build.rs` of its own.

rust-analyzer does not run build scripts against the pinned line, so it shows
false-positive errors on whichever branch is inactive; `cargo build` is the truth.

## What the shim owns

The principle: **own a tolerant decode wherever possible (no cfg), and reach for a
capability cfg only when a field's type differs per line.** Owned types
deserialize the same wire on every line (serde ignores unknown fields).

| Concern | Divergence | Shim |
| --- | --- | --- |
| Handshake harness types | `mock_engine` module absent before 0.23 (we never used its behavior, only structs) | `sim-protocol::mock_engine` owns `MockEngineSockets`/`MockEngineDataSockets`/`MockCoordinatorSockets` + `DEFAULT_MOCK_MAX_MODEL_LEN` + `default_dtype()` |
| Request-type frame | `EngineCoreRequestType::from_frame` is head-only | `sim-protocol::wire::request_type_from_frame` (1-byte decode) |
| Lora request | wire form is a positional array whose trailing fields differ per line | `LoraSpec{lora_int_id,lora_name}` (own, reads positions 0/1) for the add_lora call + registry |
| Protocol type paths | none since 0.24 rolled off; every line uses `protocol::{request,output,sampling}` | `sim-protocol::vllm` re-exports the whole surface; every crate imports from there, never from `vllm_engine_core_client::protocol` directly, so the next path move lands in one file |
| Output envelope | the classified enum serializes through a private flat struct, so it cannot be built inline | `sim-protocol::vllm::Envelope` alias + `request_batch()`/`utility()` constructors and `request_outputs()`/`scheduler_stats()`/`utility_output()` accessors |
| Engine identity | `EngineId::from_engine_index` takes u16; the CLI counts engines in u32 | `sim-protocol::vllm::engine_id_from_index(u32)` validates the two-byte wire range |
| Request-batch envelope | `RequestBatchOutputs.weight_version` exists on 0.32+ only | `request_batch()` sets it to `None` behind `vllm_weight_version` |
| Client output stream | `EngineCoreOutputStream` yields one output per item (<=0.31) vs `EngineCoreStreamDelivery` batches (0.32+) | `sim-protocol::vllm::output_stream()`, `vllm_stream_delivery`-gated; every `EngineCoreClient::call` site goes through it |
| Boxed output fields | 0.32 boxed `prefill_stats`, `kv_transfer_params`, and the other sparse `EngineCoreOutput` fields | the engine assigns `Some(value.into())`, which builds `T` or `Box<T>`; serde encodes both the same, no cfg |
| zeromq source | 0.32+ pins zmq.rs by git rev; older lines use crates.io 0.6.0 | root `[patch.crates-io]` points `zeromq` at the same git rev, so one `zeromq` (and one `PeerIdentity`) exists on every line |
| Multimodal modality | `MmFeatureSpec.modality` is the typed `MmModality` on every line | re-exported from `sim-protocol::vllm` |
| Ready response (decode) | `EngineCoreReadyResponse.vllm_version` absent before 0.23 | tap decodes its own tolerant `CapturedReadyInfo{vllm_version:Option<String>}` |
| Ready response (emit) | 0.27 frontends require the parallel-config sizes, scheduler caps, `instance_id`, and `kv_events_config`; 0.29 frontends require `supports_lora` and `max_loras` (and reject `supports_lora != (max_loras > 0)`) | sim-owned map-encoded `SimReadyResponse` superset: every field any line requires is always emitted, older frontends skip unknown keys, no cfg |
| Output decode | `decode_engine_core_outputs` takes any `AsRef<[u8]>` (<=0.28) vs `&[Bytes]` (0.29+) | the tap passes the `Bytes` frames zeromq already hands it, which satisfy both signatures, no cfg |
| Lora request | the positional `tensorizer_config_dict` slot exists below 0.33 only | `sim-protocol::vllm::lora_request()` builds it, `vllm_lora_tensorizer`-gated; the engine side reads positions 0/1 through `LoraSpec` and is unaffected |
| Utility request | `EngineCoreUtilityRequest` derives `Deserialize` only on 0.23+ (crate was client-only) | `engine_core::UtilityRequestSpec` (`Deserialize_tuple`, matches the wire tuple) |

The wire types still come from the crate (the matrix's whole point: catch drift at
compile time). The shim only covers the spots where our *decoding/server* role
needs something the client-oriented crate lacks on an older line.

The first three rows predate the current window: the lines that forced them (0.22,
0.23) have rolled off. They stay because owning them is still the right call for a
server-side role — a tolerant decode we control cannot be broken by an upstream
field addition — not because any supported line requires them.

## Testing across lines

The matrix runs per line (see `ci.yml`):

- `cargo build --workspace` — the "does the wire still compile" gate.
- `cargo test --workspace --lib` — unit tests (compile + pass on every line).
- `cargo test --test conformance` — the conformance runner (skips until goldens).
- `ci/kind-e2e.sh` (the `kind-e2e` job) — the line's slim image on kind: the real
  vllm-rs frontend and the engine over the in-pod handshake, driven over HTTP (see
  [testing.md](testing.md#per-line-kind-e2e)).

The full-stack e2e integration tests (`tests/engine_core_e2e.rs`,
`tests/tap_e2e.rs`) drive the *real* `EngineCoreClient`, whose API
is incomplete on older lines, so they target the default line via the
`build-and-test` job, not each matrix leg.

The shim's own contract (that an `Envelope` built by the constructors survives a
msgpack round trip) is unit-tested in `sim-protocol::vllm`, so it runs on every
matrix leg — that is what proves the inactive cfg branch is real and not just
compiling.

## Current window

The window is N, N-1, N-2: three stable lines, plus the two trackers riding ahead
of them. No line in the window needs a `[patch]` fork; every one builds against
upstream, so the repo has no external-fork dependency.

- **nightly** (`nightly`): tracks vLLM main, `protocol_rev` is the latest post-merge
  commit (bumped regularly). build.rs treats the non-`vX.Y` tag as the newest line, so
  all capability cfgs are on. It exists to catch wire drift before a release lands: the
  live-HEAD nightly canary pins to upstream `main`, builds, runs unit tests, runs the
  HEAD-client protocol e2e suite, and runs the conformance runner.
- **rc** (`v0.32.1rc0`): the newest release candidate ahead of the stable window,
  bumped (tag + rev) by the release watcher. Never `default`, never fidelity-validated.
- **0.33** (`v0.33.0`, default): builds against upstream `4ba787ec` (the tag
  peels to the rc3 commit). Dropped the positional
  `LoraRequest.tensorizer_config_dict` slot (`vllm_lora_tensorizer` is on below
  0.33) and added `EngineCoreOutput.error`.
- **0.32** (`v0.32.0`): builds against upstream `5df63a72`. Boxed the sparse
  `EngineCoreOutput` fields, added `RequestBatchOutputs.weight_version`
  (`vllm_weight_version`), switched the client stream to batched deliveries
  (`vllm_stream_delivery`), and moved zmq.rs to a git rev.
- **0.31** (`v0.31.0`): builds against upstream `db9527a4`. The oldest line and
  the only stable one without the 0.32 cfgs, so it keeps those branches honest.

No stable line has goldens yet: 0.27's were the last, and they left the window
with it. 0.31, 0.32, and 0.33 run non-gating until the Golden Capture workflow
registers theirs. See `conformance.md`.

### Fork patches

No line in the window needs one today, but the mechanism stays: a line can set
`patch_repo`/`patch_rev` in `compat.toml` to `[patch]`-override the crate with a
fork carrying a fix that isn't upstream yet. A fork is a *different* source, so
Cargo allows it where a same-source rev patch is rejected. `cargo xtask pin-vllm`
inserts, rewrites, or removes the block per leg; the committed `Cargo.toml` has
none, since the default line builds upstream.

The last use was the 0.22/0.21 serde-defaults fork (`vllm-project/vllm#45848`
backported for the tap's capture-time decode). Both lines left the window in the
0.26 roll, so the external `wseaton/vllm` dependency is gone with them.

## Follow-ups

None currently.
