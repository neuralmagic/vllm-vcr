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
cargo xtask pin-vllm 0.24
VLLM_TARGET_VERSION=v0.24.0 cargo build --workspace   # no --locked
```

## Capability cfgs

Where the crate's API genuinely diverges in a way owning a type can't hide (a
field whose *type* differs per line), the engine gates on a discrete capability,
not a version number. `build.rs` maps the target line to cfgs and declares them
with `cargo::rustc-check-cfg`:

- `vllm_outputs_enum` — the 0.25 protocol restructure. The types moved out of
  `protocol` into `protocol::{request, output, sampling}` (mod.rs stopped
  re-exporting them), and `EngineCoreOutputs` was repurposed from the flat wire
  struct into the classified enum 0.24 called `ClassifiedEngineCoreOutputs`, with
  the flat struct made private. Both landed in the same release, so one cfg gates
  both. On 0.25+.
- `vllm_cache_creation_tokens` — `PrefillStats` gained
  `num_cache_creation_tokens`. On 0.26+.

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
| Protocol type paths | `protocol::X` (<=0.24) vs `protocol::{request,output,sampling}::X` (0.25+) | `sim-protocol::vllm` re-exports the whole surface from the right path per line; every crate imports from there, never from `vllm_engine_core_client::protocol` directly |
| Output envelope | flat public struct (<=0.24) vs classified enum over a private wire struct (0.25+) | `sim-protocol::vllm::Envelope` alias + `request_batch()`/`utility()` constructors and `request_outputs()`/`scheduler_stats()`/`utility_output()` accessors. The wire is the same 8-field tuple on both; only the Rust API moved |
| Prefill stats | `num_cache_creation_tokens` added in 0.26 | `..Default::default()` plus a `vllm_cache_creation_tokens`-gated assignment in `engine::prefill_stats` |
| Ready response | `EngineCoreReadyResponse.vllm_version` absent before 0.23 | tap decodes its own tolerant `CapturedReadyInfo{vllm_version:Option<String>}` |
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
- **rc** (`v0.26.1rc0`): the newest release candidate ahead of the stable window,
  bumped (tag + rev) by the release watcher. Never `default`, never fidelity-validated.
- **0.26** (`v0.26.0`, default): builds against upstream `568afb3a`. Carries the 0.25
  restructure plus `PrefillStats.num_cache_creation_tokens`.
- **0.25** (`v0.25.1`): builds against upstream `752a3a50`. Same protocol layout as
  0.26 minus the prefill-stats field.
- **0.24** (`v0.24.0`): builds against upstream `ee0da84a`. The last line before the
  restructure, and the reason `vllm_outputs_enum` exists: it is the only line in the
  window where the cfg is off, so it is what keeps that branch honest.

Every line is `fidelity_validated = true`: each carries three goldens (two
prefix-cached multiturn seeds plus one nocache multiturn, Qwen3-8B on H200),
captured against that line's released engine image and replaying byte-identically.
See `conformance.md` for the capture runbook.

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

- **`cargo xtask nightly-golden-entry` hardcodes `line = "nightly"`.** It was written
  for the canary, so registering a release line's golden means hand-correcting the
  emitted entry. Give it a `--line` flag.
