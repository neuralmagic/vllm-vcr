---
name: vllm-roll
description: Roll the vLLM support window in compat.toml, absorb protocol drift in the shim, test every supported line, and cut a vllm-vcr release. Use when a new vLLM release lands, when the release watcher or nightly canary goes red, or when cutting a release of this repo.
---

# Rolling the vLLM window and cutting a release

Three jobs that share one pipeline: find the drift, absorb it, prove every line
still builds and replays. The release is the last step, not the first.

`compat.toml` is the source of truth. The manifest diff *is* the roll; everything
else (CI matrix, image tags, handshake guard, build.rs cfgs) reads from it. Read
`docs/versioning.md` for the strategy and `docs/multi-version-shim.md` for the
code side before making decisions.

## 0. What the automation already knows

The watchers do this daily and file issues when they fail. Check them first —
a red run usually has the exact compiler error waiting in it.

```bash
gh run list --workflow vllm-release-watch.yml --limit 5
gh run list --workflow nightly-canary.yml --limit 5
gh run view <run-id> --log 2>/dev/null | grep -E "^error|error\[E"
```

Then resolve upstream state yourself. Note `git ls-remote` returns commit shas
for lightweight tags; if a `<tag>^{}` row comes back, that row is the commit.

```bash
git ls-remote --tags https://github.com/vllm-project/vllm.git \
  | grep -vE '\^\{\}' | grep -E 'refs/tags/v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V -k2 | tail
git ls-remote https://github.com/vllm-project/vllm.git refs/heads/main   # nightly rev
```

## 1. Decide the window

**The window is N, N-1, N-2** — three stable lines. Default is always the newest
stable. Rolling in N+1 drops the oldest. The `nightly` and `rc` trackers ride
ahead and are never `default`.

Edit `compat.toml` by hand (or let `cargo xtask watch-stable` do it), then update
the per-line comments to say why each line is there. Drop `patch_repo`/`patch_rev`
for any line that no longer needs a fork.

## 2. Find the drift

Compare the protocol crate between the old default and the new one. The crate
lives at `rust/src/engine-core-client/` in the vLLM repo; fetching individual
files beats cloning it.

```bash
gh api "repos/vllm-project/vllm/contents/rust/src/engine-core-client/src/protocol?ref=v0.26.0" \
  -q '.[] | "\(.type) \(.name)"'
gh api "repos/vllm-project/vllm/contents/rust/src/engine-core-client/src/protocol/mod.rs?ref=v0.26.0" \
  -q .content | base64 -d
```

What to look for, in the order it bites:

- **`mod.rs` re-exports.** Types get moved into submodules and the re-export
  dropped; that breaks imports without changing the wire at all.
- **Tuple-encoded structs.** Anything deriving `Serialize_tuple` encodes
  positionally, so a new field changes the wire arity. `EngineCoreRequest`,
  `EngineCoreOutput`, and the output envelope are all tuple-encoded.
- **Types that changed meaning under the same name.** The 0.25 restructure
  repurposed `EngineCoreOutputs` from the flat wire struct into the classified
  enum. A same-name type that means something else is worse than a rename,
  because nothing fails to resolve.
- **Fields whose *type* changed** (e.g. an `OpaqueValue` becoming typed).

## 3. Absorb it in the shim, not at the call sites

The principle: **own a tolerant decode where possible; reach for a capability cfg
only when the API genuinely differs per line.** Never scatter `#[cfg]` across call
sites — put it in one place in `crates/sim-protocol/src/vllm.rs`, which is the
single import surface for the whole workspace. No crate should import from
`vllm_engine_core_client::protocol` directly.

- Path-only moves: add the re-export to `vllm.rs` behind the cfg.
- Shape changes: add a constructor/accessor pair there, so call sites stay
  identical on every line.
- New capability: declare it in `crates/sim-compat/src/capabilities.rs` (the
  `ALL` array plus a `line_at_least` branch) with a comment saying what it gates.

**Gotcha:** build-script cfgs only reach the crate that owns the script. Both the
root crate and `sim-protocol` have a `build.rs`, and both call
`sim_compat::capabilities::emit`. A crate that grows cfg-gated code needs its own
`build.rs`.

**Gotcha:** rust-analyzer does not run build scripts against the pinned line, so
it reports false errors on whichever cfg branch is inactive. `cargo build` is the
truth; ignore the squiggles.

## 4. Test every line

A per-line build needs **both** the manifest pin and the env stamp, and must omit
`--locked` (the pin moved the rev off the lockfile).

```bash
# bash, not fish: `eval` of the frontend-args assignments sets TAG/FREPO/FREF.
for line in 0.26 0.25 0.24; do
  eval "$(cargo xtask frontend-args "$line")"
  cargo xtask pin-vllm "$line"
  cargo update -p vllm-engine-core-client
  VLLM_TARGET_VERSION="$TAG" cargo build --workspace --tests
  VLLM_TARGET_VERSION="$TAG" cargo test --workspace --lib
done
cargo xtask pin-vllm <default-line>   # restore the committed pin
cargo update -p vllm-engine-core-client
```

The oldest line in the window is the one that keeps the inactive cfg branch
honest — if every line takes the same branch, the shim is untested. The
round-trip tests in `sim_protocol::vllm` are what prove the branch actually
works rather than merely compiling.

Then the default-line gates:

```bash
cargo fmt --all
cargo clippy --all --benches --tests --examples --all-features
cargo test --workspace
cargo test --test engine_core_e2e --test tap_e2e   # the real ZMQ protocol e2e
```

Also update, or the roll is half done:

- `Cargo.toml` — the committed rev must equal the default line's `protocol_rev`.
- `Dockerfile` — `VLLM_REF` and `VLLM_TARGET_VERSION` defaults.
- `docs/versioning.md`, `docs/multi-version-shim.md` — the window listing,
  capability list, and shim table.

## 5. Fidelity is a separate gate from "it builds"

`cargo build` proves the wire still parses. It says nothing about whether the sim
still *reproduces* the engine. That is conformance: golden captures replayed
GPU-free (`docs/conformance.md`).

A line enters the window as `fidelity_validated = false` and only flips to `true`
after a golden for **that line** is captured on the GPU rig, uploaded, registered
in `conformance/manifest.toml`, and passing the replay gate. Captures do not
transfer across lines — a new `protocol_rev` invalidates the old ones.

Capturing needs the cluster, not a dev box:

```bash
just capture-up && just capture-status    # wait for "forwarding frames"
just capture-run                          # drive load, fetch trace + reports
just capture-down                         # release the GPU
cargo xtask nightly-golden-entry --trace <trace.jsonl> --archive <trace.jsonl.gz> \
  --bucket-path conformance/<tag>/<gpu>/<model>/<workload>.jsonl.gz --workload <workload>
```

## 6. Cut the release

Only after every line in the window is green **and** the fidelity story is
whatever the release calls for. The sim's semver tracks *its* features; the vLLM
line is build metadata that lives in the image tag, never in the sim version
(`vllm-vcr:0.2.0-vllm0.26`). Do not conflate them.

```bash
# bump [workspace.package] version in Cargo.toml, then:
cargo update --workspace   # refresh the lock's own version entries
cargo test --workspace
git tag v<version> && git push origin v<version>   # release.yml builds the matrix
```
