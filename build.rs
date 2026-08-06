//! Stamp the target vLLM version into the binary at build time, and emit the
//! per-line capability cfgs the engine needs to compile against older lines.
//!
//! The build matrix builds one image per vLLM line; each build must know which
//! line it speaks so the handshake guard can reject mismatched peers and the
//! ready-response can advertise the right `vllm_version`. A CI matrix build
//! sets `VLLM_TARGET_VERSION` directly (the line it is building); otherwise we
//! stamp the `default = true` line from `compat.toml`.
//!
//! The capability list itself lives in `sim_compat::capabilities` so this script
//! and `sim-protocol`'s agree; cfgs from a build script only reach the crate
//! that owns it, so every crate with gated code needs its own.

use std::path::PathBuf;

fn main() {
    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by cargo");
    let compat_path = PathBuf::from(&manifest_dir).join("compat.toml");

    let target = sim_compat::capabilities::target_tag(&compat_path)
        .unwrap_or_else(|e| panic!("resolving target vLLM tag: {e}"));

    println!("cargo:rustc-env=VLLM_TARGET_VERSION={target}");
    sim_compat::capabilities::emit(&target);
}
