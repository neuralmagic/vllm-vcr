//! Emit the per-line capability cfgs for this crate.
//!
//! `sim-protocol` owns the outputs shim (`src/outputs.rs`), which is the one
//! place that has to compile differently per vLLM line. Build-script cfgs only
//! reach the crate that owns the script, so this duplicates the root crate's
//! emission; the capability list is shared via `sim_compat::capabilities` so the
//! two can't drift.

use std::path::PathBuf;

fn main() {
    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by cargo");
    // compat.toml lives at the workspace root, two levels up from crates/sim-protocol.
    let compat_path = PathBuf::from(&manifest_dir)
        .join("../..")
        .join("compat.toml");

    let target = sim_compat::capabilities::target_tag(&compat_path)
        .unwrap_or_else(|e| panic!("resolving target vLLM tag: {e}"));

    sim_compat::capabilities::emit(&target);
}
