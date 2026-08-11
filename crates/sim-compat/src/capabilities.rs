//! The per-line capability cfgs, and the build-script glue that emits them.
//!
//! Where `vllm-engine-core-client`'s API diverges across the support window in a
//! way an owned type can't hide, the code gates on a discrete capability rather
//! than a version number. Both build scripts in this workspace (the root crate's
//! and `sim-protocol`'s) call [`emit`] so they can never disagree about which
//! capabilities a given line has.
//!
//! cfgs from a build script only reach the crate that owns it, which is why the
//! list lives here instead of being duplicated per build.rs.

use std::path::Path;

use anyhow::{Context as _, Result};

use crate::{CompatManifest, minor_line};

/// Every capability cfg this workspace can emit, declared to Rust's check-cfg so
/// a typo in a `#[cfg(...)]` warns instead of silently compiling the other arm.
const ALL: &[&str] = &["vllm_cache_creation_tokens", "vllm_engine_id_u16"];

/// Resolve the vLLM tag this build targets.
///
/// A CI matrix build (and `just image-build-line`) sets `VLLM_TARGET_VERSION` to
/// the line it is pinning; otherwise the `default = true` line from
/// `compat.toml` wins. `compat_path` is the workspace-root manifest.
pub fn target_tag(compat_path: &Path) -> Result<String> {
    println!("cargo:rerun-if-changed={}", compat_path.display());
    println!("cargo:rerun-if-env-changed=VLLM_TARGET_VERSION");

    if let Ok(tag) = std::env::var("VLLM_TARGET_VERSION")
        && !tag.is_empty()
    {
        return Ok(tag);
    }

    let manifest = CompatManifest::load(compat_path)
        .with_context(|| format!("loading {}", compat_path.display()))?;
    Ok(manifest
        .default_line()
        .context("compat.toml default line")?
        .tag
        .clone())
}

/// Emit the capability cfgs for `tag` on stdout, in cargo's build-script syntax.
pub fn emit(tag: &str) {
    for name in ALL {
        println!("cargo::rustc-check-cfg=cfg({name})");
    }

    // `vllm_cache_creation_tokens`: `PrefillStats` gained
    // `num_cache_creation_tokens` (prompt tokens this prefill newly admits into
    // the local prefix cache) in 0.26.
    if line_at_least(tag, 0, 26) {
        println!("cargo::rustc-cfg=vllm_cache_creation_tokens");
    }

    // `vllm_engine_id_u16`: `EngineId::from_engine_index` narrowed its parameter
    // from u32 to u16 on vLLM main (the wire encoding was always two-byte
    // little-endian). Gates `sim_protocol::vllm::engine_id_from_index`.
    if line_at_least(tag, 0, 28) {
        println!("cargo::rustc-cfg=vllm_engine_id_u16");
    }
}

/// Whether `tag`'s `major.minor` line is >= `(major, minor)`.
///
/// A tag with no parseable `major.minor` (i.e. `"nightly"`, which tracks vLLM
/// main) is treated as the **newest** line, so every capability turns on: main
/// always carries the latest protocol surface.
fn line_at_least(tag: &str, major: u32, minor: u32) -> bool {
    let Some(line) = minor_line(tag) else {
        return true;
    };
    let mut parts = line.split('.');
    let parsed = (|| {
        let maj: u32 = parts.next()?.parse().ok()?;
        let min: u32 = parts.next()?.parse().ok()?;
        Some((maj, min))
    })();
    match parsed {
        Some((maj, min)) => (maj, min) >= (major, minor),
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use crate::capabilities::line_at_least;

    #[test]
    fn compares_release_tags_by_line() {
        assert!(line_at_least("v0.26.0", 0, 25));
        assert!(line_at_least("v0.25.1", 0, 25));
        assert!(!line_at_least("v0.24.0", 0, 25));
        assert!(!line_at_least("0.24.0.dev1+g16e9117", 0, 25));
    }

    #[test]
    fn rc_tags_resolve_to_their_release_line() {
        assert!(line_at_least("v0.26.1rc0", 0, 25));
        assert!(!line_at_least("v0.24.0rc1", 0, 25));
    }

    #[test]
    fn unparseable_tags_are_treated_as_newest() {
        assert!(line_at_least("nightly", 0, 25));
    }
}
