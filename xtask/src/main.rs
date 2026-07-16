//! Repo automation, run via `cargo xtask <cmd>`. Reuses `sim-compat` so
//! `compat.toml` is parsed in exactly one place (these used to be inline Python
//! in the workflows, re-parsing the manifest each time).

mod version;

use std::collections::BTreeSet;
use std::io::{BufRead as _, BufReader};
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;

use anyhow::{Context as _, Result, bail};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use clap::{Parser, Subcommand};
use serde::Serialize;
use serde_json::{Value as JsonValue, json};
use sha2::{Digest as _, Sha256};
use sim_compat::{CompatManifest, GoldenManifest};
use toml_edit::{DocumentMut, Item, Table, Value, value};

use crate::version::{VllmVersion, parse_ls_remote, pick_latest_rc};

const COMPAT_TOML: &str = "compat.toml";
const MANIFEST_TOML: &str = "conformance/manifest.toml";
const CARGO_TOML: &str = "Cargo.toml";
const VLLM_GIT: &str = "https://github.com/vllm-project/vllm.git";
const KUBE_OIDC_AUDIENCE_ENCODED: &str = "https%3A%2F%2Fkubernetes.default.svc";
/// The compat.toml line that tracks vLLM main (auto-bumped by the nightly canary).
const NIGHTLY_LINE: &str = "nightly";
/// The compat.toml line that tracks the newest upstream release candidate.
const RC_LINE: &str = "rc";
/// Non-stable "tracker" lines that ride ahead of the stable window and are never
/// `default`. The stable roll counts and trims everything *except* these.
const TRACKER_LINES: &[&str] = &[NIGHTLY_LINE, RC_LINE];

/// Native per-arch release runners (no cross-compile).
const PLATFORMS: &[Platform] = &[
    Platform {
        runner: "ubuntu-latest",
        target: "x86_64-unknown-linux-musl",
        os: "linux",
    },
    Platform {
        runner: "ubuntu-24.04-arm",
        target: "aarch64-unknown-linux-musl",
        os: "linux",
    },
    Platform {
        runner: "macos-14",
        target: "aarch64-apple-darwin",
        os: "macos",
    },
];

#[derive(Parser)]
#[command(name = "xtask", about = "vllm-vcr repo automation")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Emit the CI build matrix (one row per compat.toml line) as compact JSON.
    CiMatrix,
    /// Emit the Docker image build matrix as compact JSON.
    DockerMatrix,
    /// Emit the release line x platform matrix as compact JSON.
    ReleaseMatrix,
    /// Print the sim semver from `[workspace.package].version`.
    Simver,
    /// Print `TAG=…; FREPO=…; FREF=…` shell assignments for a line's vllm-rs
    /// frontend build args (`eval` it). FREPO/FREF point at the fork when the
    /// line has one, else upstream at protocol_rev.
    FrontendArgs {
        /// compat.toml line, e.g. "0.23" or "nightly".
        line: String,
    },
    /// Pin `Cargo.toml`'s vllm-engine-core-client to a compat.toml line's rev
    /// (and fork `[patch]`), or to an explicit `--rev` override.
    PinVllm {
        /// compat.toml line, e.g. "0.23" or "nightly".
        line: String,
        /// Override the line's `protocol_rev` with this sha (the nightly canary
        /// pins to the live upstream HEAD).
        #[arg(long)]
        rev: Option<String>,
    },
    /// Bump the `nightly` line's `protocol_rev` in `compat.toml` to a vLLM main
    /// sha, preserving formatting and comments. This is the persistent
    /// source-of-truth bump (vs `pin-vllm`'s transient Cargo.toml edit); the
    /// nightly canary runs it after a green build so the auto-bump PR only ever
    /// proposes a sha that already builds + passes unit tests.
    SetNightlyRev {
        /// The vllm.git main commit sha to pin the `nightly` line to.
        sha: String,
    },
    /// Refresh the `rc` line to the newest upstream release candidate newer than
    /// the latest stable release. Reads a `git ls-remote --tags` dump (the
    /// workflow fetches it) and bumps both the `rc` line's `tag` and
    /// `protocol_rev`. Prints `key=value` lines (changed/tag/line/rev) for the
    /// workflow to gate the build + PR on. A no-op (no newer rc) prints
    /// `changed=false` and leaves compat.toml untouched.
    WatchRc {
        /// File holding `git ls-remote --tags https://github.com/vllm-project/vllm.git`.
        #[arg(long)]
        tags_file: PathBuf,
        /// The latest stable release tag (e.g. `v0.23.0`), from `releases/latest`.
        #[arg(long)]
        latest_stable: String,
    },
    /// Roll the stable window to a newly released vLLM line: add it as the new
    /// `default`, demote the old default, and drop the oldest stable line once
    /// the window exceeds `--max-stable`. A release on a line already in the
    /// window is a patch bump instead: that line's tag/protocol_rev move in
    /// place and its `fidelity_validated` resets to false. No-op (prints
    /// `changed=false`) when the release is already tracked or isn't newer.
    /// The `nightly`/`rc` trackers are never touched. Prints the same
    /// `key=value` summary as `watch-rc`.
    WatchStable {
        /// File holding `git ls-remote --tags https://github.com/vllm-project/vllm.git`.
        #[arg(long)]
        tags_file: PathBuf,
        /// The latest stable release tag (e.g. `v0.24.0`), from `releases/latest`.
        #[arg(long)]
        latest_stable: String,
        /// Maximum stable lines to keep (oldest dropped beyond this).
        #[arg(long, default_value_t = 4)]
        max_stable: usize,
    },
    /// Emit one `[[golden]]` TOML entry for a captured nightly trace.
    NightlyGoldenEntry {
        /// Uncompressed trace JSONL path.
        #[arg(long)]
        trace: PathBuf,
        /// Uploaded gzip/archive path whose sha256 should be pinned.
        #[arg(long)]
        archive: PathBuf,
        /// S3 key under CONFORMANCE_BUCKET.
        #[arg(long)]
        bucket_path: String,
        /// Human-readable workload label.
        #[arg(long)]
        workload: String,
    },
    /// Replace the generated nightly-goldens block in conformance/manifest.toml.
    SetNightlyGoldens {
        /// TOML file containing generated `[[golden]]` entries.
        #[arg(long)]
        entries_file: PathBuf,
        /// Manifest to update.
        #[arg(long, default_value = MANIFEST_TOML)]
        manifest: PathBuf,
    },
    /// Write a kubeconfig that authenticates through GitHub Actions OIDC.
    GithubOidcKubeconfig {
        /// Kubernetes API server URL.
        #[arg(long)]
        cluster_url: String,
        /// Path to the built xtask binary used as the kubectl exec plugin.
        #[arg(long)]
        plugin_path: PathBuf,
        /// Kubeconfig path to write.
        #[arg(long)]
        kubeconfig: PathBuf,
    },
    /// Print a Kubernetes ExecCredential using a fresh GitHub Actions OIDC token.
    GithubOidcExecCredential,
}

#[derive(Serialize, Clone, Copy)]
struct Platform {
    runner: &'static str,
    target: &'static str,
    os: &'static str,
}

#[derive(Serialize)]
struct CiRow {
    line: String,
    tag: String,
    protocol_rev: String,
    fidelity_validated: bool,
    default: bool,
    has_goldens: bool,
}

#[derive(Serialize)]
struct DockerRow {
    line: String,
    tag: String,
    protocol_rev: String,
    patch_repo: String,
    patch_rev: String,
    fidelity_validated: bool,
    default: bool,
}

#[derive(Serialize)]
struct ReleaseRow {
    line: String,
    tag: String,
    protocol_rev: String,
    default: bool,
    runner: &'static str,
    target: &'static str,
    os: &'static str,
}

#[derive(Serialize)]
struct ExecCredential<'a> {
    #[serde(rename = "apiVersion")]
    api_version: &'static str,
    kind: &'static str,
    status: ExecCredentialStatus<'a>,
}

#[derive(Serialize)]
struct ExecCredentialStatus<'a> {
    token: &'a str,
    #[serde(
        rename = "expirationTimestamp",
        skip_serializing_if = "Option::is_none"
    )]
    expiration_timestamp: Option<String>,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::CiMatrix => ci_matrix(),
        Command::DockerMatrix => docker_matrix(),
        Command::ReleaseMatrix => release_matrix(),
        Command::Simver => simver(),
        Command::FrontendArgs { line } => frontend_args(&line),
        Command::PinVllm { line, rev } => pin_vllm(&line, rev.as_deref()),
        Command::SetNightlyRev { sha } => set_nightly_rev(&sha),
        Command::WatchRc {
            tags_file,
            latest_stable,
        } => watch_rc(&tags_file, &latest_stable),
        Command::WatchStable {
            tags_file,
            latest_stable,
            max_stable,
        } => watch_stable(&tags_file, &latest_stable, max_stable),
        Command::NightlyGoldenEntry {
            trace,
            archive,
            bucket_path,
            workload,
        } => nightly_golden_entry(&trace, &archive, &bucket_path, &workload),
        Command::SetNightlyGoldens {
            entries_file,
            manifest,
        } => set_nightly_goldens(&entries_file, &manifest),
        Command::GithubOidcKubeconfig {
            cluster_url,
            plugin_path,
            kubeconfig,
        } => github_oidc_kubeconfig(&cluster_url, &plugin_path, &kubeconfig),
        Command::GithubOidcExecCredential => github_oidc_exec_credential(),
    }
}

fn ci_matrix() -> Result<()> {
    let compat = CompatManifest::load(COMPAT_TOML)?;
    let golden_lines = golden_lines()?;
    let rows: Vec<CiRow> = compat
        .lines
        .iter()
        .map(|v| CiRow {
            line: v.line.clone(),
            tag: v.tag.clone(),
            protocol_rev: v.protocol_rev.clone(),
            fidelity_validated: v.fidelity_validated,
            default: v.default,
            has_goldens: golden_lines.contains(v.line.as_str()),
        })
        .collect();
    print_json(&rows)
}

fn docker_matrix() -> Result<()> {
    let compat = CompatManifest::load(COMPAT_TOML)?;
    let rows: Vec<DockerRow> = compat
        .lines
        .iter()
        .map(|v| DockerRow {
            line: v.line.clone(),
            tag: v.tag.clone(),
            protocol_rev: v.protocol_rev.clone(),
            patch_repo: v.patch_repo.clone().unwrap_or_default(),
            patch_rev: v.patch_rev.clone().unwrap_or_default(),
            fidelity_validated: v.fidelity_validated,
            default: v.default,
        })
        .collect();
    print_json(&rows)
}

fn release_matrix() -> Result<()> {
    let compat = CompatManifest::load(COMPAT_TOML)?;
    let rows: Vec<ReleaseRow> = compat
        .lines
        .iter()
        .flat_map(|v| {
            PLATFORMS.iter().map(move |p| ReleaseRow {
                line: v.line.clone(),
                tag: v.tag.clone(),
                protocol_rev: v.protocol_rev.clone(),
                default: v.default,
                runner: p.runner,
                target: p.target,
                os: p.os,
            })
        })
        .collect();
    print_json(&rows)
}

fn simver() -> Result<()> {
    let doc = read_cargo_toml()?;
    let version = doc["workspace"]["package"]["version"]
        .as_str()
        .context("[workspace.package].version is not a string")?;
    println!("{version}");
    Ok(())
}

fn frontend_args(line: &str) -> Result<()> {
    let compat = CompatManifest::load(COMPAT_TOML)?;
    let entry = compat
        .lines
        .iter()
        .find(|v| v.line == line)
        .with_context(|| format!("no compat.toml entry for line {line}"))?;
    let repo = entry.patch_repo.as_deref().unwrap_or(VLLM_GIT);
    let reference = entry.patch_rev.as_deref().unwrap_or(&entry.protocol_rev);
    println!("TAG={}; FREPO={repo}; FREF={reference}", entry.tag);
    Ok(())
}

fn pin_vllm(line: &str, rev_override: Option<&str>) -> Result<()> {
    let compat = CompatManifest::load(COMPAT_TOML)?;
    let entry = compat
        .lines
        .iter()
        .find(|v| v.line == line)
        .with_context(|| format!("no compat.toml entry for line {line}"))?;
    let rev = rev_override.unwrap_or(&entry.protocol_rev);

    let mut doc = read_cargo_toml()?;
    apply_pin(
        &mut doc,
        rev,
        entry.patch_repo.as_deref(),
        entry.patch_rev.as_deref(),
    )?;

    std::fs::write(CARGO_TOML, doc.to_string()).context("writing Cargo.toml")?;
    eprintln!(
        "pinned vllm-engine-core-client: line={line} rev={rev} patch={}",
        entry.patch_rev.as_deref().unwrap_or("none")
    );
    Ok(())
}

/// Set the base dependency rev and the fork `[patch]` block in `doc`. Stripping
/// any existing patch first keeps re-runs idempotent.
fn apply_pin(
    doc: &mut DocumentMut,
    rev: &str,
    patch_repo: Option<&str>,
    patch_rev: Option<&str>,
) -> Result<()> {
    // Base dependency rev in [workspace.dependencies] (never the fork [patch]).
    let dep = doc["workspace"]["dependencies"]["vllm-engine-core-client"]
        .as_inline_table_mut()
        .context("workspace.dependencies.vllm-engine-core-client must be an inline table")?;
    dep.insert("rev", rev.into());

    if let Some(patch) = doc.get_mut("patch").and_then(Item::as_table_mut) {
        if let Some(src) = patch.get_mut(VLLM_GIT).and_then(Item::as_table_mut) {
            src.remove("vllm-engine-core-client");
            if src.is_empty() {
                patch.remove(VLLM_GIT);
            }
        }
        if patch.is_empty() {
            doc.remove("patch");
        }
    }

    if let Some(patch_rev) = patch_rev {
        let patch_repo = patch_repo.context("patch_rev without patch_repo in compat.toml")?;
        let mut fork = toml_edit::InlineTable::new();
        fork.insert("git", patch_repo.into());
        fork.insert("rev", patch_rev.into());

        let patch = doc
            .entry("patch")
            .or_insert(Item::Table(Table::new()))
            .as_table_mut()
            .context("[patch] is not a table")?;
        patch.set_implicit(true); // emit only [patch."url"], not a bare [patch]
        let src = patch
            .entry(VLLM_GIT)
            .or_insert(Item::Table(Table::new()))
            .as_table_mut()
            .context("[patch.<vllm>] is not a table")?;
        src.insert("vllm-engine-core-client", Item::Value(Value::from(fork)));
    }
    Ok(())
}

fn set_nightly_rev(sha: &str) -> Result<()> {
    let mut doc = read_compat_toml()?;
    if apply_set_line(&mut doc, NIGHTLY_LINE, None, Some(sha))? {
        std::fs::write(COMPAT_TOML, doc.to_string()).context("writing compat.toml")?;
        eprintln!("bumped nightly protocol_rev -> {sha}");
    } else {
        eprintln!("nightly protocol_rev already {sha}; no change");
    }
    Ok(())
}

/// Refresh the `rc` line to the newest upstream rc tag newer than `latest_stable`.
fn watch_rc(tags_file: &Path, latest_stable: &str) -> Result<()> {
    let stable = VllmVersion::parse(latest_stable).with_context(|| {
        format!(
            "--latest-stable '{latest_stable}' is not a vX.Y.Z release tag (expected e.g. v0.24.0)"
        )
    })?;
    let tags = parse_ls_remote(&read_tags_file(tags_file)?);

    let Some((tag, rev)) = pick_latest_rc(&tags, stable) else {
        println!("changed=false");
        eprintln!("no upstream rc newer than {latest_stable}; rc line left untouched");
        return Ok(());
    };

    let mut doc = read_compat_toml()?;
    let changed = apply_set_line(&mut doc, RC_LINE, Some(&tag), Some(&rev))?;
    if changed {
        std::fs::write(COMPAT_TOML, doc.to_string()).context("writing compat.toml")?;
        eprintln!("bumped rc line -> {tag} ({rev})");
    } else {
        eprintln!("rc line already {tag}; no change");
    }
    let outcome = if changed {
        WatchOutcome::ChangedTransient
    } else {
        WatchOutcome::Unchanged
    };
    print_watch_summary(outcome, &tag, RC_LINE, &rev);
    Ok(())
}

/// Roll the stable window to `latest_stable` if it is a new line newer than the
/// current default.
fn watch_stable(tags_file: &Path, latest_stable: &str, max_stable: usize) -> Result<()> {
    let new = VllmVersion::parse(latest_stable).with_context(|| {
        format!(
            "--latest-stable '{latest_stable}' is not a vX.Y.Z release tag (expected e.g. v0.24.0)"
        )
    })?;
    if new.rc.is_some() {
        bail!("--latest-stable {latest_stable} is a pre-release; expected a final release");
    }
    let tags = parse_ls_remote(&read_tags_file(tags_file)?);
    let rev = tags
        .get(latest_stable)
        .with_context(|| format!("tag {latest_stable} not found in {}", tags_file.display()))?
        .clone();

    let manifest = CompatManifest::load(COMPAT_TOML)?;
    let minor = new.minor_line();
    // A release on a line already in the window is a patch bump (v0.23.0 ->
    // v0.23.1): move that line's tag/rev in place. The new rev invalidates any
    // fidelity captures taken against the old one, so the line is demoted back
    // to fidelity_validated = false until conformance re-passes.
    if let Some(existing) = manifest.line(&minor) {
        let current = VllmVersion::parse(&existing.tag)
            .with_context(|| format!("line {minor} tag {} is unparseable", existing.tag))?;
        if new <= current {
            println!("changed=false");
            eprintln!(
                "line {minor} already at {}; nothing to update",
                existing.tag
            );
            return Ok(());
        }
        let mut doc = read_compat_toml()?;
        apply_set_line(&mut doc, &minor, Some(latest_stable), Some(&rev))?;
        apply_demote_fidelity(&mut doc, &minor)?;
        std::fs::write(COMPAT_TOML, doc.to_string()).context("writing compat.toml")?;
        eprintln!(
            "patch-bumped line {minor}: {} -> {latest_stable} ({rev})",
            existing.tag
        );
        let outcome = if manifest.default_line()?.line == minor {
            WatchOutcome::ChangedDefault
        } else {
            WatchOutcome::ChangedTransient
        };
        print_watch_summary(outcome, latest_stable, &minor, &rev);
        return Ok(());
    }
    let default = manifest.default_line()?;
    let default_v = VllmVersion::parse(&default.tag)
        .with_context(|| format!("current default tag {} is unparseable", default.tag))?;
    if new <= default_v {
        println!("changed=false");
        eprintln!(
            "{latest_stable} is not newer than the current default {}",
            default.tag
        );
        return Ok(());
    }

    let mut doc = read_compat_toml()?;
    apply_roll_stable(&mut doc, &minor, latest_stable, &rev, max_stable)?;
    std::fs::write(COMPAT_TOML, doc.to_string()).context("writing compat.toml")?;
    eprintln!("rolled stable window: new default {latest_stable} (line {minor})");
    print_watch_summary(WatchOutcome::ChangedDefault, latest_stable, &minor, &rev);
    Ok(())
}

fn read_tags_file(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))
}

/// What a watch run did to compat.toml, and whether the committed Cargo.toml
/// pin (which tracks the `default = true` stable line) has to follow.
#[derive(Clone, Copy)]
enum WatchOutcome {
    /// compat.toml untouched; the target was already pinned.
    Unchanged,
    /// A tracker or non-default stable line moved; the CI build pin is transient.
    ChangedTransient,
    /// The default stable line moved (roll or patch bump); the committed
    /// Cargo.toml/Cargo.lock pin belongs in the auto-bump PR.
    ChangedDefault,
}

/// Emit the `key=value` lines the workflow appends to `$GITHUB_OUTPUT`.
fn print_watch_summary(outcome: WatchOutcome, tag: &str, line: &str, rev: &str) {
    let (changed, is_default) = match outcome {
        WatchOutcome::Unchanged => (false, false),
        WatchOutcome::ChangedTransient => (true, false),
        WatchOutcome::ChangedDefault => (true, true),
    };
    println!("changed={changed}");
    println!("tag={tag}");
    println!("line={line}");
    println!("rev={rev}");
    println!("is_default={is_default}");
}

fn nightly_golden_entry(
    trace: &Path,
    archive: &Path,
    bucket_path: &str,
    workload: &str,
) -> Result<()> {
    let meta = read_trace_meta(trace)?;
    let config_hash = meta
        .get("config_hash")
        .and_then(JsonValue::as_str)
        .context("trace meta missing config_hash")?;
    let sha256 = sha256_hex(archive)?;

    println!();
    println!("[[golden]]");
    println!("line = \"nightly\"");
    println!("bucket_path = {}", serde_json::to_string(bucket_path)?);
    println!("sha256 = \"{sha256}\"");
    println!("config_hash = \"{config_hash}\"");
    println!("workload = {}", serde_json::to_string(workload)?);
    println!("role = \"fidelity\"");
    Ok(())
}

fn set_nightly_goldens(entries_file: &Path, manifest: &Path) -> Result<()> {
    let entries = std::fs::read_to_string(entries_file)
        .with_context(|| format!("reading {}", entries_file.display()))?;
    let mut text = std::fs::read_to_string(manifest)
        .with_context(|| format!("reading {}", manifest.display()))?;
    text = replace_nightly_goldens_block(&text, &entries);
    std::fs::write(manifest, text).with_context(|| format!("writing {}", manifest.display()))?;
    Ok(())
}

fn github_oidc_kubeconfig(cluster_url: &str, plugin_path: &Path, kubeconfig: &Path) -> Result<()> {
    if cluster_url.trim().is_empty() {
        bail!("cluster URL is required");
    }
    if plugin_path.as_os_str().is_empty() {
        bail!("plugin path is required");
    }

    if let Some(parent) = kubeconfig.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    std::fs::write(kubeconfig, kubeconfig_json(cluster_url, plugin_path)?)
        .with_context(|| format!("writing {}", kubeconfig.display()))?;
    chmod_600(kubeconfig)?;
    eprintln!("wrote GitHub OIDC kubeconfig to {}", kubeconfig.display());
    Ok(())
}

fn github_oidc_exec_credential() -> Result<()> {
    let token = fetch_github_oidc_token()?;
    let expiration_timestamp = match jwt_expiration_timestamp(&token) {
        Ok(ts) => Some(ts),
        Err(err) => {
            eprintln!("warning: could not decode OIDC token expiration: {err:#}");
            None
        }
    };
    let credential = ExecCredential {
        api_version: "client.authentication.k8s.io/v1beta1",
        kind: "ExecCredential",
        status: ExecCredentialStatus {
            token: &token,
            expiration_timestamp,
        },
    };
    println!("{}", serde_json::to_string(&credential)?);
    Ok(())
}

fn kubeconfig_json(cluster_url: &str, plugin_path: &Path) -> Result<String> {
    let config = json!({
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": [{
            "name": "conformance",
            "cluster": {
                "server": cluster_url,
            },
        }],
        "contexts": [{
            "name": "conformance",
            "context": {
                "cluster": "conformance",
                "user": "github-oidc",
            },
        }],
        "current-context": "conformance",
        "users": [{
            "name": "github-oidc",
            "user": {
                "exec": {
                    "apiVersion": "client.authentication.k8s.io/v1beta1",
                    "command": plugin_path.display().to_string(),
                    "args": ["github-oidc-exec-credential"],
                    "interactiveMode": "Never",
                },
            },
        }],
    });
    serde_json::to_string_pretty(&config).context("serializing kubeconfig")
}

fn fetch_github_oidc_token() -> Result<String> {
    let request_url = std::env::var("ACTIONS_ID_TOKEN_REQUEST_URL")
        .context("ACTIONS_ID_TOKEN_REQUEST_URL is not set; is id-token: write enabled?")?;
    let request_token = std::env::var("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
        .context("ACTIONS_ID_TOKEN_REQUEST_TOKEN is not set; is id-token: write enabled?")?;
    let url = format!("{request_url}&audience={KUBE_OIDC_AUDIENCE_ENCODED}");
    let output = ProcessCommand::new("curl")
        .args(["-sS", "-f", "-H"])
        .arg(format!("Authorization: bearer {request_token}"))
        .arg(&url)
        .output()
        .context("requesting GitHub OIDC token with curl")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("GitHub OIDC token request failed: {}", stderr.trim());
    }
    let response: JsonValue =
        serde_json::from_slice(&output.stdout).context("parsing GitHub OIDC response JSON")?;
    response
        .get("value")
        .and_then(JsonValue::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .context("GitHub OIDC response did not include a token value")
}

fn jwt_expiration_timestamp(token: &str) -> Result<String> {
    let payload = token
        .split('.')
        .nth(1)
        .context("JWT is missing a payload segment")?;
    let payload = URL_SAFE_NO_PAD
        .decode(payload)
        .context("decoding JWT payload")?;
    let claims: JsonValue = serde_json::from_slice(&payload).context("parsing JWT payload JSON")?;
    let exp = claims
        .get("exp")
        .and_then(JsonValue::as_i64)
        .context("JWT payload is missing numeric exp")?;
    let timestamp = chrono::DateTime::<chrono::Utc>::from_timestamp(exp, 0)
        .context("JWT exp is out of range")?;
    Ok(timestamp.format("%Y-%m-%dT%H:%M:%SZ").to_string())
}

#[cfg(unix)]
fn chmod_600(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt as _;

    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
        .with_context(|| format!("setting permissions on {}", path.display()))
}

#[cfg(not(unix))]
fn chmod_600(_path: &Path) -> Result<()> {
    Ok(())
}

fn read_trace_meta(trace: &Path) -> Result<JsonValue> {
    let file =
        std::fs::File::open(trace).with_context(|| format!("opening {}", trace.display()))?;
    let mut lines = BufReader::new(file).lines();
    let first = lines
        .next()
        .transpose()
        .context("reading trace meta line")?
        .context("trace is empty")?;
    let value: JsonValue = serde_json::from_str(&first).context("parsing trace meta JSON")?;
    value
        .get("meta")
        .cloned()
        .context("first trace line must be a {\"meta\": ...} object")
}

fn sha256_hex(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

fn replace_nightly_goldens_block(manifest: &str, entries: &str) -> String {
    const START: &str = "# BEGIN NIGHTLY GOLDENS";
    const END: &str = "# END NIGHTLY GOLDENS";
    let entries = entries.trim();
    let block = format!("{START}\n{entries}\n{END}");

    if let Some(start_idx) = manifest.find(START) {
        if let Some(end_rel) = manifest[start_idx..].find(END) {
            let end_idx = start_idx + end_rel + END.len();
            let before = manifest[..start_idx].trim_end();
            let after = manifest[end_idx..].trim_start_matches('\n');
            return format!("{before}\n\n{block}\n{after}");
        }
    }

    format!("{}\n\n{block}\n", manifest.trim_end())
}

/// Set a `[[vllm]]` line's `tag` and/or `protocol_rev` in place, preserving the
/// surrounding formatting and comments. `None` leaves that field alone. Returns
/// `true` if anything changed, so the caller (and the watcher's `git diff` check)
/// can skip an empty write / no-op PR.
fn apply_set_line(
    doc: &mut DocumentMut,
    line: &str,
    tag: Option<&str>,
    rev: Option<&str>,
) -> Result<bool> {
    let tables = doc["vllm"]
        .as_array_of_tables_mut()
        .context("compat.toml [[vllm]] is not an array of tables")?;
    let entry = tables
        .iter_mut()
        .find(|t| t.get("line").and_then(Item::as_str) == Some(line))
        .with_context(|| format!("no [[vllm]] entry with line = \"{line}\" in compat.toml"))?;

    let mut changed = false;
    if let Some(tag) = tag {
        changed |= set_str_field(entry, "tag", tag)?;
    }
    if let Some(rev) = rev {
        changed |= set_str_field(entry, "protocol_rev", rev)?;
    }
    Ok(changed)
}

/// Reset a `[[vllm]]` line's `fidelity_validated` to `false`. Used when the
/// line's tag/rev moves: existing fidelity captures were taken against the old
/// rev, so the promotion has to be re-earned through the conformance gates.
fn apply_demote_fidelity(doc: &mut DocumentMut, line: &str) -> Result<()> {
    let tables = doc["vllm"]
        .as_array_of_tables_mut()
        .context("compat.toml [[vllm]] is not an array of tables")?;
    let entry = tables
        .iter_mut()
        .find(|t| t.get("line").and_then(Item::as_str) == Some(line))
        .with_context(|| format!("no [[vllm]] entry with line = \"{line}\" in compat.toml"))?;
    entry["fidelity_validated"] = value(false);
    Ok(())
}

/// Set an existing string field on a `[[vllm]]` table, returning whether it moved.
fn set_str_field(entry: &mut Table, field: &str, val: &str) -> Result<bool> {
    let item = entry
        .get_mut(field)
        .with_context(|| format!("line has no {field}"))?
        .as_value_mut()
        .with_context(|| format!("{field} is not a value"))?;
    if item.as_str() == Some(val) {
        return Ok(false);
    }
    *item = val.into();
    Ok(true)
}

/// Insert `minor` (`tag`/`rev`) as the new `default = true` stable line, demote the
/// old default, and drop the oldest stable line(s) past `max_stable`. The
/// `nightly`/`rc` trackers are never counted or removed. Caller must have already
/// verified `minor` is absent and newer than the current default.
fn apply_roll_stable(
    doc: &mut DocumentMut,
    minor: &str,
    tag: &str,
    rev: &str,
    max_stable: usize,
) -> Result<()> {
    let tables = doc["vllm"]
        .as_array_of_tables_mut()
        .context("compat.toml [[vllm]] is not an array of tables")?;

    // Demote whatever currently carries default = true.
    for table in tables.iter_mut() {
        if table.get("default").and_then(Item::as_bool) == Some(true) {
            table.remove("default");
        }
    }

    // Build the new default line and insert it ahead of the existing stable lines
    // (i.e. right after the nightly/rc trackers), so the file stays newest-first.
    let mut entry = Table::new();
    entry.decor_mut().set_prefix(
        "\n# Auto-rolled in by vllm-release-watch as the new default. Review the\n\
         # protocol_rev (pinned to the release tag's commit, not a post-release main\n\
         # sha) and whether this line needs a [patch] fork before promoting fidelity.\n",
    );
    entry["line"] = value(minor);
    entry["tag"] = value(tag);
    entry["protocol_rev"] = value(rev);
    entry["fidelity_validated"] = value(false);
    entry["default"] = value(true);

    let insert_at = first_stable_index(tables);
    tables.insert(insert_at, entry);

    // Trim the oldest stable lines (highest index) until we're within the window.
    while stable_indices(tables).len() > max_stable {
        if let Some(&oldest) = stable_indices(tables).last() {
            tables.remove(oldest);
        }
    }
    Ok(())
}

fn is_tracker(table: &Table) -> bool {
    table
        .get("line")
        .and_then(Item::as_str)
        .is_some_and(|l| TRACKER_LINES.contains(&l))
}

/// Index of the first non-tracker (stable) line, or the end if there are none.
fn first_stable_index(tables: &toml_edit::ArrayOfTables) -> usize {
    tables
        .iter()
        .position(|t| !is_tracker(t))
        .unwrap_or(tables.len())
}

/// Indices of the stable lines (everything that is not a nightly/rc tracker).
fn stable_indices(tables: &toml_edit::ArrayOfTables) -> Vec<usize> {
    tables
        .iter()
        .enumerate()
        .filter(|(_, t)| !is_tracker(t))
        .map(|(i, _)| i)
        .collect()
}

fn read_compat_toml() -> Result<DocumentMut> {
    std::fs::read_to_string(COMPAT_TOML)
        .context("reading compat.toml")?
        .parse::<DocumentMut>()
        .context("parsing compat.toml")
}

/// Lines with at least one golden registered (drives the conformance fetch leg).
fn golden_lines() -> Result<BTreeSet<String>> {
    if !Path::new(MANIFEST_TOML).exists() {
        return Ok(BTreeSet::new());
    }
    let manifest = GoldenManifest::load(MANIFEST_TOML)?;
    Ok(manifest.goldens.into_iter().map(|g| g.line).collect())
}

fn read_cargo_toml() -> Result<DocumentMut> {
    std::fs::read_to_string(CARGO_TOML)
        .context("reading Cargo.toml")?
        .parse::<DocumentMut>()
        .context("parsing Cargo.toml")
}

fn print_json<T: Serialize>(rows: &[T]) -> Result<()> {
    println!("{}", serde_json::to_string(rows)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: &str = "\
[workspace.dependencies]
vllm-engine-core-client = { git = \"https://github.com/vllm-project/vllm.git\", rev = \"oldrev\" }
anyhow = \"1\"
";

    fn pin(toml: &str, rev: &str, repo: Option<&str>, patch_rev: Option<&str>) -> String {
        let mut doc: DocumentMut = toml.parse().unwrap();
        apply_pin(&mut doc, rev, repo, patch_rev).unwrap();
        doc.to_string()
    }

    #[test]
    fn pins_rev_with_no_fork() {
        let out = pin(BASE, "newrev", None, None);
        assert!(out.contains(r#"rev = "newrev""#));
        assert!(!out.contains("oldrev"));
        assert!(
            !out.contains("[patch"),
            "no fork means no patch block: {out}"
        );
    }

    #[test]
    fn adds_fork_patch_block() {
        let out = pin(
            BASE,
            "newrev",
            Some("https://github.com/wseaton/vllm.git"),
            Some("forkrev"),
        );
        assert!(out.contains(r#"[patch."https://github.com/vllm-project/vllm.git"]"#));
        assert!(out.contains(r#"git = "https://github.com/wseaton/vllm.git""#));
        assert!(out.contains(r#"rev = "forkrev""#));
        // No bare [patch] header, only the quoted-source table.
        assert!(!out.contains("\n[patch]\n"));
    }

    #[test]
    fn re_pinning_to_no_fork_strips_the_patch() {
        let forked = pin(
            BASE,
            "r1",
            Some("https://github.com/wseaton/vllm.git"),
            Some("fork1"),
        );
        let stripped = pin(&forked, "r2", None, None);
        assert!(!stripped.contains("[patch"), "stale patch left: {stripped}");
        assert!(stripped.contains(r#"rev = "r2""#));
    }

    #[test]
    fn is_idempotent() {
        let once = pin(
            BASE,
            "r",
            Some("https://github.com/wseaton/vllm.git"),
            Some("f"),
        );
        let twice = pin(
            &once,
            "r",
            Some("https://github.com/wseaton/vllm.git"),
            Some("f"),
        );
        assert_eq!(once, twice);
    }

    // A trimmed compat.toml with the nightly line first, an inline comment on the
    // nightly entry, and a second line whose rev must stay untouched.
    const COMPAT: &str = "\
# manifest header
[[vllm]]
# nightly tracks vLLM main
line = \"nightly\"
tag = \"nightly\"
protocol_rev = \"oldsha\"
fidelity_validated = false

[[vllm]]
line = \"0.23\"
tag = \"v0.23.0\"
protocol_rev = \"keepsha\"
default = true
";

    fn set_nightly(toml: &str, sha: &str) -> (String, bool) {
        let mut doc: DocumentMut = toml.parse().unwrap();
        let changed = apply_set_line(&mut doc, "nightly", None, Some(sha)).unwrap();
        (doc.to_string(), changed)
    }

    #[test]
    fn set_nightly_rev_bumps_only_the_nightly_line() {
        let (out, changed) = set_nightly(COMPAT, "newsha");
        assert!(changed);
        assert!(out.contains(r#"protocol_rev = "newsha""#));
        assert!(!out.contains("oldsha"));
        assert!(
            out.contains(r#"protocol_rev = "keepsha""#),
            "the 0.23 line's rev must be untouched: {out}"
        );
    }

    #[test]
    fn set_nightly_rev_preserves_comments_and_structure() {
        let (out, _) = set_nightly(COMPAT, "newsha");
        assert!(out.contains("# manifest header"));
        assert!(out.contains("# nightly tracks vLLM main"));
        assert!(out.contains("fidelity_validated = false"));
        assert!(out.contains(r#"default = true"#));
    }

    #[test]
    fn set_nightly_rev_is_idempotent_no_change() {
        let (_, changed) = set_nightly(COMPAT, "oldsha");
        assert!(!changed, "setting the same sha must report no change");
    }

    #[test]
    fn set_line_bumps_tag_and_rev_together() {
        let mut doc: DocumentMut = COMPAT.parse().unwrap();
        let changed =
            apply_set_line(&mut doc, "nightly", Some("v0.24.0rc1"), Some("newsha")).unwrap();
        let out = doc.to_string();
        assert!(changed);
        assert!(out.contains(r#"tag = "v0.24.0rc1""#));
        assert!(out.contains(r#"protocol_rev = "newsha""#));
        // The other line stays put.
        assert!(out.contains(r#"protocol_rev = "keepsha""#));
    }

    // A window with both trackers (nightly, rc) and three stable lines, newest first.
    const WINDOW: &str = "\
[[vllm]]
line = \"nightly\"
tag = \"nightly\"
protocol_rev = \"mainsha\"
fidelity_validated = false

[[vllm]]
line = \"rc\"
tag = \"v0.24.0rc1\"
protocol_rev = \"rcsha\"
fidelity_validated = false

[[vllm]]
line = \"0.23\"
tag = \"v0.23.0\"
protocol_rev = \"sha23\"
default = true

[[vllm]]
line = \"0.22\"
tag = \"v0.22.1\"
protocol_rev = \"sha22\"

[[vllm]]
line = \"0.21\"
tag = \"v0.21.0\"
protocol_rev = \"sha21\"
";

    fn roll(toml: &str, minor: &str, tag: &str, rev: &str, max: usize) -> CompatManifest {
        let mut doc: DocumentMut = toml.parse().unwrap();
        apply_roll_stable(&mut doc, minor, tag, rev, max).unwrap();
        CompatManifest::parse(&doc.to_string()).unwrap()
    }

    #[test]
    fn roll_stable_makes_the_new_line_the_only_default() {
        let m = roll(WINDOW, "0.24", "v0.24.0", "sha24", 4);
        let default = m.default_line().unwrap();
        assert_eq!(default.line, "0.24");
        assert_eq!(default.tag, "v0.24.0");
        assert_eq!(default.protocol_rev, "sha24");
        assert!(!default.fidelity_validated);
        assert_eq!(m.lines.iter().filter(|l| l.default).count(), 1);
    }

    #[test]
    fn roll_stable_inserts_after_trackers_keeping_newest_first() {
        let m = roll(WINDOW, "0.24", "v0.24.0", "sha24", 4);
        let order: Vec<&str> = m.lines.iter().map(|l| l.line.as_str()).collect();
        // Trackers stay on top; new default leads the stable lines.
        assert_eq!(order, ["nightly", "rc", "0.24", "0.23", "0.22", "0.21"]);
    }

    #[test]
    fn roll_stable_keeps_four_then_drops_the_oldest() {
        // Window starts at 3 stable; rolling to 0.24 fills it to 4, none dropped.
        let four = roll(WINDOW, "0.24", "v0.24.0", "sha24", 4);
        assert!(four.line("0.21").is_some(), "0.21 survives at the 4th slot");

        // Roll again to 0.25 on the already-full window: 0.21 (oldest) ages out.
        let five = roll(&toml_for(&four), "0.25", "v0.25.0", "sha25", 4);
        let stable: Vec<&str> = five
            .lines
            .iter()
            .filter(|l| l.line != "nightly" && l.line != "rc")
            .map(|l| l.line.as_str())
            .collect();
        assert_eq!(stable, ["0.25", "0.24", "0.23", "0.22"]);
        assert!(five.line("0.21").is_none(), "oldest stable line is dropped");
        // Trackers are never touched by a roll.
        assert!(five.line("nightly").is_some());
        assert!(five.line("rc").is_some());
    }

    #[test]
    fn patch_bump_moves_the_line_in_place_and_demotes_fidelity() {
        // A patch release on an existing line (v0.22.1 -> v0.22.2) edits that
        // line only: same window shape, same default, fidelity reset to false.
        let mut doc: DocumentMut = WINDOW.parse().unwrap();
        apply_set_line(&mut doc, "0.22", Some("v0.22.2"), Some("sha222")).unwrap();
        apply_demote_fidelity(&mut doc, "0.22").unwrap();
        let m = CompatManifest::parse(&doc.to_string()).unwrap();

        let line = m.line("0.22").unwrap();
        assert_eq!(line.tag, "v0.22.2");
        assert_eq!(line.protocol_rev, "sha222");
        assert!(!line.fidelity_validated);
        assert!(!line.default, "a patch bump never moves the default");
        assert_eq!(m.default_line().unwrap().line, "0.23");
        let order: Vec<&str> = m.lines.iter().map(|l| l.line.as_str()).collect();
        assert_eq!(order, ["nightly", "rc", "0.23", "0.22", "0.21"]);
    }

    #[test]
    fn demote_fidelity_resets_a_promoted_line() {
        let toml = "\
[[vllm]]
line = \"0.23\"
tag = \"v0.23.0\"
protocol_rev = \"sha23\"
fidelity_validated = true
default = true
";
        let mut doc: DocumentMut = toml.parse().unwrap();
        apply_demote_fidelity(&mut doc, "0.23").unwrap();
        let m = CompatManifest::parse(&doc.to_string()).unwrap();
        assert!(!m.line("0.23").unwrap().fidelity_validated);
    }

    /// Re-serialize a parsed manifest back to TOML for a second roll. The field
    /// order matches what `apply_roll_stable`/the file use, so a round trip is faithful.
    fn toml_for(m: &CompatManifest) -> String {
        let mut out = String::new();
        for l in &m.lines {
            out.push_str("[[vllm]]\n");
            out.push_str(&format!("line = \"{}\"\n", l.line));
            out.push_str(&format!("tag = \"{}\"\n", l.tag));
            out.push_str(&format!("protocol_rev = \"{}\"\n", l.protocol_rev));
            out.push_str(&format!("fidelity_validated = {}\n", l.fidelity_validated));
            if l.default {
                out.push_str("default = true\n");
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn nightly_golden_block_is_appended_when_missing() {
        let out = replace_nightly_goldens_block("header\n", "[[golden]]\nline = \"nightly\"\n");
        assert!(out.contains("# BEGIN NIGHTLY GOLDENS"));
        assert!(out.contains("[[golden]]"));
        assert!(out.contains("# END NIGHTLY GOLDENS"));
    }

    #[test]
    fn nightly_golden_block_replaces_existing_block() {
        let existing = "\
header

# BEGIN NIGHTLY GOLDENS
stale-entry
# END NIGHTLY GOLDENS

tail
";
        let out = replace_nightly_goldens_block(existing, "new");
        assert!(out.contains("header"));
        assert!(out.contains("tail"));
        assert!(out.contains("new"));
        assert!(!out.contains("stale-entry"));
    }

    #[test]
    fn jwt_expiration_is_formatted_for_exec_credential() {
        let payload = URL_SAFE_NO_PAD.encode(r#"{"exp":1700000000}"#);
        let token = format!("header.{payload}.signature");
        assert_eq!(
            jwt_expiration_timestamp(&token).unwrap(),
            "2023-11-14T22:13:20Z"
        );
    }

    #[test]
    fn kubeconfig_uses_xtask_exec_plugin() {
        let json = kubeconfig_json("https://cluster.example", Path::new("/tmp/xtask")).unwrap();
        let value: JsonValue = serde_json::from_str(&json).unwrap();
        assert_eq!(value["kind"], "Config");
        assert_eq!(
            value["clusters"][0]["cluster"]["server"],
            "https://cluster.example"
        );
        assert_eq!(value["users"][0]["user"]["exec"]["command"], "/tmp/xtask");
        assert_eq!(
            value["users"][0]["user"]["exec"]["args"][0],
            "github-oidc-exec-credential"
        );
        assert_eq!(
            value["users"][0]["user"]["exec"]["interactiveMode"],
            "Never"
        );
    }
}
