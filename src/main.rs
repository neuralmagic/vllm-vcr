//! `vllm-vcr`: record, play, and inspect vLLM engine-core traces.
//!
//! One binary, three subcommands (the VCR metaphor):
//!   - `record`  — tap a live vLLM frontend ↔ engine-core link and write a trace.
//!   - `play`    — run the mock engine-core backend (replay a trace or simulate).
//!   - `inspect` — convert, summarize, Perfetto-render, and calibrate traces.
//!
//! `record` and `play` bake the vLLM engine-core protocol in (per build line);
//! `inspect` never handshakes, so it runs on any build.

mod inspect;
mod record;

use std::process::ExitCode;

use anyhow::{Context as _, Result};
use clap::{Parser, Subcommand};
use tokio_util::sync::CancellationToken;

#[derive(Parser)]
#[command(
    name = "vllm-vcr",
    version,
    about = "Record, play, and inspect vLLM engine-core traces."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Record a trace by tapping a live vLLM frontend ↔ engine-core link
    /// (transparent ZMQ proxy; frames relayed verbatim, timing observed).
    Record(record::RecordArgs),

    /// Play a trace back through the mock engine-core backend, or run it as a
    /// GPU-free vLLM engine for frontend / prefill-decode testing.
    Play(Box<vllm_vcr::Opt>),

    /// Inspect traces: convert benchmark reports, summarize, render Perfetto,
    /// and run calibration.
    #[command(subcommand)]
    Inspect(inspect::InspectCommand),

    /// Print a shell completion script to stdout. Source it from your shell rc,
    /// e.g. `vllm-vcr completions fish > ~/.config/fish/completions/vllm-vcr.fish`.
    Completions {
        /// Shell to generate completions for.
        shell: clap_complete::Shell,
    },
}

/// Logs go to stderr so `inspect`'s stdout stays clean for piping (Perfetto
/// JSON, summaries). INFO default keeps `record`/`play` debuggable; override
/// with `RUST_LOG` at startup or through the control API's `PUT /log` while
/// `play` is running.
/// Log whether OpenSSL runs with its FIPS provider, and refuse to start
/// without it when VLLM_VCR_REQUIRE_FIPS is set.
fn check_fips() -> anyhow::Result<()> {
    let fips = sim_s3::openssl_fips_enabled();
    tracing::info!(fips = ?fips, "OpenSSL FIPS provider");
    let required =
        std::env::var_os("VLLM_VCR_REQUIRE_FIPS").is_some_and(|v| !v.is_empty() && v != "0");
    if required && fips != Some(true) {
        anyhow::bail!(
            "VLLM_VCR_REQUIRE_FIPS is set but OpenSSL is not in FIPS mode (fips={fips:?})"
        );
    }
    Ok(())
}

fn init_tracing() -> vllm_vcr::LogFilter {
    use tracing_subscriber::layer::SubscriberExt as _;
    use tracing_subscriber::util::SubscriberInitExt as _;

    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let (filter, handle) = tracing_subscriber::reload::Layer::new(filter);
    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr))
        .init();
    vllm_vcr::LogFilter::new(handle)
}

/// Run the mock engine-core backend on a multi-thread runtime until a shutdown
/// signal (SIGINT/SIGTERM) or transport failure.
fn play(opt: vllm_vcr::Opt, log_filter: vllm_vcr::LogFilter) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to build Tokio runtime")?;

    runtime.block_on(async move {
        let shutdown: CancellationToken = vllm_vcr::shutdown_signal();
        vllm_vcr::run_with_log_filter(opt, shutdown, Some(log_filter)).await
    })
}

fn main() -> ExitCode {
    let log_filter = init_tracing();
    if let Err(err) = check_fips() {
        tracing::error!(%err);
        return ExitCode::FAILURE;
    }
    let cli = Cli::parse();

    let result = match cli.command {
        Command::Record(args) => record::run(args).map(|()| ExitCode::SUCCESS),
        Command::Play(opt) => play(*opt, log_filter).map(|()| ExitCode::SUCCESS),
        Command::Inspect(command) => inspect::run(command),
        Command::Completions { shell } => {
            let mut cmd = <Cli as clap::CommandFactory>::command();
            clap_complete::generate(shell, &mut cmd, "vllm-vcr", &mut std::io::stdout());
            Ok(ExitCode::SUCCESS)
        }
    };

    match result {
        Ok(code) => code,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}
