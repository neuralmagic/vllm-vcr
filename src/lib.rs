//! A mock vLLM V1 engine-core backend.
//!
//! It speaks the real ZMQ + msgpack engine-core protocol (reusing the in-tree
//! `vllm-engine-core-client` crate, so it sits behind the real vLLM Rust or Python
//! frontend unmodified), and carries a prefill/decode data plane that, with the
//! `nixl` feature, moves real KV-cache bytes over NIXL (UCX/DRAM or RDMA) without a GPU.
//!
//! Two birds:
//!   1. faithful frontend testing without a model or GPU, and
//!   2. a real P/D data path exercised over CPU RDMA.

use anyhow::{Context as _, Result, bail};
use clap::Parser;
use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use tracing::{error, info};
use vllm_engine_core_client::EngineId;
use vllm_engine_core_client::mock_engine::{
    MockEngineConfig, MockEngineSockets, connect_to_frontend,
};

pub mod dataplane;
mod engine;
mod io;
pub mod latency;

use dataplane::PdRole;
use latency::LatencyModel;

/// Mock engine-core backend for frontend + prefill/decode data-plane testing.
#[derive(Debug, Clone, Parser)]
#[command(
    name = "inference-sim",
    about = "Run a mock vLLM engine-core backend with an optional NIXL KV data plane."
)]
pub struct Opt {
    /// Frontend-owned ZMQ handshake address.
    #[arg(long, default_value = "tcp://127.0.0.1:29550")]
    pub handshake_address: String,

    /// Number of mock engine identities to register with the frontend.
    #[arg(long, default_value_t = 1)]
    pub engine_count: usize,

    /// Prefill/decode role this process plays.
    #[arg(long, value_enum, default_value_t = PdRole::Both)]
    pub pd_role: PdRole,

    /// Number of accepted output tokens included in each EngineCoreOutput.
    #[arg(long, default_value_t = 1)]
    pub output_token_chunk_size: usize,

    /// Random token IDs are sampled uniformly from 0..vocab_size.
    #[arg(long, default_value_t = 32_000)]
    pub vocab_size: u32,

    /// Base seed for deterministic random token generation.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Bytes per fabricated KV block (NIXL data plane only).
    #[arg(long, default_value_t = 128 * 1024)]
    pub kv_block_bytes: usize,

    /// Prompt tokens that map to one KV block (NIXL data plane only).
    #[arg(long, default_value_t = 16)]
    pub tokens_per_block: usize,

    /// This engine's id, advertised as `remote_engine_id` in kv_transfer_params.
    /// Set per-pod (e.g. from POD_NAME) so a decode peer can address this prefill.
    #[arg(long, env = "MOCK_ENGINE_ID", default_value = "mock-engine-0")]
    pub engine_id: String,

    /// Host advertised as `remote_host` for the NIXL metadata side channel
    /// (set to the pod IP in k8s).
    #[arg(long, env = "MOCK_SIDE_CHANNEL_HOST", default_value = "127.0.0.1")]
    pub side_channel_host: String,

    /// Port advertised as `remote_port` for the NIXL metadata side channel.
    #[arg(long, env = "MOCK_SIDE_CHANNEL_PORT", default_value_t = 5600)]
    pub side_channel_port: u32,

    /// Log a summary line for each request.
    #[arg(long)]
    pub log_requests: bool,

    // === Latency model (all milliseconds; 0 = instant, the default) ===
    // Ported from llm-d-inference-sim. The real frontend measures TTFT/ITL from when we
    // emit tokens, so these knobs drive both response timing and the vllm:* latency metrics.
    /// Fixed time-to-first-token. When this and its std-dev are 0, the token-count prefill
    /// model (`--prefill-overhead` + `--prefill-time-per-token`) is used instead.
    #[arg(long, default_value_t = 0)]
    pub time_to_first_token: u64,

    /// Standard deviation for time-to-first-token.
    #[arg(long, default_value_t = 0)]
    pub time_to_first_token_std_dev: u64,

    /// Time to generate one output token (decode step).
    #[arg(long, default_value_t = 0)]
    pub inter_token_latency: u64,

    /// Standard deviation for inter-token latency.
    #[arg(long, default_value_t = 0)]
    pub inter_token_latency_std_dev: u64,

    /// Fixed prefill overhead, added once per request in the token-count prefill model.
    #[arg(long, default_value_t = 0)]
    pub prefill_overhead: u64,

    /// Per-(uncached-)prompt-token prefill cost in the token-count prefill model.
    #[arg(long, default_value_t = 0)]
    pub prefill_time_per_token: u64,

    /// Standard deviation for the token-count prefill time.
    #[arg(long, default_value_t = 0)]
    pub prefill_time_std_dev: u64,

    /// Fixed KV-cache transfer time for a `do_remote_prefill` decode request. When this and
    /// its std-dev are 0, the per-token transfer model is used instead.
    #[arg(long, default_value_t = 0)]
    pub kv_cache_transfer_latency: u64,

    /// Standard deviation for the fixed KV-cache transfer time.
    #[arg(long, default_value_t = 0)]
    pub kv_cache_transfer_latency_std_dev: u64,

    /// Per-prompt-token KV-cache transfer cost for a `do_remote_prefill` decode request.
    #[arg(long, default_value_t = 0)]
    pub kv_cache_transfer_time_per_token: u64,

    /// Standard deviation for the per-token KV-cache transfer cost.
    #[arg(long, default_value_t = 0)]
    pub kv_cache_transfer_time_std_dev: u64,

    /// Latency multiplier at full load (`>= 1.0`; 1.0 disables load scaling). Latency grows
    /// linearly from 1.0 with one request to this value at `--max-num-seqs` concurrent ones.
    #[arg(long, default_value_t = 1.0)]
    pub time_factor_under_load: f64,

    /// Concurrency at which the load factor reaches `--time-factor-under-load`. Also the
    /// denominator for `kv_cache_usage` reporting.
    #[arg(long, default_value_t = 5)]
    pub max_num_seqs: u64,

    /// KV-cache capacity in blocks, used to report `vllm:kv_cache_usage_perc`.
    #[arg(long, default_value_t = 1024)]
    pub kv_cache_size: u64,
}

impl Opt {
    /// Build the latency model from the configured timing knobs.
    pub fn latency_model(&self) -> LatencyModel {
        LatencyModel {
            time_to_first_token: self.time_to_first_token,
            time_to_first_token_std_dev: self.time_to_first_token_std_dev,
            inter_token_latency: self.inter_token_latency,
            inter_token_latency_std_dev: self.inter_token_latency_std_dev,
            prefill_overhead: self.prefill_overhead,
            prefill_time_per_token: self.prefill_time_per_token,
            prefill_time_std_dev: self.prefill_time_std_dev,
            kv_cache_transfer_latency: self.kv_cache_transfer_latency,
            kv_cache_transfer_latency_std_dev: self.kv_cache_transfer_latency_std_dev,
            kv_cache_transfer_time_per_token: self.kv_cache_transfer_time_per_token,
            kv_cache_transfer_time_std_dev: self.kv_cache_transfer_time_std_dev,
            time_factor_under_load: self.time_factor_under_load,
            max_num_seqs: self.max_num_seqs,
        }
    }
}

/// Run one mock engine until shutdown or transport failure.
async fn run_engine(engine_index: u32, opt: Opt, shutdown: CancellationToken) -> Result<()> {
    let MockEngineSockets { data_sockets, .. } = connect_to_frontend(
        &opt.handshake_address,
        EngineId::from_engine_index(engine_index),
        MockEngineConfig::default(),
    )
    .await
    .with_context(|| format!("engine {engine_index} failed to connect to frontend"))?;

    info!(engine_index, role = ?opt.pd_role, "engine connected to frontend");

    let (input_tx, input_rx) = mpsc::unbounded_channel();
    let (output_tx, output_rx) = mpsc::channel(64);

    let mut io_loop = tokio::spawn(io::run_io_loop(
        data_sockets,
        input_tx,
        output_rx,
        shutdown.clone(),
    ));
    let mut engine_loop = tokio::spawn(engine::run_engine_loop(
        engine_index,
        opt,
        input_rx,
        output_tx,
        shutdown.clone(),
    ));

    tokio::select! {
        biased;
        _ = shutdown.cancelled() => {
            io_loop.abort();
            engine_loop.abort();
            io_loop.await.ok();
            engine_loop.await.ok();
        }
        result = &mut io_loop => {
            error!(engine_index, "engine IO loop exited unexpectedly");
            engine_loop.abort();
            engine_loop.await.ok();
            result??;
        }
        result = &mut engine_loop => {
            error!(engine_index, "engine loop exited unexpectedly");
            io_loop.abort();
            io_loop.await.ok();
            result??;
        }
    }

    info!(engine_index, "engine shut down");
    Ok(())
}

/// Run all requested mock engines until cancellation or one engine task fails.
pub async fn run(opt: Opt, shutdown: CancellationToken) -> Result<()> {
    info!(?opt, "starting mock engine");

    let mut engines = JoinSet::new();
    for engine_index in 0..opt.engine_count {
        engines.spawn(run_engine(
            engine_index as u32,
            opt.clone(),
            shutdown.clone(),
        ));
    }

    tokio::select! {
        biased;
        _ = shutdown.cancelled() => {
            engines.abort_all();
            while engines.join_next().await.is_some() {}
            Ok(())
        }
        joined = engines.join_next() => {
            match joined {
                Some(Ok(Ok(()))) => bail!("engine exited unexpectedly"),
                Some(Ok(Err(error))) => Err(error),
                Some(Err(error)) => Err(error).context("engine task join failed"),
                None => Ok(()),
            }
        }
    }
}
