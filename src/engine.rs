//! The per-engine generation loop. Adapted from vLLM's in-tree `vllm-mock-engine`
//! (`rust/src/mock-engine/src/engine.rs`), with the prefill/decode data-plane hooks
//! added at the two points where real KV bytes would move.
//!
//! Everything wire-facing comes from the `vllm-engine-core-client` crate, so this
//! stays correct as the protocol evolves upstream.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow};
use rand::rngs::StdRng;
use rand::{Rng as _, SeedableRng as _};
use rmpv::Value as MsgpackValue;
use serde::Serialize;
use serde_json::Value as JsonValue;
use tokio::sync::mpsc;
use tokio::time::{Instant, sleep_until};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};
use vllm_engine_core_client::protocol::stats::{PrefillStats, SchedulerStats};
use vllm_engine_core_client::protocol::utility::{
    EngineCoreUtilityRequest, UtilityOutput, UtilityResultEnvelope,
};
use vllm_engine_core_client::protocol::{
    EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest,
};

use crate::Opt;
use crate::dataplane::{KvDataPlane, NixlConfig, RemoteKv, RequestKv, make_data_plane};
use crate::latency::LatencyModel;

/// Derive a stable per-request seed from the CLI seed, engine, and request id.
fn request_seed(base_seed: u64, engine_index: u32, request_id: &str) -> u64 {
    use std::hash::{Hash as _, Hasher as _};
    let mut hasher = std::hash::DefaultHasher::new();
    base_seed.hash(&mut hasher);
    engine_index.hash(&mut hasher);
    request_id.hash(&mut hasher);
    hasher.finish()
}

/// Current UNIX timestamp in seconds for engine-core output envelopes.
fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or_default()
}

/// Build one request output with only token IDs and terminal status populated.
fn request_output(
    request_id: String,
    new_token_ids: Vec<u32>,
    finish_reason: Option<EngineCoreFinishReason>,
) -> EngineCoreOutput {
    EngineCoreOutput {
        request_id,
        new_token_ids,
        finish_reason,
        ..Default::default()
    }
}

/// Produce an empty output with a terminal finish reason for an invalid request.
fn empty_finish_outputs(
    engine_index: u32,
    request_id: String,
    finish_reason: EngineCoreFinishReason,
) -> EngineCoreOutputs {
    let output = request_output(request_id, Vec::new(), Some(finish_reason));
    let finished_requests = BTreeSet::from([output.request_id.clone()]);

    EngineCoreOutputs {
        engine_index,
        outputs: vec![output],
        timestamp: now_secs(),
        finished_requests: Some(finished_requests),
        ..Default::default()
    }
}

/// Encode a utility result into the protocol's msgpack value envelope.
fn utility_envelope<T>(value: T) -> Result<UtilityResultEnvelope>
where
    T: Serialize,
{
    Ok(UtilityResultEnvelope::without_type_info(
        rmpv::ext::to_value(value)?,
    ))
}

/// Produce the minimal utility responses needed by a real frontend.
fn utility_response(
    engine_index: u32,
    request: EngineCoreUtilityRequest,
) -> Result<EngineCoreOutputs> {
    let result = match request.method_name.as_str() {
        "get_supported_tasks" => utility_envelope(vec!["generate"]),
        "is_sleeping" => utility_envelope(false),
        "reset_prefix_cache" => utility_envelope(true),
        "reset_mm_cache"
        | "reset_encoder_cache"
        | "profile"
        | "sleep"
        | "wake_up"
        | "execute_dummy_batch" => utility_envelope(()),
        _ => utility_envelope(MsgpackValue::Nil),
    }?;

    Ok(EngineCoreOutputs {
        engine_index,
        utility_output: Some(UtilityOutput {
            call_id: request.call_id,
            failure_message: None,
            result: Some(result),
        }),
        timestamp: now_secs(),
        ..Default::default()
    })
}

/// The `kv_transfer_params` the frontend ferries down from the OpenAI request. The
/// server merges them into `sampling_params.extra_args["kv_transfer_params"]`
/// (mirroring Python vLLM), so that is where the P/D intent (`do_remote_prefill` /
/// `do_remote_decode` / `remote_*`) arrives. In real vLLM the produce/consume logic
/// lives in the NixlConnector inside the engine; here our data plane plays that role.
fn extract_kv_params(request: &EngineCoreRequest) -> Option<JsonValue> {
    request
        .sampling_params
        .as_ref()?
        .extra_args
        .as_ref()?
        .get("kv_transfer_params")
        .cloned()
}

/// Read a boolean flag out of a `kv_transfer_params` object.
fn kv_flag(kv: &JsonValue, key: &str) -> bool {
    kv.get(key).and_then(JsonValue::as_bool).unwrap_or(false)
}

/// Parse the `remote_*` addressing out of a decode request's `kv_transfer_params`
/// (set by the routing sidecar from the prefill response). `None` if this is not a
/// `do_remote_prefill` request or the fields are missing.
fn parse_remote_kv(kv: &JsonValue) -> Option<RemoteKv> {
    if !kv_flag(kv, "do_remote_prefill") {
        return None;
    }
    Some(RemoteKv {
        engine_id: kv.get("remote_engine_id")?.as_str()?.to_string(),
        host: kv.get("remote_host")?.as_str()?.to_string(),
        port: kv.get("remote_port")?.as_u64()? as u32,
        block_ids: kv
            .get("remote_block_ids")?
            .as_array()?
            .iter()
            .filter_map(JsonValue::as_i64)
            .collect(),
        // Mock extension fields (see RemoteKv); absent when the peer is a real vLLM.
        addr: kv
            .get("remote_buf_addr")
            .and_then(JsonValue::as_u64)
            .unwrap_or(0),
        len: kv
            .get("remote_buf_len")
            .and_then(JsonValue::as_u64)
            .unwrap_or(0),
        pattern: kv
            .get("remote_pattern")
            .and_then(JsonValue::as_u64)
            .unwrap_or(0) as u8,
    })
}

/// Build the `kv_transfer_params` a prefill engine returns, matching vLLM's
/// NixlConnector schema (`scheduler.py:664`). The routing sidecar relays these
/// `remote_*` fields into the decode request.
fn build_prefill_kv_params(
    remote: &RemoteKv,
    request_id: &str,
    remote_num_tokens: usize,
) -> JsonValue {
    serde_json::json!({
        "do_remote_prefill": true,
        "do_remote_decode": false,
        "remote_block_ids": remote.block_ids,
        "remote_engine_id": remote.engine_id,
        "remote_request_id": request_id,
        "remote_host": remote.host,
        "remote_port": remote.port,
        "tp_size": 1,
        "remote_num_tokens": remote_num_tokens,
        // Mock extension: the prefill KV buffer the decode peer reads over NIXL.
        "remote_buf_addr": remote.addr,
        "remote_buf_len": remote.len,
        "remote_pattern": remote.pattern,
    })
}

/// Message sent from the IO loop to the engine task to drive the engine loop.
pub(crate) enum EngineInput {
    Request(Box<EngineCoreRequest>),
    Abort(Vec<String>),
    Utility(EngineCoreUtilityRequest),
    StartDpWave,
}

/// Message sent from the engine task to the IO loop for one engine output batch.
pub(crate) struct EngineOutput {
    pub client_index: u32,
    pub outputs: EngineCoreOutputs,
}

/// Per-request decode state owned by one engine.
#[derive(Debug)]
struct ActiveRequest {
    request_id: String,
    client_index: u32,
    prompt_len: usize,
    max_tokens: usize,
    generated: usize,
    rng: StdRng,
    /// This request asked us to prefill for a remote decoder (`do_remote_decode`), so on
    /// finish we register its KV and stamp the `remote_*` descriptor onto its output.
    prefill_advertise: bool,
    /// This request's KV was prefilled remotely and pulled in (`do_remote_prefill`), so its
    /// prompt tokens count as externally cached for prefill stats / metrics.
    remote_prefill: bool,
    /// When the next output token is due. Set to `now + first-token delay` at admission, then
    /// advanced by the inter-token delay after each emitted token. The engine loop sleeps
    /// until the earliest deadline across all active requests, so this is the timing model.
    next_at: Instant,
}

impl ActiveRequest {
    /// Create a new active request, or return an immediate finish reason if invalid.
    ///
    /// `num_running` is the running-request count *including* this one, used to scale the
    /// first-token delay under load.
    fn new(
        engine_index: u32,
        request: Box<EngineCoreRequest>,
        opt: &Opt,
        latency: &LatencyModel,
        num_running: u64,
    ) -> Result<Self, EngineCoreFinishReason> {
        let incoming_kv = extract_kv_params(&request);
        let prefill_advertise = incoming_kv
            .as_ref()
            .map(|kv| kv_flag(kv, "do_remote_decode"))
            .unwrap_or(false);
        let remote_prefill = incoming_kv
            .as_ref()
            .map(|kv| kv_flag(kv, "do_remote_prefill"))
            .unwrap_or(false);
        let request_id = request.request_id;
        let client_index = request.client_index;
        let prompt_len = request
            .prompt_token_ids
            .as_ref()
            .map(Vec::len)
            .unwrap_or_default();

        let Some(sampling_params) = request.sampling_params else {
            warn!(
                request_id,
                "request has no sampling params; returning engine error"
            );
            return Err(EngineCoreFinishReason::Error);
        };
        let max_tokens = sampling_params.max_tokens as usize;

        if let Some(kv) = &incoming_kv {
            info!(request_id, %kv, "received kv_transfer_params from frontend");
        }

        if opt.log_requests {
            info!(
                request_id,
                prompt_len,
                max_tokens,
                chunk_size = opt.output_token_chunk_size,
                "mock request started"
            );
        }

        if max_tokens == 0 {
            return Err(EngineCoreFinishReason::Length);
        }

        let mut rng = StdRng::seed_from_u64(request_seed(opt.seed, engine_index, &request_id));
        // No prefix cache yet, so no prompt tokens are served from the local cache.
        let first_delay =
            latency.first_token_delay(&mut rng, prompt_len, 0, remote_prefill, num_running);

        Ok(ActiveRequest {
            rng,
            request_id,
            client_index,
            prompt_len,
            max_tokens,
            generated: 0,
            prefill_advertise,
            remote_prefill,
            next_at: Instant::now() + first_delay,
        })
    }

    /// Advance this request by one engine step.
    fn step(&mut self, opt: &Opt) -> EngineCoreOutput {
        let remaining = self.max_tokens - self.generated;
        let chunk_len = remaining.min(opt.output_token_chunk_size);
        let mut new_token_ids = Vec::with_capacity(chunk_len);
        for _ in 0..chunk_len {
            new_token_ids.push(self.rng.random_range(0..opt.vocab_size));
        }
        self.generated += chunk_len;

        let finished = self.generated >= self.max_tokens;
        request_output(
            self.request_id.clone(),
            new_token_ids,
            finished.then_some(EngineCoreFinishReason::Length),
        )
    }

    /// KV blocks this request currently occupies (prompt + generated so far), for the
    /// `kv_cache_usage` metric. Rounds up to whole blocks.
    fn blocks_used(&self, tokens_per_block: usize) -> u64 {
        if tokens_per_block == 0 {
            return 0;
        }
        let tokens = self.prompt_len + self.generated;
        tokens.div_ceil(tokens_per_block) as u64
    }

    /// Prefill breakdown for this request's first output, feeding the prefix-cache and
    /// KV-transfer metrics. Until the prefix-cache model lands, prompt tokens are either all
    /// local compute, or (for a remote-prefilled decode request) all externally cached.
    fn prefill_stats(&self) -> PrefillStats {
        let external = if self.remote_prefill {
            self.prompt_len as u32
        } else {
            0
        };
        PrefillStats {
            num_prompt_tokens: self.prompt_len as u32,
            num_computed_tokens: (self.prompt_len as u32).saturating_sub(external),
            num_external_cached_tokens: external,
            ..Default::default()
        }
    }
}

/// Internal state for one engine instance, owned by the engine loop task.
struct Engine {
    engine_index: u32,
    opt: Opt,
    latency: LatencyModel,
    data_plane: Box<dyn KvDataPlane>,
    active_requests: HashMap<String, ActiveRequest>,
}

impl Engine {
    /// Drain one frontend request message.
    fn handle_input(&mut self, input: EngineInput) -> Result<Vec<EngineOutput>> {
        let mut outputs = Vec::new();

        match input {
            EngineInput::Request(request) => {
                let request_id = request.request_id.clone();
                let client_index = request.client_index;

                if self.active_requests.contains_key(&request_id) {
                    warn!(
                        engine_index = self.engine_index,
                        request_id, "duplicate request id"
                    );
                    return Ok(vec![EngineOutput {
                        client_index,
                        outputs: empty_finish_outputs(
                            self.engine_index,
                            request_id,
                            EngineCoreFinishReason::Error,
                        ),
                    }]);
                }

                // === DATA PLANE: decode-side pull ===
                // A do_remote_prefill request carries the prefill engine's remote_*
                // descriptor; pull its KV over NIXL before we start generating.
                if let Some(remote) = extract_kv_params(&request)
                    .as_ref()
                    .and_then(parse_remote_kv)
                {
                    let prompt_len = request
                        .prompt_token_ids
                        .as_ref()
                        .map(Vec::len)
                        .unwrap_or_default();
                    let kv = RequestKv {
                        request_id: &request_id,
                        num_tokens: prompt_len,
                    };
                    match self.data_plane.pull_prefilled(kv, &remote) {
                        Ok(bytes) => info!(
                            request_id,
                            bytes,
                            engine_id = remote.engine_id,
                            "pulled remote KV before decode"
                        ),
                        Err(error) => warn!(request_id, %error, "remote KV pull failed"),
                    }
                }

                // Count this request among the running set for its own load-factor scaling.
                let num_running = self.active_requests.len() as u64 + 1;
                match ActiveRequest::new(
                    self.engine_index,
                    request,
                    &self.opt,
                    &self.latency,
                    num_running,
                ) {
                    Ok(request) => {
                        self.active_requests.insert(request_id, request);
                    }
                    Err(finish_reason) => {
                        return Ok(vec![EngineOutput {
                            client_index,
                            outputs: empty_finish_outputs(
                                self.engine_index,
                                request_id,
                                finish_reason,
                            ),
                        }]);
                    }
                }
            }

            EngineInput::Abort(request_ids) => {
                let mut by_client =
                    BTreeMap::<u32, (Vec<EngineCoreOutput>, BTreeSet<String>)>::new();
                for request_id in request_ids {
                    self.data_plane.release(&request_id);
                    if let Some(request) = self.active_requests.remove(&request_id) {
                        let output = request_output(
                            request_id.clone(),
                            Vec::new(),
                            Some(EngineCoreFinishReason::Abort),
                        );
                        let (outs, finished) = by_client
                            .entry(request.client_index)
                            .or_insert_with(|| (Vec::new(), BTreeSet::new()));
                        outs.push(output);
                        finished.insert(request_id.clone());
                        if self.opt.log_requests {
                            info!(request_id, finish_reason = "abort", "request aborted");
                        }
                    }
                }
                for (client_index, (client_outputs, finished_requests)) in by_client {
                    outputs.push(EngineOutput {
                        client_index,
                        outputs: EngineCoreOutputs {
                            engine_index: self.engine_index,
                            outputs: client_outputs,
                            timestamp: now_secs(),
                            finished_requests: Some(finished_requests),
                            ..Default::default()
                        },
                    });
                }
            }

            EngineInput::Utility(request) => {
                debug!(
                    engine_index = self.engine_index,
                    call_id = %request.call_id,
                    method = request.method_name,
                    "utility request"
                );
                let client_index = request.client_index;
                outputs.push(EngineOutput {
                    client_index,
                    outputs: utility_response(self.engine_index, request)?,
                });
            }

            EngineInput::StartDpWave => {
                debug!(engine_index = self.engine_index, "ignoring START_DP_WAVE");
            }
        }

        Ok(outputs)
    }

    /// Advance every request whose token is due, returning one batched engine output per
    /// client. Each emitted token reschedules the request by the inter-token delay; the
    /// engine loop sleeps until the earliest deadline, so this is paced by the latency model.
    fn step(&mut self) -> Vec<EngineOutput> {
        if self.active_requests.is_empty() {
            return Vec::new();
        }

        let now = Instant::now();
        // Running-request count drives the inter-token load factor; snapshot before the loop.
        let num_running = self.active_requests.len() as u64;

        let mut by_client = BTreeMap::<u32, (Vec<EngineCoreOutput>, BTreeSet<String>)>::new();
        let mut finished_ids = BTreeSet::new();

        // Collect (client_index, request_id, num_tokens) for finished prefill requests so
        // we can advertise their KV after the borrow on active_requests ends.
        let mut to_advertise: Vec<(u32, String, usize)> = Vec::new();

        for request in self.active_requests.values_mut() {
            if request.next_at > now {
                // Not due yet; its deadline keeps the engine loop's sleep honest.
                continue;
            }
            let client_index = request.client_index;
            let is_first = request.generated == 0;
            let mut output = request.step(&self.opt);
            let request_id = request.request_id.clone();
            let finished = output.finished();

            // The first output of a request carries its prefill breakdown for metrics.
            if is_first {
                output.prefill_stats = Some(request.prefill_stats());
            }

            if finished {
                finished_ids.insert(request_id.clone());
                if request.prefill_advertise {
                    to_advertise.push((client_index, request_id.clone(), request.prompt_len));
                }
                if self.opt.log_requests {
                    info!(
                        request_id,
                        prompt_len = request.prompt_len,
                        output_tokens = request.generated,
                        finish_reason = "length",
                        "request finished"
                    );
                }
            } else {
                // Schedule the next decode token.
                request.next_at = now
                    + self
                        .latency
                        .inter_token_delay(&mut request.rng, num_running);
            }

            let (outs, finished_set) = by_client
                .entry(client_index)
                .or_insert_with(|| (Vec::new(), BTreeSet::new()));
            if finished {
                finished_set.insert(request_id);
            }
            // Hold the output so we can attach kv_transfer_params below if advertised.
            outs.push(std::mem::take(&mut output));
        }

        for request_id in &finished_ids {
            self.active_requests.remove(request_id);
        }

        // === DATA PLANE: prefill-side advertise ===
        // Once prefill finishes, register the fake KV and stamp the real kv_transfer_params
        // (remote_engine_id/host/port/block_ids) onto the finishing output for the decoder.
        for (client_index, request_id, num_tokens) in to_advertise {
            let kv = RequestKv {
                request_id: &request_id,
                num_tokens,
            };
            match self.data_plane.advertise_prefilled(kv) {
                Ok(remote) => {
                    if let Some((outs, _)) = by_client.get_mut(&client_index)
                        && let Some(out) = outs.iter_mut().find(|o| o.request_id == request_id)
                    {
                        out.kv_transfer_params =
                            Some(build_prefill_kv_params(&remote, &request_id, num_tokens));
                    }
                }
                Err(error) => warn!(request_id, %error, "prefill KV advertise failed"),
            }
        }

        // Computed once after removals so the gauges reflect post-step state (e.g. the batch
        // that finishes the last request reports num_running = 0). Cloned onto each client.
        let stats = self.scheduler_stats();

        by_client
            .into_iter()
            .filter_map(|(client_index, (outputs, finished_requests))| {
                (!outputs.is_empty()).then(|| EngineOutput {
                    client_index,
                    outputs: EngineCoreOutputs {
                        engine_index: self.engine_index,
                        outputs,
                        scheduler_stats: Some(Box::new(stats.clone())),
                        timestamp: now_secs(),
                        finished_requests: (!finished_requests.is_empty())
                            .then_some(finished_requests),
                        ..Default::default()
                    },
                })
            })
            .collect()
    }

    /// Snapshot of scheduler state for the frontend's `vllm:*` gauges. `num_waiting_reqs`
    /// stays 0 until the admission-queue model (Phase 3) lands.
    fn scheduler_stats(&self) -> SchedulerStats {
        let used_blocks: u64 = self
            .active_requests
            .values()
            .map(|request| request.blocks_used(self.opt.tokens_per_block))
            .sum();
        let kv_cache_usage = if self.opt.kv_cache_size == 0 {
            0.0
        } else {
            (used_blocks as f64 / self.opt.kv_cache_size as f64).min(1.0)
        };

        SchedulerStats {
            num_running_reqs: self.active_requests.len() as u64,
            num_waiting_reqs: 0,
            kv_cache_usage,
            ..Default::default()
        }
    }

    /// The soonest a token is due across all active requests; `None` when idle. The engine
    /// loop sleeps until this instant before calling `step`.
    fn earliest_deadline(&self) -> Option<Instant> {
        self.active_requests
            .values()
            .map(|request| request.next_at)
            .min()
    }
}

/// Run the main loop for one engine, receiving `EngineInput` and sending `EngineOutput`
/// until `shutdown` is cancelled.
pub(crate) async fn run_engine_loop(
    engine_index: u32,
    opt: Opt,
    mut input_rx: mpsc::UnboundedReceiver<EngineInput>,
    output_tx: mpsc::Sender<EngineOutput>,
    shutdown: CancellationToken,
) -> Result<()> {
    let role = opt.pd_role;
    let cfg = NixlConfig {
        kv_block_bytes: opt.kv_block_bytes,
        tokens_per_block: opt.tokens_per_block,
        engine_id: opt.engine_id.clone(),
        side_channel_host: opt.side_channel_host.clone(),
        side_channel_port: opt.side_channel_port,
    };
    let latency = opt.latency_model();
    let mut engine = Engine {
        engine_index,
        opt,
        latency,
        data_plane: make_data_plane(role, cfg),
        active_requests: HashMap::new(),
    };

    loop {
        // Wake at the soonest token deadline; disabled (no timer branch) when idle.
        let next_deadline = engine.earliest_deadline();
        let outputs = tokio::select! {
            biased;
            _ = shutdown.cancelled() => break,

            input = input_rx.recv() => {
                let input = input.ok_or_else(|| anyhow!("engine input channel closed"))?;
                engine.handle_input(input)?
            }

            _ = async { sleep_until(next_deadline.unwrap_or_else(Instant::now)).await },
                if next_deadline.is_some() =>
            {
                engine.step()
            }
        };

        for output in outputs {
            output_tx
                .send(output)
                .await
                .map_err(|_| anyhow!("engine IO task shut down"))?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use clap::Parser as _;
    use vllm_engine_core_client::protocol::{EngineCoreRequest, EngineCoreSamplingParams};

    use super::*;
    use crate::dataplane::{NixlConfig, PdRole, make_data_plane};

    fn test_opt() -> Opt {
        // clap fills every field with its declared default (all latency knobs = 0 / instant).
        Opt::parse_from(["inference-sim"])
    }

    fn test_engine(opt: Opt) -> Engine {
        let cfg = NixlConfig {
            kv_block_bytes: opt.kv_block_bytes,
            tokens_per_block: opt.tokens_per_block,
            engine_id: opt.engine_id.clone(),
            side_channel_host: opt.side_channel_host.clone(),
            side_channel_port: opt.side_channel_port,
        };
        Engine {
            engine_index: 0,
            latency: opt.latency_model(),
            data_plane: make_data_plane(PdRole::Both, cfg),
            active_requests: HashMap::new(),
            opt,
        }
    }

    fn request(id: &str, prompt_len: usize, max_tokens: u32) -> EngineCoreRequest {
        EngineCoreRequest {
            request_id: id.to_string(),
            prompt_token_ids: Some(vec![0u32; prompt_len]),
            sampling_params: Some(EngineCoreSamplingParams {
                max_tokens,
                ..EngineCoreSamplingParams::for_test()
            }),
            ..Default::default()
        }
    }

    fn add(engine: &mut Engine, req: EngineCoreRequest) {
        engine
            .handle_input(EngineInput::Request(Box::new(req)))
            .expect("handle_input");
    }

    /// Drain steps until the engine is idle, returning the flat output list. Safe only when
    /// the latency model is instant (deadlines never in the future), as in these tests.
    fn drain(engine: &mut Engine) -> Vec<EngineCoreOutput> {
        let mut all = Vec::new();
        while !engine.active_requests.is_empty() {
            let batch = engine.step();
            assert!(
                !batch.is_empty(),
                "instant model must make progress each step"
            );
            for output in batch {
                all.extend(output.outputs.outputs);
            }
        }
        all
    }

    #[test]
    fn unconfigured_engine_is_instant() {
        let mut engine = test_engine(test_opt());
        add(&mut engine, request("r1", 4, 3));
        // Instant model: the first token is due immediately.
        assert!(engine.earliest_deadline().unwrap() <= Instant::now());
        let outputs = drain(&mut engine);
        let tokens: usize = outputs.iter().map(|o| o.new_token_ids.len()).sum();
        assert_eq!(tokens, 3);
        assert!(outputs.last().unwrap().finished());
    }

    #[test]
    fn ttft_delays_the_first_token() {
        let mut opt = test_opt();
        opt.time_to_first_token = 10_000; // 10s, no std-dev -> exact, comfortably in the future
        let mut engine = test_engine(opt);

        let before = Instant::now();
        add(&mut engine, request("r1", 4, 2));
        let deadline = engine.earliest_deadline().expect("a deadline");
        assert!(deadline >= before + Duration::from_millis(9_000));

        // The token is not due yet, so a step right now produces nothing and keeps the request.
        assert!(engine.step().is_empty());
        assert_eq!(engine.active_requests.len(), 1);
    }

    #[test]
    fn prefill_stats_only_on_first_output() {
        let mut engine = test_engine(test_opt());
        add(&mut engine, request("r1", 7, 4));
        let outputs = drain(&mut engine);

        let with_prefill: Vec<_> = outputs
            .iter()
            .filter(|o| o.prefill_stats.is_some())
            .collect();
        assert_eq!(
            with_prefill.len(),
            1,
            "exactly one output carries prefill stats"
        );
        let stats = with_prefill[0].prefill_stats.as_ref().unwrap();
        assert_eq!(stats.num_prompt_tokens, 7);
        assert_eq!(stats.num_computed_tokens, 7);
        assert_eq!(stats.num_external_cached_tokens, 0);
    }

    #[test]
    fn scheduler_stats_report_running_and_kv_usage() {
        let mut opt = test_opt();
        opt.tokens_per_block = 16;
        opt.kv_cache_size = 100; // blocks
        let mut engine = test_engine(opt);

        // One request, 32 prompt tokens -> ceil(32/16) = 2 blocks while running.
        add(&mut engine, request("r1", 32, 5));
        let stats = engine.scheduler_stats();
        assert_eq!(stats.num_running_reqs, 1);
        assert_eq!(stats.num_waiting_reqs, 0);
        assert!(
            (stats.kv_cache_usage - 0.02).abs() < 1e-9,
            "got {}",
            stats.kv_cache_usage
        );

        // Drain to completion; the engine is empty so usage is back to zero.
        let _ = drain(&mut engine);
        let idle = engine.scheduler_stats();
        assert_eq!(idle.num_running_reqs, 0);
        assert_eq!(idle.kv_cache_usage, 0.0);
    }

    #[test]
    fn final_batch_reports_zero_running() {
        let mut engine = test_engine(test_opt());
        add(&mut engine, request("r1", 4, 1)); // single output token, finishes in one step
        let batch = engine.step();
        let stats = batch[0]
            .outputs
            .scheduler_stats
            .as_ref()
            .expect("stats on batch");
        // Computed after the finished request is removed, so the gauge drops to 0.
        assert_eq!(stats.num_running_reqs, 0);
        assert!(engine.active_requests.is_empty());
    }

    #[test]
    fn remote_prefill_request_counts_prompt_as_external_cached() {
        let mut engine = test_engine(test_opt());
        let mut req = request("r1", 9, 2);
        let mut extra = HashMap::new();
        extra.insert(
            "kv_transfer_params".to_string(),
            serde_json::json!({ "do_remote_prefill": true }),
        );
        req.sampling_params.as_mut().unwrap().extra_args = Some(extra);
        add(&mut engine, req);

        let outputs = drain(&mut engine);
        let stats = outputs
            .iter()
            .find_map(|o| o.prefill_stats.as_ref())
            .expect("prefill stats");
        assert_eq!(stats.num_external_cached_tokens, 9);
        assert_eq!(stats.num_computed_tokens, 0);
    }
}
