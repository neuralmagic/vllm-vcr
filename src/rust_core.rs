//! `play --use-rust-core`: vLLM's own scheduler and KV cache manager
//! (`vllm-engine-core`), stepped against a simulated GPU worker.
//!
//! ```text
//!   frontend ──ZMQ──▶ io loop ──EngineInput──▶ RustCoreEngine
//!                                                │ add_request / finish_requests
//!                                                ▼
//!                          ┌──────────── Scheduler (vLLM) ◀───────────┐
//!                          │ schedule()                               │ update_from_output()
//!                          ▼                                          │
//!                     SimRunner::execute() ── step duration ──▶ (step clock)
//!                                                                     │
//!   frontend ◀──ZMQ── io loop ◀──EngineOutput── emission queue ◀──────┘
//! ```

mod convert;
mod nixl;
mod runner;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context as _, Result, anyhow, bail};
use rand::SeedableRng as _;
use rand::rngs::StdRng;
use sim_protocol::mock_engine::DEFAULT_MOCK_MAX_MODEL_LEN;
use sim_protocol::vllm::{
    self as vllm, EngineCoreFinishReason, EngineCoreOutput, EngineCoreRequest, SchedulerStats,
};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};
use vllm_engine_core::hash::RequestBlockHasher;
use vllm_engine_core::kv_cache::KvCacheEvent;
use vllm_engine_core::sched::output::SchedulerOutput;
use vllm_engine_core::sched::request::{Request, RequestStatus};
use vllm_engine_core::sched::scheduler::{AcceptAllStructuredOutput, ModelRunnerOutput, Scheduler};
use vllm_engine_core_connectors::nixl::{NixlKvConnector, NixlKvTransferParams};

use crate::Opt;
use crate::control::{Config, ConfigPatch, ControlError, ControlReply, ControlRequest, Stats};
use crate::engine::{
    empty_finish_outputs, fail_on_arrival, now_monotonic, request_output, request_seed,
    utility_envelope, utility_response, utility_result_outputs,
};
use crate::engine_core::{self, EngineCore, EngineInput, EngineOutput, UtilityRequestSpec};
use crate::kvevents::KvEventTx;
use crate::kvparams::extract_kv_params;
use crate::latency::LatencyModel;
use crate::lora::LoraRegistry;
use crate::rust_core::nixl::NixlWorker;
use crate::rust_core::runner::{OutputLimits, SimRunner, StepResult};
use crate::tokens::TokenSource;

/// How often a step that could schedule nothing is retried, as in vLLM's engine loop.
const IDLE_POLL: Duration = Duration::from_millis(1);

/// Reject `play` options the scheduler-backed engine does not implement.
pub(crate) fn validate(opt: &Opt, latency: &dyn LatencyModel) -> Result<()> {
    if opt.output_token_chunk_size != 1 {
        bail!("--use-rust-core emits one token per step; --output-token-chunk-size must be 1");
    }
    if opt.replay_steps.is_some() {
        bail!("--use-rust-core does not support --replay-steps yet");
    }
    if latency.spec_tokens().is_some() {
        bail!("--use-rust-core does not support multi-token (speculative) latency traces yet");
    }
    Ok(())
}

/// Run the engine loop on its own thread until shutdown or `abort`.
pub(crate) fn spawn(
    engine_index: u32,
    opt: Opt,
    events: Option<KvEventTx>,
    input_rx: mpsc::UnboundedReceiver<EngineInput>,
    output_tx: mpsc::UnboundedSender<EngineOutput>,
    shutdown: CancellationToken,
    abort: CancellationToken,
) -> JoinHandle<Result<()>> {
    tokio::task::spawn_blocking(move || {
        let shutdown_timeout = Duration::from_secs(opt.shutdown_timeout);
        let engine = RustCoreEngine::new(engine_index, opt, events)?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .context("building the engine-core runtime")?;
        runtime.block_on(async move {
            tokio::select! {
                result = engine_core::run_loop(
                    engine, input_rx, output_tx, shutdown, shutdown_timeout,
                ) => result,
                _ = abort.cancelled() => Ok(()),
            }
        })
    })
}

/// `kv_transfer_params` with null-valued keys dropped. The llm-d routing sidecar
/// sends the `remote_*` keys as null on a prefill request.
fn without_nulls(mut params: serde_json::Value) -> serde_json::Value {
    if let Some(dict) = params.as_object_mut() {
        dict.retain(|_, value| !value.is_null());
    }
    params
}

/// A scheduled step the worker is still executing.
struct InFlightStep {
    end: Instant,
    scheduled: SchedulerOutput,
    result: ModelRunnerOutput,
}

struct PendingEmit {
    client_index: u32,
    output: EngineCoreOutput,
}

type ClientBatches = BTreeMap<u32, (Vec<EngineCoreOutput>, BTreeSet<String>)>;

pub(crate) struct RustCoreEngine {
    engine_index: u32,
    opt: Opt,
    scheduler: Scheduler,
    /// The scheduler's KV connector, shared to register each request's transfer params.
    nixl: NixlKvConnector,
    runner: SimRunner,
    hasher: Arc<RequestBlockHasher>,
    events: Option<KvEventTx>,
    failure_rng: StdRng,
    loras: LoraRegistry,
    /// Client of every request that has not sent its final output yet.
    owed: HashMap<String, u32>,
    in_flight: Option<InFlightStep>,
    /// Outputs waiting for their emission time, in `(due, sequence)` order.
    emits: BTreeMap<(Instant, u64), PendingEmit>,
    emit_seq: u64,
    /// Scheduler stats not yet sent to a frontend.
    pending_stats: Option<SchedulerStats>,
    last_sent_stats: Option<SchedulerStats>,
    stats: Stats,
}

impl RustCoreEngine {
    fn new(engine_index: u32, opt: Opt, events: Option<KvEventTx>) -> Result<Self> {
        let latency = opt.build_latency()?;
        let token_source = opt.build_token_source()?;
        Self::from_parts(engine_index, opt, latency, token_source, events)
    }

    fn from_parts(
        engine_index: u32,
        opt: Opt,
        latency: Box<dyn LatencyModel>,
        token_source: Box<dyn TokenSource>,
        events: Option<KvEventTx>,
    ) -> Result<Self> {
        validate(&opt, latency.as_ref())?;
        let nixl = nixl::scheduler_half(&opt, engine_index)?;
        let scheduler = Scheduler::new(
            convert::scheduler_config(&opt),
            convert::kv_cache_config(&opt),
            Some(Box::new(nixl.clone())),
            Box::new(AcceptAllStructuredOutput),
        )
        .map_err(|error| anyhow!("building the vLLM scheduler: {error}"))?;
        Ok(RustCoreEngine {
            engine_index,
            scheduler,
            nixl,
            runner: SimRunner::new(
                latency,
                token_source,
                NixlWorker::new(&opt, engine_index),
                opt.tokens_per_block,
                opt.max_num_batched_tokens.max(1) as usize,
                opt.time_scale,
            ),
            hasher: convert::request_block_hasher(&opt),
            events,
            failure_rng: StdRng::seed_from_u64(
                opt.seed ^ u64::from(engine_index).wrapping_mul(0x9e3779b9),
            ),
            loras: LoraRegistry::new(opt.max_loras as usize),
            owed: HashMap::new(),
            in_flight: None,
            emits: BTreeMap::new(),
            emit_seq: 0,
            pending_stats: None,
            last_sent_stats: None,
            stats: Stats::default(),
            opt,
        })
    }

    fn finish_on_arrival(
        &mut self,
        client_index: u32,
        request_id: String,
        reason: EngineCoreFinishReason,
    ) -> Vec<EngineOutput> {
        self.stats.requests_failed += 1;
        if self.opt.log_requests {
            info!(request_id, ?reason, "request finished on arrival");
        }
        vec![EngineOutput {
            client_index,
            outputs: empty_finish_outputs(self.engine_index, request_id, reason),
        }]
    }

    fn add_request(&mut self, request: Box<EngineCoreRequest>) -> Vec<EngineOutput> {
        let request_id = request.request_id.clone();
        let client_index = request.client_index;
        self.stats.requests_received += 1;

        if self.owed.contains_key(&request_id) {
            warn!(
                engine_index = self.engine_index,
                request_id, "duplicate request id"
            );
            return self.finish_on_arrival(client_index, request_id, EngineCoreFinishReason::Error);
        }
        if let Some(reason) = fail_on_arrival(&self.opt, &mut self.failure_rng, &request) {
            return self.finish_on_arrival(client_index, request_id, reason);
        }
        let Some(sampling) = request.sampling_params.as_ref() else {
            warn!(
                request_id,
                "request has no sampling params; returning engine error"
            );
            return self.finish_on_arrival(client_index, request_id, EngineCoreFinishReason::Error);
        };
        let max_tokens = sampling.max_tokens as usize;
        if max_tokens == 0 {
            return self.finish_on_arrival(
                client_index,
                request_id,
                EngineCoreFinishReason::Length,
            );
        }
        let kv_params = match extract_kv_params(&request)
            .map(|params| NixlKvTransferParams::from_wire(&without_nulls(params)))
            .transpose()
        {
            Ok(params) => params.flatten(),
            Err(error) => {
                warn!(request_id, error, "request has unusable kv_transfer_params");
                return self.finish_on_arrival(
                    client_index,
                    request_id,
                    EngineCoreFinishReason::Error,
                );
            }
        };

        let prompt_token_ids = request.prompt_token_ids.clone().unwrap_or_default();
        let prompt_len = prompt_token_ids.len();
        let max_model_len = if self.opt.max_model_len > 0 {
            self.opt.max_model_len
        } else {
            DEFAULT_MOCK_MAX_MODEL_LEN
        } as usize;
        let eos_terminable = sampling.eos_token_id.is_some();
        let open_ended = prompt_len.saturating_add(max_tokens) >= max_model_len;
        let output_len = self.runner.register(
            &request_id,
            request_seed(self.opt.seed, self.engine_index, &request_id),
            prompt_token_ids,
            OutputLimits {
                max_tokens,
                min_tokens: sampling.min_tokens as usize,
                sample_output_len: self.opt.replay_tokens.is_none()
                    && (eos_terminable || open_ended),
            },
        );

        let mut new_request = convert::new_request(&request, sampling, output_len, &self.hasher);
        new_request.do_remote_prefill = kv_params
            .as_ref()
            .is_some_and(|params| params.do_remote_prefill);
        let admitted = Request::new(new_request)
            .map_err(|error| error.to_string())
            .and_then(|mut request| {
                self.nixl
                    .add_wire_request(&mut request, kv_params, self.opt.tokens_per_block, 0)
                    .map_err(|error| error.to_string())?;
                Ok(request)
            });
        match admitted {
            Ok(request) => self.scheduler.add_request(request),
            Err(error) => {
                warn!(request_id, error, "scheduler rejected the request");
                self.runner.forget(&request_id);
                return self.finish_on_arrival(
                    client_index,
                    request_id,
                    EngineCoreFinishReason::Error,
                );
            }
        }
        if self.opt.log_requests {
            info!(
                request_id,
                prompt_len,
                max_tokens = output_len,
                "mock request started"
            );
        }
        self.owed.insert(request_id, client_index);
        self.ensure_step(Instant::now());
        Vec::new()
    }

    /// Abort every listed request that still owes output. Unknown ids are ignored.
    fn abort_requests(&mut self, request_ids: Vec<String>) -> Vec<EngineOutput> {
        let known: BTreeSet<String> = request_ids
            .into_iter()
            .filter(|request_id| self.owed.contains_key(request_id))
            .collect();
        if known.is_empty() {
            return Vec::new();
        }
        let ids: Vec<&str> = known.iter().map(String::as_str).collect();
        self.scheduler
            .finish_requests(Some(&ids), RequestStatus::FinishedAborted);
        self.emits
            .retain(|_, emit| !known.contains(&emit.output.request_id));

        let mut by_client = ClientBatches::new();
        for request_id in known {
            let Some(client_index) = self.owed.remove(&request_id) else {
                continue;
            };
            self.stats.requests_aborted += 1;
            if self.opt.log_requests {
                info!(request_id, finish_reason = "abort", "request aborted");
            }
            let (outputs, finished) = by_client.entry(client_index).or_default();
            finished.insert(request_id.clone());
            outputs.push(request_output(
                request_id,
                Vec::new(),
                Some(EngineCoreFinishReason::Abort),
            ));
        }
        self.ensure_step(Instant::now());
        if let Some(stats) = self.scheduler.make_stats(None) {
            convert::merge_stats(&mut self.pending_stats, convert::scheduler_stats(stats));
        }
        self.envelopes(by_client)
    }

    fn handle_utility(&mut self, request: UtilityRequestSpec) -> Result<EngineOutput> {
        debug!(
            engine_index = self.engine_index,
            call_id = %request.call_id,
            method = request.method_name,
            "utility request"
        );
        let client_index = request.client_index;
        let answer = match request.method_name.as_str() {
            "reset_prefix_cache" => Some(self.reset_prefix_cache()),
            method => self.loras.handle_utility(method, &request.args)?,
        };
        let outputs = match answer {
            Some(ok) => {
                utility_result_outputs(self.engine_index, request.call_id, utility_envelope(ok)?)
            }
            None => utility_response(self.engine_index, request)?,
        };
        Ok(EngineOutput {
            client_index,
            outputs,
        })
    }

    fn reset_prefix_cache(&mut self) -> bool {
        let reset = self.scheduler.reset_prefix_cache(false, false);
        if !reset {
            warn!(
                engine_index = self.engine_index,
                "refusing reset_prefix_cache: KV blocks are still in use"
            );
        }
        let events = self.scheduler.kv_cache_manager_mut().take_events();
        self.publish(events);
        reset
    }

    fn handle_control(&mut self, request: ControlRequest) -> Result<ControlReply, ControlError> {
        match request {
            ControlRequest::GetConfig => Ok(ControlReply::Config(Config::from_opt(&self.opt))),
            ControlRequest::PatchConfig(patch) => {
                self.apply_patch(*patch)?;
                Ok(ControlReply::Config(Config::from_opt(&self.opt)))
            }
            ControlRequest::GetStats => Ok(ControlReply::Stats(self.control_stats())),
            ControlRequest::ResetStats => {
                self.stats = Stats::default();
                Ok(ControlReply::Stats(self.control_stats()))
            }
        }
    }

    fn control_stats(&self) -> Stats {
        let (running, waiting) = self.scheduler.get_request_counts();
        Stats {
            running: running as u64,
            waiting: waiting as u64,
            ..self.stats
        }
    }

    fn apply_patch(&mut self, patch: ConfigPatch) -> Result<(), ControlError> {
        if patch.max_num_seqs.is_some() || patch.max_num_batched_tokens.is_some() {
            return Err(ControlError::conflict(
                "max_num_seqs and max_num_batched_tokens are fixed at startup with --use-rust-core",
            ));
        }
        let rebuild_latency = patch.changes_latency();
        if rebuild_latency && self.opt.latency_trace.is_some() {
            return Err(ControlError::conflict(
                "latency knobs cannot be patched while --latency-trace is in use",
            ));
        }
        let mut opt = self.opt.clone();
        patch.apply(&mut opt);
        if rebuild_latency {
            self.runner
                .set_latency(opt.build_latency().map_err(|error| {
                    ControlError::internal(format!("rebuilding latency model: {error:#}"))
                })?);
        }
        self.opt = opt;
        info!(
            engine_index = self.engine_index,
            config = ?Config::from_opt(&self.opt),
            "config patched via control API"
        );
        Ok(())
    }

    fn publish(&self, events: Vec<KvCacheEvent>) {
        if let Some(tx) = &self.events {
            tx.publish(events.into_iter().map(convert::kv_event).collect());
        }
    }

    /// Schedule the next step if none is executing and the scheduler has work.
    fn ensure_step(&mut self, anchor: Instant) {
        if self.in_flight.is_some() || !self.scheduler.has_requests() {
            return;
        }
        let scheduled = self.scheduler.schedule(false);
        let (num_running, _) = self.scheduler.get_request_counts();
        let StepResult { output, duration } = self.runner.execute(&scheduled, num_running as u64);
        let duration = if scheduled.total_num_scheduled_tokens > 0 {
            duration
        } else if self.scheduler.has_requests() {
            IDLE_POLL
        } else {
            Duration::ZERO
        };
        self.in_flight = Some(InFlightStep {
            end: anchor + duration,
            scheduled,
            result: output,
        });
    }

    /// Hand a finished step's results to the scheduler and queue its outputs.
    fn complete_step(&mut self, step: &InFlightStep) {
        let update = self
            .scheduler
            .update_from_output(&step.scheduled, &step.result);
        self.publish(update.kv_cache_events);

        for (client_index, client_outputs) in update.engine_core_outputs {
            if let Some(stats) = client_outputs.scheduler_stats {
                convert::merge_stats(&mut self.pending_stats, convert::scheduler_stats(stats));
            }
            for output in client_outputs.outputs {
                let mut output = convert::output(output);
                if output.finish_reason == Some(EngineCoreFinishReason::Length)
                    && let Some(recorded) = self.runner.recorded_finish_reason(&output.request_id)
                {
                    output.finish_reason = Some(recorded);
                }
                let due = step.end + self.runner.emit_offset(&output.request_id);
                self.emits.insert(
                    (due, self.emit_seq),
                    PendingEmit {
                        client_index: client_index as u32,
                        output,
                    },
                );
                self.emit_seq += 1;
            }
        }
    }

    fn step(&mut self) -> Vec<EngineOutput> {
        let now = Instant::now();
        while let Some(step) = self.in_flight.take_if(|step| step.end <= now) {
            self.complete_step(&step);
            self.ensure_step(step.end);
        }

        let mut by_client = ClientBatches::new();
        while let Some(entry) = self.emits.first_entry()
            && entry.key().0 <= now
        {
            let PendingEmit {
                client_index,
                output,
            } = entry.remove();
            let (outputs, finished) = by_client.entry(client_index).or_default();
            if let Some(finish_reason) = output.finish_reason {
                self.owed.remove(&output.request_id);
                finished.insert(output.request_id.clone());
                if finish_reason == EngineCoreFinishReason::Error {
                    self.stats.requests_failed += 1;
                } else {
                    self.stats.requests_completed += 1;
                }
                if self.opt.log_requests {
                    info!(
                        request_id = %output.request_id,
                        ?finish_reason,
                        "request finished"
                    );
                }
            }
            outputs.push(output);
        }
        self.envelopes(by_client)
    }

    /// One envelope per client, the first carrying any unsent scheduler stats.
    /// With no outputs to send, the stats go to client 0 on their own, as vLLM does,
    /// unless they repeat what was last sent.
    fn envelopes(&mut self, by_client: ClientBatches) -> Vec<EngineOutput> {
        let mut stats = self.pending_stats.take();
        if by_client.is_empty() && stats == self.last_sent_stats {
            stats = None;
        }
        if stats.is_some() {
            self.last_sent_stats.clone_from(&stats);
        }
        let mut stats = stats.map(Box::new);
        let timestamp = now_monotonic();
        if by_client.is_empty() {
            return stats
                .map(|stats| EngineOutput {
                    client_index: 0,
                    outputs: vllm::request_batch(
                        self.engine_index,
                        Vec::new(),
                        Some(stats),
                        timestamp,
                        None,
                    ),
                })
                .into_iter()
                .collect();
        }
        by_client
            .into_iter()
            .map(|(client_index, (outputs, finished))| EngineOutput {
                client_index,
                outputs: vllm::request_batch(
                    self.engine_index,
                    outputs,
                    stats.take(),
                    timestamp,
                    (!finished.is_empty()).then_some(finished),
                ),
            })
            .collect()
    }
}

impl EngineCore for RustCoreEngine {
    type Internal = ();

    fn handle_input(&mut self, input: EngineInput) -> Result<Vec<EngineOutput>> {
        Ok(match input {
            EngineInput::Request(request) => self.add_request(request),
            EngineInput::Abort(request_ids) => self.abort_requests(request_ids),
            EngineInput::Utility(request) => vec![self.handle_utility(request)?],
            EngineInput::Control(call) => {
                if call.reply.send(self.handle_control(call.request)).is_err() {
                    debug!(engine_index = self.engine_index, "control caller went away");
                }
                Vec::new()
            }
            EngineInput::StartDpWave => {
                debug!(engine_index = self.engine_index, "ignoring START_DP_WAVE");
                Vec::new()
            }
        })
    }

    fn take_internal_rx(&mut self) -> mpsc::UnboundedReceiver<()> {
        let (_tx, rx) = mpsc::unbounded_channel();
        rx
    }

    fn on_internal(&mut self, _event: ()) -> Vec<EngineOutput> {
        Vec::new()
    }

    fn earliest_deadline(&self) -> Option<Instant> {
        let step_end = self.in_flight.as_ref().map(|step| step.end);
        let emit = self.emits.keys().next().map(|&(due, _)| due);
        match (step_end, emit) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }

    fn step(&mut self) -> Vec<EngineOutput> {
        RustCoreEngine::step(self)
    }

    fn num_unfinished_requests(&self) -> usize {
        self.owed.len()
    }

    fn abort_all_requests(&mut self) -> Vec<EngineOutput> {
        let request_ids = self.owed.keys().cloned().collect();
        self.abort_requests(request_ids)
    }

    fn reject_request(&mut self, request: Box<EngineCoreRequest>) -> EngineOutput {
        self.stats.requests_received += 1;
        self.stats.requests_aborted += 1;
        debug!(
            engine_index = self.engine_index,
            request_id = request.request_id,
            "rejecting new request during shutdown"
        );
        EngineOutput {
            client_index: request.client_index,
            outputs: empty_finish_outputs(
                self.engine_index,
                request.request_id,
                EngineCoreFinishReason::Abort,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use rand::rngs::StdRng;
    use serde_json::json;
    use sim_protocol::vllm::{
        self as vllm, EngineCoreEventType, EngineCoreFinishReason, EngineCoreOutput,
        EngineCoreRequest, EngineCoreSamplingParams, SchedulerStats,
    };
    use tokio::time::Instant;
    use vllm_engine_core_client::protocol::utility::UtilityCallId;

    use crate::blockpool::KvCacheEvent;
    use crate::control::{ConfigPatch, ControlCall, ControlError, ControlReply, ControlRequest};
    use crate::dataplane::PdRole;
    use crate::engine_core::{EngineCore, EngineInput, EngineOutput, UtilityRequestSpec};
    use crate::kvevents::KvEventTx;
    use crate::latency::{FirstTokenCtx, LatencyModel};
    use crate::rust_core::RustCoreEngine;
    use crate::{KvTransferMode, Opt};

    const SLOW: u64 = 10_000;

    fn test_opt() -> Opt {
        Opt::parse_from(["play", "--use-rust-core"])
    }

    fn engine(opt: Opt) -> RustCoreEngine {
        let latency = opt.build_latency().unwrap();
        engine_with_latency(opt, latency)
    }

    fn engine_with_latency(opt: Opt, latency: Box<dyn LatencyModel>) -> RustCoreEngine {
        let token_source = opt.build_token_source().unwrap();
        RustCoreEngine::from_parts(0, opt, latency, token_source, None).unwrap()
    }

    fn prompt(seed: u32, len: usize) -> Vec<u32> {
        (0..len as u32).map(|i| seed * 10_000 + i).collect()
    }

    fn request(id: &str, prompt_token_ids: Vec<u32>, max_tokens: u32) -> EngineCoreRequest {
        EngineCoreRequest {
            request_id: id.to_string(),
            prompt_token_ids: Some(prompt_token_ids),
            sampling_params: Some(EngineCoreSamplingParams {
                max_tokens,
                ..EngineCoreSamplingParams::for_test()
            }),
            ..Default::default()
        }
    }

    fn add(engine: &mut RustCoreEngine, request: EngineCoreRequest) -> Vec<EngineOutput> {
        engine
            .handle_input(EngineInput::Request(Box::new(request)))
            .expect("handle_input")
    }

    fn outputs(batches: &[EngineOutput]) -> Vec<EngineCoreOutput> {
        batches
            .iter()
            .flat_map(|batch| vllm::request_outputs(&batch.outputs).to_vec())
            .collect()
    }

    fn stats(batches: &[EngineOutput]) -> SchedulerStats {
        batches
            .iter()
            .find_map(|batch| vllm::scheduler_stats(&batch.outputs))
            .expect("a batch carrying scheduler stats")
            .clone()
    }

    fn outputs_of(outputs: &[EngineCoreOutput], id: &str) -> Vec<EngineCoreOutput> {
        outputs
            .iter()
            .filter(|output| output.request_id == id)
            .cloned()
            .collect()
    }

    fn num_tokens(outputs: &[EngineCoreOutput]) -> usize {
        outputs
            .iter()
            .map(|output| output.new_token_ids.len())
            .sum()
    }

    fn call_utility<T: serde::de::DeserializeOwned, A: serde::Serialize>(
        engine: &mut RustCoreEngine,
        method: &str,
        args: A,
    ) -> T {
        let request = UtilityRequestSpec {
            client_index: 0,
            call_id: UtilityCallId::from(1u64),
            method_name: method.to_string(),
            args: rmpv::ext::to_value(&args).expect("encode utility args"),
        };
        let mut out = engine
            .handle_input(EngineInput::Utility(request))
            .expect("handle_input");
        vllm::utility_output(&out.remove(0).outputs)
            .expect("utility output")
            .clone()
            .into_typed_result::<T>(method)
            .expect("typed result")
    }

    fn control(
        engine: &mut RustCoreEngine,
        request: ControlRequest,
    ) -> Result<ControlReply, ControlError> {
        let (reply, mut reply_rx) = tokio::sync::oneshot::channel();
        engine
            .handle_input(EngineInput::Control(ControlCall { request, reply }))
            .expect("handle_input");
        reply_rx
            .try_recv()
            .expect("engine answers control calls inline")
    }

    #[test]
    fn a_request_generates_max_tokens_and_finishes_with_length() {
        let mut engine = engine(test_opt());
        assert!(add(&mut engine, request("r1", prompt(1, 4), 3)).is_empty());
        assert!(engine.earliest_deadline().unwrap() <= Instant::now());

        let outputs = outputs(&engine.step());
        assert_eq!(num_tokens(&outputs), 3);
        assert_eq!(outputs.len(), 3);
        assert_eq!(
            outputs.last().unwrap().finish_reason,
            Some(EngineCoreFinishReason::Length)
        );
        assert!(
            outputs[..2]
                .iter()
                .all(|output| output.finish_reason.is_none())
        );
        assert_eq!(engine.num_unfinished_requests(), 0);
        assert!(engine.earliest_deadline().is_none());
    }

    #[test]
    fn the_first_output_carries_lifecycle_events_and_prefill_stats() {
        let mut engine = engine(test_opt());
        add(&mut engine, request("r1", prompt(1, 40), 2));
        let outputs = outputs(&engine.step());

        let events: Vec<_> = outputs[0]
            .events
            .as_ref()
            .expect("events on the first output")
            .iter()
            .map(|event| event.r#type)
            .collect();
        assert_eq!(
            events,
            [EngineCoreEventType::Queued, EngineCoreEventType::Scheduled]
        );
        let prefill = outputs[0].prefill_stats.as_ref().expect("prefill stats");
        assert_eq!(prefill.num_prompt_tokens, 40);
        assert_eq!(prefill.num_computed_tokens, 40);
        assert_eq!(prefill.num_cached_tokens, 0);
        assert!(outputs[1].prefill_stats.is_none());
        assert!(outputs[1].events.is_none());
    }

    #[test]
    fn the_running_batch_is_capped_at_max_num_seqs() {
        let mut opt = test_opt();
        opt.max_num_seqs = 2;
        opt.inter_token_latency = SLOW;
        let mut engine = engine(opt);
        for id in ["a", "b", "c"] {
            add(&mut engine, request(id, prompt(1, 8), 4));
        }

        let batches = engine.step();
        assert_eq!(outputs(&batches).len(), 1);
        assert!(stats(&batches).kv_cache_usage > 0.0);
        let next_step = engine.in_flight.as_ref().expect("a scheduled step");
        let scheduled: Vec<_> = next_step.scheduled.num_scheduled_tokens.keys().collect();
        assert_eq!(scheduled, ["a", "b"]);
        assert_eq!(engine.scheduler.get_request_counts(), (2, 1));
        assert_eq!(engine.num_unfinished_requests(), 3);
    }

    #[test]
    fn a_long_prompt_is_chunked_under_the_step_token_budget() {
        let mut opt = test_opt();
        opt.max_num_batched_tokens = 128;
        let mut engine = engine(opt);
        add(&mut engine, request("a", prompt(1, 300), 2));

        let first_step = engine.in_flight.as_ref().expect("a scheduled step");
        assert_eq!(first_step.scheduled.num_scheduled_tokens["a"], 128);
        assert!(first_step.result.sampled_token_ids[0].is_empty());

        let outputs = outputs(&engine.step());
        assert_eq!(num_tokens(&outputs), 2);
        assert_eq!(
            outputs[0].prefill_stats.as_ref().unwrap().num_prompt_tokens,
            300
        );
    }

    #[test]
    fn a_shared_prompt_prefix_is_served_from_the_prefix_cache() {
        let mut engine = engine(test_opt());
        add(&mut engine, request("first", prompt(1, 200), 1));
        engine.step();

        let mut shared = prompt(1, 160);
        shared.extend(prompt(2, 40));
        add(&mut engine, request("second", shared, 1));
        let batches = engine.step();
        let prefill = outputs(&batches)[0].prefill_stats.clone().unwrap();
        assert_eq!(prefill.num_local_cached_tokens, 160);
        assert_eq!(prefill.num_cached_tokens, 160);
        assert_eq!(prefill.num_computed_tokens, 40);
        assert_eq!(stats(&batches).prefix_cache_stats.base.hits, 160);
    }

    #[test]
    fn cache_salt_and_lora_isolate_identical_prompts() {
        let mut engine = engine(test_opt());
        add(&mut engine, request("plain", prompt(1, 70), 1));
        engine.step();

        let salted = EngineCoreRequest {
            cache_salt: Some("tenant-a".to_string()),
            ..request("salted", prompt(1, 70), 1)
        };
        add(&mut engine, salted);
        let prefill = outputs(&engine.step())[0].prefill_stats.clone().unwrap();
        assert_eq!(prefill.num_local_cached_tokens, 0);

        let adapted = EngineCoreRequest {
            lora_request: Some(vllm::lora_request("adapter", 1, "/loras/adapter")),
            ..request("adapted", prompt(1, 70), 1)
        };
        add(&mut engine, adapted);
        let prefill = outputs(&engine.step())[0].prefill_stats.clone().unwrap();
        assert_eq!(prefill.num_local_cached_tokens, 0);

        add(&mut engine, request("plain-again", prompt(1, 70), 1));
        let prefill = outputs(&engine.step())[0].prefill_stats.clone().unwrap();
        assert_eq!(prefill.num_local_cached_tokens, 64);
    }

    #[test]
    fn a_fully_cached_prompt_still_recomputes_its_last_block() {
        let mut engine = engine(test_opt());
        add(&mut engine, request("first", prompt(1, 64), 1));
        engine.step();

        add(&mut engine, request("again", prompt(1, 64), 1));
        let prefill = outputs(&engine.step())[0].prefill_stats.clone().unwrap();
        assert_eq!(prefill.num_local_cached_tokens, 48);
        assert_eq!(prefill.num_computed_tokens, 16);
    }

    #[test]
    fn an_eos_token_from_the_worker_stops_the_request() {
        let mut opt = test_opt();
        opt.vocab_size = 1;
        let mut engine = engine(opt);
        let mut eos = request("r1", prompt(1, 4), 50);
        eos.sampling_params.as_mut().unwrap().eos_token_id = Some(0);
        add(&mut engine, eos);

        let outputs = outputs(&engine.step());
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].new_token_ids, [0]);
        assert_eq!(outputs[0].finish_reason, Some(EngineCoreFinishReason::Stop));
    }

    /// Instant model that caps every EOS-terminable request at a fixed length.
    struct CappingLatency(usize);

    impl LatencyModel for CappingLatency {
        fn first_token_delay(&self, _rng: &mut StdRng, _ctx: &FirstTokenCtx) -> Duration {
            Duration::ZERO
        }
        fn inter_token_delay(&self, _rng: &mut StdRng, _num_running: u64) -> Duration {
            Duration::ZERO
        }
        fn sample_output_len(&self, _rng: &mut StdRng) -> Option<usize> {
            Some(self.0)
        }
    }

    #[test]
    fn a_sampled_output_length_caps_only_eos_terminable_requests() {
        let mut engine = engine_with_latency(test_opt(), Box::new(CappingLatency(5)));
        let mut eos = request("eos", prompt(1, 4), 100);
        eos.sampling_params.as_mut().unwrap().eos_token_id = Some(u32::MAX);
        add(&mut engine, eos);
        add(&mut engine, request("bounded", prompt(2, 4), 9));

        let outputs = outputs(&engine.step());
        assert_eq!(num_tokens(&outputs_of(&outputs, "eos")), 5);
        assert_eq!(num_tokens(&outputs_of(&outputs, "bounded")), 9);
    }

    #[test]
    fn time_to_first_token_delays_emission_but_not_the_step_clock() {
        let mut opt = test_opt();
        opt.time_to_first_token = SLOW;
        let mut engine = engine(opt);
        let before = Instant::now();
        add(&mut engine, request("r1", prompt(1, 4), 2));

        let batches = engine.step();
        assert!(outputs(&batches).is_empty());
        assert_eq!(stats(&batches).num_running_reqs, 0);
        assert_eq!(engine.num_unfinished_requests(), 1);
        let deadline = engine.earliest_deadline().expect("a pending emission");
        assert!(deadline >= before + Duration::from_millis(SLOW - 1_000));
    }

    #[test]
    fn aborting_a_running_request_frees_its_slot_and_reports_post_abort_stats() {
        let mut opt = test_opt();
        opt.max_num_seqs = 1;
        opt.inter_token_latency = SLOW;
        let mut engine = engine(opt);
        add(&mut engine, request("a", prompt(1, 8), 4));
        add(&mut engine, request("b", prompt(2, 8), 4));
        engine.step();

        let batches = engine
            .handle_input(EngineInput::Abort(vec![
                "a".to_string(),
                "nope".to_string(),
            ]))
            .unwrap();
        let outputs = outputs(&batches);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].request_id, "a");
        assert_eq!(
            outputs[0].finish_reason,
            Some(EngineCoreFinishReason::Abort)
        );
        let stats = stats(&batches);
        assert_eq!(stats.num_running_reqs, 0);
        assert_eq!(stats.num_waiting_reqs, 1);
        assert_eq!(engine.num_unfinished_requests(), 1);
    }

    #[test]
    fn aborting_a_request_drops_its_delayed_outputs() {
        let mut opt = test_opt();
        opt.time_to_first_token = SLOW;
        let mut engine = engine(opt);
        add(&mut engine, request("r1", prompt(1, 4), 2));
        engine.step();
        assert!(!engine.emits.is_empty());

        let batches = engine.abort_all_requests();
        assert_eq!(
            outputs(&batches)[0].finish_reason,
            Some(EngineCoreFinishReason::Abort)
        );
        assert!(engine.emits.is_empty());
        assert_eq!(engine.num_unfinished_requests(), 0);
        engine.step();
        assert!(engine.earliest_deadline().is_none());
    }

    #[test]
    fn abort_all_covers_running_and_waiting_requests() {
        let mut opt = test_opt();
        opt.max_num_seqs = 1;
        opt.inter_token_latency = SLOW;
        let mut engine = engine(opt);
        add(&mut engine, request("a", prompt(1, 8), 4));
        add(&mut engine, request("b", prompt(2, 8), 4));
        engine.step();

        let outputs = outputs(&engine.abort_all_requests());
        let mut aborted: Vec<_> = outputs.iter().map(|o| o.request_id.as_str()).collect();
        aborted.sort_unstable();
        assert_eq!(aborted, ["a", "b"]);
        assert_eq!(engine.num_unfinished_requests(), 0);
    }

    #[test]
    fn requests_that_cannot_run_finish_on_arrival() {
        let mut opt = test_opt();
        opt.inter_token_latency = SLOW;
        let mut engine = engine(opt);
        add(&mut engine, request("dup", prompt(1, 4), 4));

        let duplicate = outputs(&add(&mut engine, request("dup", prompt(1, 4), 4)));
        assert_eq!(
            duplicate[0].finish_reason,
            Some(EngineCoreFinishReason::Error)
        );

        let empty = outputs(&add(&mut engine, request("zero", prompt(1, 4), 0)));
        assert_eq!(empty[0].finish_reason, Some(EngineCoreFinishReason::Length));

        let mut no_params = request("bare", prompt(1, 4), 4);
        no_params.sampling_params = None;
        let bare = outputs(&add(&mut engine, no_params));
        assert_eq!(bare[0].finish_reason, Some(EngineCoreFinishReason::Error));

        assert_eq!(engine.num_unfinished_requests(), 1);
    }

    #[test]
    fn failure_injection_and_the_context_limit_fail_requests_on_arrival() {
        let mut opt = test_opt();
        opt.max_model_len = 16;
        let mut engine = engine(opt);
        let too_long = outputs(&add(&mut engine, request("long", prompt(1, 12), 8)));
        assert_eq!(
            too_long[0].finish_reason,
            Some(EngineCoreFinishReason::Length)
        );

        let mut opt = test_opt();
        opt.failure_injection_rate = 1.0;
        let mut engine = self::engine(opt);
        let injected = outputs(&add(&mut engine, request("r1", prompt(1, 4), 4)));
        assert_eq!(
            injected[0].finish_reason,
            Some(EngineCoreFinishReason::Error)
        );
        assert_eq!(engine.num_unfinished_requests(), 0);
    }

    #[test]
    fn a_preempted_request_resumes_and_generates_every_token() {
        let mut opt = test_opt();
        opt.kv_cache_size = 8;
        let mut engine = engine(opt);
        add(&mut engine, request("g", prompt(1, 48), 40));
        add(&mut engine, request("h", prompt(2, 48), 40));

        let outputs = outputs(&engine.step());
        for id in ["g", "h"] {
            let request_outputs = outputs_of(&outputs, id);
            assert_eq!(num_tokens(&request_outputs), 40, "{id}");
            assert_eq!(
                request_outputs.last().unwrap().finish_reason,
                Some(EngineCoreFinishReason::Length)
            );
        }
        let preempted = outputs
            .iter()
            .flat_map(|output| output.events.iter().flatten())
            .filter(|event| event.r#type == EngineCoreEventType::Preempted)
            .count();
        assert!(preempted > 0);
    }

    #[test]
    fn reset_prefix_cache_is_refused_while_blocks_are_in_use() {
        let mut opt = test_opt();
        opt.inter_token_latency = SLOW;
        let mut engine = engine(opt);
        add(&mut engine, request("r1", prompt(1, 64), 4));
        engine.step();
        assert!(!call_utility::<bool, _>(
            &mut engine,
            "reset_prefix_cache",
            ()
        ));

        engine.abort_all_requests();
        engine.step();
        assert!(call_utility::<bool, _>(
            &mut engine,
            "reset_prefix_cache",
            ()
        ));

        let mut opt = test_opt();
        opt.inter_token_latency = SLOW;
        let mut engine = self::engine(opt);
        add(&mut engine, request("r2", prompt(1, 64), 1));
        let prefill = outputs(&engine.step())[0].prefill_stats.clone().unwrap();
        assert_eq!(prefill.num_local_cached_tokens, 0);
    }

    #[test]
    fn prefix_cache_changes_are_published_as_kv_events() {
        let mut opt = test_opt();
        opt.enable_kv_cache_events = true;
        let latency = opt.build_latency().unwrap();
        let token_source = opt.build_token_source().unwrap();
        let (tx, mut rx) = KvEventTx::channel();
        let mut engine =
            RustCoreEngine::from_parts(0, opt, latency, token_source, Some(tx)).unwrap();

        let prompt_token_ids = prompt(1, 40);
        add(&mut engine, request("r1", prompt_token_ids.clone(), 1));
        engine.step();

        let mut stored_hashes = Vec::new();
        let mut stored_tokens = Vec::new();
        while let Ok(batch) = rx.try_recv() {
            for event in batch {
                let KvCacheEvent::Stored {
                    block_hashes,
                    token_ids,
                    block_size,
                    lora_name,
                    ..
                } = event
                else {
                    panic!("expected only stored blocks, got {event:?}");
                };
                assert_eq!(block_size, 16);
                assert_eq!(lora_name, None);
                stored_hashes.extend(block_hashes);
                stored_tokens.extend(token_ids);
            }
        }
        assert_eq!(stored_hashes.len(), 2);
        assert_ne!(stored_hashes[0], stored_hashes[1]);
        assert_eq!(stored_tokens, prompt_token_ids[..32]);

        assert!(call_utility::<bool, _>(
            &mut engine,
            "reset_prefix_cache",
            ()
        ));
        assert_eq!(rx.try_recv().unwrap(), [KvCacheEvent::AllCleared]);
    }

    #[test]
    fn lora_utilities_report_whether_the_adapter_was_loaded() {
        let mut engine = engine(test_opt());
        let adapter = vllm::lora_request("adapter", 7, "/loras/adapter");
        assert!(call_utility::<bool, _>(&mut engine, "add_lora", (adapter,)));
        assert!(call_utility::<bool, _>(&mut engine, "remove_lora", (7u64,)));
        assert!(!call_utility::<bool, _>(
            &mut engine,
            "remove_lora",
            (7u64,)
        ));
    }

    #[test]
    fn control_reports_counters_and_refuses_scheduler_cap_patches() {
        let mut opt = test_opt();
        opt.max_num_seqs = 1;
        opt.inter_token_latency = SLOW;
        let mut engine = engine(opt);
        add(&mut engine, request("a", prompt(1, 8), 1));
        add(&mut engine, request("b", prompt(2, 8), 4));
        add(&mut engine, request("c", prompt(3, 8), 4));
        add(&mut engine, request("zero", prompt(4, 8), 0));
        engine.step();

        let Ok(ControlReply::Stats(stats)) = control(&mut engine, ControlRequest::GetStats) else {
            panic!("expected stats");
        };
        assert_eq!(stats.requests_received, 4);
        assert_eq!(stats.requests_completed, 1);
        assert_eq!(stats.requests_failed, 1);
        assert_eq!(stats.running, 1);
        assert_eq!(stats.waiting, 1);

        let caps = ConfigPatch {
            max_num_seqs: Some(4),
            ..Default::default()
        };
        let refused = control(&mut engine, ControlRequest::PatchConfig(Box::new(caps)));
        assert_eq!(refused.unwrap_err().status, 409);

        let latency = ConfigPatch {
            inter_token_latency: Some(5),
            ..Default::default()
        };
        let Ok(ControlReply::Config(config)) =
            control(&mut engine, ControlRequest::PatchConfig(Box::new(latency)))
        else {
            panic!("expected config");
        };
        assert_eq!(config.inter_token_latency, 5);
    }

    fn kv_request(
        id: &str,
        prompt_token_ids: Vec<u32>,
        max_tokens: u32,
        kv_transfer_params: serde_json::Value,
    ) -> EngineCoreRequest {
        let mut request = request(id, prompt_token_ids, max_tokens);
        request.sampling_params.as_mut().unwrap().extra_args = Some(HashMap::from([(
            "kv_transfer_params".to_string(),
            kv_transfer_params,
        )]));
        request
    }

    /// Step until every request has sent its final output and the scheduler has
    /// nothing left to do, sleeping to each deadline.
    fn run_to_idle(engine: &mut RustCoreEngine) -> Vec<EngineOutput> {
        let give_up = std::time::Instant::now() + Duration::from_secs(10);
        let mut batches = Vec::new();
        while let Some(deadline) = engine.earliest_deadline() {
            assert!(
                std::time::Instant::now() < give_up,
                "engine never went idle"
            );
            std::thread::sleep(deadline.saturating_duration_since(Instant::now()));
            batches.extend(engine.step());
        }
        assert_eq!(engine.num_unfinished_requests(), 0);
        batches
    }

    /// Run a prefill request on its own engine and return the `kv_transfer_params`
    /// its final output hands to the decoder.
    fn prefill(prompt_token_ids: Vec<u32>) -> serde_json::Value {
        let mut opt = test_opt();
        opt.pd_role = PdRole::Prefill;
        opt.engine_id = "prefill-0".to_string();
        let mut engine = engine(opt);
        add(
            &mut engine,
            kv_request(
                "p-1",
                prompt_token_ids,
                1,
                json!({"do_remote_decode": true, "do_remote_prefill": false}),
            ),
        );
        let outputs = outputs(&run_to_idle(&mut engine));
        assert!(!engine.scheduler.has_requests());
        assert_eq!(engine.scheduler.get_kv_cache_usage(), 0.0);
        let params = outputs
            .last()
            .and_then(|output| output.kv_transfer_params.clone())
            .expect("the final prefill output carries kv_transfer_params");
        *params
    }

    #[test]
    fn a_finished_prefill_hands_its_blocks_to_the_decoder() {
        let params = prefill(prompt(1, 40));
        assert_eq!(params["do_remote_prefill"], true);
        assert_eq!(params["do_remote_decode"], false);
        assert_eq!(params["remote_engine_id"], "prefill-0");
        assert_eq!(params["remote_request_id"], "p-1");
        assert_eq!(params["remote_host"], "127.0.0.1");
        assert_eq!(params["remote_port"], 5600);
        assert_eq!(params["remote_num_tokens"], 40);
        assert_eq!(params["transfer_mode"], "pull");
        let groups = params["remote_block_ids"].as_array().unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].as_array().unwrap().len(), 3);
    }

    #[test]
    fn a_prefill_request_with_null_remote_fields_is_accepted() {
        let mut opt = test_opt();
        opt.pd_role = PdRole::Prefill;
        let mut engine = engine(opt);
        let from_sidecar = json!({
            "do_remote_decode": true,
            "do_remote_prefill": false,
            "remote_engine_id": null,
            "remote_block_ids": null,
            "remote_host": null,
            "remote_port": null,
        });
        add(
            &mut engine,
            kv_request("p-1", prompt(1, 40), 1, from_sidecar),
        );
        let outputs = outputs(&run_to_idle(&mut engine));
        assert_eq!(
            outputs.last().unwrap().finish_reason,
            Some(EngineCoreFinishReason::Length)
        );
        let params = outputs.last().unwrap().kv_transfer_params.clone().unwrap();
        assert_eq!(params["remote_num_tokens"], 40);
    }

    #[test]
    fn a_decode_request_loads_its_prompt_from_the_prefill_engine() {
        let params = prefill(prompt(1, 40));
        let mut opt = test_opt();
        opt.pd_role = PdRole::Decode;
        let mut engine = engine(opt);
        add(&mut engine, kv_request("d-1", prompt(1, 40), 3, params));
        assert_eq!(engine.scheduler.get_request_counts(), (0, 1));

        let batches = run_to_idle(&mut engine);
        let connector_hits: u64 = batches
            .iter()
            .filter_map(|batch| vllm::scheduler_stats(&batch.outputs))
            .filter_map(|stats| stats.connector_prefix_cache_stats.as_ref())
            .map(|connector| connector.base.hits)
            .sum();
        assert_eq!(connector_hits, 40);
        let outputs = outputs(&batches);
        assert_eq!(num_tokens(&outputs), 3);
        let prefill = outputs[0].prefill_stats.clone().unwrap();
        assert_eq!(prefill.num_prompt_tokens, 40);
        assert_eq!(prefill.num_external_cached_tokens, 40);
        assert_eq!(prefill.num_local_cached_tokens, 0);
        assert_eq!(prefill.num_computed_tokens, 0);
    }

    #[test]
    fn a_decode_request_waits_out_the_modeled_transfer_time() {
        let params = prefill(prompt(1, 40));
        let mut opt = test_opt();
        opt.pd_role = PdRole::Decode;
        opt.kv_cache_transfer_latency = 150;
        let mut engine = engine(opt);
        let started = std::time::Instant::now();
        add(&mut engine, kv_request("d-1", prompt(1, 40), 1, params));

        let outputs = outputs(&run_to_idle(&mut engine));
        assert_eq!(num_tokens(&outputs), 1);
        assert!(started.elapsed() >= Duration::from_millis(150));
    }

    #[test]
    fn a_failed_remote_load_is_recomputed_locally() {
        let mut params = prefill(prompt(1, 40));
        params["remote_block_ids"] = json!([[7]]);
        let mut opt = test_opt();
        opt.pd_role = PdRole::Decode;
        let mut engine = engine(opt);
        add(&mut engine, kv_request("d-1", prompt(1, 40), 2, params));

        let outputs = outputs(&run_to_idle(&mut engine));
        assert_eq!(num_tokens(&outputs), 2);
        let prefill = outputs[0].prefill_stats.clone().unwrap();
        assert_eq!(prefill.num_external_cached_tokens, 0);
        assert_eq!(prefill.num_computed_tokens, 40);
    }

    #[test]
    fn a_decode_request_with_a_local_prefix_hit_loads_only_the_rest() {
        let params = prefill(prompt(1, 80));
        let mut opt = test_opt();
        opt.pd_role = PdRole::Decode;
        let mut engine = engine(opt);
        add(&mut engine, request("warm", prompt(1, 32), 1));
        run_to_idle(&mut engine);

        add(&mut engine, kv_request("d-1", prompt(1, 80), 1, params));
        let outputs = outputs(&run_to_idle(&mut engine));
        let prefill = outputs[0].prefill_stats.clone().unwrap();
        assert_eq!(prefill.num_local_cached_tokens, 32);
        assert_eq!(prefill.num_external_cached_tokens, 48);
        assert_eq!(prefill.num_computed_tokens, 0);
    }

    #[test]
    fn push_mode_hands_off_without_the_decoder_reading() {
        let mut opt = test_opt();
        opt.pd_role = PdRole::Prefill;
        opt.kv_transfer_mode = KvTransferMode::Push;
        opt.engine_id = "prefill-0".to_string();
        let mut prefill = engine(opt);
        add(
            &mut prefill,
            kv_request(
                "cmpl-7-0123abcd",
                prompt(1, 40),
                1,
                json!({"do_remote_decode": true}),
            ),
        );
        let outputs = outputs(&run_to_idle(&mut prefill));
        let params = outputs.last().unwrap().kv_transfer_params.clone().unwrap();
        assert_eq!(params["transfer_mode"], "push");
        assert_eq!(params["do_remote_prefill"], true);
        assert!(params.get("remote_blocks_expiry_time").is_none());
        assert!(!prefill.scheduler.has_requests());

        let mut opt = test_opt();
        opt.pd_role = PdRole::Decode;
        opt.kv_transfer_mode = KvTransferMode::Push;
        let mut decode = engine(opt);
        add(
            &mut decode,
            kv_request(
                "cmpl-7-89abcdef",
                prompt(1, 40),
                2,
                json!({
                    "do_remote_prefill": true,
                    "remote_engine_id": "prefill-0",
                    "remote_request_id": "cmpl-7-0123abcd",
                    "remote_host": "127.0.0.1",
                    "remote_port": 5600,
                    "tp_size": 1,
                }),
            ),
        );
        let outputs = self::outputs(&run_to_idle(&mut decode));
        assert_eq!(num_tokens(&outputs), 2);
        let prefill_stats = outputs[0].prefill_stats.clone().unwrap();
        assert_eq!(prefill_stats.num_external_cached_tokens, 40);
        assert_eq!(prefill_stats.num_computed_tokens, 0);
    }

    #[test]
    fn a_push_mode_decode_request_must_name_its_prefill_engine() {
        let mut opt = test_opt();
        opt.pd_role = PdRole::Decode;
        opt.kv_transfer_mode = KvTransferMode::Push;
        let mut engine = engine(opt);
        let incomplete = kv_request(
            "d-1",
            prompt(1, 40),
            1,
            json!({"do_remote_prefill": true, "remote_engine_id": "prefill-0"}),
        );
        let outputs = outputs(&add(&mut engine, incomplete));
        assert_eq!(
            outputs[0].finish_reason,
            Some(EngineCoreFinishReason::Error)
        );
        assert_eq!(engine.num_unfinished_requests(), 0);
    }

    #[test]
    fn push_mode_is_refused_without_the_rust_core() {
        let opt = Opt::parse_from(["play", "--kv-transfer-mode", "push"]);
        let latency = opt.build_latency().unwrap();
        assert!(opt.validate_engine(latency.as_ref()).is_err());
    }

    #[test]
    fn unusable_kv_transfer_params_fail_the_request_on_arrival() {
        let mut engine = engine(test_opt());
        let bad = kv_request(
            "d-1",
            prompt(1, 40),
            1,
            json!({"do_remote_prefill": true, "remote_port": "not-a-port"}),
        );
        let outputs = outputs(&add(&mut engine, bad));
        assert_eq!(
            outputs[0].finish_reason,
            Some(EngineCoreFinishReason::Error)
        );
        assert_eq!(engine.num_unfinished_requests(), 0);
    }

    #[test]
    fn unsupported_options_are_rejected_at_startup() {
        let cases: [fn(&mut Opt); 1] = [|opt| opt.output_token_chunk_size = 2];
        for unsupported in cases {
            let mut opt = test_opt();
            unsupported(&mut opt);
            let latency = opt.build_latency().unwrap();
            assert!(opt.validate_engine(latency.as_ref()).is_err());
        }
        let opt = test_opt();
        let latency = opt.build_latency().unwrap();
        assert!(opt.validate_engine(latency.as_ref()).is_ok());
    }
}
