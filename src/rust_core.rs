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
mod recorder;
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
use sim_trace::steps::{KvTransferTable, StepTable, read_step_trace_file};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};
use vllm_engine_core::hash::RequestBlockHasher;
use vllm_engine_core::kv_cache::KvCacheEvent;
use vllm_engine_core::sched::output::SchedulerOutput;
use vllm_engine_core::sched::request::{Request, RequestStatus};
use vllm_engine_core::sched::scheduler::{
    AcceptAllStructuredOutput, DraftTokenIds, ModelRunnerOutput, Scheduler,
};
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
use crate::lora::LoraRegistry;
use crate::rust_core::nixl::NixlWorker;
use crate::rust_core::recorder::StepRecorder;
use crate::rust_core::runner::{OutputLimits, SimRunner, Sources, StepCosts, StepResult};

/// How often a step that could schedule nothing is retried, as in vLLM's engine loop.
const IDLE_POLL: Duration = Duration::from_millis(1);

/// Reject `play` options the scheduler-backed engine does not implement.
pub(crate) fn validate(opt: &Opt) -> Result<()> {
    if opt.output_token_chunk_size != 1 {
        bail!(
            "--use-rust-core frames output by its replayed steps; \
             --output-token-chunk-size must be 1"
        );
    }
    if opt.step_trace_out.is_some() && opt.engine_count > 1 {
        bail!("--step-trace-out writes one engine's steps; --engine-count must be 1");
    }
    Ok(())
}

/// The recorded step and KV load times `--step-trace` names, if it is set.
pub(crate) fn step_costs(opt: &Opt, engine_index: u32) -> Result<Option<StepCosts>> {
    let Some(uri) = &opt.step_trace else {
        return Ok(None);
    };
    let trace = read_step_trace_file(crate::local_input(uri, "--step-trace")?)?;
    let steps =
        StepTable::from_records(&trace.steps).with_context(|| format!("--step-trace {uri}"))?;
    let kv_loads = KvTransferTable::from_records(&trace.kv_transfers);
    info!(
        engine_index,
        steps = trace.steps.len(),
        shapes = steps.num_shapes(),
        kv_transfers = trace.kv_transfers.len(),
        "pricing steps from a recorded step trace"
    );
    Ok(Some(StepCosts {
        steps,
        kv_loads,
        rng: StdRng::seed_from_u64(
            opt.seed ^ (u64::from(engine_index) + 1).wrapping_mul(0x85eb_ca6b),
        ),
    }))
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
    drafts: DraftTokenIds,
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
    recorder: Option<StepRecorder>,
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
        let sources = Sources {
            latency: opt.build_latency()?,
            token_source: opt.build_token_source()?,
            step_source: opt.build_step_source()?,
            step_costs: step_costs(&opt, engine_index)?,
        };
        Self::from_parts(engine_index, opt, sources, events)
    }

    fn from_parts(
        engine_index: u32,
        opt: Opt,
        sources: Sources,
        events: Option<KvEventTx>,
    ) -> Result<Self> {
        validate(&opt)?;
        let recorder = opt
            .step_trace_out
            .as_deref()
            .map(|path| StepRecorder::create(path, &opt))
            .transpose()?;
        let runner = SimRunner::new(
            sources,
            NixlWorker::new(&opt, engine_index),
            opt.tokens_per_block,
            opt.max_num_batched_tokens.max(1) as usize,
            opt.time_scale,
        );
        let nixl = nixl::scheduler_half(&opt, engine_index)?;
        let scheduler = Scheduler::new(
            convert::scheduler_config(&opt, runner.spec_tokens()),
            convert::kv_cache_config(&opt),
            Some(Box::new(nixl.clone())),
            Box::new(AcceptAllStructuredOutput),
        )
        .map_err(|error| anyhow!("building the vLLM scheduler: {error}"))?;
        Ok(RustCoreEngine {
            engine_index,
            scheduler,
            nixl,
            runner,
            recorder,
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
        let result = self.runner.execute(&scheduled, num_running as u64);
        if let Some(recorder) = &mut self.recorder {
            recorder.record(anchor, &result);
        }
        let StepResult {
            output,
            drafts,
            duration,
            ..
        } = result;
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
            drafts,
        });
    }

    /// Hand a finished step's results to the scheduler and queue its outputs.
    fn complete_step(&mut self, step: InFlightStep) {
        let update = self
            .scheduler
            .update_from_output(&step.scheduled, &step.result);
        self.scheduler.update_draft_token_ids(step.drafts);
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
            let end = step.end;
            self.complete_step(step);
            self.ensure_step(end);
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

    use rand::SeedableRng as _;
    use rand::rngs::StdRng;
    use serde_json::json;
    use sim_protocol::vllm::{
        self as vllm, EngineCoreEventType, EngineCoreFinishReason, EngineCoreOutput,
        EngineCoreRequest, EngineCoreSamplingParams, SchedulerStats,
    };
    use sim_trace::steps::{
        DecodeWork, KvDirection, KvOutcome, KvTransferRecord, KvTransferTable, PrefillChunk,
        StepRecord, StepShape, StepTable, read_step_trace_file,
    };
    use tokio::time::Instant;
    use vllm_engine_core_client::protocol::stats::SpecDecodingStats;
    use vllm_engine_core_client::protocol::utility::UtilityCallId;

    use crate::blockpool::KvCacheEvent;
    use crate::control::{ConfigPatch, ControlCall, ControlError, ControlReply, ControlRequest};
    use crate::dataplane::PdRole;
    use crate::engine_core::{EngineCore, EngineInput, EngineOutput, UtilityRequestSpec};
    use crate::kvevents::KvEventTx;
    use crate::latency::{FirstTokenCtx, LatencyModel, TraceLatency};
    use crate::replay_steps::{IndexSteps, StepSource};
    use crate::rust_core::RustCoreEngine;
    use crate::rust_core::runner::{Sources, StepCosts};
    use crate::tokens::ReplayTokens;
    use crate::trace::TraceRecord;
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
        engine_with_sources(opt, latency, None)
    }

    fn engine_with_sources(
        opt: Opt,
        latency: Box<dyn LatencyModel>,
        step_source: Option<Box<dyn StepSource>>,
    ) -> RustCoreEngine {
        let sources = Sources {
            latency,
            token_source: opt.build_token_source().unwrap(),
            step_source,
            step_costs: None,
        };
        RustCoreEngine::from_parts(0, opt, sources, None).unwrap()
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
        let sources = Sources {
            latency: opt.build_latency().unwrap(),
            token_source: opt.build_token_source().unwrap(),
            step_source: None,
            step_costs: None,
        };
        let (tx, mut rx) = KvEventTx::channel();
        let mut engine = RustCoreEngine::from_parts(0, opt, sources, Some(tx)).unwrap();

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

    fn recorded(output_tokens: usize, gaps_ms: &[f64], bursts: &[u32]) -> Box<dyn StepSource> {
        Box::new(IndexSteps::from_records(vec![TraceRecord {
            prompt_tokens: 8,
            output_tokens,
            ttft_ms: 1.0,
            itl_ms: Some(gaps_ms.to_vec()),
            itl_tokens: Some(bursts.to_vec()),
            ..Default::default()
        }]))
    }

    fn chunk_sizes(outputs: &[EngineCoreOutput]) -> Vec<usize> {
        outputs
            .iter()
            .map(|output| output.new_token_ids.len())
            .filter(|&len| len > 0)
            .collect()
    }

    #[test]
    fn recorded_bursts_are_delivered_as_accepted_drafts() {
        let opt = test_opt();
        let latency = opt.build_latency().unwrap();
        let steps = recorded(9, &[0.0; 3], &[3, 1, 4]);
        let mut engine = engine_with_sources(opt, latency, Some(steps));
        add(&mut engine, request("replay-0", prompt(1, 8), 50));

        let batches = engine.step();
        let outputs = outputs(&batches);
        assert_eq!(chunk_sizes(&outputs), [1, 3, 1, 4]);
        assert_eq!(
            outputs.last().unwrap().finish_reason,
            Some(EngineCoreFinishReason::Length)
        );
        assert_eq!(engine.num_unfinished_requests(), 0);

        let spec = stats(&batches).spec_decoding_stats.expect("spec stats");
        assert_eq!(spec.num_spec_tokens, 3);
        assert_eq!(spec.num_drafts, 3);
        assert_eq!(spec.num_draft_tokens, 9);
        assert_eq!(spec.num_accepted_tokens, 5);
        assert_eq!(spec.num_accepted_tokens_per_pos, [2, 2, 1]);
    }

    #[test]
    fn a_recorded_first_block_is_delivered_whole_one_step_after_the_prefill() {
        let opt = test_opt();
        let latency = opt.build_latency().unwrap();
        let steps = recorded(24, &[0.0; 2], &[8, 8]);
        let mut engine = engine_with_sources(opt, latency, Some(steps));
        add(&mut engine, request("replay-0", prompt(1, 40), 24));

        let prefill = engine.in_flight.as_ref().expect("the prefill step");
        assert_eq!(prefill.scheduled.num_scheduled_tokens["replay-0"], 40);
        assert!(prefill.result.sampled_token_ids[0].is_empty());
        assert_eq!(prefill.drafts.draft_token_ids[0].len(), 7);

        let batches = engine.step();
        let outputs = outputs(&batches);
        assert_eq!(chunk_sizes(&outputs), [8, 8, 8]);
        assert_eq!(
            outputs.last().unwrap().finish_reason,
            Some(EngineCoreFinishReason::Length)
        );
        let spec = stats(&batches).spec_decoding_stats.expect("spec stats");
        assert_eq!(spec.num_spec_tokens, 7);
        assert_eq!(spec.num_drafts, 3);
        assert_eq!(spec.num_draft_tokens, 21);
        assert_eq!(spec.num_accepted_tokens, 21);
    }

    #[test]
    fn a_recorded_gap_sets_the_step_clock() {
        let opt = test_opt();
        let latency = opt.build_latency().unwrap();
        let steps = recorded(5, &[SLOW as f64], &[4]);
        let mut engine = engine_with_sources(opt, latency, Some(steps));
        let before = Instant::now();
        add(&mut engine, request("replay-0", prompt(1, 8), 5));

        assert_eq!(chunk_sizes(&outputs(&engine.step())), [1]);
        let burst = engine.in_flight.as_ref().expect("the recorded decode step");
        assert_eq!(burst.scheduled.num_scheduled_tokens["replay-0"], 4);
        assert_eq!(
            burst.scheduled.scheduled_spec_decode_tokens["replay-0"].len(),
            3
        );
        assert_eq!(burst.result.sampled_token_ids[0].len(), 4);
        assert!(burst.end >= before + Duration::from_millis(SLOW));
    }

    #[test]
    fn a_request_without_a_record_stays_on_the_latency_model() {
        let opt = test_opt();
        let latency = opt.build_latency().unwrap();
        let steps = recorded(9, &[0.0; 2], &[4, 4]);
        let mut engine = engine_with_sources(opt, latency, Some(steps));
        add(&mut engine, request("replay-7", prompt(1, 8), 5));

        let batches = engine.step();
        assert_eq!(chunk_sizes(&outputs(&batches)), [1; 5]);
        let spec = stats(&batches).spec_decoding_stats.expect("spec stats");
        assert_eq!(spec.num_drafts, 4);
        assert_eq!(spec.num_draft_tokens, 12);
        assert_eq!(spec.num_accepted_tokens, 0);
    }

    #[test]
    fn concurrent_recorded_requests_keep_their_own_bursts() {
        let opt = test_opt();
        let latency = opt.build_latency().unwrap();
        let records = [(7, vec![2, 4]), (6, vec![3, 2])]
            .into_iter()
            .map(|(output_tokens, bursts)| TraceRecord {
                prompt_tokens: 8,
                output_tokens,
                ttft_ms: 1.0,
                itl_ms: Some(vec![0.0; 2]),
                itl_tokens: Some(bursts),
                ..Default::default()
            })
            .collect();
        let steps: Box<dyn StepSource> = Box::new(IndexSteps::from_records(records));
        let mut engine = engine_with_sources(opt, latency, Some(steps));
        add(&mut engine, request("replay-0", prompt(1, 8), 50));
        add(&mut engine, request("replay-1", prompt(2, 8), 50));

        let outputs = outputs(&engine.step());
        assert_eq!(chunk_sizes(&outputs_of(&outputs, "replay-0")), [1, 2, 4]);
        assert_eq!(chunk_sizes(&outputs_of(&outputs, "replay-1")), [1, 3, 2]);
        assert_eq!(engine.num_unfinished_requests(), 0);
        assert_eq!(engine.scheduler.get_request_counts(), (0, 0));
    }

    #[test]
    fn a_stop_token_inside_a_burst_trims_it_and_finishes_the_request() {
        let opt = test_opt();
        let record = TraceRecord {
            prompt_tokens: 8,
            output_tokens: 9,
            ttft_ms: 1.0,
            itl_ms: Some(vec![0.0; 2]),
            itl_tokens: Some(vec![4, 4]),
            output_token_ids: Some(vec![5, 6, 0, 7, 8, 9, 10, 11, 12]),
            ..Default::default()
        };
        let sources = Sources {
            latency: opt.build_latency().unwrap(),
            token_source: Box::new(ReplayTokens::from_records(
                std::slice::from_ref(&record),
                opt.vocab_size,
            )),
            step_source: Some(Box::new(IndexSteps::from_records(vec![record]))),
            step_costs: None,
        };
        let mut engine = RustCoreEngine::from_parts(0, opt, sources, None).unwrap();
        let mut stopping = request("replay-0", prompt(1, 8), 50);
        stopping.sampling_params.as_mut().unwrap().stop_token_ids = vec![0];
        add(&mut engine, stopping);

        let outputs = outputs(&engine.step());
        let tokens: Vec<_> = outputs
            .iter()
            .map(|output| output.new_token_ids.clone())
            .filter(|tokens| !tokens.is_empty())
            .collect();
        assert_eq!(tokens, [vec![5], vec![6, 0]]);
        assert_eq!(
            outputs.last().unwrap().finish_reason,
            Some(EngineCoreFinishReason::Stop)
        );
        assert_eq!(engine.num_unfinished_requests(), 0);
        assert_eq!(engine.scheduler.get_request_counts(), (0, 0));
    }

    #[test]
    fn a_multi_token_latency_trace_paces_and_frames_its_bursts() {
        let opt = test_opt();
        let records: Vec<TraceRecord> = (0..4)
            .map(|_| TraceRecord {
                prompt_tokens: 4,
                output_tokens: 49,
                ttft_ms: 1.0,
                itl_ms: Some(vec![10.0; 12]),
                itl_tokens: Some(vec![4; 12]),
                concurrency: 1,
                ..Default::default()
            })
            .collect();
        let latency = TraceLatency::from_records(
            &records,
            opt.latency_model(),
            opt.max_num_batched_tokens as usize,
        )
        .unwrap();
        let mut engine = engine_with_latency(opt, Box::new(latency));
        let start = Instant::now();
        add(&mut engine, request("r1", prompt(1, 4), 9));

        let mut chunks = Vec::new();
        let mut spec = SpecDecodingStats::default();
        while engine.num_unfinished_requests() > 0 {
            let deadline = engine.earliest_deadline().expect("a deadline while busy");
            std::thread::sleep(deadline.saturating_duration_since(Instant::now()));
            let batches = engine.step();
            chunks.extend(chunk_sizes(&outputs(&batches)));
            for batch in &batches {
                let step = vllm::scheduler_stats(&batch.outputs)
                    .and_then(|stats| stats.spec_decoding_stats.as_ref());
                if let Some(step) = step {
                    assert_eq!(step.num_spec_tokens, 3);
                    spec.num_drafts += step.num_drafts;
                    spec.num_draft_tokens += step.num_draft_tokens;
                    spec.num_accepted_tokens += step.num_accepted_tokens;
                }
            }
        }
        assert_eq!(chunks, [1, 4, 4]);
        assert!(start.elapsed() >= Duration::from_millis(20));
        assert_eq!(spec.num_drafts, 2);
        assert_eq!(spec.num_draft_tokens, 6);
        assert_eq!(spec.num_accepted_tokens, 6);
    }

    fn decode_shape(reqs: u32, context_tokens: u64) -> StepShape {
        StepShape {
            decode: DecodeWork {
                reqs,
                tokens: reqs,
                drafts: 0,
                context_tokens,
            },
            prefill: Vec::new(),
        }
    }

    fn prefill_shape(tokens: u32, kv_depth: u32) -> StepShape {
        StepShape {
            decode: DecodeWork::default(),
            prefill: vec![PrefillChunk {
                tokens,
                kv_depth,
                last: true,
            }],
        }
    }

    fn recorded_step(shape: StepShape, step_ms: f64) -> StepRecord {
        StepRecord {
            step: 0,
            ts_ms: 0.0,
            exec_ms: step_ms,
            step_ms,
            sched_ms: 0.0,
            update_ms: 0.0,
            shape,
            sampled: 0,
            accepted: 0,
        }
    }

    fn recorded_load(blocks: u32, wall_ms: f64, outcome: KvOutcome) -> KvTransferRecord {
        KvTransferRecord {
            direction: KvDirection::Recv,
            blocks,
            start_step: 0,
            end_step: 1,
            wall_ms,
            outcome,
        }
    }

    fn engine_with_costs(
        opt: Opt,
        steps: &[StepRecord],
        loads: &[KvTransferRecord],
    ) -> RustCoreEngine {
        let sources = Sources {
            latency: opt.build_latency().unwrap(),
            token_source: opt.build_token_source().unwrap(),
            step_source: None,
            step_costs: Some(StepCosts {
                steps: StepTable::from_records(steps).unwrap(),
                kv_loads: KvTransferTable::from_records(loads),
                rng: StdRng::seed_from_u64(1),
            }),
        };
        RustCoreEngine::from_parts(0, opt, sources, None).unwrap()
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("vcr-rust-core-{name}-{}.jsonl", std::process::id()))
    }

    #[test]
    fn a_step_trace_prices_each_step_by_its_shape() {
        let steps = [
            recorded_step(prefill_shape(40, 0), 300.0),
            recorded_step(decode_shape(1, 40), SLOW as f64),
        ];
        let mut engine = engine_with_costs(test_opt(), &steps, &[]);
        let before = Instant::now();
        add(&mut engine, request("r1", prompt(1, 40), 3));

        let prefill = engine.in_flight.as_ref().expect("the prefill step");
        assert!(prefill.end >= before + Duration::from_millis(300));
        assert!(prefill.end < before + Duration::from_millis(SLOW));

        std::thread::sleep(Duration::from_millis(320));
        assert_eq!(num_tokens(&outputs(&engine.step())), 1);
        let decode = engine.in_flight.as_ref().expect("the first decode step");
        assert!(decode.end >= before + Duration::from_millis(300 + SLOW));
    }

    #[test]
    fn a_step_trace_replaces_the_latency_models_step_times() {
        let mut opt = test_opt();
        opt.inter_token_latency = SLOW;
        opt.prefill_time_per_token = SLOW;
        opt.prefill_overhead = SLOW;
        let steps = [
            recorded_step(prefill_shape(40, 0), 0.0),
            recorded_step(decode_shape(1, 40), 0.0),
        ];
        let mut engine = engine_with_costs(opt, &steps, &[]);
        add(&mut engine, request("r1", prompt(1, 40), 4));

        let outputs = outputs(&engine.step());
        assert_eq!(num_tokens(&outputs), 4);
        assert_eq!(engine.num_unfinished_requests(), 0);
    }

    #[test]
    fn a_recorded_run_replays_its_step_times_on_another_engine() {
        let path = temp_path("step-round-trip");
        let mut opt = test_opt();
        opt.inter_token_latency = 30;
        opt.max_num_batched_tokens = 64;
        opt.step_trace_out = Some(path.clone());
        let mut source = engine(opt);
        add(&mut source, request("r1", prompt(1, 100), 6));
        assert_eq!(num_tokens(&outputs(&run_to_idle(&mut source))), 6);
        drop(source);

        let trace = read_step_trace_file(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(trace.meta.source.as_deref(), Some("vllm-vcr play"));
        let shapes: Vec<&StepShape> = trace.steps.iter().map(|step| &step.shape).collect();
        assert_eq!(
            shapes[0].prefill,
            [PrefillChunk {
                tokens: 64,
                kv_depth: 0,
                last: false
            }]
        );
        assert_eq!(
            shapes[1].prefill,
            [PrefillChunk {
                tokens: 36,
                kv_depth: 64,
                last: true
            }]
        );
        let decodes: Vec<_> = shapes[2..].iter().map(|shape| shape.decode).collect();
        assert_eq!(
            decodes,
            (0..5)
                .map(|step| decode_shape(1, 100 + step).decode)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            trace.steps.iter().map(|step| step.step).collect::<Vec<_>>(),
            (1..=7).collect::<Vec<_>>()
        );
        assert_eq!(trace.steps.iter().map(|step| step.sampled).sum::<u32>(), 6);
        assert!(trace.steps[2..].iter().all(|step| step.step_ms == 30.0));
        assert!(trace.steps[..2].iter().all(|step| step.step_ms == 0.0));

        let mut opt = test_opt();
        opt.max_num_batched_tokens = 64;
        let mut replay = engine_with_costs(opt, &trace.steps, &trace.kv_transfers);
        let started = std::time::Instant::now();
        add(&mut replay, request("other", prompt(2, 100), 6));
        assert_eq!(num_tokens(&outputs(&run_to_idle(&mut replay))), 6);
        let elapsed = started.elapsed();
        assert!(
            elapsed >= Duration::from_millis(150),
            "replayed in {elapsed:?}"
        );
        assert!(
            elapsed < Duration::from_millis(600),
            "replayed in {elapsed:?}"
        );
    }

    #[test]
    fn a_remote_load_takes_its_recorded_time() {
        let params = prefill(prompt(1, 40));
        let mut opt = test_opt();
        opt.pd_role = PdRole::Decode;
        let steps = [recorded_step(decode_shape(1, 40), 0.0)];
        let loads = [recorded_load(3, 150.0, KvOutcome::Ok)];
        let mut engine = engine_with_costs(opt, &steps, &loads);
        let started = std::time::Instant::now();
        add(&mut engine, kv_request("d-1", prompt(1, 40), 1, params));

        let outputs = outputs(&run_to_idle(&mut engine));
        assert_eq!(num_tokens(&outputs), 1);
        assert!(started.elapsed() >= Duration::from_millis(150));
        let prefill = outputs[0].prefill_stats.clone().unwrap();
        assert!(prefill.num_external_cached_tokens > 0);
    }

    #[test]
    fn a_recorded_load_failure_is_replayed_recomputed_and_recorded_again() {
        let path = temp_path("kv-failure");
        let params = prefill(prompt(1, 40));
        let mut opt = test_opt();
        opt.pd_role = PdRole::Decode;
        opt.step_trace_out = Some(path.clone());
        let steps = [recorded_step(decode_shape(1, 40), 0.0)];
        let loads = [recorded_load(3, 120.0, KvOutcome::Failed)];
        let mut engine = engine_with_costs(opt, &steps, &loads);
        let started = std::time::Instant::now();
        add(&mut engine, kv_request("d-1", prompt(1, 40), 2, params));

        let outputs = outputs(&run_to_idle(&mut engine));
        assert_eq!(num_tokens(&outputs), 2);
        assert!(started.elapsed() >= Duration::from_millis(120));
        let prefill = outputs[0].prefill_stats.clone().unwrap();
        assert_eq!(prefill.num_external_cached_tokens, 0);
        assert_eq!(prefill.num_computed_tokens, 40);
        drop(engine);

        let trace = read_step_trace_file(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(trace.kv_transfers.len(), 1);
        assert_eq!(trace.kv_transfers[0].outcome, KvOutcome::Failed);
        assert_eq!(trace.kv_transfers[0].blocks, 3);
        assert!(trace.kv_transfers[0].wall_ms >= 120.0);
    }

    #[test]
    fn remote_loads_are_recorded_with_their_size_time_and_outcome() {
        let path = temp_path("kv-record");
        let mut opt = test_opt();
        opt.pd_role = PdRole::Decode;
        opt.kv_cache_transfer_latency = 80;
        opt.step_trace_out = Some(path.clone());
        let mut engine = engine(opt);
        add(
            &mut engine,
            kv_request("ok", prompt(1, 40), 1, prefill(prompt(1, 40))),
        );
        run_to_idle(&mut engine);
        let mut unreadable = prefill(prompt(2, 40));
        unreadable["remote_block_ids"] = json!([[7]]);
        add(&mut engine, kv_request("bad", prompt(2, 40), 1, unreadable));
        run_to_idle(&mut engine);
        drop(engine);

        let trace = read_step_trace_file(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let loads = &trace.kv_transfers;
        assert_eq!(loads.len(), 1, "{loads:?}");
        assert_eq!(loads[0].direction, KvDirection::Recv);
        assert_eq!(loads[0].outcome, KvOutcome::Ok);
        assert_eq!(loads[0].blocks, 3);
        assert!(loads[0].wall_ms >= 80.0, "{loads:?}");
        assert!(loads[0].end_step >= loads[0].start_step);
    }

    #[test]
    fn step_trace_flags_need_the_rust_core_and_one_engine() {
        let mut opt = Opt::parse_from(["play"]);
        opt.step_trace_out = Some(temp_path("unused"));
        assert!(opt.validate_engine().is_err());

        let mut opt = test_opt();
        opt.step_trace_out = Some(temp_path("unused"));
        assert!(opt.validate_engine().is_ok());
        opt.engine_count = 2;
        assert!(opt.validate_engine().is_err());
    }

    #[test]
    fn an_unusable_step_trace_fails_at_startup() {
        let path = temp_path("empty-step-trace");
        std::fs::write(&path, "{\"meta\": {}}\n").unwrap();
        let mut opt = test_opt();
        opt.step_trace = Some(path.to_str().unwrap().parse().unwrap());
        let error = opt.validate_engine().unwrap_err();
        std::fs::remove_file(&path).unwrap();
        assert!(format!("{error:#}").contains("no executed steps"));
    }

    #[test]
    fn push_mode_is_refused_without_the_rust_core() {
        let opt = Opt::parse_from(["play", "--kv-transfer-mode", "push"]);
        assert!(opt.validate_engine().is_err());
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
        let mut opt = test_opt();
        opt.output_token_chunk_size = 2;
        assert!(opt.validate_engine().is_err());
        assert!(test_opt().validate_engine().is_ok());
    }
}
