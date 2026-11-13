//! The simulated GPU worker behind vLLM's scheduler.
//!
//! ```text
//!   Scheduler::schedule() ──SchedulerOutput──▶ SimRunner::execute()
//!                                                │  per scheduled request:
//!                                                │    prefill chunk -> LatencyModel::prefill_chunk_cost
//!                                                │    decode step   -> recorded step, else LatencyModel::paced_step
//!                                                │    prompt done   -> TokenSource::next_tokens
//!                                                │  the step: its shape -> a recorded step's time,
//!                                                │    when a step trace is replayed
//!                                                ▼
//!   Scheduler::update_from_output() ◀──ModelRunnerOutput + step duration
//!   Scheduler::update_draft_token_ids() ◀──DraftTokenIds
//! ```
//!
//! A burst of `n > 1` tokens goes through vLLM's speculative decoding path, with
//! `K = spec_tokens` drafts per request per step:
//!
//! ```text
//!   step N    samples 1 token              drafts K tokens for step N+1
//!   step N+1  scheduled 1 + K tokens       returns n tokens: n - 1 drafts accepted,
//!                                          K - (n - 1) rolled back by the scheduler
//! ```
//!
//! A recorded first chunk of `F > 1` tokens takes the same path one step late:
//!
//! ```text
//!   last prefill step  samples nothing     drafts F - 1 tokens
//!   next step          scheduled F - 1     returns F tokens: F - 1 drafts accepted
//! ```

use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use rand::SeedableRng as _;
use rand::rngs::StdRng;
use sim_protocol::vllm::EngineCoreFinishReason;
use sim_trace::steps::{KvOutcome, KvTransferTable, PrefillChunk, StepShape, StepTable};
use tokio::time::Instant;
use tracing::warn;
use vllm_engine_core::sched::output::{NewRequestData, SchedulerOutput};
use vllm_engine_core::sched::scheduler::{DraftTokenIds, KvConnectorOutput, ModelRunnerOutput};
use vllm_engine_core_connectors::nixl::{NixlConnectorMetadata, TransferMode};

use crate::dataplane::{BlockRead, block_pattern};
use crate::latency::{BIG_CHUNK_TOKENS, Churn, DecodePacing, FirstTokenCtx, LatencyModel};
use crate::replay_steps::{ScriptedDecode, StepSource};
use crate::rust_core::nixl::{NixlWorker, remote_block_ids};
use crate::tokens::{TokenCtx, TokenSource};

/// Token id of every simulated draft.
const DRAFT_TOKEN_ID: i32 = 0;

/// One executed step: what the worker hands back and how long it took.
pub(crate) struct StepResult {
    pub output: ModelRunnerOutput,
    /// Drafts for the next step of each request that can use them.
    pub drafts: DraftTokenIds,
    pub duration: Duration,
    pub shape: StepShape,
    /// Speculative drafts the step accepted.
    pub accepted: usize,
    /// Remote KV loads this step started, with the blocks each one moves.
    pub loads_started: Vec<(String, usize)>,
}

/// Recorded step and KV load times that replace the latency model's.
pub(crate) struct StepCosts {
    pub steps: StepTable,
    pub kv_loads: Option<KvTransferTable>,
    pub rng: StdRng,
}

/// Where the worker gets its timing, output tokens and recorded decode steps.
pub(crate) struct Sources {
    pub latency: Box<dyn LatencyModel>,
    pub token_source: Box<dyn TokenSource>,
    pub step_source: Option<Box<dyn StepSource>>,
    pub step_costs: Option<StepCosts>,
}

/// The output-length bounds a request arrives with.
pub(crate) struct OutputLimits {
    pub max_tokens: usize,
    pub min_tokens: usize,
    /// Draw the output length from the latency model's recorded distribution.
    pub sample_output_len: bool,
}

struct WorkerRequest {
    rng: StdRng,
    prompt_token_ids: Vec<u32>,
    /// Prompt plus sampled tokens, as of the last schedule.
    num_tokens: usize,
    num_computed: usize,
    num_generated: usize,
    /// Output tokens this request generates before it finishes.
    output_len: usize,
    /// The current prefill has reached its last prompt token.
    prefilled: bool,
    /// Recorded decode steps left to replay.
    script: Option<ScriptedDecode>,
    /// Size of a recorded first chunk that is waiting for its drafts.
    pending_first_chunk: Option<usize>,
    /// Prompt tokens already in KV when the request was first scheduled.
    num_cached_tokens: usize,
    pacing: Option<DecodePacing>,
    emit_offset: Option<Duration>,
    /// KV blocks holding this request, in prompt order.
    block_ids: Vec<u32>,
}

impl WorkerRequest {
    /// Fill byte of the `index`-th block of the prompt, if the prompt reaches it.
    fn block_pattern(&self, index: usize, block_size: usize) -> Option<u8> {
        let start = index.checked_mul(block_size)?;
        let end = self.prompt_token_ids.len().min(start + block_size);
        (start < end).then(|| block_pattern(&self.prompt_token_ids[start..end]))
    }
}

pub(crate) struct SimRunner {
    latency: Box<dyn LatencyModel>,
    token_source: Box<dyn TokenSource>,
    step_source: Option<Box<dyn StepSource>>,
    step_costs: Option<StepCosts>,
    /// Drafts proposed per request per step, vLLM's `num_speculative_tokens`.
    spec_tokens: usize,
    requests: HashMap<String, WorkerRequest>,
    kv: NixlWorker,
    block_size: usize,
    max_num_batched_tokens: usize,
    time_scale: f64,
    recent_big_admissions: VecDeque<Instant>,
}

impl SimRunner {
    pub(crate) fn new(
        sources: Sources,
        kv: NixlWorker,
        block_size: usize,
        max_num_batched_tokens: usize,
        time_scale: f64,
    ) -> Self {
        let Sources {
            latency,
            token_source,
            step_source,
            step_costs,
        } = sources;
        let spec_tokens = step_source
            .as_ref()
            .and_then(|source| source.spec_tokens())
            .max(latency.spec_tokens())
            .unwrap_or(0) as usize;
        SimRunner {
            latency,
            token_source,
            step_source,
            step_costs,
            spec_tokens,
            requests: HashMap::new(),
            kv,
            block_size: block_size.max(1),
            max_num_batched_tokens,
            time_scale: if time_scale > 0.0 { time_scale } else { 1.0 },
            recent_big_admissions: VecDeque::new(),
        }
    }

    pub(crate) fn set_latency(&mut self, latency: Box<dyn LatencyModel>) {
        self.latency = latency;
    }

    /// The widest recorded burst minus one: what the scheduler must be configured
    /// with as `num_spec_tokens`. Zero when every step delivers one token.
    pub(crate) fn spec_tokens(&self) -> usize {
        self.spec_tokens
    }

    /// Start tracking a request and return the number of tokens it will generate.
    pub(crate) fn register(
        &mut self,
        request_id: &str,
        seed: u64,
        prompt_token_ids: Vec<u32>,
        limits: OutputLimits,
    ) -> usize {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut output_len = limits.max_tokens;
        if limits.sample_output_len
            && let Some(sampled) = self.latency.sample_output_len(&mut rng)
        {
            let lo = limits.min_tokens.max(1);
            output_len = sampled.clamp(lo, limits.max_tokens.max(lo));
        }
        if let Some(recorded) = self
            .token_source
            .on_request_added(request_id, &prompt_token_ids)
        {
            output_len = output_len.min(recorded).max(1);
        }
        let script = self
            .step_source
            .as_mut()
            .and_then(|source| source.take_schedule(request_id, &prompt_token_ids));
        if let Some(script) = &script {
            output_len = output_len.min(script.output_tokens()).max(1);
        }
        self.requests.insert(
            request_id.to_string(),
            WorkerRequest {
                rng,
                prompt_token_ids,
                num_tokens: 0,
                num_computed: 0,
                num_generated: 0,
                output_len,
                prefilled: false,
                script,
                pending_first_chunk: None,
                num_cached_tokens: 0,
                pacing: None,
                emit_offset: None,
                block_ids: Vec::new(),
            },
        );
        output_len
    }

    /// Stop tracking a request the scheduler never saw.
    pub(crate) fn forget(&mut self, request_id: &str) {
        if self.requests.remove(request_id).is_some() {
            self.token_source.on_request_finished(request_id);
        }
    }

    /// How far this request's outputs trail the step that produced them.
    pub(crate) fn emit_offset(&self, request_id: &str) -> Duration {
        self.requests
            .get(request_id)
            .and_then(|request| request.emit_offset)
            .unwrap_or_default()
            .div_f64(self.time_scale)
    }

    /// The finish reason a replayed stream recorded, if the token source knows one.
    pub(crate) fn recorded_finish_reason(
        &self,
        request_id: &str,
    ) -> Option<EngineCoreFinishReason> {
        self.token_source.finish_reason(request_id)
    }

    /// Run one scheduled step: advance every scheduled request, sample the tokens
    /// due to each one whose prompt is fully computed, and price the step.
    pub(crate) fn execute(&mut self, scheduled: &SchedulerOutput, num_running: u64) -> StepResult {
        let mut loads_started = Vec::new();
        let kv_connector_output = self.transfer_kv(scheduled, num_running, &mut loads_started);
        for request_id in &scheduled.finished_req_ids {
            self.forget(request_id);
        }
        let now = Instant::now();
        for new in &scheduled.scheduled_new_reqs {
            self.admit(new, now);
        }
        let cached = &scheduled.scheduled_cached_reqs;
        for (i, request_id) in cached.req_ids.iter().enumerate() {
            let Some(request) = self.requests.get_mut(request_id) else {
                continue;
            };
            request.num_computed = cached.num_computed_tokens[i];
            if let Some(Some(new_block_ids)) = cached.new_block_ids.get(i) {
                request
                    .block_ids
                    .extend(new_block_ids.first().into_iter().flatten());
            }
        }

        let num_scheduled = scheduled.num_scheduled_tokens.len();
        let mut req_ids = Vec::with_capacity(num_scheduled);
        let mut sampled_token_ids = Vec::with_capacity(num_scheduled);
        let mut drafts = DraftTokenIds::default();
        let mut decode_cost = Duration::ZERO;
        let mut deepest_context = None;
        let mut prefill_cost = Duration::ZERO;
        let mut prefill_tokens = 0;
        let mut shape = StepShape::default();
        let mut accepted = 0;

        for (request_id, &num_tokens) in &scheduled.num_scheduled_tokens {
            req_ids.push(request_id.clone());
            let Some(request) = self.requests.get_mut(request_id) else {
                warn!(request_id, "scheduled request is unknown to the worker");
                sampled_token_ids.push(Vec::new());
                continue;
            };
            let samples = request.num_computed + num_tokens >= request.num_tokens;
            let num_drafts = scheduled
                .scheduled_spec_decode_tokens
                .get(request_id)
                .map_or(0, Vec::len);
            if request.prefilled && request.pacing.is_some() {
                shape.decode.reqs += 1;
                shape.decode.tokens += num_tokens as u32;
                shape.decode.drafts += num_drafts as u32;
                shape.decode.context_tokens += request.num_computed as u64;
            } else {
                shape.prefill.push(PrefillChunk {
                    tokens: num_tokens as u32,
                    kv_depth: request.num_computed as u32,
                    last: samples,
                });
            }

            let mut burst = 1;
            match request.pacing.as_mut().filter(|_| request.prefilled) {
                Some(_) if request.pending_first_chunk.is_some() => {}
                Some(pacing) => {
                    let recorded = request
                        .script
                        .as_mut()
                        .filter(|_| samples)
                        .and_then(|script| script.steps.pop_front());
                    let draw = match recorded {
                        Some(step) => step,
                        None => self
                            .latency
                            .paced_step(&mut request.rng, num_running, pacing),
                    };
                    let context = pacing.context_bucket();
                    if deepest_context.is_none_or(|deepest| context > deepest) {
                        deepest_context = Some(context);
                        decode_cost = draw.delay;
                    }
                    burst = draw.tokens as usize;
                }
                None => {
                    prefill_cost += self
                        .latency
                        .prefill_chunk_cost(num_tokens, request.num_computed);
                    prefill_tokens += num_tokens;
                }
            }

            request.num_computed += num_tokens;
            if !samples {
                sampled_token_ids.push(Vec::new());
                continue;
            }

            let remaining = request
                .output_len
                .saturating_sub(request.num_generated)
                .max(1);
            let wanted = if request.prefilled {
                request.pending_first_chunk.take().unwrap_or(burst)
            } else {
                request.prefilled = true;
                if request.emit_offset.is_none() {
                    request.emit_offset = Some(match self.step_costs {
                        Some(_) => Duration::ZERO,
                        None => self.latency.first_token_overhead(
                            &mut request.rng,
                            &FirstTokenCtx {
                                num_prompt_tokens: request.prompt_token_ids.len(),
                                num_cached_tokens: request.num_cached_tokens,
                                do_remote_prefill: false,
                                num_running,
                            },
                        ),
                    });
                }
                let first_chunk = request
                    .script
                    .as_ref()
                    .filter(|_| request.num_generated == 0)
                    .map_or(1, |script| script.first_chunk_tokens as usize)
                    .min(remaining)
                    .min(1 + self.spec_tokens);
                if first_chunk > 1 + num_drafts {
                    request.pending_first_chunk = Some(first_chunk);
                    drafts.req_ids.push(request_id.clone());
                    drafts
                        .draft_token_ids
                        .push(vec![DRAFT_TOKEN_ID; first_chunk - 1]);
                    sampled_token_ids.push(Vec::new());
                    continue;
                }
                first_chunk
            };

            let tokens = self.token_source.next_tokens(
                &TokenCtx {
                    request_id,
                    prompt_token_ids: &request.prompt_token_ids,
                    num_generated: request.num_generated,
                },
                wanted.clamp(1, remaining.min(1 + num_drafts)),
                &mut request.rng,
            );
            request.num_generated += tokens.len();
            request.num_tokens += tokens.len();
            if num_drafts > 0 {
                accepted += tokens.len().saturating_sub(1);
            }
            sampled_token_ids.push(tokens.into_iter().map(|token| token as i32).collect());
            if self.spec_tokens > 0 {
                drafts.req_ids.push(request_id.clone());
                drafts
                    .draft_token_ids
                    .push(vec![DRAFT_TOKEN_ID; self.spec_tokens]);
            }
        }

        if prefill_tokens > 0 && scheduled.total_num_scheduled_tokens >= self.max_num_batched_tokens
        {
            prefill_cost = prefill_cost.max(self.latency.budget_full_chunk_floor());
        }

        let cost = match &mut self.step_costs {
            Some(costs) => costs.steps.price(&shape, &mut costs.rng),
            None => decode_cost.max(prefill_cost),
        };
        StepResult {
            output: ModelRunnerOutput {
                req_ids,
                sampled_token_ids,
                kv_connector_output,
                ..Default::default()
            },
            drafts,
            duration: cost.div_f64(self.time_scale),
            shape,
            accepted,
            loads_started,
        }
    }

    /// Act on the KV connector's instructions for this step and report the
    /// transfers that finished.
    fn transfer_kv(
        &mut self,
        scheduled: &SchedulerOutput,
        num_running: u64,
        loads_started: &mut Vec<(String, usize)>,
    ) -> Option<KvConnectorOutput> {
        let metadata = scheduled
            .kv_connector_metadata
            .as_ref()
            .and_then(|metadata| metadata.0.downcast_ref::<NixlConnectorMetadata>());
        let Some(metadata) = metadata else {
            return self.kv.poll();
        };

        for request_id in &metadata.reqs_not_processed {
            self.kv.drop_lease(request_id);
        }
        for (request_id, &deadline) in &metadata.reqs_to_send {
            let blocks: Vec<(usize, u8)> = self
                .requests
                .get(request_id)
                .map(|request| {
                    request
                        .block_ids
                        .iter()
                        .enumerate()
                        .filter_map(|(index, &block_id)| {
                            let pattern = request.block_pattern(index, self.block_size)?;
                            Some((block_id as usize, pattern))
                        })
                        .collect()
                })
                .unwrap_or_default();
            let lease = Duration::from_secs_f64((deadline - metadata.scheduler_clock).max(0.0));
            self.kv.stage(request_id, &blocks, lease);
        }

        for (request_id, block_ids) in &metadata.push_finished_blocks {
            let block_ids = block_ids.items.first().cloned().unwrap_or_default();
            self.kv.finished_blocks(request_id, block_ids);
        }
        for (engine_id, heartbeat) in &metadata.heartbeat_by_engine {
            self.kv.heartbeat(
                engine_id,
                &heartbeat.host,
                heartbeat.port,
                &heartbeat.req_ids,
            );
        }
        for registration in metadata.push_registrations.values() {
            self.kv.register_push(registration);
        }

        for (request_id, recv) in &metadata.reqs_to_recv {
            let Some(remote) = &recv.remote else {
                continue;
            };
            let local: &[u32] = recv
                .local_block_ids
                .items
                .first()
                .map_or(&[], Vec::as_slice);
            let push = self.kv.mode() == TransferMode::Push;
            if local.is_empty() {
                if !push {
                    self.kv.release_remote(remote);
                }
                if recv.awaiting_kvs {
                    self.kv.finish_load(request_id);
                }
                continue;
            }
            let Some(request) = self.requests.get_mut(request_id) else {
                self.kv
                    .fail_load(request_id, "request is unknown to the worker");
                continue;
            };
            let prompt_blocks = request.prompt_token_ids.len().div_ceil(self.block_size);
            let first_block = prompt_blocks.saturating_sub(local.len());
            let expected: Option<Vec<(usize, u8)>> = local
                .iter()
                .enumerate()
                .map(|(offset, &block_id)| {
                    let pattern = request.block_pattern(first_block + offset, self.block_size)?;
                    Some((block_id as usize, pattern))
                })
                .collect();
            let Some(expected) = expected else {
                self.kv
                    .fail_load(request_id, "the prompt does not cover the blocks to load");
                continue;
            };
            let modeled = self.latency.first_token_overhead(
                &mut request.rng,
                &FirstTokenCtx {
                    num_prompt_tokens: request.prompt_token_ids.len(),
                    num_cached_tokens: 0,
                    do_remote_prefill: true,
                    num_running,
                },
            );
            request.emit_offset = Some(Duration::ZERO);
            let recorded = self.step_costs.as_mut().and_then(|costs| {
                let loads = costs.kv_loads.as_ref()?;
                Some(loads.draw(local.len(), &mut costs.rng))
            });
            let transfer_time = recorded
                .map_or(modeled, |load| load.duration)
                .div_f64(self.time_scale);
            loads_started.push((request_id.clone(), local.len()));
            if recorded.is_some_and(|load| load.outcome != KvOutcome::Ok) {
                if !push {
                    self.kv.release_remote(remote);
                }
                self.kv
                    .fail_load_after(request_id, "the recorded load failed", transfer_time);
                continue;
            }

            if push {
                self.kv.expect_push(request_id, expected, transfer_time);
                continue;
            }
            let remote_ids = remote_block_ids(remote);
            let Some(skipped) = remote_ids.len().checked_sub(local.len()) else {
                self.kv.fail_load(
                    request_id,
                    "decode allocated more KV blocks than the prefill engine supplied",
                );
                continue;
            };
            let reads = expected
                .into_iter()
                .zip(&remote_ids[skipped..])
                .map(|((local_block_id, expected), &remote_block_id)| BlockRead {
                    remote_block_id,
                    local_block_id,
                    expected,
                })
                .collect();
            self.kv.start_pull(request_id, remote, reads, transfer_time);
        }
        self.kv.poll()
    }

    /// A request scheduled for the first time, or again after a preemption.
    fn admit(&mut self, new: &NewRequestData, now: Instant) {
        let Some(request) = self.requests.get_mut(&new.req_id) else {
            return;
        };
        request.num_tokens = new.prefill_token_ids.len();
        request.num_computed = new.num_computed_tokens;
        request.prefilled = false;
        request.pending_first_chunk = None;
        request.block_ids = new.block_ids.first().cloned().unwrap_or_default();
        if request.pacing.is_some() {
            return;
        }

        let churn_window = Duration::from_secs_f64(1.0 / self.time_scale);
        while self
            .recent_big_admissions
            .front()
            .is_some_and(|&admitted| now.duration_since(admitted) > churn_window)
        {
            self.recent_big_admissions.pop_front();
        }
        let churn = if self.recent_big_admissions.is_empty() {
            Churn::Calm
        } else {
            Churn::BigChunk
        };
        request.pacing = Some(DecodePacing::for_prompt(
            request.prompt_token_ids.len(),
            churn,
        ));
        request.num_cached_tokens = new.num_computed_tokens;
        if request.num_tokens - request.num_computed > BIG_CHUNK_TOKENS as usize {
            self.recent_big_admissions.push_back(now);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use rand::rngs::StdRng;
    use vllm_engine_core::sched::output::{CachedRequestData, NewRequestData, SchedulerOutput};

    use crate::Opt;
    use crate::latency::{FirstTokenCtx, LatencyModel};
    use crate::replay_steps::{IndexSteps, StepSource};
    use crate::rust_core::nixl::NixlWorker;
    use crate::rust_core::runner::{OutputLimits, SimRunner, Sources};
    use crate::tokens::RandomTokens;
    use crate::trace::TraceRecord;

    /// 10ms per decode step, 1ms per prefill token, 500ms first-token overhead.
    struct FixedCosts;

    impl LatencyModel for FixedCosts {
        fn first_token_delay(&self, _rng: &mut StdRng, _ctx: &FirstTokenCtx) -> Duration {
            Duration::from_millis(500)
        }
        fn inter_token_delay(&self, _rng: &mut StdRng, _num_running: u64) -> Duration {
            Duration::from_millis(10)
        }
        fn prefill_chunk_cost(&self, chunk_tokens: usize, _kv_depth: usize) -> Duration {
            Duration::from_millis(chunk_tokens as u64)
        }
        fn budget_full_chunk_floor(&self) -> Duration {
            Duration::from_millis(200)
        }
    }

    fn runner(time_scale: f64) -> SimRunner {
        runner_with_steps(time_scale, None)
    }

    /// A runner replaying one recorded request, matched by the id `replay-0`.
    fn replaying(output_tokens: usize, gaps_ms: &[f64], bursts: &[u32]) -> SimRunner {
        let record = TraceRecord {
            prompt_tokens: 20,
            output_tokens,
            ttft_ms: 1.0,
            itl_ms: Some(gaps_ms.to_vec()),
            itl_tokens: Some(bursts.to_vec()),
            ..Default::default()
        };
        runner_with_steps(1.0, Some(Box::new(IndexSteps::from_records(vec![record]))))
    }

    fn runner_with_steps(time_scale: f64, step_source: Option<Box<dyn StepSource>>) -> SimRunner {
        SimRunner::new(
            Sources {
                latency: Box::new(FixedCosts),
                token_source: Box::new(RandomTokens { vocab_size: 100 }),
                step_source,
                step_costs: None,
            },
            NixlWorker::new(&Opt::parse_from(["play"]), 0),
            16,
            128,
            time_scale,
        )
    }

    fn register(runner: &mut SimRunner, request_id: &str, prompt_len: usize) {
        register_up_to(runner, request_id, prompt_len, 8);
    }

    /// Register a request and return the output length the worker settled on.
    fn register_up_to(
        runner: &mut SimRunner,
        request_id: &str,
        prompt_len: usize,
        max_tokens: usize,
    ) -> usize {
        runner.register(
            request_id,
            7,
            vec![1; prompt_len],
            OutputLimits {
                max_tokens,
                min_tokens: 0,
                sample_output_len: false,
            },
        )
    }

    fn new_request(request_id: &str, prefill_len: usize, num_computed: usize) -> NewRequestData {
        NewRequestData {
            req_id: request_id.to_string(),
            prompt_token_ids: Some(vec![1; prefill_len]),
            block_ids: vec![Vec::new()],
            num_computed_tokens: num_computed,
            prefill_token_ids: vec![1; prefill_len],
            prompt_is_token_ids: None,
            replay_start: 0,
        }
    }

    fn step(
        new: Vec<NewRequestData>,
        cached: &[(&str, usize)],
        scheduled: &[(&str, usize)],
    ) -> SchedulerOutput {
        SchedulerOutput {
            scheduled_new_reqs: new,
            scheduled_cached_reqs: CachedRequestData::new(
                cached.iter().map(|(id, _)| id.to_string()).collect(),
                Vec::new(),
                vec![None; cached.len()],
                cached.iter().map(|&(_, computed)| computed).collect(),
                vec![0; cached.len()],
            ),
            num_scheduled_tokens: scheduled
                .iter()
                .map(|&(id, tokens)| (id.to_string(), tokens))
                .collect(),
            total_num_scheduled_tokens: scheduled.iter().map(|&(_, tokens)| tokens).sum(),
            ..Default::default()
        }
    }

    #[test]
    fn a_token_is_sampled_only_once_the_prompt_is_fully_computed() {
        let mut runner = runner(1.0);
        register(&mut runner, "a", 100);

        let first = runner.execute(&step(vec![new_request("a", 100, 0)], &[], &[("a", 60)]), 1);
        assert_eq!(first.output.req_ids, ["a"]);
        assert_eq!(first.output.sampled_token_ids, [Vec::<i32>::new()]);
        assert_eq!(first.duration, Duration::from_millis(60));

        let last = runner.execute(&step(Vec::new(), &[("a", 60)], &[("a", 40)]), 1);
        assert_eq!(last.output.sampled_token_ids[0].len(), 1);
        assert_eq!(last.duration, Duration::from_millis(40));

        let decode = runner.execute(&step(Vec::new(), &[("a", 100)], &[("a", 1)]), 1);
        assert_eq!(decode.output.sampled_token_ids[0].len(), 1);
        assert_eq!(decode.duration, Duration::from_millis(10));
    }

    #[test]
    fn a_step_costs_the_larger_of_its_decode_and_prefill_work() {
        let mut runner = runner(1.0);
        register(&mut runner, "decoder", 4);
        register(&mut runner, "small", 5);
        register(&mut runner, "big", 50);
        runner.execute(
            &step(vec![new_request("decoder", 4, 0)], &[], &[("decoder", 4)]),
            1,
        );

        let hidden = runner.execute(
            &step(
                vec![new_request("small", 5, 0)],
                &[("decoder", 4)],
                &[("decoder", 1), ("small", 5)],
            ),
            2,
        );
        assert_eq!(hidden.duration, Duration::from_millis(10));

        let dominated = runner.execute(
            &step(
                vec![new_request("big", 50, 0)],
                &[("decoder", 5)],
                &[("decoder", 1), ("big", 50)],
            ),
            3,
        );
        assert_eq!(dominated.duration, Duration::from_millis(50));
    }

    #[test]
    fn a_budget_saturating_prefill_pays_the_full_chunk_floor() {
        let mut runner = runner(1.0);
        register(&mut runner, "a", 400);
        let result = runner.execute(&step(vec![new_request("a", 400, 0)], &[], &[("a", 128)]), 1);
        assert_eq!(result.duration, Duration::from_millis(200));
    }

    #[test]
    fn time_scale_compresses_step_and_emission_delays() {
        let mut runner = runner(10.0);
        register(&mut runner, "a", 100);
        let result = runner.execute(&step(vec![new_request("a", 100, 0)], &[], &[("a", 100)]), 1);
        assert_eq!(result.duration, Duration::from_millis(10));
        assert_eq!(runner.emit_offset("a"), Duration::from_millis(50));
    }

    #[test]
    fn a_preempted_request_prefills_again_with_its_emission_offset_unchanged() {
        let mut runner = runner(1.0);
        register(&mut runner, "a", 20);
        runner.execute(&step(vec![new_request("a", 20, 0)], &[], &[("a", 20)]), 1);
        let offset = runner.emit_offset("a");

        let resumed = runner.execute(&step(vec![new_request("a", 21, 0)], &[], &[("a", 21)]), 1);
        assert_eq!(resumed.duration, Duration::from_millis(21));
        assert_eq!(resumed.output.sampled_token_ids[0].len(), 1);
        assert_eq!(runner.emit_offset("a"), offset);
    }

    #[test]
    fn finished_requests_are_dropped_from_worker_state() {
        let mut runner = runner(1.0);
        register(&mut runner, "a", 4);
        runner.execute(&step(vec![new_request("a", 4, 0)], &[], &[("a", 4)]), 1);
        assert_eq!(runner.emit_offset("a"), Duration::from_millis(500));

        let mut flush = SchedulerOutput::default();
        flush.finished_req_ids.insert("a".to_string());
        runner.execute(&flush, 0);
        assert_eq!(runner.emit_offset("a"), Duration::ZERO);
    }

    fn with_drafts(
        mut scheduled: SchedulerOutput,
        request_id: &str,
        drafts: usize,
    ) -> SchedulerOutput {
        scheduled
            .scheduled_spec_decode_tokens
            .insert(request_id.to_string(), vec![0; drafts]);
        scheduled
    }

    #[test]
    fn a_recorded_burst_accepts_drafts_and_takes_its_recorded_gap() {
        let mut runner = replaying(9, &[35.0, 45.0], &[4, 4]);
        assert_eq!(runner.spec_tokens(), 3);
        assert_eq!(register_up_to(&mut runner, "replay-0", 20, 100), 9);

        let prefill = runner.execute(
            &step(
                vec![new_request("replay-0", 20, 0)],
                &[],
                &[("replay-0", 20)],
            ),
            1,
        );
        assert_eq!(prefill.output.sampled_token_ids[0].len(), 1);
        assert_eq!(prefill.drafts.req_ids, ["replay-0"]);
        assert_eq!(prefill.drafts.draft_token_ids, [vec![0; 3]]);

        let burst = runner.execute(
            &with_drafts(
                step(Vec::new(), &[("replay-0", 20)], &[("replay-0", 4)]),
                "replay-0",
                3,
            ),
            1,
        );
        assert_eq!(burst.output.sampled_token_ids[0].len(), 4);
        assert_eq!(burst.duration, Duration::from_millis(35));
        assert_eq!(burst.drafts.draft_token_ids, [vec![0; 3]]);
    }

    #[test]
    fn a_burst_is_capped_by_the_drafts_scheduled_and_the_output_left() {
        let mut runner = replaying(9, &[35.0, 45.0, 55.0], &[4, 2, 2]);
        assert_eq!(register_up_to(&mut runner, "replay-0", 20, 6), 6);
        runner.execute(
            &step(
                vec![new_request("replay-0", 20, 0)],
                &[],
                &[("replay-0", 20)],
            ),
            1,
        );

        let short_of_drafts = runner.execute(
            &with_drafts(
                step(Vec::new(), &[("replay-0", 20)], &[("replay-0", 3)]),
                "replay-0",
                2,
            ),
            1,
        );
        assert_eq!(short_of_drafts.output.sampled_token_ids[0].len(), 3);

        let undrafted = runner.execute(
            &step(Vec::new(), &[("replay-0", 23)], &[("replay-0", 1)]),
            1,
        );
        assert_eq!(undrafted.output.sampled_token_ids[0].len(), 1);
        assert_eq!(undrafted.duration, Duration::from_millis(45));

        let last = runner.execute(
            &with_drafts(
                step(Vec::new(), &[("replay-0", 24)], &[("replay-0", 4)]),
                "replay-0",
                3,
            ),
            1,
        );
        assert_eq!(last.output.sampled_token_ids[0].len(), 1);
        assert_eq!(last.duration, Duration::from_millis(55));
    }

    #[test]
    fn a_recorded_first_chunk_waits_one_step_for_its_drafts() {
        let mut runner = replaying(24, &[80.0, 80.0], &[8, 8]);
        assert_eq!(runner.spec_tokens(), 7);
        assert_eq!(register_up_to(&mut runner, "replay-0", 20, 100), 24);

        let prefill = runner.execute(
            &step(
                vec![new_request("replay-0", 20, 0)],
                &[],
                &[("replay-0", 20)],
            ),
            1,
        );
        assert_eq!(prefill.output.sampled_token_ids, [Vec::<i32>::new()]);
        assert_eq!(prefill.drafts.draft_token_ids, [vec![0; 7]]);
        assert_eq!(prefill.duration, Duration::from_millis(20));

        let first_chunk = runner.execute(
            &with_drafts(
                step(Vec::new(), &[("replay-0", 20)], &[("replay-0", 7)]),
                "replay-0",
                7,
            ),
            1,
        );
        assert_eq!(first_chunk.output.sampled_token_ids[0].len(), 8);
        assert_eq!(first_chunk.duration, Duration::ZERO);

        let block = runner.execute(
            &with_drafts(
                step(Vec::new(), &[("replay-0", 27)], &[("replay-0", 8)]),
                "replay-0",
                7,
            ),
            1,
        );
        assert_eq!(block.output.sampled_token_ids[0].len(), 8);
        assert_eq!(block.duration, Duration::from_millis(80));
    }

    #[test]
    fn no_drafts_are_proposed_without_multi_token_steps() {
        let mut runner = runner(1.0);
        register(&mut runner, "a", 4);
        let result = runner.execute(&step(vec![new_request("a", 4, 0)], &[], &[("a", 4)]), 1);
        assert_eq!(runner.spec_tokens(), 0);
        assert!(result.drafts.req_ids.is_empty());
    }
}
