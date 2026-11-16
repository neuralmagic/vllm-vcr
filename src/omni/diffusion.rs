//! A mock vLLM-Omni diffusion stage.
//!
//! Answers the head's generate requests with synthetic images after a modeled
//! delay. The delay comes from [`Timing`]: the knob model, a fixed overhead
//! plus a per-step time that scales with pixel count and image count, or a
//! recorded trace. `concurrency` requests execute at once and the rest queue
//! in arrival order.
//!
//! [`Server`] is the stage's whole state and is only touched by the loop in
//! [`DiffusionStage::run`], between socket reads. Control calls arrive on a
//! channel of their own, so a config patch lands between two requests.

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use serde::{Deserialize, Serialize};
use sim_protocol::omni::diffusion::{
    self as wire, DiffusionSockets, GenerateRequest, GeneratedImages, StageRequest,
};
use sim_trace::diffusion::{
    DiffusionReplay, ImageSize, Service, StepModel, read_trace_file, step_units,
};
use tokio::sync::mpsc;
use tokio::task::AbortHandle;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};
use zeromq::prelude::{SocketRecv as _, SocketSend as _};
use zeromq::{PushSocket, ZmqMessage};

use crate::LogFilter;
use crate::control::{
    self, ControlCall, ControlChannel, ControlError, ControlReply, ControlRequest, Patch, Stats,
};
use crate::engine_core::UnfinishedGauge;
use crate::omni::Stage;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// Pixels one request may return across all of its images.
const MAX_REQUEST_PIXELS: u64 = 64 * 1024 * 1024;
const BAD_REQUEST: u16 = 400;
const INTERNAL_ERROR: u16 = 500;

/// Where a mock diffusion stage's execution times come from.
#[derive(Debug, Clone)]
pub(crate) enum Timing {
    /// The `--omni-diffusion-step-time` and `--omni-diffusion-overhead` knobs.
    Knobs(StepModel),
    /// A recorded trace (`--latency-trace`): a request of a recorded shape
    /// replays that shape's recorded times, any other is timed by the model
    /// fitted to the trace.
    Trace(DiffusionReplay),
}

impl Timing {
    /// Build the timing for a stage started with `knobs` and, optionally, a
    /// recorded trace. The two are exclusive.
    pub(crate) fn new(knobs: StepModel, latency_trace: Option<&Path>) -> Result<Self> {
        let Some(path) = latency_trace else {
            return Ok(Timing::Knobs(knobs));
        };
        if !knobs.step_time.is_zero() || !knobs.overhead.is_zero() {
            bail!(
                "--latency-trace is mutually exclusive with --omni-diffusion-step-time and \
                 --omni-diffusion-overhead"
            );
        }
        let (_meta, records) = read_trace_file(path)?;
        let replay = DiffusionReplay::from_records(&records)
            .with_context(|| format!("building diffusion timing from {}", path.display()))?;
        Ok(Timing::Trace(replay))
    }

    /// The model that times a request with no recorded time of its own.
    fn step_model(&self) -> StepModel {
        match self {
            Timing::Knobs(model) => *model,
            Timing::Trace(replay) => replay.fitted(),
        }
    }
}

/// Timing and capacity of a mock diffusion stage.
#[derive(Debug, Clone)]
pub(crate) struct DiffusionConfig {
    pub timing: Timing,
    pub default_steps: u32,
    pub default_size: ImageSize,
    /// Requests executed at once.
    pub concurrency: u32,
}

/// A diffusion stage: image generation and editing behind an omni head.
#[derive(Debug, Clone)]
pub(crate) struct DiffusionStage {
    pub stage: Stage,
    pub config: DiffusionConfig,
}

/// A generate request with the stage defaults filled in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Job {
    size: ImageSize,
    count: u32,
    steps: u32,
    /// The service time a trace recorded for a request of this shape.
    recorded: Option<Service>,
}

impl DiffusionConfig {
    /// Resolve a request against the trace and the stage defaults. A request
    /// that sets no size takes the size a recorded request of its shape
    /// returned, before the stage default.
    fn job(&mut self, request: &GenerateRequest) -> Result<Job, String> {
        let recorded = match &mut self.timing {
            Timing::Knobs(_) => None,
            Timing::Trace(replay) => replay.next_recorded(&request.shape()),
        };
        let job = Job {
            size: request
                .size
                .or(recorded.and_then(|recorded| recorded.output_size))
                .unwrap_or(self.default_size),
            count: request.num_outputs,
            steps: request.num_inference_steps.unwrap_or(self.default_steps),
            recorded: recorded.map(|recorded| recorded.service),
        };
        if job.count == 0 || job.size.pixels() == 0 {
            return Err("request asks for no image".to_string());
        }
        let pixels = job.size.pixels().saturating_mul(u64::from(job.count));
        if pixels > MAX_REQUEST_PIXELS {
            return Err(format!(
                "request asks for {pixels} pixels, above the mock stage's limit of {MAX_REQUEST_PIXELS}"
            ));
        }
        Ok(job)
    }

    /// How long a job holds an execution slot, and how long after that its
    /// result is sent. Only a recorded time has a tail: the modeled time is
    /// spent in the slot.
    fn service(&self, job: Job) -> Service {
        job.recorded.unwrap_or_else(|| Service {
            slot: self
                .timing
                .step_model()
                .exec_time(step_units(job.size, job.steps, job.count)),
            tail: Duration::ZERO,
        })
    }
}

fn whole_ms(duration: Duration) -> u64 {
    (duration.as_secs_f64() * 1000.0).round() as u64
}

/// Which [`Timing`] a stage runs on, as `GET /config` reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimingSource {
    Knobs,
    Trace,
}

/// The stage knobs that can change at runtime, as `GET /config` reports them.
/// Milliseconds for the latency fields, matching the `--omni-diffusion-*` flags.
/// On a stage timed by a trace, `step_time` and `overhead` are the model
/// fitted to the trace and cannot be patched.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffusionKnobs {
    pub timing: TimingSource,
    pub step_time: u64,
    pub overhead: u64,
    pub default_steps: u32,
    pub default_width: u32,
    pub default_height: u32,
    pub concurrency: u32,
    pub log_requests: bool,
}

/// Partial update for `PATCH /config`. Absent fields keep their current value.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffusionKnobsPatch {
    pub step_time: Option<u64>,
    pub overhead: Option<u64>,
    pub default_steps: Option<u32>,
    pub default_width: Option<u32>,
    pub default_height: Option<u32>,
    pub concurrency: Option<u32>,
    pub log_requests: Option<bool>,
}

impl Patch for DiffusionKnobsPatch {
    fn validate(&self) -> Result<(), ControlError> {
        for (name, value) in [
            ("default_steps", self.default_steps),
            ("default_width", self.default_width),
            ("default_height", self.default_height),
            ("concurrency", self.concurrency),
        ] {
            if value == Some(0) {
                return Err(ControlError::bad_request(format!(
                    "{name} must be at least 1"
                )));
            }
        }
        Ok(())
    }
}

type StageControlCall = ControlCall<DiffusionKnobs, DiffusionKnobsPatch>;

struct Waiting {
    request_id: String,
    job: Job,
    arrived: Instant,
}

/// A request past the queue: holding an execution slot, or finishing after
/// it released one.
struct Started {
    job: Job,
    queue_wait: Duration,
    service: Service,
    /// Reports the request id when the current phase has elapsed.
    timer: AbortHandle,
}

/// Frames to send the head after handling one of its messages.
#[derive(Debug, Default, PartialEq, Eq)]
struct Reaction {
    replies: Vec<Vec<u8>>,
    /// The head asked the stage to stop.
    shutdown: bool,
}

/// The stage's request queue, config, and counters.
struct Server {
    config: DiffusionConfig,
    log_requests: bool,
    waiting: VecDeque<Waiting>,
    /// Requests holding an execution slot.
    running: HashMap<String, Started>,
    /// Requests that released their slot and whose result is not sent yet.
    finishing: HashMap<String, Started>,
    /// Receives a request id when the phase it is in has elapsed.
    elapsed_tx: mpsc::UnboundedSender<String>,
    received: u64,
    completed: u64,
    failed: u64,
    aborted: u64,
}

impl Server {
    fn new(
        config: DiffusionConfig,
        log_requests: bool,
        elapsed_tx: mpsc::UnboundedSender<String>,
    ) -> Self {
        Self {
            config,
            log_requests,
            waiting: VecDeque::new(),
            running: HashMap::new(),
            finishing: HashMap::new(),
            elapsed_tx,
            received: 0,
            completed: 0,
            failed: 0,
            aborted: 0,
        }
    }

    fn unfinished(&self) -> usize {
        self.waiting.len() + self.running.len() + self.finishing.len()
    }

    /// Report `request_id` on the elapsed channel after `delay`.
    fn timer(&self, request_id: &str, delay: Duration) -> AbortHandle {
        let elapsed_tx = self.elapsed_tx.clone();
        let request_id = request_id.to_string();
        tokio::spawn(async move {
            tokio::time::sleep(delay).await;
            elapsed_tx.send(request_id).ok();
        })
        .abort_handle()
    }

    /// Move waiting requests into free execution slots, in arrival order.
    fn start_ready(&mut self) {
        while self.running.len() < self.config.concurrency as usize {
            let Some(Waiting {
                request_id,
                job,
                arrived,
            }) = self.waiting.pop_front()
            else {
                return;
            };
            let queue_wait = arrived.elapsed();
            let service = self.config.service(job);
            let timer = self.timer(&request_id, service.slot);
            if self.log_requests {
                info!(
                    request_id,
                    exec_ms = service.slot.as_millis() as u64,
                    tail_ms = service.tail.as_millis() as u64,
                    recorded = job.recorded.is_some(),
                    waiting = self.waiting.len(),
                    "diffusion request executing"
                );
            }
            self.running.insert(
                request_id,
                Started {
                    job,
                    queue_wait,
                    service,
                    timer,
                },
            );
        }
    }

    fn drop_request(&mut self, request_id: &str) -> bool {
        let started = self
            .running
            .remove(request_id)
            .or_else(|| self.finishing.remove(request_id));
        if let Some(started) = started {
            started.timer.abort();
            return true;
        }
        let queued = self.waiting.len();
        self.waiting
            .retain(|waiting| waiting.request_id != request_id);
        self.waiting.len() < queued
    }

    fn on_request(&mut self, request: StageRequest) -> Result<Reaction> {
        let mut reaction = Reaction::default();
        match request {
            StageRequest::Generate(request) => {
                self.received += 1;
                match self.config.job(&request) {
                    Ok(job) => {
                        if self.log_requests {
                            info!(
                                request_id = %request.request_id,
                                width = job.size.width,
                                height = job.size.height,
                                images = job.count,
                                steps = job.steps,
                                input_images = request.input_images,
                                "diffusion request received"
                            );
                        }
                        self.drop_request(&request.request_id);
                        self.waiting.push_back(Waiting {
                            request_id: request.request_id,
                            job,
                            arrived: Instant::now(),
                        });
                    }
                    Err(reason) => {
                        warn!(request_id = %request.request_id, reason, "rejecting diffusion request");
                        self.failed += 1;
                        reaction.replies.push(wire::encode_error(
                            &request.request_id,
                            &reason,
                            BAD_REQUEST,
                        )?);
                    }
                }
            }
            StageRequest::Abort(request_ids) => {
                for request_id in request_ids {
                    if self.drop_request(&request_id) {
                        self.aborted += 1;
                        if self.log_requests {
                            info!(request_id, "diffusion request aborted");
                        }
                    }
                }
            }
            StageRequest::Rpc { rpc_id, method } => {
                debug!(method, "diffusion rpc; replying nil");
                reaction.replies.push(wire::encode_rpc_result(&rpc_id)?);
            }
            StageRequest::Shutdown => {
                info!("omni head asked the diffusion stage to shut down");
                reaction.shutdown = true;
            }
            StageRequest::Unknown(kind) => {
                warn!(kind, "unknown diffusion message type; ignoring");
            }
        }
        self.start_ready();
        Ok(reaction)
    }

    /// Move a request on when the phase it is in has elapsed. A request that
    /// held a slot releases it and, if it has a tail, finishes after it. The
    /// return is the result frame of a request that is done; `None` when the
    /// request has a tail to wait out, or was aborted in the meantime.
    fn on_elapsed(&mut self, request_id: &str) -> Result<Option<Vec<u8>>> {
        let done = match self.running.remove(request_id) {
            Some(mut started) => {
                self.start_ready();
                if started.service.tail.is_zero() {
                    started
                } else {
                    started.timer = self.timer(request_id, started.service.tail);
                    self.finishing.insert(request_id.to_string(), started);
                    return Ok(None);
                }
            }
            None => match self.finishing.remove(request_id) {
                Some(started) => started,
                None => return Ok(None),
            },
        };
        self.completed += 1;
        if self.log_requests {
            info!(request_id, "diffusion request finished");
        }
        wire::encode_result(&GeneratedImages {
            request_id,
            size: done.job.size,
            count: done.job.count,
            num_inference_steps: done.job.steps,
            queue_wait: done.queue_wait,
            exec_time: done.service.slot,
        })
        .or_else(|error| wire::encode_error(request_id, &format!("{error:#}"), INTERNAL_ERROR))
        .map(Some)
    }

    fn knobs(&self) -> DiffusionKnobs {
        let model = self.config.timing.step_model();
        DiffusionKnobs {
            timing: match self.config.timing {
                Timing::Knobs(_) => TimingSource::Knobs,
                Timing::Trace(_) => TimingSource::Trace,
            },
            step_time: whole_ms(model.step_time),
            overhead: whole_ms(model.overhead),
            default_steps: self.config.default_steps,
            default_width: self.config.default_size.width,
            default_height: self.config.default_size.height,
            concurrency: self.config.concurrency,
            log_requests: self.log_requests,
        }
    }

    fn apply(&mut self, patch: DiffusionKnobsPatch) -> Result<(), ControlError> {
        let config = &mut self.config;
        match &mut config.timing {
            Timing::Knobs(model) => {
                if let Some(step_time) = patch.step_time {
                    model.step_time = Duration::from_millis(step_time);
                }
                if let Some(overhead) = patch.overhead {
                    model.overhead = Duration::from_millis(overhead);
                }
            }
            Timing::Trace(_) if patch.step_time.is_some() || patch.overhead.is_some() => {
                return Err(ControlError::conflict(
                    "step_time and overhead cannot change at runtime under --latency-trace",
                ));
            }
            Timing::Trace(_) => {}
        }
        if let Some(default_steps) = patch.default_steps {
            config.default_steps = default_steps;
        }
        if let Some(width) = patch.default_width {
            config.default_size.width = width;
        }
        if let Some(height) = patch.default_height {
            config.default_size.height = height;
        }
        if let Some(concurrency) = patch.concurrency {
            config.concurrency = concurrency;
        }
        if let Some(log_requests) = patch.log_requests {
            self.log_requests = log_requests;
        }
        info!(config = ?self.config, "diffusion config patched via control API");
        self.start_ready();
        Ok(())
    }

    fn stats(&self) -> Stats {
        Stats {
            requests_received: self.received,
            requests_completed: self.completed,
            requests_failed: self.failed,
            requests_aborted: self.aborted,
            running: (self.running.len() + self.finishing.len()) as u64,
            waiting: self.waiting.len() as u64,
        }
    }

    fn on_control(&mut self, call: StageControlCall) {
        let reply = match call.request {
            ControlRequest::GetConfig => Ok(ControlReply::Config(self.knobs())),
            ControlRequest::PatchConfig(patch) => self
                .apply(*patch)
                .map(|()| ControlReply::Config(self.knobs())),
            ControlRequest::GetStats => Ok(ControlReply::Stats(self.stats())),
            ControlRequest::ResetStats => {
                self.received = 0;
                self.completed = 0;
                self.failed = 0;
                self.aborted = 0;
                Ok(ControlReply::Stats(self.stats()))
            }
        };
        call.reply.send(reply).ok();
    }

    fn abort_all(&mut self) {
        for (_, started) in self.running.drain().chain(self.finishing.drain()) {
            started.timer.abort();
        }
        self.waiting.clear();
    }
}

async fn send(responses: &mut PushSocket, frame: Vec<u8>) -> Result<()> {
    responses
        .send(ZmqMessage::from(frame))
        .await
        .context("sending diffusion response")
}

async fn send_all(responses: &mut PushSocket, frames: Vec<Vec<u8>>) -> Result<()> {
    for frame in frames {
        send(responses, frame).await?;
    }
    Ok(())
}

impl DiffusionStage {
    /// Attach to the head and serve requests until shutdown. With
    /// `control_address`, also serve the control API for this stage.
    pub(crate) async fn run(
        &self,
        log_requests: bool,
        control_address: Option<String>,
        log_filter: Option<LogFilter>,
        shutdown: CancellationToken,
    ) -> Result<()> {
        let attach = async {
            let replica = self.stage.register().await?;
            let sockets = wire::connect(replica.handshake_address(), CONNECT_TIMEOUT).await?;
            anyhow::Ok((replica, sockets))
        };
        let (replica, sockets) = tokio::select! {
            biased;
            _ = shutdown.cancelled() => {
                info!("shutdown requested before the omni handshake completed");
                return Ok(());
            }
            attached = attach => attached.context("diffusion stage failed to attach to the omni head")?,
        };
        let DiffusionSockets {
            mut requests,
            mut responses,
            input_address,
            output_address,
        } = sockets;
        let unfinished = UnfinishedGauge::default();
        let heartbeats = replica
            .announce(&input_address, &output_address, unfinished.clone())
            .await?;
        info!(config = ?self.config, "diffusion stage attached to omni head");

        let (control_tx, mut control_rx) = mpsc::unbounded_channel::<StageControlCall>();
        let stopped = CancellationToken::new();
        let mut control_server = match control_address {
            Some(address) => tokio::spawn(control::serve(
                address,
                ControlChannel(control_tx),
                log_filter,
                stopped.clone(),
            )),
            None => tokio::spawn(std::future::pending()),
        };

        let (elapsed_tx, mut elapsed_rx) = mpsc::unbounded_channel::<String>();
        let mut server = Server::new(self.config.clone(), log_requests, elapsed_tx);

        let outcome = loop {
            unfinished.set(server.unfinished());
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => break Ok(()),

                exited = &mut control_server => {
                    break match exited {
                        Ok(Ok(())) => Err(anyhow::anyhow!("control API server exited")),
                        Ok(Err(error)) => Err(error),
                        Err(error) => Err(error).context("control API task join failed"),
                    };
                }

                Some(call) = control_rx.recv() => server.on_control(call),

                Some(request_id) = elapsed_rx.recv() => {
                    let sent = match server.on_elapsed(&request_id) {
                        Ok(Some(frame)) => send(&mut responses, frame).await,
                        Ok(None) => Ok(()),
                        Err(error) => Err(error),
                    };
                    if let Err(error) = sent {
                        break Err(error);
                    }
                }

                message = requests.recv() => {
                    let message = match message.context("receiving diffusion request") {
                        Ok(message) => message,
                        Err(error) => break Err(error),
                    };
                    let request = match message.get(0).map(|frame| wire::decode_request(frame)) {
                        Some(Ok(request)) => request,
                        Some(Err(error)) => {
                            warn!(%error, "failed to decode diffusion request; ignoring");
                            continue;
                        }
                        None => continue,
                    };
                    let reaction = match server.on_request(request) {
                        Ok(reaction) => reaction,
                        Err(error) => break Err(error),
                    };
                    if let Err(error) = send_all(&mut responses, reaction.replies).await {
                        break Err(error);
                    }
                    if reaction.shutdown {
                        break Ok(());
                    }
                }
            }
        };

        server.abort_all();
        stopped.cancel();
        control_server.abort();
        heartbeats.stop().await;
        info!("diffusion stage shut down");
        outcome
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use std::time::Duration;

    use rmpv::Value;
    use sim_protocol::omni::diffusion::{GenerateRequest, StageRequest};
    use sim_trace::diffusion::{
        DiffusionOutcome, DiffusionRecord, DiffusionReplay, DiffusionShape, EXEC_TIME_METRIC,
        ImageSize, StepModel, append_record, write_meta,
    };
    use sim_trace::trace::{TraceKind, TraceMeta};
    use tokio::sync::{mpsc, oneshot};

    use crate::control::{ControlCall, ControlError, ControlReply, ControlRequest, Patch, Stats};
    use crate::omni::diffusion::{
        DiffusionConfig, DiffusionKnobs, DiffusionKnobsPatch, Job, MAX_REQUEST_PIXELS, Server,
        Timing, TimingSource,
    };

    const SQUARE_1024: ImageSize = ImageSize {
        width: 1024,
        height: 1024,
    };

    const SQUARE_512: ImageSize = ImageSize {
        width: 512,
        height: 512,
    };

    fn knobs(step_time: Duration, overhead: Duration) -> Timing {
        Timing::Knobs(StepModel {
            overhead,
            step_time,
        })
    }

    fn config() -> DiffusionConfig {
        DiffusionConfig {
            timing: knobs(Duration::from_millis(40), Duration::from_millis(300)),
            default_steps: 50,
            default_size: SQUARE_1024,
            concurrency: 1,
        }
    }

    fn recorded(
        arrival_ms: f64,
        size: Option<ImageSize>,
        steps: Option<u32>,
        latency_ms: f64,
    ) -> DiffusionRecord {
        DiffusionRecord {
            arrival_ms,
            shape: DiffusionShape {
                size,
                steps,
                images: 1,
                input_images: 0,
            },
            concurrency: 1,
            latency_ms,
            outcome: DiffusionOutcome::Finished,
            output_size: size,
            stage_metrics: BTreeMap::new(),
        }
    }

    /// A trace of 512x512 and 1024x1024 requests at 4 and 8 steps whose
    /// least-squares fit is exactly `80 ms + 25 ms per step`, plus one
    /// prompt-only request that came back 768x512.
    fn trace_records() -> Vec<DiffusionRecord> {
        let mut records: Vec<DiffusionRecord> = [
            (SQUARE_512, 4, 103.0),
            (SQUARE_512, 8, 130.0),
            (SQUARE_1024, 4, 180.0),
            (SQUARE_1024, 8, 280.0),
            (SQUARE_512, 4, 107.0),
        ]
        .into_iter()
        .enumerate()
        .map(|(index, (size, steps, latency_ms))| {
            recorded(index as f64, Some(size), Some(steps), latency_ms)
        })
        .collect();
        records.push(DiffusionRecord {
            output_size: Some(ImageSize {
                width: 768,
                height: 512,
            }),
            ..recorded(10.0, None, None, 900.0)
        });
        records
    }

    fn trace_config() -> DiffusionConfig {
        DiffusionConfig {
            timing: Timing::Trace(DiffusionReplay::from_records(&trace_records()).expect("replay")),
            ..config()
        }
    }

    fn request(
        size: Option<ImageSize>,
        num_outputs: u32,
        num_inference_steps: Option<u32>,
    ) -> GenerateRequest {
        GenerateRequest {
            request_id: "img_gen-1".to_string(),
            size,
            num_outputs,
            num_inference_steps,
            input_images: 0,
        }
    }

    #[test]
    fn job_fills_stage_defaults() {
        assert_eq!(
            config().job(&request(None, 1, None)),
            Ok(Job {
                size: SQUARE_1024,
                count: 1,
                steps: 50,
                recorded: None,
            })
        );
    }

    #[test]
    fn job_keeps_what_the_request_set() {
        let size = ImageSize {
            width: 768,
            height: 512,
        };
        assert_eq!(
            config().job(&request(Some(size), 3, Some(4))),
            Ok(Job {
                size,
                count: 3,
                steps: 4,
                recorded: None,
            })
        );
    }

    #[test]
    fn job_rejects_empty_and_oversized_requests() {
        assert!(config().job(&request(None, 0, None)).is_err());
        let empty = ImageSize {
            width: 0,
            height: 512,
        };
        assert!(config().job(&request(Some(empty), 1, None)).is_err());

        let at_limit = (MAX_REQUEST_PIXELS / SQUARE_1024.pixels()) as u32;
        assert!(config().job(&request(None, at_limit, None)).is_ok());
        assert!(config().job(&request(None, at_limit + 1, None)).is_err());
    }

    #[test]
    fn exec_time_is_overhead_plus_steps_at_the_reference_size() {
        let job = Job {
            size: SQUARE_1024,
            count: 1,
            steps: 50,
            recorded: None,
        };
        assert_eq!(
            config().service(job).slot,
            Duration::from_millis(300 + 50 * 40)
        );
    }

    #[test]
    fn exec_time_scales_with_pixels_and_image_count() {
        let quarter = Job {
            size: SQUARE_512,
            count: 1,
            steps: 20,
            recorded: None,
        };
        assert_eq!(
            config().service(quarter).slot,
            Duration::from_millis(300 + 20 * 40 / 4)
        );

        let four_images = Job {
            count: 4,
            ..quarter
        };
        assert_eq!(
            config().service(four_images).slot,
            Duration::from_millis(300 + 20 * 40)
        );
    }

    #[test]
    fn unset_knobs_make_requests_instant() {
        let instant = DiffusionConfig {
            timing: knobs(Duration::ZERO, Duration::ZERO),
            ..config()
        };
        let job = Job {
            size: SQUARE_1024,
            count: 2,
            steps: 50,
            recorded: None,
        };
        assert_eq!(instant.service(job).total(), Duration::ZERO);
    }

    fn exec_ms(config: &mut DiffusionConfig, request: &GenerateRequest) -> u64 {
        let job = config.job(request).expect("job");
        let service = config.service(job);
        assert_eq!(service.tail, Duration::ZERO);
        (service.slot.as_secs_f64() * 1000.0).round() as u64
    }

    #[test]
    fn trace_replays_a_recorded_shapes_times_in_order() {
        let mut config = trace_config();
        let small = request(Some(SQUARE_512), 1, Some(4));
        let times: Vec<u64> = (0..3).map(|_| exec_ms(&mut config, &small)).collect();
        assert_eq!(times, [103, 107, 103]);
        assert_eq!(
            exec_ms(&mut config, &request(Some(SQUARE_1024), 1, Some(8))),
            280
        );
    }

    #[test]
    fn trace_times_an_unseen_shape_with_the_fitted_model() {
        let mut config = trace_config();
        let unseen = request(Some(SQUARE_1024), 1, Some(20));
        assert_eq!(config.job(&unseen).expect("job").recorded, None);
        assert_eq!(exec_ms(&mut config, &unseen), 80 + 25 * 20);

        let two_images = request(Some(SQUARE_512), 2, Some(4));
        assert_eq!(exec_ms(&mut config, &two_images), 80 + 25 * 2);
    }

    #[test]
    fn trace_gives_a_sizeless_request_the_size_the_recorded_one_returned() {
        let mut config = trace_config();
        let job = config.job(&request(None, 1, None)).expect("job");
        assert_eq!(
            job.size,
            ImageSize {
                width: 768,
                height: 512
            }
        );
        assert_eq!(
            job.recorded.map(|service| service.total()),
            Some(Duration::from_millis(900))
        );

        let unseen = config.job(&request(None, 3, None)).expect("job");
        assert_eq!(
            unseen.size, SQUARE_1024,
            "an unseen shape takes the default"
        );
    }

    struct TraceFile(PathBuf);

    impl TraceFile {
        fn write(name: &str, contents: &[u8]) -> Self {
            let path = std::env::temp_dir().join(format!(
                "vllm-vcr-diffusion-{name}-{}.jsonl",
                std::process::id()
            ));
            std::fs::write(&path, contents).expect("write trace");
            Self(path)
        }
    }

    impl Drop for TraceFile {
        fn drop(&mut self) {
            std::fs::remove_file(&self.0).ok();
        }
    }

    fn trace_file(name: &str, records: &[DiffusionRecord]) -> TraceFile {
        let mut contents = Vec::new();
        let meta = TraceMeta {
            kind: TraceKind::Diffusion,
            ..TraceMeta::default()
        };
        write_meta(&mut contents, &meta).expect("meta");
        for record in records {
            append_record(&mut contents, record).expect("record");
        }
        TraceFile::write(name, &contents)
    }

    #[test]
    fn timing_reads_a_trace_and_refuses_knobs_alongside_it() {
        let file = trace_file("knobs", &trace_records());
        let unset = StepModel {
            overhead: Duration::ZERO,
            step_time: Duration::ZERO,
        };
        assert!(matches!(
            Timing::new(unset, None).expect("knobs"),
            Timing::Knobs(_)
        ));
        let Timing::Trace(replay) = Timing::new(unset, Some(&file.0)).expect("trace") else {
            panic!("expected trace timing");
        };
        assert_eq!(replay.shapes(), 5);

        let set = StepModel {
            overhead: Duration::ZERO,
            step_time: Duration::from_millis(40),
        };
        let error = Timing::new(set, Some(&file.0)).expect_err("knobs with a trace");
        assert!(error.to_string().contains("mutually exclusive"));
    }

    #[test]
    fn timing_refuses_a_token_trace_and_a_trace_with_nothing_finished() {
        let unset = StepModel {
            overhead: Duration::ZERO,
            step_time: Duration::ZERO,
        };
        let token_trace = TraceFile::write(
            "tokens",
            b"{\"meta\":{\"model\":\"m\"}}\n{\"prompt_tokens\":1,\"output_tokens\":1,\"ttft_ms\":1.0}\n",
        );
        let error = Timing::new(unset, Some(&token_trace.0)).expect_err("token trace");
        assert!(format!("{error:#}").contains("not a diffusion trace"));

        let aborted = DiffusionRecord {
            outcome: DiffusionOutcome::Aborted,
            ..recorded(0.0, Some(SQUARE_512), Some(4), 5.0)
        };
        let file = trace_file("aborted", &[aborted]);
        let error = Timing::new(unset, Some(&file.0)).expect_err("nothing finished");
        assert!(format!("{error:#}").contains("no finished request"));
    }

    /// A server whose requests never elapse on their own: tests drive
    /// completion through `on_elapsed`.
    fn server(concurrency: u32) -> (Server, mpsc::UnboundedReceiver<String>) {
        let (elapsed_tx, elapsed_rx) = mpsc::unbounded_channel();
        let config = DiffusionConfig {
            timing: knobs(Duration::from_secs(3600), Duration::from_millis(300)),
            concurrency,
            ..config()
        };
        (Server::new(config, false, elapsed_tx), elapsed_rx)
    }

    fn generate(request_id: &str) -> StageRequest {
        StageRequest::Generate(GenerateRequest {
            request_id: request_id.to_string(),
            ..request(None, 1, Some(1))
        })
    }

    fn message_type(frame: &[u8]) -> String {
        let message: Value = rmpv::decode::read_value(&mut &frame[..]).expect("decode");
        message
            .as_map()
            .expect("map")
            .iter()
            .find(|(key, _)| key.as_str() == Some("type"))
            .and_then(|(_, value)| value.as_str())
            .expect("type")
            .to_string()
    }

    fn metric(frame: &[u8], name: &str) -> f64 {
        let message: Value = rmpv::decode::read_value(&mut &frame[..]).expect("decode");
        let find = |map: &Value, key: &str| {
            map.as_map()
                .expect("map")
                .iter()
                .find(|(k, _)| k.as_str() == Some(key))
                .map(|(_, value)| value.clone())
                .unwrap_or_else(|| panic!("no {key}"))
        };
        find(&find(&find(&message, "output"), "metrics"), name)
            .as_f64()
            .expect("numeric metric")
    }

    fn counts(server: &Server) -> (u64, u64) {
        let stats = server.stats();
        (stats.running, stats.waiting)
    }

    #[tokio::test]
    async fn requests_past_the_concurrency_limit_wait_in_arrival_order() {
        let (mut server, _elapsed) = server(1);
        for id in ["a", "b", "c"] {
            let reaction = server.on_request(generate(id)).expect("request");
            assert!(reaction.replies.is_empty());
        }
        assert_eq!(counts(&server), (1, 2));
        assert!(server.running.contains_key("a"));

        let frame = server.on_elapsed("a").expect("elapsed").expect("result");
        assert_eq!(message_type(&frame), "result");
        assert_eq!(counts(&server), (1, 1));
        assert!(server.running.contains_key("b"), "b arrived before c");
        assert_eq!(server.unfinished(), 2);
    }

    #[tokio::test]
    async fn abort_drops_running_and_waiting_requests_without_a_reply() {
        let (mut server, _elapsed) = server(1);
        for id in ["a", "b", "c"] {
            server.on_request(generate(id)).expect("request");
        }

        let reaction = server
            .on_request(StageRequest::Abort(vec![
                "a".to_string(),
                "c".to_string(),
                "never-seen".to_string(),
            ]))
            .expect("abort");

        assert!(reaction.replies.is_empty());
        assert!(server.running.contains_key("b"), "the freed slot goes to b");
        assert_eq!(counts(&server), (1, 0));
        assert_eq!(server.stats().requests_aborted, 2);
        assert_eq!(server.on_elapsed("a").expect("elapsed"), None);
    }

    #[tokio::test]
    async fn rejected_request_gets_a_client_error_and_takes_no_slot() {
        let (mut server, _elapsed) = server(1);
        let huge = StageRequest::Generate(request(
            Some(ImageSize {
                width: 65_536,
                height: 65_536,
            }),
            1,
            None,
        ));

        let reaction = server.on_request(huge).expect("request");

        assert_eq!(reaction.replies.len(), 1);
        assert_eq!(message_type(&reaction.replies[0]), "error");
        assert_eq!(counts(&server), (0, 0));
        assert_eq!(
            server.stats(),
            Stats {
                requests_received: 1,
                requests_failed: 1,
                ..Stats::default()
            }
        );
    }

    #[tokio::test]
    async fn rpc_gets_a_reply_and_shutdown_stops_the_stage() {
        let (mut server, _elapsed) = server(1);
        let rpc = server
            .on_request(StageRequest::Rpc {
                rpc_id: "01".to_string(),
                method: "profile".to_string(),
            })
            .expect("rpc");
        assert_eq!(message_type(&rpc.replies[0]), "rpc_result");
        assert!(!rpc.shutdown);

        let shutdown = server.on_request(StageRequest::Shutdown).expect("shutdown");
        assert!(shutdown.shutdown);
        assert!(shutdown.replies.is_empty());
    }

    #[tokio::test]
    async fn resubmitted_request_id_replaces_the_earlier_request() {
        let (mut server, _elapsed) = server(2);
        server.on_request(generate("a")).expect("request");
        server.on_request(generate("a")).expect("request");
        assert_eq!(counts(&server), (1, 0));
    }

    #[tokio::test]
    async fn elapsed_timer_reports_the_request_id() {
        let (elapsed_tx, mut elapsed_rx) = mpsc::unbounded_channel();
        let instant = DiffusionConfig {
            timing: knobs(Duration::ZERO, Duration::ZERO),
            ..config()
        };
        let mut server = Server::new(instant, false, elapsed_tx);
        server.on_request(generate("a")).expect("request");

        let elapsed = tokio::time::timeout(Duration::from_secs(5), elapsed_rx.recv())
            .await
            .expect("timer fired")
            .expect("request id");
        assert_eq!(elapsed, "a");
    }

    fn try_control(
        server: &mut Server,
        request: ControlRequest<DiffusionKnobsPatch>,
    ) -> Result<ControlReply<DiffusionKnobs>, ControlError> {
        let (reply, mut answer) = oneshot::channel();
        server.on_control(ControlCall { request, reply });
        answer.try_recv().expect("answered")
    }

    fn control(
        server: &mut Server,
        request: ControlRequest<DiffusionKnobsPatch>,
    ) -> ControlReply<DiffusionKnobs> {
        try_control(server, request).expect("ok")
    }

    fn trace_server() -> Server {
        let (elapsed_tx, _elapsed_rx) = mpsc::unbounded_channel();
        Server::new(trace_config(), false, elapsed_tx)
    }

    #[tokio::test]
    async fn config_under_a_trace_reports_the_fitted_model() {
        let mut server = trace_server();
        let ControlReply::Config(knobs) = control(&mut server, ControlRequest::GetConfig) else {
            panic!("expected a config");
        };
        assert_eq!(knobs.timing, TimingSource::Trace);
        assert_eq!((knobs.overhead, knobs.step_time), (80, 25));
    }

    #[tokio::test]
    async fn latency_patch_under_a_trace_is_a_conflict_and_changes_nothing() {
        let mut server = trace_server();
        for patch in [
            DiffusionKnobsPatch {
                step_time: Some(7),
                concurrency: Some(4),
                ..DiffusionKnobsPatch::default()
            },
            DiffusionKnobsPatch {
                overhead: Some(7),
                ..DiffusionKnobsPatch::default()
            },
        ] {
            let error = try_control(&mut server, ControlRequest::PatchConfig(Box::new(patch)))
                .expect_err("latency patch");
            assert_eq!(error.status, 409);
            assert!(error.error.contains("--latency-trace"));
        }
        let ControlReply::Config(knobs) = control(&mut server, ControlRequest::GetConfig) else {
            panic!("expected a config");
        };
        assert_eq!(
            knobs.concurrency, 1,
            "a rejected patch applies none of its fields"
        );
        assert_eq!((knobs.overhead, knobs.step_time), (80, 25));
    }

    #[tokio::test]
    async fn capacity_patch_under_a_trace_applies() {
        let mut server = trace_server();
        let patch = DiffusionKnobsPatch {
            concurrency: Some(3),
            default_steps: Some(8),
            log_requests: Some(true),
            ..DiffusionKnobsPatch::default()
        };
        let ControlReply::Config(knobs) =
            control(&mut server, ControlRequest::PatchConfig(Box::new(patch)))
        else {
            panic!("expected a config");
        };
        assert_eq!(knobs.timing, TimingSource::Trace);
        assert_eq!((knobs.concurrency, knobs.default_steps), (3, 8));
        assert!(knobs.log_requests);
    }

    /// A stage replaying requests that held a slot for an hour and then took
    /// another hour to post-process.
    fn tailed_server() -> Server {
        let mut tailed = recorded(0.0, Some(SQUARE_512), Some(4), 7_200_000.0);
        tailed.stage_metrics = BTreeMap::from([(EXEC_TIME_METRIC.to_string(), 3_600_000.0)]);
        let (elapsed_tx, _elapsed_rx) = mpsc::unbounded_channel();
        let config = DiffusionConfig {
            timing: Timing::Trace(DiffusionReplay::from_records(&[tailed]).expect("replay")),
            ..config()
        };
        Server::new(config, false, elapsed_tx)
    }

    fn tailed(request_id: &str) -> StageRequest {
        StageRequest::Generate(GenerateRequest {
            request_id: request_id.to_string(),
            ..request(Some(SQUARE_512), 1, Some(4))
        })
    }

    #[tokio::test]
    async fn recorded_tail_frees_the_slot_before_the_result_is_sent() {
        let mut server = tailed_server();
        server.on_request(tailed("a")).expect("request");
        server.on_request(tailed("b")).expect("request");
        assert_eq!(counts(&server), (1, 1));

        assert_eq!(
            server.on_elapsed("a").expect("slot elapsed"),
            None,
            "a has its tail to wait out"
        );
        assert!(server.running.contains_key("b"), "a released its slot");
        assert!(server.finishing.contains_key("a"));
        assert_eq!(
            counts(&server),
            (2, 0),
            "a finishing request counts as running"
        );
        assert_eq!(server.unfinished(), 2);
        assert_eq!(server.stats().requests_completed, 0);

        let frame = server
            .on_elapsed("a")
            .expect("tail elapsed")
            .expect("result");
        assert_eq!(message_type(&frame), "result");
        assert_eq!(
            metric(&frame, "diffusion_engine_exec_time_ms").round(),
            3_600_000.0,
            "the tail is not part of the stage's exec time"
        );
        assert_eq!(counts(&server), (1, 0));
        assert_eq!(server.stats().requests_completed, 1);
    }

    #[tokio::test]
    async fn abort_during_the_tail_drops_the_result() {
        let mut server = tailed_server();
        server.on_request(tailed("a")).expect("request");
        server.on_elapsed("a").expect("slot elapsed");

        server
            .on_request(StageRequest::Abort(vec!["a".to_string()]))
            .expect("abort");

        assert_eq!(server.stats().requests_aborted, 1);
        assert_eq!(counts(&server), (0, 0));
        assert_eq!(server.on_elapsed("a").expect("elapsed"), None);
    }

    #[tokio::test]
    async fn result_reports_the_time_a_request_waited_for_a_slot() {
        let (mut server, _elapsed) = server(1);
        server.on_request(generate("a")).expect("request");
        server.on_request(generate("b")).expect("request");
        tokio::time::sleep(Duration::from_millis(60)).await;

        let first = server.on_elapsed("a").expect("elapsed").expect("result");
        let first_wait = metric(&first, "scheduler_queue_wait_ms");
        assert!(
            first_wait < 30.0,
            "a started at once, waited {first_wait} ms"
        );

        let second = server.on_elapsed("b").expect("elapsed").expect("result");
        let second_wait = metric(&second, "scheduler_queue_wait_ms");
        assert!(second_wait >= 60.0, "b waited for a, got {second_wait} ms");
        let exec = metric(&second, "diffusion_engine_exec_time_ms");
        assert!(
            (exec - (second_wait + 3_600_000.0 + 300.0)).abs() < 1e-3,
            "exec time runs from admission, so it includes the wait: {exec} ms"
        );
    }

    #[tokio::test]
    async fn config_reports_the_knobs_in_flag_units() {
        let (mut server, _elapsed) = server(3);
        let ControlReply::Config(knobs) = control(&mut server, ControlRequest::GetConfig) else {
            panic!("expected a config");
        };
        assert_eq!(knobs.timing, TimingSource::Knobs);
        assert_eq!(knobs.step_time, 3_600_000);
        assert_eq!(knobs.overhead, 300);
        assert_eq!(knobs.default_steps, 50);
        assert_eq!((knobs.default_width, knobs.default_height), (1024, 1024));
        assert_eq!(knobs.concurrency, 3);
        assert!(!knobs.log_requests);
    }

    #[tokio::test]
    async fn patch_changes_only_the_fields_it_names() {
        let (mut server, _elapsed) = server(1);
        let patch = DiffusionKnobsPatch {
            step_time: Some(7),
            default_width: Some(640),
            log_requests: Some(true),
            ..DiffusionKnobsPatch::default()
        };
        let ControlReply::Config(knobs) =
            control(&mut server, ControlRequest::PatchConfig(Box::new(patch)))
        else {
            panic!("expected a config");
        };
        assert_eq!(knobs.step_time, 7);
        assert_eq!((knobs.default_width, knobs.default_height), (640, 1024));
        assert_eq!(knobs.overhead, 300);
        assert!(knobs.log_requests);
    }

    #[tokio::test]
    async fn raising_concurrency_starts_waiting_requests() {
        let (mut server, _elapsed) = server(1);
        for id in ["a", "b", "c"] {
            server.on_request(generate(id)).expect("request");
        }
        assert_eq!(counts(&server), (1, 2));

        let patch = DiffusionKnobsPatch {
            concurrency: Some(2),
            ..DiffusionKnobsPatch::default()
        };
        control(&mut server, ControlRequest::PatchConfig(Box::new(patch)));

        assert_eq!(counts(&server), (2, 1));
    }

    #[tokio::test]
    async fn stats_count_outcomes_and_reset_keeps_the_gauges() {
        let (mut server, _elapsed) = server(1);
        for id in ["a", "b", "c"] {
            server.on_request(generate(id)).expect("request");
        }
        server.on_elapsed("a").expect("elapsed");
        server
            .on_request(StageRequest::Abort(vec!["b".to_string()]))
            .expect("abort");

        let ControlReply::Stats(stats) = control(&mut server, ControlRequest::GetStats) else {
            panic!("expected stats");
        };
        assert_eq!(
            stats,
            Stats {
                requests_received: 3,
                requests_completed: 1,
                requests_failed: 0,
                requests_aborted: 1,
                running: 1,
                waiting: 0,
            }
        );

        let ControlReply::Stats(reset) = control(&mut server, ControlRequest::ResetStats) else {
            panic!("expected stats");
        };
        assert_eq!(
            reset,
            Stats {
                running: 1,
                ..Stats::default()
            }
        );
    }

    #[test]
    fn patch_rejects_zero_for_counts_and_sizes() {
        for patch in [
            DiffusionKnobsPatch {
                concurrency: Some(0),
                ..DiffusionKnobsPatch::default()
            },
            DiffusionKnobsPatch {
                default_steps: Some(0),
                ..DiffusionKnobsPatch::default()
            },
            DiffusionKnobsPatch {
                default_width: Some(0),
                ..DiffusionKnobsPatch::default()
            },
            DiffusionKnobsPatch {
                default_height: Some(0),
                ..DiffusionKnobsPatch::default()
            },
        ] {
            assert!(patch.validate().is_err(), "{patch:?} should be rejected");
        }
        let instant = DiffusionKnobsPatch {
            step_time: Some(0),
            overhead: Some(0),
            ..DiffusionKnobsPatch::default()
        };
        assert!(instant.validate().is_ok());
    }
}
