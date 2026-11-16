//! A mock vLLM-Omni diffusion stage.
//!
//! Answers the head's generate requests with flat gray images after a modeled
//! delay: a fixed overhead plus a per-step time that scales with pixel count
//! and image count. `concurrency` requests execute at once and the rest queue
//! in arrival order.
//!
//! [`Server`] is the stage's whole state and is only touched by the loop in
//! [`DiffusionStage::run`], between socket reads. Control calls arrive on a
//! channel of their own, so a config patch lands between two requests.

use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};
use sim_protocol::omni::diffusion::{
    self as wire, DiffusionSockets, GenerateRequest, GeneratedImages, ImageSize, StageRequest,
};
use tokio::sync::mpsc;
use tokio::task::AbortHandle;
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
/// Pixel count `step_time` is quoted for.
const REFERENCE_PIXELS: u64 = 1024 * 1024;
/// Pixels one request may return across all of its images.
const MAX_REQUEST_PIXELS: u64 = 64 * 1024 * 1024;
const BAD_REQUEST: u16 = 400;
const INTERNAL_ERROR: u16 = 500;

/// Latency model and capacity of a mock diffusion stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DiffusionConfig {
    /// One denoising step for one image of [`REFERENCE_PIXELS`].
    pub step_time: Duration,
    /// Fixed cost per request.
    pub overhead: Duration,
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
}

impl DiffusionConfig {
    fn job(&self, request: &GenerateRequest) -> Result<Job, String> {
        let job = Job {
            size: request.size.unwrap_or(self.default_size),
            count: request.num_outputs,
            steps: request.num_inference_steps.unwrap_or(self.default_steps),
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

    fn exec_time(&self, job: Job) -> Duration {
        let scale = job.size.pixels() as f64 / REFERENCE_PIXELS as f64;
        let steps = f64::from(job.steps) * f64::from(job.count) * scale;
        self.overhead + self.step_time.mul_f64(steps)
    }
}

/// The stage knobs that can change at runtime, as `GET /config` reports them.
/// Milliseconds for the latency fields, matching the `--omni-diffusion-*` flags.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffusionKnobs {
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
}

struct Running {
    job: Job,
    exec_time: Duration,
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
    running: HashMap<String, Running>,
    /// Receives a request id when its modeled execution has elapsed.
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
            elapsed_tx,
            received: 0,
            completed: 0,
            failed: 0,
            aborted: 0,
        }
    }

    fn unfinished(&self) -> usize {
        self.waiting.len() + self.running.len()
    }

    /// Move waiting requests into free execution slots, in arrival order.
    fn start_ready(&mut self) {
        while self.running.len() < self.config.concurrency as usize {
            let Some(Waiting { request_id, job }) = self.waiting.pop_front() else {
                return;
            };
            let exec_time = self.config.exec_time(job);
            let elapsed_tx = self.elapsed_tx.clone();
            let elapsed_id = request_id.clone();
            let timer = tokio::spawn(async move {
                tokio::time::sleep(exec_time).await;
                elapsed_tx.send(elapsed_id).ok();
            })
            .abort_handle();
            if self.log_requests {
                info!(
                    request_id,
                    exec_ms = exec_time.as_millis() as u64,
                    waiting = self.waiting.len(),
                    "diffusion request executing"
                );
            }
            self.running.insert(
                request_id,
                Running {
                    job,
                    exec_time,
                    timer,
                },
            );
        }
    }

    fn drop_request(&mut self, request_id: &str) -> bool {
        if let Some(running) = self.running.remove(request_id) {
            running.timer.abort();
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

    /// The result frame for a request whose execution has elapsed. `None` when
    /// the request was aborted in the meantime.
    fn on_elapsed(&mut self, request_id: &str) -> Result<Option<Vec<u8>>> {
        let Some(Running { job, exec_time, .. }) = self.running.remove(request_id) else {
            return Ok(None);
        };
        self.completed += 1;
        if self.log_requests {
            info!(request_id, "diffusion request finished");
        }
        self.start_ready();
        wire::encode_result(&GeneratedImages {
            request_id,
            size: job.size,
            count: job.count,
            num_inference_steps: job.steps,
            exec_time,
        })
        .or_else(|error| wire::encode_error(request_id, &format!("{error:#}"), INTERNAL_ERROR))
        .map(Some)
    }

    fn knobs(&self) -> DiffusionKnobs {
        DiffusionKnobs {
            step_time: self.config.step_time.as_millis() as u64,
            overhead: self.config.overhead.as_millis() as u64,
            default_steps: self.config.default_steps,
            default_width: self.config.default_size.width,
            default_height: self.config.default_size.height,
            concurrency: self.config.concurrency,
            log_requests: self.log_requests,
        }
    }

    fn apply(&mut self, patch: DiffusionKnobsPatch) {
        let config = &mut self.config;
        if let Some(step_time) = patch.step_time {
            config.step_time = Duration::from_millis(step_time);
        }
        if let Some(overhead) = patch.overhead {
            config.overhead = Duration::from_millis(overhead);
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
    }

    fn stats(&self) -> Stats {
        Stats {
            requests_received: self.received,
            requests_completed: self.completed,
            requests_failed: self.failed,
            requests_aborted: self.aborted,
            running: self.running.len() as u64,
            waiting: self.waiting.len() as u64,
        }
    }

    fn on_control(&mut self, call: StageControlCall) {
        let reply = match call.request {
            ControlRequest::GetConfig => ControlReply::Config(self.knobs()),
            ControlRequest::PatchConfig(patch) => {
                self.apply(*patch);
                ControlReply::Config(self.knobs())
            }
            ControlRequest::GetStats => ControlReply::Stats(self.stats()),
            ControlRequest::ResetStats => {
                self.received = 0;
                self.completed = 0;
                self.failed = 0;
                self.aborted = 0;
                ControlReply::Stats(self.stats())
            }
        };
        call.reply.send(Ok(reply)).ok();
    }

    fn abort_all(&mut self) {
        for (_, running) in self.running.drain() {
            running.timer.abort();
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
        let mut server = Server::new(self.config, log_requests, elapsed_tx);

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
    use std::time::Duration;

    use rmpv::Value;
    use sim_protocol::omni::diffusion::{GenerateRequest, ImageSize, StageRequest};
    use tokio::sync::{mpsc, oneshot};

    use crate::control::{ControlCall, ControlReply, ControlRequest, Patch, Stats};
    use crate::omni::diffusion::{
        DiffusionConfig, DiffusionKnobs, DiffusionKnobsPatch, Job, MAX_REQUEST_PIXELS, Server,
    };

    const SQUARE_1024: ImageSize = ImageSize {
        width: 1024,
        height: 1024,
    };

    fn config() -> DiffusionConfig {
        DiffusionConfig {
            step_time: Duration::from_millis(40),
            overhead: Duration::from_millis(300),
            default_steps: 50,
            default_size: SQUARE_1024,
            concurrency: 1,
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
                steps: 50
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
                steps: 4
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
        };
        assert_eq!(
            config().exec_time(job),
            Duration::from_millis(300 + 50 * 40)
        );
    }

    #[test]
    fn exec_time_scales_with_pixels_and_image_count() {
        let quarter = Job {
            size: ImageSize {
                width: 512,
                height: 512,
            },
            count: 1,
            steps: 20,
        };
        assert_eq!(
            config().exec_time(quarter),
            Duration::from_millis(300 + 20 * 40 / 4)
        );

        let four_images = Job {
            count: 4,
            ..quarter
        };
        assert_eq!(
            config().exec_time(four_images),
            Duration::from_millis(300 + 20 * 40)
        );
    }

    #[test]
    fn unset_knobs_make_requests_instant() {
        let instant = DiffusionConfig {
            step_time: Duration::ZERO,
            overhead: Duration::ZERO,
            ..config()
        };
        let job = Job {
            size: SQUARE_1024,
            count: 2,
            steps: 50,
        };
        assert_eq!(instant.exec_time(job), Duration::ZERO);
    }

    /// A server whose requests never elapse on their own: tests drive
    /// completion through `on_elapsed`.
    fn server(concurrency: u32) -> (Server, mpsc::UnboundedReceiver<String>) {
        let (elapsed_tx, elapsed_rx) = mpsc::unbounded_channel();
        let config = DiffusionConfig {
            step_time: Duration::from_secs(3600),
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
            step_time: Duration::ZERO,
            overhead: Duration::ZERO,
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

    fn control(
        server: &mut Server,
        request: ControlRequest<DiffusionKnobsPatch>,
    ) -> ControlReply<DiffusionKnobs> {
        let (reply, mut answer) = oneshot::channel();
        server.on_control(ControlCall { request, reply });
        answer.try_recv().expect("answered").expect("ok")
    }

    #[tokio::test]
    async fn config_reports_the_knobs_in_flag_units() {
        let (mut server, _elapsed) = server(3);
        let ControlReply::Config(knobs) = control(&mut server, ControlRequest::GetConfig) else {
            panic!("expected a config");
        };
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
