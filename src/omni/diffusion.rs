//! A mock vLLM-Omni diffusion stage.
//!
//! Answers the head's generate requests with flat gray images after a modeled
//! delay: a fixed overhead plus a per-step time that scales with pixel count
//! and image count. `concurrency` requests execute at once and the rest queue
//! in arrival order.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context as _, Result};
use sim_protocol::omni::diffusion::{
    self as wire, DiffusionSockets, GenerateRequest, GeneratedImages, ImageSize, StageRequest,
};
use tokio::sync::{Semaphore, mpsc};
use tokio::task::AbortHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};
use zeromq::prelude::{SocketRecv as _, SocketSend as _};
use zeromq::{PushSocket, ZmqMessage};

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

/// A request whose modeled execution has elapsed.
struct Finished {
    request_id: String,
    frame: Vec<u8>,
}

fn result_frame(request_id: &str, job: Job, exec_time: Duration) -> Result<Vec<u8>> {
    wire::encode_result(&GeneratedImages {
        request_id,
        size: job.size,
        count: job.count,
        num_inference_steps: job.steps,
        exec_time,
    })
    .or_else(|error| wire::encode_error(request_id, &format!("{error:#}"), INTERNAL_ERROR))
}

async fn send(responses: &mut PushSocket, frame: Vec<u8>) -> Result<()> {
    responses
        .send(ZmqMessage::from(frame))
        .await
        .context("sending diffusion response")
}

impl DiffusionStage {
    /// Attach to the head and serve requests until shutdown.
    pub(crate) async fn run(&self, log_requests: bool, shutdown: CancellationToken) -> Result<()> {
        let config = self.config;
        serve(&self.stage, config, log_requests, shutdown).await
    }
}

async fn serve(
    stage: &Stage,
    config: DiffusionConfig,
    log_requests: bool,
    shutdown: CancellationToken,
) -> Result<()> {
    let attach = async {
        let replica = stage.register().await?;
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
    info!(?config, "diffusion stage attached to omni head");

    let permits = Arc::new(Semaphore::new(config.concurrency as usize));
    let (finished_tx, mut finished_rx) = mpsc::unbounded_channel::<Finished>();
    let mut running: HashMap<String, AbortHandle> = HashMap::new();

    let outcome = loop {
        unfinished.set(running.len());
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => break Ok(()),

            Some(finished) = finished_rx.recv() => {
                if running.remove(&finished.request_id).is_none() {
                    continue;
                }
                if log_requests {
                    info!(request_id = %finished.request_id, "diffusion request finished");
                }
                if let Err(error) = send(&mut responses, finished.frame).await {
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
                match request {
                    StageRequest::Generate(request) => {
                        let job = match config.job(&request) {
                            Ok(job) => job,
                            Err(reason) => {
                                warn!(request_id = %request.request_id, reason, "rejecting diffusion request");
                                let rejection = match wire::encode_error(&request.request_id, &reason, BAD_REQUEST) {
                                    Ok(frame) => send(&mut responses, frame).await,
                                    Err(error) => Err(error),
                                };
                                if let Err(error) = rejection {
                                    break Err(error);
                                }
                                continue;
                            }
                        };
                        let exec_time = config.exec_time(job);
                        if log_requests {
                            info!(
                                request_id = %request.request_id,
                                width = job.size.width,
                                height = job.size.height,
                                images = job.count,
                                steps = job.steps,
                                input_images = request.input_images,
                                exec_ms = exec_time.as_millis() as u64,
                                queued = running.len(),
                                "diffusion request started"
                            );
                        }
                        let permits = Arc::clone(&permits);
                        let finished_tx = finished_tx.clone();
                        let request_id = request.request_id.clone();
                        let task = tokio::spawn(async move {
                            let Ok(_permit) = permits.acquire_owned().await else {
                                return;
                            };
                            tokio::time::sleep(exec_time).await;
                            match result_frame(&request_id, job, exec_time) {
                                Ok(frame) => {
                                    finished_tx.send(Finished { request_id, frame }).ok();
                                }
                                Err(error) => {
                                    warn!(request_id, %error, "failed to encode diffusion result");
                                }
                            }
                        });
                        if let Some(replaced) = running.insert(request.request_id, task.abort_handle()) {
                            replaced.abort();
                        }
                    }
                    StageRequest::Abort(request_ids) => {
                        for request_id in request_ids {
                            if let Some(task) = running.remove(&request_id) {
                                task.abort();
                                if log_requests {
                                    info!(request_id, "diffusion request aborted");
                                }
                            }
                        }
                    }
                    StageRequest::Rpc { rpc_id, method } => {
                        debug!(method, "diffusion rpc; replying nil");
                        let reply = match wire::encode_rpc_result(&rpc_id) {
                            Ok(frame) => send(&mut responses, frame).await,
                            Err(error) => Err(error),
                        };
                        if let Err(error) = reply {
                            break Err(error);
                        }
                    }
                    StageRequest::Shutdown => {
                        info!("omni head asked the diffusion stage to shut down");
                        break Ok(());
                    }
                    StageRequest::Unknown(kind) => {
                        warn!(kind, "unknown diffusion message type; ignoring");
                    }
                }
            }
        }
    };

    for task in running.into_values() {
        task.abort();
    }
    heartbeats.stop().await;
    info!("diffusion stage shut down");
    outcome
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use sim_protocol::omni::diffusion::{GenerateRequest, ImageSize};

    use crate::omni::diffusion::{DiffusionConfig, Job, MAX_REQUEST_PIXELS};

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
}
