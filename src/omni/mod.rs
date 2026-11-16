//! Attaching `play` to a vLLM-Omni head as a headless stage replica.
//!
//! Every stage kind attaches the same way: register with the head's
//! `OmniMasterServer`, handshake on the address it hands back, then report
//! status to the head's coordinator ([`Stage`], [`Replica`], [`Heartbeats`]).
//! What differs is the data path. An [`LlmStage`] is the engine-core loop with
//! omni's request and output tails ([`crate::io::Codec`]); a
//! [`DiffusionStage`] is its own loop.

use std::time::Duration;

use anyhow::{Result, bail};
use clap::Args;
use sim_protocol::omni::attach::{
    CoordinatorClient, ReplicaEventType, ReplicaStatus, StageRegistration, register_stage,
};
use sim_protocol::omni::diffusion::ImageSize;
use sim_protocol::omni::llm::AudioOutput;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::engine_core::UnfinishedGauge;
use crate::omni::diffusion::{DiffusionConfig, DiffusionStage};

pub(crate) mod diffusion;

const REGISTRATION_TIMEOUT: Duration = Duration::from_secs(30);
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);
const DOWN_REPORT_TIMEOUT: Duration = Duration::from_secs(2);

/// Which omni runtime a stage belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum StageKind {
    /// An engine-core stage: thinker, talker, code2wav.
    Llm,
    /// A diffusion stage: image generation and editing.
    Diffusion,
}

fn parse_image_size(value: &str) -> Result<ImageSize, String> {
    let parsed = value.split_once(['x', 'X']).and_then(|(width, height)| {
        Some(ImageSize {
            width: width.parse().ok()?,
            height: height.parse().ok()?,
        })
    });
    match parsed {
        Some(size) if size.pixels() > 0 => Ok(size),
        _ => Err(format!(
            "expected WIDTHxHEIGHT, e.g. 1024x1024, got `{value}`"
        )),
    }
}

/// vLLM-Omni stage options. Setting `--omni-master-address` turns `play` into
/// one stage of an omni pipeline.
#[derive(Debug, Clone, Args)]
#[command(next_help_heading = "vLLM-Omni stage")]
pub struct OmniOpt {
    /// vLLM-Omni head `OmniMasterServer` endpoint (`tcp://host:port`). Registers this
    /// process as a headless replica of `--omni-stage-id` and takes the handshake
    /// address from the head instead of `--handshake-address`.
    #[arg(long = "omni-master-address", requires = "stage_id")]
    pub master_address: Option<String>,

    /// Stage of the omni pipeline this process serves.
    #[arg(long = "omni-stage-id", requires = "master_address")]
    pub stage_id: Option<u32>,

    /// Host the omni head records for this replica's connector endpoints.
    #[arg(long = "omni-replica-address", requires = "master_address")]
    pub replica_address: Option<String>,

    /// Runtime the stage belongs to. A diffusion stage serves image generation and
    /// editing and uses the `--omni-diffusion-*` latency knobs instead of the token ones.
    #[arg(
        long = "omni-stage-kind",
        value_enum,
        default_value_t = StageKind::Llm,
        requires = "master_address"
    )]
    pub stage_kind: StageKind,

    /// Waveform samples per generated token. Set on the pipeline's final audio stage
    /// (code2wav): every output then carries that much silence as `multimodal_output`,
    /// which the omni head returns as the response audio. At a 24 kHz head default,
    /// 480 makes each token 20 ms of audio.
    #[arg(
        long = "omni-audio-samples-per-token",
        requires = "master_address",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub audio_samples_per_token: Option<u32>,

    /// Sample rate reported with the audio. Unset leaves the head's default (24 kHz).
    #[arg(long = "omni-audio-sample-rate", requires = "audio_samples_per_token")]
    pub audio_sample_rate: Option<u32>,

    /// Milliseconds one denoising step takes for one 1024x1024 image. Scales linearly
    /// with pixel count and with the number of images in the request.
    #[arg(long = "omni-diffusion-step-time", default_value_t = 0)]
    pub diffusion_step_time: u64,

    /// Fixed milliseconds per diffusion request (text encoding, VAE decode).
    #[arg(long = "omni-diffusion-overhead", default_value_t = 0)]
    pub diffusion_overhead: u64,

    /// Denoising steps for a request that does not set `num_inference_steps`.
    #[arg(
        long = "omni-diffusion-default-steps",
        default_value_t = 50,
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub diffusion_default_steps: u32,

    /// Image size for a request that does not set one.
    #[arg(
        long = "omni-diffusion-default-size",
        default_value = "1024x1024",
        value_parser = parse_image_size
    )]
    pub diffusion_default_size: ImageSize,

    /// Diffusion requests executed at once; the rest queue in arrival order.
    #[arg(
        long = "omni-diffusion-concurrency",
        default_value_t = 1,
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub diffusion_concurrency: u32,
}

impl OmniOpt {
    /// The stage this process serves, when `--omni-master-address` is set.
    pub(crate) fn stage(&self) -> Result<Option<OmniStage>> {
        let (Some(master_address), Some(stage_id)) = (self.master_address.clone(), self.stage_id)
        else {
            return Ok(None);
        };
        let stage = Stage {
            master_address,
            stage_id,
            replica_address: self.replica_address.clone(),
        };
        let audio = self
            .audio_samples_per_token
            .map(|samples_per_token| AudioOutput {
                samples_per_token,
                sample_rate: self.audio_sample_rate,
            });
        Ok(Some(match self.stage_kind {
            StageKind::Llm => OmniStage::Llm(LlmStage { stage, audio }),
            StageKind::Diffusion => {
                if audio.is_some() {
                    bail!("--omni-audio-samples-per-token does not apply to a diffusion stage");
                }
                OmniStage::Diffusion(DiffusionStage {
                    stage,
                    config: DiffusionConfig {
                        step_time: Duration::from_millis(self.diffusion_step_time),
                        overhead: Duration::from_millis(self.diffusion_overhead),
                        default_steps: self.diffusion_default_steps,
                        default_size: self.diffusion_default_size,
                        concurrency: self.diffusion_concurrency,
                    },
                })
            }
        }))
    }
}

/// The stage `--omni-*` asks this process to be.
#[derive(Debug, Clone)]
pub(crate) enum OmniStage {
    Llm(LlmStage),
    Diffusion(DiffusionStage),
}

/// An engine-core stage: the mock engine behind an omni head.
#[derive(Debug, Clone)]
pub(crate) struct LlmStage {
    pub stage: Stage,
    /// Set on the pipeline's final audio stage.
    pub audio: Option<AudioOutput>,
}

/// One stage of an omni pipeline, on the head that runs it.
#[derive(Debug, Clone)]
pub(crate) struct Stage {
    master_address: String,
    stage_id: u32,
    replica_address: Option<String>,
}

impl Stage {
    pub(crate) async fn register(&self) -> Result<Replica> {
        let registration = register_stage(
            &self.master_address,
            self.stage_id,
            self.replica_address.as_deref(),
            REGISTRATION_TIMEOUT,
        )
        .await?;
        info!(
            stage_id = self.stage_id,
            replica_id = registration.replica_id,
            handshake_address = %registration.handshake_address,
            coordinator = ?registration.coordinator_router_address,
            "registered with omni master"
        );
        Ok(Replica {
            stage_id: self.stage_id,
            registration,
        })
    }
}

/// A replica slot the head allocated for this process.
#[derive(Debug)]
pub(crate) struct Replica {
    stage_id: u32,
    registration: StageRegistration,
}

impl Replica {
    pub(crate) fn handshake_address(&self) -> &str {
        &self.registration.handshake_address
    }

    /// Report this replica up to the head's coordinator and keep heartbeating.
    /// `input_address` / `output_address` are its data addresses from the
    /// handshake INIT, which the head routes by. Call once the data sockets are
    /// connected: the head sends requests as soon as the replica is up.
    pub(crate) async fn announce(
        &self,
        input_address: &str,
        output_address: &str,
        unfinished: UnfinishedGauge,
    ) -> Result<Heartbeats> {
        let stop = CancellationToken::new();
        let Some(coordinator_address) = &self.registration.coordinator_router_address else {
            return Ok(Heartbeats { stop, task: None });
        };
        let mut client = CoordinatorClient::connect(
            coordinator_address,
            self.stage_id,
            input_address.to_string(),
            output_address.to_string(),
        )
        .await?;
        client
            .send(
                ReplicaEventType::Update,
                ReplicaStatus::Up,
                unfinished.get(),
            )
            .await?;

        let stopped = stop.clone();
        let task = tokio::spawn(async move {
            loop {
                tokio::select! {
                    biased;
                    _ = stopped.cancelled() => {
                        if let Err(error) = client
                            .send(ReplicaEventType::Update, ReplicaStatus::Down, 0)
                            .await
                        {
                            warn!(%error, "failed to report replica down to omni coordinator");
                        }
                        return;
                    }
                    _ = tokio::time::sleep(HEARTBEAT_INTERVAL) => {
                        if let Err(error) = client
                            .send(ReplicaEventType::Heartbeat, ReplicaStatus::Up, unfinished.get())
                            .await
                        {
                            warn!(%error, "omni coordinator heartbeat failed");
                        }
                    }
                }
            }
        });
        Ok(Heartbeats {
            stop,
            task: Some(task),
        })
    }
}

/// A replica's running status reports to the head's coordinator.
pub(crate) struct Heartbeats {
    stop: CancellationToken,
    task: Option<JoinHandle<()>>,
}

impl Heartbeats {
    /// Report the replica down and stop.
    pub(crate) async fn stop(self) {
        self.stop.cancel();
        if let Some(task) = self.task
            && tokio::time::timeout(DOWN_REPORT_TIMEOUT, task)
                .await
                .is_err()
        {
            warn!("timed out reporting replica down to omni coordinator");
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;
    use sim_protocol::omni::diffusion::ImageSize;

    use crate::omni::{LlmStage, OmniOpt, OmniStage, parse_image_size};

    #[derive(Debug, Parser)]
    struct Cli {
        #[command(flatten)]
        omni: OmniOpt,
    }

    fn parse(args: &[&str]) -> Result<OmniOpt, clap::Error> {
        Cli::try_parse_from(std::iter::once("play").chain(args.iter().copied())).map(|cli| cli.omni)
    }

    const ATTACH: [&str; 4] = [
        "--omni-master-address",
        "tcp://127.0.0.1:21212",
        "--omni-stage-id",
        "2",
    ];

    fn attached(extra: &[&str]) -> Result<OmniOpt, clap::Error> {
        parse(&[&ATTACH[..], extra].concat())
    }

    #[test]
    fn no_omni_flags_means_no_stage() {
        let opt = parse(&[]).expect("parse");
        assert!(opt.stage().expect("stage").is_none());
    }

    #[test]
    fn master_address_and_stage_id_need_each_other() {
        assert!(parse(&["--omni-master-address", "tcp://127.0.0.1:1"]).is_err());
        assert!(parse(&["--omni-stage-id", "0"]).is_err());
    }

    #[test]
    fn stage_flags_need_a_master_address() {
        assert!(parse(&["--omni-stage-kind", "diffusion"]).is_err());
        assert!(parse(&["--omni-audio-samples-per-token", "480"]).is_err());
    }

    #[test]
    fn llm_stage_is_the_default_role() {
        let stage = attached(&[]).expect("parse").stage().expect("stage");
        assert!(matches!(
            stage,
            Some(OmniStage::Llm(LlmStage { audio: None, .. }))
        ));
    }

    #[test]
    fn audio_flags_configure_the_final_stage() {
        let opt = attached(&[
            "--omni-audio-samples-per-token",
            "480",
            "--omni-audio-sample-rate",
            "44100",
        ])
        .expect("parse");
        let Some(OmniStage::Llm(LlmStage {
            audio: Some(audio), ..
        })) = opt.stage().expect("stage")
        else {
            panic!("expected an llm stage with audio");
        };
        assert_eq!(audio.samples_per_token, 480);
        assert_eq!(audio.sample_rate, Some(44_100));
    }

    #[test]
    fn audio_sample_rate_needs_samples_per_token() {
        assert!(attached(&["--omni-audio-sample-rate", "44100"]).is_err());
        assert!(attached(&["--omni-audio-samples-per-token", "0"]).is_err());
    }

    #[test]
    fn diffusion_flags_build_the_diffusion_config() {
        let opt = attached(&[
            "--omni-stage-kind",
            "diffusion",
            "--omni-diffusion-step-time",
            "40",
            "--omni-diffusion-overhead",
            "300",
            "--omni-diffusion-default-steps",
            "28",
            "--omni-diffusion-default-size",
            "768x512",
            "--omni-diffusion-concurrency",
            "2",
        ])
        .expect("parse");
        let Some(OmniStage::Diffusion(stage)) = opt.stage().expect("stage") else {
            panic!("expected a diffusion stage");
        };
        let config = stage.config;
        assert_eq!(config.step_time.as_millis(), 40);
        assert_eq!(config.overhead.as_millis(), 300);
        assert_eq!(config.default_steps, 28);
        assert_eq!(
            config.default_size,
            ImageSize {
                width: 768,
                height: 512
            }
        );
        assert_eq!(config.concurrency, 2);
    }

    #[test]
    fn diffusion_stage_rejects_audio_flags() {
        let opt = attached(&[
            "--omni-stage-kind",
            "diffusion",
            "--omni-audio-samples-per-token",
            "480",
        ])
        .expect("parse");
        assert!(opt.stage().is_err());
    }

    #[test]
    fn image_size_parses_width_by_height() {
        assert_eq!(
            parse_image_size("1280x720"),
            Ok(ImageSize {
                width: 1280,
                height: 720
            })
        );
        assert!(parse_image_size("1024X1024").is_ok());
        for bad in ["1024", "x", "0x512", "512x0", "axb", "512x512x3", ""] {
            assert!(parse_image_size(bad).is_err(), "{bad} should not parse");
        }
    }
}
