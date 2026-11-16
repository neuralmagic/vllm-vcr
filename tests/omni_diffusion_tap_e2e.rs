//! End-to-end tests for recording and replaying a vLLM-Omni diffusion stage.
//!
//! Recording chains the scripted omni head in `omni_head`, the real tap
//! (`sim_tap::diffusion::run_diffusion_tap`), and a stage, all over real ZMQ
//! sockets:
//!
//! ```text
//!   scripted head <--> tap <--> stage
//! ```
//!
//! The stage is the real `vllm_vcr::run` task where the test is about what
//! lands in the trace, and a stage scripted by hand where it is about the
//! bytes the tap relays. Replay attaches `play --latency-trace` straight to a
//! scripted head.
//!
//! The contract under test:
//!   - The stage registers with the tap and the head receives that
//!     registration; the stage's status events reach the head carrying the
//!     head-side data addresses.
//!   - Every frame crosses the tap unchanged in both directions, whether or
//!     not the tap can decode it, the raw death sentinel included.
//!   - Each request that reaches an outcome becomes one record: finished with
//!     the stage's own metrics, error with its status, or aborted.
//!   - A mock stage given the trace replays a recorded shape's times in
//!     order, times an unrecorded shape from the model fitted to the trace,
//!     and refuses a latency patch.

use std::io::BufReader;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use rmpv::Value;
use sim_protocol::frontend_connect::OpenHandshake;
use sim_protocol::omni::attach::{
    CoordinatorClient, ReplicaEventType, ReplicaStatus, register_stage,
};
use sim_protocol::omni::diffusion::{self as wire, DiffusionSockets, PROC_DEAD};
use sim_protocol::vllm::engine_id_from_index;
use sim_tap::diffusion::{DiffusionTapConfig, DiffusionTapMeta, run_diffusion_tap};
use sim_tap::omni::OmniTap;
use sim_trace::diffusion::{
    DiffusionOutcome, DiffusionRecord, DiffusionReplay, DiffusionShape, ImageSize, read_trace,
    step_units,
};
use sim_trace::trace::{TraceKind, TraceMeta};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use vllm_vcr::{Opt, run};
use zeromq::prelude::{SocketRecv as _, SocketSend as _};
use zeromq::{PullSocket, PushSocket, ZmqMessage};

use crate::omni_head::{
    Head, Stage, control, decode, encode, field, generate, hex, map, unbound_endpoint, within,
};

mod omni_head;

/// The real tap, running against a scripted head.
struct Tap {
    task: JoinHandle<(anyhow::Result<()>, Vec<u8>)>,
    shutdown: CancellationToken,
    master_address: String,
    input_address: String,
}

impl Tap {
    fn spawn(head: &Head) -> Self {
        let master_address = unbound_endpoint();
        let input_address = unbound_endpoint();
        let config = DiffusionTapConfig {
            omni: OmniTap {
                head_master: head.master_address.clone(),
                master_listen: master_address.clone(),
                coordinator_listen: unbound_endpoint(),
            },
            engine_handshake: unbound_endpoint(),
            input_address: input_address.clone(),
            output_address: unbound_endpoint(),
        };
        let meta = DiffusionTapMeta {
            model: "test-diffusion-model".to_string(),
            gpu: Some("none".to_string()),
            ..DiffusionTapMeta::default()
        };
        let shutdown = CancellationToken::new();
        let task = tokio::spawn({
            let shutdown = shutdown.clone();
            async move {
                let mut trace = Vec::new();
                let result = run_diffusion_tap(config, meta, &mut trace, shutdown).await;
                (result, trace)
            }
        });
        Self {
            task,
            shutdown,
            master_address,
            input_address,
        }
    }

    /// Stop the tap and parse the trace it wrote.
    async fn finish(self) -> (TraceMeta, Vec<DiffusionRecord>, Vec<u8>) {
        self.shutdown.cancel();
        let (result, trace) = within("tap exit", self.task).await.expect("tap join");
        result.expect("tap result");
        let (meta, records) = read_trace(BufReader::new(trace.as_slice())).expect("read trace");
        (meta, records, trace)
    }
}

/// The head's data sockets for a diffusion stage.
struct Data {
    requests: PushSocket,
    responses: PullSocket,
}

impl Data {
    async fn send(&mut self, frame: Vec<u8>) {
        self.requests
            .send(ZmqMessage::from(frame))
            .await
            .expect("send request");
    }

    async fn raw_response(&mut self) -> Vec<u8> {
        let frames = within("diffusion response", self.responses.recv())
            .await
            .expect("response")
            .into_vec();
        assert_eq!(frames.len(), 1, "diffusion responses are single-frame");
        frames[0].to_vec()
    }

    async fn response(&mut self) -> Value {
        decode(&self.raw_response().await)
    }

    /// Send a request and return its result's request id, the size of its
    /// first image, and how long the answer took.
    async fn timed(&mut self, frame: Vec<u8>) -> (String, ImageSize, Duration) {
        let sent = Instant::now();
        self.send(frame).await;
        let response = self.response().await;
        let elapsed = sent.elapsed();
        assert_eq!(field(&response, "type").as_str(), Some("result"));
        let output = field(&response, "output");
        let shape = field(
            &field(output, "images").as_array().expect("images")[0],
            "shape",
        )
        .as_array()
        .expect("shape")
        .clone();
        let dimension = |index: usize| shape[index].as_u64().expect("dimension") as u32;
        (
            field(output, "request_id")
                .as_str()
                .expect("request id")
                .to_string(),
            ImageSize {
                width: dimension(1),
                height: dimension(0),
            },
            elapsed,
        )
    }
}

const SLOW: ImageSize = ImageSize {
    width: 256,
    height: 256,
};
const SLOW_STEPS: u32 = 8;
const FAST: ImageSize = ImageSize {
    width: 128,
    height: 128,
};
const FAST_STEPS: u32 = 4;
/// `--omni-diffusion-step-time 400 --omni-diffusion-overhead 20`: the slow
/// shape is half a reference step-image per step, 220 ms; the fast one 45 ms.
const STAGE_KNOBS: [&str; 4] = [
    "--omni-diffusion-step-time",
    "400",
    "--omni-diffusion-overhead",
    "20",
];
const SLOW_MS: f64 = 220.0;
const FAST_MS: f64 = 45.0;

fn slow(request_id: &str) -> Vec<u8> {
    generate(request_id, SLOW.width, SLOW.height, SLOW_STEPS)
}

fn fast(request_id: &str) -> Vec<u8> {
    generate(request_id, FAST.width, FAST.height, FAST_STEPS)
}

/// An edit of one input image at the fast shape.
fn edit(request_id: &str) -> Vec<u8> {
    let input = map(vec![
        ("__pil_image__", Value::from(true)),
        ("mode", Value::from("RGB")),
        (
            "shape",
            Value::Array(vec![Value::from(4), Value::from(6), Value::from(3)]),
        ),
        ("data", Value::Binary(vec![7; 4 * 6 * 3])),
    ]);
    encode(&map(vec![
        ("type", Value::from("add_request")),
        ("request_id", Value::from(request_id)),
        (
            "prompt",
            map(vec![
                ("prompt", Value::from("make it night")),
                (
                    "multi_modal_data",
                    map(vec![("image", Value::Array(vec![input]))]),
                ),
            ]),
        ),
        (
            "sampling_params",
            map(vec![
                ("num_outputs_per_prompt", Value::from(1)),
                ("width", Value::from(FAST.width)),
                ("height", Value::from(FAST.height)),
                ("num_inference_steps", Value::from(FAST_STEPS)),
            ]),
        ),
    ]))
}

fn abort(request_id: &str) -> Vec<u8> {
    encode(&map(vec![
        ("type", Value::from("abort")),
        ("request_ids", Value::Array(vec![Value::from(request_id)])),
    ]))
}

fn shape(size: ImageSize, steps: u32, input_images: u32) -> DiffusionShape {
    DiffusionShape {
        size: Some(size),
        steps: Some(steps),
        images: 1,
        input_images,
    }
}

/// Record a `play` diffusion stage through the tap under a fixed workload:
/// two slow requests and a fast one, a queued request aborted behind the
/// second slow one, an oversized request, and an edit.
async fn record_session() -> (TraceMeta, Vec<DiffusionRecord>, Vec<u8>) {
    let mut head = Head::start().await;
    let tap = Tap::spawn(&head);
    let mut flags = vec!["--omni-stage-kind", "diffusion"];
    flags.extend_from_slice(&STAGE_KNOBS);
    let stage = Stage::spawn(&tap.master_address, "0", &flags);

    let registration = head.accept_registration().await;
    assert_eq!(field(&registration, "stage_id"), &Value::from(0));
    assert_eq!(field(&registration, "replica_id"), &Value::from(-1));
    assert!(!field(&registration, "stage_config").is_nil());
    head.handshake().await;
    let (requests, responses) = head.bind_diffusion().await;
    let mut data = Data {
        requests,
        responses,
    };
    assert_ne!(tap.input_address, head.input_address);
    head.expect_replica_up(0).await;

    let (id, size, _) = data.timed(slow("img_gen-slow-1")).await;
    assert_eq!((id.as_str(), size), ("img_gen-slow-1", SLOW));
    let (id, size, _) = data.timed(fast("img_gen-fast-1")).await;
    assert_eq!((id.as_str(), size), ("img_gen-fast-1", FAST));

    data.send(slow("img_gen-slow-2")).await;
    data.send(fast("img_gen-queued")).await;
    data.send(abort("img_gen-queued")).await;
    let response = data.response().await;
    assert_eq!(
        field(field(&response, "output"), "request_id").as_str(),
        Some("img_gen-slow-2"),
        "the aborted request never answers"
    );

    data.send(generate("img_gen-huge", 65_536, 65_536, 1)).await;
    let response = data.response().await;
    assert_eq!(field(&response, "type").as_str(), Some("error"));
    assert_eq!(field(&response, "status_code"), &Value::from(400));

    let (id, size, _) = data.timed(edit("img_edit-1")).await;
    assert_eq!((id.as_str(), size), ("img_edit-1", FAST));

    stage.shutdown.cancel();
    head.expect_replica_down().await;
    stage.finished().await;
    tap.finish().await
}

#[tokio::test]
async fn tap_records_a_diffusion_stage_between_the_head_and_the_stage() {
    let (meta, records, _) = record_session().await;

    assert_eq!(meta.kind, TraceKind::Diffusion);
    assert_eq!(meta.source.as_deref(), Some("tap"));
    assert_eq!(meta.model.as_deref(), Some("test-diffusion-model"));
    assert_eq!(meta.gpu.as_deref(), Some("none"));
    assert_eq!(meta.extra.get("omni_stage_id"), Some(&serde_json::json!(0)));
    assert!(meta.config_hash.is_some());

    let outcomes: Vec<(DiffusionShape, DiffusionOutcome, u32)> = records
        .iter()
        .map(|record| (record.shape, record.outcome, record.concurrency))
        .collect();
    let huge = ImageSize {
        width: 65_536,
        height: 65_536,
    };
    assert_eq!(
        outcomes,
        [
            (shape(SLOW, SLOW_STEPS, 0), DiffusionOutcome::Finished, 1),
            (shape(FAST, FAST_STEPS, 0), DiffusionOutcome::Finished, 1),
            (shape(FAST, FAST_STEPS, 0), DiffusionOutcome::Aborted, 2),
            (shape(SLOW, SLOW_STEPS, 0), DiffusionOutcome::Finished, 1),
            (
                shape(huge, 1, 0),
                DiffusionOutcome::Error { status: Some(400) },
                1
            ),
            (shape(FAST, FAST_STEPS, 1), DiffusionOutcome::Finished, 1),
        ],
        "records are written as requests close"
    );

    for (index, expected_ms, size) in [
        (0, SLOW_MS, SLOW),
        (1, FAST_MS, FAST),
        (3, SLOW_MS, SLOW),
        (5, FAST_MS, FAST),
    ] {
        let record = &records[index];
        assert!(
            (expected_ms..expected_ms + 2000.0).contains(&record.latency_ms),
            "record {index}: latency {} ms for a {expected_ms} ms request",
            record.latency_ms
        );
        assert_eq!(record.output_size, Some(size));
        let exec_ms = record.stage_metrics["diffusion_engine_exec_time_ms"];
        let waited_ms = record.stage_metrics["scheduler_queue_wait_ms"];
        assert!(
            (exec_ms - waited_ms - expected_ms).abs() < 1e-6,
            "record {index}: the stage's own exec time is recorded, got {exec_ms} ms"
        );
        assert_eq!(
            record.stage_metrics["num_inference_steps"],
            f64::from(record.shape.steps.expect("steps"))
        );
        assert!(record.service_ms() <= record.latency_ms);
    }

    let aborted = &records[2];
    assert!(aborted.stage_metrics.is_empty());
    assert_eq!(aborted.output_size, None);
    assert!(
        aborted.latency_ms < SLOW_MS,
        "an aborted request's latency runs to the abort"
    );
    assert_eq!(records[4].output_size, None);

    let mut by_arrival: Vec<f64> = records.iter().map(|record| record.arrival_ms).collect();
    assert!(by_arrival.iter().all(|arrival| *arrival >= 0.0));
    let slow_2 = by_arrival.remove(3);
    let queued = by_arrival.remove(2);
    assert!(slow_2 <= queued, "the queued request arrived second");
}

/// The trace on disk, removed on drop.
struct TraceFile(PathBuf);

impl TraceFile {
    fn write(name: &str, contents: &[u8]) -> Self {
        let path = std::env::temp_dir().join(format!(
            "vllm-vcr-diffusion-e2e-{name}-{}.jsonl",
            std::process::id()
        ));
        std::fs::write(&path, contents).expect("write trace");
        Self(path)
    }

    fn path(&self) -> &str {
        self.0.to_str().expect("utf-8 temp path")
    }
}

impl Drop for TraceFile {
    fn drop(&mut self) {
        std::fs::remove_file(&self.0).ok();
    }
}

fn at_least(elapsed: Duration, ms: f64) -> bool {
    elapsed.as_secs_f64() * 1000.0 >= ms
}

#[tokio::test]
async fn mock_stage_replays_the_recorded_trace() {
    let (_, records, trace) = record_session().await;
    let file = TraceFile::write("replay", &trace);
    let slow_recorded: Vec<f64> = records
        .iter()
        .filter(|record| record.shape == shape(SLOW, SLOW_STEPS, 0))
        .map(DiffusionRecord::service_ms)
        .collect();
    assert_eq!(slow_recorded.len(), 2);
    let fast_recorded = records[1].service_ms();
    let edit_recorded = records[5].service_ms();
    let fitted = DiffusionReplay::from_records(&records)
        .expect("replay")
        .fitted();

    let control_address = unbound_endpoint().replace("tcp://", "");
    let mut head = Head::start().await;
    let stage = Stage::spawn(
        &head.master_address,
        "0",
        &[
            "--omni-stage-kind",
            "diffusion",
            "--latency-trace",
            file.path(),
            "--control-address",
            &control_address,
        ],
    );
    head.accept_registration().await;
    head.handshake().await;
    let (requests, responses) = head.bind_diffusion().await;
    let mut data = Data {
        requests,
        responses,
    };
    head.expect_replica_up(0).await;

    let (status, config) = control(&control_address, "GET", "/config", "").await;
    assert_eq!(status, 200);
    assert_eq!(config["timing"], "trace");
    assert_eq!(
        config["step_time"],
        (fitted.step_time.as_secs_f64() * 1000.0).round() as u64
    );
    let (status, error) =
        control(&control_address, "PATCH", "/config", r#"{"step_time": 1}"#).await;
    assert_eq!(status, 409);
    assert!(
        error["error"]
            .as_str()
            .is_some_and(|error| error.contains("--latency-trace"))
    );

    for (index, recorded_ms) in slow_recorded.iter().chain(&slow_recorded[..1]).enumerate() {
        let (_, size, elapsed) = data
            .timed(slow(&format!("img_gen-replay-slow-{index}")))
            .await;
        assert_eq!(size, SLOW);
        assert!(
            at_least(elapsed, *recorded_ms) && !at_least(elapsed, recorded_ms + 2000.0),
            "slow request {index} took {elapsed:?}, recorded {recorded_ms} ms"
        );
    }
    let (_, size, elapsed) = data.timed(fast("img_gen-replay-fast")).await;
    assert_eq!(size, FAST);
    assert!(
        at_least(elapsed, fast_recorded) && !at_least(elapsed, slow_recorded[0]),
        "the fast shape replays its own time: {elapsed:?}, recorded {fast_recorded} ms"
    );
    let (_, _, elapsed) = data.timed(edit("img_edit-replay")).await;
    assert!(
        at_least(elapsed, edit_recorded) && !at_least(elapsed, slow_recorded[0]),
        "an edit replays the recorded edit: {elapsed:?}, recorded {edit_recorded} ms"
    );

    let unseen_steps = 4;
    let fitted_ms = fitted
        .exec_time(step_units(SLOW, unseen_steps, 1))
        .as_secs_f64()
        * 1000.0;
    assert!(
        fitted_ms > fast_recorded && fitted_ms < slow_recorded[0],
        "the fit puts the unrecorded shape between the recorded ones, got {fitted_ms} ms"
    );
    let (_, size, elapsed) = data
        .timed(generate(
            "img_gen-replay-unseen",
            SLOW.width,
            SLOW.height,
            unseen_steps,
        ))
        .await;
    assert_eq!(size, SLOW);
    assert!(
        at_least(elapsed, fitted_ms) && !at_least(elapsed, slow_recorded[0]),
        "an unrecorded shape is timed by the fit: {elapsed:?}, fitted {fitted_ms} ms"
    );

    stage.shutdown.cancel();
    head.expect_replica_down().await;
    stage.finished().await;
}

#[tokio::test]
async fn llm_engine_refuses_a_diffusion_trace_as_its_latency_trace() {
    let trace = concat!(
        r#"{"meta":{"kind":"diffusion"}}"#,
        "\n",
        r#"{"arrival_ms":0.0,"images":1,"concurrency":1,"latency_ms":5.0,"outcome":"finished"}"#,
        "\n",
    );
    let file = TraceFile::write("wrong-kind", trace.as_bytes());
    let opt = Opt::parse_from(["play", "--latency-trace", file.path()]);

    let error = within("play exit", run(opt, CancellationToken::new()))
        .await
        .expect_err("a diffusion trace has no token timing");
    let message = format!("{error:#}");
    assert!(message.contains("diffusion trace"), "got: {message}");
}

/// A diffusion stage played by hand: the real attach, then whatever bytes the
/// test sends.
struct ScriptedStage {
    sockets: DiffusionSockets,
    _coordinator: CoordinatorClient,
}

impl ScriptedStage {
    async fn attach(master_address: String) -> Self {
        let timeout = Duration::from_secs(10);
        let registration = register_stage(&master_address, 0, None, timeout)
            .await
            .expect("register with the tap");
        let handshake = OpenHandshake::open(
            &registration.handshake_address,
            engine_id_from_index(0).expect("engine id"),
            false,
            true,
            timeout,
        )
        .await
        .expect("handshake with the tap");
        let sockets = wire::attach(handshake).await.expect("data sockets");
        assert_eq!(sockets.input_address, registration.input_address);

        let mut coordinator = CoordinatorClient::connect(
            registration
                .coordinator_router_address
                .as_deref()
                .expect("the tap offers a coordinator"),
            0,
            sockets.input_address.clone(),
            sockets.output_address.clone(),
        )
        .await
        .expect("coordinator");
        coordinator
            .send(ReplicaEventType::Update, ReplicaStatus::Up, 0)
            .await
            .expect("report up");
        Self {
            sockets,
            _coordinator: coordinator,
        }
    }

    async fn request(&mut self) -> Vec<u8> {
        let frames = within("request at the stage", self.sockets.requests.recv())
            .await
            .expect("request")
            .into_vec();
        assert_eq!(frames.len(), 1);
        frames[0].to_vec()
    }

    async fn send(&mut self, frame: Vec<u8>) {
        self.sockets
            .responses
            .send(ZmqMessage::from(frame))
            .await
            .expect("send response");
    }
}

/// Frames vLLM-Omni v0.32.0's own encoder produced (deploy/omni/encode_groundtruth.py).
macro_rules! fixture {
    ($name:literal) => {
        hex(include_str!(concat!(
            "../crates/sim-protocol/tests/fixtures/omni_v0.32.0_diffusion_",
            $name,
            ".hex"
        )))
    };
}

#[tokio::test]
async fn tap_relays_every_frame_unchanged_in_both_directions() {
    let mut head = Head::start().await;
    let tap = Tap::spawn(&head);
    let attach = tokio::spawn(ScriptedStage::attach(tap.master_address.clone()));

    head.accept_registration().await;
    head.handshake().await;
    let (requests, responses) = head.bind_diffusion().await;
    let mut data = Data {
        requests,
        responses,
    };
    head.expect_replica_up(0).await;
    let mut stage = within("stage attach", attach).await.expect("attach join");

    let collective_rpc = encode(&map(vec![
        ("type", Value::from("collective_rpc")),
        ("rpc_id", Value::from("0123456789abcdef")),
        ("method", Value::from("profile")),
        ("timeout", Value::Nil),
        ("args", Value::Array(Vec::new())),
        ("kwargs", Value::Map(Vec::new())),
    ]));
    let from_head = [
        fixture!("t2i"),
        fixture!("bare"),
        fixture!("edit"),
        generate("img_gen-a", 64, 64, 2),
        fixture!("abort"),
        collective_rpc,
        b"\xc1 not msgpack".to_vec(),
        encode(&map(vec![("type", Value::from("shutdown"))])),
    ];
    for frame in &from_head {
        data.send(frame.clone()).await;
        assert_eq!(&stage.request().await, frame);
    }

    let rpc_result = encode(&map(vec![
        ("type", Value::from("rpc_result")),
        ("rpc_id", Value::from("0123456789abcdef")),
        ("result", Value::Nil),
    ]));
    let from_stage = [
        fixture!("result_started"),
        fixture!("result"),
        fixture!("error"),
        fixture!("abort_metrics"),
        rpc_result,
        b"\xc1 not msgpack".to_vec(),
        PROC_DEAD.to_vec(),
    ];
    for frame in &from_stage {
        stage.send(frame.clone()).await;
        assert_eq!(&data.raw_response().await, frame);
    }

    let (meta, records, _) = tap.finish().await;
    assert_eq!(meta.kind, TraceKind::Diffusion);

    let outcomes: Vec<(DiffusionShape, DiffusionOutcome)> = records
        .iter()
        .map(|record| (record.shape, record.outcome))
        .collect();
    let t2i = DiffusionShape {
        size: Some(ImageSize {
            width: 768,
            height: 512,
        }),
        steps: Some(30),
        images: 2,
        input_images: 0,
    };
    let bare = DiffusionShape {
        size: None,
        steps: None,
        images: 1,
        input_images: 0,
    };
    assert_eq!(
        outcomes,
        [
            (
                shape(
                    ImageSize {
                        width: 64,
                        height: 64
                    },
                    2,
                    0
                ),
                DiffusionOutcome::Aborted
            ),
            (t2i, DiffusionOutcome::Finished),
            (bare, DiffusionOutcome::Error { status: Some(400) }),
        ],
        "the edit never answered, so it has no record"
    );

    let finished = &records[1];
    assert_eq!(
        finished.output_size,
        Some(ImageSize {
            width: 6,
            height: 4
        })
    );
    assert_eq!(
        finished.stage_metrics["diffusion_engine_exec_time_ms"],
        1500.5
    );
    assert_eq!(finished.stage_metrics["scheduler_queue_wait_ms"], 420.0);
    assert_eq!(finished.concurrency, 1);
    assert_eq!(records[0].concurrency, 4);
}
