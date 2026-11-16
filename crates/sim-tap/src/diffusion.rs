//! Tapping a vLLM-Omni diffusion stage.
//!
//! The stage attaches through the tap the way an LLM stage does
//! ([`crate::omni`]), but its data path is not engine-core: single-frame
//! msgpack maps over a PUSH/PULL pair in each direction.
//!
//! ```text
//!   head  PUSH ──requests──> PULL  tap  PUSH ──requests──> PULL  stage
//!         PULL <─responses── PUSH       PULL <─responses── PUSH
//!        (binds)          (connects)   (binds)          (connects)
//! ```
//!
//! Every frame is relayed unchanged. The tap decodes a copy to observe it: a
//! request opens on `add_request` and closes on the stage's `result` or
//! `error`, or on the head's `abort`, and each closed request becomes one
//! [`DiffusionRecord`].

use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::time::Duration;

use anyhow::{Context as _, Result, anyhow};
use sim_protocol::frontend_connect::OpenHandshake;
use sim_protocol::omni::diffusion::{
    self as wire, DiffusionSockets, StageRequest, StageResponse, decode_request, decode_response,
};
use sim_protocol::vllm::engine_id_from_index;
use sim_trace::diffusion::{
    DiffusionOutcome, DiffusionRecord, DiffusionShape, ImageSize, append_record, write_meta,
};
use sim_trace::trace::{TraceKind, TraceMeta};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};
use zeromq::prelude::{Socket as _, SocketRecv as _, SocketSend as _};
use zeromq::{PullSocket, PushSocket, ZmqError, ZmqMessage};

use crate::omni::{self, OmniTap, TapAddresses};
use crate::tap::{EngineHandshake, config_hash, ms_between};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// How often a send is retried while its PUSH socket has no peer.
const PEER_POLL: Duration = Duration::from_millis(10);

/// Where the tap meets an omni head and the diffusion stage it records.
#[derive(Debug, Clone)]
pub struct DiffusionTapConfig {
    pub omni: OmniTap,
    /// Handshake address the tap binds for the real stage.
    pub engine_handshake: String,
    /// Address the tap binds its request PUSH on, for the stage's PULL.
    pub input_address: String,
    /// Address the tap binds its response PULL on, for the stage's PUSH.
    pub output_address: String,
}

/// Inputs for a diffusion trace's meta line.
#[derive(Debug, Clone, Default)]
pub struct DiffusionTapMeta {
    pub model: String,
    pub gpu: Option<String>,
    /// Explicit `config_hash`. When `None` it is the fingerprint over `gpu`,
    /// `vllm_tag`, and `engine_config`.
    pub config_hash: Option<String>,
    /// vLLM tag the stage runs, as the caller states it.
    pub vllm_tag: Option<String>,
    pub engine_config: String,
}

fn build_meta(meta: &DiffusionTapMeta, stage_id: u32) -> TraceMeta {
    TraceMeta {
        kind: TraceKind::Diffusion,
        source: Some("tap".to_string()),
        model: (!meta.model.is_empty()).then(|| meta.model.clone()),
        gpu: meta.gpu.clone(),
        config_hash: Some(config_hash(
            meta.config_hash.as_deref(),
            meta.gpu.as_deref(),
            meta.vllm_tag.as_deref().unwrap_or_default(),
            &meta.engine_config,
        )),
        extra: HashMap::from([("omni_stage_id".to_string(), stage_id.into())]),
        ..TraceMeta::default()
    }
}

/// A request the head sent that has not reached an outcome.
struct InFlight {
    arrival: Instant,
    shape: DiffusionShape,
    concurrency: u32,
}

/// The requests in flight through the tap, and the records they close into.
struct Observer {
    capture_start: Instant,
    in_flight: HashMap<String, InFlight>,
}

impl Observer {
    fn new(capture_start: Instant) -> Self {
        Self {
            capture_start,
            in_flight: HashMap::new(),
        }
    }

    fn close(
        &mut self,
        request_id: &str,
        at: Instant,
        outcome: DiffusionOutcome,
        output_size: Option<ImageSize>,
        stage_metrics: BTreeMap<String, f64>,
    ) -> Option<DiffusionRecord> {
        let request = self.in_flight.remove(request_id)?;
        Some(DiffusionRecord {
            arrival_ms: ms_between(self.capture_start, request.arrival),
            shape: request.shape,
            concurrency: request.concurrency,
            latency_ms: ms_between(request.arrival, at),
            outcome,
            output_size,
            stage_metrics,
        })
    }

    /// Observe a frame from the head. An `abort` closes the requests it names
    /// that are still in flight.
    fn on_request(&mut self, frame: &[u8], at: Instant) -> Vec<DiffusionRecord> {
        match decode_request(frame) {
            Ok(StageRequest::Generate(request)) => {
                let shape = request.shape();
                let replaced = self.in_flight.remove(&request.request_id).is_some();
                let concurrency = u32::try_from(self.in_flight.len() + 1).unwrap_or(u32::MAX);
                debug!(
                    request_id = %request.request_id,
                    ?shape,
                    concurrency,
                    replaced,
                    "tap: observed diffusion request"
                );
                self.in_flight.insert(
                    request.request_id,
                    InFlight {
                        arrival: at,
                        shape,
                        concurrency,
                    },
                );
                Vec::new()
            }
            Ok(StageRequest::Abort(request_ids)) => request_ids
                .iter()
                .filter_map(|request_id| {
                    self.close(
                        request_id,
                        at,
                        DiffusionOutcome::Aborted,
                        None,
                        BTreeMap::new(),
                    )
                })
                .collect(),
            Ok(StageRequest::Rpc { .. } | StageRequest::Shutdown | StageRequest::Unknown(_)) => {
                Vec::new()
            }
            Err(error) => {
                warn!(%error, "tap: failed to decode a diffusion request for observation (still forwarded)");
                Vec::new()
            }
        }
    }

    /// Observe a frame from the stage. A finished `result` or an `error`
    /// closes the request it names.
    fn on_response(&mut self, frame: &[u8], at: Instant) -> Option<DiffusionRecord> {
        match decode_response(frame) {
            Ok(StageResponse::Finished(finished)) => self.close(
                &finished.request_id,
                at,
                DiffusionOutcome::Finished,
                finished.output_size,
                finished.metrics,
            ),
            Ok(StageResponse::Failed(failed)) => self.close(
                &failed.request_id,
                at,
                DiffusionOutcome::Error {
                    status: failed.status_code,
                },
                None,
                failed.metrics,
            ),
            Ok(StageResponse::Dead) => {
                error!(
                    in_flight = self.in_flight.len(),
                    "tap: the diffusion stage reported itself dead; its in-flight requests are not recorded"
                );
                None
            }
            Ok(StageResponse::Progress { .. } | StageResponse::Other(_)) => None,
            Err(error) => {
                warn!(%error, "tap: failed to decode a diffusion response for observation (still forwarded)");
                None
            }
        }
    }
}

/// The tap's sockets once both sides are attached.
struct Attached {
    stage_id: u32,
    from_head: PullSocket,
    to_head: PushSocket,
    to_stage: PushSocket,
    from_stage: PullSocket,
    status_relay: Option<JoinHandle<()>>,
}

/// Attach the stage through the tap. HELLO and READY reach the head when the
/// stage sends them.
async fn attach(config: &DiffusionTapConfig) -> Result<Attached> {
    let attachment = omni::accept_stage(
        &config.omni,
        &TapAddresses {
            handshake: &config.engine_handshake,
            input: &config.input_address,
            output: &config.output_address,
        },
    )
    .await?;
    let stage_id = attachment.stage_id;

    let mut to_stage = PushSocket::new();
    to_stage
        .bind(&config.input_address)
        .await
        .with_context(|| format!("binding tap request push at {}", config.input_address))?;
    let mut from_stage = PullSocket::new();
    from_stage
        .bind(&config.output_address)
        .await
        .with_context(|| format!("binding tap response pull at {}", config.output_address))?;

    let stage_handshake = EngineHandshake::accept(
        &config.engine_handshake,
        &config.input_address,
        &config.output_address,
    )
    .await?;
    let head_handshake = OpenHandshake::open(
        &attachment.head.handshake_address,
        engine_id_from_index(0)?,
        stage_handshake.hello.local.unwrap_or(false),
        stage_handshake.hello.headless.unwrap_or(true),
        CONNECT_TIMEOUT,
    )
    .await
    .map_err(|e| anyhow!("opening the handshake with the omni head: {e}"))?;
    info!(
        stage_id,
        "waiting for the real diffusion stage to report READY"
    );
    stage_handshake.ready().await?;

    let DiffusionSockets {
        requests: from_head,
        responses: to_head,
        input_address,
        output_address,
    } = wire::attach(head_handshake).await?;
    let status_relay = attachment.relay_status(input_address, output_address);

    Ok(Attached {
        stage_id,
        from_head,
        to_head,
        to_stage,
        from_stage,
        status_relay,
    })
}

/// Read a socket in a task of its own, stamping each message as it arrives.
fn drain(
    mut socket: PullSocket,
    what: &'static str,
) -> (
    mpsc::UnboundedReceiver<(Instant, ZmqMessage)>,
    JoinHandle<()>,
) {
    let (tx, rx) = mpsc::unbounded_channel();
    let task = tokio::spawn(async move {
        loop {
            match socket.recv().await {
                Ok(message) => {
                    if tx.send((Instant::now(), message)).is_err() {
                        return;
                    }
                }
                Err(error) => {
                    warn!(%error, what, "tap: recv error, stopping the drain");
                    return;
                }
            }
        }
    });
    (rx, task)
}

/// Send on a PUSH socket, waiting for a peer when there is none.
async fn push(
    socket: &mut PushSocket,
    mut message: ZmqMessage,
    what: &'static str,
    shutdown: &CancellationToken,
) -> Result<()> {
    let mut waiting = false;
    loop {
        match socket.send(message).await {
            Ok(()) => return Ok(()),
            Err(ZmqError::ReturnToSender {
                message: returned, ..
            }) => {
                if shutdown.is_cancelled() {
                    return Ok(());
                }
                if !waiting {
                    warn!(what, "tap: no peer connected yet, holding the frame");
                    waiting = true;
                }
                message = returned;
                tokio::time::sleep(PEER_POLL).await;
            }
            Err(error) => return Err(error).with_context(|| format!("forwarding {what}")),
        }
    }
}

fn write_record(writer: &mut impl Write, record: &DiffusionRecord) {
    debug!(?record, "tap: writing diffusion trace record");
    if let Err(error) = append_record(writer, record) {
        warn!(%error, "tap: failed to write trace record");
    }
    if let Err(error) = writer.flush() {
        warn!(%error, "tap: failed to flush trace writer");
    }
}

/// Run the tap between an omni head and a real diffusion stage until
/// shutdown or a transport failure, writing a diffusion trace to `writer`.
/// The meta line is written once the stage is attached.
pub async fn run_diffusion_tap<W: Write>(
    config: DiffusionTapConfig,
    meta: DiffusionTapMeta,
    writer: &mut W,
    shutdown: CancellationToken,
) -> Result<()> {
    let attached = tokio::select! {
        biased;
        _ = shutdown.cancelled() => {
            info!("tap shutting down before the stage attached");
            return Ok(());
        }
        attached = attach(&config) => attached?,
    };
    let Attached {
        stage_id,
        from_head,
        mut to_head,
        mut to_stage,
        from_stage,
        status_relay,
    } = attached;

    write_meta(writer, &build_meta(&meta, stage_id))?;
    writer.flush().context("flushing trace writer")?;
    info!(
        stage_id,
        "diffusion tap connected to both sides, forwarding frames"
    );

    let mut observer = Observer::new(Instant::now());
    let (mut requests, request_drain) = drain(from_head, "head requests");
    let (mut responses, response_drain) = drain(from_stage, "stage responses");

    let outcome: Result<()> = loop {
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => {
                info!("tap shutting down");
                break Ok(());
            }

            request = requests.recv() => {
                let Some((at, message)) = request else {
                    break Err(anyhow!("request drain task ended (head disconnected)"));
                };
                let frame = message.get(0).cloned();
                if let Err(error) = push(&mut to_stage, message, "a request to the stage", &shutdown).await {
                    break Err(error);
                }
                if let Some(frame) = frame {
                    for record in observer.on_request(&frame, at) {
                        write_record(writer, &record);
                    }
                }
            }

            response = responses.recv() => {
                let Some((at, message)) = response else {
                    break Err(anyhow!("response drain task ended (stage disconnected)"));
                };
                let frame = message.get(0).cloned();
                if let Err(error) = push(&mut to_head, message, "a response to the head", &shutdown).await {
                    break Err(error);
                }
                if let Some(record) = frame.and_then(|frame| observer.on_response(&frame, at)) {
                    write_record(writer, &record);
                }
            }
        }
    };

    request_drain.abort();
    response_drain.abort();
    if let Some(status_relay) = status_relay {
        status_relay.abort();
    }
    if !observer.in_flight.is_empty() {
        info!(
            in_flight = observer.in_flight.len(),
            "tap: requests still in flight at shutdown are not recorded"
        );
    }
    outcome?;
    writer.flush().context("flushing trace writer")
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::time::Duration;

    use rmpv::Value;
    use sim_protocol::omni::diffusion::{
        GeneratedImages, PROC_DEAD, encode_error, encode_result, encode_rpc_result,
    };
    use sim_trace::diffusion::{DiffusionOutcome, DiffusionShape, ImageSize};
    use sim_trace::trace::TraceKind;
    use tokio::time::Instant;

    use crate::diffusion::{DiffusionTapMeta, Observer, build_meta};

    fn groundtruth(hex: &str) -> Vec<u8> {
        let hex = hex.trim();
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex"))
            .collect()
    }

    /// Frames vLLM-Omni v0.32.0's own encoder produced
    /// (deploy/omni/encode_groundtruth.py).
    macro_rules! fixture {
        ($name:literal) => {
            groundtruth(include_str!(concat!(
                "../../sim-protocol/tests/fixtures/omni_v0.32.0_diffusion_",
                $name,
                ".hex"
            )))
        };
    }

    const T2I_ID: &str = "img_gen-0123456789abcdef-01234567";
    const BARE_ID: &str = "img_gen-fedcba9876543210-76543210";
    const EDIT_ID: &str = "img_edit-0123456789abcdef-89abcdef";

    fn encode(value: &Value) -> Vec<u8> {
        let mut out = Vec::new();
        rmpv::encode::write_value(&mut out, value).expect("encode");
        out
    }

    fn map(entries: Vec<(&str, Value)>) -> Value {
        Value::Map(
            entries
                .into_iter()
                .map(|(key, value)| (Value::from(key), value))
                .collect(),
        )
    }

    fn generate(request_id: &str) -> Vec<u8> {
        encode(&map(vec![
            ("type", Value::from("add_request")),
            ("request_id", Value::from(request_id)),
            ("prompt", map(vec![("prompt", Value::from("a red fox"))])),
            (
                "sampling_params",
                map(vec![
                    ("num_outputs_per_prompt", Value::from(1)),
                    ("width", Value::from(8)),
                    ("height", Value::from(8)),
                    ("num_inference_steps", Value::from(2)),
                ]),
            ),
        ]))
    }

    fn abort(request_ids: &[&str]) -> Vec<u8> {
        encode(&map(vec![
            ("type", Value::from("abort")),
            (
                "request_ids",
                Value::Array(request_ids.iter().map(|id| Value::from(*id)).collect()),
            ),
        ]))
    }

    fn result(request_id: &str) -> Vec<u8> {
        encode_result(&GeneratedImages {
            request_id,
            size: ImageSize {
                width: 8,
                height: 8,
            },
            count: 1,
            num_inference_steps: 2,
            queue_wait: Duration::ZERO,
            exec_time: Duration::from_millis(5),
        })
        .expect("encode")
    }

    fn after(start: Instant, ms: u64) -> Instant {
        start + Duration::from_millis(ms)
    }

    #[test]
    fn groundtruth_request_and_result_close_into_a_finished_record() {
        let start = Instant::now();
        let mut observer = Observer::new(start);

        assert!(
            observer
                .on_request(&fixture!("t2i"), after(start, 100))
                .is_empty()
        );
        assert_eq!(
            observer.on_response(&fixture!("result_started"), after(start, 150)),
            None,
            "a lifecycle marker does not finish the request"
        );
        let record = observer
            .on_response(&fixture!("result"), after(start, 1700))
            .expect("record");

        assert_eq!(record.arrival_ms, 100.0);
        assert_eq!(
            record.shape,
            DiffusionShape {
                size: Some(ImageSize {
                    width: 768,
                    height: 512
                }),
                steps: Some(30),
                images: 2,
                input_images: 0,
            }
        );
        assert_eq!(record.concurrency, 1);
        assert_eq!(record.latency_ms, 1600.0);
        assert_eq!(record.outcome, DiffusionOutcome::Finished);
        assert_eq!(
            record.output_size,
            Some(ImageSize {
                width: 6,
                height: 4
            }),
            "the size is read off the returned image"
        );
        assert_eq!(
            record.stage_metrics.get("diffusion_engine_exec_time_ms"),
            Some(&1500.5)
        );
        assert_eq!(
            record.stage_metrics.get("scheduler_queue_wait_ms"),
            Some(&420.0)
        );
        assert_eq!(record.service_ms(), 1180.0);
        assert!(observer.in_flight.is_empty());
    }

    #[test]
    fn groundtruth_edit_request_records_its_input_images() {
        let start = Instant::now();
        let mut observer = Observer::new(start);
        observer.on_request(&fixture!("edit"), start);
        let record = observer
            .on_response(&result(EDIT_ID), after(start, 40))
            .expect("record");
        assert_eq!(record.shape.input_images, 2);
        assert_eq!(record.shape.steps, None);
    }

    #[test]
    fn groundtruth_error_closes_into_an_error_record_with_its_status() {
        let start = Instant::now();
        let mut observer = Observer::new(start);
        observer.on_request(&fixture!("bare"), start);
        let record = observer
            .on_response(&fixture!("error"), after(start, 3))
            .expect("record");

        assert_eq!(
            record.shape,
            DiffusionShape {
                size: None,
                steps: None,
                images: 1,
                input_images: 0,
            }
        );
        assert_eq!(
            record.outcome,
            DiffusionOutcome::Error { status: Some(400) }
        );
        assert_eq!(record.latency_ms, 3.0);
        assert_eq!(record.output_size, None);
        assert_eq!(
            record.stage_metrics,
            BTreeMap::from([("scheduler_num_waiting_reqs".to_string(), 1.0)])
        );
    }

    #[test]
    fn concurrency_counts_the_requests_in_flight_at_arrival() {
        let start = Instant::now();
        let mut observer = Observer::new(start);
        for id in ["a", "b", "c"] {
            observer.on_request(&generate(id), start);
        }
        let concurrency = |observer: &mut Observer, id: &str| {
            observer
                .on_response(&result(id), after(start, 10))
                .expect("record")
                .concurrency
        };
        assert_eq!(concurrency(&mut observer, "a"), 1);
        assert_eq!(concurrency(&mut observer, "c"), 3);

        observer.on_request(&generate("d"), after(start, 20));
        assert_eq!(
            concurrency(&mut observer, "d"),
            2,
            "b is still in flight when d arrives"
        );
    }

    #[test]
    fn abort_closes_only_the_requests_still_in_flight() {
        let start = Instant::now();
        let mut observer = Observer::new(start);
        observer.on_request(&generate("img_gen-a"), start);
        observer.on_request(&generate("img_gen-b"), after(start, 5));
        observer
            .on_response(&result("img_gen-b"), after(start, 30))
            .expect("b finished");

        let records = observer.on_request(&fixture!("abort"), after(start, 50));

        assert_eq!(records.len(), 1, "b finished before the abort");
        assert_eq!(records[0].outcome, DiffusionOutcome::Aborted);
        assert_eq!(records[0].arrival_ms, 0.0);
        assert_eq!(records[0].latency_ms, 50.0);
        assert!(records[0].stage_metrics.is_empty());

        assert_eq!(
            observer.on_response(&result("img_gen-a"), after(start, 60)),
            None,
            "a result that lost the race with the abort is not a second record"
        );
        assert!(
            observer
                .on_request(&abort(&["never-seen"]), start)
                .is_empty()
        );
    }

    #[test]
    fn resubmitted_request_id_restarts_the_request() {
        let start = Instant::now();
        let mut observer = Observer::new(start);
        observer.on_request(&generate("a"), start);
        observer.on_request(&generate("a"), after(start, 100));
        let record = observer
            .on_response(&result("a"), after(start, 130))
            .expect("record");
        assert_eq!(record.arrival_ms, 100.0);
        assert_eq!(record.latency_ms, 30.0);
        assert_eq!(record.concurrency, 1);
    }

    #[test]
    fn frames_that_end_no_request_write_nothing() {
        let start = Instant::now();
        let mut observer = Observer::new(start);
        observer.on_request(&fixture!("t2i"), start);

        let rpc = encode(&map(vec![
            ("type", Value::from("collective_rpc")),
            ("rpc_id", Value::from("0123")),
            ("method", Value::from("profile")),
        ]));
        let shutdown = encode(&map(vec![("type", Value::from("shutdown"))]));
        for request in [rpc, shutdown, vec![0xc1], b"not msgpack".to_vec()] {
            assert!(observer.on_request(&request, start).is_empty());
        }

        let rpc_result = encode_rpc_result("0123").expect("encode");
        for response in [
            fixture!("abort_metrics"),
            rpc_result,
            PROC_DEAD.to_vec(),
            vec![0xc1],
            result("some-other-request"),
            encode_error("some-other-request", "nope", 500).expect("encode"),
        ] {
            assert_eq!(observer.on_response(&response, start), None);
        }
        assert_eq!(observer.in_flight.len(), 1);
        assert!(observer.in_flight.contains_key(T2I_ID));
        assert!(!observer.in_flight.contains_key(BARE_ID));
    }

    #[test]
    fn meta_marks_the_trace_as_diffusion_and_names_the_stage() {
        let meta = build_meta(
            &DiffusionTapMeta {
                model: "black-forest-labs/FLUX.2-klein-4B".to_string(),
                gpu: Some("H200".to_string()),
                ..DiffusionTapMeta::default()
            },
            3,
        );
        assert_eq!(meta.kind, TraceKind::Diffusion);
        assert_eq!(meta.source.as_deref(), Some("tap"));
        assert_eq!(
            meta.model.as_deref(),
            Some("black-forest-labs/FLUX.2-klein-4B")
        );
        assert_eq!(meta.extra.get("omni_stage_id"), Some(&serde_json::json!(3)));
        assert!(meta.config_hash.is_some());
        assert_eq!(meta.vllm_version, None);

        let unnamed = build_meta(&DiffusionTapMeta::default(), 0);
        assert_eq!(unnamed.model, None);

        let explicit = build_meta(
            &DiffusionTapMeta {
                config_hash: Some("manual".to_string()),
                ..DiffusionTapMeta::default()
            },
            0,
        );
        assert_eq!(explicit.config_hash.as_deref(), Some("manual"));
    }
}
