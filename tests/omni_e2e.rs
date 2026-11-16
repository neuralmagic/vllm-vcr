//! End-to-end tests for vLLM-Omni stages over real ZMQ transport.
//!
//! Each test runs the full `vllm_vcr::run` task against a scripted omni head:
//! real sockets, and the head's side of the wire written out here by hand. The
//! head mirrors what a v0.32.0 head does, including the part plain vLLM does
//! not: the data sockets are bound only after the stage reports READY.
//!
//! The contract under test:
//!   - A stage registers with the master, handshakes on the address it is
//!     handed, connects its data sockets after READY, and then reports up to
//!     the coordinator with the addresses from the handshake.
//!   - An LLM stage accepts a request carrying the omni tail and, as a final
//!     audio stage, returns `multimodal_output` on its outputs.
//!   - A diffusion stage serves generate requests, honors abort, and stops on
//!     the head's shutdown message.
//!   - Either stage reports down when it stops.

use std::future::Future;
use std::time::Duration;

use rmpv::Value;
use sim_protocol::vllm::{
    EngineCoreFinishReason, EngineCoreOutput, EngineCoreRequest, EngineCoreSamplingParams,
    Envelope, decode_msgpack, encode_msgpack, request_outputs,
};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use vllm_vcr::{Opt, run};
use zeromq::prelude::{Socket as _, SocketRecv as _, SocketSend as _};
use zeromq::{PullSocket, PushSocket, RouterSocket, ZmqMessage};

const TIMEOUT: Duration = Duration::from_secs(10);

async fn within<T>(what: &str, future: impl Future<Output = T>) -> T {
    tokio::time::timeout(TIMEOUT, future)
        .await
        .unwrap_or_else(|_| panic!("timed out: {what}"))
}

fn map(entries: Vec<(&str, Value)>) -> Value {
    Value::Map(
        entries
            .into_iter()
            .map(|(key, value)| (Value::from(key), value))
            .collect(),
    )
}

fn field<'a>(map: &'a Value, key: &str) -> &'a Value {
    map.as_map()
        .unwrap_or_else(|| panic!("not a map: {map}"))
        .iter()
        .find(|(k, _)| k.as_str() == Some(key))
        .map(|(_, v)| v)
        .unwrap_or_else(|| panic!("missing key {key} in {map}"))
}

fn encode(value: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, value).expect("encode");
    out
}

fn decode(bytes: &[u8]) -> Value {
    rmpv::decode::read_value(&mut &bytes[..]).expect("decode")
}

/// A local TCP endpoint nothing is bound to yet.
fn unbound_endpoint() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("reserve port");
    let port = listener.local_addr().expect("local addr").port();
    format!("tcp://127.0.0.1:{port}")
}

async fn bound_router() -> (RouterSocket, String) {
    let mut socket = RouterSocket::new();
    let endpoint = socket.bind("tcp://127.0.0.1:0").await.expect("bind");
    (socket, endpoint.to_string())
}

/// The head's control plane: master, handshake, and coordinator sockets, plus
/// the data addresses it will bind once the stage is READY.
struct Head {
    master: RouterSocket,
    master_address: String,
    handshake: RouterSocket,
    handshake_address: String,
    coordinator: RouterSocket,
    coordinator_address: String,
    input_address: String,
    output_address: String,
}

impl Head {
    async fn start() -> Self {
        let (master, master_address) = bound_router().await;
        let (handshake, handshake_address) = bound_router().await;
        let (coordinator, coordinator_address) = bound_router().await;
        Self {
            master,
            master_address,
            handshake,
            handshake_address,
            coordinator,
            coordinator_address,
            input_address: unbound_endpoint(),
            output_address: unbound_endpoint(),
        }
    }

    /// Answer one stage registration and return what the stage sent.
    async fn accept_registration(&mut self) -> Value {
        let frames = within("registration", self.master.recv())
            .await
            .expect("registration")
            .into_vec();
        let mut reply = ZmqMessage::from(frames[0].to_vec());
        reply.push_back(
            encode(&map(vec![
                (
                    "handshake_address",
                    Value::from(self.handshake_address.as_str()),
                ),
                ("input_address", Value::from(self.input_address.as_str())),
                ("output_address", Value::from(self.output_address.as_str())),
                ("replica_id", Value::from(0)),
                (
                    "coordinator_router_address",
                    Value::from(self.coordinator_address.as_str()),
                ),
            ]))
            .into(),
        );
        self.master.send(reply).await.expect("registration reply");
        decode(frames.last().expect("registration payload"))
    }

    /// HELLO, INIT, READY. Returns the stage's handshake identity.
    async fn handshake(&mut self) -> Vec<u8> {
        let hello = within("HELLO", self.handshake.recv())
            .await
            .expect("HELLO")
            .into_vec();
        let identity = hello[0].to_vec();
        let message = decode(&hello[1]);
        assert_eq!(field(&message, "status").as_str(), Some("HELLO"));
        assert_eq!(field(&message, "local"), &Value::from(false));
        assert_eq!(field(&message, "headless"), &Value::from(true));

        let mut init = ZmqMessage::from(identity.clone());
        init.push_back(
            encode(&map(vec![
                (
                    "addresses",
                    map(vec![
                        (
                            "inputs",
                            Value::Array(vec![Value::from(self.input_address.as_str())]),
                        ),
                        (
                            "outputs",
                            Value::Array(vec![Value::from(self.output_address.as_str())]),
                        ),
                        ("coordinator_input", Value::Nil),
                        ("coordinator_output", Value::Nil),
                        ("frontend_stats_publish_address", Value::Nil),
                    ]),
                ),
                ("parallel_config", Value::Map(Vec::new())),
            ]))
            .into(),
        );
        self.handshake.send(init).await.expect("INIT");

        let ready = within("READY", self.handshake.recv())
            .await
            .expect("READY")
            .into_vec();
        assert_eq!(
            field(&decode(&ready[1]), "status").as_str(),
            Some("READY"),
            "the stage must report READY before its data sockets can connect"
        );
        identity
    }

    async fn replica_event(&mut self) -> serde_json::Value {
        let frames = within("replica event", self.coordinator.recv())
            .await
            .expect("replica event")
            .into_vec();
        serde_json::from_slice(frames.last().expect("event payload")).expect("event json")
    }

    /// The stage's first report, sent once it is attached.
    async fn expect_replica_up(&mut self, stage_id: u32) {
        assert_eq!(
            self.replica_event().await,
            serde_json::json!({
                "input_addr": self.input_address,
                "output_addr": self.output_address,
                "stage_id": stage_id,
                "event_type": "update",
                "status": "up",
                "queue_length": 0,
            })
        );
    }

    async fn expect_replica_down(&mut self) {
        loop {
            let event = self.replica_event().await;
            if event["event_type"] == "heartbeat" {
                continue;
            }
            assert_eq!(event["event_type"], "update");
            assert_eq!(event["status"], "down");
            return;
        }
    }
}

struct Stage {
    task: Option<JoinHandle<anyhow::Result<()>>>,
    shutdown: CancellationToken,
}

impl Stage {
    fn spawn(head: &Head, stage_id: &str, flags: &[&str]) -> Self {
        let mut args = vec![
            "play",
            "--omni-master-address",
            head.master_address.as_str(),
            "--omni-stage-id",
            stage_id,
        ];
        args.extend_from_slice(flags);
        let opt = Opt::parse_from(&args);
        let shutdown = CancellationToken::new();
        let task = Some(tokio::spawn(run(opt, shutdown.clone())));
        Self { task, shutdown }
    }

    async fn finished(mut self) {
        let task = self.task.take().expect("stage task");
        within("stage exit", task)
            .await
            .expect("stage task join")
            .expect("stage result");
    }
}

impl Drop for Stage {
    fn drop(&mut self) {
        self.shutdown.cancel();
    }
}

fn request_with_tail(request: &EngineCoreRequest, tail: &[Value]) -> Vec<u8> {
    let Value::Array(mut fields) = decode(&encode_msgpack(request).expect("encode request")) else {
        panic!("request is not an array");
    };
    fields.extend_from_slice(tail);
    encode(&Value::Array(fields))
}

/// The samples in an output's `multimodal_output.audio`, when it has one.
fn audio_samples(output: &[Value], vllm_fields: usize) -> Option<u64> {
    let multimodal = output.get(vllm_fields)?;
    let tensor = field(multimodal, "audio")
        .as_array()
        .expect("tensor triple");
    assert_eq!(tensor[0].as_str(), Some("float32"));
    let shape = tensor[1].as_array().expect("shape");
    assert_eq!(shape.len(), 1);
    let samples = shape[0].as_u64().expect("samples");
    let Value::Ext(_, data) = &tensor[2] else {
        panic!("tensor data is not an ext");
    };
    assert_eq!(data.len() as u64, samples * 4);
    Some(samples)
}

#[tokio::test]
async fn llm_stage_attaches_and_returns_audio_for_a_request_with_an_omni_tail() {
    let mut head = Head::start().await;
    let stage = Stage::spawn(
        &head,
        "2",
        &[
            "--output-tokens",
            "3",
            "--omni-audio-samples-per-token",
            "10",
        ],
    );

    let registration = head.accept_registration().await;
    assert_eq!(field(&registration, "stage_id"), &Value::from(2));
    assert_eq!(field(&registration, "replica_id"), &Value::from(-1));
    assert!(!field(&registration, "stage_config").is_nil());

    let identity = head.handshake().await;

    let mut input = RouterSocket::new();
    input.bind(&head.input_address).await.expect("bind input");
    let mut output = PullSocket::new();
    output
        .bind(&head.output_address)
        .await
        .expect("bind output");

    let ready_response = within("ready response", input.recv())
        .await
        .expect("ready response")
        .into_vec();
    assert_eq!(ready_response[0].as_ref(), identity.as_slice());
    assert!(field(&decode(&ready_response[1]), "max_model_len").is_u64());

    head.expect_replica_up(2).await;

    let request = EngineCoreRequest {
        request_id: "chatcmpl-omni-1".to_string(),
        prompt_token_ids: Some(vec![42u32; 4]),
        sampling_params: Some(EngineCoreSamplingParams {
            max_tokens: 64,
            ..EngineCoreSamplingParams::for_test()
        }),
        ..Default::default()
    };
    let vllm_fields = decode(&encode_msgpack(&EngineCoreOutput::default()).expect("encode"))
        .as_array()
        .expect("output array")
        .len();
    let tail = [
        Value::Nil,
        Value::Nil,
        map(vec![("zmq_port", Value::from(50051))]),
    ];
    let mut add = ZmqMessage::from(identity);
    add.push_back(vec![0u8].into());
    add.push_back(request_with_tail(&request, &tail).into());
    input.send(add).await.expect("send add");

    let mut tokens = 0;
    let mut samples = 0;
    let finish_reason = loop {
        let frames = within("outputs", output.recv())
            .await
            .expect("outputs")
            .into_vec();
        let envelope: Envelope = decode_msgpack(&frames[0]).expect("envelope");
        let Value::Array(fields) = decode(&frames[0]) else {
            panic!("envelope is not an array");
        };
        let wire_outputs = fields[1].as_array().expect("outputs array");
        let outputs = request_outputs(&envelope);
        assert_eq!(wire_outputs.len(), outputs.len());

        let mut finished = None;
        for (output, wire) in outputs.iter().zip(wire_outputs) {
            assert_eq!(output.request_id, "chatcmpl-omni-1");
            let wire = wire.as_array().expect("output array");
            let chunk_samples = audio_samples(wire, vllm_fields);
            if output.new_token_ids.is_empty() {
                assert_eq!(chunk_samples, None);
            } else {
                assert_eq!(chunk_samples, Some(output.new_token_ids.len() as u64 * 10));
            }
            tokens += output.new_token_ids.len();
            samples += chunk_samples.unwrap_or(0);
            finished = finished.or(output.finish_reason);
        }
        if let Some(reason) = finished {
            break reason;
        }
    };
    assert_eq!(tokens, 3);
    assert_eq!(samples, 30);
    assert_eq!(finish_reason, EngineCoreFinishReason::Stop);

    stage.shutdown.cancel();
    head.expect_replica_down().await;
    stage.finished().await;
}

fn hex(fixture: &str) -> Vec<u8> {
    let fixture = fixture.trim();
    (0..fixture.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&fixture[i..i + 2], 16).expect("hex"))
        .collect()
}

/// Frames vLLM-Omni v0.32.0's own encoder produced (deploy/omni/encode_groundtruth.py).
const T2I_REQUEST: &str =
    include_str!("../crates/sim-protocol/tests/fixtures/omni_v0.32.0_diffusion_t2i.hex");
const PROMPT_ONLY_REQUEST: &str =
    include_str!("../crates/sim-protocol/tests/fixtures/omni_v0.32.0_diffusion_bare.hex");

/// A diffusion stage attached to a head, with the head's data sockets bound.
struct DiffusionRig {
    head: Head,
    stage: Stage,
    requests: PushSocket,
    responses: PullSocket,
}

impl DiffusionRig {
    async fn attach(flags: &[&str]) -> Self {
        let mut head = Head::start().await;
        let mut all_flags = vec!["--omni-stage-kind", "diffusion"];
        all_flags.extend_from_slice(flags);
        let stage = Stage::spawn(&head, "0", &all_flags);

        let registration = head.accept_registration().await;
        assert_eq!(field(&registration, "stage_id"), &Value::from(0));
        assert!(!field(&registration, "stage_config").is_nil());
        head.handshake().await;

        let mut requests = PushSocket::new();
        requests
            .bind(&head.input_address)
            .await
            .expect("bind input");
        let mut responses = PullSocket::new();
        responses
            .bind(&head.output_address)
            .await
            .expect("bind output");
        head.expect_replica_up(0).await;
        Self {
            head,
            stage,
            requests,
            responses,
        }
    }

    async fn send(&mut self, frame: Vec<u8>) {
        self.requests
            .send(ZmqMessage::from(frame))
            .await
            .expect("send request");
    }

    async fn response(&mut self) -> Value {
        let frames = within("diffusion response", self.responses.recv())
            .await
            .expect("response")
            .into_vec();
        assert_eq!(frames.len(), 1, "diffusion responses are single-frame");
        decode(&frames[0])
    }
}

fn result_images(response: &Value) -> Vec<(u64, u64, usize)> {
    assert_eq!(field(response, "type").as_str(), Some("result"));
    let output = field(response, "output");
    assert_eq!(field(output, "finished"), &Value::from(true));
    assert_eq!(field(output, "final_output_type").as_str(), Some("image"));
    field(output, "images")
        .as_array()
        .expect("images")
        .iter()
        .map(|image| {
            assert_eq!(field(image, "mode").as_str(), Some("RGB"));
            let shape = field(image, "shape").as_array().expect("shape");
            let Value::Binary(data) = field(image, "data") else {
                panic!("image data must be msgpack bin");
            };
            (
                shape[1].as_u64().expect("width"),
                shape[0].as_u64().expect("height"),
                data.len(),
            )
        })
        .collect()
}

#[tokio::test]
async fn diffusion_stage_serves_a_head_encoded_generate_request() {
    let mut rig = DiffusionRig::attach(&[]).await;

    rig.send(hex(T2I_REQUEST)).await;
    let response = rig.response().await;

    assert_eq!(
        field(field(&response, "output"), "request_id").as_str(),
        Some("img_gen-0123456789abcdef-01234567")
    );
    assert_eq!(
        result_images(&response),
        vec![(768, 512, 768 * 512 * 3), (768, 512, 768 * 512 * 3)]
    );

    rig.stage.shutdown.cancel();
    rig.head.expect_replica_down().await;
    rig.stage.finished().await;
}

#[tokio::test]
async fn diffusion_stage_uses_its_defaults_for_a_prompt_only_request() {
    let mut rig = DiffusionRig::attach(&["--omni-diffusion-default-size", "96x64"]).await;

    rig.send(hex(PROMPT_ONLY_REQUEST)).await;

    assert_eq!(
        result_images(&rig.response().await),
        vec![(96, 64, 96 * 64 * 3)]
    );
}

fn generate(request_id: &str, width: u32, height: u32, steps: u32) -> Vec<u8> {
    encode(&map(vec![
        ("type", Value::from("add_request")),
        ("request_id", Value::from(request_id)),
        ("prompt", map(vec![("prompt", Value::from("a red fox"))])),
        (
            "sampling_params",
            map(vec![
                ("num_outputs_per_prompt", Value::from(1)),
                ("width", Value::from(width)),
                ("height", Value::from(height)),
                ("num_inference_steps", Value::from(steps)),
            ]),
        ),
    ]))
}

/// With one execution slot, a request queued behind a long one only runs once
/// the long one is aborted, and the aborted request never answers.
#[tokio::test]
async fn diffusion_abort_frees_the_execution_slot_and_sends_nothing() {
    let mut rig = DiffusionRig::attach(&["--omni-diffusion-step-time", "60000"]).await;

    rig.send(generate("img_gen-slow", 1024, 1024, 50)).await;
    rig.send(generate("img_gen-quick", 8, 8, 1)).await;
    rig.send(encode(&map(vec![
        ("type", Value::from("abort")),
        (
            "request_ids",
            Value::Array(vec![Value::from("img_gen-slow")]),
        ),
    ])))
    .await;

    let response = rig.response().await;
    assert_eq!(
        field(field(&response, "output"), "request_id").as_str(),
        Some("img_gen-quick")
    );
}

#[tokio::test]
async fn diffusion_stage_rejects_an_oversized_request_with_a_client_error() {
    let mut rig = DiffusionRig::attach(&[]).await;

    rig.send(generate("img_gen-huge", 65_536, 65_536, 1)).await;

    let response = rig.response().await;
    assert_eq!(field(&response, "type").as_str(), Some("error"));
    assert_eq!(
        field(&response, "request_id").as_str(),
        Some("img_gen-huge")
    );
    assert_eq!(field(&response, "status_code"), &Value::from(400));
}

#[tokio::test]
async fn diffusion_stage_answers_rpc_and_stops_on_the_shutdown_message() {
    let mut rig = DiffusionRig::attach(&[]).await;

    rig.send(encode(&map(vec![
        ("type", Value::from("collective_rpc")),
        ("rpc_id", Value::from("0123456789abcdef")),
        ("method", Value::from("profile")),
        ("timeout", Value::Nil),
        ("args", Value::Array(Vec::new())),
        ("kwargs", Value::Map(Vec::new())),
    ])))
    .await;
    let reply = rig.response().await;
    assert_eq!(field(&reply, "type").as_str(), Some("rpc_result"));
    assert_eq!(field(&reply, "rpc_id").as_str(), Some("0123456789abcdef"));

    rig.send(encode(&map(vec![("type", Value::from("shutdown"))])))
        .await;
    rig.head.expect_replica_down().await;
    rig.stage.finished().await;
}
