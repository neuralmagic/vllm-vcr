//! End-to-end tests for vLLM-Omni stages over real ZMQ transport.
//!
//! Each test runs the full `vllm_vcr::run` task against the scripted omni head
//! in `omni_head`.
//!
//! The contract under test:
//!   - A stage registers with the master, handshakes on the address it is
//!     handed, connects its data sockets after READY, and then reports up to
//!     the coordinator with the addresses from the handshake.
//!   - An LLM stage accepts a request carrying the omni tail and, as a final
//!     audio stage, returns `multimodal_output` on its outputs.
//!   - A diffusion stage serves generate requests, honors abort, and stops on
//!     the head's shutdown message.
//!   - A diffusion stage serves the control API: a config patch applies to
//!     requests that have not started, and stats report the queue.
//!   - Either stage reports down when it stops.

use std::time::Duration;

use rmpv::Value;
use sim_protocol::vllm::{
    EngineCoreFinishReason, EngineCoreOutput, EngineCoreRequest, EngineCoreSamplingParams,
    Envelope, decode_msgpack, encode_msgpack, request_outputs,
};
use zeromq::prelude::{Socket as _, SocketRecv as _, SocketSend as _};
use zeromq::{PullSocket, PushSocket, RouterSocket, ZmqMessage};

use crate::omni_head::{
    Head, Stage, audio_samples, decode, encode, field, map, request_with_tail, unbound_endpoint,
    within,
};

mod omni_head;

#[tokio::test]
async fn llm_stage_attaches_and_returns_audio_for_a_request_with_an_omni_tail() {
    let mut head = Head::start().await;
    let stage = Stage::spawn(
        &head.master_address,
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
        let stage = Stage::spawn(&head.master_address, "0", &all_flags);

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

/// One HTTP/1.1 exchange with the control API, retried until the server is
/// listening.
async fn control(address: &str, method: &str, path: &str, body: &str) -> (u16, serde_json::Value) {
    use std::io::{Read as _, Write as _};

    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: control\r\nConnection: close\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let address = address.to_string();
    let exchange = tokio::task::spawn_blocking(move || {
        let mut stream = loop {
            match std::net::TcpStream::connect(&address) {
                Ok(stream) => break stream,
                Err(_) => std::thread::sleep(Duration::from_millis(20)),
            }
        };
        stream.write_all(request.as_bytes()).expect("write request");
        let mut response = String::new();
        stream.read_to_string(&mut response).expect("read response");
        response
    });
    let response = within("control API exchange", exchange)
        .await
        .expect("control exchange join");
    let status = response
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .unwrap_or_else(|| panic!("no status line in {response:?}"));
    let body = response.split_once("\r\n\r\n").map_or("", |(_, body)| body);
    (
        status,
        serde_json::from_str(body).unwrap_or(serde_json::Value::Null),
    )
}

async fn stats_when(address: &str, received: u64) -> serde_json::Value {
    loop {
        let (status, stats) = control(address, "GET", "/stats", "").await;
        assert_eq!(status, 200);
        if stats["requests_received"] == received {
            return stats;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[tokio::test]
async fn diffusion_control_api_patches_the_queue_and_reports_stats() {
    let control_address = unbound_endpoint().replace("tcp://", "");
    let mut rig = DiffusionRig::attach(&[
        "--omni-diffusion-step-time",
        "60000",
        "--control-address",
        &control_address,
    ])
    .await;

    let (status, config) = control(&control_address, "GET", "/config", "").await;
    assert_eq!(status, 200);
    assert_eq!(config["step_time"], 60_000);
    assert_eq!(config["concurrency"], 1);

    rig.send(generate("img_gen-slow", 1024, 1024, 50)).await;
    rig.send(generate("img_gen-queued", 8, 8, 1)).await;
    let stats = within("both requests received", stats_when(&control_address, 2)).await;
    assert_eq!(stats["running"], 1);
    assert_eq!(stats["waiting"], 1);

    let (status, config) = control(
        &control_address,
        "PATCH",
        "/config",
        r#"{"concurrency": 2, "step_time": 0}"#,
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(config["concurrency"], 2);
    assert_eq!(config["step_time"], 0);

    let response = rig.response().await;
    assert_eq!(
        field(field(&response, "output"), "request_id").as_str(),
        Some("img_gen-queued"),
        "the queued request starts under the patched config; the running one keeps its pace"
    );
    let (_, stats) = control(&control_address, "GET", "/stats", "").await;
    assert_eq!(stats["requests_completed"], 1);
    assert_eq!(stats["running"], 1);
    assert_eq!(stats["waiting"], 0);

    let (status, error) = control(
        &control_address,
        "PATCH",
        "/config",
        r#"{"concurrency": 0}"#,
    )
    .await;
    assert_eq!(status, 400);
    assert!(
        error["error"]
            .as_str()
            .is_some_and(|e| e.contains("concurrency"))
    );
}
