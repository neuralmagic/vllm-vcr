//! End-to-end test for recording a vLLM-Omni LLM stage.
//!
//! The chain is the scripted omni head in `omni_head`, the real tap
//! (`sim_tap::tap::run_tap`), and the real `vllm_vcr::run` task as the stage,
//! all over real ZMQ sockets:
//!
//! ```text
//!   scripted head <--> tap <--> play (omni LLM stage)
//! ```
//!
//! The contract under test:
//!   - The stage registers with the tap, and the head receives that
//!     registration unchanged.
//!   - The tap reports READY to the head before its data sockets can connect.
//!   - The stage's status events reach the head carrying the head-side data
//!     addresses, not the tap-side ones the stage knows.
//!   - A request with the omni tail and its outputs with the audio tail pass
//!     through untouched, and the request lands in the trace.

use std::io::BufReader;

use rmpv::Value;
use sim_protocol::vllm::{
    EngineCoreFinishReason, EngineCoreOutput, EngineCoreRequest, EngineCoreSamplingParams,
    Envelope, decode_msgpack, encode_msgpack, request_outputs,
};
use sim_tap::omni::OmniTap;
use sim_tap::tap::{TapConfig, TapMetaConfig, TokenRecording, run_tap};
use tokio_util::sync::CancellationToken;
use vllm_vcr::trace::{TraceFinishReason, read_trace};
use zeromq::prelude::{Socket as _, SocketRecv as _, SocketSend as _};
use zeromq::{PullSocket, RouterSocket, ZmqMessage};

use crate::omni_head::{
    Head, Stage, audio_samples, decode, field, map, request_with_tail, unbound_endpoint, within,
};

mod omni_head;

#[tokio::test]
async fn tap_records_an_omni_stage_between_the_head_and_the_stage() {
    let mut head = Head::start().await;

    let tap_master = unbound_endpoint();
    let tap_input = unbound_endpoint();
    let tap_output = unbound_endpoint();
    let config = TapConfig {
        frontend_handshake: String::new(),
        engine_handshake: unbound_endpoint(),
        input_address: tap_input.clone(),
        output_address: tap_output.clone(),
        block_size: 16,
        record_tokens: TokenRecording::On,
        omni: Some(OmniTap {
            head_master: head.master_address.clone(),
            master_listen: tap_master.clone(),
            coordinator_listen: unbound_endpoint(),
        }),
    };
    let meta = TapMetaConfig {
        model: "test-model".to_string(),
        gpu: None,
        tp: None,
        max_num_seqs: None,
        block_size: 16,
        config_hash: None,
        vllm_tag: None,
        engine_config: String::new(),
    };
    let tap_shutdown = CancellationToken::new();
    let tap = tokio::spawn({
        let shutdown = tap_shutdown.clone();
        async move {
            let mut trace = Vec::new();
            let result = run_tap(config, meta, &mut trace, None::<Vec<u8>>, shutdown).await;
            (result, trace)
        }
    });

    let stage = Stage::spawn(
        &tap_master,
        "1",
        &[
            "--output-tokens",
            "3",
            "--omni-audio-samples-per-token",
            "10",
        ],
    );

    let registration = head.accept_registration().await;
    assert_eq!(field(&registration, "stage_id"), &Value::from(1));
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
    assert!(
        field(&decode(&ready_response[1]), "max_model_len").is_u64(),
        "the stage's ready response is relayed to the head"
    );

    assert_ne!(tap_input, head.input_address);
    head.expect_replica_up(1).await;

    let request = EngineCoreRequest {
        request_id: "chatcmpl-tapped-1".to_string(),
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

        let mut finished = None;
        for (output, wire) in request_outputs(&envelope).iter().zip(wire_outputs) {
            let wire = wire.as_array().expect("output array");
            tokens += output.new_token_ids.len();
            samples += audio_samples(wire, vllm_fields).unwrap_or(0);
            finished = finished.or(output.finish_reason);
        }
        if let Some(reason) = finished {
            break reason;
        }
    };
    assert_eq!(tokens, 3);
    assert_eq!(samples, 30, "the audio tail passes through the tap");
    assert_eq!(finish_reason, EngineCoreFinishReason::Stop);

    stage.shutdown.cancel();
    head.expect_replica_down().await;
    stage.finished().await;

    tap_shutdown.cancel();
    let (result, trace) = within("tap exit", tap).await.expect("tap join");
    result.expect("tap result");

    let (meta, records) = read_trace(BufReader::new(trace.as_slice())).expect("read trace");
    assert_eq!(meta.source.as_deref(), Some("tap"));
    assert_eq!(meta.model.as_deref(), Some("test-model"));
    assert_eq!(
        meta.extra.get("omni_stage_id"),
        Some(&serde_json::json!(1)),
        "the trace names the stage it was captured from"
    );

    assert_eq!(records.len(), 1);
    let record = &records[0];
    assert_eq!(record.prompt_tokens, 4);
    assert_eq!(record.output_tokens, 3);
    assert_eq!(record.finish_reason, Some(TraceFinishReason::Stop));
    assert_eq!(
        record.output_token_ids.as_ref().map(Vec::len),
        Some(3),
        "token ids are recorded with --record-tokens"
    );
}
