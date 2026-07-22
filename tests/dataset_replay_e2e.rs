//! End-to-end tests for dataset-driven token replay (`--replay-tokens` with a
//! HuggingFace dataset file): real engine, real ZMQ transport, real tokenizer.
//! Ignored by default because they download the Qwen tokenizer from the HF Hub;
//! run with `cargo test --test dataset_replay_e2e -- --ignored`.

use std::time::Duration;

use futures::StreamExt;
use tokio_util::sync::CancellationToken;
use vllm_engine_core_client::protocol::{EngineCoreRequest, EngineCoreSamplingParams};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};
use vllm_vcr::{Opt, run};

const TIMEOUT: Duration = Duration::from_secs(60);
const MODEL: &str = "Qwen/Qwen3-0.6B";
const DATASET: &str = "tests/fixtures/datasets/hf_dataset_sample.jsonl";

fn load_tokenizer() -> tokenizers::Tokenizer {
    let api = hf_hub::api::sync::Api::new().expect("hf api");
    let file = api
        .model(MODEL.to_string())
        .get("tokenizer.json")
        .expect("download tokenizer");
    tokenizers::Tokenizer::from_file(file).expect("load tokenizer")
}

fn dataset_row(idx: usize) -> (String, String) {
    let rows = sim_trace::dataset_convert::parse_dataset(std::path::Path::new(DATASET))
        .expect("parse dataset");
    let prompt = sim_trace::dataset_convert::extract_prompt(&rows[idx]).expect("prompt");
    let response = sim_trace::dataset_convert::extract_response(&rows[idx]).expect("response");
    (prompt, response)
}

fn encode(tokenizer: &tokenizers::Tokenizer, text: &str) -> Vec<u32> {
    tokenizer
        .encode(text, false)
        .expect("encode")
        .get_ids()
        .to_vec()
}

async fn harness(test_name: &str, extra_flags: &[&str]) -> (EngineCoreClient, CancellationToken) {
    let addr = format!(
        "ipc:///tmp/dataset-e2e-{}-{test_name}.ipc",
        std::process::id()
    );
    let mut args = vec![
        "play",
        "--handshake-address",
        &addr,
        "--replay-tokens",
        DATASET,
        "--model-name",
        MODEL,
    ];
    args.extend_from_slice(extra_flags);
    let opt = Opt::parse_from(&args);
    let token = CancellationToken::new();
    let sim_token = token.clone();
    tokio::spawn(async move {
        if let Err(e) = run(opt, sim_token).await {
            eprintln!("sim exited with error: {e:#}");
        }
    });
    let config = EngineCoreClientConfig::new_single(&addr);
    let client = tokio::time::timeout(TIMEOUT, EngineCoreClient::connect(config))
        .await
        .expect("connect timed out")
        .expect("connect failed");
    (client, token)
}

fn request(id: &str, prompt_token_ids: Vec<u32>, max_tokens: u32) -> EngineCoreRequest {
    EngineCoreRequest {
        request_id: id.to_string(),
        prompt_token_ids: Some(prompt_token_ids),
        sampling_params: Some(EngineCoreSamplingParams {
            max_tokens,
            ..EngineCoreSamplingParams::for_test()
        }),
        ..Default::default()
    }
}

async fn collect(client: &EngineCoreClient, req: EngineCoreRequest) -> Vec<u32> {
    let stream = client.call(req).await.expect("call failed");
    let outputs: Vec<_> = tokio::time::timeout(TIMEOUT, stream.collect::<Vec<_>>())
        .await
        .expect("stream timed out");
    outputs
        .into_iter()
        .flat_map(|r| r.expect("stream item").new_token_ids.clone())
        .collect()
}

#[tokio::test]
#[ignore = "network: downloads the Qwen tokenizer from the HF Hub"]
async fn dataset_index_match_serves_tokenized_response() {
    let tokenizer = load_tokenizer();
    let (_, response) = dataset_row(0);
    let expected = encode(&tokenizer, &response);
    assert!(!expected.is_empty());

    let (client, guard) = harness("index", &["--replay-match", "index"]).await;
    let got = collect(
        &client,
        request("replay-0", vec![42u32; 16], expected.len() as u32),
    )
    .await;
    assert_eq!(got, expected);
    guard.cancel();
}

#[tokio::test]
#[ignore = "network: downloads the Qwen tokenizer from the HF Hub"]
async fn dataset_prefix_match_serves_tokenized_response() {
    let tokenizer = load_tokenizer();
    let (prompt, response) = dataset_row(2);
    let prompt_ids = encode(&tokenizer, &prompt);
    let expected = encode(&tokenizer, &response);
    assert!(prompt_ids.len() >= 4, "need at least one block of prompt");

    let (client, guard) = harness(
        "prefix",
        &["--replay-match", "prefix", "--tokens-per-block", "4"],
    )
    .await;
    let got = collect(
        &client,
        request("live-abc", prompt_ids, expected.len() as u32),
    )
    .await;
    assert_eq!(got, expected);
    guard.cancel();
}

#[tokio::test]
#[ignore = "network: downloads the Qwen tokenizer from the HF Hub"]
async fn dataset_overrequest_is_clamped_to_row_length() {
    let tokenizer = load_tokenizer();
    let (_, response) = dataset_row(1);
    let expected = encode(&tokenizer, &response);

    let (client, guard) = harness("overask", &["--replay-match", "index"]).await;
    let got = collect(
        &client,
        request("replay-1", vec![42u32; 16], (expected.len() + 8) as u32),
    )
    .await;
    assert_eq!(got, expected, "stream must end at the dataset row");
    guard.cancel();
}
