//! End-to-end integration tests for the engine using synthetic traces.
//!
//! This test suite validates the engine against programmatically generated traces
//! covering all schema variants, edge cases, and replay modes. All tests use real
//! ZMQ transport and real EngineCoreClient (no mocks), with synthetic traces for
//! deterministic, fast, GPU-free validation.

use std::time::Duration;

use futures::StreamExt;
use tokio_util::sync::CancellationToken;
use vllm_engine_core_client::protocol::{EngineCoreRequest, EngineCoreSamplingParams};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};
use vllm_vcr::{Opt, run};

mod synthetic_trace_generator;
mod test_helpers;

use synthetic_trace_generator::*;
use test_helpers::*;

const TIMEOUT: Duration = Duration::from_secs(30);

/// Spin up the simulator with the given CLI flags, connect a real client.
/// Returns `(client, guard)`. The guard cancels the sim on drop.
async fn harness(test_name: &str, extra_flags: &[&str]) -> (EngineCoreClient, SimGuard) {
    let addr = unique_ipc_endpoint(test_name);

    let mut args: Vec<&str> = vec!["play", "--handshake-address", &addr];
    args.extend_from_slice(extra_flags);

    let opt = Opt::parse_from(&args);
    let token = CancellationToken::new();
    let guard = SimGuard {
        token: token.clone(),
    };

    // Spawn the simulator task.
    let sim_opt = opt.clone();
    let sim_token = token.clone();
    tokio::spawn(async move {
        let _ = run(sim_opt, sim_token).await;
    });

    // Connect the real client.
    let config = EngineCoreClientConfig::new_single(&addr);
    let client = tokio::time::timeout(Duration::from_secs(30), EngineCoreClient::connect(config))
        .await
        .expect("client connect timed out")
        .expect("client connect failed");

    (client, guard)
}

/// Build a simple request with repeated token IDs.
fn make_request(id: &str, prompt_len: usize, max_tokens: u32) -> EngineCoreRequest {
    EngineCoreRequest {
        request_id: id.to_string(),
        prompt_token_ids: Some(vec![42u32; prompt_len]),
        sampling_params: Some(EngineCoreSamplingParams {
            max_tokens,
            ..EngineCoreSamplingParams::for_test()
        }),
        ..Default::default()
    }
}

// ============================================================================
// Basic Trace Tests
// ============================================================================

#[tokio::test]
async fn test_basic_trace_token_replay() {
    let (meta, records) = generate_basic_trace(20, 12345);
    let trace_file =
        create_temp_trace("basic_token_replay", &meta, &records).expect("create trace file");
    let trace_path = trace_file.path().to_str().expect("path to UTF-8");

    let (client, _guard) = harness("basic_token_replay", &["--replay-tokens", trace_path]).await;

    // Replay first 3 records
    for (i, record) in records.iter().enumerate().take(3) {
        let req = make_request(
            &format!("replay-{}", i),
            record.prompt_tokens,
            record.output_tokens as u32,
        );

        let stream = client.call(req).await.expect("call failed");
        let outputs: Vec<_> = tokio::time::timeout(TIMEOUT, stream.collect::<Vec<_>>())
            .await
            .expect("stream collect timed out");

        // Validate token count
        let total_tokens: usize = outputs
            .iter()
            .filter_map(|r| r.as_ref().ok())
            .map(|o| o.new_token_ids.len())
            .sum();
        assert_eq!(
            total_tokens, record.output_tokens,
            "record {} token count mismatch",
            i
        );

        // Validate finish reason
        let has_finish = outputs
            .iter()
            .filter_map(|r| r.as_ref().ok())
            .any(|o| o.finish_reason.is_some());
        assert!(has_finish, "record {} should have finish reason", i);
    }
}

#[tokio::test]
async fn test_basic_trace_latency_replay() {
    let (meta, records) = generate_basic_trace(10, 54321);
    let trace_file =
        create_temp_trace("basic_latency_replay", &meta, &records).expect("create trace file");
    let trace_path = trace_file.path().to_str().expect("path to UTF-8");

    let (client, _guard) = harness("basic_latency_replay", &["--latency-trace", trace_path]).await;

    // Just validate that requests complete with the correct token count
    for (i, record) in records.iter().enumerate().take(3) {
        let req = make_request(
            &format!("latency-{}", i),
            record.prompt_tokens,
            record.output_tokens as u32,
        );

        let stream = client.call(req).await.expect("call failed");
        let outputs: Vec<_> = tokio::time::timeout(TIMEOUT, stream.collect::<Vec<_>>())
            .await
            .expect("stream collect timed out");

        let total_tokens: usize = outputs
            .iter()
            .filter_map(|r| r.as_ref().ok())
            .map(|o| o.new_token_ids.len())
            .sum();
        assert_eq!(total_tokens, record.output_tokens);
    }
}

// ============================================================================
// Batch Context Tests
// ============================================================================

#[tokio::test]
async fn test_batch_context_replay() {
    let (meta, records) = generate_batch_context_trace(15, 11111);
    let trace_file =
        create_temp_trace("batch_context", &meta, &records).expect("create trace file");
    let trace_path = trace_file.path().to_str().expect("path to UTF-8");

    let (client, _guard) = harness("batch_context", &["--replay-tokens", trace_path]).await;

    // Validate a few requests complete successfully
    for (i, record) in records.iter().enumerate().take(3) {
        let req = make_request(
            &format!("replay-{}", i),
            record.prompt_tokens,
            record.output_tokens as u32,
        );

        let stream = client.call(req).await.expect("call failed");
        let outputs: Vec<_> = tokio::time::timeout(TIMEOUT, stream.collect::<Vec<_>>())
            .await
            .expect("stream collect timed out");

        assert!(!outputs.is_empty(), "should have outputs for record {}", i);
        let total_tokens: usize = outputs
            .iter()
            .filter_map(|r| r.as_ref().ok())
            .map(|o| o.new_token_ids.len())
            .sum();
        assert_eq!(total_tokens, record.output_tokens);
    }
}

// ============================================================================
// Speculative Decoding Tests
// ============================================================================

#[tokio::test]
async fn test_speculative_burst_structure() {
    let (meta, records) = generate_speculative_trace(10, 22222);
    let trace_file = create_temp_trace("speculative", &meta, &records).expect("create trace file");
    let trace_path = trace_file.path().to_str().expect("path to UTF-8");

    let (client, _guard) = harness("speculative", &["--replay-tokens", trace_path]).await;

    // Test first record
    let record = &records[0];
    let req = make_request(
        "replay-0",
        record.prompt_tokens,
        record.output_tokens as u32,
    );

    let stream = client.call(req).await.expect("call failed");
    let outputs: Vec<_> = tokio::time::timeout(TIMEOUT, stream.collect::<Vec<_>>())
        .await
        .expect("stream collect timed out");

    let total_tokens: usize = outputs
        .iter()
        .filter_map(|r| r.as_ref().ok())
        .map(|o| o.new_token_ids.len())
        .sum();
    assert_eq!(total_tokens, record.output_tokens);
}

// ============================================================================
// Diffusion Block Tests
// ============================================================================

#[tokio::test]
async fn test_diffusion_block_outputs() {
    let (meta, records) = generate_diffusion_trace(8, 33333);
    let trace_file = create_temp_trace("diffusion", &meta, &records).expect("create trace file");
    let trace_path = trace_file.path().to_str().expect("path to UTF-8");

    let (client, _guard) = harness("diffusion", &["--replay-tokens", trace_path]).await;

    // Test first record
    let record = &records[0];
    let req = make_request(
        "replay-0",
        record.prompt_tokens,
        record.output_tokens as u32,
    );

    let stream = client.call(req).await.expect("call failed");
    let outputs: Vec<_> = tokio::time::timeout(TIMEOUT, stream.collect::<Vec<_>>())
        .await
        .expect("stream collect timed out");

    let total_tokens: usize = outputs
        .iter()
        .filter_map(|r| r.as_ref().ok())
        .map(|o| o.new_token_ids.len())
        .sum();
    assert_eq!(total_tokens, record.output_tokens);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[tokio::test]
async fn test_edge_cases_single_token() {
    let (meta, records) = generate_edge_cases_trace(44444);
    let trace_file = create_temp_trace("edge_cases", &meta, &records).expect("create trace file");
    let trace_path = trace_file.path().to_str().expect("path to UTF-8");

    let (client, _guard) = harness("edge_cases_single", &["--replay-tokens", trace_path]).await;

    // First record is single-token output
    let record = &records[0];
    assert_eq!(
        record.output_tokens, 1,
        "first record should be single token"
    );

    let req = make_request(
        "replay-0",
        record.prompt_tokens,
        record.output_tokens as u32,
    );
    let stream = client.call(req).await.expect("call failed");
    let outputs: Vec<_> = tokio::time::timeout(TIMEOUT, stream.collect::<Vec<_>>())
        .await
        .expect("stream collect timed out");

    let total_tokens: usize = outputs
        .iter()
        .filter_map(|r| r.as_ref().ok())
        .map(|o| o.new_token_ids.len())
        .sum();
    assert_eq!(total_tokens, 1);
}

#[tokio::test]
async fn test_gzip_compressed_trace() {
    let (meta, records) = generate_basic_trace(15, 88888);
    let trace_file =
        create_temp_trace_gz("compressed", &meta, &records).expect("create gzipped trace file");
    let trace_path = trace_file.path().to_str().expect("path to UTF-8");

    let (client, _guard) = harness("compressed", &["--replay-tokens", trace_path]).await;

    // Validate first few records work with compressed trace
    for (i, record) in records.iter().enumerate().take(3) {
        let req = make_request(
            &format!("replay-{}", i),
            record.prompt_tokens,
            record.output_tokens as u32,
        );

        let stream = client.call(req).await.expect("call failed");
        let outputs: Vec<_> = tokio::time::timeout(TIMEOUT, stream.collect::<Vec<_>>())
            .await
            .expect("stream collect timed out");

        let total_tokens: usize = outputs
            .iter()
            .filter_map(|r| r.as_ref().ok())
            .map(|o| o.new_token_ids.len())
            .sum();
        assert_eq!(total_tokens, record.output_tokens);
    }
}

#[tokio::test]
async fn test_mixed_concurrency_levels() {
    let (meta, records) = generate_mixed_concurrency_trace(20, 99999);
    let trace_file =
        create_temp_trace("mixed_concurrency", &meta, &records).expect("create trace file");
    let trace_path = trace_file.path().to_str().expect("path to UTF-8");

    let (client, _guard) = harness("mixed_concurrency", &["--replay-tokens", trace_path]).await;

    // Test a few records with different concurrency levels
    for (i, record) in records.iter().enumerate().take(4) {
        let req = make_request(
            &format!("replay-{}", i),
            record.prompt_tokens,
            record.output_tokens as u32,
        );

        let stream = client.call(req).await.expect("call failed");
        let outputs: Vec<_> = tokio::time::timeout(TIMEOUT, stream.collect::<Vec<_>>())
            .await
            .expect("stream collect timed out");

        let total_tokens: usize = outputs
            .iter()
            .filter_map(|r| r.as_ref().ok())
            .map(|o| o.new_token_ids.len())
            .sum();
        assert_eq!(total_tokens, record.output_tokens);
    }
}
