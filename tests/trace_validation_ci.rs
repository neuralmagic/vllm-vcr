//! CI trace validation test that validates synthetic trace fixtures.
//!
//! This test validates that synthetic trace files with pseudo TTFT/ITL timing data:
//! 1. Pass structural validation rules (itl_ms/itl_ctx/itl_tokens alignment)
//! 2. Can be serialized/deserialized correctly
//! 3. Can be replayed through the simulator without errors
//!
//! The test uses the same harness pattern as engine_core_e2e.rs: spawn the simulator
//! with tokio::spawn, connect a real EngineCoreClient over ZMQ, and drive requests.

use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use futures::StreamExt;
use sim_trace::trace::read_trace_file;
use tokio_util::sync::CancellationToken;
use vllm_engine_core_client::protocol::{EngineCoreRequest, EngineCoreSamplingParams};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};
use vllm_vcr::{Opt, run};

const TIMEOUT: Duration = Duration::from_secs(10);

/// RAII guard that cancels the simulator task on drop, even if the test panics.
struct SimGuard {
    token: CancellationToken,
}

impl Drop for SimGuard {
    fn drop(&mut self) {
        self.token.cancel();
    }
}

/// Spin up the simulator with replay flags and connect a real client.
/// Returns `(client, guard)`. The guard cancels the sim on drop.
async fn harness_with_replay(
    name: &str,
    replay_trace: &str,
) -> Result<(EngineCoreClient, SimGuard)> {
    let addr = format!("ipc:///tmp/trace-val-{}-{name}.ipc", std::process::id());

    let args = vec![
        "play",
        "--handshake-address",
        &addr,
        "--replay-tokens",
        replay_trace,
        "--replay-steps",
        replay_trace,
        "--replay-match",
        "index",
    ];
    let opt = Opt::parse_from(&args);

    let token = CancellationToken::new();
    let guard = SimGuard {
        token: token.clone(),
    };

    // Spawn the simulator task
    let sim_opt = opt.clone();
    let sim_token = token.clone();
    tokio::spawn(async move {
        let _ = run(sim_opt, sim_token).await;
    });

    // Give simulator time to bind ZMQ socket
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Connect the real client
    let config = EngineCoreClientConfig::new_single(&addr);
    let client = tokio::time::timeout(Duration::from_secs(30), EngineCoreClient::connect(config))
        .await
        .expect("client connect timed out")?;

    Ok((client, guard))
}

/// Build a request for replay matching. The replay engine matches by index when
/// --replay-match=index, so request_id should be "replay-{i}".
fn make_replay_request(index: usize, prompt_tokens: usize, max_tokens: u32) -> EngineCoreRequest {
    EngineCoreRequest {
        request_id: format!("replay-{index}"),
        prompt_token_ids: Some(vec![42u32; prompt_tokens]),
        sampling_params: Some(EngineCoreSamplingParams {
            max_tokens,
            ..EngineCoreSamplingParams::for_test()
        }),
        ..Default::default()
    }
}

#[tokio::test]
async fn replay_simple_trace() -> Result<()> {
    let fixture = "tests/fixtures/synthetic/simple_trace.jsonl";
    let (meta, records) = read_trace_file(Path::new(fixture))?;

    assert_eq!(meta.model, Some("synthetic-model".into()));
    assert_eq!(records.len(), 4);

    let (client, _guard) = harness_with_replay("simple", fixture).await?;

    // Replay each record
    for (i, record) in records.iter().enumerate() {
        let req = make_replay_request(i, record.prompt_tokens, record.output_tokens as u32);

        let stream = client.call(req).await?;
        let outputs: Vec<_> = tokio::time::timeout(TIMEOUT, stream.collect::<Vec<_>>())
            .await
            .expect("stream collect timed out");

        let token_count: usize = outputs
            .iter()
            .map(|r| {
                let o = r.as_ref().expect("stream item error");
                o.new_token_ids.len()
            })
            .sum();

        assert_eq!(
            token_count, record.output_tokens,
            "fixture={fixture}, request={i}"
        );
    }

    Ok(())
}

#[tokio::test]
async fn replay_batch_context_trace() -> Result<()> {
    let fixture = "tests/fixtures/synthetic/batch_context_trace.jsonl";
    let (meta, records) = read_trace_file(Path::new(fixture))?;

    assert_eq!(meta.model, Some("synthetic-model".into()));
    assert_eq!(records.len(), 5);

    // Validate itl_ctx fields are present and properly aligned
    for (i, record) in records.iter().enumerate() {
        if let Some(ref itl_ms) = record.itl_ms {
            let gaps = itl_ms.len();

            if let Some(ref ctx) = record.itl_ctx {
                assert_eq!(
                    ctx.num_running.len(),
                    gaps,
                    "record {i}: num_running len must parallel itl_ms"
                );
                assert_eq!(
                    ctx.prefill_tokens.len(),
                    gaps,
                    "record {i}: prefill_tokens len must parallel itl_ms"
                );
            }
        }
    }

    let (client, _guard) = harness_with_replay("batch", fixture).await?;

    // Replay each record
    for (i, record) in records.iter().enumerate() {
        let req = make_replay_request(i, record.prompt_tokens, record.output_tokens as u32);

        let stream = client.call(req).await?;
        let outputs: Vec<_> = tokio::time::timeout(TIMEOUT, stream.collect::<Vec<_>>())
            .await
            .expect("stream collect timed out");

        let token_count: usize = outputs
            .iter()
            .map(|r| {
                let o = r.as_ref().expect("stream item error");
                o.new_token_ids.len()
            })
            .sum();

        assert_eq!(
            token_count, record.output_tokens,
            "fixture={fixture}, request={i}"
        );
    }

    Ok(())
}

#[tokio::test]
async fn replay_speculative_trace() -> Result<()> {
    let fixture = "tests/fixtures/synthetic/speculative_trace.jsonl";
    let (meta, records) = read_trace_file(Path::new(fixture))?;

    assert_eq!(meta.model, Some("synthetic-spec-model".into()));
    assert_eq!(records.len(), 4);

    // Validate itl_tokens fields are properly aligned
    for (i, record) in records.iter().enumerate() {
        if let Some(ref itl_tokens) = record.itl_tokens {
            let gaps = record.itl_ms.as_ref().expect("itl_ms required").len();

            assert_eq!(
                itl_tokens.len(),
                gaps,
                "record {i}: itl_tokens len must parallel itl_ms"
            );

            // Validate no zero tokens
            assert!(
                !itl_tokens.contains(&0),
                "record {i}: itl_tokens must not contain zero"
            );

            // Validate sum < output_tokens
            let chunk_total: u64 = itl_tokens.iter().map(|&t| u64::from(t)).sum();
            assert!(
                chunk_total < record.output_tokens as u64,
                "record {i}: itl_tokens sum ({chunk_total}) must be < output_tokens ({})",
                record.output_tokens
            );
        }
    }

    let (client, _guard) = harness_with_replay("spec", fixture).await?;

    // Replay each record
    for (i, record) in records.iter().enumerate() {
        let req = make_replay_request(i, record.prompt_tokens, record.output_tokens as u32);

        let stream = client.call(req).await?;
        let outputs: Vec<_> = tokio::time::timeout(TIMEOUT, stream.collect::<Vec<_>>())
            .await
            .expect("stream collect timed out");

        let token_count: usize = outputs
            .iter()
            .map(|r| {
                let o = r.as_ref().expect("stream item error");
                o.new_token_ids.len()
            })
            .sum();

        assert_eq!(
            token_count, record.output_tokens,
            "fixture={fixture}, request={i}"
        );
    }

    Ok(())
}

#[tokio::test]
async fn replay_edge_cases_trace() -> Result<()> {
    let fixture = "tests/fixtures/synthetic/edge_cases_trace.jsonl";
    let (meta, records) = read_trace_file(Path::new(fixture))?;

    assert_eq!(meta.model, Some("synthetic-edge-model".into()));
    assert_eq!(records.len(), 8);

    // Validate edge cases:
    // - Single-token output (no ITL required)
    assert_eq!(records[0].output_tokens, 1);
    assert!(records[0].itl_ms.is_none());
    assert!(records[0].itl_summary.is_none());

    // - Large outputs with itl_summary
    assert_eq!(records[1].output_tokens, 100);
    assert!(records[1].itl_summary.is_some());
    assert_eq!(records[1].itl_summary.as_ref().unwrap().count, 99);

    assert_eq!(records[2].output_tokens, 256);
    assert!(records[2].itl_summary.is_some());

    // - Different finish_reason values
    assert_eq!(
        records[3].finish_reason.as_ref().map(|f| format!("{f:?}")),
        Some("Abort".into())
    );
    assert_eq!(
        records[4].finish_reason.as_ref().map(|f| format!("{f:?}")),
        Some("Error".into())
    );
    assert_eq!(
        records[5].finish_reason.as_ref().map(|f| format!("{f:?}")),
        Some("Repetition".into())
    );

    let (client, _guard) = harness_with_replay("edge", fixture).await?;

    // Replay each record
    for (i, record) in records.iter().enumerate() {
        let req = make_replay_request(i, record.prompt_tokens, record.output_tokens as u32);

        let stream = client.call(req).await?;
        let outputs: Vec<_> = tokio::time::timeout(TIMEOUT, stream.collect::<Vec<_>>())
            .await
            .expect("stream collect timed out");

        let token_count: usize = outputs
            .iter()
            .map(|r| {
                let o = r.as_ref().expect("stream item error");
                o.new_token_ids.len()
            })
            .sum();

        assert_eq!(
            token_count, record.output_tokens,
            "fixture={fixture}, request={i}"
        );
    }

    Ok(())
}

#[tokio::test]
async fn validate_all_traces_parseable() -> Result<()> {
    // Ensure all synthetic traces can be parsed without errors
    let fixtures = [
        "tests/fixtures/synthetic/simple_trace.jsonl",
        "tests/fixtures/synthetic/batch_context_trace.jsonl",
        "tests/fixtures/synthetic/speculative_trace.jsonl",
        "tests/fixtures/synthetic/edge_cases_trace.jsonl",
    ];

    for fixture in fixtures {
        let (meta, records) = read_trace_file(Path::new(fixture))?;

        assert!(
            meta.model.is_some(),
            "fixture={fixture}: meta.model missing"
        );
        assert!(!records.is_empty(), "fixture={fixture}: no records");

        // Validate each record passes validation
        for (i, record) in records.iter().enumerate() {
            // Multi-token output requires ITL data
            if record.output_tokens > 1 {
                assert!(
                    record.itl_ms.is_some() || record.itl_summary.is_some(),
                    "fixture={fixture}, record={i}: multi-token output requires itl_ms or itl_summary"
                );
            }

            // If itl_ctx present, validate parallel arrays
            if let Some(ref ctx) = record.itl_ctx {
                let gaps = record.itl_ms.as_ref().expect("itl_ms required").len();
                assert_eq!(
                    ctx.num_running.len(),
                    gaps,
                    "fixture={fixture}, record={i}: num_running must parallel itl_ms"
                );
                assert_eq!(
                    ctx.prefill_tokens.len(),
                    gaps,
                    "fixture={fixture}, record={i}: prefill_tokens must parallel itl_ms"
                );
            }

            // If itl_tokens present, validate parallel and constraints
            if let Some(ref tokens) = record.itl_tokens {
                let gaps = record.itl_ms.as_ref().expect("itl_ms required").len();
                assert_eq!(
                    tokens.len(),
                    gaps,
                    "fixture={fixture}, record={i}: itl_tokens must parallel itl_ms"
                );
                assert!(
                    !tokens.contains(&0),
                    "fixture={fixture}, record={i}: itl_tokens must not contain zero"
                );

                let chunk_total: u64 = tokens.iter().map(|&t| u64::from(t)).sum();
                assert!(
                    chunk_total < record.output_tokens as u64,
                    "fixture={fixture}, record={i}: itl_tokens sum must be < output_tokens"
                );
            }

            // If output_token_ids present, validate length
            if let Some(ref ids) = record.output_token_ids {
                assert_eq!(
                    ids.len(),
                    record.output_tokens,
                    "fixture={fixture}, record={i}: output_token_ids len must equal output_tokens"
                );
            }
        }
    }

    Ok(())
}
