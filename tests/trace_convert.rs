//! Integration tests for guidellm-to-trace conversion and summarization.
//!
//! The fixture at tests/fixtures/guidellm_sample.json mirrors the v0.6.0 schema from:
//!   - src/guidellm/benchmark/schemas/generative/report.py   (GenerativeBenchmarksReport)
//!   - src/guidellm/benchmark/schemas/generative/benchmark.py (GenerativeBenchmark)
//!   - src/guidellm/schemas/request_stats.py                  (GenerativeRequestStats)
//!   - src/guidellm/scheduler/strategies.py                   (ConcurrentStrategy, SynchronousStrategy)

use std::io::BufReader;

use inference_simulator_rs::trace;
use inference_simulator_rs::trace_convert::{
    ConvertOptions, convert_guidellm, summarize_trace, write_conversion, write_summary,
};

fn load_fixture() -> String {
    std::fs::read_to_string("tests/fixtures/guidellm_sample.json")
        .expect("fixture file should exist")
}

#[test]
fn fixture_converts_to_correct_record_count() {
    let json = load_fixture();
    let (meta, records) = convert_guidellm(&json, &ConvertOptions::default()).unwrap();

    // 3 successful in benchmark[0] (concurrent, streams=4) + 1 in benchmark[1] (synchronous)
    assert_eq!(records.len(), 4);
    assert_eq!(meta.source.as_deref(), Some("guidellm"));
    assert_eq!(
        meta.model.as_deref(),
        Some("meta-llama/Llama-3.1-8B-Instruct")
    );
}

#[test]
fn fixture_ttft_values_match() {
    let json = load_fixture();
    let (_, records) = convert_guidellm(&json, &ConvertOptions::default()).unwrap();

    // req-001: time_to_first_token_ms = 45.2
    assert!((records[0].ttft_ms - 45.2).abs() < 0.01);
    // req-002: time_to_first_token_ms = 80.0
    assert!((records[1].ttft_ms - 80.0).abs() < 0.01);
    // req-003: time_to_first_token_ms = 30.0
    assert!((records[2].ttft_ms - 30.0).abs() < 0.01);
    // req-004 (synchronous benchmark): time_to_first_token_ms = 25.0
    assert!((records[3].ttft_ms - 25.0).abs() < 0.01);
}

#[test]
fn fixture_concurrency_from_strategy() {
    let json = load_fixture();
    let (_, records) = convert_guidellm(&json, &ConvertOptions::default()).unwrap();

    // First 3 records from concurrent strategy with streams=4.
    for r in &records[..3] {
        assert_eq!(
            r.concurrency, 4,
            "concurrent benchmark should have concurrency 4"
        );
    }
    // Last record from synchronous strategy.
    assert_eq!(
        records[3].concurrency, 1,
        "synchronous benchmark should have concurrency 1"
    );
}

#[test]
fn fixture_cached_tokens_always_zero() {
    let json = load_fixture();
    let (_, records) = convert_guidellm(&json, &ConvertOptions::default()).unwrap();

    for r in &records {
        assert_eq!(r.cached_tokens, 0, "guidellm cannot know cached tokens");
    }
}

#[test]
fn fixture_itl_summary_present_for_multi_token() {
    let json = load_fixture();
    let (_, records) = convert_guidellm(&json, &ConvertOptions::default()).unwrap();

    // req-001: 100 output tokens, inter_token_latency_ms = 14.7
    let itl = records[0].itl_summary.as_ref().unwrap();
    assert!((itl.mean_ms - 14.7).abs() < 0.01);
    assert_eq!(itl.count, 99); // output_tokens - 1

    // All records have output_tokens > 1, so all should have itl_summary.
    for r in &records {
        assert!(r.itl_summary.is_some());
    }
}

#[test]
fn fixture_meta_overrides() {
    let json = load_fixture();
    let opts = ConvertOptions {
        model: Some("override-model".into()),
        gpu: Some("A100-SXM".into()),
        tp: Some(2),
    };
    let (meta, _) = convert_guidellm(&json, &opts).unwrap();
    assert_eq!(meta.model.as_deref(), Some("override-model"));
    assert_eq!(meta.gpu.as_deref(), Some("A100-SXM"));
    assert_eq!(meta.tp, Some(2));
}

#[test]
fn fixture_round_trip_through_trace_format() {
    let json = load_fixture();
    let (meta, records) = convert_guidellm(&json, &ConvertOptions::default()).unwrap();

    let mut buf = Vec::new();
    write_conversion(&mut buf, &meta, &records).unwrap();

    let (parsed_meta, parsed_records) = trace::read_trace(BufReader::new(buf.as_slice())).unwrap();
    assert_eq!(parsed_meta, meta);
    assert_eq!(parsed_records.len(), records.len());
    assert_eq!(parsed_records, records);
}

#[test]
fn fixture_summarize_produces_buckets() {
    let json = load_fixture();
    let (meta, records) = convert_guidellm(&json, &ConvertOptions::default()).unwrap();

    let mut buf = Vec::new();
    write_conversion(&mut buf, &meta, &records).unwrap();

    let (smeta, stats) = summarize_trace(BufReader::new(buf.as_slice())).unwrap();
    assert_eq!(smeta.source.as_deref(), Some("guidellm"));

    // Two concurrency buckets: 1 (synchronous) and 4 (concurrent).
    assert_eq!(stats.len(), 2);

    let bucket1 = stats.iter().find(|s| s.concurrency == 1).unwrap();
    assert_eq!(bucket1.count, 1);
    assert!((bucket1.ttft_p50 - 25.0).abs() < 0.01);
    assert_eq!(bucket1.prompt_min, 64);
    assert_eq!(bucket1.prompt_max, 64);

    let bucket4 = stats.iter().find(|s| s.concurrency == 4).unwrap();
    assert_eq!(bucket4.count, 3);
    assert_eq!(bucket4.prompt_min, 128);
    assert_eq!(bucket4.prompt_max, 256);
}

#[test]
fn fixture_summary_text_output() {
    let json = load_fixture();
    let (meta, records) = convert_guidellm(&json, &ConvertOptions::default()).unwrap();

    let mut buf = Vec::new();
    write_conversion(&mut buf, &meta, &records).unwrap();

    let (smeta, stats) = summarize_trace(BufReader::new(buf.as_slice())).unwrap();
    let mut out = Vec::new();
    write_summary(&mut out, &smeta, &stats).unwrap();
    let text = String::from_utf8(out).unwrap();

    assert!(text.contains("meta-llama/Llama-3.1-8B-Instruct"));
    assert!(text.contains("guidellm"));
    // Should have rows for both concurrency levels.
    assert!(text.contains("1"));
    assert!(text.contains("4"));
}
