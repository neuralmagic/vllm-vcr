#![cfg(feature = "rust-core")]
//! Record -> replay round trip on the `--use-rust-core` engine.
//!
//! ```text
//!   generation 1                              generation 2
//!   client ─▶ vllm-vcr record ─▶ play         client ─▶ vllm-vcr record ─▶ play
//!                  │           (source)                      │    (--replay-tokens trace 1
//!                  ▼                                         ▼     --replay-steps  trace 1)
//!               trace 1 ───────────────────────────────▶  trace 2
//! ```
//!
//! Both generations run vLLM's scheduler. The client streams of the two
//! generations must agree on token ids, per-message framing and finish reason,
//! and so must the two traces.

use std::io::BufReader;
use std::process::{Child, Command};
use std::time::Duration;

use futures::StreamExt;
use futures::future::join_all;
use sim_protocol::vllm::{
    EngineCoreFinishReason, EngineCoreRequest, EngineCoreSamplingParams, EngineCoreStreamOutput,
};
use sim_trace::steps::{StepRecord, read_step_trace_file};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};
use vllm_vcr::trace::{
    TraceFinishReason, TraceMeta, TraceRecord, prompt_block_hashes, read_trace, replay_subset,
    write_trace,
};
use vllm_vcr::{Opt, run};

const TIMEOUT: Duration = Duration::from_secs(30);
const BLOCK_SIZE: usize = 16;

#[derive(Clone)]
struct Ask {
    prompt_len: usize,
    max_tokens: u32,
    stop_token_ids: Vec<u32>,
}

impl Ask {
    fn new(prompt_len: usize, max_tokens: u32) -> Self {
        Ask {
            prompt_len,
            max_tokens,
            stop_token_ids: Vec::new(),
        }
    }
}

/// One request as the client saw it.
#[derive(Debug, PartialEq)]
struct Stream {
    chunks: Vec<Vec<u32>>,
    finish_reason: Option<EngineCoreFinishReason>,
}

impl Stream {
    fn framing(&self) -> Vec<usize> {
        self.chunks.iter().map(Vec::len).collect()
    }
}

struct Generation {
    streams: Vec<Stream>,
    /// First to last token-bearing message of each stream.
    decode_times: Vec<Duration>,
    /// The tap's records in replay order.
    records: Vec<TraceRecord>,
    trace_path: String,
}

struct Rig {
    tap: Child,
    engine: CancellationToken,
    ipc_paths: Vec<String>,
}

impl Drop for Rig {
    fn drop(&mut self) {
        self.engine.cancel();
        let _ = self.tap.kill();
        let _ = self.tap.wait();
        for path in &self.ipc_paths {
            let _ = std::fs::remove_file(path);
        }
    }
}

fn prompt(index: usize, len: usize) -> Vec<u32> {
    (0..len).map(|i| (index * 1000 + i) as u32).collect()
}

async fn collect(
    mut outputs: impl futures::Stream<Item = vllm_engine_core_client::Result<EngineCoreStreamOutput>>
    + Unpin,
) -> (Stream, Duration) {
    let mut stream = Stream {
        chunks: Vec::new(),
        finish_reason: None,
    };
    let mut first = None;
    let mut last = None;
    while let Some(output) = tokio::time::timeout(TIMEOUT, outputs.next())
        .await
        .expect("stream timed out")
    {
        let output = output.expect("stream error");
        if !output.new_token_ids.is_empty() {
            let now = Instant::now();
            first.get_or_insert(now);
            last = Some(now);
            stream.chunks.push(output.new_token_ids.clone());
        }
        if output.finish_reason.is_some() {
            stream.finish_reason = output.finish_reason;
        }
    }
    let decode_time = match (first, last) {
        (Some(first), Some(last)) => last.duration_since(first),
        _ => Duration::ZERO,
    };
    (stream, decode_time)
}

/// Run `play --use-rust-core <engine_args>` behind the recording tap and drive
/// `waves` through it: the requests of a wave are sent one after another and
/// are in flight together, and a wave starts when the one before it has
/// finished. Request `i` in send order is named `replay-{i}`.
async fn generation(name: &str, engine_args: &[&str], waves: &[Vec<Ask>]) -> Generation {
    let tag = format!("rc-rt-{name}-{}", std::process::id());
    let ipc = |role: &str| format!("/tmp/{tag}-{role}.ipc");
    let ipc_paths = vec![ipc("frontend"), ipc("engine"), ipc("input"), ipc("output")];
    for path in &ipc_paths {
        let _ = std::fs::remove_file(path);
    }
    let trace_path = format!("/tmp/{tag}-trace.jsonl");
    let _ = std::fs::remove_file(&trace_path);

    let tap = Command::new(env!("CARGO_BIN_EXE_vllm-vcr"))
        .arg("record")
        .args(["--frontend-handshake", &format!("ipc://{}", ipc_paths[0])])
        .args(["--engine-handshake", &format!("ipc://{}", ipc_paths[1])])
        .args(["--input-address", &format!("ipc://{}", ipc_paths[2])])
        .args(["--output-address", &format!("ipc://{}", ipc_paths[3])])
        .args(["--trace-out", &trace_path])
        .args(["--model", "round-trip"])
        .arg("--record-tokens")
        .env("RUST_LOG", "warn")
        .spawn()
        .expect("failed to spawn vllm-vcr record");
    tokio::time::sleep(Duration::from_millis(500)).await;

    let engine_handshake = format!("ipc://{}", ipc_paths[1]);
    let mut args = vec![
        "play",
        "--use-rust-core",
        "--handshake-address",
        &engine_handshake,
        "--tokens-per-block",
        "16",
    ];
    args.extend_from_slice(engine_args);
    let opt = Opt::parse_from(&args);
    let engine = CancellationToken::new();
    let engine_token = engine.clone();
    tokio::spawn(async move {
        run(opt, engine_token).await.expect("engine failed");
    });
    let rig = Rig {
        tap,
        engine,
        ipc_paths,
    };
    tokio::time::sleep(Duration::from_millis(500)).await;

    let config = EngineCoreClientConfig::new_single(format!("ipc://{}", rig.ipc_paths[0]));
    let client = tokio::time::timeout(TIMEOUT, EngineCoreClient::connect(config))
        .await
        .expect("client connect timed out")
        .expect("client connect failed");

    let mut streams = Vec::new();
    let mut decode_times = Vec::new();
    let mut index = 0;
    for wave in waves {
        let requests: Vec<EngineCoreRequest> = wave
            .iter()
            .map(|ask| {
                let request = EngineCoreRequest {
                    request_id: format!("replay-{index}"),
                    prompt_token_ids: Some(prompt(index, ask.prompt_len)),
                    sampling_params: Some(EngineCoreSamplingParams {
                        max_tokens: ask.max_tokens,
                        stop_token_ids: ask.stop_token_ids.clone(),
                        ..EngineCoreSamplingParams::for_test()
                    }),
                    ..Default::default()
                };
                index += 1;
                request
            })
            .collect();
        let mut in_flight = Vec::new();
        for request in requests {
            let outputs = client.call(request).await.expect("call failed");
            in_flight.push(collect(sim_protocol::vllm::output_stream(outputs)));
        }
        for (stream, decode_time) in join_all(in_flight).await {
            streams.push(stream);
            decode_times.push(decode_time);
        }
    }

    tokio::time::sleep(Duration::from_millis(200)).await;
    drop(rig);

    let file = std::fs::File::open(&trace_path).expect("the tap wrote a trace");
    let (_, records) = read_trace(BufReader::new(file)).expect("the trace parses");
    Generation {
        streams,
        decode_times,
        records: replay_subset(records),
        trace_path,
    }
}

/// Per-message token counts a trace record says the request was delivered in.
fn recorded_framing(record: &TraceRecord) -> Vec<usize> {
    let gaps = record.itl_ms.as_ref().map_or(0, Vec::len);
    let bursts: Vec<usize> = match &record.itl_tokens {
        Some(tokens) => tokens.iter().map(|&tokens| tokens as usize).collect(),
        None => vec![1; gaps],
    };
    let first = record.output_tokens - bursts.iter().sum::<usize>();
    std::iter::once(first).chain(bursts).collect()
}

fn trace_finish_reason(reason: Option<EngineCoreFinishReason>) -> Option<TraceFinishReason> {
    match reason? {
        EngineCoreFinishReason::Stop => Some(TraceFinishReason::Stop),
        EngineCoreFinishReason::Length => Some(TraceFinishReason::Length),
        other => panic!("unexpected finish reason {other:?}"),
    }
}

/// The tap's trace must describe exactly what the client saw.
fn assert_trace_matches_streams(generation: &Generation) {
    assert_eq!(generation.records.len(), generation.streams.len());
    for (i, (record, stream)) in generation
        .records
        .iter()
        .zip(&generation.streams)
        .enumerate()
    {
        assert_eq!(
            record.output_token_ids.as_deref(),
            Some(stream.chunks.concat().as_slice()),
            "request {i}: recorded token ids"
        );
        assert_eq!(
            recorded_framing(record),
            stream.framing(),
            "request {i}: recorded step framing"
        );
        assert_eq!(
            record.finish_reason,
            trace_finish_reason(stream.finish_reason),
            "request {i}: recorded finish reason"
        );
    }
}

/// Replay `recorded` on a fresh engine the way the arrival-replay harness does,
/// asking each request for its recorded length, and require the replay to
/// reproduce the capture.
async fn assert_replays(name: &str, recorded: &Generation, waves: &[Vec<Ask>]) {
    let mut lengths = recorded.records.iter().map(|record| record.output_tokens);
    let replay_waves: Vec<Vec<Ask>> = waves
        .iter()
        .map(|wave| {
            wave.iter()
                .zip(&mut lengths)
                .map(|(ask, length)| Ask::new(ask.prompt_len, length as u32))
                .collect()
        })
        .collect();
    let replayed = generation(
        name,
        &[
            "--replay-tokens",
            &recorded.trace_path,
            "--replay-steps",
            &recorded.trace_path,
        ],
        &replay_waves,
    )
    .await;

    for (i, (replay, capture)) in replayed.streams.iter().zip(&recorded.streams).enumerate() {
        assert_eq!(replay, capture, "request {i}: replayed stream");
    }
    assert_trace_matches_streams(&replayed);
    for (i, (replay, capture)) in replayed.records.iter().zip(&recorded.records).enumerate() {
        assert_eq!(replay.prompt_tokens, capture.prompt_tokens, "request {i}");
        assert_eq!(replay.output_tokens, capture.output_tokens, "request {i}");
        assert_eq!(replay.itl_tokens, capture.itl_tokens, "request {i}");

        let recorded_decode_ms: f64 = capture.itl_ms.iter().flatten().sum();
        let replayed_decode_ms = replayed.decode_times[i].as_secs_f64() * 1000.0;
        assert!(
            replayed_decode_ms >= 0.5 * recorded_decode_ms,
            "request {i}: replay decoded in {replayed_decode_ms:.1}ms, \
             the capture in {recorded_decode_ms:.1}ms"
        );
    }

    let _ = std::fs::remove_file(&replayed.trace_path);
}

#[tokio::test]
async fn speculative_bursts_survive_a_record_replay_round_trip() {
    let mut stopping = Ask::new(24, 2000);
    stopping.stop_token_ids = vec![3];
    let waves = vec![
        vec![Ask::new(32, 43)],
        vec![stopping],
        vec![Ask::new(40, 30), Ask::new(20, 25)],
    ];

    let captured = generation(
        "spec-capture",
        &[
            "--latency-trace",
            "tests/fixtures/spec_decode_trace.jsonl",
            "--vocab-size",
            "64",
        ],
        &waves,
    )
    .await;

    assert_trace_matches_streams(&captured);
    let lengths: Vec<usize> = captured
        .streams
        .iter()
        .map(|stream| stream.chunks.concat().len())
        .collect();
    assert_eq!([lengths[0], lengths[2], lengths[3]], [43, 30, 25]);
    assert_eq!(
        captured.streams[1].finish_reason,
        Some(EngineCoreFinishReason::Stop)
    );
    assert_eq!(captured.streams[1].chunks.concat().last(), Some(&3));
    assert!(
        captured
            .streams
            .iter()
            .any(|stream| stream.framing().iter().any(|&tokens| tokens > 1)),
        "the capture must contain multi-token steps"
    );

    assert_replays("spec-replay", &captured, &waves).await;
    let _ = std::fs::remove_file(&captured.trace_path);
}

#[tokio::test]
async fn a_first_block_and_a_recorded_stop_survive_a_record_replay_round_trip() {
    let prompt_len = 48;
    let output_token_ids: Vec<u32> = (5000..5020).collect();
    let seed_record = TraceRecord {
        prompt_tokens: prompt_len,
        output_tokens: output_token_ids.len(),
        ttft_ms: 20.0,
        itl_ms: Some(vec![15.0, 15.0]),
        itl_tokens: Some(vec![8, 4]),
        concurrency: 1,
        arrival_ms: Some(0.0),
        block_hashes: prompt_block_hashes(&prompt(0, prompt_len), BLOCK_SIZE),
        output_token_ids: Some(output_token_ids.clone()),
        finish_reason: Some(TraceFinishReason::Stop),
        ..Default::default()
    };
    let seed_path = format!("/tmp/rc-rt-seed-{}.jsonl", std::process::id());
    let meta = TraceMeta {
        source: Some("synthetic".to_string()),
        block_size: Some(BLOCK_SIZE),
        ..TraceMeta::default()
    };
    let mut seed_file = std::fs::File::create(&seed_path).expect("create the seed trace");
    write_trace(&mut seed_file, &meta, std::slice::from_ref(&seed_record))
        .expect("write the seed trace");
    drop(seed_file);

    let waves = vec![vec![Ask::new(prompt_len, 10_000)]];
    let captured = generation(
        "block-capture",
        &["--replay-tokens", &seed_path, "--replay-steps", &seed_path],
        &waves,
    )
    .await;

    assert_trace_matches_streams(&captured);
    assert_eq!(captured.streams[0].chunks.concat(), output_token_ids);
    assert_eq!(captured.streams[0].framing(), [8, 8, 4]);
    assert_eq!(
        captured.streams[0].finish_reason,
        Some(EngineCoreFinishReason::Stop)
    );

    assert_replays("block-replay", &captured, &waves).await;
    let _ = std::fs::remove_file(&captured.trace_path);
    let _ = std::fs::remove_file(&seed_path);
}

#[tokio::test]
async fn recorded_step_times_survive_a_record_replay_round_trip() {
    let step_trace = format!("/tmp/rc-rt-steps-{}.jsonl", std::process::id());
    let waves = vec![
        vec![Ask::new(40, 12)],
        vec![Ask::new(24, 8), Ask::new(56, 8)],
    ];

    let captured = generation(
        "steps-capture",
        &[
            "--inter-token-latency",
            "25",
            "--step-trace-out",
            &step_trace,
        ],
        &waves,
    )
    .await;
    let recorded = read_step_trace_file(std::path::Path::new(&step_trace)).expect("step trace");
    let prefill_tokens: u64 = recorded
        .steps
        .iter()
        .map(|step| step.shape.prefill_tokens())
        .sum();
    assert_eq!(prefill_tokens, 40 + 24 + 56);
    let sampled: u32 = recorded.steps.iter().map(|step| step.sampled).sum();
    assert_eq!(sampled, 12 + 8 + 8);
    assert!(
        recorded
            .steps
            .iter()
            .any(|step| step.shape.decode.reqs == 2),
        "the concurrent wave must show up as two-request decode steps"
    );

    let replayed = generation("steps-replay", &["--step-trace", &step_trace], &waves).await;
    for (i, (replay, capture)) in replayed
        .decode_times
        .iter()
        .zip(&captured.decode_times)
        .enumerate()
    {
        let (replay, capture) = (replay.as_secs_f64(), capture.as_secs_f64());
        assert!(
            (0.6 * capture..=2.0 * capture).contains(&replay),
            "request {i}: replay decoded in {replay:.3}s, the capture in {capture:.3}s"
        );
    }

    for path in [&step_trace, &captured.trace_path, &replayed.trace_path] {
        let _ = std::fs::remove_file(path);
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// Replays an engine-step trace captured from vLLM (`VCR_STEP_TRACE`, e.g. the
/// `engine-steps.jsonl` that `just capture-run` fetches) and compares the step
/// gaps the tap sees with the step times recorded at the same decode
/// concurrency. Run it through `just replay-step-capture <trace>`.
#[tokio::test]
#[ignore = "needs VCR_STEP_TRACE, an engine-step trace captured from vLLM"]
async fn a_captured_step_trace_replays_its_step_times() {
    let step_trace = std::env::var("VCR_STEP_TRACE").expect("VCR_STEP_TRACE names a step trace");
    let recorded = read_step_trace_file(std::path::Path::new(&step_trace)).expect("step trace");

    for concurrency in [1usize, 16] {
        let decodes: Vec<&StepRecord> = recorded
            .steps
            .iter()
            .filter(|step| {
                step.shape.prefill.is_empty() && step.shape.decode.reqs as usize == concurrency
            })
            .collect();
        assert!(
            !decodes.is_empty(),
            "the capture has no decode-only step with {concurrency} requests"
        );
        let mut want: Vec<f64> = decodes.iter().map(|step| step.step_ms).collect();
        let context: u64 = decodes
            .iter()
            .map(|step| step.shape.decode.context_tokens)
            .sum();
        let prompt_len = context as usize / (decodes.len() * concurrency);

        let waves = vec![vec![Ask::new(prompt_len, 48); concurrency]];
        let replayed = generation(
            &format!("capture-replay-{concurrency}"),
            &["--step-trace", &step_trace],
            &waves,
        )
        .await;
        let mut got: Vec<f64> = replayed
            .records
            .iter()
            .flat_map(|record| record.itl_ms.clone().unwrap_or_default())
            .collect();
        assert_eq!(got.len(), concurrency * 47);

        let (want, got) = (median(&mut want), median(&mut got));
        assert!(
            (0.9 * want..=1.1 * want).contains(&got),
            "{concurrency} concurrent: replayed median step gap {got:.2}ms, captured {want:.2}ms"
        );
        println!("{concurrency} concurrent: replayed {got:.2}ms, captured {want:.2}ms");
        let _ = std::fs::remove_file(&replayed.trace_path);
    }
}
