//! Engine-step trace: what vLLM's model runner and KV connector did, one record
//! per executed step and per KV transfer.
//!
//! ```text
//!   Scheduler::schedule() ──SchedulerOutput──▶ model runner ──ModelRunnerOutput──▶ update_from_output()
//!                         │                                                     │
//!                         └──────────── StepRecord: shape + wall time ──────────┘
//!
//!   connector metadata names a load ─────── KvTransferRecord ─────── finished_recving names it
//! ```
//!
//! The file is JSONL. Each line is one of `{"meta": {...}}` (first line,
//! optional, the same header a request trace carries), `{"step": {...}}` or
//! `{"kv_transfer": {...}}`. It sits next to the request trace the tap writes:
//! the tap sees the wire, this sees the two boundaries that are not sockets.

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::Path;
use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use rand::Rng as _;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::trace::{TraceMeta, open_trace_reader};

/// The decode work of one step: requests past their prompt.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecodeWork {
    pub reqs: u32,
    /// Tokens scheduled for those requests, drafts included.
    pub tokens: u32,
    /// Speculative drafts among `tokens`.
    #[serde(default)]
    pub drafts: u32,
    /// Tokens already in KV across those requests.
    pub context_tokens: u64,
}

/// One request's share of a step's prefill work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefillChunk {
    pub tokens: u32,
    /// Tokens of this request already in KV when the chunk ran.
    pub kv_depth: u32,
    /// The chunk reached the end of the prompt and sampled.
    pub last: bool,
}

/// What the model runner was asked to compute in one step.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepShape {
    pub decode: DecodeWork,
    #[serde(default)]
    pub prefill: Vec<PrefillChunk>,
}

impl StepShape {
    pub fn prefill_tokens(&self) -> u64 {
        self.prefill
            .iter()
            .map(|chunk| u64::from(chunk.tokens))
            .sum()
    }

    pub fn total_tokens(&self) -> u64 {
        u64::from(self.decode.tokens) + self.prefill_tokens()
    }
}

/// One executed engine step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepRecord {
    pub step: u64,
    /// When `schedule()` was called, in milliseconds since capture start.
    pub ts_ms: f64,
    /// Model runner wall time: from `schedule()` returning, or the step before
    /// leaving the runner if that was later, to `update_from_output()` being called.
    pub exec_ms: f64,
    /// Engine loop time this step accounts for: from `schedule()` being called,
    /// or the step before finishing its update if that was later, to this step
    /// finishing its update. Under async scheduling the scheduler runs while the
    /// runner is busy, so this is the step period rather than a sum of parts.
    pub step_ms: f64,
    #[serde(default)]
    pub sched_ms: f64,
    #[serde(default)]
    pub update_ms: f64,
    #[serde(flatten)]
    pub shape: StepShape,
    /// Tokens the model runner returned.
    #[serde(default)]
    pub sampled: u32,
    /// Speculative drafts it accepted.
    #[serde(default)]
    pub accepted: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvDirection {
    Recv,
    Send,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvOutcome {
    Ok,
    /// The connector reported the request in `failed_recving`.
    Failed,
    /// The connector reported some of the request's blocks as invalid.
    InvalidBlocks,
}

/// One KV transfer, from the step whose connector metadata started it to the
/// step whose connector output reported it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvTransferRecord {
    pub direction: KvDirection,
    pub blocks: u32,
    pub start_step: u64,
    pub end_step: u64,
    pub wall_ms: f64,
    pub outcome: KvOutcome,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Line {
    Meta(TraceMeta),
    Step(StepRecord),
    KvTransfer(KvTransferRecord),
}

/// A parsed engine-step trace.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StepTrace {
    pub meta: TraceMeta,
    pub steps: Vec<StepRecord>,
    pub kv_transfers: Vec<KvTransferRecord>,
}

fn append(writer: &mut impl Write, line: &Line) -> Result<()> {
    serde_json::to_writer(&mut *writer, line)?;
    writeln!(writer)?;
    Ok(())
}

pub fn append_step_meta(writer: &mut impl Write, meta: &TraceMeta) -> Result<()> {
    append(writer, &Line::Meta(meta.clone()))
}

pub fn append_step(writer: &mut impl Write, record: &StepRecord) -> Result<()> {
    append(writer, &Line::Step(record.clone()))
}

pub fn append_kv_transfer(writer: &mut impl Write, record: &KvTransferRecord) -> Result<()> {
    append(writer, &Line::KvTransfer(record.clone()))
}

/// Parse an engine-step trace. A last line cut off mid-write, as a file copied
/// out of a running capture has, is skipped with a warning.
pub fn read_step_trace(mut reader: impl BufRead) -> Result<StepTrace> {
    let mut trace = StepTrace::default();
    let mut line = String::new();
    let mut line_num = 0;
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Ok(trace);
        }
        line_num += 1;
        let complete = line.ends_with('\n');
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        match serde_json::from_str::<Line>(text) {
            Ok(Line::Meta(meta)) => trace.meta = meta,
            Ok(Line::Step(step)) => {
                validate_step(&step).with_context(|| format!("line {line_num}"))?;
                trace.steps.push(step);
            }
            Ok(Line::KvTransfer(transfer)) => {
                if !transfer.wall_ms.is_finite() || transfer.wall_ms < 0.0 {
                    bail!("line {line_num}: wall_ms must be a non-negative number");
                }
                trace.kv_transfers.push(transfer);
            }
            Err(error) if !complete => {
                warn!(line_num, %error, "step trace ends in a partial line; skipping it");
                return Ok(trace);
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("line {line_num}: not a step trace line"));
            }
        }
    }
}

pub fn read_step_trace_file(path: &Path) -> Result<StepTrace> {
    read_step_trace(open_trace_reader(path)?)
        .with_context(|| format!("reading step trace {}", path.display()))
}

fn validate_step(step: &StepRecord) -> Result<()> {
    for (name, value) in [("exec_ms", step.exec_ms), ("step_ms", step.step_ms)] {
        if !value.is_finite() || value < 0.0 {
            bail!("{name} must be a non-negative number, got {value}");
        }
    }
    if step.shape.decode.reqs == 0 && step.shape.decode.tokens > 0 {
        bail!("decode tokens without decode requests");
    }
    if step.shape.prefill.iter().any(|chunk| chunk.tokens == 0) {
        bail!("a prefill chunk of zero tokens");
    }
    Ok(())
}

/// Position of `value` on a scale of four buckets per doubling, zero kept apart.
fn bucket(value: u64) -> u8 {
    if value == 0 {
        return 0;
    }
    1 + ((value as f64).log2() * 4.0) as u8
}

/// Distance between two bucketed amounts of work. Work present on one side and
/// absent on the other costs `absent` more.
fn distance(a: u8, b: u8, absent: u32) -> u32 {
    u32::from((a == 0) != (b == 0)) * absent + u32::from(a.abs_diff(b))
}

/// The features a step's wall time is looked up by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct ShapeKey {
    decode_reqs: u8,
    /// Mean context per decode request.
    decode_context: u8,
    prefill_tokens: u8,
    /// Deepest KV any chunk extends.
    prefill_depth: u8,
}

impl ShapeKey {
    fn of(shape: &StepShape) -> Self {
        let reqs = u64::from(shape.decode.reqs);
        ShapeKey {
            decode_reqs: bucket(reqs),
            decode_context: bucket(shape.decode.context_tokens.checked_div(reqs).unwrap_or(0)),
            prefill_tokens: bucket(shape.prefill_tokens()),
            prefill_depth: bucket(
                shape
                    .prefill
                    .iter()
                    .map(|chunk| u64::from(chunk.kv_depth))
                    .max()
                    .unwrap_or(0),
            ),
        }
    }

    fn distance(self, other: Self) -> u32 {
        distance(self.decode_reqs, other.decode_reqs, 256)
            + distance(self.prefill_tokens, other.prefill_tokens, 1024)
            + u32::from(self.decode_context.abs_diff(other.decode_context))
            + u32::from(self.prefill_depth.abs_diff(other.prefill_depth))
    }
}

/// Recorded step times, looked up by the shape of the step being priced.
///
/// ```text
///   StepShape ─▶ ShapeKey ─▶ steps recorded with that key ─▶ one of their times
///                   │ no such key
///                   └──▶ the recorded key closest to it
/// ```
pub struct StepTable {
    /// Step times in milliseconds per key, in key order.
    samples: Vec<(ShapeKey, Vec<f64>)>,
    index: HashMap<ShapeKey, usize>,
}

impl StepTable {
    pub fn from_records(records: &[StepRecord]) -> Result<Self> {
        let mut by_key: HashMap<ShapeKey, Vec<f64>> = HashMap::new();
        for record in records {
            if record.shape.total_tokens() == 0 {
                continue;
            }
            let ms = if record.step_ms > 0.0 {
                record.step_ms
            } else {
                record.exec_ms
            };
            by_key
                .entry(ShapeKey::of(&record.shape))
                .or_default()
                .push(ms);
        }
        if by_key.is_empty() {
            bail!("the step trace has no executed steps to price from");
        }
        let mut samples: Vec<_> = by_key.into_iter().collect();
        samples.sort_by_key(|&(key, _)| key);
        let index = samples
            .iter()
            .enumerate()
            .map(|(i, &(key, _))| (key, i))
            .collect();
        Ok(StepTable { samples, index })
    }

    /// Number of distinct step shapes recorded.
    pub fn num_shapes(&self) -> usize {
        self.samples.len()
    }

    /// A recorded time of a step shaped like `shape`. Zero for a step that
    /// schedules nothing.
    pub fn price(&self, shape: &StepShape, rng: &mut StdRng) -> Duration {
        if shape.total_tokens() == 0 {
            return Duration::ZERO;
        }
        let key = ShapeKey::of(shape);
        let nearest = self.index.get(&key).copied().or_else(|| {
            self.samples
                .iter()
                .enumerate()
                .min_by_key(|&(_, &(recorded, _))| key.distance(recorded))
                .map(|(i, _)| i)
        });
        let Some((_, times)) = nearest.and_then(|i| self.samples.get(i)) else {
            return Duration::ZERO;
        };
        let ms = times[rng.random_range(0..times.len())];
        Duration::from_secs_f64(ms / 1000.0)
    }
}

/// How a replayed KV load goes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KvTransferDraw {
    pub duration: Duration,
    pub outcome: KvOutcome,
}

/// Recorded KV loads, looked up by how many blocks move.
pub struct KvTransferTable {
    /// Recorded loads per block-count bucket, in bucket order.
    samples: Vec<(u8, Vec<KvTransferDraw>)>,
}

impl KvTransferTable {
    /// `None` when the trace recorded no loads.
    pub fn from_records(records: &[KvTransferRecord]) -> Option<Self> {
        let mut by_bucket: HashMap<u8, Vec<KvTransferDraw>> = HashMap::new();
        for record in records {
            if record.direction != KvDirection::Recv {
                continue;
            }
            by_bucket
                .entry(bucket(u64::from(record.blocks)))
                .or_default()
                .push(KvTransferDraw {
                    duration: Duration::from_secs_f64(record.wall_ms / 1000.0),
                    outcome: record.outcome,
                });
        }
        if by_bucket.is_empty() {
            return None;
        }
        let mut samples: Vec<_> = by_bucket.into_iter().collect();
        samples.sort_by_key(|&(blocks, _)| blocks);
        Some(KvTransferTable { samples })
    }

    /// A recorded load of about `blocks` blocks.
    pub fn draw(&self, blocks: usize, rng: &mut StdRng) -> KvTransferDraw {
        let wanted = bucket(blocks as u64);
        let nearest = self
            .samples
            .iter()
            .min_by_key(|&&(recorded, _)| distance(wanted, recorded, 0));
        match nearest {
            Some((_, draws)) => draws[rng.random_range(0..draws.len())],
            None => KvTransferDraw {
                duration: Duration::ZERO,
                outcome: KvOutcome::Ok,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::time::Duration;

    use rand::SeedableRng as _;
    use rand::rngs::StdRng;

    use crate::steps::{
        DecodeWork, KvDirection, KvOutcome, KvTransferRecord, KvTransferTable, PrefillChunk,
        StepRecord, StepShape, StepTable, StepTrace, append_kv_transfer, append_step,
        append_step_meta, bucket, read_step_trace,
    };
    use crate::trace::TraceMeta;

    fn decode(reqs: u32, context_per_req: u64) -> StepShape {
        StepShape {
            decode: DecodeWork {
                reqs,
                tokens: reqs,
                drafts: 0,
                context_tokens: u64::from(reqs) * context_per_req,
            },
            prefill: Vec::new(),
        }
    }

    fn prefill(tokens: u32, kv_depth: u32) -> StepShape {
        StepShape {
            decode: DecodeWork::default(),
            prefill: vec![PrefillChunk {
                tokens,
                kv_depth,
                last: true,
            }],
        }
    }

    fn step(shape: StepShape, step_ms: f64) -> StepRecord {
        StepRecord {
            step: 0,
            ts_ms: 0.0,
            exec_ms: step_ms,
            step_ms,
            sched_ms: 0.0,
            update_ms: 0.0,
            shape,
            sampled: 0,
            accepted: 0,
        }
    }

    fn transfer(blocks: u32, wall_ms: f64, outcome: KvOutcome) -> KvTransferRecord {
        KvTransferRecord {
            direction: KvDirection::Recv,
            blocks,
            start_step: 1,
            end_step: 2,
            wall_ms,
            outcome,
        }
    }

    fn rng() -> StdRng {
        StdRng::seed_from_u64(7)
    }

    #[test]
    fn buckets_keep_small_counts_apart_and_zero_alone() {
        let small: Vec<u8> = (0..=8).map(bucket).collect();
        let mut distinct = small.clone();
        distinct.dedup();
        assert_eq!(small, distinct);
        assert_eq!(bucket(0), 0);
        assert_eq!(bucket(1024), bucket(1100));
        assert_ne!(bucket(1024), bucket(1300));
    }

    #[test]
    fn a_trace_round_trips_through_its_file_format() {
        let meta = TraceMeta {
            model: Some("m".to_string()),
            block_size: Some(16),
            ..TraceMeta::default()
        };
        let mut mixed = prefill(512, 2048);
        mixed.decode = decode(3, 900).decode;
        let steps = vec![step(decode(2, 100), 11.5), step(mixed, 48.25)];
        let transfers = vec![
            transfer(64, 30.0, KvOutcome::Ok),
            transfer(8, 2.0, KvOutcome::Failed),
        ];

        let mut file = Vec::new();
        append_step_meta(&mut file, &meta).unwrap();
        append_step(&mut file, &steps[0]).unwrap();
        append_kv_transfer(&mut file, &transfers[0]).unwrap();
        append_step(&mut file, &steps[1]).unwrap();
        append_kv_transfer(&mut file, &transfers[1]).unwrap();

        let trace = read_step_trace(Cursor::new(file)).unwrap();
        assert_eq!(
            trace,
            StepTrace {
                meta,
                steps,
                kv_transfers: transfers
            }
        );
    }

    #[test]
    fn the_recorder_line_format_parses() {
        let file = concat!(
            r#"{"meta": {"source": "vcr_step_recorder", "async_scheduling": true}}"#,
            "\n",
            r#"{"step": {"step": 4, "ts_ms": 10.5, "exec_ms": 20.0, "step_ms": 21.0, "#,
            r#""sched_ms": 0.4, "update_ms": 0.6, "#,
            r#""decode": {"reqs": 2, "tokens": 10, "drafts": 8, "context_tokens": 300}, "#,
            r#""prefill": [{"tokens": 512, "kv_depth": 1024, "last": false}], "#,
            r#""sampled": 5, "accepted": 3}}"#,
            "\n",
            r#"{"kv_transfer": {"direction": "recv", "blocks": 33, "start_step": 4, "#,
            r#""end_step": 6, "wall_ms": 41.5, "outcome": "invalid_blocks"}}"#,
            "\n",
        );
        let trace = read_step_trace(Cursor::new(file)).unwrap();
        assert_eq!(trace.meta.source.as_deref(), Some("vcr_step_recorder"));
        assert_eq!(trace.steps[0].shape.decode.drafts, 8);
        assert_eq!(trace.steps[0].shape.prefill[0].kv_depth, 1024);
        assert_eq!(trace.steps[0].shape.total_tokens(), 522);
        assert_eq!(trace.kv_transfers[0].outcome, KvOutcome::InvalidBlocks);
    }

    #[test]
    fn a_trace_written_by_the_vllm_recorder_parses_and_prices() {
        let file = include_str!("../../../tests/fixtures/engine_steps_vllm.jsonl");
        let trace = read_step_trace(Cursor::new(file)).unwrap();
        assert_eq!(trace.meta.source.as_deref(), Some("vcr_step_recorder"));
        let shapes: Vec<&StepShape> = trace.steps.iter().map(|step| &step.shape).collect();
        assert_eq!(
            shapes[0].prefill,
            [PrefillChunk {
                tokens: 64,
                kv_depth: 0,
                last: false
            }]
        );
        assert_eq!(
            shapes[1].prefill,
            [PrefillChunk {
                tokens: 36,
                kv_depth: 64,
                last: true
            }]
        );
        let decodes: Vec<&StepShape> = shapes[2..].to_vec();
        let expected: Vec<StepShape> = (0..5).map(|step| decode(1, 100 + step)).collect();
        assert_eq!(decodes, expected.iter().collect::<Vec<_>>());
        assert_eq!(trace.steps.iter().map(|step| step.sampled).sum::<u32>(), 8);

        let table = StepTable::from_records(&trace.steps).unwrap();
        assert_eq!(
            table.price(&prefill(64, 0), &mut rng()),
            Duration::from_secs_f64(trace.steps[0].step_ms / 1000.0)
        );
    }

    #[test]
    fn a_partial_last_line_is_skipped_and_a_bad_full_line_is_an_error() {
        let mut file = Vec::new();
        append_step(&mut file, &step(decode(1, 10), 5.0)).unwrap();
        let complete = file.clone();
        file.extend_from_slice(br#"{"step": {"step": 2, "ts_"#);
        assert_eq!(read_step_trace(Cursor::new(file)).unwrap().steps.len(), 1);

        let mut bad = complete;
        bad.extend_from_slice(b"{\"request\": {}}\n");
        let error = read_step_trace(Cursor::new(bad)).unwrap_err();
        assert!(format!("{error:#}").contains("line 2"));
    }

    #[test]
    fn invalid_records_are_rejected() {
        let mut negative = Vec::new();
        append_step(&mut negative, &step(decode(1, 10), -1.0)).unwrap();
        assert!(read_step_trace(Cursor::new(negative)).is_err());

        let mut empty_chunk = Vec::new();
        append_step(&mut empty_chunk, &step(prefill(0, 0), 1.0)).unwrap();
        assert!(read_step_trace(Cursor::new(empty_chunk)).is_err());
    }

    #[test]
    fn a_recorded_shape_is_priced_at_its_recorded_time() {
        let table = StepTable::from_records(&[
            step(decode(1, 100), 8.0),
            step(decode(16, 100), 20.0),
            step(prefill(2048, 0), 90.0),
            step(prefill(2048, 8192), 140.0),
        ])
        .unwrap();
        assert_eq!(table.num_shapes(), 4);
        let mut rng = rng();
        let ms = |shape: StepShape, rng: &mut StdRng| table.price(&shape, rng).as_secs_f64() * 1e3;
        assert_eq!(ms(decode(1, 100), &mut rng), 8.0);
        assert_eq!(ms(decode(16, 100), &mut rng), 20.0);
        assert_eq!(ms(prefill(2048, 0), &mut rng), 90.0);
        assert_eq!(ms(prefill(2048, 8192), &mut rng), 140.0);
    }

    #[test]
    fn an_unrecorded_shape_takes_the_closest_recorded_one() {
        let table = StepTable::from_records(&[
            step(decode(1, 100), 8.0),
            step(decode(16, 100), 20.0),
            step(prefill(2048, 0), 90.0),
        ])
        .unwrap();
        let mut rng = rng();
        assert_eq!(
            table.price(&decode(14, 110), &mut rng),
            Duration::from_millis(20)
        );
        assert_eq!(
            table.price(&decode(2, 90), &mut rng),
            Duration::from_millis(8)
        );
        assert_eq!(
            table.price(&prefill(1500, 0), &mut rng),
            Duration::from_millis(90)
        );
        let mut mixed = prefill(2048, 0);
        mixed.decode = decode(1, 100).decode;
        assert_eq!(table.price(&mixed, &mut rng), Duration::from_millis(90));
    }

    #[test]
    fn a_shape_recorded_many_times_replays_its_distribution() {
        let records: Vec<_> = [10.0, 12.0, 14.0]
            .into_iter()
            .map(|ms| step(decode(4, 500), ms))
            .collect();
        let table = StepTable::from_records(&records).unwrap();
        let mut rng = rng();
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..200 {
            seen.insert(table.price(&decode(4, 500), &mut rng).as_millis());
        }
        assert_eq!(seen.into_iter().collect::<Vec<_>>(), [10, 12, 14]);
    }

    #[test]
    fn step_ms_prices_a_step_and_exec_ms_stands_in_when_it_is_missing() {
        let mut both = step(decode(1, 10), 30.0);
        both.exec_ms = 25.0;
        let mut exec_only = step(decode(8, 10), 0.0);
        exec_only.exec_ms = 40.0;
        let table = StepTable::from_records(&[both, exec_only]).unwrap();
        let mut rng = rng();
        assert_eq!(
            table.price(&decode(1, 10), &mut rng),
            Duration::from_millis(30)
        );
        assert_eq!(
            table.price(&decode(8, 10), &mut rng),
            Duration::from_millis(40)
        );
    }

    #[test]
    fn an_empty_step_costs_nothing_and_an_empty_trace_cannot_price() {
        let table = StepTable::from_records(&[step(decode(1, 10), 5.0)]).unwrap();
        assert_eq!(
            table.price(&StepShape::default(), &mut rng()),
            Duration::ZERO
        );
        assert!(StepTable::from_records(&[]).is_err());
        assert!(StepTable::from_records(&[step(StepShape::default(), 1.0)]).is_err());
    }

    #[test]
    fn a_kv_load_is_drawn_from_loads_of_a_similar_size() {
        let table = KvTransferTable::from_records(&[
            transfer(8, 2.0, KvOutcome::Ok),
            transfer(512, 60.0, KvOutcome::Failed),
        ])
        .unwrap();
        let mut rng = rng();
        let small = table.draw(10, &mut rng);
        assert_eq!(small.duration, Duration::from_millis(2));
        assert_eq!(small.outcome, KvOutcome::Ok);
        let large = table.draw(400, &mut rng);
        assert_eq!(large.duration, Duration::from_millis(60));
        assert_eq!(large.outcome, KvOutcome::Failed);
    }

    #[test]
    fn sends_alone_give_no_load_table() {
        let mut send = transfer(8, 2.0, KvOutcome::Ok);
        send.direction = KvDirection::Send;
        assert!(KvTransferTable::from_records(&[send]).is_none());
        assert!(KvTransferTable::from_records(&[]).is_none());
    }
}
