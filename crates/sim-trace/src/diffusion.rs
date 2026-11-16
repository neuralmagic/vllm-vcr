//! JSONL trace of a vLLM-Omni diffusion stage, and the timing model a replay
//! builds from it.
//!
//! A diffusion request has no tokens, so it does not fit [`TraceRecord`]. The
//! file layout is the same as a token trace: a `{"meta": {...}}` line, whose
//! `kind` is [`TraceKind::Diffusion`], then one [`DiffusionRecord`] per
//! request that reached an outcome.
//!
//! [`TraceRecord`]: crate::trace::TraceRecord

use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, Write};
use std::path::Path;
use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use serde::{Deserialize, Serialize};

use crate::trace::{
    TraceKind, TraceMeta, is_unfinalized_gz_eof, open_trace_reader, parse_meta_line,
    warn_unfinalized_trace,
};

/// Pixel count a [`StepModel`]'s step time is quoted for.
pub const REFERENCE_PIXELS: u64 = 1024 * 1024;

/// Name of the stage metric holding the time a request waited for an
/// execution slot inside the stage.
pub const QUEUE_WAIT_METRIC: &str = "scheduler_queue_wait_ms";

/// Name of the stage metric holding the time from a request's admission to
/// its output leaving the execution slot. It includes the wait for the slot.
pub const EXEC_TIME_METRIC: &str = "diffusion_engine_exec_time_ms";

/// Pixel dimensions of an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ImageSize {
    pub width: u32,
    pub height: u32,
}

impl ImageSize {
    pub fn pixels(self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

/// What a request asked for, as the head sent it. Unset fields are the ones
/// the client left to the stage's defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DiffusionShape {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<ImageSize>,
    /// `num_inference_steps`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    /// `num_outputs_per_prompt`.
    pub images: u32,
    /// Images sent with the request: zero for text-to-image, one or more for
    /// an edit.
    #[serde(default)]
    pub input_images: u32,
}

/// How a request ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum DiffusionOutcome {
    /// The stage returned images.
    Finished,
    /// The stage answered with an error. `status` is its `status_code`, which
    /// a stage may leave unset.
    Error {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        status: Option<u16>,
    },
    /// The head aborted the request before the stage answered.
    Aborted,
}

/// One request's observed timing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiffusionRecord {
    /// Arrival in milliseconds since the start of the capture.
    pub arrival_ms: f64,
    #[serde(flatten)]
    pub shape: DiffusionShape,
    /// Requests in flight at arrival, this one included.
    pub concurrency: u32,
    /// Arrival to the stage's answer as the tap saw it. For an aborted
    /// request, arrival to the abort.
    pub latency_ms: f64,
    #[serde(flatten)]
    pub outcome: DiffusionOutcome,
    /// Size of the images the stage returned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_size: Option<ImageSize>,
    /// The numeric entries of the stage's own `metrics` for this request
    /// (`diffusion_engine_exec_time_ms`, `scheduler_queue_wait_ms`, ...).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub stage_metrics: BTreeMap<String, f64>,
}

impl DiffusionRecord {
    /// The size the request ran at: what it asked for, or what came back when
    /// it left the size to the stage.
    pub fn resolved_size(&self) -> Option<ImageSize> {
        self.shape.size.or(self.output_size)
    }

    /// Time the stage spent on this request alone: the latency less the time
    /// the stage reports the request waited for an execution slot.
    ///
    /// The stage's exec time splits it in two: the slot runs to the end of
    /// the exec time, the tail from there to the answer. Without the stage's
    /// exec time it is all slot.
    pub fn service(&self) -> Service {
        let duration = |ms: f64| Duration::from_secs_f64(ms.max(0.0) / 1000.0);
        let waited = self
            .stage_metrics
            .get(QUEUE_WAIT_METRIC)
            .copied()
            .unwrap_or(0.0);
        match self.stage_metrics.get(EXEC_TIME_METRIC) {
            Some(exec) => Service {
                slot: duration(exec - waited),
                tail: duration(self.latency_ms - exec),
            },
            None => Service {
                slot: duration(self.latency_ms - waited),
                tail: Duration::ZERO,
            },
        }
    }

    /// [`DiffusionRecord::service`] in total, in milliseconds.
    pub fn service_ms(&self) -> f64 {
        self.service().total().as_secs_f64() * 1000.0
    }
}

/// A request's time at the stage, less its wait for an execution slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Service {
    /// Time the request held an execution slot.
    pub slot: Duration,
    /// Time from releasing the slot to the answer reaching the tap.
    pub tail: Duration,
}

impl Service {
    pub fn total(&self) -> Duration {
        self.slot + self.tail
    }
}

/// Write the `{"meta": ...}` line of a diffusion trace.
pub fn write_meta(writer: &mut impl Write, meta: &TraceMeta) -> Result<()> {
    if meta.kind != TraceKind::Diffusion {
        bail!("a diffusion trace's meta must have kind `diffusion`");
    }
    crate::trace::write_meta(writer, meta)
}

pub fn append_record(writer: &mut impl Write, record: &DiffusionRecord) -> Result<()> {
    serde_json::to_writer(&mut *writer, record)?;
    writeln!(writer)?;
    Ok(())
}

/// Open and parse a diffusion trace file (gzip-aware).
pub fn read_trace_file(path: &Path) -> Result<(TraceMeta, Vec<DiffusionRecord>)> {
    read_trace(open_trace_reader(path)?)
        .with_context(|| format!("parsing diffusion trace file: {}", path.display()))
}

/// Parse a diffusion trace. The first non-blank line must be a meta line of
/// kind `diffusion`: a token trace, or a file with no meta line, is an error.
pub fn read_trace(reader: impl BufRead) -> Result<(TraceMeta, Vec<DiffusionRecord>)> {
    let mut meta = None;
    let mut records = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        let line_num = index + 1;
        let line = match line {
            Ok(line) => line,
            Err(error) if is_unfinalized_gz_eof(&error) => {
                warn_unfinalized_trace(line_num, records.len());
                break;
            }
            Err(error) => return Err(error).with_context(|| format!("reading line {line_num}")),
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if meta.is_none() {
            let parsed = parse_meta_line(line, line_num)?;
            match parsed {
                Some(parsed) if parsed.kind == TraceKind::Diffusion => meta = Some(parsed),
                Some(_) | None => bail!(
                    "not a diffusion trace: line {line_num} is not a meta line of kind `diffusion`"
                ),
            }
            continue;
        }
        let record: DiffusionRecord = serde_json::from_str(line)
            .with_context(|| format!("line {line_num}: invalid diffusion record"))?;
        records.push(record);
    }
    let Some(meta) = meta else {
        bail!("not a diffusion trace: the file is empty");
    };
    Ok((meta, records))
}

/// The knob latency model: a fixed cost per request plus a per-step cost that
/// scales with pixel count and image count.
///
/// ```text
/// time = overhead + step_time x steps x images x (width x height / 1024^2)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepModel {
    /// Fixed cost per request.
    pub overhead: Duration,
    /// One denoising step for one image of [`REFERENCE_PIXELS`].
    pub step_time: Duration,
}

/// Denoising steps a request runs, in units of one step on one
/// [`REFERENCE_PIXELS`] image.
pub fn step_units(size: ImageSize, steps: u32, images: u32) -> f64 {
    f64::from(steps) * f64::from(images) * (size.pixels() as f64 / REFERENCE_PIXELS as f64)
}

impl StepModel {
    pub fn exec_time(&self, step_units: f64) -> Duration {
        self.overhead + self.step_time.mul_f64(step_units)
    }

    /// Fit the model to `(step_units, service_ms)` points by least squares.
    ///
    /// When the points hold one distinct step count, or the fit has a
    /// negative overhead or step time, the fit is a line through the origin,
    /// `step_time = sum(service) / sum(step_units)`, and a constant
    /// `overhead` when that slope is not positive either. `None` when there
    /// are no points.
    pub fn fit(points: &[(f64, f64)]) -> Option<Self> {
        if points.is_empty() {
            return None;
        }
        let count = points.len() as f64;
        let mean_units = points.iter().map(|(units, _)| units).sum::<f64>() / count;
        let mean_service = points.iter().map(|(_, service)| service).sum::<f64>() / count;
        let spread: f64 = points
            .iter()
            .map(|(units, _)| (units - mean_units).powi(2))
            .sum();
        if spread > f64::EPSILON {
            let covariance: f64 = points
                .iter()
                .map(|(units, service)| (units - mean_units) * (service - mean_service))
                .sum();
            let step_ms = covariance / spread;
            let overhead_ms = mean_service - step_ms * mean_units;
            if step_ms >= 0.0 && overhead_ms >= 0.0 {
                return Some(Self::from_ms(overhead_ms, step_ms));
            }
        }
        if mean_units > 0.0 && mean_service > 0.0 {
            return Some(Self::from_ms(0.0, mean_service / mean_units));
        }
        Some(Self::from_ms(mean_service.max(0.0), 0.0))
    }

    fn from_ms(overhead_ms: f64, step_ms: f64) -> Self {
        Self {
            overhead: Duration::from_secs_f64(overhead_ms / 1000.0),
            step_time: Duration::from_secs_f64(step_ms / 1000.0),
        }
    }
}

/// A recorded request's timing, replayed for a request of the same shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Recorded {
    /// The recorded request's service time.
    pub service: Service,
    /// The size of the images the recorded request returned.
    pub output_size: Option<ImageSize>,
}

#[derive(Clone)]
struct ShapeSamples {
    recorded: Vec<Recorded>,
    next: usize,
    /// The size requests of this shape ran at, when any record shows it.
    size: Option<ImageSize>,
}

impl ShapeSamples {
    fn push(&mut self, record: &DiffusionRecord) {
        self.recorded.push(Recorded {
            service: record.service(),
            output_size: record.output_size,
        });
        self.size = self.size.or(record.resolved_size());
    }

    fn median_service_ms(&self) -> f64 {
        median(
            self.recorded
                .iter()
                .map(|recorded| recorded.service.total().as_secs_f64() * 1000.0)
                .collect(),
        )
    }
}

/// The middle value, or the mean of the middle two. Zero for no values.
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    match values.len() {
        0 => 0.0,
        len if len % 2 == 1 => values[middle],
        _ => (values[middle - 1] + values[middle]) / 2.0,
    }
}

/// Timing for a mock diffusion stage, taken from a recorded trace.
///
/// A request whose [`DiffusionShape`] the trace contains replays that shape's
/// recorded service times in arrival order, wrapping around. Any other request
/// is timed by [`DiffusionReplay::fitted`], a [`StepModel`] fitted to the
/// median service time of each recorded shape. Only finished requests are
/// used.
#[derive(Clone)]
pub struct DiffusionReplay {
    by_shape: HashMap<DiffusionShape, ShapeSamples>,
    fitted: StepModel,
}

impl std::fmt::Debug for DiffusionReplay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiffusionReplay")
            .field("shapes", &self.by_shape.len())
            .field("fitted", &self.fitted)
            .finish()
    }
}

impl DiffusionReplay {
    pub fn from_records(records: &[DiffusionRecord]) -> Result<Self> {
        let mut finished: Vec<&DiffusionRecord> = records
            .iter()
            .filter(|record| record.outcome == DiffusionOutcome::Finished)
            .collect();
        if finished.is_empty() {
            bail!("diffusion trace has no finished request to take timing from");
        }
        finished.sort_by(|a, b| a.arrival_ms.total_cmp(&b.arrival_ms));

        let mut by_shape: HashMap<DiffusionShape, ShapeSamples> = HashMap::new();
        for record in &finished {
            by_shape
                .entry(record.shape)
                .or_insert_with(|| ShapeSamples {
                    recorded: Vec::new(),
                    next: 0,
                    size: None,
                })
                .push(record);
        }

        let points: Vec<(f64, f64)> = by_shape
            .iter()
            .filter_map(|(shape, samples)| {
                Some((
                    step_units(samples.size?, shape.steps?, shape.images),
                    samples.median_service_ms(),
                ))
            })
            .collect();
        let fitted = match StepModel::fit(&points) {
            Some(fitted) => fitted,
            None => StepModel::from_ms(
                median(finished.iter().map(|record| record.service_ms()).collect()),
                0.0,
            ),
        };
        Ok(Self { by_shape, fitted })
    }

    /// The next recorded timing for `shape`. `None` when the trace has no
    /// finished request of that shape.
    pub fn next_recorded(&mut self, shape: &DiffusionShape) -> Option<Recorded> {
        let samples = self.by_shape.get_mut(shape)?;
        let recorded = samples.recorded.get(samples.next).copied()?;
        samples.next = (samples.next + 1) % samples.recorded.len();
        Some(recorded)
    }

    /// The model that times a request shape the trace does not contain.
    ///
    /// It is fitted to the recorded shapes whose size and step count are
    /// known. When the trace has none, it is a constant: the median service
    /// time of every finished request.
    pub fn fitted(&self) -> StepModel {
        self.fitted
    }

    /// Distinct request shapes the trace has timing for.
    pub fn shapes(&self) -> usize {
        self.by_shape.len()
    }
}

/// Counts and latency quantiles for the requests of one shape.
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeStats {
    pub shape: DiffusionShape,
    pub finished: usize,
    pub errors: usize,
    pub aborted: usize,
    /// Quantiles over the finished requests; zero when there are none.
    pub latency_p50: f64,
    pub latency_p90: f64,
    pub latency_max: f64,
    pub service_p50: f64,
}

fn quantile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (fraction * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

/// Per-shape statistics, ordered by pixel count, steps, image count, and
/// input images.
pub fn summarize(records: &[DiffusionRecord]) -> Vec<ShapeStats> {
    let mut groups: HashMap<DiffusionShape, Vec<&DiffusionRecord>> = HashMap::new();
    for record in records {
        groups.entry(record.shape).or_default().push(record);
    }
    let mut stats: Vec<ShapeStats> = groups
        .into_iter()
        .map(|(shape, group)| {
            let finished: Vec<&&DiffusionRecord> = group
                .iter()
                .filter(|record| record.outcome == DiffusionOutcome::Finished)
                .collect();
            let mut latency: Vec<f64> = finished.iter().map(|record| record.latency_ms).collect();
            latency.sort_by(f64::total_cmp);
            let mut service: Vec<f64> = finished.iter().map(|record| record.service_ms()).collect();
            service.sort_by(f64::total_cmp);
            ShapeStats {
                shape,
                finished: finished.len(),
                errors: group
                    .iter()
                    .filter(|record| matches!(record.outcome, DiffusionOutcome::Error { .. }))
                    .count(),
                aborted: group
                    .iter()
                    .filter(|record| record.outcome == DiffusionOutcome::Aborted)
                    .count(),
                latency_p50: quantile(&latency, 0.5),
                latency_p90: quantile(&latency, 0.9),
                latency_max: latency.last().copied().unwrap_or(0.0),
                service_p50: quantile(&service, 0.5),
            }
        })
        .collect();
    stats.sort_by_key(|stats| {
        let shape = stats.shape;
        (
            shape.size.map(ImageSize::pixels),
            shape.size.map(|size| size.width),
            shape.steps,
            shape.images,
            shape.input_images,
        )
    });
    stats
}

fn or_default<T: std::fmt::Display>(value: Option<T>) -> String {
    value.map_or_else(|| "default".to_string(), |value| value.to_string())
}

/// Write the summary as aligned plain text.
pub fn write_summary(
    writer: &mut impl Write,
    meta: &TraceMeta,
    records: &[DiffusionRecord],
) -> Result<()> {
    writeln!(writer, "kind:  diffusion")?;
    if let Some(model) = &meta.model {
        writeln!(writer, "model: {model}")?;
    }
    if let Some(gpu) = &meta.gpu {
        writeln!(writer, "gpu:   {gpu}")?;
    }
    if let Some(source) = &meta.source {
        writeln!(writer, "src:   {source}")?;
    }
    if let Ok(replay) = DiffusionReplay::from_records(records) {
        let fitted = replay.fitted();
        writeln!(
            writer,
            "fit:   overhead {:.1} ms + {:.2} ms per step at 1024x1024",
            fitted.overhead.as_secs_f64() * 1000.0,
            fitted.step_time.as_secs_f64() * 1000.0,
        )?;
    }
    writeln!(writer)?;
    writeln!(
        writer,
        "{:>11}  {:>7}  {:>6}  {:>6}  {:>8}  {:>6}  {:>7}  {:>30}  {:>11}",
        "size",
        "steps",
        "images",
        "inputs",
        "finished",
        "errors",
        "aborted",
        "latency_ms (p50/p90/max)",
        "service_p50"
    )?;
    for stats in summarize(records) {
        let shape = stats.shape;
        writeln!(
            writer,
            "{:>11}  {:>7}  {:>6}  {:>6}  {:>8}  {:>6}  {:>7}  {:>9.1}/{:>9.1}/{:>9.1}  {:>11.1}",
            or_default(
                shape
                    .size
                    .map(|size| format!("{}x{}", size.width, size.height))
            ),
            or_default(shape.steps),
            shape.images,
            shape.input_images,
            stats.finished,
            stats.errors,
            stats.aborted,
            stats.latency_p50,
            stats.latency_p90,
            stats.latency_max,
            stats.service_p50,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::io::BufReader;
    use std::time::Duration;

    use crate::diffusion::{
        DiffusionOutcome, DiffusionRecord, DiffusionReplay, DiffusionShape, EXEC_TIME_METRIC,
        ImageSize, QUEUE_WAIT_METRIC, Recorded, Service, StepModel, append_record, read_trace,
        step_units, summarize, write_meta, write_summary,
    };
    use crate::trace::{TraceKind, TraceMeta};

    const SQUARE_1024: ImageSize = ImageSize {
        width: 1024,
        height: 1024,
    };
    const SQUARE_512: ImageSize = ImageSize {
        width: 512,
        height: 512,
    };

    fn shape(size: Option<ImageSize>, steps: Option<u32>) -> DiffusionShape {
        DiffusionShape {
            size,
            steps,
            images: 1,
            input_images: 0,
        }
    }

    fn record(arrival_ms: f64, shape: DiffusionShape, latency_ms: f64) -> DiffusionRecord {
        DiffusionRecord {
            arrival_ms,
            shape,
            concurrency: 1,
            latency_ms,
            outcome: DiffusionOutcome::Finished,
            output_size: shape.size,
            stage_metrics: BTreeMap::new(),
        }
    }

    fn meta() -> TraceMeta {
        TraceMeta {
            kind: TraceKind::Diffusion,
            model: Some("test-model".to_string()),
            source: Some("tap".to_string()),
            ..TraceMeta::default()
        }
    }

    fn written(meta: &TraceMeta, records: &[DiffusionRecord]) -> Vec<u8> {
        let mut out = Vec::new();
        write_meta(&mut out, meta).expect("meta");
        for record in records {
            append_record(&mut out, record).expect("record");
        }
        out
    }

    #[test]
    fn record_serializes_flat_with_unset_fields_omitted() {
        let bare = DiffusionRecord {
            output_size: Some(SQUARE_1024),
            ..record(12.5, shape(None, None), 480.0)
        };
        assert_eq!(
            serde_json::to_value(&bare).expect("json"),
            serde_json::json!({
                "arrival_ms": 12.5,
                "images": 1,
                "input_images": 0,
                "concurrency": 1,
                "latency_ms": 480.0,
                "outcome": "finished",
                "output_size": {"width": 1024, "height": 1024},
            })
        );

        let failed = DiffusionRecord {
            outcome: DiffusionOutcome::Error { status: Some(400) },
            output_size: None,
            stage_metrics: BTreeMap::from([("scheduler_num_waiting_reqs".to_string(), 1.0)]),
            ..record(0.0, shape(Some(SQUARE_512), Some(4)), 3.0)
        };
        assert_eq!(
            serde_json::to_value(&failed).expect("json"),
            serde_json::json!({
                "arrival_ms": 0.0,
                "size": {"width": 512, "height": 512},
                "steps": 4,
                "images": 1,
                "input_images": 0,
                "concurrency": 1,
                "latency_ms": 3.0,
                "outcome": "error",
                "status": 400,
                "stage_metrics": {"scheduler_num_waiting_reqs": 1.0},
            })
        );
    }

    #[test]
    fn every_outcome_round_trips() {
        let records: Vec<DiffusionRecord> = [
            DiffusionOutcome::Finished,
            DiffusionOutcome::Error { status: Some(500) },
            DiffusionOutcome::Error { status: None },
            DiffusionOutcome::Aborted,
        ]
        .into_iter()
        .map(|outcome| DiffusionRecord {
            outcome,
            ..record(1.0, shape(Some(SQUARE_512), Some(4)), 2.0)
        })
        .collect();

        let bytes = written(&meta(), &records);
        let (read_meta, read_records) = read_trace(BufReader::new(bytes.as_slice())).expect("read");
        assert_eq!(read_meta, meta());
        assert_eq!(read_records, records);
    }

    #[test]
    fn reader_refuses_a_token_trace_and_a_file_without_meta() {
        let token_trace =
            b"{\"meta\":{\"model\":\"m\"}}\n{\"prompt_tokens\":1,\"output_tokens\":1,\"ttft_ms\":1.0}\n";
        let error = read_trace(BufReader::new(&token_trace[..])).expect_err("token trace");
        assert!(format!("{error:#}").contains("not a diffusion trace"));

        let no_meta = serde_json::to_vec(&record(0.0, shape(None, None), 1.0)).expect("json");
        assert!(read_trace(BufReader::new(no_meta.as_slice())).is_err());
        assert!(read_trace(BufReader::new(&b""[..])).is_err());
    }

    #[test]
    fn writer_refuses_a_meta_of_the_wrong_kind() {
        assert!(write_meta(&mut Vec::new(), &TraceMeta::default()).is_err());
    }

    #[test]
    fn service_time_excludes_the_stages_queue_wait() {
        let mut queued = record(0.0, shape(Some(SQUARE_512), Some(4)), 900.0);
        assert_eq!(queued.service_ms(), 900.0);
        queued
            .stage_metrics
            .insert(QUEUE_WAIT_METRIC.to_string(), 700.0);
        assert_eq!(
            queued.service(),
            Service {
                slot: Duration::from_millis(200),
                tail: Duration::ZERO,
            },
            "without the stage's exec time the whole service holds the slot"
        );
        queued
            .stage_metrics
            .insert(QUEUE_WAIT_METRIC.to_string(), 5000.0);
        assert_eq!(queued.service_ms(), 0.0);
    }

    #[test]
    fn stage_exec_time_splits_service_into_slot_and_tail() {
        let mut queued = record(0.0, shape(Some(SQUARE_512), Some(4)), 900.0);
        queued.stage_metrics = BTreeMap::from([
            (QUEUE_WAIT_METRIC.to_string(), 500.0),
            (EXEC_TIME_METRIC.to_string(), 840.0),
        ]);
        assert_eq!(
            queued.service(),
            Service {
                slot: Duration::from_millis(340),
                tail: Duration::from_millis(60),
            }
        );
        assert_eq!(queued.service_ms(), 400.0);

        queued
            .stage_metrics
            .insert(EXEC_TIME_METRIC.to_string(), 950.0);
        assert_eq!(
            queued.service().tail,
            Duration::ZERO,
            "an exec time past the tap's latency leaves no tail"
        );
    }

    #[test]
    fn step_units_scale_with_steps_images_and_pixels() {
        assert_eq!(step_units(SQUARE_1024, 50, 1), 50.0);
        assert_eq!(step_units(SQUARE_512, 20, 1), 5.0);
        assert_eq!(step_units(SQUARE_512, 20, 4), 20.0);
    }

    fn assert_model(model: StepModel, overhead_ms: f64, step_ms: f64) {
        let close =
            |duration: Duration, ms: f64| (duration.as_secs_f64() * 1000.0 - ms).abs() < 1e-6;
        assert!(
            close(model.overhead, overhead_ms) && close(model.step_time, step_ms),
            "expected {overhead_ms} ms + {step_ms} ms/step, got {model:?}"
        );
    }

    #[test]
    fn fit_recovers_overhead_and_step_time_from_exact_points() {
        let points: Vec<(f64, f64)> = [1.0, 4.0, 16.0, 50.0]
            .into_iter()
            .map(|units| (units, 300.0 + 40.0 * units))
            .collect();
        assert_model(StepModel::fit(&points).expect("fit"), 300.0, 40.0);
    }

    #[test]
    fn fit_through_one_step_count_is_a_line_through_the_origin() {
        let points = [(4.0, 190.0), (4.0, 210.0)];
        assert_model(StepModel::fit(&points).expect("fit"), 0.0, 50.0);
    }

    #[test]
    fn fit_with_a_negative_intercept_is_a_line_through_the_origin() {
        let points = [(1.0, 10.0), (2.0, 40.0)];
        assert_model(StepModel::fit(&points).expect("fit"), 0.0, 50.0 / 3.0);
    }

    #[test]
    fn fit_with_a_negative_slope_is_still_non_negative() {
        let points = [(1.0, 100.0), (2.0, 80.0), (3.0, 60.0)];
        let fitted = StepModel::fit(&points).expect("fit");
        assert_model(fitted, 0.0, 40.0);
    }

    #[test]
    fn fit_of_nothing_is_none() {
        assert_eq!(StepModel::fit(&[]), None);
    }

    #[test]
    fn replay_cycles_a_shapes_recorded_times_in_arrival_order() {
        let small = shape(Some(SQUARE_512), Some(4));
        let large = shape(Some(SQUARE_1024), Some(4));
        let records = [
            record(30.0, small, 130.0),
            record(10.0, small, 110.0),
            record(20.0, large, 400.0),
            record(40.0, small, 120.0),
        ];
        let mut replay = DiffusionReplay::from_records(&records).expect("replay");
        assert_eq!(replay.shapes(), 2);

        let small_times: Vec<u128> = (0..4)
            .map(|_| {
                replay
                    .next_recorded(&small)
                    .expect("recorded shape")
                    .service
                    .total()
                    .as_millis()
            })
            .collect();
        assert_eq!(small_times, [110, 130, 120, 110]);
        assert_eq!(
            replay.next_recorded(&large),
            Some(Recorded {
                service: Service {
                    slot: Duration::from_millis(400),
                    tail: Duration::ZERO,
                },
                output_size: Some(SQUARE_1024),
            })
        );
    }

    #[test]
    fn replay_has_no_recorded_time_for_an_unseen_shape() {
        let records = [record(0.0, shape(Some(SQUARE_512), Some(4)), 100.0)];
        let mut replay = DiffusionReplay::from_records(&records).expect("replay");
        assert_eq!(
            replay.next_recorded(&shape(Some(SQUARE_512), Some(8))),
            None
        );
        let edit = DiffusionShape {
            input_images: 1,
            ..shape(Some(SQUARE_512), Some(4))
        };
        assert_eq!(replay.next_recorded(&edit), None);
    }

    #[test]
    fn replay_fits_the_step_model_for_unseen_shapes() {
        let records: Vec<DiffusionRecord> = [
            (SQUARE_512, 4),
            (SQUARE_512, 8),
            (SQUARE_1024, 4),
            (SQUARE_1024, 8),
        ]
        .into_iter()
        .enumerate()
        .map(|(index, (size, steps))| {
            let units = step_units(size, steps, 1);
            record(
                index as f64,
                shape(Some(size), Some(steps)),
                80.0 + 25.0 * units,
            )
        })
        .collect();
        let replay = DiffusionReplay::from_records(&records).expect("replay");
        assert_model(replay.fitted(), 80.0, 25.0);
    }

    #[test]
    fn replay_fit_uses_the_returned_size_when_the_request_set_none() {
        let records = [
            DiffusionRecord {
                output_size: Some(SQUARE_1024),
                ..record(0.0, shape(None, Some(4)), 100.0 + 10.0 * 4.0)
            },
            DiffusionRecord {
                output_size: Some(SQUARE_1024),
                ..record(1.0, shape(None, Some(8)), 100.0 + 10.0 * 8.0)
            },
        ];
        let replay = DiffusionReplay::from_records(&records).expect("replay");
        assert_model(replay.fitted(), 100.0, 10.0);
    }

    #[test]
    fn replay_fit_takes_each_shapes_median_so_a_cold_first_request_does_not_skew_it() {
        let mut records = Vec::new();
        for (size, steps) in [
            (SQUARE_512, 4),
            (SQUARE_512, 8),
            (SQUARE_1024, 4),
            (SQUARE_1024, 8),
        ] {
            let warm = 80.0 + 25.0 * step_units(size, steps, 1);
            for latency_ms in [16_000.0, warm, warm] {
                records.push(record(
                    records.len() as f64,
                    shape(Some(size), Some(steps)),
                    latency_ms,
                ));
            }
        }
        let mut replay = DiffusionReplay::from_records(&records).expect("replay");
        assert_model(replay.fitted(), 80.0, 25.0);
        assert_eq!(
            replay
                .next_recorded(&shape(Some(SQUARE_512), Some(4)))
                .expect("recorded")
                .service
                .total(),
            Duration::from_secs(16),
            "the cold request is still replayed where it was recorded"
        );
    }

    #[test]
    fn replay_recorded_times_keep_the_slot_and_tail_split() {
        let small = shape(Some(SQUARE_512), Some(4));
        let mut queued = record(0.0, small, 900.0);
        queued.stage_metrics = BTreeMap::from([
            (QUEUE_WAIT_METRIC.to_string(), 500.0),
            (EXEC_TIME_METRIC.to_string(), 840.0),
        ]);
        let mut replay = DiffusionReplay::from_records(&[queued]).expect("replay");
        assert_eq!(
            replay.next_recorded(&small).expect("recorded").service,
            Service {
                slot: Duration::from_millis(340),
                tail: Duration::from_millis(60),
            }
        );
    }

    #[test]
    fn replay_without_known_steps_falls_back_to_the_median_service_time() {
        let records = [
            record(0.0, shape(Some(SQUARE_512), None), 100.0),
            record(1.0, shape(Some(SQUARE_1024), None), 300.0),
        ];
        let replay = DiffusionReplay::from_records(&records).expect("replay");
        assert_model(replay.fitted(), 200.0, 0.0);
    }

    #[test]
    fn replay_ignores_errors_and_aborts_and_needs_a_finished_request() {
        let small = shape(Some(SQUARE_512), Some(4));
        let failed = DiffusionRecord {
            outcome: DiffusionOutcome::Error { status: Some(500) },
            ..record(0.0, small, 5.0)
        };
        let aborted = DiffusionRecord {
            outcome: DiffusionOutcome::Aborted,
            ..record(1.0, small, 7.0)
        };
        assert!(DiffusionReplay::from_records(&[failed.clone(), aborted.clone()]).is_err());

        let mut replay =
            DiffusionReplay::from_records(&[failed, aborted, record(2.0, small, 100.0)])
                .expect("replay");
        for _ in 0..3 {
            assert_eq!(
                replay
                    .next_recorded(&small)
                    .expect("recorded")
                    .service
                    .total(),
                Duration::from_millis(100)
            );
        }
    }

    #[test]
    fn summary_groups_by_shape_and_counts_outcomes() {
        let small = shape(Some(SQUARE_512), Some(4));
        let large = shape(Some(SQUARE_1024), Some(4));
        let records = [
            record(0.0, large, 400.0),
            record(1.0, small, 100.0),
            record(2.0, small, 300.0),
            record(3.0, small, 200.0),
            DiffusionRecord {
                outcome: DiffusionOutcome::Error { status: Some(400) },
                ..record(4.0, small, 1.0)
            },
            DiffusionRecord {
                outcome: DiffusionOutcome::Aborted,
                ..record(5.0, large, 2.0)
            },
        ];
        let stats = summarize(&records);
        assert_eq!(stats.len(), 2);
        assert_eq!(stats[0].shape, small, "smaller images sort first");
        assert_eq!(
            (stats[0].finished, stats[0].errors, stats[0].aborted),
            (3, 1, 0)
        );
        assert_eq!(stats[0].latency_p50, 200.0);
        assert_eq!(stats[0].latency_max, 300.0);
        assert_eq!(
            (stats[1].finished, stats[1].errors, stats[1].aborted),
            (1, 0, 1)
        );

        let mut text = Vec::new();
        write_summary(&mut text, &meta(), &records).expect("summary");
        let text = String::from_utf8(text).expect("utf8");
        assert!(text.contains("kind:  diffusion"));
        assert!(text.contains("model: test-model"));
        assert!(text.contains("512x512"));
        assert!(text.contains("1024x1024"));
    }
}
