//! Writes the engine-step trace of a simulated run (`--step-trace-out`), in the
//! format the vLLM recorder in `deploy/trace-capture/vcr_step_recorder.py` writes
//! for a real engine.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write as _};
use std::path::Path;

use anyhow::{Context as _, Result};
use sim_trace::steps::{
    KvDirection, KvOutcome, KvTransferRecord, StepRecord, append_kv_transfer, append_step,
    append_step_meta,
};
use sim_trace::trace::TraceMeta;
use tokio::time::Instant;
use tracing::warn;

use crate::Opt;
use crate::rust_core::runner::StepResult;

struct PendingLoad {
    start_step: u64,
    started: Instant,
    blocks: usize,
}

pub(crate) struct StepRecorder {
    out: Option<BufWriter<File>>,
    capture_start: Instant,
    time_scale: f64,
    /// Steps that scheduled tokens so far.
    step: u64,
    loads: HashMap<String, PendingLoad>,
}

impl StepRecorder {
    pub(crate) fn create(path: &Path, opt: &Opt) -> Result<Self> {
        let file = File::create(path)
            .with_context(|| format!("creating step trace {}", path.display()))?;
        let mut out = BufWriter::new(file);
        let meta = TraceMeta {
            source: Some("vllm-vcr play".to_string()),
            max_num_seqs: Some(opt.max_num_seqs),
            block_size: Some(opt.tokens_per_block),
            ..TraceMeta::default()
        };
        append_step_meta(&mut out, &meta)?;
        out.flush()?;
        Ok(StepRecorder {
            out: Some(out),
            capture_start: Instant::now(),
            time_scale: if opt.time_scale > 0.0 {
                opt.time_scale
            } else {
                1.0
            },
            step: 0,
            loads: HashMap::new(),
        })
    }

    fn ms(&self, from: Instant, to: Instant) -> f64 {
        to.saturating_duration_since(from).as_secs_f64() * 1000.0 * self.time_scale
    }

    /// Record the step that started at `started` and the KV loads it reported.
    pub(crate) fn record(&mut self, started: Instant, result: &StepResult) {
        let executed = result.shape.total_tokens() > 0;
        if executed {
            self.step += 1;
        }
        let mut transfers = Vec::new();
        if let Some(kv) = &result.output.kv_connector_output {
            for request_id in kv.finished_recving.iter().flatten() {
                let Some(load) = self.loads.remove(request_id) else {
                    continue;
                };
                transfers.push(KvTransferRecord {
                    direction: KvDirection::Recv,
                    blocks: load.blocks as u32,
                    start_step: load.start_step,
                    end_step: self.step,
                    wall_ms: self.ms(load.started, started),
                    outcome: if kv.failed_recving.contains(request_id) {
                        KvOutcome::Failed
                    } else {
                        KvOutcome::Ok
                    },
                });
            }
        }
        for (request_id, blocks) in &result.loads_started {
            self.loads.insert(
                request_id.clone(),
                PendingLoad {
                    start_step: self.step,
                    started,
                    blocks: *blocks,
                },
            );
        }

        let step_ms = result.duration.as_secs_f64() * 1000.0 * self.time_scale;
        let step = executed.then(|| StepRecord {
            step: self.step,
            ts_ms: self.ms(self.capture_start, started),
            exec_ms: step_ms,
            step_ms,
            sched_ms: 0.0,
            update_ms: 0.0,
            shape: result.shape.clone(),
            sampled: result
                .output
                .sampled_token_ids
                .iter()
                .map(Vec::len)
                .sum::<usize>() as u32,
            accepted: result.accepted as u32,
        });
        if let Err(error) = self.write(&transfers, step.as_ref()) {
            warn!(%error, "writing the step trace failed; recording stops");
            self.out = None;
        }
    }

    fn write(&mut self, transfers: &[KvTransferRecord], step: Option<&StepRecord>) -> Result<()> {
        let Some(out) = &mut self.out else {
            return Ok(());
        };
        for transfer in transfers {
            append_kv_transfer(out, transfer)?;
        }
        if let Some(step) = step {
            append_step(out, step)?;
        }
        out.flush()?;
        Ok(())
    }
}
