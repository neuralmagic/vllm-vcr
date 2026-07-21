//! Shared test utilities for engine E2E integration tests.
//!
//! Provides RAII cleanup guards, IPC endpoint generation, temporary trace file
//! management, and common assertion helpers used across the test suite.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use futures::Stream;
use tokio_util::sync::CancellationToken;
use vllm_engine_core_client::protocol::EngineCoreOutput;

use sim_trace::trace::{TraceMeta, TraceRecord, TraceWriter, write_trace};

/// RAII guard that cancels the simulator task on drop, even if the test panics.
/// Also holds the sim task's JoinHandle so callers can detect early exits.
pub struct SimGuard {
    pub token: CancellationToken,
    pub sim_handle: Option<tokio::task::JoinHandle<anyhow::Result<()>>>,
}

impl SimGuard {
    /// Check whether the simulator task exited early (error or panic).
    /// Call this after connect succeeds but before issuing requests.
    pub fn assert_sim_running(&mut self) {
        if let Some(ref handle) = self.sim_handle {
            if handle.is_finished() {
                let handle = self.sim_handle.take().unwrap();
                match futures::executor::block_on(handle) {
                    Ok(Ok(())) => panic!("simulator exited unexpectedly (no error)"),
                    Ok(Err(e)) => panic!("simulator failed to start: {e:#}"),
                    Err(e) => panic!("simulator task panicked: {e}"),
                }
            }
        }
    }
}

impl Drop for SimGuard {
    fn drop(&mut self) {
        self.token.cancel();
    }
}

/// Generate a unique IPC endpoint for a test to prevent cross-test interference.
pub fn unique_ipc_endpoint(test_name: &str) -> String {
    format!(
        "ipc:///tmp/inf-sim-synthetic-{}-{}.ipc",
        std::process::id(),
        test_name
    )
}

/// RAII wrapper for a temporary trace file that cleans up on drop.
pub struct TempTraceFile {
    path: PathBuf,
}

impl TempTraceFile {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempTraceFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Create a temporary trace file and write the given meta + records to it.
/// Returns a TempTraceFile that cleans up the file on drop.
pub fn create_temp_trace(
    name: &str,
    meta: &TraceMeta,
    records: &[TraceRecord],
) -> Result<TempTraceFile> {
    let path = PathBuf::from(format!(
        "/tmp/inf-sim-synthetic-trace-{}-{}.jsonl",
        std::process::id(),
        name
    ));

    let mut writer = TraceWriter::create(&path)
        .with_context(|| format!("failed to create trace file at {:?}", path))?;

    write_trace(&mut writer, meta, records).context("failed to write trace")?;

    writer.finish().context("failed to finalize trace file")?;

    Ok(TempTraceFile { path })
}

/// Create a temporary gzip-compressed trace file.
pub fn create_temp_trace_gz(
    name: &str,
    meta: &TraceMeta,
    records: &[TraceRecord],
) -> Result<TempTraceFile> {
    let path = PathBuf::from(format!(
        "/tmp/inf-sim-synthetic-trace-{}-{}.jsonl.gz",
        std::process::id(),
        name
    ));

    let mut writer = TraceWriter::create(&path)
        .with_context(|| format!("failed to create gzipped trace file at {:?}", path))?;

    write_trace(&mut writer, meta, records).context("failed to write trace")?;

    writer.finish().context("failed to finalize trace file")?;

    Ok(TempTraceFile { path })
}

/// Collect all tokens from a stream of engine outputs.
#[allow(dead_code)]
pub async fn collect_tokens<S, E>(mut stream: S) -> Result<Vec<u32>>
where
    S: Stream<Item = Result<EngineCoreOutput, E>> + Unpin,
    E: std::error::Error + Send + Sync + 'static,
{
    use futures::StreamExt;

    let mut tokens = Vec::new();
    while let Some(result) = stream.next().await {
        let output = result.map_err(|e| anyhow::anyhow!("stream error: {}", e))?;
        tokens.extend(output.new_token_ids);
    }
    Ok(tokens)
}

/// Validate that the total number of tokens in outputs matches the expected count.
#[allow(dead_code)]
pub fn assert_token_count(outputs: &[EngineCoreOutput], expected: usize) {
    let total: usize = outputs.iter().map(|o| o.new_token_ids.len()).sum();
    assert_eq!(
        total, expected,
        "expected {} tokens total, got {}",
        expected, total
    );
}

/// Validate that the last output has a finish reason.
#[allow(dead_code)]
pub fn assert_has_finish_reason(outputs: &[EngineCoreOutput]) {
    assert!(
        !outputs.is_empty(),
        "outputs empty, cannot check finish reason"
    );
    let last = outputs.last().unwrap();
    assert!(
        last.finish_reason.is_some(),
        "last output should have a finish reason, got None"
    );
}
