//! Synthetic trace generation utilities for E2E engine testing.
//!
//! Provides functions to generate various types of synthetic traces programmatically,
//! covering different schema variants, edge cases, and workload patterns. All
//! generation is deterministic (seeded RNG) for reproducible tests.

use rand::rngs::StdRng;
use rand::{Rng as _, SeedableRng};

use sim_trace::trace::{
    ItlContext, ItlSummary, TraceFinishReason, TraceMeta, TraceRecord, prompt_block_hashes,
};

/// Check if fast mode is enabled via environment variable (for CI).
fn is_fast_mode() -> bool {
    std::env::var("SYNTHETIC_E2E_FAST_MODE").is_ok()
}

/// Enrich records with `arrival_ms` and `output_token_ids` so they work with
/// `--replay-tokens`. The replay path filters out records missing these fields.
fn make_replayable(records: &mut [TraceRecord], seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut arrival = 0.0_f64;
    for r in records.iter_mut() {
        if r.arrival_ms.is_none() {
            r.arrival_ms = Some(arrival);
            arrival += rng.random_range(50.0..500.0);
        }
        if r.output_token_ids.is_none() {
            r.output_token_ids = Some(
                (0..r.output_tokens)
                    .map(|_| rng.random_range(100..50000_u32))
                    .collect(),
            );
        }
    }
}

/// Generate a basic synthetic trace with varying prompt/output lengths.
/// Uses realistic lognormal distributions for timing (via calibrate::gen_demo).
pub fn generate_basic_trace(num_records: usize, seed: u64) -> (TraceMeta, Vec<TraceRecord>) {
    let (meta, mut records) = if is_fast_mode() {
        vllm_vcr::calibrate::gen_demo_fast(num_records, seed)
    } else {
        vllm_vcr::calibrate::gen_demo(num_records, seed)
    };
    make_replayable(&mut records, seed.wrapping_add(1));
    (meta, records)
}

/// Generate a trace with batch context (itl_ctx) data.
/// Simulates realistic batch interference with varying num_running and prefill_tokens.
pub fn generate_batch_context_trace(
    num_records: usize,
    seed: u64,
) -> (TraceMeta, Vec<TraceRecord>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let meta = TraceMeta {
        model: Some("synthetic-batch-context".to_string()),
        source: Some("synthetic-e2e".to_string()),
        max_num_seqs: Some(8),
        ..Default::default()
    };

    let mut records = Vec::with_capacity(num_records);
    for _ in 0..num_records {
        let prompt_tokens = rng.random_range(10..200);
        let output_tokens = rng.random_range(5..50);
        let ttft_ms = if is_fast_mode() {
            rng.random_range(15.0..40.0)
        } else {
            rng.random_range(50.0..200.0)
        };

        // Generate ITL data with batch context
        let mut itl_ms = Vec::with_capacity(output_tokens - 1);
        let mut num_running = Vec::with_capacity(output_tokens - 1);
        let mut prefill_tokens = Vec::with_capacity(output_tokens - 1);

        for i in 0..(output_tokens - 1) {
            let base_itl = if is_fast_mode() { 8.0 } else { 15.0 };
            itl_ms.push(rng.random_range(base_itl..base_itl * 2.0));

            // Varying batch size
            num_running.push(rng.random_range(1..=8));

            // Occasional prefill interference
            prefill_tokens.push(if i % 3 == 0 {
                rng.random_range(0..800)
            } else {
                0
            });
        }

        records.push(TraceRecord {
            prompt_tokens,
            output_tokens,
            ttft_ms,
            itl_ms: Some(itl_ms),
            itl_ctx: Some(ItlContext {
                num_running,
                prefill_tokens,
            }),
            concurrency: rng.random_range(1..=4),
            finish_reason: Some(TraceFinishReason::Stop),
            ..Default::default()
        });
    }

    make_replayable(&mut records, seed.wrapping_add(1));
    (meta, records)
}

/// Generate a speculative decoding trace with multi-token chunks (itl_tokens).
/// Simulates EAGLE-style K=4 speculative decoding.
pub fn generate_speculative_trace(num_records: usize, seed: u64) -> (TraceMeta, Vec<TraceRecord>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let meta = TraceMeta {
        model: Some("synthetic-speculative".to_string()),
        source: Some("synthetic-e2e".to_string()),
        max_num_seqs: Some(4),
        ..Default::default()
    };

    let mut records = Vec::with_capacity(num_records);
    for _ in 0..num_records {
        let prompt_tokens = rng.random_range(50..256);
        // Total output: first chunk + K chunks of varying size
        let num_chunks = rng.random_range(3..8);
        let first_chunk_size = rng.random_range(1..=4_u32);

        let mut itl_tokens = Vec::with_capacity(num_chunks);
        let mut total_output = first_chunk_size as usize;

        // Each subsequent chunk (after TTFT) has K tokens (with some variation)
        for _ in 0..num_chunks {
            let chunk_size = rng.random_range(2..=5_u32); // K=4 ± 1
            itl_tokens.push(chunk_size);
            total_output += chunk_size as usize;
        }

        let ttft_ms = if is_fast_mode() {
            rng.random_range(20.0..50.0)
        } else {
            rng.random_range(80.0..300.0)
        };

        // ITL gaps (one per chunk after the first)
        let base_itl = if is_fast_mode() { 5.0 } else { 10.0 };
        let itl_ms: Vec<f64> = (0..num_chunks)
            .map(|_| rng.random_range(base_itl..base_itl * 2.0))
            .collect();

        records.push(TraceRecord {
            prompt_tokens,
            output_tokens: total_output,
            ttft_ms,
            itl_ms: Some(itl_ms),
            itl_tokens: Some(itl_tokens),
            concurrency: 1,
            finish_reason: Some(TraceFinishReason::Length),
            ..Default::default()
        });
    }

    make_replayable(&mut records, seed.wrapping_add(1));
    (meta, records)
}

/// Generate a diffusion model trace with block outputs (8-token bursts).
pub fn generate_diffusion_trace(num_records: usize, seed: u64) -> (TraceMeta, Vec<TraceRecord>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let meta = TraceMeta {
        model: Some("synthetic-diffusion".to_string()),
        source: Some("synthetic-e2e".to_string()),
        max_num_seqs: Some(2),
        ..Default::default()
    };

    let mut records = Vec::with_capacity(num_records);
    const BLOCK_SIZE: u32 = 8;

    for _ in 0..num_records {
        let prompt_tokens = rng.random_range(20..128);
        let num_blocks = rng.random_range(2..6);
        let first_block = BLOCK_SIZE;
        let total_output = first_block as usize + (num_blocks * BLOCK_SIZE) as usize;

        let ttft_ms = if is_fast_mode() {
            rng.random_range(30.0..80.0)
        } else {
            rng.random_range(100.0..400.0)
        };

        // Each block takes similar time (diffusion steps)
        let base_block_time = if is_fast_mode() { 20.0 } else { 50.0 };
        let itl_ms: Vec<f64> = (0..num_blocks)
            .map(|_| rng.random_range(base_block_time..base_block_time * 1.5))
            .collect();

        let itl_tokens: Vec<u32> = vec![BLOCK_SIZE; num_blocks as usize];

        records.push(TraceRecord {
            prompt_tokens,
            output_tokens: total_output,
            ttft_ms,
            itl_ms: Some(itl_ms),
            itl_tokens: Some(itl_tokens),
            concurrency: 1,
            finish_reason: Some(TraceFinishReason::Stop),
            ..Default::default()
        });
    }

    make_replayable(&mut records, seed.wrapping_add(1));
    (meta, records)
}

/// Generate a trace with various edge cases.
pub fn generate_edge_cases_trace(seed: u64) -> (TraceMeta, Vec<TraceRecord>) {
    let meta = TraceMeta {
        model: Some("synthetic-edge-cases".to_string()),
        source: Some("synthetic-e2e".to_string()),
        ..Default::default()
    };

    let mut records = Vec::new();

    // Edge case 1: Single-token output (no ITL data)
    records.push(TraceRecord {
        prompt_tokens: 10,
        output_tokens: 1,
        ttft_ms: 25.0,
        itl_ms: None,
        finish_reason: Some(TraceFinishReason::Stop),
        ..Default::default()
    });

    // Edge case 2: Large output (1000+ tokens)
    let large_output = 1200;
    records.push(TraceRecord {
        prompt_tokens: 50,
        output_tokens: large_output,
        ttft_ms: 100.0,
        itl_ms: Some(vec![10.0; large_output - 1]),
        finish_reason: Some(TraceFinishReason::Length),
        ..Default::default()
    });

    // Edge case 3: All finish reasons
    for (i, reason) in [
        TraceFinishReason::Stop,
        TraceFinishReason::Length,
        TraceFinishReason::Abort,
        TraceFinishReason::Error,
        TraceFinishReason::Repetition,
    ]
    .iter()
    .enumerate()
    {
        records.push(TraceRecord {
            prompt_tokens: 20 + i * 10,
            output_tokens: 5,
            ttft_ms: 30.0,
            itl_ms: Some(vec![12.0; 4]),
            finish_reason: Some(*reason),
            ..Default::default()
        });
    }

    // Edge case 4: High cache hit rate
    records.push(TraceRecord {
        prompt_tokens: 100,
        cached_tokens: 95,
        output_tokens: 10,
        ttft_ms: 15.0, // Faster due to cache
        itl_ms: Some(vec![8.0; 9]),
        finish_reason: Some(TraceFinishReason::Stop),
        ..Default::default()
    });

    // Edge case 5: Zero cached tokens (cold start)
    records.push(TraceRecord {
        prompt_tokens: 200,
        cached_tokens: 0,
        output_tokens: 20,
        ttft_ms: 150.0,
        itl_ms: Some(vec![12.0; 19]),
        finish_reason: Some(TraceFinishReason::Stop),
        ..Default::default()
    });

    // Edge case 6: Minimal record (only required fields, no optional data)
    records.push(TraceRecord {
        prompt_tokens: 15,
        output_tokens: 3,
        ttft_ms: 20.0,
        itl_ms: Some(vec![10.0; 2]),
        // No arrival_ms, no itl_ctx, no block_hashes, no output_token_ids
        ..Default::default()
    });

    // Edge case 7: ITL summary instead of full array
    records.push(TraceRecord {
        prompt_tokens: 50,
        output_tokens: 100,
        ttft_ms: 60.0,
        itl_ms: None,
        itl_summary: Some(ItlSummary {
            mean_ms: 12.5,
            count: 99,
        }),
        finish_reason: Some(TraceFinishReason::Stop),
        ..Default::default()
    });

    make_replayable(&mut records, seed.wrapping_add(1));
    (meta, records)
}

/// Generate a trace with prefix sharing (multi-turn conversation).
/// Uses block hashes to simulate cache hits across turns.
#[allow(dead_code)]
pub fn generate_prefix_sharing_trace(
    num_turns: usize,
    block_size: usize,
    seed: u64,
) -> (TraceMeta, Vec<TraceRecord>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let meta = TraceMeta {
        model: Some("synthetic-prefix-sharing".to_string()),
        source: Some("synthetic-e2e".to_string()),
        block_size: Some(block_size),
        ..Default::default()
    };

    let mut records = Vec::with_capacity(num_turns);
    let mut conversation_tokens = Vec::new();

    for turn in 0..num_turns {
        // Each turn extends the conversation
        let turn_tokens: Vec<u32> = (0..rng.random_range(20..50))
            .map(|_| rng.random_range(100..1000))
            .collect();

        conversation_tokens.extend(&turn_tokens);

        // Compute block hashes for the full conversation so far
        let block_hashes = prompt_block_hashes(&conversation_tokens, block_size);

        let output_tokens = rng.random_range(5..15);
        let ttft_ms = if is_fast_mode() {
            rng.random_range(10.0..30.0)
        } else {
            rng.random_range(30.0..100.0)
        };

        let itl_ms: Vec<f64> = (0..output_tokens - 1)
            .map(|_| {
                let base = if is_fast_mode() { 8.0 } else { 12.0 };
                rng.random_range(base..base * 1.5)
            })
            .collect();

        // Generate output tokens for this turn
        let output_token_ids: Vec<u32> = (0..output_tokens)
            .map(|_| rng.random_range(100..1000))
            .collect();

        records.push(TraceRecord {
            prompt_tokens: conversation_tokens.len(),
            cached_tokens: if turn > 0 {
                // Simulate cache hits from previous turns
                conversation_tokens.len() - turn_tokens.len()
            } else {
                0
            },
            output_tokens,
            ttft_ms,
            itl_ms: Some(itl_ms),
            block_hashes,
            output_token_ids: Some(output_token_ids.clone()),
            finish_reason: Some(TraceFinishReason::Stop),
            ..Default::default()
        });

        // Add this turn's output to the conversation
        conversation_tokens.extend(&output_token_ids);
    }

    (meta, records)
}

/// Generate a trace with explicit arrival times for open-loop replay.
#[allow(dead_code)]
pub fn generate_arrival_schedule_trace(
    num_records: usize,
    seed: u64,
) -> (TraceMeta, Vec<TraceRecord>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let meta = TraceMeta {
        model: Some("synthetic-arrival-schedule".to_string()),
        source: Some("synthetic-e2e".to_string()),
        ..Default::default()
    };

    let mut records = Vec::with_capacity(num_records);
    let mut current_time = 0.0;

    for _ in 0..num_records {
        let prompt_tokens = rng.random_range(20..150);
        let output_tokens = rng.random_range(5..30);
        let ttft_ms = if is_fast_mode() {
            rng.random_range(15.0..40.0)
        } else {
            rng.random_range(50.0..150.0)
        };

        let itl_ms: Vec<f64> = (0..output_tokens - 1)
            .map(|_| {
                let base = if is_fast_mode() { 8.0 } else { 12.0 };
                rng.random_range(base..base * 1.5)
            })
            .collect();

        records.push(TraceRecord {
            prompt_tokens,
            output_tokens,
            ttft_ms,
            itl_ms: Some(itl_ms),
            arrival_ms: Some(current_time),
            finish_reason: Some(TraceFinishReason::Stop),
            ..Default::default()
        });

        // Next arrival with some inter-arrival time
        current_time += rng.random_range(50.0..500.0);
    }

    (meta, records)
}

/// Generate a trace with mixed concurrency levels.
pub fn generate_mixed_concurrency_trace(
    num_records: usize,
    seed: u64,
) -> (TraceMeta, Vec<TraceRecord>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let meta = TraceMeta {
        model: Some("synthetic-mixed-concurrency".to_string()),
        source: Some("synthetic-e2e".to_string()),
        max_num_seqs: Some(16),
        ..Default::default()
    };

    let concurrency_levels = [1, 4, 8, 16];
    let mut records = Vec::with_capacity(num_records);

    for i in 0..num_records {
        let concurrency = concurrency_levels[i % concurrency_levels.len()];
        let prompt_tokens = rng.random_range(50..300);
        let output_tokens = rng.random_range(10..50);
        let ttft_ms = if is_fast_mode() {
            rng.random_range(15.0..60.0)
        } else {
            rng.random_range(50.0..250.0)
        };

        let itl_ms: Vec<f64> = (0..output_tokens - 1)
            .map(|_| {
                let base = if is_fast_mode() { 8.0 } else { 12.0 };
                // Higher concurrency = slightly higher latency
                rng.random_range(base..base * (1.0 + concurrency as f64 * 0.1))
            })
            .collect();

        records.push(TraceRecord {
            prompt_tokens,
            output_tokens,
            ttft_ms,
            itl_ms: Some(itl_ms),
            concurrency,
            finish_reason: Some(TraceFinishReason::Stop),
            ..Default::default()
        });
    }

    make_replayable(&mut records, seed.wrapping_add(1));
    (meta, records)
}
