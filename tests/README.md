# vLLM VCR Test Suite

This directory contains the integration test suite for the vLLM VCR engine simulator.

## Test Categories

### 1. Engine Core E2E Tests (`engine_core_e2e.rs`)
Full-stack integration tests using real ZMQ transport and `EngineCoreClient`. Tests core engine functionality:
- Token streaming and finish reasons
- Request abort and graceful shutdown
- Prefix cache reset
- LoRA lifecycle (when supported)
- Prefill/Decode (P/D) handoff with KV transfer

### 2. Synthetic E2E Tests (`engine_synthetic_e2e.rs`) ✨ NEW
Comprehensive E2E tests using **programmatically generated synthetic traces**. Covers:

**Trace Schema Variants:**
- Basic traces (simple prompt/output pairs)
- Batch context traces (`itl_ctx` with interference patterns)
- Speculative decoding traces (`itl_tokens` for multi-token chunks)
- Diffusion model traces (block outputs)

**Edge Cases:**
- Single-token outputs
- Large outputs (1000+ tokens)
- All finish reasons (Stop, Length, Abort, Error, Repetition)
- High/low cache hit rates
- Missing optional fields

**Replay Modes:**
- Token replay (`--replay-tokens`)
- Latency replay (`--latency-trace`)
- Prefix matching (`--replay-match prefix`)
- Compressed traces (`.jsonl.gz`)

**Workload Patterns:**
- Mixed concurrency levels (1, 4, 8, 16)
- Varied prompt/output lengths
- Arrival schedules for open-loop replay
- Prefix sharing (multi-turn conversations)

### 3. Real Trace Replay Tests
Tests using actual GPU captures:
- `real_trace_replay.rs` - Byte-identical replay of real traces
- `spec_replay_fidelity.rs` - Speculative decoding fidelity
- `diffusion_replay_demo.rs` - Diffusion model replay
- `gemma4_replay_demo.rs` - Multimodal (vision) trace replay
- `closed_loop_prefix_replay.rs` - Prefix cache matching

### 4. Trace Validation Tests (`trace_validation_ci.rs`)
Structural validation of synthetic trace fixtures in `fixtures/synthetic/`:
- Schema conformance
- Array length alignment (`itl_ms`, `itl_ctx`, `itl_tokens`)
- Required vs optional field handling

### 5. Calibration Tests (`calibrate.rs`)
Timing model validation:
- Lognormal distribution trace generation
- Quantile fidelity (p50, p90, p99)
- Trace vs knob-based latency models
- Open-loop arrival schedule replay

### 6. Conformance Tests (`conformance.rs`)
Golden testing against real vLLM captures from S3:
- SHA256 integrity validation
- vLLM version line compatibility
- Config hash provenance
- Protocol schema conformance

### 7. Recording Pipeline Tests
- `tap_e2e.rs` - Full recording tap integration
- `kv_events_pubsub.rs` - KV cache event publishing

### 8. Data Plane Tests
- `nixl_loopback.rs` - NIXL KV transfer over loopback (requires `nixl` feature)

## Running Tests

### Run all tests
```bash
cargo test --workspace
```

### Run specific test suite
```bash
# Synthetic E2E tests
cargo test --test engine_synthetic_e2e

# Engine core E2E tests
cargo test --test engine_core_e2e

# Trace validation
cargo test --test trace_validation_ci

# Calibration
cargo test --test calibrate

# Conformance (requires AWS credentials)
cargo test --test conformance
```

### Run with output
```bash
cargo test --test engine_synthetic_e2e -- --nocapture
```

### Run specific test
```bash
cargo test --test engine_synthetic_e2e test_basic_trace_token_replay
```

### Parallel execution control
```bash
# Limit parallel test threads
cargo test --test engine_synthetic_e2e -- --test-threads=4
```

## Synthetic vs Real Traces

### Synthetic Traces (`engine_synthetic_e2e.rs`)
**Purpose:** Fast, deterministic, comprehensive coverage of engine behavior

**Characteristics:**
- ✓ Generated programmatically with seeded RNG
- ✓ No GPU or external dependencies required
- ✓ Fast execution (<30 seconds for full suite)
- ✓ Covers edge cases and schema variants systematically
- ✓ CI-friendly (deterministic, parallel-safe)

**When to use:**
- Testing engine logic and replay modes
- Validating schema support (batch context, speculative, diffusion)
- Edge case coverage (single tokens, large outputs, all finish reasons)
- Regression testing in CI

### Real Traces (`real_trace_replay.rs`, `conformance.rs`, etc.)
**Purpose:** Validate fidelity against actual vLLM GPU captures

**Characteristics:**
- ✓ Byte-identical replay of real workloads
- ✓ Validates production behavior
- ✓ Tests against multiple vLLM versions
- ✗ Requires trace files (fixtures or S3)
- ✗ Slower (larger traces, network I/O for S3)

**When to use:**
- Proving byte-identical replay of real captures
- Multi-version vLLM compatibility (conformance suite)
- Fidelity validation for production deployments

## Test Utilities

### `test_helpers.rs`
Shared utilities for engine tests:
- `SimGuard` - RAII cleanup for simulator tasks
- `unique_ipc_endpoint()` - Per-test IPC endpoint generation
- `create_temp_trace()` - Temporary trace file management
- `collect_tokens()` - Stream token collection helpers
- `assert_*()` - Common assertions

### `synthetic_trace_generator.rs`
Synthetic trace generation functions:
- `generate_basic_trace()` - Simple traces with lognormal timing
- `generate_batch_context_trace()` - Batch interference patterns
- `generate_speculative_trace()` - Multi-token chunks (EAGLE-style)
- `generate_diffusion_trace()` - Block outputs
- `generate_edge_cases_trace()` - Edge cases collection
- `generate_prefix_sharing_trace()` - Multi-turn conversations
- `generate_mixed_concurrency_trace()` - Varying batch sizes
- `generate_arrival_schedule_trace()` - Open-loop replay

All generators respect the `SYNTHETIC_E2E_FAST_MODE` environment variable for CI.

## Fast Mode (CI)

Synthetic tests support a fast mode for CI via environment variable:

```bash
# Enable fast mode (15-40ms TTFT instead of 50-400ms)
SYNTHETIC_E2E_FAST_MODE=1 cargo test --test engine_synthetic_e2e
```

This is automatically enabled in GitHub Actions CI for faster builds while maintaining correctness validation.

## Adding New Synthetic Tests

1. **Add generator function** in `synthetic_trace_generator.rs`:
   ```rust
   pub fn generate_my_trace(num_records: usize, seed: u64) -> (TraceMeta, Vec<TraceRecord>) {
       // Respect fast mode
       let timing = if is_fast_mode() { /* fast */ } else { /* realistic */ };
       // ... generate records
   }
   ```

2. **Add test** in `engine_synthetic_e2e.rs`:
   ```rust
   #[tokio::test]
   async fn test_my_feature() {
       let (meta, records) = generate_my_trace(10, 12345);
       let trace_file = create_temp_trace("my_feature", &meta, &records)
           .expect("create trace");
       let (client, _guard) = harness_with_trace("my_feature", trace_file.path(), &[/* flags */]).await;
       // ... validate behavior
   }
   ```

3. **Run and verify:**
   ```bash
   cargo test --test engine_synthetic_e2e test_my_feature
   ```

## Test Coverage Matrix

| Trace Type | Token Replay | Latency Replay | Prefix Match | Compression | Edge Cases |
|------------|--------------|----------------|--------------|-------------|------------|
| Basic      | ✓            | ✓              | ✓            | ✓           | -          |
| Batch Context | ✓         | ✓              | -            | -           | -          |
| Speculative | ✓           | -              | -            | -           | ✓          |
| Diffusion  | ✓            | -              | -            | -           | -          |
| Edge Cases | ✓            | -              | -            | -           | ✓          |
| Mixed Workload | ✓       | -              | -            | -           | -          |

## CI Integration

The synthetic E2E test suite runs in GitHub Actions as a separate job after `build-and-test`:

```yaml
synthetic-e2e:
  runs-on: ubuntu-latest
  needs: build-and-test
  env:
    SYNTHETIC_E2E_FAST_MODE: "1"
```

See `.github/workflows/ci.yml` for full configuration.

## Troubleshooting

### Tests hang or timeout
- Check for unique IPC endpoints (test name collision)
- Verify no zombie simulator processes: `pkill -f inf-sim`
- Increase `TIMEOUT` constant if tests are slow

### Flaky tests
- Ensure tests use seeded RNG (deterministic)
- Check for shared state between tests
- Verify RAII guards (`SimGuard`, `TempTraceFile`) are working

### Token count mismatches
- Verify trace record `output_tokens` matches `itl_ms.len() + 1`
- Check `itl_tokens` array alignment (speculative/diffusion)
- Validate finish reason is present

### CI failures
- Check if `SYNTHETIC_E2E_FAST_MODE` is set
- Verify sccache is working (speeds up rebuild)
- Look for resource exhaustion (too many parallel tests)

## Further Reading

- [Conformance Testing](../docs/conformance.md) - Golden capture validation
- [Trace Format](../crates/sim-trace/src/trace.rs) - JSONL schema documentation
- [Calibration](../src/calibrate.rs) - Timing model validation
- [Engine Core](../src/engine_core.rs) - Engine trait and loop
