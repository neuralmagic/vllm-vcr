# Engine-step traces

The tap records what crosses the frontend socket. Two boundaries inside vLLM are
not sockets: the model runner (one `SchedulerOutput` in, one `ModelRunnerOutput`
out) and the KV connector. An engine-step trace records both, and
`play --use-rust-core --step-trace` replays them: vLLM's own scheduler decides
what each step contains, and the step takes the time a real step of that shape
took.

```text
  capture                                        replay
  vllm serve --scheduler-cls                     vllm-vcr play --use-rust-core
      vcr_step_recorder.recording_scheduler          --step-trace engine-steps.jsonl
        │                                              │
        ▼                                              ▼
  engine-steps.jsonl ───────────────────────▶  step shape ─▶ recorded step time
  {"step": ...} {"kv_transfer": ...}           KV load size ─▶ recorded load time, outcome
```

## What is recorded

One line per executed step:

| Field | Meaning |
| --- | --- |
| `step`, `ts_ms` | Executed-step index, and when `schedule()` was called, since capture start |
| `exec_ms` | Model runner wall time: `schedule()` returning to `update_from_output()` being called |
| `step_ms` | Engine loop time the step accounts for, which is what replay uses. Under async scheduling this is the step period |
| `sched_ms`, `update_ms` | Time inside `schedule()` and `update_from_output()` |
| `decode` | `reqs`, `tokens` (drafts included), `drafts`, `context_tokens` already in KV |
| `prefill` | One `{tokens, kv_depth, last}` per request still on its prompt |
| `sampled`, `accepted` | Tokens the runner returned, and speculative drafts accepted. Both count what the runner returned, before the scheduler trims tokens past a stop |

One line per KV transfer: `direction` (`recv` or `send`), `blocks`, `start_step`,
`end_step`, `wall_ms`, and `outcome` (`ok`, `failed`, `invalid_blocks`). A transfer
starts in the step whose connector metadata names it and ends in the step whose
connector output reports it. NIXL's own per-transfer telemetry (bytes, post and
transfer duration) is in the tap's `--step-stats-out` sidecar, under
`scheduler.kv_connector_stats`.

The first line is the same `{"meta": {...}}` header a request trace carries, so
`--expect-config-hash` covers the file. Set `VCR_CONFIG_HASH` on the engine to
stamp it.

## Capturing from vLLM

`deploy/trace-capture/vcr_step_recorder.py` is one file with no dependencies
beyond vLLM. Put it on the engine's `PYTHONPATH` and start the engine with:

```bash
VCR_STEP_TRACE_OUT=/trace/engine-steps.jsonl \
  vllm serve ... --scheduler-cls vcr_step_recorder.recording_scheduler
```

`recording_scheduler` asks vLLM which scheduler it would have built without the
flag (sync, async, diffusion, or the Rust core) and subclasses it, so the
scheduling policy and async scheduling are unchanged. On the capture rig:

```bash
just capture-up
just capture-steps-on     # configmap + engine patch; restarts the pod
just capture-run          # also fetches engine-steps.jsonl
just capture-up           # back to an engine without the recorder
```

Under async scheduling the runner executes steps for requests that have already
finished, and vLLM leaves those out of `spec_decoding_stats`. The trace counts
them, because the runner executed them, so its draft totals come out above
vLLM's (2.2% in the [GPU run](./step-recorder-report.md)). Under sync scheduling
the draft and accepted totals equal vLLM's.

### Cost on the engine

The engine thread does two clock reads and one `deque.append` in `schedule()` and
again in `update_from_output()`, where it also takes the length of each request's
sampled-token list before the scheduler trims it. Everything else (per-request
loops, JSON, file writes) runs on a second thread off a 50 ms timer, giving up
the GIL after each step it processes.

`just check-step-recorder <vllm-src>` measures this against vLLM's real scheduler,
with vLLM's scheduler test harness standing in for the model runner:

| Measurement | Result (Apple M-series, CPython 3.12) |
| --- | --- |
| The two overrides, over a scheduler that does nothing, 16 requests per step | 0.6 to 1.0 µs per step |
| Python scheduler, engine thread blocks 1 ms on the runner | within the harness noise of ±20 µs per step |
| Python scheduler, engine thread never blocks | about 10 µs per step, the second thread's work contending for the GIL |

On an H200 the recorder-on runs fell inside the pod-to-pod spread of the
recorder-off runs (about 0.6 ms per step); see
[Step recorder on a GPU](./step-recorder-report.md). On other hardware, run the
same load with the recorder off and on, and compare the two tap traces, which
observe the engine from outside:

```bash
just summarize /tmp/recorder-off/tap-trace.jsonl
just summarize /tmp/recorder-on/tap-trace.jsonl
```

## Replaying

```bash
vllm-vcr play --use-rust-core --step-trace engine-steps.jsonl
```

Each simulated step is looked up by decode request count, mean decode context,
prefill tokens and deepest prefill KV depth, bucketed at four buckets per
doubling. A shape that was recorded takes one of its recorded times at random; a
shape that was not takes the closest recorded shape's. Each remote KV load takes
the time and outcome of a recorded load of similar block count, so recorded
failures are replayed and recomputed locally.

`--step-trace` replaces every time the latency model would supply, including the
first-token overhead, which the recorded steps already contain. Burst sizes and
output lengths still come from `--latency-trace` and the `--replay-*` flags.

`play --use-rust-core --step-trace-out <file>` writes the same format from a
simulated run.

`just replay-step-capture <engine-steps.jsonl>` replays a capture and compares
the median step gap with the median recorded step time at decode concurrency 1
and 16.
