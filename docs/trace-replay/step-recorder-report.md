# Step recorder on a GPU: measured cost and accuracy

One capture session on one GPU, 2026-11-12. It answers three questions about
[`vcr_step_recorder.py`](./step-trace.md): does turning it on change the engine's
latency, does its trace agree with what the tap and vLLM's own stats saw, and
does the trace replay to the recorded step times.

## Setup

| | |
| --- | --- |
| Rig | `deploy/trace-capture/base/h200-capture.yaml`: loadgen → `vllm-rs` → tap → headless vLLM |
| GPU | 1x NVIDIA H200 (CoreWeave `gd-8xh200ib-i128`), every run on the same node |
| Engine | `vllm/vllm-openai:v0.33.0`, `Qwen/Qwen3-8B`, `--enforce-eager`, ngram speculative decoding with 4 drafts |
| Scheduling | async (`AsyncScheduler`), vLLM's default for this config |
| Tap, frontend | `ghcr.io/neuralmagic/vllm-vcr:vllm0.33` |
| Load | `run-capture.sh`: 60 s closed loop at concurrency 1, then 120 s at 16; 795-token prompts, 128 output tokens, fixed seed |

The recorder can only be switched by restarting the engine, so an off/on
comparison is also a comparison between two pods. Five pods ran the load, in the
order off, on, off, on, on. Two were lost to preemption by another workload:
one mid-load (no data), one after its load finished and before its tap trace was
fetched (client-side numbers only).

## Latency with the recorder off and on

Step gaps as the tap saw them at the engine socket (`itl_ms` of the tap trace):

| Run | Pod | Recorder | c=1 requests | c=1 gap p50 / p90 / p99 (ms) | c>=12 requests | c>=12 steps | c>=12 gap mean / p50 / p90 / p99 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| off1 | 1 | off | 118 | 12.12 / 12.51 / 15.68 | 3033 | 108,512 | 14.86 / 12.85 / 20.50 / 41.41 |
| off2 | 1 | off | 122 | 12.17 / 12.59 / 16.21 | 3018 | 107,786 | 14.99 / 12.90 / 21.88 / 42.80 |
| on1 | 2 | on | 114 | 12.32 / 12.73 / 16.17 | 2969 | 106,050 | 15.07 / 13.15 / 20.83 / 41.72 |
| off3 | 3 | off | 109 | 12.74 / 14.01 / 18.89 | 2954 | 105,606 | 15.19 / 13.21 / 21.51 / 41.51 |

What the load generator saw through `kubectl port-forward`:

| Run | Recorder | c=1 mean ITL p50 (ms) | c=16 requests in 120 s | c=16 req/s | c=16 TTFT p50 (ms) | c=16 mean ITL p50 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| off1 | off | 11.7 | 3053 | 25.44 | 76 | 14.4 |
| off2 | off | 11.9 | 3057 | 25.48 | 82 | 14.6 |
| on1 | on | 12.0 | 2996 | 24.97 | 80 | 14.6 |
| off3 | off | 12.5 | 2978 | 24.82 | 80 | 14.7 |
| on4 | on | 11.9 | 2996 | 24.97 | 79 | 14.7 |

Two recorder-off pods differ from each other by 0.62 ms (5%) in the median step
gap at concurrency 1 and by 2.6% in throughput at concurrency 16. Both
recorder-on pods fall between them on every column. Two runs on the same pod
(off1, off2) agree within 1%.

So this method shows no recorder effect, and cannot resolve one smaller than the
pod-to-pod spread of about 0.6 ms per step. The engine-thread cost measured
without a GPU is about 1 µs per step ([Cost on the engine](./step-trace.md#cost-on-the-engine)).

## The trace against the tap and vLLM's stats

Run on1 (async scheduling, 3109 requests, 11,783 executed steps):

| Quantity | Step trace | Independent source |
| --- | --- | --- |
| Requests finishing prefill | 3109 | 3109 requests in the tap trace |
| Prefill tokens | 2,448,286 | 2,448,274 prompt minus cached tokens in the tap trace |
| Decode-only step time at 1 request, p50 | 12.31 ms (4012 steps) | 12.32 ms tap gap p50 at concurrency 1 |
| Decode-only step time at 12+ requests, p50 | 13.07 ms (5263 steps) | 13.15 ms tap gap p50 at concurrency 12+ |
| Draft tokens | 453,716 | 443,820 in vLLM's `spec_decoding_stats` |

The 12-token prefill difference is unexplained. The draft-token excess is
9,896, which is 2,474 request-steps of 4 drafts: under async scheduling the
runner executes steps for requests that have already finished, and vLLM's
`update_from_output` skips those when it counts. The trace counts what the
runner executed.

A 45 s run at concurrency 16 with `--no-async-scheduling`, where no such steps
exist (873 requests, 2163 executed steps):

| Quantity | Step trace | vLLM / tap |
| --- | --- | --- |
| Draft tokens | 125,060 | 125,060 |
| Accepted drafts | 81,233 | 81,233 |
| Prefill tokens | 698,930 | 698,930 |
| Sampled tokens | 113,371 | 111,744 emitted |

The 1,627 sampled tokens that were not emitted are the ones the scheduler trims
past a length stop.

The first sync run reported 78,567 accepted against vLLM's 80,233. The recorder
was reading the sampled-token lists on its background thread, after the
scheduler had trimmed them in place. It now takes their lengths on the engine
thread before the scheduler runs, and the table above is from the fixed
recorder. Run on1 predates the fix, so its accepted and sampled counts are not
listed.

## Replaying the capture

`just replay-step-capture <engine-steps.jsonl>` replays the trace on
`play --use-rust-core --step-trace` and compares the median step gap the tap
sees with the median recorded step time at the same decode concurrency. For run
on1:

| Decode concurrency | Captured p50 | Replayed p50 |
| --- | --- | --- |
| 1 | 12.31 ms | 12.53 ms (47 gaps) |
| 16 | 12.80 ms | 12.67 ms (752 gaps) |

## What this does not show

- One GPU model, one node, one model, eager mode. Nothing about CUDA graphs,
  larger batches, tensor parallelism or other schedulers.
- Three recorder-off runs on two pods and two recorder-on runs on two pods. That
  bounds the effect by the pod-to-pod spread, it does not measure it.
- The recorder-on run with tap-side data (on1) used the recorder before the
  sampled-length fix. The fixed recorder has client-side numbers only (on4).
- No KV transfers: the rig has no connector. Transfer records are checked only
  against vLLM's scheduler harness with the NIXL connector, without a GPU.
- The replay check compares medians at two concurrencies, not full distributions
  or mixed prefill and decode steps.

## Reproducing

```bash
just capture-up                       # recorder off
OUT_DIR=/tmp/recorder-off just capture-run
just capture-steps-on                 # restarts the engine with the recorder
OUT_DIR=/tmp/recorder-on just capture-run
just capture-down

just summarize /tmp/recorder-off/tap-trace.jsonl
just summarize /tmp/recorder-on/tap-trace.jsonl
just replay-step-capture /tmp/recorder-on/engine-steps.jsonl
```

Pin the pod to one node (`nodeSelector`) before comparing, and repeat the
recorder-off run on a fresh pod to see the pod-to-pod spread on your hardware.
