# Trace replay and calibration

Trace replay has two separate axes:

- **Workload replay:** recorded arrivals, sessions, prompt prefix structure, and
  optionally output token ids.
- **Timing replay:** either sampled from a fitted latency model, replayed verbatim
  from recorded step gaps, or replaced with explicit timing knobs.

Keeping those axes separate is important. You can replay a captured workload while
testing a latency model fit from a different capture, or you can serve the same
recorded token stream with synthetic timing for fast client tests.

The trace files used to build the committed figures live under `traces/`, which is
gitignored. See
[traces/README.md](https://github.com/neuralmagic/vllm-vcr/blob/main/traces/README.md)
for the local inventory and which captures are fitting inputs versus gate seeds.

Start with the [Simulation guide](../simulation-guide.md) for trace format, prefill,
budget, and how `play` consumes captures. Then read [Concepts](./concepts.md) for
terminology, and use the scenario pages for arrival replay, prefix-cache workloads,
content-identical replay, and multi-token step replay.
