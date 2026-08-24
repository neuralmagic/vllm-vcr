# Runtime control API

`vllm-vcr play --control-address HOST:PORT` (env `MOCK_CONTROL_ADDRESS`) serves a
small HTTP/JSON API for changing the engine's behavior while it runs and for
reading per-request counters. It is off unless the flag is set; the container
image enables it on port `8001` (`MOCK_CONTROL_PORT`).

The API exists for test harnesses. A test that needs a slow model, a burst of
failures, or proof that exactly N requests reached the engine can get all three
without restarting the pod or scraping logs.

## How it fits the engine

The engine loop is single-owner: `SimEngine` is not behind a lock, and
`run_loop` (`src/engine_core.rs`) is the only thing that touches it. The control
server does not break that. Every call becomes an `EngineInput::Control`
message on the same channel the frontend's requests arrive on, and the engine
answers it over a oneshot between two steps. A patch therefore lands at a step
boundary and applies to the next arriving request; in-flight requests keep the
pacing they were admitted with.

With `--engine-count N`, every call is fanned out to all N engines in order.
`GET /config` returns the first engine's config (they only diverge if a patch
fails part-way), `GET /stats` returns the sum.

## Routes

| Method | Path | Body | Response |
|---|---|---|---|
| `GET` | `/config` | | current [config](#config) |
| `PATCH` | `/config` | partial config (absent fields unchanged) | resulting config |
| `GET` | `/stats` | | [counters](#stats) summed over engines |
| `POST` | `/stats/reset` | | `204`; counters zeroed, gauges untouched |
| `GET` | `/log` | | `{"filter": "<EnvFilter directive>"}` |
| `PUT` | `/log` | `{"filter": "vllm_vcr=debug,info"}` | the filter now in effect |

Errors are JSON `{"error": "..."}` with a 4xx/5xx status:

- `400` a value fails validation (`failure_injection_rate` outside `[0, 1]`,
  `time_factor_under_load < 1.0`, empty `failure_types`, an unparsable log
  filter).
- `409` a latency knob was patched while `--latency-trace` is in use. The
  trace paces those requests, not the knobs; patching them would silently do
  nothing.
- `422` the body does not match the schema (unknown `failure_types` value,
  wrong field type).
- `500` an engine loop is gone.
- `501` `/log` on a process that did not install a reloadable subscriber
  (only the `vllm-vcr` binary does; embedding `run()` in a test does not).

### Config

Every field mirrors the CLI flag of the same name. Latency fields are
milliseconds.

```json
{
  "time_to_first_token": 200,
  "time_to_first_token_std_dev": 0,
  "inter_token_latency": 30,
  "inter_token_latency_std_dev": 0,
  "prefill_overhead": 0,
  "prefill_time_per_token": 0,
  "prefill_time_std_dev": 0,
  "time_factor_under_load": 1.0,
  "max_num_seqs": 128,
  "max_num_batched_tokens": 2048,
  "max_model_len": 0,
  "failure_injection_rate": 0.0,
  "failure_types": ["error"],
  "log_requests": true
}
```

Patching a latency field rebuilds the knob latency model from the patched
options. Raising `max_num_seqs` admits waiting requests immediately.
`max_model_len` changes the arrival check only; the value advertised to the
frontend at the handshake does not change.

`failure_types` values: `error` (a retryable engine error, which the vLLM
frontend turns into a 500), `length`, `repetition`.

Not patchable: `time_scale` (baked into every scheduled deadline),
`shutdown_timeout` (read once when the loop starts), replay and trace inputs,
and anything that reaches the wire at the handshake (`kv_cache_size`,
`tokens_per_block`, engine identity).

### Stats

```json
{
  "requests_received": 12,
  "requests_completed": 9,
  "requests_failed": 1,
  "requests_aborted": 2,
  "running": 0,
  "waiting": 0
}
```

| Field | Counts |
|---|---|
| `requests_received` | every request the frontend handed the engine, including ones rejected during shutdown |
| `requests_completed` | requests that ran to their stop condition |
| `requests_failed` | injected failures, `max_model_len` rejects, failed KV pulls, duplicate request ids |
| `requests_aborted` | aborts from the frontend (client disconnect, cancel) and from shutdown |
| `running` | gauge: requests in the batch or waiting on a KV pull right now |
| `waiting` | gauge: requests queued for a batch slot right now |

`received == completed + failed + aborted` once the engine is idle.

The counters are the authoritative witness for "how many requests actually
reached the engine", which is the number a gateway or router test wants when it
asserts no request was executed twice or that a request the API reported as
failed never ran.

## Examples

Make every request take about six seconds so a pod restart mid-request can be
observed:

```bash
curl -s -X PATCH localhost:8001/config \
  -H 'content-type: application/json' \
  -d '{"time_to_first_token": 1000, "inter_token_latency": 50}'
```

Choke the engine down to one running request so a router in front of it starts
shedding load, then release it:

```bash
curl -s -X PATCH localhost:8001/config -H 'content-type: application/json' \
  -d '{"max_num_seqs": 1, "inter_token_latency": 2000}'
# ... drive load, observe backpressure ...
curl -s -X PATCH localhost:8001/config -H 'content-type: application/json' \
  -d '{"max_num_seqs": 128, "inter_token_latency": 30}'
```

Fail every request until told otherwise:

```bash
curl -s -X PATCH localhost:8001/config -H 'content-type: application/json' \
  -d '{"failure_injection_rate": 1.0, "failure_types": ["error"]}'
```

Count what the engine served during a test window:

```bash
curl -s -X POST localhost:8001/stats/reset
# ... run the scenario ...
curl -s localhost:8001/stats | jq .requests_received
```

Turn on engine tracing for one scenario without a restart:

```bash
curl -s -X PUT localhost:8001/log -H 'content-type: application/json' \
  -d '{"filter": "vllm_vcr::engine=trace,info"}'
```

The filter string is a `tracing` `EnvFilter` directive, the same syntax as
`RUST_LOG`.
