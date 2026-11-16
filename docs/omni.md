# vLLM-Omni stages

`vllm-vcr play` can stand in for any stage of a
[vLLM-Omni](https://github.com/vllm-project/vllm-omni) pipeline. The omni head (API
server and orchestrator) is real and unmodified. Each stage it would normally run on a
GPU is a `play` process with no model.

<div class="vcr-flow vcr-flow-omni" role="img" aria-label="An omni head drives play processes as its stages: thinker, talker, and code2wav for speech, or one diffusion stage for images.">
  <div class="vcr-node">
    <span class="vcr-node-kicker">real</span>
    <strong>vLLM-Omni head</strong>
    <span>API server + orchestrator</span>
  </div>
  <div class="vcr-connector"><span>one socket pair per stage</span></div>
  <div class="vcr-node-stack">
    <div class="vcr-node">
      <span class="vcr-node-kicker">llm stages</span>
      <strong>play × N</strong>
      <span>thinker, talker, code2wav</span>
    </div>
    <div class="vcr-node">
      <span class="vcr-node-kicker">diffusion stage</span>
      <strong>play</strong>
      <span>image generation and edit</span>
    </div>
  </div>
</div>

What has been run through a real head:

| Endpoint | Pipeline | Stages mocked |
| --- | --- | --- |
| `/v1/chat/completions`, text | Qwen2.5-Omni | thinker |
| `/v1/chat/completions`, text and audio | Qwen2.5-Omni | thinker, talker, code2wav |
| `/v1/audio/speech` | Qwen3-TTS | talker, code2wav |
| `/v1/images/generations` | any single-stage diffusion pipeline | diffusion |
| `/v1/images/edits` | any single-stage diffusion pipeline | diffusion |

All of them work streaming and non-streaming where the endpoint offers both.

## How a stage attaches

Every stage, of either kind, attaches the same way:

1. **Register.** The stage sends its stage id to the head's `OmniMasterServer` and gets
   back a handshake address, data addresses, and a coordinator address.
2. **Handshake.** vLLM's HELLO / INIT / READY on the handshake address. The head binds
   the data sockets only after READY, so READY goes first. Plain vLLM does it the other
   way round.
3. **Report.** The stage tells the head's coordinator it is up, then heartbeats every
   five seconds with its queue length. The head routes only to replicas that reported
   up, and detaches one that reports down or goes quiet.

The head becomes ready once every stage of the pipeline has attached.

## LLM stages

An omni LLM stage (thinker, talker, code2wav) is a subclass of vLLM's engine core, so
it speaks the protocol `play` already implements. Everything `play` does for vLLM
applies: the latency knobs, trace replay, scheduler caps, and failure injection. Omni
adds two things to the wire:

- **Request tail.** `OmniEngineCoreRequest` appends fields to the request tuple
  (`additional_information`, `model_intermediate_buffer`, `payload_sender_info`). The
  mock drops them.
- **Output tail.** `OmniEngineCoreOutput` appends fields to the output tuple. The first
  is `multimodal_output`, which the pipeline's final audio stage fills with the
  waveform.

Start the head with every stage remote, then attach one `play` per stage:

```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni --stage-id none \
  --omni-master-address 127.0.0.1 --omni-master-port 21212

# stage 0: thinker (text)
vllm-vcr play --omni-master-address tcp://127.0.0.1:21212 --omni-stage-id 0 \
  --time-to-first-token 100 --inter-token-latency 20

# stage 1: talker (codec tokens)
vllm-vcr play --omni-master-address tcp://127.0.0.1:21212 --omni-stage-id 1 \
  --inter-token-latency 5 --output-tokens 250

# stage 2: code2wav (audio)
vllm-vcr play --omni-master-address tcp://127.0.0.1:21212 --omni-stage-id 2 \
  --inter-token-latency 1 --output-tokens 250 --omni-audio-samples-per-token 480
```

### Audio

`--omni-audio-samples-per-token` marks the final audio stage. Each output it emits
carries that many samples of silence per token, so the response audio lasts
`output tokens × samples per token ÷ sample rate`. At the head's 24 kHz default, 480
samples per token is 20 ms per token; a 12.5 Hz codec such as Qwen3-TTS's is 1920.
`--omni-audio-sample-rate` reports a different rate with the audio.

### Stage lengths

A synthetic `play` stage runs to the request's `max_tokens` unless `--output-tokens`
is set, and downstream stages have large defaults (4096 for a talker, 65536 for
code2wav). Set `--output-tokens` on them.

- Some TTS adapters reject a talker that ends on `length`. Qwen3-TTS answers 500 with
  `did not emit codec EOS before its token budget` unless stage 0 stops on EOS, which
  `--output-tokens` provides.
- Leave `--max-model-len` at its default on a stage whose `max_tokens` is large. A
  stage rejects a request whose `prompt + max_tokens` exceeds it.

## Diffusion stages

A diffusion stage does not speak engine-core. After the same attach, the head sends
single-frame msgpack maps (`add_request`, `abort`, `collective_rpc`, `shutdown`) over a
PUSH socket and reads results from a PULL socket. Images travel inline as raw RGB.
`--omni-stage-kind diffusion` switches `play` to this protocol.

```bash
vllm serve black-forest-labs/FLUX.2-klein-4B --omni --stage-id none \
  --omni-master-address 127.0.0.1 --omni-master-port 21212

vllm-vcr play --omni-master-address tcp://127.0.0.1:21212 --omni-stage-id 0 \
  --omni-stage-kind diffusion \
  --omni-diffusion-step-time 40 --omni-diffusion-overhead 300
```

The head serves both `/v1/images/generations` and `/v1/images/edits` from the one
stage. With every stage remote it reads only the pipeline's `model_index.json`, so a
tiny pipeline such as `hf-internal-testing/tiny-flux2-klein` is enough to stand a head
up.

### Latency model

```text
time = overhead + step_time × steps × images × (width × height ÷ 1024²)
```

| Flag | Meaning | Default |
| --- | --- | --- |
| `--omni-diffusion-step-time` | Milliseconds per denoising step for one 1024x1024 image. | 0 |
| `--omni-diffusion-overhead` | Fixed milliseconds per request (text encoding, VAE decode). | 0 |
| `--omni-diffusion-default-steps` | Steps when the request sets no `num_inference_steps`. | 50 |
| `--omni-diffusion-default-size` | Size when the request sets none. | 1024x1024 |
| `--omni-diffusion-concurrency` | Requests executed at once. The rest queue in arrival order. | 1 |

An edit request is timed the same way as a generation of its output size. The input
images are received and counted, not modeled.

The stage returns RGB images of the requested size and count, each noise in its top
half and flat gray below. The head PNG-encodes them without compression, so response
size matches a real image of the same dimensions. The noise is there for the head's
encoder, whose time depends on content: at 1024x1024 it takes 8 ms on a flat image,
64 ms on noise, 45 ms on a FLUX.2-klein output, and 36 ms on the mock's image. The
same request always returns the same bytes.

A request for more than 64 megapixels in total is answered with a 400. An aborted
request frees its execution slot and sends nothing, as a real stage does.

### Trace replay

`--latency-trace` takes a [recorded diffusion trace](#recording-a-diffusion-stage) and
replaces the two latency knobs. It is exclusive with `--omni-diffusion-step-time` and
`--omni-diffusion-overhead`.

```bash
vllm-vcr play --omni-master-address tcp://127.0.0.1:21212 --omni-stage-id 0 \
  --omni-stage-kind diffusion --latency-trace flux.jsonl
```

A request is matched to the trace by its shape: size, step count, image count, and
number of input images, each as the head sent it (a field the client left unset
matches only requests that left it unset).

- **A recorded shape** replays that shape's recorded times in the order they were
  recorded, wrapping around. A slow first request of a shape (a cold compile) is
  replayed where it was recorded. A request that set no size returns images of the
  size the recorded request returned.
- **Any other shape** is timed by the latency model above, with `overhead` and
  `step_time` fitted by least squares to the median time of each recorded shape. If
  the trace holds one step count, or the fit would be negative, the fit is a line
  through the origin. If no recorded request has a known size and step count, every
  unrecorded shape takes the median recorded time. The fit does not know edits from
  generations.

A recorded request's latency at the tap is replayed in three parts, split by the
stage's own metrics:

| Part | From the trace | In the mock |
| --- | --- | --- |
| Queue wait | `scheduler_queue_wait_ms` | Not replayed. The mock queues requests itself. |
| Slot | `diffusion_engine_exec_time_ms` less the wait | Holds one of `--omni-diffusion-concurrency` execution slots. |
| Tail | latency less `diffusion_engine_exec_time_ms` | Runs after the slot is released, before the result is sent. |

The tail is post-processing and serialization, which a real stage does while its next
request is already denoising. A trace without the stage's exec time replays the whole
time in the slot, as a fitted time does.

Set `--omni-diffusion-concurrency` to what the real stage runs at once; the trace does
not record it. Recorded times carry whatever contention the stage was under, so a
trace recorded at one load replays best at a similar one.

Only finished requests carry timing. Recorded errors and aborts are in the trace and
are not replayed.

### Control API

`--control-address` serves the [runtime control API](./control-api.md#diffusion-stages)
on a diffusion stage too: read and patch the latency knobs and concurrency, and read
request counters with exact running and waiting counts. Under `--latency-trace` the
latency fields report the fitted model and a patch to them is refused with `409`.

## Recording a stage

`vllm-vcr record` can sit between the head and a real stage of either kind. An LLM
stage is captured the way a plain engine is: per-token timing, token ids with
`--record-tokens`, and the trace metadata, plus `omni_stage_id`. The trace then
drives a mock of the same stage through `--latency-trace` and `--replay-tokens`. A
diffusion stage is captured into [its own trace](#recording-a-diffusion-stage).

<div class="vcr-flow vcr-flow-capture" role="img" aria-label="The omni head talks to the recording tap, which talks to the real stage; the tap writes a trace.">
  <div class="vcr-node">
    <span class="vcr-node-kicker">real</span>
    <strong>vLLM-Omni head</strong>
    <span>master :21212</span>
  </div>
  <div class="vcr-connector"><span>registration, handshake, data, status</span></div>
  <div class="vcr-node vcr-node-accent">
    <span class="vcr-node-kicker">tap</span>
    <strong>vllm-vcr record</strong>
    <span>master :21213</span>
  </div>
  <div class="vcr-connector"><span>same four, tap addresses</span></div>
  <div class="vcr-node">
    <span class="vcr-node-kicker">real, GPU</span>
    <strong>headless stage</strong>
    <span>thinker, talker, or diffusion</span>
  </div>
  <div class="vcr-connector"><span>JSONL</span></div>
  <div class="vcr-node">
    <span class="vcr-node-kicker">output</span>
    <strong>trace</strong>
    <span>one per stage</span>
  </div>
</div>

Besides the handshake and data frames the tap already proxies, it stands in the
middle of the two omni-only exchanges:

- **Registration.** The stage registers with the tap. The tap passes that registration
  to the head unchanged and answers the stage with its own addresses.
- **Status.** The stage reports to the tap's coordinator address. The tap forwards each
  event to the head with the tap-side data addresses swapped for the head-side ones,
  which are what the head routes by.

```bash
# The tap: registers with the head on :21212, listens for the stage on :21213.
vllm-vcr record --omni-master-address tcp://127.0.0.1:21212 \
  --trace-out thinker.jsonl --model Qwen/Qwen2.5-Omni-7B --record-tokens

# The real stage registers with the tap instead of the head.
vllm serve Qwen/Qwen2.5-Omni-7B --omni --headless --stage-id 0 \
  --omni-master-address 127.0.0.1 --omni-master-port 21213
```

One tap records one stage. To record two stages, run two taps on different ports
(`--omni-master-listen`, `--omni-coordinator-listen`, `--engine-handshake`,
`--input-address`, `--output-address`).

The tap added no measurable latency: the same 24 image requests averaged 366.0 ms
through it and 365.8 ms without it.

To replay, give the mock stage the trace and the real stage's scheduler limits. The
limits are in the trace's `ready_response_hex`; the thinker below runs one sequence
at a time.

```bash
vllm-vcr play --omni-master-address tcp://127.0.0.1:21212 --omni-stage-id 0 \
  --latency-trace thinker.jsonl --replay-tokens thinker.jsonl --replay-match prefix \
  --max-num-seqs 1 --max-model-len 32768 --kv-cache-size 48824
```

### Measured fidelity

Qwen2.5-Omni-7B on one H200, vLLM-Omni v0.32.0. The thinker was recorded under 48
streaming text requests, then replaced on the same head by a mock replaying the
trace, and the same requests were sent again. Times are client-side means.

| Workload | | Real thinker | Mock replaying the trace |
| --- | --- | --- | --- |
| Concurrency 1 | TTFT | 29.7 ms | 38.0 ms |
| | Total | 951.6 ms | 965.7 ms |
| Concurrency 4 | TTFT | 2601 ms | 2523 ms |
| | Total | 3526 ms | 3453 ms |

Totals are within about 2%. The mock returned the recorded text and the same number
of chunks. The talker records the same way: 112 codec tokens per request at an 11 ms
inter-token gap, ending on `stop`.

### Recording a diffusion stage

`--omni-stage-kind diffusion` records a diffusion stage. Registration and status are
relayed the same way. The data path is the diffusion one: the tap connects a PULL and
a PUSH to the head and binds a PUSH and a PULL for the stage, and relays every frame
unchanged in both directions, including the ones it cannot decode.

```bash
vllm-vcr record --omni-master-address tcp://127.0.0.1:21212 \
  --omni-stage-kind diffusion \
  --trace-out flux.jsonl --model black-forest-labs/FLUX.2-klein-4B --gpu H200

vllm serve black-forest-labs/FLUX.2-klein-4B --omni --headless --stage-id 0 \
  --omni-master-address 127.0.0.1 --omni-master-port 21213
```

The head sees the stage's HELLO when the stage sends it and READY when the stage has
loaded its model, so the head's startup deadline is the one it would apply without the
tap. `--record-tokens`, `--step-stats-out`, `--tp`, `--max-num-seqs`, and
`--block-size` do not apply.

#### The diffusion trace

A diffusion request has no tokens, so the trace has its own record type. The meta line
carries `"kind": "diffusion"`. A reader checks it: token replay, `--latency-trace` on an
LLM stage, and `inspect perfetto`, `calibrate`, and `calibrate-e2e` refuse a diffusion
trace, and a diffusion stage refuses a token trace. `inspect summarize` reads both.

```json
{"meta": {"kind": "diffusion", "model": "black-forest-labs/FLUX.2-klein-4B", "gpu": "H200", "source": "tap", "omni_stage_id": 0}}
{"arrival_ms": 82840.1, "size": {"width": 768, "height": 512}, "steps": 8, "images": 1, "input_images": 0,
 "concurrency": 4, "latency_ms": 899.2, "outcome": "finished", "output_size": {"width": 768, "height": 512},
 "stage_metrics": {"diffusion_engine_exec_time_ms": 858.1, "scheduler_queue_wait_ms": 523.0, "postprocess_time_ms": 39.2}}
```

| Field | Meaning |
| --- | --- |
| `arrival_ms` | When the request reached the tap, in milliseconds since the capture started. |
| `size`, `steps`, `images` | `width` and `height`, `num_inference_steps`, and `num_outputs_per_prompt` as the head sent them. `size` and `steps` are absent when the client left them to the stage. |
| `input_images` | Images sent with the request: 0 for a generation, 1 or more for an edit. |
| `concurrency` | Requests in flight through the tap at arrival, this one included. |
| `latency_ms` | Arrival to the stage's answer at the tap. For an aborted request, arrival to the abort. |
| `outcome` | `finished`, `error` (with `status`, the stage's `status_code`), or `aborted` (the head sent `abort` first). |
| `output_size` | Size of the images returned. |
| `stage_metrics` | Every numeric entry of the stage's own `metrics` for the request. |

A record is written when a request reaches an outcome. Requests still in flight when
the tap stops, or when the stage reports itself dead, are not recorded.

```bash
vllm-vcr inspect summarize flux.jsonl
```

```text
kind:  diffusion
model: black-forest-labs/FLUX.2-klein-4B
gpu:   H200
src:   tap
fit:   overhead 100.6 ms + 70.64 ms per step at 1024x1024

       size    steps  images  inputs  finished  errors  aborted        latency_ms (p50/p90/max)  service_p50
    512x512        4       1       0         9       0        0      134.9/    919.9/    919.9        134.8
    512x512        4       1       1         9       0        0      207.6/    975.4/    975.4        207.4
    512x512        4       2       0         9       0        0      237.5/  16373.2/  16373.2        237.4
    512x512        8       1       0         9       0        0      216.8/   1499.3/   1499.3        216.7
    768x512        8       1       0         9       0        0      280.0/    902.5/    902.5        279.9
  1024x1024        4       1       0         9       0        0      389.3/   1127.7/   1127.7        389.2
  1024x1024        4       1       1         9       0        0      463.2/   1476.3/   1476.3        463.1
  1024x1024        8       1       0         9       0        0      642.6/   1462.0/   1462.0        642.5
```

`fit` is the model unrecorded shapes are timed by. `service_p50` is the median
latency less the stage's queue wait.

### Measured diffusion fidelity

FLUX.2-klein-4B on one H200, vLLM-Omni v0.32.0. The stage was recorded under 72 image
requests (`deploy/omni/loadgen_images.py`: eight shapes, generations and edits, 512x512
to 1024x1024, 4 and 8 steps, one and two images). The real stage and a mock replaying
that trace with `--omni-diffusion-concurrency 1` were then each attached to the same
head and sent the same requests. Times are client-side means over 24 requests.

| Workload | | Real stage | Mock replaying the trace |
| --- | --- | --- | --- |
| Concurrency 1 | Mean | 366 ms | 368 ms |
| | 512x512, 4 steps | 158 ms | 157 ms |
| | 1024x1024, 4 steps | 462 ms | 464 ms |
| | 1024x1024, 8 steps | 721 ms | 712 ms |
| | Edit, 1024x1024, 4 steps | 539 ms | 543 ms |
| Concurrency 4 | Mean | 1177 ms | 1186 ms |
| | Median | 1202 ms | 1210 ms |

Means are within 1%. The head's own per-stage timing (`OmniTiming`) was the same for
both, to its 10 ms resolution.

What the mock's images contain matters to this number. With flat gray images the mock
was 6% fast at concurrency 1, and with all-noise images 7% slow, because of the head's
PNG encoder. Half noise puts it where a real image is.

Request shapes the trace did not contain, timed by the fit (median of three, after one
warm-up request on the real stage):

| Shape | Real stage | Mock |
| --- | --- | --- |
| 256x256, 4 steps | 112 ms | 127 ms |
| 1024x512, 4 steps | 247 ms | 279 ms |
| 768x768, 6 steps | 346 ms | 383 ms |
| 512x512, 16 steps | 394 ms | 404 ms |
| 1536x1024, 4 steps | 705 ms | 647 ms |

The fit is within about 13% of the real stage. It is high for small images and low at
1536x1024, which is larger than anything recorded.

## Flags

| Flag | Meaning |
| --- | --- |
| `--omni-master-address` | The head's `OmniMasterServer` endpoint. Replaces `--handshake-address`. |
| `--omni-stage-id` | Pipeline stage this process serves. |
| `--omni-stage-kind` | `llm` (default) or `diffusion`. |
| `--omni-replica-address` | Host the head records for this replica's connector endpoints. |
| `--omni-audio-samples-per-token` | LLM stages. Marks the final audio stage and sets audio duration per token. |
| `--omni-audio-sample-rate` | LLM stages. Sample rate reported with the audio. |
| `--output-tokens` | LLM stages. Synthetic EOS point: a request allowed more tokens finishes here with `stop`. |
| `--control-address` | Either kind. Serves the runtime control API for the stage. |
| `--omni-diffusion-*` | Diffusion stages. See the latency model. |
| `--latency-trace` | Either kind. A token trace on an LLM stage, a diffusion trace on a diffusion stage. |

## Versions

Build `play` for the vLLM line the omni release pins. LLM-stage requests and outputs
are positional tuples, so a mismatched line shifts the omni tail. vLLM-Omni v0.32.0
pins vLLM 0.32.0:

```bash
cargo xtask pin-vllm 0.32
VLLM_TARGET_VERSION=v0.32.0 cargo build --release -p vllm-vcr
```

The diffusion protocol does not depend on the vLLM line. The omni code builds and its
tests pass on every line in `compat.toml`.

## The rig

`deploy/omni` runs a real head and `play` stages in one pod. The head container needs
a GPU even though it loads no model, because vLLM's platform probe fails on the CUDA
image without one.

```bash
just omni-up                 # schedule the pod
just omni-build              # copy the working tree in, build play for VLLM_LINE (0.32)

just omni-head Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
just omni-stage 0 --inter-token-latency 5 --output-tokens 60
just omni-stage 1 --inter-token-latency 5 --output-tokens 60 --omni-audio-samples-per-token 1920

just omni-status             # head health, attached replicas, request timings
just omni-logs head          # or a stage id
just omni-down               # release the GPU
```

To record, run the stages for real and put a tap in front of one of them:

```bash
just omni-head Qwen/Qwen2.5-Omni-7B
just omni-record 0 --model Qwen/Qwen2.5-Omni-7B --record-tokens   # prints the tap's port
just omni-real-stage 0 Qwen/Qwen2.5-Omni-7B 21213                 # through the tap
just omni-real-stage 1 Qwen/Qwen2.5-Omni-7B                       # straight to the head
just omni-real-stage 2 Qwen/Qwen2.5-Omni-7B

kubectl -n inference-sim exec -i omni-rig -c head -- \
  python3 - --model Qwen/Qwen2.5-Omni-7B --requests 24 < deploy/omni/loadgen.py

just omni-stop-records                                            # finalizes the trace
just omni-fetch /work/stage0.jsonl /tmp/thinker.jsonl
```

A diffusion stage records the same way, with an image workload:

```bash
just omni-head black-forest-labs/FLUX.2-klein-4B
just omni-record 0 --omni-stage-kind diffusion --model black-forest-labs/FLUX.2-klein-4B --gpu H200
just omni-real-stage 0 black-forest-labs/FLUX.2-klein-4B 21213

kubectl -n inference-sim exec -i omni-rig -c head -- \
  python3 - --model black-forest-labs/FLUX.2-klein-4B --requests 24 < deploy/omni/loadgen_images.py

just omni-stop-records
just omni-fetch /work/stage0.jsonl /tmp/flux.jsonl

just omni-stop-stages                                             # swap in the mock
just omni-stage 0 --omni-stage-kind diffusion --latency-trace /work/stage0.jsonl
```

The first request of each shape and image count compiles on a real stage (16 seconds
for the first two-image request above). Send the workload once to warm up before the
passes you compare.

Two things to know when the real stages share one GPU:

- Start them one at a time and let each finish loading. Loading together races on
  GPU memory and one of them fails with `No available memory for the cache blocks`.
- A pipeline's deploy config may place stages on several GPUs and give each most of
  one. Pass `--stage-overrides` to fit them on one, for example
  `{"0":{"gpu_memory_utilization":0.45},"1":{"devices":"0","gpu_memory_utilization":0.3},"2":{"gpu_memory_utilization":0.1}}`.
  `just` does not pass JSON through intact; call `bash deploy/omni/rig.sh real-stage
  ...` directly for this flag.

A stage that dies while the head is still starting leaves its replica slot stuck: the
head keeps waiting on it. Restart the head. Once the head is ready, stages can come
and go.

`just omni-head <model>` restarts the head for another pipeline; run `just
omni-stop-stages` first and attach new stages after. A stage can be restarted on its
own: the head detaches the old replica and attaches the new one under a fresh replica
id.

## Limits

- **No data plane between stages.** Real stages pass hidden states and codec chunks
  through connectors (shared memory, NIXL, Mooncake). A mock stage neither sends nor
  waits for them, so a downstream stage's timing is independent of the upstream one.
  In an `async_chunk` pipeline a mock code2wav can finish before the mock talker.
- **Content is not modeled.** Audio is silence and images are half noise. Duration,
  size, and count are. The share of noise is set so the head's encode time matches a
  FLUX.2-klein image; another model's images may encode faster or slower.
- **Diffusion replay is timing only.** Recorded errors and aborts are not replayed,
  and the mock stage's concurrency is a flag, not something the trace carries.
- **Video and multi-stage image pipelines are untested.** A diffusion stage answers
  one result per request, which is what the image endpoints use. The diffusion tap
  relays a stage's lifecycle markers and streamed chunks and records the request at
  its finished result. AR-plus-DiT pipelines (HunyuanImage-3, BAGEL) hand data
  between stages through connectors.

## Where the code is

| Path | What |
| --- | --- |
| `crates/sim-protocol/src/omni/attach.rs` | Registration and coordinator status events. |
| `crates/sim-protocol/src/omni/llm.rs` | Request tail decoder, audio output encoder. |
| `crates/sim-protocol/src/omni/diffusion.rs` | Diffusion stage wire format. |
| `src/omni/mod.rs` | `--omni-*` options, stage types, attach and heartbeats. |
| `crates/sim-trace/src/diffusion.rs` | The diffusion trace record, its reader and summary, the latency model, and the replay built from a trace. |
| `src/omni/diffusion.rs` | The mock diffusion stage: queue, knob and trace timing, control API. |
| `crates/sim-tap/src/omni.rs` | The tap's registration and status relay, for either stage kind. |
| `crates/sim-tap/src/diffusion.rs` | The diffusion tap: its data path and what it records. |
| `tests/omni_e2e.rs` | Both stage kinds against a scripted head over real sockets. |
| `tests/omni_tap_e2e.rs` | Scripted head, the tap, and a `play` LLM stage in one chain. |
| `tests/omni_diffusion_tap_e2e.rs` | The same chain for a diffusion stage, byte-exact relay, and replay of the recorded trace. |
| `crates/sim-trace/tests/diffusion_trace.rs` | A trace recorded from a real FLUX.2-klein stage. |
| `deploy/omni/` | The rig, text and image load generators, and the script that dumps ground-truth frames from vLLM-Omni's own encoder. |
