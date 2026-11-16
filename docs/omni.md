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

The stage returns flat gray RGB images of the requested size and count. The head
PNG-encodes them without compression, so response size matches a real image of the
same dimensions.

A request for more than 64 megapixels in total is answered with a 400. An aborted
request frees its execution slot and sends nothing, as a real stage does.

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
| `--omni-diffusion-*` | Diffusion stages. See the latency model. |

## Versions

Build `play` for the vLLM line the omni release pins. LLM-stage requests and outputs
are positional tuples, so a mismatched line shifts the omni tail. vLLM-Omni v0.32.0
pins vLLM 0.32.0:

```bash
cargo xtask pin-vllm 0.32
VLLM_TARGET_VERSION=v0.32.0 cargo build --release -p vllm-vcr
```

The diffusion protocol does not depend on the vLLM line.

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

`just omni-head <model>` restarts the head for another pipeline; run `just
omni-stop-stages` first and attach new stages after. A stage can be restarted on its
own: the head detaches the old replica and attaches the new one under a fresh replica
id.

## Limits

- **No data plane between stages.** Real stages pass hidden states and codec chunks
  through connectors (shared memory, NIXL, Mooncake). A mock stage neither sends nor
  waits for them, so a downstream stage's timing is independent of the upstream one.
  In an `async_chunk` pipeline a mock code2wav can finish before the mock talker.
- **Content is not modeled.** Audio is silence and images are flat gray. Duration,
  size, and count are.
- **Diffusion has no trace replay.** The trace schema is token-based. A diffusion
  stage is knob-driven only.
- **Video and multi-stage image pipelines are untested.** A diffusion stage answers
  one result per request, which is what the image endpoints use. AR-plus-DiT pipelines
  (HunyuanImage-3, BAGEL) hand data between stages through connectors.
- **`record` does not attach to omni stages.**
- **No control API on a diffusion stage.** `--control-address` applies to LLM stages.

## Where the code is

| Path | What |
| --- | --- |
| `crates/sim-protocol/src/omni/attach.rs` | Registration and coordinator status events. |
| `crates/sim-protocol/src/omni/llm.rs` | Request tail decoder, audio output encoder. |
| `crates/sim-protocol/src/omni/diffusion.rs` | Diffusion stage wire format. |
| `src/omni/mod.rs` | `--omni-*` options, stage types, attach and heartbeats. |
| `src/omni/diffusion.rs` | The mock diffusion stage and its latency model. |
| `tests/omni_e2e.rs` | Both stage kinds against a scripted head over real sockets. |
| `deploy/omni/` | The rig, and the script that dumps ground-truth frames from the head's encoder. |
