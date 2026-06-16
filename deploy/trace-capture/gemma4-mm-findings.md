# gemma-4 multimodal trace-capture: findings

Date: 2026-06-16. Cluster: coreweave-waldorf, ns `weaton-dev`.
Stack: vLLM main `16e91176` (the DiffusionGemma rev), Python frontend + headless
engine, recording tap `tap-16e91176-mmfix`.

## TL;DR

Captured a real, **coherent** image-in/text-out multimodal trace from
`google/gemma-4-E4B-it` end to end through the tap and replayed it byte-identically
(`tests/gemma4_replay_demo.rs`, fixture `tests/fixtures/gemma4_mm_trace.jsonl`). The
pipeline works with **zero new infra**: no vLLM rev bump, no tap rebuild.

The first target, `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic`, loads and runs but
emits **gibberish** image output at every shippable vLLM rev, due to open upstream
bug [vllm#40106](https://github.com/vllm-project/vllm/issues/40106). E4B is the same
arch but does NOT carry the flag that triggers it, so it produces coherent output
and is the canonical capture here.

## What differed from the DiffusionGemma run

The DiffusionGemma findings (`diffusiongemma-mm-frame-findings.md`) anticipated a
"central setup task": pick a newer vLLM rev for gemma-4 and rebuild the tap against
its protocol. **Both evaporated** for the whole gemma-4 family.

| | DiffusionGemma | gemma-4 (26B / E4B) |
|---|---|---|
| Engine arch | `diffusiongemma` (diffusion) | `Gemma4ForConditionalGeneration` -> `gemma4_mm` (autoregressive) |
| vLLM rev | main `16e91176` | **same `16e91176`** (already registers `gemma4_mm`; models predate the image) |
| Tap rebuild | (built at 16e91176 + mmfix) | **none** — protocol crate unchanged, reuse `tap-16e91176-mmfix` |
| V2 model runner | required (`VLLM_USE_V2_MODEL_RUNNER=1`) | **not needed** — standard runner |
| Output ITL | diffusion blocks (`itl_tokens` > 1/step) | **plain per-token** (`itl_tokens` absent) |
| Diffusion flags | `--diffusion-config`, entropy `--hf-overrides`, `--max-num-seqs=4`, `max_soft_tokens` | **all dropped** |
| Prompt shape | ~259 (text + 256 image placeholders) | ~271-285 (text + **280** vision soft-tokens) |

So each gemma-4 capture deployment is the DiffusionGemma one minus every diffusion knob: headless
engine + Python frontend (dpl=0, on a GPU for platform init) + the same mmfix tap,
its own PVC, image-only `--limit-mm-per-prompt`. Bring-up was clean: the tap
brokered the frontend<->engine handshake first try, no restarts (the DiffusionGemma
handshake/wedge fragility did NOT recur — the two tap fixes hold).

Inline-base64 images only (frontend server-side URL fetch 403s through istio
egress; `g4-image-smoke.sh` builds a PNG with stdlib zlib, no PIL).

## The 26B gibberish: upstream bug #40106 (NOT ours)

`RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic`: text-only chat is coherent ("The capital
of France is Paris."), but every image request returns token salad ("The own
discriminant own fact이나 own这意味着...", heavy repetition of token id 1852).
Tested a degenerate 64x64 solid-red PNG and a structured 256x256 four-quadrant
image; both gibberish, so it is not the input.

Root cause, confirmed against config + tracker: the 26B's `text_config.
use_bidirectional_attention = "vision"`, and it is one of the two models named in
open bug [vllm#40106](https://github.com/vllm-project/vllm/issues/40106): vLLM
**silently ignores that flag and runs standard causal attention over vision
tokens** instead of the bidirectional mask HF applies (`create_causal_mask_mapping`).
The issue reports KL divergence 0.03-0.09/token concentrated on image positions vs
~0.0004 for non-bidirectional variants; at greedy decode over tens of tokens that
compounds into the degenerate output observed. Text-only is unaffected (no vision
tokens). **No fix PR is merged**, so no newer rev fixes the 26B.

## E4B: the coherent gemma-4 multimodal model

`google/gemma-4-E4B-it` is the same arch (`Gemma4ForConditionalGeneration` ->
`gemma4_mm`, so supported at 16e91176, no rebuild) but a dense ~4B-effective omni
variant. Its config has `use_bidirectional_attention: null` (NOT `"vision"`), so the
#40106 trigger is absent. It is anonymously pullable (gated: none), so no HF_TOKEN
(the namespace hf-token-secret is expired anyway). vision soft-tokens=280,
image_token_id=258880.

Image output is coherent and correct. The four-quadrant smoke:

```
> Describe the colors and their positions in this image.
This image is divided into four equal quadrants, each with a distinct color.
*   Top-Left Quadrant: Red
*   Top-Right Quadrant: Green
*   Bottom-Left Quadrant: Blue
*   Bottom-Right Quadrant: Yellow
```

Bring-up was fast (<2 min pull + ~7 s engine warmup). Captured 6 multimodal records
(prompt ~271-281 = text + 280 vision soft-tokens, output 32-128, mixed stop/length).
Saved artifacts: `/tmp/e4b-mm-trace/{trace.jsonl,step-stats.jsonl}`; committed subset
`tests/fixtures/gemma4_mm_trace.jsonl` (4 records: cold-start/stop + 32/64/128
length).

(`RedHatAI/gemma-3-27b-it-FP8-dynamic` was also verified coherent on the same
deployment — gemma-3's mature MM path is unaffected by #40106 — and is a drop-in
alternative if a bigger/dense MM model is wanted. Manifest: `gemma3-capture.yaml`.)

## Replay validation

Replay fidelity does not depend on output coherence: the tap records the engine's
REAL token ids + timing; the sim replays them verbatim. Validated on the E4B trace:

- **Byte-identical content replay PASS.** `cargo test --test gemma4_replay_demo`
  (committed E4B fixture) and `REPLAY_TRACE=/tmp/e4b-mm-trace/trace.jsonl cargo test
  --test real_trace_replay` (full capture): all streams reproduce the recorded
  `output_token_ids` byte-for-byte with matching finish reasons.
- **Per-token ITL replay near-exact.** `inference-sim-trace calibrate`: ITL max
  relative error 0.0022; request-total 0.0185 (within tolerance).
- **TTFT calibrate "FAIL" (~0.50) is a capture artifact, not a sim gap.** Only the
  first-ever image request paid mm-processor warmup (TTFT ~14.5 s); every
  steady-state request is 45-67 ms. With 6 records the p90/p99 TTFT quantile is
  dominated by that single cold-start sample. Steady-state TTFT and all ITL replay
  cleanly. (The 26B garbage trace replays byte-identically too, with the same TTFT
  cold-start artifact — fidelity is independent of coherence.)

## Reproduce (E4B)

```bash
kubectl apply -f deploy/trace-capture/gemma4-e4b-model-cache-pvc.yaml
kubectl apply -f deploy/trace-capture/gemma4-e4b-capture.yaml
kubectl -n weaton-dev rollout status deploy/trace-capture-gemma4-e4b   # <2 min first pull
# inline-image smoke (expect a coherent caption):
kubectl -n weaton-dev port-forward deploy/trace-capture-gemma4-e4b 8000:8000 &
# ... POST /v1/chat/completions with an inline data: image_url, max_tokens set ...
kubectl -n weaton-dev cp -c tap <pod>:/trace/trace.jsonl ./trace.jsonl
kubectl -n weaton-dev scale deploy/trace-capture-gemma4-e4b --replicas=0
cargo test --test gemma4_replay_demo
```

Capture deployments: `gemma4-e4b-capture.yaml` (E4B, coherent — canonical),
`gemma4-capture.yaml` (26B, gibberish — #40106 repro),
`gemma3-capture.yaml` (gemma-3-27b, coherent alternative).
