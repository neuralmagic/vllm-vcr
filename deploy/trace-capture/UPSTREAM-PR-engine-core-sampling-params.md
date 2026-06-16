# Upstream PR draft: `EngineCoreSamplingParams` must default omittable fields

Target: `vllm-project/vllm`, path `rust/src/engine-core-client/src/protocol/mod.rs`.
Patch: `engine-core-sampling-params-serde-default.patch` (in this dir).

## Title

`[rust] EngineCoreSamplingParams: add serde defaults for omit_defaults fields`

## Problem

Python `SamplingParams` is `msgspec.Struct(..., omit_defaults=True)`, so it
serializes to a map that **omits every field sitting at its default value**. A
typical request's `sampling_params` on the wire is just:

```
{stop, stop_token_ids, bad_words, skip_reading_prefix_cache}
```

The Rust `EngineCoreSamplingParams` (a map-decoded struct) carries
`#[serde(default)]` on only a subset of fields, so `decode_msgpack` of a real
request fails on the first omitted-but-required field:

```
Decode { target_type: "EngineCoreRequest", message: "missing field `temperature`" }
```

Any request that leaves a sampling field at its default (i.e. essentially all of
them) fails to decode. The crate's own `python_compat` test does not catch this
because the request fixtures it round-trips are encoded by the *Rust* serializer
(which writes every field); only Python's `omit_defaults` produces the short map.

## Fix

Add `#[serde(default)]` to every omittable field, with default fns for the
non-zero Python defaults (`temperature` = 1.0, `max_tokens` = 16). Fields covered:
`temperature`, `seed`, `max_tokens`, `logprobs`, `prompt_logprobs`,
`frequency_penalty`, `presence_penalty`, `stop_token_ids`, `_eos_token_id`,
`_all_stop_token_ids`. (`top_p`, `top_k`, `min_p`, `min_tokens`,
`repetition_penalty`, and the rest already had defaults.)

Python defaults matched: temperature 1.0, top_p 1.0, top_k 0, min_p 0.0,
presence/frequency_penalty 0.0, repetition_penalty 1.0, seed None, max_tokens 16,
min_tokens 0, stop_token_ids None/[].

## Suggested test addition

Extend `python_compat.py` with a request whose `SamplingParams` is left at
defaults (so msgspec omits the fields) and assert the Rust decode succeeds. A
captured fixture lives at
`inference-simulator-rs/crates/sim-tap/tests/fixtures/mm_request_*.hex` (real
`MsgpackEncoder` output for an image request) and the failing/passing decode is
exercised by `crates/sim-tap/tests/mm_decode_groundtruth.rs`.
