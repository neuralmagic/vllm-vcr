# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Plot the step recorder's GPU captures (docs/trace-replay/step-recorder-report.md).

  step-recorder-ab.png    step gaps at the engine socket, recorder off and on
  step-trace-replay.png   decode-only steps at a fixed concurrency: gaps at the
                          tap, times in the step trace, and replayed gaps
  kv-load-replay.png      captured against replayed remote KV load times, per size

Usage (what `just step-capture-figures <dir>` runs):
  uv run scripts/plot_step_capture.py \
      --off "off, pod 1=off1/tap-trace.jsonl" --off "off, pod 3=off3/tap-trace.jsonl" \
      --on "on, pod 2=on1/tap-trace.jsonl" \
      --step-trace on1/engine-steps.jsonl --tap-trace on1/tap-trace.jsonl \
      --kv-trace kv/decode-steps.jsonl \
      --replay-dir replay --out-dir docs/images

A tap trace spec may end in `:N` to skip its first N requests, for a second
run appended to the same pod's trace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plot_calibration import C_CAP, C_MOD, C_MUT, C_REF, style_axes, survival, title

import matplotlib.pyplot as plt  # isort: skip
from matplotlib.ticker import NullFormatter, ScalarFormatter  # isort: skip

OFF_STYLES = [(C_CAP, "-", 2.6), (C_MUT, "--", 1.8), (C_REF, "-", 1.8)]


def records(path: Path) -> list[dict]:
    lines = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return [record for record in lines if "meta" not in record]


def tap_gaps(spec: str) -> tuple[str, np.ndarray, np.ndarray]:
    """`label=path[:skip]` -> label, step gaps at concurrency 1, at 12 and above."""
    label, _, rest = spec.partition("=")
    path, _, skip = rest.partition(":")
    requests = records(Path(path))[int(skip or 0) :]
    alone = [g for r in requests if r.get("concurrency", 1) <= 1 for g in r.get("itl_ms") or []]
    loaded = [g for r in requests if r.get("concurrency", 1) >= 12 for g in r.get("itl_ms") or []]
    return label, np.array(alone), np.array(loaded)


def decode_only_gaps(requests: list[dict], running: int) -> np.ndarray:
    """Gaps that closed on a step with `running` requests and no prefill work."""
    gaps = []
    for request in requests:
        context = request.get("itl_ctx")
        if not context:
            continue
        for gap, num_running, prefill in zip(
            request.get("itl_ms") or [], context["num_running"], context["prefill_tokens"]
        ):
            if num_running == running and prefill == 0:
                gaps.append(gap)
    return np.array(gaps)


def survival_axes(ax, xlabel: str, heading: str) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(xlabel)
    ax.set_ylabel("P(time > x)")
    title(ax, heading)
    ax.legend()
    style_axes(ax)


def recorder_figure(off: list[str], on: list[str], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    runs = [(tap_gaps(spec), style) for spec, style in zip(off, OFF_STYLES)]
    runs += [(tap_gaps(spec), (C_MOD, "-", 1.8)) for spec in on]
    for (label, alone, loaded), (color, ls, lw) in runs:
        for ax, gaps in zip(axes, (alone, loaded)):
            survival(ax, gaps, f"recorder {label} ({len(gaps):,} steps)", color, lw, ls)
    survival_axes(axes[0], "step gap at the tap (ms)", "Step gap, one request at a time")
    survival_axes(axes[1], "step gap at the tap (ms)", "Step gap, 12 to 16 requests in flight")
    for ax in axes:
        ax.set_xlim(left=8)
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


def step_replay_figure(step_trace: Path, tap_trace: Path, replay_dir: Path, out: Path) -> None:
    lines = map(json.loads, step_trace.read_text().splitlines())
    steps = [record["step"] for record in lines if "step" in record]
    tapped = records(tap_trace)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, running in zip(axes, (1, 16)):
        at_tap = decode_only_gaps(tapped, running)
        in_trace = np.array(
            [s["step_ms"] for s in steps if not s["prefill"] and s["decode"]["reqs"] == running]
        )
        replayed = decode_only_gaps(records(replay_dir / f"replay-c{running}.jsonl"), running)
        survival(ax, at_tap, f"captured, gap at the tap ({len(at_tap):,})", C_CAP, 2.6)
        survival(ax, in_trace, f"captured, step trace ({len(in_trace):,})", C_REF, 1.8, "--")
        survival(ax, replayed, f"replayed, gap at the tap ({len(replayed):,})", C_MOD, 1.8)
        plural = "s" if running > 1 else ""
        survival_axes(ax, "decode-only step (ms)", f"Decode-only steps, {running} request{plural}")
        ax.set_xlim(8, 60)
        ax.set_xticks([10, 12, 15, 20, 30, 50])
        ax.xaxis.set_major_formatter(ScalarFormatter())
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


def loads(path: Path) -> dict[int, list[dict]]:
    by_blocks: dict[int, list[dict]] = {}
    for record in map(json.loads, path.read_text().splitlines()):
        load = record.get("kv_transfer")
        if load and load["direction"] == "recv":
            by_blocks.setdefault(load["blocks"], []).append(load)
    return by_blocks


def kv_figure(kv_trace: Path, replay: Path, out: Path) -> None:
    captured, replayed = loads(kv_trace), loads(replay)
    sizes = sorted(captured)
    fig, axes = plt.subplots(
        1, len(sizes), figsize=(3.4 * len(sizes), 4.4), sharex=True, sharey=True
    )
    for ax, blocks in zip(np.atleast_1d(axes), sizes):
        for label, group, color, lw in (
            ("captured", captured[blocks], C_CAP, 2.6),
            ("replayed", replayed.get(blocks, []), C_MOD, 1.8),
        ):
            failed = sum(1 for load in group if load["outcome"] != "ok")
            note = f"{len(group)} loads" + (f", {failed} failed" if failed else "")
            survival(ax, np.array([load["wall_ms"] for load in group]), f"{label} ({note})", color, lw)
        survival_axes(ax, "remote KV load (ms)", f"{blocks} blocks")
        ax.legend(fontsize=8.5)
    for ax in np.atleast_1d(axes)[1:]:
        ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--off", action="append", default=[], metavar="LABEL=TAP_TRACE[:SKIP]")
    parser.add_argument("--on", action="append", default=[], metavar="LABEL=TAP_TRACE[:SKIP]")
    parser.add_argument("--step-trace", type=Path, help="engine-step trace of a capture")
    parser.add_argument("--tap-trace", type=Path, help="tap trace of the same capture")
    parser.add_argument("--kv-trace", type=Path, help="decode engine's step trace of a P/D capture")
    parser.add_argument("--replay-dir", type=Path, help="VCR_REPLAY_OUT of the two replay recipes")
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.off or args.on:
        recorder_figure(args.off, args.on, args.out_dir / "step-recorder-ab.png")
    if args.step_trace and args.tap_trace and args.replay_dir:
        step_replay_figure(
            args.step_trace,
            args.tap_trace,
            args.replay_dir,
            args.out_dir / "step-trace-replay.png",
        )
    if args.kv_trace and args.replay_dir:
        kv_figure(args.kv_trace, args.replay_dir / "replay-kv.jsonl", args.out_dir / "kv-load-replay.png")


if __name__ == "__main__":
    main()
