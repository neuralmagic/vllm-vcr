# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Plot calibration sample dumps produced by `inference-sim-trace calibrate --dump-samples`.

Produces two figures:
  replay-fidelity.png   source vs replay vs knob-fit: survival curves + Q-Q plots
  mean-vs-pertoken.png  per-token ITL vs per-request mean ITL from the trace itself

Usage:
  uv run scripts/plot_calibration.py \
      --samples /tmp/calib-samples.json \
      --trace /tmp/trace.jsonl \
      --out-dir docs/images
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

C_SRC, C_REP, C_KNB = "#222222", "#d62728", "#1f77b4"


def survival(ax: Axes, data: np.ndarray, label: str, color: str, lw: float, ls: str = "-") -> None:
    x = np.sort(data)
    y = np.maximum(1.0 - np.arange(1, len(x) + 1) / len(x), 1.0 / len(x))
    ax.step(x, y, where="post", label=label, color=color, lw=lw, ls=ls)


def qq(ax: Axes, src: np.ndarray, rep: np.ndarray, color: str) -> None:
    qs = np.linspace(0.001, 0.999, 400)
    ax.plot(np.quantile(src, qs), np.quantile(rep, qs), ".", ms=3, color=color)
    lim = [
        min(src.min(), rep.min()) * 0.9,
        max(np.quantile(src, 0.999), np.quantile(rep, 0.999)) * 1.1,
    ]
    ax.plot(lim, lim, "--", color="#999999", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)


def fidelity_figure(dump: dict, out: Path) -> None:
    src_t, src_i = np.array(dump["source"]["ttft_ms"]), np.array(dump["source"]["itl_ms"])
    rep_t, rep_i = np.array(dump["replay"]["ttft_ms"]), np.array(dump["replay"]["itl_ms"])
    knb_t, knb_i = np.array(dump["knobfit"]["ttft_ms"]), np.array(dump["knobfit"]["itl_ms"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0][0]
    survival(ax, src_i, "source (per-token)", C_SRC, 2.6)
    survival(ax, rep_i, "trace replay", C_REP, 1.6)
    survival(ax, knb_i, "knob-fit", C_KNB, 1.4, "--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("inter-token latency (ms)")
    ax.set_ylabel("P(ITL > x)")
    ax.set_title("ITL survival")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    ax = axes[0][1]
    survival(ax, src_t, "source", C_SRC, 2.6)
    survival(ax, rep_t, "trace replay", C_REP, 1.6)
    survival(ax, knb_t, "knob-fit", C_KNB, 1.4, "--")
    ax.set_yscale("log")
    ax.set_xlabel("TTFT (ms)")
    ax.set_ylabel("P(TTFT > x)")
    ax.set_title("TTFT survival")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    ax = axes[1][0]
    qq(ax, src_i, rep_i, C_REP)
    ax.set_xlabel("source ITL quantiles (ms)")
    ax.set_ylabel("replay ITL quantiles (ms)")
    ax.set_title("Q-Q: ITL")
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    qq(ax, src_t, rep_t, C_REP)
    ax.set_xlabel("source TTFT quantiles (ms)")
    ax.set_ylabel("replay TTFT quantiles (ms)")
    ax.set_title("Q-Q: TTFT")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def mean_vs_pertoken_figure(dump: dict, trace_path: Path, out: Path) -> None:
    src_i = np.array(dump["source"]["itl_ms"])
    per_req_means = []
    with open(trace_path) as f:
        for line in f:
            r = json.loads(line)
            if "meta" in r:
                continue
            itls = r.get("itl_ms")
            if itls:
                per_req_means.append(sum(itls) / len(itls))
    if not per_req_means:
        print(f"skipping {out}: trace has no per-token itl_ms arrays")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    survival(ax, src_i, "per-token ITL", C_SRC, 2.6)
    survival(ax, np.array(per_req_means), "per-request mean ITL", C_KNB, 2.0)
    # Sub-ms outliers (chunked finish gaps) would drag the log axis into empty
    # space; anchor the left edge near the bulk of the distribution instead.
    left = float(np.quantile(src_i, 0.01)) * 0.5
    p999 = float(np.quantile(src_i, 0.999))
    ax.axvline(p999, color="#d62728", ls=":", lw=1)
    ax.annotate(
        f"per-token p99.9 = {p999:.1f} ms",
        xy=(p999, 1.5e-3),
        xytext=(left * 3.0, 1e-4),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "#d62728"},
        color="#d62728",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left, src_i.max() * 1.5)
    ax.set_xlabel("inter-token latency (ms)")
    ax.set_ylabel("P(ITL > x)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def load_trace_samples(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Pool TTFT and per-token ITL samples from a trace JSONL."""
    ttfts: list[float] = []
    itls: list[float] = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "meta" in r:
                continue
            ttfts.append(r["ttft_ms"])
            if r.get("itl_ms"):
                itls.extend(r["itl_ms"])
            elif r.get("itl_summary"):
                itls.extend([r["itl_summary"]["mean_ms"]] * r["itl_summary"]["count"])
    return np.array(ttfts), np.array(itls)


def comparison_figure(traces: list[tuple[str, Path]], out: Path) -> None:
    """Survival curves for several traces on shared axes: one labeled trace per curve."""
    palette = [C_SRC, C_REP, C_KNB, "#2ca02c", "#9467bd"]
    fig, (ax_itl, ax_ttft) = plt.subplots(1, 2, figsize=(13, 5))

    for (label, path), color in zip(traces, palette):
        ttfts, itls = load_trace_samples(path)
        lw = 2.6 if color == C_SRC else 1.8
        survival(ax_itl, itls, label, color, lw)
        survival(ax_ttft, ttfts, label, color, lw)

    ax_itl.set_xscale("log")
    ax_itl.set_yscale("log")
    ax_itl.set_xlabel("inter-token latency (ms)")
    ax_itl.set_ylabel("P(ITL > x)")
    ax_itl.set_title("ITL survival")
    ax_itl.legend()
    ax_itl.grid(alpha=0.3, which="both")

    ax_ttft.set_yscale("log")
    ax_ttft.set_xlabel("TTFT (ms)")
    ax_ttft.set_ylabel("P(TTFT > x)")
    ax_ttft.set_title("TTFT survival")
    ax_ttft.legend()
    ax_ttft.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def load_records(path: Path) -> list[dict]:
    """All non-meta records of a trace JSONL, as dicts."""
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "meta" in r:
                continue
            records.append(r)
    return records


def turn_depths(records: list[dict]) -> list[int]:
    """Session-turn depth per record (1 = session root), inferred from chained
    block_hashes the same way the replay harness infers sessions: a record's
    parent is the most recent earlier record whose full hash chain is a proper
    prefix of its own chain."""
    by_last_hash: dict[int, int] = {}
    depths: list[int] = []
    for i, r in enumerate(records):
        chain = r.get("block_hashes") or []
        depth = 1
        for k in range(len(chain) - 1, 0, -1):
            p = by_last_hash.get(chain[k - 1])
            if p is not None and len(records[p].get("block_hashes") or []) == k:
                depth = depths[p] + 1
                break
        if chain:
            by_last_hash[chain[-1]] = i
        depths.append(depth)
    return depths


def cache_effect_figure(traces: list[tuple[str, Path]], out: Path) -> None:
    """Prove the prefix-cache effect is reproduced shape-wise, not just in the
    pooled marginal: cohort requests by session-turn depth (turn 1 hits only
    the shared prefix; deeper turns hit their session's growing context) and
    overlay real vs replay TTFT survival per cohort. Compensating errors
    between cohorts would show here and not in the pooled curve. An optional
    nocache=... trace adds the cache-off what-if for magnitude."""
    by_label = dict(traces)
    src = load_records(by_label["real"])
    src.sort(key=lambda r: r["arrival_ms"])
    depths = turn_depths(src)

    def by_arrival(label: str) -> dict[float, dict]:
        return {round(r["arrival_ms"], 3): r for r in load_records(by_label[label])}

    rep = by_arrival("replay")
    cold = by_arrival("nocache") if "nocache" in by_label else {}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cohorts = [
        ("turn 1 (shared-prefix hit)", lambda d: d == 1, axes[0]),
        ("turns 2+ (growing-context hit)", lambda d: d >= 2, axes[1]),
    ]
    for title, pred, ax in cohorts:
        keys = [round(r["arrival_ms"], 3) for r, d in zip(src, depths) if pred(d)]
        real = np.array([r["ttft_ms"] for r, d in zip(src, depths) if pred(d)])
        replay = np.array([rep[k]["ttft_ms"] for k in keys if k in rep])
        survival(ax, real, "real", C_SRC, 2.6)
        survival(ax, replay, "replay", C_REP, 1.8)
        if cold:
            nocache = np.array([cold[k]["ttft_ms"] for k in keys if k in cold])
            survival(ax, nocache, "no prefix cache (what-if)", C_KNB, 1.8, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("TTFT (ms)")
        ax.set_ylabel("P(TTFT > x)")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def parse_labeled_trace(spec: str) -> tuple[str, Path]:
    label, _, path = spec.partition("=")
    if not path:
        raise argparse.ArgumentTypeError(f"expected LABEL=PATH, got {spec!r}")
    return label, Path(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=Path, help="calibrate --dump-samples output")
    p.add_argument("--trace", type=Path, help="the source trace JSONL")
    p.add_argument(
        "--compare",
        type=parse_labeled_trace,
        action="append",
        metavar="LABEL=PATH",
        help="repeatable; plot survival curves of several traces on shared axes "
        "(first entry is drawn as the reference)",
    )
    p.add_argument(
        "--cache-effect",
        type=parse_labeled_trace,
        action="append",
        metavar="LABEL=PATH",
        help="repeatable; per-turn-cohort TTFT survival from real=trace.jsonl, "
        "replay=measured.jsonl, and optional nocache=measured.jsonl (traces "
        "are joined on arrival_ms; real needs block_hashes)",
    )
    p.add_argument("--out-dir", type=Path, default=Path("."))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.samples and args.trace:
        dump = json.load(open(args.samples))
        fidelity_figure(dump, args.out_dir / "replay-fidelity.png")
        mean_vs_pertoken_figure(dump, args.trace, args.out_dir / "mean-vs-pertoken.png")
    if args.compare:
        comparison_figure(args.compare, args.out_dir / "sim-comparison.png")
    if args.cache_effect:
        labels = {label for label, _ in args.cache_effect}
        if not {"real", "replay"} <= labels:
            p.error("--cache-effect needs at least real=... and replay=...")
        cache_effect_figure(args.cache_effect, args.out_dir / "cache-effect.png")
    if not (args.samples and args.trace) and not args.compare and not args.cache_effect:
        p.error("nothing to do: pass --samples/--trace, --compare, and/or --cache-effect")


if __name__ == "__main__":
    main()
