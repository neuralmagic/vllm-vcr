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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=Path, required=True, help="calibrate --dump-samples output")
    p.add_argument("--trace", type=Path, required=True, help="the source trace JSONL")
    p.add_argument("--out-dir", type=Path, default=Path("."))
    args = p.parse_args()

    dump = json.load(open(args.samples))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fidelity_figure(dump, args.out_dir / "replay-fidelity.png")
    mean_vs_pertoken_figure(dump, args.trace, args.out_dir / "mean-vs-pertoken.png")


if __name__ == "__main__":
    main()
