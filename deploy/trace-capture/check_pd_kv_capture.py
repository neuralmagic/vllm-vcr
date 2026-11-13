"""Compare the step recorder's KV transfer records from a P/D capture with what
the driver and vLLM's own NIXL metrics saw (`just check-kv-capture <dir>`).

Reads the files `just capture-kv-run` fetches: both engines' step traces, the
driver's per-request results, and /metrics before and after the load.
"""

from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def trace(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    steps, transfers = [], []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if "step" in record:
            steps.append(record["step"])
        elif "kv_transfer" in record:
            transfers.append(record["kv_transfer"])
    return steps, transfers


def metrics(path: Path) -> dict[str, float]:
    """Sample name -> value summed over label sets."""
    totals: dict[str, float] = collections.defaultdict(float)
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name_and_labels, _, value = line.rpartition(" ")
        if name_and_labels.endswith("}") and 'le="' in name_and_labels:
            continue
        totals[name_and_labels.partition("{")[0]] += float(value)
    return totals


def delta(directory: Path, engine: str) -> dict[str, float]:
    before = metrics(directory / f"metrics-{engine}-before.txt")
    after = metrics(directory / f"metrics-{engine}-after.txt")
    return {name: value - before.get(name, 0.0) for name, value in after.items()}


def nixl(values: dict[str, float]) -> dict[str, float]:
    return {
        name.removeprefix("vllm:nixl_"): value
        for name, value in sorted(values.items())
        if name.startswith("vllm:nixl_") and not name.endswith("_created")
    }


def main() -> None:
    directory = Path(sys.argv[1])
    results = json.loads((directory / "pd-results.json").read_text())
    _, decode_transfers = trace(directory / "decode-steps.jsonl")
    _, prefill_transfers = trace(directory / "prefill-steps.jsonl")
    decode_nixl = nixl(delta(directory, "decode"))
    prefill_nixl = nixl(delta(directory, "prefill"))

    by_phase = collections.Counter(result["phase"] for result in results)
    errors = [r for r in results if r.get("prefill_error") or r.get("decode_error")]
    print(f"driver: {dict(by_phase)}, {len(errors)} HTTP errors")

    recv = [t for t in decode_transfers if t["direction"] == "recv"]
    outcomes = collections.Counter(t["outcome"] for t in recv)
    print(f"decode trace: {len(recv)} recv records, outcomes {dict(outcomes)}")
    print(f"decode vLLM NIXL metrics (delta): {decode_nixl}")

    ok = [t for t in recv if t["outcome"] == "ok"]
    decoded = [r for r in results if r["phase"] != "expired"]
    loaded = [r for r in decoded if not r["phase"].startswith("failed-")]
    broken = [r for r in decoded if r["phase"].startswith("failed-")]
    print(
        f"loads: trace {len(ok)} ok and {len(recv) - len(ok)} not ok; "
        f"driver {len(loaded)} expected to load and {len(broken)} expected to fail"
    )
    want_blocks = sorted(r["remote_blocks"] or 0 for r in loaded)
    got_blocks = sorted(t["blocks"] for t in ok)
    print(
        f"blocks per load: trace total {sum(got_blocks)} over {len(got_blocks)} loads, "
        f"driver total {sum(want_blocks)} over {len(want_blocks)} requests, "
        f"same multiset: {got_blocks == want_blocks}"
    )
    for transfer in recv:
        if transfer["outcome"] != "ok":
            print(
                f"  {transfer['outcome']}: {transfer['blocks']} blocks, "
                f"wall_ms {transfer['wall_ms']:.2f}"
            )
    floors = sorted(r["prompt_tokens"] // 16 for r in loaded)
    print(f"blocks equal prompt_tokens // 16 for every load: {got_blocks == floors}")
    total_bytes = decode_nixl.get("bytes_transferred_sum", 0.0)
    if got_blocks and total_bytes:
        print(
            f"bytes: vLLM {total_bytes:.0f} over {sum(got_blocks)} traced blocks = "
            f"{total_bytes / sum(got_blocks):.1f} bytes per block"
        )

    by_size: dict[int, list[float]] = collections.defaultdict(list)
    for transfer in ok:
        by_size[transfer["blocks"]].append(transfer["wall_ms"])
    for blocks, walls in sorted(by_size.items()):
        print(
            f"  {blocks:5d} blocks: {len(walls):3d} loads, wall_ms median "
            f"{statistics.median(walls):8.2f}, min {min(walls):8.2f}, max {max(walls):8.2f}"
        )
    count = decode_nixl.get("xfer_time_seconds_count", 0.0)
    if count and ok:
        print(
            f"time: trace mean wall_ms {statistics.fmean(t['wall_ms'] for t in ok):.2f} "
            f"over {len(ok)} loads; vLLM mean transfer "
            f"{1000 * decode_nixl['xfer_time_seconds_sum'] / count:.2f} ms and post "
            f"{1000 * decode_nixl.get('post_time_seconds_sum', 0.0) / count:.2f} ms "
            f"over {count:.0f} transfers"
        )

    sends = [t for t in prefill_transfers if t["direction"] == "send"]
    print(
        f"prefill trace: {len(sends)} send records for {len(results)} prefill requests; "
        f"prefill vLLM NIXL metrics (delta): {prefill_nixl}"
    )


if __name__ == "__main__":
    main()
