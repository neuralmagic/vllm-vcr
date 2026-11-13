"""Drive a prefill/decode vLLM pair the way a P/D proxy does, from inside the
pd-kv-capture pod (`just capture-kv-run`).

Each request is a list of random token ids, so its prompt length is exact. The
prefill engine gets it with `do_remote_decode` and one output token; the decode
engine gets it again with the `kv_transfer_params` the prefill returned.

    ok             P then D, at each prompt size, sequentially and four at a time
    failed-engine  D is handed transfer params naming an engine it has never met,
                   at a closed side channel port, so the handshake fails
    failed-blocks  D is handed transfer params naming blocks the prefill lacks
    expired        P only; the decode never comes, so the prefill's lease runs out

Writes the per-request results and both engines' /metrics before and after.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

MODEL = "Qwen/Qwen3-8B"


def post(url: str, body: dict[str, Any], request_id: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{url}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-Request-Id": request_id},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        return {"error": error.code, "detail": error.read().decode()[:500]}


def metrics(url: str) -> str:
    with urllib.request.urlopen(f"{url}/metrics", timeout=30) as response:
        return response.read().decode()


def prefill(url: str, tokens: list[int], request_id: str) -> dict[str, Any]:
    return post(
        url,
        {
            "model": MODEL,
            "prompt": tokens,
            "max_tokens": 1,
            "stream": False,
            "kv_transfer_params": {
                "do_remote_decode": True,
                "do_remote_prefill": False,
                "remote_engine_id": None,
                "remote_block_ids": None,
                "remote_host": None,
                "remote_port": None,
            },
        },
        request_id,
    )


def decode(
    url: str, tokens: list[int], request_id: str, params: dict[str, Any], max_tokens: int
) -> dict[str, Any]:
    return post(
        url,
        {
            "model": MODEL,
            "prompt": tokens,
            "max_tokens": max_tokens,
            "stream": False,
            "kv_transfer_params": params,
        },
        request_id,
    )


def one(args: argparse.Namespace, phase: str, tokens: list[int]) -> dict[str, Any]:
    request_id = uuid.uuid4().hex
    result: dict[str, Any] = {"phase": phase, "prompt_tokens": len(tokens)}
    start = time.perf_counter()
    prefilled = prefill(args.prefill, tokens, request_id)
    result["prefill_s"] = round(time.perf_counter() - start, 4)
    params = prefilled.get("kv_transfer_params")
    result["prefill_error"] = prefilled.get("error")
    result["remote_blocks"] = _num_blocks(params)
    if phase == "expired" or params is None:
        return result
    if phase == "failed-engine":
        params = dict(params, remote_engine_id=str(uuid.uuid4()), remote_port=1)
    elif phase == "failed-blocks":
        params = dict(params, remote_block_ids=_shift(params["remote_block_ids"], 10**9))
    start = time.perf_counter()
    decoded = decode(args.decode, tokens, request_id, params, args.max_tokens)
    result["decode_s"] = round(time.perf_counter() - start, 4)
    result["decode_error"] = decoded.get("error")
    result["decode_detail"] = decoded.get("detail")
    usage = decoded.get("usage") or {}
    result["completion_tokens"] = usage.get("completion_tokens")
    result["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return result


def _num_blocks(params: dict[str, Any] | None) -> int | None:
    blocks = (params or {}).get("remote_block_ids")
    if not blocks:
        return None
    if isinstance(blocks[0], list):
        return len(blocks[0])
    return len(blocks)


def _shift(blocks: list[Any], by: int) -> list[Any]:
    return [_shift(b, by) if isinstance(b, list) else b + by for b in blocks]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prefill", default="http://localhost:8100")
    parser.add_argument("--decode", default="http://localhost:8200")
    parser.add_argument("--sizes", default="256,1024,4096,8192")
    parser.add_argument("--per-size", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--failures", type=int, default=3, help="per failure mode")
    parser.add_argument("--failure-modes", default="engine,blocks")
    parser.add_argument("--expiries", type=int, default=5)
    parser.add_argument("--lease-wait", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("/trace"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    sizes = [int(size) for size in args.sizes.split(",")]

    def prompt(size: int) -> list[int]:
        return [rng.randrange(1000, 100000) for _ in range(size)]

    for name, url in (("prefill", args.prefill), ("decode", args.decode)):
        (args.out / f"metrics-{name}-before.txt").write_text(metrics(url))

    results: list[dict[str, Any]] = []
    for size in sizes:
        for _ in range(args.per_size):
            results.append(one(args, "ok", prompt(size)))
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [prompt(size) for size in sizes for _ in range(args.per_size)]
        results.extend(pool.map(lambda tokens: one(args, "ok-concurrent", tokens), jobs))
    for mode in filter(None, args.failure_modes.split(",")):
        for _ in range(args.failures):
            results.append(one(args, f"failed-{mode}", prompt(sizes[1])))
    for _ in range(args.expiries):
        results.append(one(args, "expired", prompt(sizes[1])))
    time.sleep(args.lease_wait)

    for name, url in (("prefill", args.prefill), ("decode", args.decode)):
        (args.out / f"metrics-{name}-after.txt").write_text(metrics(url))
    (args.out / "pd-results.json").write_text(json.dumps(results, indent=1))
    errors = sum(1 for r in results if r.get("prefill_error") or r.get("decode_error"))
    print(f"{len(results)} requests, {errors} with an HTTP error")


if __name__ == "__main__":
    main()
