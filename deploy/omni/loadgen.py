"""Drive streaming text chat requests at an omni head and report client-side timing.

Run it in the rig's head container:

    kubectl -n inference-sim exec -i omni-rig -c head -- python3 - --model <model> < deploy/omni/loadgen.py

Prints one JSON object: per-request TTFT, total time, and chunk count, plus their means.
The same prompts are sent in the same order on every run, so a capture and its replay
can be compared request for request.
"""

import argparse
import http.client
import json
import statistics
import threading
import time

PROMPTS = [
    "Explain what a mutex is in two sentences.",
    "List three uses for a paperclip.",
    "Write a haiku about rain.",
    "What is the capital of Australia, and why is it not Sydney?",
    "Summarize how TCP's three-way handshake works.",
    "Give me a recipe for scrambled eggs.",
    "Describe the color blue to someone who cannot see.",
    "Why is the sky dark at night?",
]


def one(host: str, port: int, model: str, prompt: str, max_tokens: int, out: list, index: int) -> None:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "modalities": ["text"],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
        }
    )
    conn = http.client.HTTPConnection(host, port, timeout=300)
    start = time.monotonic()
    conn.request("POST", "/v1/chat/completions", body, {"content-type": "application/json"})
    response = conn.getresponse()
    first = None
    chunks = 0
    for raw in response:
        line = raw.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"):
            continue
        delta = json.loads(line[5:])["choices"][0]["delta"]
        if delta.get("content"):
            chunks += 1
            if first is None:
                first = time.monotonic()
    end = time.monotonic()
    out[index] = {
        "status": response.status,
        "ttft_ms": None if first is None else (first - start) * 1000,
        "total_ms": (end - start) * 1000,
        "chunks": chunks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=96)
    args = parser.parse_args()

    results: list = [None] * args.requests
    gate = threading.Semaphore(args.concurrency)

    def run(index: int) -> None:
        with gate:
            one(args.host, args.port, args.model, PROMPTS[index % len(PROMPTS)], args.max_tokens, results, index)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(args.requests)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    ttfts = [r["ttft_ms"] for r in results if r and r["ttft_ms"] is not None]
    totals = [r["total_ms"] for r in results if r]
    print(
        json.dumps(
            {
                "requests": results,
                "mean_ttft_ms": statistics.mean(ttfts) if ttfts else None,
                "mean_total_ms": statistics.mean(totals) if totals else None,
                "mean_chunks": statistics.mean(r["chunks"] for r in results if r),
            }
        )
    )


if __name__ == "__main__":
    main()
