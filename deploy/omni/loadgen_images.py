"""Drive image generation and edit requests at an omni head and report client-side timing.

Run it in the rig's head container:

    kubectl -n inference-sim exec -i omni-rig -c head -- python3 - --model <model> < deploy/omni/loadgen_images.py

Prints one JSON object: per-request status, total time, and response size, plus the mean
total time per request shape. The same requests are sent in the same order on every run,
so a capture and its replay can be compared request for request.
"""

import argparse
import http.client
import io
import json
import statistics
import threading
import time
import uuid

# (endpoint, size, num_inference_steps, n)
WORKLOAD = [
    ("generations", "512x512", 4, 1),
    ("generations", "1024x1024", 4, 1),
    ("generations", "768x512", 8, 1),
    ("edits", "512x512", 4, 1),
    ("generations", "1024x1024", 8, 1),
    ("generations", "512x512", 4, 2),
    ("edits", "1024x1024", 4, 1),
    ("generations", "512x512", 8, 1),
]
PROMPT = "a red fox in fresh snow, golden hour"
EDIT_PROMPT = "make it night, with stars"


def input_png() -> bytes:
    from PIL import Image

    image = Image.new("RGB", (512, 512))
    image.putdata([(x % 256, y % 256, (x + y) % 256) for y in range(512) for x in range(512)])
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def multipart(fields: dict, png: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="input.png"\r\n'
        f"Content-Type: image/png\r\n\r\n".encode()
        + png
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def one(host: str, port: int, model: str, case: tuple, png: bytes, out: list, index: int) -> None:
    endpoint, size, steps, n = case
    if endpoint == "edits":
        body, content_type = multipart(
            {"model": model, "prompt": EDIT_PROMPT, "size": size, "n": n, "num_inference_steps": steps, "seed": 7},
            png,
        )
    else:
        body = json.dumps(
            {"model": model, "prompt": PROMPT, "size": size, "n": n, "num_inference_steps": steps, "seed": 7}
        ).encode()
        content_type = "application/json"
    conn = http.client.HTTPConnection(host, port, timeout=600)
    start = time.monotonic()
    conn.request("POST", f"/v1/images/{endpoint}", body, {"content-type": content_type})
    response = conn.getresponse()
    payload = response.read()
    end = time.monotonic()
    images = len(json.loads(payload).get("data", [])) if response.status == 200 else 0
    out[index] = {
        "shape": f"{endpoint} {size} steps={steps} n={n}",
        "status": response.status,
        "total_ms": (end - start) * 1000,
        "images": images,
        "bytes": len(payload),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()

    png = input_png()
    results: list = [None] * args.requests
    gate = threading.Semaphore(args.concurrency)

    def run(index: int) -> None:
        with gate:
            one(args.host, args.port, args.model, WORKLOAD[index % len(WORKLOAD)], png, results, index)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(args.requests)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    by_shape: dict = {}
    for result in results:
        if result and result["status"] == 200:
            by_shape.setdefault(result["shape"], []).append(result["total_ms"])
    totals = [r["total_ms"] for r in results if r and r["status"] == 200]
    print(
        json.dumps(
            {
                "requests": results,
                "mean_total_ms_by_shape": {shape: statistics.mean(times) for shape, times in by_shape.items()},
                "mean_total_ms": statistics.mean(totals) if totals else None,
                "failed": sum(1 for r in results if not r or r["status"] != 200),
            }
        )
    )


if __name__ == "__main__":
    main()
