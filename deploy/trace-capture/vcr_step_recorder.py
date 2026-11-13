"""Record what vLLM's model runner and KV connector did, one JSONL line per
executed step and per KV transfer, for `vllm-vcr play --step-trace`.

Enable it on the engine process, with this file on PYTHONPATH:

    VCR_STEP_TRACE_OUT=/trace/engine-steps.jsonl \\
    vllm serve ... --scheduler-cls vcr_step_recorder.recording_scheduler

`recording_scheduler` builds the scheduler vLLM would have built without the
flag (sync, async, diffusion, or the Rust core) as a subclass that overrides
`schedule` and `update_from_output`:

    engine thread                                  drain thread
    ─────────────                                  ────────────
    schedule()            2 clock reads, 1 append  every 50 ms, off a timer:
    execute_model()       blocks on the runner       step shapes, KV transfers,
    update_from_output()  2 clock reads, 1 append    JSON, file writes, giving
                                                     up the GIL after each item

The file format is `crates/sim-trace/src/steps.rs`. A `{pid}` in the path is
replaced by the process id, for engines that run more than one core process.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
from collections import deque
from typing import IO, Any

logger = logging.getLogger("vllm.vcr_step_recorder")

OUT_ENV = "VCR_STEP_TRACE_OUT"
CONFIG_HASH_ENV = "VCR_CONFIG_HASH"

_SCHEDULE = 0
_UPDATE = 1
_BACKLOG_LIMIT = 65536
_DRAIN_INTERVAL_S = 0.05


def _flat_block_ids(block_ids: Any) -> list[int]:
    """Block ids of a connector's `BlockIds`, one list per KV cache group or flat."""
    flat: list[int] = []
    for item in block_ids or ():
        if isinstance(item, int):
            flat.append(item)
        else:
            flat.extend(item)
    return flat


def _connector_metadata(metadata: Any) -> list[Any]:
    """`metadata` itself, or the per-connector metadata a MultiConnector wraps."""
    if metadata is None:
        return []
    nested = getattr(metadata, "metadata", None)
    if isinstance(nested, (list, tuple)):
        return [inner for item in nested for inner in _connector_metadata(item)]
    return [metadata]


class StepRecorder:
    """Turns the scheduler's inputs and outputs into step trace lines."""

    def __init__(self, out: IO[str], meta: dict[str, Any]) -> None:
        self.items: deque[tuple[Any, ...]] = deque(maxlen=_BACKLOG_LIMIT)
        self._stop = threading.Event()
        self._out = out
        self._start_ns = time.perf_counter_ns()
        self._step = 0
        self._pending: dict[int, tuple[int, int, int, Any]] = {}
        self._last_update_entry_ns = self._start_ns
        self._last_update_exit_ns = self._start_ns
        self._prefilling: set[str] = set()
        self._loads: dict[str, tuple[int, int, list[int]]] = {}
        self._sends: dict[str, tuple[int, int]] = {}
        self._max_backlog = 0
        self._closed = False
        self._write({"meta": meta})
        self._out.flush()
        self._thread = threading.Thread(
            target=self._drain, name="vcr-step-recorder", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._thread.join(timeout=5)
        logger.info(
            "step recorder closed: %d steps, drain thread CPU %.3fs, max backlog %d",
            self._step,
            self._drain_cpu_s,
            self._max_backlog,
        )

    _drain_cpu_s = 0.0

    def _drain(self) -> None:
        try:
            stopping = False
            while not stopping:
                stopping = self._stop.wait(_DRAIN_INTERVAL_S)
                if not self.items:
                    continue
                self._max_backlog = max(self._max_backlog, len(self.items))
                while self.items:
                    self._process(self.items.popleft())
                    time.sleep(0)
                self._out.flush()
        except Exception:
            logger.exception("step recorder stopped; the trace is incomplete")
        finally:
            self._drain_cpu_s = time.thread_time()
            self._out.close()

    def _write(self, line: dict[str, Any]) -> None:
        self._out.write(json.dumps(line, separators=(",", ":")))
        self._out.write("\n")

    def _ms(self, start_ns: int, end_ns: int) -> float:
        return round(max(end_ns - start_ns, 0) / 1e6, 4)

    def _process(self, item: tuple[Any, ...]) -> None:
        if item[0] == _SCHEDULE:
            _, call_ns, return_ns, scheduled = item
            if scheduled.num_scheduled_tokens:
                self._step += 1
            self._pending[id(scheduled)] = (self._step, call_ns, return_ns, scheduled)
            self._kv_started(scheduled, return_ns)
            return
        _, entry_ns, exit_ns, scheduled, output, sampled_lens = item
        pending = self._pending.pop(id(scheduled), None)
        if pending is None:
            return
        step, call_ns, return_ns, _ = pending
        self._kv_finished(output, step, entry_ns)
        if scheduled.num_scheduled_tokens:
            record = self._shape(scheduled, output, sampled_lens)
            record.update(
                step=step,
                ts_ms=self._ms(self._start_ns, call_ns),
                exec_ms=self._ms(max(return_ns, self._last_update_entry_ns), entry_ns),
                step_ms=self._ms(max(call_ns, self._last_update_exit_ns), exit_ns),
                sched_ms=self._ms(call_ns, return_ns),
                update_ms=self._ms(entry_ns, exit_ns),
            )
            self._write({"step": record})
        self._last_update_entry_ns = entry_ns
        self._last_update_exit_ns = exit_ns

    def _shape(
        self, scheduled: Any, output: Any, sampled_lens: Any
    ) -> dict[str, Any]:
        prefilling = self._prefilling
        depth: dict[str, int] = {}
        for new in scheduled.scheduled_new_reqs:
            depth[new.req_id] = new.num_computed_tokens
            prefilling.add(new.req_id)
        cached = scheduled.scheduled_cached_reqs
        for req_id, computed in zip(cached.req_ids, cached.num_computed_tokens):
            depth[req_id] = computed
        prefilling.update(getattr(cached, "resumed_req_ids", None) or ())

        sampled_by_req: dict[str, int] = {}
        if sampled_lens:
            sampled_by_req.update(zip(output.req_ids, sampled_lens))
        drafts_by_req = scheduled.scheduled_spec_decode_tokens or {}

        reqs = tokens = drafts = context = sampled = accepted = 0
        prefill: list[dict[str, Any]] = []
        for req_id, num_tokens in scheduled.num_scheduled_tokens.items():
            got = sampled_by_req.get(req_id, 0)
            sampled += got
            if req_id in prefilling:
                prefill.append(
                    {
                        "tokens": num_tokens,
                        "kv_depth": depth.get(req_id, 0),
                        "last": got > 0,
                    }
                )
                if got:
                    prefilling.discard(req_id)
                continue
            num_drafts = len(drafts_by_req.get(req_id, ()))
            reqs += 1
            tokens += num_tokens
            drafts += num_drafts
            context += depth.get(req_id, 0)
            if num_drafts and got:
                accepted += got - 1
        prefilling.difference_update(scheduled.finished_req_ids or ())
        return {
            "decode": {
                "reqs": reqs,
                "tokens": tokens,
                "drafts": drafts,
                "context_tokens": context,
            },
            "prefill": prefill,
            "sampled": sampled,
            "accepted": accepted,
        }

    def _kv_started(self, scheduled: Any, now_ns: int) -> None:
        for metadata in _connector_metadata(scheduled.kv_connector_metadata):
            for req_id, load in (getattr(metadata, "reqs_to_recv", None) or {}).items():
                block_ids = _flat_block_ids(getattr(load, "local_block_ids", None))
                if block_ids:
                    self._loads[req_id] = (self._step, now_ns, block_ids)
            for req_id in getattr(metadata, "reqs_to_send", None) or ():
                self._sends.setdefault(req_id, (self._step, now_ns))

    def _kv_finished(self, output: Any, step: int, now_ns: int) -> None:
        kv = getattr(output, "kv_connector_output", None)
        if kv is None:
            return
        failed = getattr(kv, "failed_recving", None) or ()
        invalid = getattr(kv, "invalid_block_ids", None) or ()
        for req_id in kv.finished_recving or ():
            load = self._loads.pop(req_id, None)
            if load is None:
                continue
            start_step, start_ns, block_ids = load
            if req_id in failed:
                outcome = "failed"
            elif invalid and not invalid.isdisjoint(block_ids):
                outcome = "invalid_blocks"
            else:
                outcome = "ok"
            self._write(
                {
                    "kv_transfer": {
                        "direction": "recv",
                        "blocks": len(block_ids),
                        "start_step": start_step,
                        "end_step": step,
                        "wall_ms": self._ms(start_ns, now_ns),
                        "outcome": outcome,
                    }
                }
            )
        for req_id in kv.finished_sending or ():
            send = self._sends.pop(req_id, None)
            if send is None:
                continue
            start_step, start_ns = send
            self._write(
                {
                    "kv_transfer": {
                        "direction": "send",
                        "blocks": 0,
                        "start_step": start_step,
                        "end_step": step,
                        "wall_ms": self._ms(start_ns, now_ns),
                        "outcome": "ok",
                    }
                }
            )


def recording_cls(base: type, recorder: StepRecorder) -> type:
    """A subclass of `base` that hands every step to `recorder`."""
    base_schedule = base.schedule
    base_update = base.update_from_output
    base_shutdown = base.shutdown
    append = recorder.items.append
    now = time.perf_counter_ns

    class Recording(base):  # type: ignore[misc, valid-type]
        def schedule(self, *args: Any, **kwargs: Any) -> Any:
            call_ns = now()
            scheduled = base_schedule(self, *args, **kwargs)
            append((_SCHEDULE, call_ns, now(), scheduled))
            return scheduled

        def update_from_output(
            self, scheduler_output: Any, model_runner_output: Any, *args: Any, **kwargs: Any
        ) -> Any:
            entry_ns = now()
            tokens = getattr(model_runner_output, "sampled_token_ids", None)
            sampled_lens = list(map(len, tokens)) if tokens else ()
            outputs = base_update(
                self, scheduler_output, model_runner_output, *args, **kwargs
            )
            append(
                (
                    _UPDATE,
                    entry_ns,
                    now(),
                    scheduler_output,
                    model_runner_output,
                    sampled_lens,
                )
            )
            return outputs

        def shutdown(self) -> None:
            base_shutdown(self)
            recorder.close()

    Recording.__name__ = Recording.__qualname__ = f"Recording{base.__name__}"
    return Recording


def default_scheduler_cls(vllm_config: Any) -> type:
    """The scheduler class vLLM picks when `--scheduler-cls` is not given."""
    scheduler_config = vllm_config.scheduler_config
    if getattr(vllm_config, "diffusion_config", None) is not None:
        from vllm.v1.core.sched import diffusion_scheduler

        if scheduler_config.async_scheduling:
            return diffusion_scheduler.DiffusionAsyncScheduler
        return diffusion_scheduler.DiffusionScheduler
    configured = scheduler_config.scheduler_cls
    scheduler_config.scheduler_cls = None
    try:
        return scheduler_config.get_scheduler_cls()
    finally:
        scheduler_config.scheduler_cls = configured


def _meta(vllm_config: Any, base: type, block_size: int | None) -> dict[str, Any]:
    scheduler_config = vllm_config.scheduler_config
    meta: dict[str, Any] = {
        "source": "vcr_step_recorder",
        "model": getattr(getattr(vllm_config, "model_config", None), "model", None),
        "max_num_seqs": getattr(scheduler_config, "max_num_seqs", None),
        "block_size": block_size,
        "config_hash": os.environ.get(CONFIG_HASH_ENV),
        "scheduler": base.__name__,
        "async_scheduling": bool(getattr(scheduler_config, "async_scheduling", False)),
        "max_num_batched_tokens": getattr(scheduler_config, "max_num_batched_tokens", None),
        "unix_start_ms": time.time_ns() // 1_000_000,
    }
    try:
        from vllm import __version__

        meta["vllm_version"] = __version__
    except ImportError:
        pass
    return {key: value for key, value in meta.items() if value is not None}


def recording_scheduler(vllm_config: Any, **kwargs: Any) -> Any:
    """`--scheduler-cls` target: vLLM's default scheduler, recording its steps."""
    path = os.environ.get(OUT_ENV)
    if not path:
        raise ValueError(f"{OUT_ENV} must name the step trace file to write")
    base = default_scheduler_cls(vllm_config)
    out = open(path.replace("{pid}", str(os.getpid())), "w", buffering=1 << 16)
    recorder = StepRecorder(out, _meta(vllm_config, base, kwargs.get("block_size")))
    atexit.register(recorder.close)
    logger.info("recording %s steps to %s", base.__name__, out.name)
    return recording_cls(base, recorder)(vllm_config=vllm_config, **kwargs)
