"""Check vcr_step_recorder.py against the real scheduler of a vLLM source checkout.

No GPU: the scheduler is vLLM's own, the model runner output is produced by
vLLM's scheduler test harness. Run it through `just check-step-recorder <vllm-src>`.

    scenario   one request through known steps; the trace must match exactly
    fuzz       vLLM's differential harness (tests/v1/core/rust_core_diff.py):
               a plain scheduler and a recording one in lockstep must agree on
               every output, and the trace must account for every scheduled
               and sampled token and every finished KV load
    overhead   time spent inside schedule() + update_from_output() per step,
               plain scheduler against recording scheduler, with the engine
               thread blocking between the two calls as it does on the model
               runner, and with it never blocking
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vcr_step_recorder as recorder_module  # noqa: E402


def recording(base: type, path: Path) -> tuple[type, recorder_module.StepRecorder]:
    recorder = recorder_module.StepRecorder(
        open(path, "w"), {"source": "vcr_step_recorder", "scheduler": base.__name__}
    )
    return recorder_module.recording_cls(base, recorder), recorder


def read_trace(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    steps: list[dict[str, Any]] = []
    transfers: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if "step" in record:
            steps.append(record["step"])
        elif "kv_transfer" in record:
            transfers.append(record["kv_transfer"])
    return steps, transfers


def model_output(scheduled: Any, sampled: list[list[int]], **extra: Any) -> Any:
    from vllm.v1.outputs import ModelRunnerOutput

    req_ids = list(scheduled.num_scheduled_tokens)
    fields = ModelRunnerOutput.__dataclass_fields__
    if "req_id_to_index" in fields:
        extra["req_id_to_index"] = {req_id: i for i, req_id in enumerate(req_ids)}
    return ModelRunnerOutput(req_ids=req_ids, sampled_token_ids=sampled, **extra)


def shape(step: dict[str, Any]) -> dict[str, Any]:
    return {key: step[key] for key in ("decode", "prefill", "sampled", "accepted")}


def harness_scheduler(cls: type, **kwargs: Any) -> Any:
    """vLLM's harness scheduler built from `cls`. Older harnesses take no
    `scheduler_cls` and build the `Scheduler` their module names."""
    from tests.v1.core import utils

    if "scheduler_cls" in inspect.signature(utils.create_scheduler).parameters:
        return utils.create_scheduler(scheduler_cls=cls, **kwargs)
    plain = utils.Scheduler
    utils.Scheduler = cls
    try:
        return utils.create_scheduler(**kwargs)
    finally:
        utils.Scheduler = plain


def check_scenario(fixture: Path | None) -> None:
    """A 100-token prompt under a 64-token step budget, six output tokens. The
    last step returns two tokens past the length stop, which the scheduler trims."""
    from tests.v1.core.utils import create_requests
    from vllm.v1.core.sched.scheduler import Scheduler

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "steps.jsonl"
        cls, recorder = recording(Scheduler, path)
        scheduler = harness_scheduler(cls, max_num_batched_tokens=64)
        (request,) = create_requests(1, num_tokens=100, max_tokens=6, ignore_eos=True)
        scheduler.add_request(request)
        while scheduler.has_requests():
            scheduled = scheduler.schedule()
            if request.num_computed_tokens < request.num_tokens:
                tokens = []
            elif request.num_output_tokens == 5:
                tokens = [7, 7, 7]
            else:
                tokens = [7]
            sampled = [tokens for _ in scheduled.num_scheduled_tokens]
            scheduler.update_from_output(scheduled, model_output(scheduled, sampled))
            assert len(tokens) <= 1, "the scheduler trims what follows a stop in place"
        recorder.close()
        steps, transfers = read_trace(path)
        if fixture is not None:
            fixture.write_text(path.read_text())

    def decode(context: int, sampled: int = 1) -> dict[str, Any]:
        return {
            "decode": {"reqs": 1, "tokens": 1, "drafts": 0, "context_tokens": context},
            "prefill": [],
            "sampled": sampled,
            "accepted": 0,
        }

    def prefill(tokens: int, kv_depth: int, last: bool) -> dict[str, Any]:
        return {
            "decode": {"reqs": 0, "tokens": 0, "drafts": 0, "context_tokens": 0},
            "prefill": [{"tokens": tokens, "kv_depth": kv_depth, "last": last}],
            "sampled": int(last),
            "accepted": 0,
        }

    expected = [prefill(64, 0, False), prefill(36, 64, True)]
    expected += [decode(100 + i) for i in range(4)] + [decode(104, sampled=3)]
    assert [shape(step) for step in steps] == expected, steps
    assert [step["step"] for step in steps] == list(range(1, 8)), steps
    assert transfers == []
    for step in steps:
        assert step["exec_ms"] >= 0 and step["step_ms"] >= step["update_ms"], step
    print(f"scenario OK: {len(steps)} steps")


@dataclasses.dataclass
class Truth:
    """What one executed step scheduled and returned, read off the scheduler's
    own inputs and outputs by a class below the recorder."""

    scheduled_tokens: int
    drafts: int
    sampled: int = 0
    loads_finished: int = 0
    sends_finished: int = 0


def tallying(base: type, truth: list[Truth]) -> type:
    by_output: dict[int, Truth] = {}
    counted = {"loads": 0, "sends": 0}

    class Tallying(base):  # type: ignore[misc, valid-type]
        def schedule(self, *args: Any, **kwargs: Any) -> Any:
            scheduled = super().schedule(*args, **kwargs)
            if scheduled.total_num_scheduled_tokens:
                drafts = sum(map(len, scheduled.scheduled_spec_decode_tokens.values()))
                by_output[id(scheduled)] = Truth(
                    scheduled.total_num_scheduled_tokens, drafts
                )
            return scheduled

        def update_from_output(self, scheduled: Any, output: Any, *args: Any, **kwargs: Any) -> Any:
            kv = output.kv_connector_output
            if kv is not None:
                counted["loads"] += len(kv.finished_recving or ())
                counted["sends"] += len(kv.finished_sending or ())
            step = by_output.pop(id(scheduled), None)
            if step is not None:
                step.sampled = sum(map(len, output.sampled_token_ids))
                truth.append(step)
            return super().update_from_output(scheduled, output, *args, **kwargs)

    Tallying.counted = counted  # type: ignore[attr-defined]
    return Tallying


def check_fuzz(seeds: range, use_async: bool, nixl: bool) -> None:
    from tests.v1.core.rust_core_diff import FuzzConfig, build_scheduler, run_one
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.scheduler import Scheduler

    base = AsyncScheduler if use_async else Scheduler
    label = f"fuzz {'async' if use_async else 'sync'}{' nixl' if nixl else ''}"
    totals = {"steps": 0, "prefill_chunks": 0, "drafted": 0, "loads": 0, "sends": 0}
    outcomes: dict[str, int] = {}
    truth_drafted = 0
    unsupported: list[int] = []
    for seed in seeds:
        cfg = FuzzConfig.sample_nixl(seed) if nixl else FuzzConfig.sample(seed)
        try:
            build_scheduler(cfg, base)
        except ValueError as error:
            if "does not yet support" not in str(error):
                raise
            unsupported.append(seed)
            continue
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "steps.jsonl"
            truth: list[Truth] = []
            tally = tallying(base, truth)
            cls, recorder = recording(tally, path)
            ok, message = run_one(
                cfg,
                lambda cfg: build_scheduler(cfg, base),
                lambda cfg, cls=cls: build_scheduler(cfg, cls),
                use_async=use_async,
            )
            recorder.close()
            steps, transfers = read_trace(path)
        assert ok, f"{label} seed {seed}: the recording scheduler diverged: {message}"
        assert len(steps) == len(truth), (label, seed, len(steps), len(truth))
        truth_drafted += sum(real.drafts for real in truth)
        for step, real in zip(steps, truth):
            tokens = step["decode"]["tokens"] + sum(c["tokens"] for c in step["prefill"])
            assert tokens == real.scheduled_tokens, (label, seed, step, real)
            assert step["sampled"] == real.sampled, (label, seed, step, real)
            assert step["decode"]["drafts"] <= real.drafts, (label, seed, step, real)
            assert step["accepted"] <= step["decode"]["drafts"], (label, seed, step)
            assert all(c["tokens"] > 0 for c in step["prefill"]), (label, seed, step)
        counted = tally.counted  # type: ignore[attr-defined]
        recv = [t for t in transfers if t["direction"] == "recv"]
        send = [t for t in transfers if t["direction"] == "send"]
        assert len(recv) <= counted["loads"], (label, seed, len(recv), counted)
        assert len(send) <= counted["sends"], (label, seed, len(send), counted)
        for transfer in transfers:
            assert transfer["end_step"] >= transfer["start_step"], (label, seed, transfer)
            outcomes[transfer["outcome"]] = outcomes.get(transfer["outcome"], 0) + 1
        totals["steps"] += len(steps)
        totals["prefill_chunks"] += sum(len(step["prefill"]) for step in steps)
        totals["drafted"] += sum(step["decode"]["drafts"] for step in steps)
        totals["loads"] += len(recv)
        totals["sends"] += len(send)
    print(f"{label} OK: {len(seeds) - len(unsupported)} seeds, {totals}, outcomes {outcomes}")
    if unsupported:
        print(f"  seeds {unsupported} not run: this platform cannot build their config")
    assert (totals["drafted"] > 0) == (truth_drafted > 0), (label, totals, truth_drafted)
    if nixl:
        assert totals["loads"] > 0, "the NIXL fuzz recorded no loads"


def scheduler_time_ns(cfg: Any, scheduler_cls: type, runner_wait_s: float) -> tuple[int, int]:
    """Executed steps and nanoseconds spent in schedule() + update_from_output(),
    sleeping `runner_wait_s` between the two where the engine waits on the runner."""
    from tests.v1.core.rust_core_diff import (
        Workload,
        build_scheduler,
        fake_model_output,
        make_request,
    )
    from vllm.utils.hashing import sha256, sha256_cbor
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import RequestStatus

    hash_fn = {"sha256": sha256, "sha256_cbor": sha256_cbor}[cfg.hash_fn]
    init_none_hash(hash_fn)
    hasher = get_request_block_hasher(cfg.block_size, hash_fn)
    scheduler = build_scheduler(cfg, scheduler_cls)
    workload = Workload(cfg)
    now = time.perf_counter_ns
    spent = executed = 0
    for step in range(cfg.num_steps):
        for spec in workload.new_request_specs():
            scheduler.add_request(make_request(spec, hasher))
        aborts = workload.aborts(list(scheduler.requests))
        if aborts:
            scheduler.finish_requests(aborts, RequestStatus.FINISHED_ABORTED)
        start = now()
        scheduled = scheduler.schedule()
        spent += now() - start
        output, drafts = fake_model_output(cfg, step, scheduler, scheduled)
        if runner_wait_s and scheduled.total_num_scheduled_tokens:
            time.sleep(runner_wait_s)
        start = now()
        scheduler.update_from_output(scheduled, output)
        spent += now() - start
        if drafts is not None:
            scheduler.update_draft_token_ids(drafts)
        executed += bool(scheduled.total_num_scheduled_tokens)
    return executed, spent


def check_overhead(seeds: Any, rounds: int, steps: int) -> None:
    """Scheduler time per executed step, the fastest of `rounds` runs per seed."""
    from tests.v1.core.rust_core_diff import FuzzConfig
    from vllm.v1.core.sched.scheduler import Scheduler

    seeds = [seed for seed in seeds if FuzzConfig.sample(seed).num_spec_tokens == 0][:3]

    for label, runner_wait_s in (("runner wait 1ms", 0.001), ("no runner wait", 0.0)):
        plain_us: list[float] = []
        added_us: list[float] = []
        drain_us: list[float] = []
        for seed in seeds:
            cfg = dataclasses.replace(FuzzConfig.sample(seed), num_steps=steps)
            plain = recorded = drain = float("inf")
            for _ in range(rounds):
                executed, spent = scheduler_time_ns(cfg, Scheduler, runner_wait_s)
                plain = min(plain, spent / executed / 1e3)
                with tempfile.TemporaryDirectory() as tmp:
                    cls, recorder = recording(Scheduler, Path(tmp) / "steps.jsonl")
                    recorded_steps, spent = scheduler_time_ns(cfg, cls, runner_wait_s)
                    recorder.close()
                assert executed == recorded_steps
                recorded = min(recorded, spent / executed / 1e3)
                drain = min(drain, recorder._drain_cpu_s * 1e6 / executed)
            plain_us.append(plain)
            added_us.append(recorded - plain)
            drain_us.append(drain)
        print(
            f"overhead per executed step, {label}: "
            f"plain scheduler {statistics.median(plain_us):.1f}us, "
            f"recording adds {statistics.median(added_us):+.2f}us on the engine thread "
            f"(per seed: {', '.join(f'{us:+.2f}' for us in added_us)}), "
            f"drain thread {statistics.median(drain_us):.1f}us CPU"
        )


def check_hot_path(steps: int) -> None:
    """Cost of the two overridden methods over a scheduler that does nothing,
    with 16 requests' sampled tokens in every model runner output."""

    class Scheduled:
        num_scheduled_tokens: dict[str, int] = {}
        kv_connector_metadata = None

    class Output:
        req_ids = [str(i) for i in range(16)]
        sampled_token_ids = [[7] for _ in range(16)]

    class Idle:
        def schedule(self) -> Scheduled:
            return scheduled

        def update_from_output(self, scheduler_output: Any, model_runner_output: Any) -> None:
            return None

        def shutdown(self) -> None:
            return None

    scheduled = Scheduled()
    runner_output = Output()

    def per_step_ns(scheduler: Any, runner_wait_s: float, steps: int) -> float:
        now = time.perf_counter_ns
        best = float("inf")
        for _ in range(5):
            spent = 0
            for _ in range(steps):
                start = now()
                output = scheduler.schedule()
                spent += now() - start
                if runner_wait_s:
                    time.sleep(runner_wait_s)
                start = now()
                scheduler.update_from_output(output, runner_output)
                spent += now() - start
            best = min(best, spent / steps)
        return best

    for label, runner_wait_s, count in (
        ("runner wait 50us", 0.00005, steps // 50),
        ("no runner wait", 0.0, steps),
    ):
        with tempfile.TemporaryDirectory() as tmp:
            cls, recorder = recording(Idle, Path(tmp) / "steps.jsonl")
            added = per_step_ns(cls(), runner_wait_s, count) - per_step_ns(
                Idle(), runner_wait_s, count
            )
            recorder.close()
        print(f"hot path, {label}: the overrides add {added:.0f}ns per step")


def parse_seeds(text: str) -> range:
    first, _, last = text.partition("-")
    return range(int(first), int(last or first) + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vllm-src", required=True, type=Path)
    parser.add_argument("--seeds", default="0-11", type=parse_seeds)
    parser.add_argument("--overhead-rounds", default=3, type=int)
    parser.add_argument("--overhead-steps", default=1000, type=int)
    parser.add_argument("--write-fixture", type=Path)
    parser.add_argument("--skip-fuzz", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.vllm_src))

    check_scenario(args.write_fixture)
    if args.skip_fuzz:
        return
    check_fuzz(args.seeds, use_async=False, nixl=False)
    check_fuzz(args.seeds, use_async=True, nixl=False)
    check_fuzz(args.seeds, use_async=False, nixl=True)
    check_fuzz(args.seeds, use_async=True, nixl=True)
    check_hot_path(200_000)
    check_overhead(args.seeds, args.overhead_rounds, args.overhead_steps)


if __name__ == "__main__":
    main()
