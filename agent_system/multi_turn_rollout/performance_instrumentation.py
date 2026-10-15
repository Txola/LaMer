"""Pure aggregation helpers for low-overhead Meta-RL rollout profiling."""

from __future__ import annotations

from collections import defaultdict
from math import isfinite
from typing import Any, Iterable


TIMED_STAGES = (
    "env_pool_acquire_s",
    "env_manager_init_s",
    "env_reset_execution_s",
    "env_reset_prompt_s",
    "env_restart_execution_s",
    "env_restart_prompt_s",
    "env_step_projection_s",
    "env_step_execution_s",
    "env_step_prompt_s",
    "reflection_prompt_s",
    "reflection_projection_s",
    "client_prompt_tokenization_s",
    "client_response_processing_s",
)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def _phase_label(value: Any) -> str:
    phase = str(value or "unknown")
    return "reflection" if phase == "reflect" else phase


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _record_groups(fields: dict[str, Any]) -> tuple[str, ...]:
    phase = _phase_label(fields.get("phase"))
    try:
        attempt = int(fields.get("attempt_index")) + 1
    except (TypeError, ValueError):
        attempt = 0
    groups = ["overall", f"phase/{phase}"]
    if attempt > 0:
        groups.extend(
            [
                f"attempt/{attempt}",
                f"phase_attempt/{phase}_{attempt}",
            ]
        )
    return tuple(groups)


def summarize_rollout_profile(
    records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate request/cache measurements without changing rollout behavior.

    Cache-hit denominators include only records for which vLLM returned an
    authoritative ``num_cached_tokens`` value. Prompt and generated-token
    totals still include every record, and coverage makes missing cache
    metadata explicit rather than silently treating it as a miss.
    """

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    trajectory_completion: dict[tuple[str, str], float] = {}
    rollout_starts: list[float] = []
    rollout_ends: list[float] = []
    request_intervals: list[tuple[float, float]] = []
    worker_ids: set[str] = set()
    worker_intervals: dict[str, list[tuple[float, float, float, float]]] = (
        defaultdict(list)
    )

    materialized = list(records)
    for fields in materialized:
        for group in _record_groups(fields):
            groups[group].append(fields)

        start = _finite_float(fields.get("request_start_monotonic_s"))
        end = _finite_float(fields.get("request_end_monotonic_s"))
        if start is not None:
            rollout_starts.append(start)
        if end is not None:
            rollout_ends.append(end)
            if start is not None and end >= start:
                request_intervals.append((start, end))
            task_uid = str(fields.get("task_uid", ""))
            trajectory_uid = str(fields.get("traj_uid", ""))
            if task_uid or trajectory_uid:
                key = (task_uid, trajectory_uid)
                trajectory_completion[key] = max(
                    end,
                    trajectory_completion.get(key, end),
                )
        worker_id = fields.get("rollout_worker_id")
        if worker_id is not None:
            worker_key = str(worker_id)
            worker_ids.add(worker_key)
            cpu_start = _finite_float(fields.get("worker_process_cpu_start_s"))
            cpu_end = _finite_float(fields.get("worker_process_cpu_end_s"))
            if (
                start is not None
                and end is not None
                and cpu_start is not None
                and cpu_end is not None
                and end >= start
                and cpu_end >= cpu_start
            ):
                worker_intervals[worker_key].append(
                    (start, end, cpu_start, cpu_end)
                )

    summaries: dict[str, dict[str, float | int]] = {}
    for group_name, members in sorted(groups.items()):
        prompt_tokens = 0
        generated_tokens = 0
        cached_tokens = 0
        cache_prompt_tokens = 0
        cache_known_records = 0
        cache_invalid_records = 0
        request_latencies: list[float] = []
        queue_latencies: list[float] = []
        prefill_latencies: list[float] = []
        decode_latencies: list[float] = []
        stage_values: dict[str, list[float]] = {
            name: [] for name in TIMED_STAGES
        }
        preemptions: list[float] = []

        for fields in members:
            prompt = max(0, int(fields.get("prompt_token_count", 0) or 0))
            generated = max(0, int(fields.get("response_token_count", 0) or 0))
            prompt_tokens += prompt
            generated_tokens += generated

            cached_value = fields.get("num_cached_tokens")
            if cached_value is not None:
                cached = max(0, int(cached_value))
                cache_known_records += 1
                cache_prompt_tokens += prompt
                cached_tokens += cached
                cache_invalid_records += int(cached > prompt)

            for key, destination in (
                ("request_latency_s", request_latencies),
                ("vllm_queue_latency_s", queue_latencies),
                ("vllm_prefill_latency_s", prefill_latencies),
                ("vllm_decode_latency_s", decode_latencies),
            ):
                value = _finite_float(fields.get(key))
                if value is not None and value >= 0.0:
                    destination.append(value)
            for name, destination in stage_values.items():
                value = _finite_float(fields.get(name))
                if value is not None and value >= 0.0:
                    destination.append(value)
            preempted = _finite_float(fields.get("rollout_num_preempted"))
            if preempted is not None and preempted >= 0.0:
                preemptions.append(preempted)

        def mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        summaries[group_name] = {
            "record_count": len(members),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "cached_prompt_tokens": cached_tokens,
            "cache_prompt_tokens_with_metadata": cache_prompt_tokens,
            "cache_known_records": cache_known_records,
            "cache_unknown_records": len(members) - cache_known_records,
            "cache_invalid_records": cache_invalid_records,
            "cache_hit_fraction": (
                cached_tokens / cache_prompt_tokens if cache_prompt_tokens else 0.0
            ),
            "request_latency_mean_s": mean(request_latencies),
            "request_latency_max_s": max(request_latencies, default=0.0),
            "request_latency_p50_s": _percentile(request_latencies, 0.5),
            "request_latency_p90_s": _percentile(request_latencies, 0.9),
            "request_latency_p99_s": _percentile(request_latencies, 0.99),
            "request_latency_count": len(request_latencies),
            "queue_latency_mean_s": mean(queue_latencies),
            "queue_latency_count": len(queue_latencies),
            "prefill_latency_mean_s": mean(prefill_latencies),
            "prefill_latency_count": len(prefill_latencies),
            "decode_latency_mean_s": mean(decode_latencies),
            "decode_latency_count": len(decode_latencies),
            "preemption_count": len(preemptions),
            "preemption_total": sum(preemptions),
            **{
                f"{name}_{suffix}": result
                for name, values in stage_values.items()
                for suffix, result in (
                    ("count", len(values)),
                    ("mean", mean(values)),
                    ("total", sum(values)),
                )
            },
        }

    completions = sorted(trajectory_completion.values())
    slowest_tail = (
        completions[-1] - completions[-2] if len(completions) >= 2 else 0.0
    )
    rollout_span = (
        max(rollout_ends) - min(rollout_starts)
        if rollout_starts and rollout_ends
        else 0.0
    )
    concurrency_events = sorted(
        [event for start, end in request_intervals for event in ((start, 1), (end, -1))],
        key=lambda event: (event[0], event[1]),
    )
    in_flight = 0
    max_in_flight = 0
    for _, change in concurrency_events:
        in_flight += change
        max_in_flight = max(max_in_flight, in_flight)
    worker_cpu_core_fractions: list[float] = []
    worker_request_gap_seconds: list[float] = []
    for intervals in worker_intervals.values():
        by_start = sorted(intervals)
        wall_start = by_start[0][0]
        wall_end = max(interval[1] for interval in by_start)
        cpu_start = min(by_start, key=lambda interval: interval[0])[2]
        cpu_end = max(by_start, key=lambda interval: interval[1])[3]
        if wall_end > wall_start:
            worker_cpu_core_fractions.append(
                max(0.0, cpu_end - cpu_start) / (wall_end - wall_start)
            )

        gap_seconds = 0.0
        active_end = by_start[0][1]
        for start, end, _cpu_start, _cpu_end in by_start[1:]:
            if start > active_end:
                gap_seconds += start - active_end
            active_end = max(active_end, end)
        worker_request_gap_seconds.append(gap_seconds)

    return {
        "groups": summaries,
        "trajectory_count_with_timing": len(completions),
        "rollout_worker_count": len(worker_ids),
        "rollout_request_span_s": rollout_span,
        "generated_tokens_per_request_span_s": (
            summaries.get("overall", {}).get("generated_tokens", 0) / rollout_span
            if rollout_span > 0.0
            else 0.0
        ),
        "prompt_tokens_per_request_span_s": (
            summaries.get("overall", {}).get("prompt_tokens", 0) / rollout_span
            if rollout_span > 0.0
            else 0.0
        ),
        "request_interval_count": len(request_intervals),
        "average_requests_in_flight": (
            sum(end - start for start, end in request_intervals) / rollout_span
            if rollout_span > 0.0
            else 0.0
        ),
        "max_requests_in_flight": max_in_flight,
        "slowest_trajectory_tail_s": slowest_tail,
        "worker_cpu_core_fraction_mean": (
            sum(worker_cpu_core_fractions) / len(worker_cpu_core_fractions)
            if worker_cpu_core_fractions
            else 0.0
        ),
        "worker_cpu_core_fraction_max": max(
            worker_cpu_core_fractions, default=0.0
        ),
        "worker_request_gap_seconds_mean": (
            sum(worker_request_gap_seconds) / len(worker_request_gap_seconds)
            if worker_request_gap_seconds
            else 0.0
        ),
    }


def rollout_profile_metrics(records: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Return numeric trainer metrics for :func:`summarize_rollout_profile`."""

    summary = summarize_rollout_profile(records)
    metrics: dict[str, float] = {
        "rollout_profile/trajectory_count_with_timing": float(
            summary["trajectory_count_with_timing"]
        ),
        "rollout_profile/worker_count": float(summary["rollout_worker_count"]),
        "rollout_profile/request_span_s": float(summary["rollout_request_span_s"]),
        "rollout_profile/generated_tokens_per_request_span_s": float(
            summary["generated_tokens_per_request_span_s"]
        ),
        "rollout_profile/prompt_tokens_per_request_span_s": float(
            summary["prompt_tokens_per_request_span_s"]
        ),
        "rollout_profile/request_interval_count": float(
            summary["request_interval_count"]
        ),
        "rollout_profile/average_requests_in_flight": float(
            summary["average_requests_in_flight"]
        ),
        "rollout_profile/max_requests_in_flight": float(
            summary["max_requests_in_flight"]
        ),
        "rollout_profile/slowest_trajectory_tail_s": float(
            summary["slowest_trajectory_tail_s"]
        ),
        "rollout_profile/worker_cpu_core_fraction_mean": float(
            summary["worker_cpu_core_fraction_mean"]
        ),
        "rollout_profile/worker_cpu_core_fraction_max": float(
            summary["worker_cpu_core_fraction_max"]
        ),
        "rollout_profile/worker_request_gap_seconds_mean": float(
            summary["worker_request_gap_seconds_mean"]
        ),
    }
    for group, values in summary["groups"].items():
        prefix = f"rollout_profile/{group}"
        for key, value in values.items():
            metrics[f"{prefix}/{key}"] = float(value)
    return metrics
