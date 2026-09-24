"""Detailed, read-only diagnostics for agent validation rollouts."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Any

import numpy as np


def _json_value(value: Any) -> Any:
    """Convert tensors/NumPy/object metadata to JSON-compatible Python values."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        value = value.tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _metadata_at(data, key: str, index: int, default=None):
    values = data.non_tensor_batch.get(key)
    if values is None:
        return default
    return _json_value(values[index])


def collect_validation_interactions(data, tokenizer) -> list[dict[str, Any]]:
    """Convert a validation DataProto into human-readable interaction records."""
    responses = data.batch["responses"]
    prompts = data.batch["prompts"]
    attention_mask = data.batch["attention_mask"]
    response_width = responses.shape[-1]
    records = []

    for index in range(len(data)):
        response_mask = attention_mask[index, -response_width:].bool()
        prompt_mask = attention_mask[index, :-response_width].bool()
        response_ids = responses[index][response_mask]
        prompt_ids = prompts[index][prompt_mask]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        phase = str(_metadata_at(data, "phase", index, "unknown"))
        attempt_zero_based = int(_metadata_at(data, "traj_idx", index, -1))
        turn_zero_based = int(_metadata_at(data, "turn_idx", index, -1))
        response_tokens = int(response_mask.sum().item())

        anchor_obs = _metadata_at(data, "anchor_obs", index)
        next_anchor_obs = _metadata_at(data, "next_anchor_obs", index)
        environment_effective_flag = _metadata_at(data, "action_is_effective", index)
        board_changed = (
            anchor_obs != next_anchor_obs
            if phase == "play" and anchor_obs is not None and next_anchor_obs is not None
            else None
        )

        record = {
            "uid": str(_metadata_at(data, "uid", index, "")),
            "traj_uid": str(_metadata_at(data, "traj_uid", index, "")),
            "phase": phase,
            "attempt": attempt_zero_based + 1,
            "attempt_zero_based": attempt_zero_based,
            "turn": turn_zero_based + 1,
            "turn_zero_based": turn_zero_based,
            "is_parse_valid": bool(_metadata_at(data, "is_action_valid", index, False)),
            # board_changed is the reliable pre/post observation comparison.
            # The environment flag can be stale when MineField returns early.
            "is_effective": board_changed,
            "board_changed": board_changed,
            "environment_action_is_effective": environment_effective_flag,
            "done": bool(_metadata_at(data, "action_done", index, False)),
            "won": bool(_metadata_at(data, "action_won", index, False)),
            "immediate_reward": float(_metadata_at(data, "rewards", index, 0.0)),
            "prompt_tokens": int(prompt_mask.sum().item()),
            "response_tokens": response_tokens,
            "response_token_limit": int(response_width),
            "response_reached_token_limit": response_tokens >= response_width,
            "has_complete_action_tag": "<action>" in response and "</action>" in response,
            "has_complete_reflection_tag": "<remark>" in response and "</remark>" in response,
            "parsed_action": _metadata_at(data, "parsed_action", index),
            "anchor_obs": anchor_obs,
            "next_anchor_obs": next_anchor_obs,
            "previous_reflections": _metadata_at(data, "previous_reflections", index, []),
            "prompt": prompt,
            "response": response,
        }
        if record["is_effective"] is not None:
            record["is_effective"] = bool(record["is_effective"])
            record["board_changed"] = bool(record["board_changed"])
        if record["environment_action_is_effective"] is not None:
            record["environment_action_is_effective"] = bool(
                record["environment_action_is_effective"]
            )
        records.append(record)

    return records


def _mean(values) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values else None


def _rate(records, predicate) -> float | None:
    return _mean(float(bool(predicate(record))) for record in records)


def summarize_validation_interactions(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return flat trainer metrics plus per-task summaries for one validation run."""
    plays = [record for record in records if record["phase"] == "play"]
    reflections = [record for record in records if record["phase"] == "reflect"]
    parsed_plays = [record for record in plays if record["is_parse_valid"]]
    parsed_effective_known = [
        record for record in parsed_plays if record["is_effective"] is not None
    ]
    effective_known = [record for record in plays if record["is_effective"] is not None]
    environment_effective_known = [
        record
        for record in effective_known
        if record.get("environment_action_is_effective") is not None
    ]

    metrics: dict[str, float | int | None] = {
        "val/diagnostics/record_count": len(records),
        "val/diagnostics/play_record_count": len(plays),
        "val/diagnostics/reflection_record_count": len(reflections),
        # This matches the phase-mixed, record-weighted training valid_action_ratio.
        "val/diagnostics/all_record_parse_rate": _rate(records, lambda row: row["is_parse_valid"]),
        "val/diagnostics/play_complete_action_tag_rate": _rate(plays, lambda row: row["has_complete_action_tag"]),
        "val/diagnostics/play_parse_rate": _rate(plays, lambda row: row["is_parse_valid"]),
        "val/diagnostics/play_effective_rate": _rate(effective_known, lambda row: row["is_effective"]),
        "val/diagnostics/environment_effective_flag_mismatch_rate": _rate(
            environment_effective_known,
            lambda row: row["environment_action_is_effective"] != row["is_effective"],
        ),
        "val/diagnostics/effective_given_parsed_rate": _rate(
            parsed_effective_known, lambda row: row["is_effective"]
        ),
        "val/diagnostics/parsed_but_ineffective_rate": _rate(
            parsed_effective_known, lambda row: not row["is_effective"]
        ),
        "val/diagnostics/play_response_token_limit_rate": _rate(plays, lambda row: row["response_reached_token_limit"]),
        "val/diagnostics/reflection_parse_rate": _rate(reflections, lambda row: row["is_parse_valid"]),
        "val/diagnostics/reflection_response_token_limit_rate": _rate(reflections, lambda row: row["response_reached_token_limit"]),
        "val/diagnostics/mean_play_response_tokens": _mean(row["response_tokens"] for row in plays),
        "val/diagnostics/mean_reflection_response_tokens": _mean(row["response_tokens"] for row in reflections),
        "val/diagnostics/mean_immediate_play_reward": _mean(row["immediate_reward"] for row in plays),
        "val/diagnostics/win_transition_count": sum(row["won"] for row in plays),
        "val/diagnostics/terminal_loss_count": sum(row["done"] and not row["won"] for row in plays),
    }

    for reward in (-1.0, -0.1, 0.5, 2.0, 10.0):
        label = str(reward).replace("-", "minus_").replace(".", "_")
        metrics[f"val/diagnostics/reward_{label}_rate"] = _rate(
            plays, lambda row, target=reward: np.isclose(row["immediate_reward"], target)
        )

    for attempt in sorted({row["attempt"] for row in plays}):
        attempt_records = [row for row in plays if row["attempt"] == attempt]
        known = [row for row in attempt_records if row["is_effective"] is not None]
        metrics[f"val/diagnostics/attempt_{attempt}_play_count"] = len(attempt_records)
        metrics[f"val/diagnostics/attempt_{attempt}_parse_rate"] = _rate(
            attempt_records, lambda row: row["is_parse_valid"]
        )
        metrics[f"val/diagnostics/attempt_{attempt}_effective_rate"] = _rate(
            known, lambda row: row["is_effective"]
        )
        metrics[f"val/diagnostics/attempt_{attempt}_token_limit_rate"] = _rate(
            attempt_records, lambda row: row["response_reached_token_limit"]
        )

    by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in plays:
        by_uid[row["uid"]].append(row)

    task_summaries = []
    for uid, task_records in sorted(by_uid.items()):
        known = [row for row in task_records if row["is_effective"] is not None]
        task_summaries.append({
            "uid": uid,
            "play_records": len(task_records),
            "parse_rate": _rate(task_records, lambda row: row["is_parse_valid"]),
            "effective_rate": _rate(known, lambda row: row["is_effective"]),
            "token_limit_rate": _rate(task_records, lambda row: row["response_reached_token_limit"]),
            "total_immediate_reward": float(sum(row["immediate_reward"] for row in task_records)),
            "won": any(row["won"] for row in task_records),
            "attempt_play_counts": dict(sorted(Counter(row["attempt"] for row in task_records).items())),
        })

    metrics["val/diagnostics/task_count"] = len(task_summaries)
    metrics["val/diagnostics/task_mean_play_parse_rate"] = _mean(
        row["parse_rate"] for row in task_summaries
    )
    metrics["val/diagnostics/task_mean_play_effective_rate"] = _mean(
        row["effective_rate"] for row in task_summaries if row["effective_rate"] is not None
    )
    metrics["val/diagnostics/tasks_with_zero_parsed_action_rate"] = _rate(
        task_summaries, lambda row: row["parse_rate"] == 0.0
    )
    metrics["val/diagnostics/tasks_with_zero_effective_action_rate"] = _rate(
        task_summaries, lambda row: row["effective_rate"] == 0.0
    )
    metrics["val/diagnostics/tasks_with_any_token_limit_response_rate"] = _rate(
        task_summaries, lambda row: row["token_limit_rate"] > 0.0
    )

    # None values are useful in the JSON summary but cannot be logged as scalars.
    trainer_metrics = {key: value for key, value in metrics.items() if value is not None}
    return {"metrics": trainer_metrics, "task_summaries": task_summaries}


def dump_validation_interactions(
    records: list[dict[str, Any]], summary: dict[str, Any], dump_path: str, global_step: int
) -> tuple[str, str]:
    """Write every interaction and its aggregate summary before validation returns."""
    os.makedirs(dump_path, exist_ok=True)
    stem = f"step_{global_step:06d}"
    interactions_path = os.path.join(dump_path, f"{stem}_interactions.jsonl")
    summary_path = os.path.join(dump_path, f"{stem}_summary.json")

    with open(interactions_path, "w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    payload = {
        "global_step": global_step,
        "interaction_file": os.path.basename(interactions_path),
        **summary,
    }
    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    return interactions_path, summary_path
