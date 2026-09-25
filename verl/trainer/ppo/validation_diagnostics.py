"""Detailed, read-only diagnostics for agent validation rollouts."""

from __future__ import annotations

import hashlib
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


def collect_validation_interactions(
    data, tokenizer, include_heavy_fields: bool = True
) -> list[dict[str, Any]]:
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
        response = tokenizer.decode(
            response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        phase = str(_metadata_at(data, "phase", index, "unknown"))
        attempt_zero_based = int(_metadata_at(data, "traj_idx", index, -1))
        turn_zero_based = int(_metadata_at(data, "turn_idx", index, -1))
        response_tokens = int(response_mask.sum().item())

        anchor_obs = _metadata_at(data, "anchor_obs", index)
        next_anchor_obs = _metadata_at(data, "next_anchor_obs", index)
        environment_effective_flag = _metadata_at(data, "action_is_effective", index)
        observation_changed = (
            anchor_obs != next_anchor_obs
            if phase == "play" and anchor_obs is not None and next_anchor_obs is not None
            else None
        )

        record = {
            "uid": str(_metadata_at(data, "uid", index, "")),
            "traj_uid": str(_metadata_at(data, "traj_uid", index, "")),
            "task_type": _metadata_at(data, "task_type", index),
            "gamefile": _metadata_at(data, "gamefile", index),
            "phase": phase,
            "attempt": attempt_zero_based + 1,
            "attempt_zero_based": attempt_zero_based,
            "turn": turn_zero_based + 1,
            "turn_zero_based": turn_zero_based,
            "is_parse_valid": bool(_metadata_at(data, "is_action_valid", index, False)),
            # The pre/post observation comparison is the environment-agnostic
            # effectiveness signal. Keep board_changed as a compatibility alias.
            # The environment flag can be stale when MineField returns early.
            "is_effective": observation_changed,
            "observation_changed": observation_changed,
            "board_changed": observation_changed,
            "environment_action_is_effective": environment_effective_flag,
            "done": bool(_metadata_at(data, "action_done", index, False)),
            "won": bool(_metadata_at(data, "action_won", index, False)),
            "immediate_reward": float(_metadata_at(data, "rewards", index, 0.0)),
            "prompt_tokens": int(prompt_mask.sum().item()),
            "response_tokens": response_tokens,
            "response_token_limit": int(response_width),
            "response_reached_token_limit": response_tokens >= response_width,
            "has_action_open_tag": "<action>" in response,
            "has_complete_action_tag": "<action>" in response and "</action>" in response,
            "has_complete_reflection_tag": "<remark>" in response and "</remark>" in response,
            "parsed_action": _metadata_at(data, "parsed_action", index),
            "anchor_obs": anchor_obs,
            "next_anchor_obs": next_anchor_obs,
            "previous_reflections": _metadata_at(data, "previous_reflections", index, []),
            "resolved_sampling_params": _metadata_at(
                data, "resolved_sampling_params", index
            ),
            "rollout_model_path": _metadata_at(data, "rollout_model_path", index),
            "active_lora_ids": _metadata_at(data, "active_lora_ids", index, []),
            "lora_sync_fingerprint": _metadata_at(
                data, "lora_sync_fingerprint", index
            ),
            "generation_do_sample": _metadata_at(
                data, "generation_do_sample", index
            ),
            "generation_validate": _metadata_at(
                data, "generation_validate", index
            ),
            "generation_engine_seed": _metadata_at(
                data, "generation_engine_seed", index
            ),
            "generation_finish_reason": _metadata_at(
                data, "generation_finish_reason", index
            ),
            "generation_stop_reason": _metadata_at(
                data, "generation_stop_reason", index
            ),
            "response": response,
        }
        if include_heavy_fields:
            record.update({
                "prompt_token_ids": prompt_ids.tolist(),
                "generated_token_ids": response_ids.tolist(),
                "model_facing_prompt": tokenizer.decode(
                    prompt_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "raw_decoded_response": tokenizer.decode(
                    response_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "prompt": tokenizer.decode(
                    prompt_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ),
            })
        if record["is_effective"] is not None:
            record["is_effective"] = bool(record["is_effective"])
            record["observation_changed"] = bool(record["observation_changed"])
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
            "task_type": task_records[0].get("task_type"),
            "gamefile": task_records[0].get("gamefile"),
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
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    dump_path: str,
    global_step: int,
    validation_metrics: dict[str, Any] | None = None,
    dump_all_interactions: bool = True,
) -> tuple[str | None, str]:
    """Write validation metrics/summary and, when requested, every interaction."""
    os.makedirs(dump_path, exist_ok=True)
    stem = f"step_{global_step:06d}"
    interactions_path = os.path.join(dump_path, f"{stem}_interactions.jsonl")
    summary_path = os.path.join(dump_path, f"{stem}_summary.json")
    metrics_path = os.path.join(dump_path, f"{stem}_metrics.json")

    interaction_file = None
    if dump_all_interactions:
        interaction_file = os.path.basename(interactions_path)
        with open(interactions_path, "w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics_file = None
    if validation_metrics is not None:
        metrics_file = os.path.basename(metrics_path)
        with open(metrics_path, "w", encoding="utf-8") as stream:
            json.dump(_json_value(validation_metrics), stream, ensure_ascii=False, indent=2)
            stream.write("\n")

    payload = {
        "global_step": global_step,
        "interaction_file": interaction_file,
        "metrics_file": metrics_file,
        "validation_metrics": _json_value(validation_metrics or {}),
        **summary,
    }
    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    return interactions_path if dump_all_interactions else None, summary_path


def _markdown_block(value: Any) -> str:
    text = str(value if value is not None else "")
    return "\n".join(f"    {line}" for line in text.splitlines()) or "    "


def _trajectory_key(record: dict[str, Any]) -> tuple[str, str]:
    identity = str(record.get("gamefile") or record.get("uid") or record.get("traj_uid"))
    task_type = str(record.get("task_type") or "unknown")
    return task_type, identity


def select_validation_trajectory_keys(
    records: list[dict[str, Any]], samples_per_task: int, sample_seed: int
) -> set[tuple[str, str]]:
    """Select stable trajectory identities without relying on Python's hash seed."""
    by_task: dict[str, set[str]] = defaultdict(set)
    for record in records:
        task_type, identity = _trajectory_key(record)
        by_task[task_type].add(identity)

    selected: set[tuple[str, str]] = set()
    for task_type in sorted(by_task):
        identities = sorted(
            by_task[task_type],
            key=lambda identity: hashlib.sha256(
                f"{sample_seed}:{identity}".encode("utf-8")
            ).hexdigest(),
        )
        selected.update(
            (task_type, identity) for identity in identities[:samples_per_task]
        )
    return selected


def attach_prompts_to_selected_trajectories(
    all_records: list[dict[str, Any]],
    new_records: list[dict[str, Any]],
    data,
    tokenizer,
    samples_per_task: int,
    sample_seed: int,
) -> None:
    """Decode prompts only for the currently selected trajectory samples."""
    selected = select_validation_trajectory_keys(
        all_records, samples_per_task, sample_seed
    )
    for record in all_records:
        if _trajectory_key(record) not in selected:
            record.pop("prompt", None)

    responses = data.batch["responses"]
    prompts = data.batch["prompts"]
    attention_mask = data.batch["attention_mask"]
    response_width = responses.shape[-1]
    for index, record in enumerate(new_records):
        if _trajectory_key(record) not in selected:
            continue
        prompt_mask = attention_mask[index, :-response_width].bool()
        prompt_ids = prompts[index][prompt_mask]
        record["prompt"] = tokenizer.decode(
            prompt_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


def dump_validation_trajectory_samples(
    records: list[dict[str, Any]],
    dump_path: str,
    global_step: int,
    samples_per_task: int,
    sample_seed: int = 0,
) -> str | None:
    """Write deterministic, task-balanced, human-readable trajectory samples."""
    if samples_per_task <= 0:
        return None

    selected_keys = select_validation_trajectory_keys(
        records, samples_per_task, sample_seed
    )
    by_trajectory: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = _trajectory_key(record)
        if key in selected_keys:
            by_trajectory[key].append(record)

    selected: list[tuple[str, str, list[dict[str, Any]]]] = []
    for task_type, identity in sorted(selected_keys):
        selected.append((task_type, identity, by_trajectory[(task_type, identity)]))

    os.makedirs(dump_path, exist_ok=True)
    output_path = os.path.join(
        dump_path, f"step_{global_step:06d}_trajectory_samples.md"
    )
    with open(output_path, "w", encoding="utf-8") as stream:
        stream.write("# Validation trajectory samples\n\n")
        stream.write(
            f"Selected deterministically by task type with seed {sample_seed}; "
            "this file contains only the sampled trajectories.\n\n"
        )
        for task_type, identity, trajectory_records in selected:
            won = any(record.get("won", False) for record in trajectory_records)
            stream.write(f"## {task_type}\n\n")
            stream.write(f"- Game: `{identity}`\n")
            stream.write(f"- Success within allowed attempts: `{won}`\n\n")
            first_play = next(
                (record for record in trajectory_records if record.get("phase") == "play"),
                None,
            )
            if first_play is not None:
                stream.write("### Initial observation\n\n")
                stream.write(_markdown_block(first_play.get("anchor_obs")) + "\n\n")

            for record in trajectory_records:
                phase = record.get("phase", "unknown")
                attempt = record.get("attempt", "?")
                turn = record.get("turn", "?")
                stream.write(f"### Attempt {attempt}, {phase}, turn {turn}\n\n")
                stream.write("Prompt sent to the model:\n\n")
                stream.write(_markdown_block(record.get("prompt", "unavailable")) + "\n\n")
                stream.write("Model response:\n\n")
                stream.write(_markdown_block(record.get("response")) + "\n\n")
                stream.write("Parsed action/reflection:\n\n")
                stream.write(_markdown_block(record.get("parsed_action")) + "\n\n")
                if phase == "play":
                    stream.write("Resulting observation:\n\n")
                    stream.write(_markdown_block(record.get("next_anchor_obs")) + "\n\n")
                stream.write(
                    "Outcome: "
                    f"parse_valid={record.get('is_parse_valid')}, "
                    f"observation_changed={record.get('observation_changed')}, "
                    f"done={record.get('done')}, won={record.get('won')}, "
                    f"reward={record.get('immediate_reward')}\n\n"
                )

    return output_path
