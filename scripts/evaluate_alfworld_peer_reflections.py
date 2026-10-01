#!/usr/bin/env python3
"""Frozen-policy ALFWorld diagnostic for failed-peer reflections.

Attempt one and its reflection are generated exactly once per rollout. The
same failed recipient is then retried from the initial state with nested sets
of failed same-task peer information. The appended-reflection variant adds
peer reflections to the retry prompt. The assisted-reflection variant instead
generates one reflection from the recipient history and peer histories, then
uses only that reflection in the normal retry prompt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import ray
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from agent_system.environments.alfworld.env_manager import AlfWorldEnvironmentManager
from agent_system.environments.alfworld.envs import build_alfworld_envs
from agent_system.environments.alfworld.projection import alfworld_projection
from agent_system.environments.alfworld.prompt import (
    format_failed_peer_histories,
    format_failed_peer_reflections,
    format_reflection_merge_prompt,
    format_unrelated_failed_reflections,
)


APPENDED_CONDITIONS = (
    ("self_k0", "self", 0),
    ("failed_peer_k1", "same_task", 1),
    ("failed_peer_k2", "same_task", 2),
    ("failed_peer_k4", "same_task", 4),
    ("unrelated_k2", "unrelated", 2),
)

ASSISTED_CONDITIONS = (
    ("assisted_k0", "same_task", 0),
    ("assisted_k1", "same_task", 1),
    ("assisted_k2", "same_task", 2),
    ("assisted_k4", "same_task", 4),
)

MERGED_CONDITIONS = (
    ("self_k0", "self", 0),
    ("merged_same_k4", "same_task", 4),
    ("merged_unrelated_k4", "unrelated", 4),
)


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF


def json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=json_default) + "\n")


def append_jsonl(stream, value: dict[str, Any]) -> None:
    stream.write(json.dumps(value, default=json_default) + "\n")
    stream.flush()


def task_identifier(gamefile: str) -> str:
    return hashlib.sha1(task_instance_key(gamefile).encode()).hexdigest()[:12]


def task_instance_key(gamefile: str) -> str:
    """Return a machine-independent path identifying one ALFWorld game."""
    parts = Path(gamefile).parts
    if "json" in parts:
        return "/".join(parts[parts.index("json") + 1:])
    return "/".join(parts[-4:])


class FrozenVLLMGenerator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=False
        )
        self.llm = LLM(
            model=args.model_path,
            tokenizer=args.model_path,
            trust_remote_code=False,
            tensor_parallel_size=1,
            dtype="bfloat16",
            load_format="safetensors",
            seed=args.engine_seed,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_model_len=args.max_model_len,
            enable_chunked_prefill=False,
            enforce_eager=args.enforce_eager,
        )

    def chat_prompt(self, prompt: str) -> str:
        chat = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False
            )

    def generate(
        self, requests: list[tuple[int, str, int | None]]
    ) -> dict[int, dict[str, Any]]:
        if not requests:
            return {}
        chat_prompts = [self.chat_prompt(prompt) for _, prompt, _ in requests]
        sampling = [
            SamplingParams(
                n=1,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                top_k=self.args.top_k,
                seed=seed,
                max_tokens=self.args.max_response_tokens,
                min_tokens=5,
            )
            for _, _, seed in requests
        ]
        outputs = self.llm.generate(chat_prompts, sampling_params=sampling, use_tqdm=False)
        generated = {}
        for (slot, prompt, seed), output in zip(requests, outputs):
            candidate = output.outputs[0]
            generated[slot] = {
                "prompt": prompt,
                "response": candidate.text,
                "input_tokens": len(output.prompt_token_ids),
                "output_tokens": len(candidate.token_ids),
                "request_generation_seed": seed,
                "finish_reason": candidate.finish_reason,
                "stop_reason": candidate.stop_reason,
            }
        return generated


def run_play_attempt(
    manager: AlfWorldEnvironmentManager,
    observation: dict[str, Any],
    active: np.ndarray,
    generator: FrozenVLLMGenerator,
    rollout_ids: list[str],
    generation_seed: int | None,
    max_turns: int,
) -> tuple[list[list[dict[str, Any]]], np.ndarray, np.ndarray]:
    trajectories: list[list[dict[str, Any]]] = [
        [] for _ in range(manager.num_processes)
    ]
    succeeded = np.zeros(manager.num_processes, dtype=bool)
    total_rewards = np.zeros(manager.num_processes, dtype=float)

    for turn in range(max_turns):
        requests = [
            (
                slot,
                observation["text"][slot],
                (
                    stable_seed(generation_seed, rollout_ids[slot], "play", turn)
                    if generation_seed is not None else None
                ),
            )
            for slot in range(manager.num_processes)
            if active[slot]
        ]
        generated = generator.generate(requests)
        responses = [
            generated[slot]["response"] if slot in generated else ""
            for slot in range(manager.num_processes)
        ]
        next_observation, rewards, dones, infos = manager.step(responses, phase="play")
        rewards = np.asarray(rewards, dtype=float)
        dones = np.asarray(dones, dtype=bool)

        for slot, generation in generated.items():
            info = infos[slot]
            reward = float(rewards[slot])
            won = bool(info.get("won", False))
            total_rewards[slot] += reward
            succeeded[slot] = succeeded[slot] or won
            trajectories[slot].append(
                {
                    "turn": turn + 1,
                    **generation,
                    "parsed_action": info.get("diagnostic_parsed_action"),
                    "action_parse_valid": bool(info.get("is_action_valid", False)),
                    "action_effective": bool(info.get("action_is_effective", False)),
                    "observation_before": observation["anchor"][slot],
                    "observation_after": next_observation["anchor"][slot],
                    "reward": reward,
                    "done": bool(dones[slot]),
                    "won": won,
                }
            )

        active = active & ~dones & ~succeeded
        observation = next_observation
        if not active.any():
            break

    return trajectories, succeeded, total_rewards


def generate_reflections(
    manager: AlfWorldEnvironmentManager,
    generator: FrozenVLLMGenerator,
    active: np.ndarray | None = None,
    additional_contexts: list[str] | None = None,
    attempt1_turn_idx: int | None = None,
) -> list[dict[str, Any] | None]:
    if active is None:
        active = np.ones(manager.num_processes, dtype=bool)
    if additional_contexts is None:
        additional_contexts = ["" for _ in range(manager.num_processes)]
    if attempt1_turn_idx is None:
        observation, _ = manager.reflect(additional_contexts)
    else:
        observation, _ = manager.reflect_for_paired_retry(
            attempt1_turn_idx, additional_contexts
        )
    requests = [
        (
            slot,
            observation["text"][slot],
            None,
        )
        for slot in range(manager.num_processes)
        if active[slot]
    ]
    generated = generator.generate(requests)
    responses = [
        generated[slot]["response"] if slot in generated else ""
        for slot in range(manager.num_processes)
    ]
    manager.step(responses, phase="reflect")
    records: list[dict[str, Any] | None] = [None] * manager.num_processes
    for slot, generation in generated.items():
        records[slot] = {
            **generation,
            "parsed_reflection": manager.reflections[slot].get(0, ""),
            "reflection_parse_valid": bool(manager.reflections[slot].get(0, "")),
        }
    return records


def generate_merged_reflections(
    generator: FrozenVLLMGenerator,
    prompts: list[str],
    active: np.ndarray,
) -> list[dict[str, Any] | None]:
    """Generate and parse exactly one merged reflection for every active slot."""
    requests = [
        (slot, prompts[slot], None)
        for slot in range(len(prompts))
        if active[slot]
    ]
    generated = generator.generate(requests)
    responses = [
        generated[slot]["response"] if slot in generated else ""
        for slot in range(len(prompts))
    ]
    reflections, valids = alfworld_projection(responses, phase="reflect")
    records: list[dict[str, Any] | None] = [None] * len(prompts)
    for slot, generation in generated.items():
        records[slot] = {
            **generation,
            "parsed_reflection": reflections[slot],
            "reflection_parse_valid": bool(valids[slot]),
        }
    return records


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def bootstrap_task_mean(
    task_values: dict[str, float], samples: int, seed: int
) -> list[float | None]:
    if not task_values or samples <= 0:
        return [None, None]
    task_ids = sorted(task_values)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        selected = [rng.choice(task_ids) for _ in task_ids]
        estimates.append(statistics.fmean(task_values[task] for task in selected))
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def paired_contrast(
    records_by_condition: dict[str, list[dict[str, Any]]],
    left: str,
    right: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    right_map = {
        (row["recipient_rollout_id"], row["retry_generation_seed"]): row
        for row in records_by_condition.get(right, [])
        if row["evaluated"]
    }
    pairs = []
    for row in records_by_condition.get(left, []):
        if not row["evaluated"]:
            continue
        other = right_map.get(
            (row["recipient_rollout_id"], row["retry_generation_seed"])
        )
        if other is not None:
            pairs.append((row, other))
    per_task: dict[str, list[float]] = defaultdict(list)
    for left_row, right_row in pairs:
        per_task[left_row["task_id"]].append(
            float(left_row["attempt2_success"])
            - float(right_row["attempt2_success"])
        )
    task_values = {
        task: statistics.fmean(values) for task, values in per_task.items()
    }
    pooled = statistics.fmean(
        float(a["attempt2_success"]) - float(b["attempt2_success"])
        for a, b in pairs
    ) if pairs else None
    macro = statistics.fmean(task_values.values()) if task_values else None
    return {
        "left": left,
        "right": right,
        "paired_evaluations": len(pairs),
        "paired_tasks": len(task_values),
        "pooled_gain": pooled,
        "task_macro_gain": macro,
        "task_macro_gain_95ci": bootstrap_task_mean(
            task_values, bootstrap_samples, bootstrap_seed
        ),
    }


def summarize(
    retry_records: list[dict[str, Any]],
    selected_recipient_count: int,
    total_failed_count: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    conditions: tuple[tuple[str, str, int], ...],
    baseline_condition: str,
    contrast_pairs: tuple[tuple[str, str], ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in retry_records:
        by_condition[row["condition"]].append(row)
    baseline_map = {
        (row["recipient_rollout_id"], row["retry_generation_seed"]): row
        for row in by_condition[baseline_condition]
        if row["evaluated"]
    }
    condition_summaries = []
    task_rows = []

    for condition, _, peers_requested in conditions:
        all_rows = by_condition[condition]
        evaluated = [row for row in all_rows if row["evaluated"]]
        eligible_ids = {row["recipient_rollout_id"] for row in evaluated}
        skipped_ids = {
            row["recipient_rollout_id"] for row in all_rows if not row["evaluated"]
        }
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in evaluated:
            by_task[row["task_id"]].append(row)

        task_success = {}
        task_gain = {}
        for task_id, rows in by_task.items():
            success_rate = statistics.fmean(float(row["attempt2_success"]) for row in rows)
            paired = [
                (row, baseline_map[(row["recipient_rollout_id"], row["retry_generation_seed"])])
                for row in rows
                if (row["recipient_rollout_id"], row["retry_generation_seed"])
                in baseline_map
            ]
            gain = statistics.fmean(
                float(row["attempt2_success"]) - float(self_row["attempt2_success"])
                for row, self_row in paired
            ) if paired else None
            task_success[task_id] = success_rate
            if gain is not None:
                task_gain[task_id] = gain
            first = rows[0]
            task_rows.append(
                {
                    "condition": condition,
                    "task_id": task_id,
                    "task_type": first["task_type"],
                    "gamefile": first["gamefile"],
                    "evaluations": len(rows),
                    "success_count": sum(bool(row["attempt2_success"]) for row in rows),
                    "success_rate": success_rate,
                    "paired_gain_vs_baseline": gain,
                }
            )

        paired_rows = [
            (row, baseline_map[(row["recipient_rollout_id"], row["retry_generation_seed"])])
            for row in evaluated
            if (row["recipient_rollout_id"], row["retry_generation_seed"])
            in baseline_map
        ]
        pooled_rate = statistics.fmean(
            float(row["attempt2_success"]) for row in evaluated
        ) if evaluated else None
        pooled_gain = statistics.fmean(
            float(row["attempt2_success"]) - float(self_row["attempt2_success"])
            for row, self_row in paired_rows
        ) if paired_rows else None
        paired_baseline_success_count = sum(
            bool(baseline_row["attempt2_success"])
            for _, baseline_row in paired_rows
        )
        rescued_count = sum(
            not bool(baseline_row["attempt2_success"])
            and bool(row["attempt2_success"])
            for row, baseline_row in paired_rows
        )
        harmed_count = sum(
            bool(baseline_row["attempt2_success"])
            and not bool(row["attempt2_success"])
            for row, baseline_row in paired_rows
        )

        def mean_field(field: str) -> float | None:
            values = [row[field] for row in evaluated if row.get(field) is not None]
            return statistics.fmean(values) if values else None

        condition_summaries.append(
            {
                "condition": condition,
                "baseline_condition": baseline_condition,
                "peers_requested": peers_requested,
                "total_failed_attempt1_rollouts": total_failed_count,
                "selected_failed_recipients": selected_recipient_count,
                "eligible_recipients": len(eligible_ids),
                "skipped_recipients": len(skipped_ids),
                "evaluated_recipient_seed_pairs": len(evaluated),
                "success_count": sum(bool(row["attempt2_success"]) for row in evaluated),
                "pooled_recipient_success_rate": pooled_rate,
                "paired_baseline_success_count": paired_baseline_success_count,
                "paired_baseline_success_rate": (
                    paired_baseline_success_count / len(paired_rows)
                    if paired_rows else None
                ),
                "pooled_paired_gain_vs_baseline": pooled_gain,
                "pooled_paired_gain_vs_k0": pooled_gain,
                "paired_rescued_count": rescued_count,
                "paired_harmed_count": harmed_count,
                "paired_net_success_count": rescued_count - harmed_count,
                "task_count": len(task_success),
                "task_macro_success_rate": (
                    statistics.fmean(task_success.values()) if task_success else None
                ),
                "task_macro_success_95ci": bootstrap_task_mean(
                    task_success,
                    bootstrap_samples,
                    stable_seed(bootstrap_seed, condition, "success"),
                ),
                "task_macro_paired_gain_vs_baseline": (
                    statistics.fmean(task_gain.values()) if task_gain else None
                ),
                "task_macro_paired_gain_vs_k0": (
                    statistics.fmean(task_gain.values()) if task_gain else None
                ),
                "task_macro_gain_95ci": bootstrap_task_mean(
                    task_gain,
                    bootstrap_samples,
                    stable_seed(bootstrap_seed, condition, "gain"),
                ),
                "mean_peers_used": (
                    statistics.fmean(row["peers_used"] for row in evaluated)
                    if evaluated else None
                ),
                "mean_reflection_generation_input_tokens": mean_field(
                    "reflection_generation_input_tokens"
                ),
                "mean_reflection_generation_output_tokens": mean_field(
                    "reflection_generation_output_tokens"
                ),
                "mean_merge_generation_input_tokens": mean_field(
                    "merge_generation_input_tokens"
                ),
                "mean_merge_generation_output_tokens": mean_field(
                    "merge_generation_output_tokens"
                ),
                "mean_retry_play_input_tokens": mean_field(
                    "attempt2_total_input_tokens"
                ),
                "mean_retry_play_output_tokens": mean_field(
                    "attempt2_total_output_tokens"
                ),
                "mean_total_tokens": mean_field("total_tokens"),
            }
        )

    contrasts = [
        paired_contrast(
            by_condition,
            left,
            right,
            bootstrap_samples,
            stable_seed(bootstrap_seed, left, right),
        )
        for left, right in contrast_pairs
    ]
    return {"conditions": condition_summaries, "contrasts": contrasts}, task_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def qualitative_category(condition_rows: dict[str, dict[str, Any]]) -> str:
    self_success = bool(condition_rows["self_k0"]["attempt2_success"])
    peer_success = bool(condition_rows["failed_peer_k2"]["attempt2_success"])
    unrelated_success = bool(condition_rows["unrelated_k2"]["attempt2_success"])
    if not self_success and peer_success and not unrelated_success:
        return "failed_peers_help_specifically"
    if not self_success and peer_success:
        return "failed_peers_help"
    if self_success and not peer_success:
        return "failed_peers_hurt"
    if peer_success != unrelated_success:
        return "peer_vs_unrelated_difference"
    if not self_success and not peer_success and not unrelated_success:
        return "persistent_failure"
    return "consistent_success"


def select_qualitative_examples(
    retry_records: list[dict[str, Any]],
    attempt1_rows: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in retry_records:
        if row["evaluated"]:
            grouped[(row["recipient_rollout_id"], row["retry_generation_seed"])][
                row["condition"]
            ] = row
    required = {"self_k0", "failed_peer_k2", "unrelated_k2"}
    candidates = []
    for key, conditions in grouped.items():
        if required <= conditions.keys():
            candidates.append((qualitative_category(conditions), key, conditions))

    category_order = (
        "failed_peers_help_specifically",
        "failed_peers_help",
        "failed_peers_hurt",
        "peer_vs_unrelated_difference",
        "persistent_failure",
        "consistent_success",
    )
    attempt1_by_id = {row["recipient_rollout_id"]: row for row in attempt1_rows}
    selected = []
    used_recipients = set()
    used_tasks = set()

    def add_candidate(candidate, prefer_new_task: bool) -> bool:
        category, (recipient_id, retry_seed), conditions = candidate
        task_id = conditions["self_k0"]["task_id"]
        if recipient_id in used_recipients:
            return False
        if prefer_new_task and task_id in used_tasks:
            return False
        selected.append(
            {
                "selection_category": category,
                "recipient_rollout_id": recipient_id,
                "retry_generation_seed": retry_seed,
                "attempt1": attempt1_by_id[recipient_id],
                "conditions": {
                    name: conditions[name] for name in sorted(conditions)
                },
            }
        )
        used_recipients.add(recipient_id)
        used_tasks.add(task_id)
        return True

    for category in category_order:
        for candidate in candidates:
            if candidate[0] == category and add_candidate(candidate, True):
                break
        if len(selected) >= limit:
            return selected
    for prefer_new_task in (True, False):
        for candidate in candidates:
            if add_candidate(candidate, prefer_new_task) and len(selected) >= limit:
                return selected
    return selected


def compact_text(value: Any, limit: int = 500) -> str:
    text = "" if value is None else str(value).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def write_qualitative_markdown(
    path: Path, examples: list[dict[str, Any]], json_filename: str
) -> None:
    lines = [
        "# ALFWorld failed-peer reflection examples",
        "",
        "These examples are selected by outcome pattern, not at random. Full prompts,",
        f"responses, observations, and provenance are stored in `{json_filename}`.",
    ]
    for index, example in enumerate(examples, start=1):
        attempt1 = example["attempt1"]
        lines.extend(
            [
                "",
                f"## Example {index}: {example['selection_category']}",
                "",
                f"- Task: `{attempt1['task_id']}` ({attempt1['task_type']})",
                f"- Recipient: `{example['recipient_rollout_id']}`",
                f"- Retry seed: `{example['retry_generation_seed']}`",
                f"- Instruction: {attempt1['task_description']}",
                "",
                "### Attempt 1 actions",
                "",
            ]
        )
        for step in attempt1["attempt1_trajectory"]:
            lines.append(
                f"{step['turn']}. `{step['parsed_action']}` → "
                f"{compact_text(step['observation_after'], 300)}"
            )
        lines.extend(
            [
                "",
                "### Own reflection",
                "",
                compact_text(attempt1["own_reflection"], 1200),
            ]
        )
        for condition in ("self_k0", "failed_peer_k2", "unrelated_k2"):
            row = example["conditions"][condition]
            lines.extend(
                [
                    "",
                    f"### {condition}: {'SUCCESS' if row['attempt2_success'] else 'failure'}",
                    "",
                    f"Donors: {', '.join(row['donor_rollout_ids']) or 'none'}",
                ]
            )
            for donor_id, reflection in zip(
                row["donor_rollout_ids"], row["donor_reflections"]
            ):
                lines.extend(
                    [
                        "",
                        f"- `{donor_id}`: {compact_text(reflection, 700)}",
                    ]
                )
            lines.extend(["", "Actions:", ""])
            for step in row["attempt2_trajectory"]:
                lines.append(
                    f"{step['turn']}. `{step['parsed_action']}` → "
                    f"{compact_text(step['observation_after'], 300)}"
                )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-variant",
        choices=(
            "appended_reflections",
            "assisted_reflection",
            "merged_reflections",
        ),
        default="appended_reflections",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--alfworld-data", required=True, type=Path)
    parser.add_argument("--eval-dataset", default="eval_all")
    parser.add_argument("--num-tasks", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-failed-recipients", type=int, default=160)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-response-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--validation-seed", type=int, default=1000)
    parser.add_argument("--donor-seed", type=int, default=22)
    parser.add_argument("--recipient-seed", type=int, default=23)
    parser.add_argument("--retry-seeds", default="24")
    parser.add_argument("--engine-seed", type=int, default=20)
    parser.add_argument("--bootstrap-seed", type=int, default=25)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--qualitative-examples", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ray-num-cpus", type=int, default=18)
    parser.add_argument("--env-cpus-per-worker", type=float, default=1.0)
    parser.add_argument("--games-per-worker", type=int, default=16)
    parser.add_argument("--ray-tmpdir", type=Path, default=Path("/tmp/lamer-peer-ray"))
    args = parser.parse_args()
    args.retry_seeds = [int(seed) for seed in args.retry_seeds.split(",") if seed]
    for name in (
        "num_tasks",
        "group_size",
        "max_failed_recipients",
        "max_turns",
        "max_response_tokens",
        "max_model_len",
        "max_num_batched_tokens",
        "ray_num_cpus",
        "games_per_worker",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_num_batched_tokens < args.max_model_len:
        parser.error("--max-num-batched-tokens must be at least --max-model-len")
    worker_count = (
        args.num_tasks * args.group_size + args.games_per_worker - 1
    ) // args.games_per_worker
    required_worker_cpus = worker_count * args.env_cpus_per_worker
    if required_worker_cpus > args.ray_num_cpus:
        parser.error(
            f"environment workers require {required_worker_cpus:g} Ray CPUs "
            f"({worker_count} workers), but --ray-num-cpus is {args.ray_num_cpus}; "
            "increase --games-per-worker or reduce --env-cpus-per-worker"
        )
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between 0 and 1")
    if not args.retry_seeds:
        parser.error("--retry-seeds must contain at least one integer")
    if args.qualitative_examples < 0:
        parser.error("--qualitative-examples must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    if args.experiment_variant == "assisted_reflection":
        conditions = ASSISTED_CONDITIONS
        baseline_condition = "assisted_k0"
        contrast_pairs = tuple(
            (condition, baseline_condition)
            for condition, _, peers_requested in conditions
            if peers_requested > 0
        )
    elif args.experiment_variant == "merged_reflections":
        conditions = MERGED_CONDITIONS
        baseline_condition = "self_k0"
        contrast_pairs = (
            ("merged_same_k4", baseline_condition),
            ("merged_unrelated_k4", baseline_condition),
        )
    else:
        conditions = APPENDED_CONDITIONS
        baseline_condition = "self_k0"
        contrast_pairs = (
            ("failed_peer_k2", "self_k0"),
            ("unrelated_k2", "self_k0"),
            ("failed_peer_k2", "unrelated_k2"),
        )
    os.environ["ALFWORLD_DATA"] = str(args.alfworld_data.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completion_marker = args.output_dir / "summary.json"
    if completion_marker.exists():
        raise SystemExit(
            f"A completed experiment already exists at {args.output_dir}; "
            "choose a new output directory."
        )
    args.ray_tmpdir.mkdir(parents=True, exist_ok=True)
    ray.init(
        num_cpus=args.ray_num_cpus,
        include_dashboard=False,
        _temp_dir=str(args.ray_tmpdir.resolve()),
    )

    config = OmegaConf.create(
        {
            "env": {
                "env_name": "alfworld/AlfredTWEnv",
                "max_turns": args.max_turns,
                "reflection_type": "reflection_only",
            }
        }
    )
    alf_config = Path(__file__).resolve().parents[1] / (
        "agent_system/environments/alfworld/configs/config_tw.yaml"
    )
    envs = build_alfworld_envs(
        str(alf_config),
        args.validation_seed,
        args.num_tasks,
        args.group_size,
        is_train=False,
        env_kwargs={
            "eval_dataset": args.eval_dataset,
            "num_cpus_per_worker": args.env_cpus_per_worker,
            "num_gpus_per_worker": 0,
            "games_per_worker": args.games_per_worker,
            "preserve_game_order": True,
        },
    )
    manager = AlfWorldEnvironmentManager(
        envs, alfworld_projection, num_attempts=2, do_reflection=True, config=config
    )

    started = datetime.now(timezone.utc)
    retry_records: list[dict[str, Any]] = []
    attempt1_path = args.output_dir / "attempt1_rollouts.jsonl"
    donor_plan_path = args.output_dir / "donor_assignments.jsonl"
    reflection_path = args.output_dir / "reflection_generations.jsonl"
    retry_path = args.output_dir / "retry_evaluations.jsonl"

    try:
        observation, _ = manager.reset()
        expected = args.num_tasks * args.group_size
        if manager.num_processes != expected:
            raise RuntimeError(
                f"Expected {expected} repeated evaluation environments, got "
                f"{manager.num_processes}"
            )
        rollout_ids = [
            f"task{task_index:04d}_rollout{rollout_index:02d}"
            for task_index in range(args.num_tasks)
            for rollout_index in range(args.group_size)
        ]
        task_ids = [task_identifier(gamefile) for gamefile in manager.gamefile]
        task_instance_keys = [task_instance_key(gamefile) for gamefile in manager.gamefile]
        for start in range(0, expected, args.group_size):
            if len(set(manager.gamefile[start:start + args.group_size])) != 1:
                raise RuntimeError("Evaluation group does not contain one repeated task")

        # Validate environment grouping before allocating the model and KV cache.
        generator = FrozenVLLMGenerator(args)

        attempt1_trajectories, attempt1_success, attempt1_rewards = run_play_attempt(
            manager,
            observation,
            np.ones(expected, dtype=bool),
            generator,
            rollout_ids,
            None,
            args.max_turns,
        )
        failed = ~attempt1_success
        attempt1_turn_idx = manager.curr_turn_idx
        attempt1_histories, _ = manager.memories[0].fetch(
            history_length=args.max_turns
        )
        base_reflection_records = generate_reflections(
            manager,
            generator,
        )
        base_reflections = [
            record["parsed_reflection"] if record is not None else ""
            for record in base_reflection_records
        ]

        attempt1_rows = []
        with attempt1_path.open("w") as stream:
            for slot in range(expected):
                row = {
                    "checkpoint": str(Path(args.model_path).resolve())
                    if Path(args.model_path).exists() else args.model_path,
                    "task_id": task_ids[slot],
                    "task_instance_identifier": task_instance_keys[slot],
                    "task_description": manager.tasks[slot],
                    "task_type": manager.task_types[slot],
                    "gamefile": manager.gamefile[slot],
                    "recipient_rollout_id": rollout_ids[slot],
                    "task_group_index": slot // args.group_size,
                    "rollout_index": slot % args.group_size,
                    "attempt1_reward": float(attempt1_rewards[slot]),
                    "attempt1_success": bool(attempt1_success[slot]),
                    "attempt1_seed_mode": "vllm_engine",
                    "attempt1_engine_seed": args.engine_seed,
                    "attempt1_trajectory": attempt1_trajectories[slot],
                    "reflection": base_reflection_records[slot],
                    "own_reflection": base_reflections[slot],
                }
                attempt1_rows.append(row)
                append_jsonl(stream, row)

        failed_slots = [slot for slot in range(expected) if failed[slot]]
        same_task_orders: dict[int, list[int]] = {}
        unrelated_orders: dict[int, list[int]] = {}
        for slot in failed_slots:
            same = [
                donor for donor in failed_slots
                if task_ids[donor] == task_ids[slot] and donor != slot
            ]
            unrelated = [
                donor for donor in failed_slots if task_ids[donor] != task_ids[slot]
            ]
            random.Random(stable_seed(args.donor_seed, rollout_ids[slot], "same")).shuffle(same)
            random.Random(stable_seed(args.donor_seed, rollout_ids[slot], "unrelated")).shuffle(unrelated)
            same_task_orders[slot] = same
            unrelated_orders[slot] = unrelated

        if args.experiment_variant in {"assisted_reflection", "merged_reflections"}:
            selection_pool = [
                slot for slot in failed_slots if len(same_task_orders[slot]) >= 4
            ]
        else:
            selection_pool = list(failed_slots)
        selected_slots = list(selection_pool)
        random.Random(args.recipient_seed).shuffle(selected_slots)
        selected_slots = selected_slots[: args.max_failed_recipients]
        selected_set = set(selected_slots)

        with donor_plan_path.open("w") as stream:
            for slot in selected_slots:
                append_jsonl(
                    stream,
                    {
                        "recipient_rollout_id": rollout_ids[slot],
                        "task_id": task_ids[slot],
                        "donor_permutation_seed": args.donor_seed,
                        "failed_same_task_donor_order": [
                            rollout_ids[donor] for donor in same_task_orders[slot]
                        ],
                        "failed_unrelated_task_donor_order": [
                            rollout_ids[donor] for donor in unrelated_orders[slot]
                        ],
                    },
                )

        with (
            retry_path.open("w") as retry_stream,
            reflection_path.open("w") as reflection_stream,
        ):
            for condition, donor_kind, peers_requested in conditions:
                donor_slots: dict[int, list[int]] = {}
                skip_reasons: dict[int, str] = {}
                for slot in selected_slots:
                    pool = (
                        same_task_orders[slot]
                        if donor_kind == "same_task"
                        else unrelated_orders[slot]
                    )
                    if donor_kind == "self":
                        donor_slots[slot] = []
                    elif len(pool) < peers_requested:
                        skip_reasons[slot] = (
                            f"only {len(pool)} eligible {donor_kind} failed donors; "
                            f"{peers_requested} required"
                        )
                    else:
                        donor_slots[slot] = pool[:peers_requested]

                play_contexts = ["" for _ in range(expected)]
                merge_reflection_records: list[dict[str, Any] | None] = [
                    None for _ in range(expected)
                ]
                if args.experiment_variant == "assisted_reflection":
                    reflection_contexts = ["" for _ in range(expected)]
                    for slot, donors in donor_slots.items():
                        reflection_contexts[slot] = format_failed_peer_histories(
                            [attempt1_histories[donor] for donor in donors]
                        )
                    if peers_requested == 0:
                        condition_reflection_records = base_reflection_records
                    else:
                        condition_reflection_records = generate_reflections(
                            manager,
                            generator,
                            active=np.array(
                                [slot in donor_slots for slot in range(expected)],
                                dtype=bool,
                            ),
                            additional_contexts=reflection_contexts,
                            attempt1_turn_idx=attempt1_turn_idx,
                        )
                    condition_reflections = [
                        record["parsed_reflection"] if record is not None else ""
                        for record in condition_reflection_records
                    ]
                elif args.experiment_variant == "merged_reflections":
                    if peers_requested == 0:
                        condition_reflection_records = base_reflection_records
                    else:
                        merge_prompts = ["" for _ in range(expected)]
                        for slot, donors in donor_slots.items():
                            merge_prompts[slot] = format_reflection_merge_prompt(
                                manager.tasks[slot],
                                base_reflections[slot],
                                [base_reflections[donor] for donor in donors],
                                donor_kind,
                            )
                        merge_reflection_records = generate_merged_reflections(
                            generator,
                            merge_prompts,
                            np.array(
                                [slot in donor_slots for slot in range(expected)],
                                dtype=bool,
                            ),
                        )
                        condition_reflection_records = merge_reflection_records
                    condition_reflections = [
                        record["parsed_reflection"] if record is not None else ""
                        for record in condition_reflection_records
                    ]
                else:
                    condition_reflection_records = base_reflection_records
                    condition_reflections = base_reflections
                    for slot, donors in donor_slots.items():
                        reflections = [base_reflections[donor] for donor in donors]
                        if donor_kind == "same_task":
                            play_contexts[slot] = format_failed_peer_reflections(reflections)
                        elif donor_kind == "unrelated":
                            play_contexts[slot] = format_unrelated_failed_reflections(reflections)

                for slot in selected_slots:
                    donors = donor_slots.get(slot, [])
                    record = condition_reflection_records[slot]
                    append_jsonl(
                        reflection_stream,
                        {
                            "condition": condition,
                            "recipient_rollout_id": rollout_ids[slot],
                            "task_id": task_ids[slot],
                            "peers_used": len(donors),
                            "donor_rollout_ids": [rollout_ids[d] for d in donors],
                            "generation": record,
                            "normal_reflection_generation": (
                                base_reflection_records[slot]
                                if args.experiment_variant == "merged_reflections"
                                else None
                            ),
                            "merge_generation": merge_reflection_records[slot],
                        },
                    )

                for retry_seed in args.retry_seeds:
                    active = np.array(
                        [slot in donor_slots for slot in range(expected)], dtype=bool
                    )
                    retry_observation, _ = manager.restart_for_paired_retry(
                        condition_reflections, play_contexts
                    )
                    trajectories, successes, rewards = run_play_attempt(
                        manager,
                        retry_observation,
                        active,
                        generator,
                        rollout_ids,
                        retry_seed,
                        args.max_turns,
                    )

                    for slot in selected_slots:
                        donors = donor_slots.get(slot, [])
                        common = {
                            "checkpoint": attempt1_rows[slot]["checkpoint"],
                            "task_id": task_ids[slot],
                            "task_instance_identifier": task_instance_keys[slot],
                            "task_description": manager.tasks[slot],
                            "task_type": manager.task_types[slot],
                            "gamefile": manager.gamefile[slot],
                            "recipient_rollout_id": rollout_ids[slot],
                            "recipient_attempt1_reward": float(attempt1_rewards[slot]),
                            "recipient_attempt1_success": bool(attempt1_success[slot]),
                            "own_reflection": base_reflections[slot],
                            "generated_reflection": condition_reflections[slot],
                            "condition": condition,
                            "peers_requested": peers_requested,
                            "peers_used": len(donors),
                            "donor_rollout_ids": [rollout_ids[d] for d in donors],
                            "donor_rewards": [float(attempt1_rewards[d]) for d in donors],
                            "donor_reflections": [base_reflections[d] for d in donors],
                            "donor_task_ids": [task_ids[d] for d in donors],
                            "donor_task_types": [manager.task_types[d] for d in donors],
                            "attempt1_seed_mode": "vllm_engine",
                            "attempt1_engine_seed": args.engine_seed,
                            "reflection_seed_mode": "continued_vllm_engine_stream",
                            "reflection_engine_seed": args.engine_seed,
                            "retry_seed_mode": "paired_per_request",
                            "donor_permutation_seed": args.donor_seed,
                            "retry_generation_seed": retry_seed,
                            "attempt1_trajectory_file": attempt1_path.name,
                            "donor_assignment_file": donor_plan_path.name,
                            "reflection_generation_file": reflection_path.name,
                        }
                        reflection_record = condition_reflection_records[slot]
                        if slot in skip_reasons:
                            row = {
                                **common,
                                "evaluated": False,
                                "skip_reason": skip_reasons[slot],
                                "attempt2_prompt": None,
                                "attempt2_trajectory": [],
                                "attempt2_reward": None,
                                "attempt2_success": None,
                                "attempt2_first_prompt_tokens": None,
                                "attempt2_total_input_tokens": None,
                                "attempt2_total_output_tokens": None,
                                "reflection_generation_input_tokens": None,
                                "reflection_generation_output_tokens": None,
                                "merge_generation_input_tokens": None,
                                "merge_generation_output_tokens": None,
                                "total_tokens": None,
                            }
                        else:
                            trajectory = trajectories[slot]
                            play_input_tokens = sum(
                                step["input_tokens"] for step in trajectory
                            )
                            play_output_tokens = sum(
                                step["output_tokens"] for step in trajectory
                            )
                            if args.experiment_variant == "merged_reflections":
                                normal_reflection_record = base_reflection_records[slot]
                                reflection_input_tokens = normal_reflection_record[
                                    "input_tokens"
                                ]
                                reflection_output_tokens = normal_reflection_record[
                                    "output_tokens"
                                ]
                                merge_record = merge_reflection_records[slot]
                                merge_input_tokens = (
                                    merge_record["input_tokens"]
                                    if merge_record is not None else 0
                                )
                                merge_output_tokens = (
                                    merge_record["output_tokens"]
                                    if merge_record is not None else 0
                                )
                            else:
                                reflection_input_tokens = reflection_record["input_tokens"]
                                reflection_output_tokens = reflection_record["output_tokens"]
                                merge_input_tokens = 0
                                merge_output_tokens = 0
                            row = {
                                **common,
                                "evaluated": True,
                                "skip_reason": None,
                                "attempt2_prompt": trajectory[0]["prompt"] if trajectory else None,
                                "attempt2_trajectory": trajectory,
                                "attempt2_reward": float(rewards[slot]),
                                "attempt2_success": bool(successes[slot]),
                                "attempt2_first_prompt_tokens": (
                                    trajectory[0]["input_tokens"] if trajectory else 0
                                ),
                                "attempt2_total_input_tokens": play_input_tokens,
                                "attempt2_total_output_tokens": play_output_tokens,
                                "reflection_generation_input_tokens": reflection_input_tokens,
                                "reflection_generation_output_tokens": reflection_output_tokens,
                                "merge_generation_input_tokens": merge_input_tokens,
                                "merge_generation_output_tokens": merge_output_tokens,
                                "total_tokens": (
                                    reflection_input_tokens
                                    + reflection_output_tokens
                                    + merge_input_tokens
                                    + merge_output_tokens
                                    + play_input_tokens
                                    + play_output_tokens
                                ),
                            }
                        retry_records.append(row)
                        append_jsonl(retry_stream, row)

        results, task_rows = summarize(
            retry_records,
            len(selected_set),
            len(failed_slots),
            args.bootstrap_samples,
            args.bootstrap_seed,
            conditions,
            baseline_condition,
            contrast_pairs,
        )
        qualitative_examples = (
            select_qualitative_examples(
                retry_records, attempt1_rows, args.qualitative_examples
            )
            if args.experiment_variant == "appended_reflections"
            else []
        )
        qualitative_json = args.output_dir / "qualitative_examples.json"
        qualitative_markdown = args.output_dir / "qualitative_examples.md"
        write_json(qualitative_json, qualitative_examples)
        write_qualitative_markdown(
            qualitative_markdown, qualitative_examples, qualitative_json.name
        )
        finished = datetime.now(timezone.utc)
        manifest = {
            "experiment": f"alfworld_failed_peer_reflections_{args.experiment_variant}",
            "inference_only": True,
            "started_at_utc": started.isoformat(),
            "finished_at_utc": finished.isoformat(),
            "elapsed_seconds": (finished - started).total_seconds(),
            "checkpoint": attempt1_rows[0]["checkpoint"],
            "configuration": {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in vars(args).items()
            },
            "selected_task_ids": [
                task_ids[index] for index in range(0, expected, args.group_size)
            ],
            "attempt1_rollouts": expected,
            "attempt1_failures": len(failed_slots),
            "k4_eligible_failed_recipients": sum(
                len(same_task_orders[slot]) >= 4 for slot in failed_slots
            ),
            "selected_failed_recipients": len(selected_set),
            "token_accounting": {
                "reflection_generation_input_tokens": "reflection prompt input",
                "reflection_generation_output_tokens": "generated reflection output",
                "merge_generation_input_tokens": "optional merge-pass prompt input",
                "merge_generation_output_tokens": "optional merged-reflection output",
                "retry_play_input_tokens": "sum of retry play-prompt inputs across turns",
                "retry_play_output_tokens": "sum of retry model outputs across turns",
                "total_tokens": (
                    "reflection input + reflection output + merge input + merge output "
                    "+ retry play input + retry play output"
                ),
            },
            "files": {
                "attempt1_rollouts": attempt1_path.name,
                "retry_evaluations": retry_path.name,
                "donor_assignments": donor_plan_path.name,
                "reflection_generations": reflection_path.name,
                "condition_summary": "condition_summary.csv",
                "task_summary": "task_summary.csv",
                "contrast_summary": "contrast_summary.csv",
                "qualitative_examples_json": qualitative_json.name,
                "qualitative_examples_markdown": qualitative_markdown.name,
            },
            **results,
        }
        write_json(args.output_dir / "summary.json", manifest)
        write_csv(args.output_dir / "condition_summary.csv", results["conditions"])
        write_csv(args.output_dir / "task_summary.csv", task_rows)
        write_csv(args.output_dir / "contrast_summary.csv", results["contrasts"])
        print(json.dumps(results, indent=2, default=json_default))
        print(f"Results written to {args.output_dir}")
    finally:
        manager.close()
        ray.shutdown()


if __name__ == "__main__":
    main()
