"""LaMer GiGPO support for VERL's v1 synchronous trainer."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch
import transfer_queue as tq
from tensordict import TensorDict
from transfer_queue import KVBatchMeta

from verl import DataProto
from verl.trainer.ppo.v1 import PPOTrainerSync, register_trainer
from verl.trainer.ppo.rollout_corr_helper import (
    compute_rollout_correction_and_add_to_batch,
)
from verl.utils.tracking import Tracking
from verl.workers.utils.padding import response_to_nested


def _install_live_wandb_logging() -> None:
    """Commit explicit W&B steps immediately for live long-running charts."""
    current_log = Tracking.log
    if getattr(current_log, "_lamer_commits_wandb_steps", False):
        return

    def log_with_commit(self, data, step, backend=None):
        def enabled(name):
            return backend is None or name in backend

        if "wandb" in self.logger and enabled("wandb"):
            self.logger["wandb"].log(data=data, step=step, commit=True)

        remaining_backends = [
            name for name in self.logger if name != "wandb" and enabled(name)
        ]
        if remaining_backends:
            current_log(self, data=data, step=step, backend=remaining_backends)

    log_with_commit._lamer_commits_wandb_steps = True
    Tracking.log = log_with_commit


def _hashable(value: Any) -> Any:
    if isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return tuple(value.flatten())
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _hashable(item)) for key, item in value.items()))
    raise TypeError(f"Unsupported GiGPO observation type: {type(value)}")


def _center_grouped_scores(
    scores: torch.Tensor,
    groups: list[Any],
    *,
    divide_by_std: bool,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Center scalar scores within groups using LaMer's original convention."""
    grouped_indices: dict[Any, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        grouped_indices[group].append(index)

    centered = torch.empty_like(scores)
    with torch.no_grad():
        for indices in grouped_indices.values():
            index_tensor = torch.tensor(indices, dtype=torch.long, device=scores.device)
            values = scores[index_tensor]
            mean = values.mean()
            if divide_by_std and len(indices) > 1:
                centered[index_tensor] = (values - mean) / (values.std() + epsilon)
            else:
                centered[index_tensor] = values - mean
    return centered


def compute_gigpo_advantages(
    episode_scores: torch.Tensor,
    step_returns: torch.Tensor,
    response_mask: torch.Tensor,
    task_uids: list[str],
    anchor_observations: list[Any],
    *,
    step_advantage_weight: float = 1.0,
    mode: str = "mean_norm",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the episode- and state-group components of GiGPO.

    ``episode_scores`` contains one scalar for every generated action or
    reflection.  This deliberately retains the original LaMer behavior: the
    task-group mean is weighted by the number of generated records in each
    trajectory, rather than first reducing to one scalar per trajectory.
    """
    if mode == "mean_norm":
        divide_by_std = False
    elif mode == "mean_std_norm":
        divide_by_std = True
    else:
        raise ValueError(f"Unknown GiGPO normalization mode: {mode}")

    episode_advantages = _center_grouped_scores(
        episode_scores,
        task_uids,
        divide_by_std=divide_by_std,
    )
    step_groups = [
        (task_uid, _hashable(anchor))
        for task_uid, anchor in zip(task_uids, anchor_observations, strict=True)
    ]
    step_advantages = _center_grouped_scores(
        step_returns,
        step_groups,
        divide_by_std=divide_by_std,
    )
    scalar_advantages = episode_advantages + step_advantage_weight * step_advantages
    token_advantages = scalar_advantages.unsqueeze(-1) * response_mask
    return token_advantages, token_advantages.clone()


def _gigpo_metrics(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    task_uids: list[str],
    trajectory_uids: list[str],
    episode_rewards: list[float],
) -> dict[str, float]:
    trajectory_outcomes: dict[tuple[str, str], float] = defaultdict(float)
    for task_uid, trajectory_uid, reward in zip(
        task_uids, trajectory_uids, episode_rewards, strict=True
    ):
        key = (task_uid, trajectory_uid)
        trajectory_outcomes[key] = max(trajectory_outcomes[key], float(reward))

    outcomes_by_task: dict[str, list[float]] = defaultdict(list)
    for (task_uid, _), reward in trajectory_outcomes.items():
        outcomes_by_task[task_uid].append(reward)

    successful_counts = [
        sum(reward > 0.0 for reward in outcomes) for outcomes in outcomes_by_task.values()
    ]
    group_count = len(successful_counts)
    all_failed = sum(count == 0 for count in successful_counts)
    all_succeeded = sum(
        count == len(outcomes)
        for count, outcomes in zip(successful_counts, outcomes_by_task.values(), strict=True)
    )
    mixed = group_count - all_failed - all_succeeded

    valid_advantages = advantages.masked_select(response_mask.bool())
    metrics = {
        "gigpo/groups/count": float(group_count),
        "gigpo/groups/mixed_outcome_fraction": mixed / group_count if group_count else 0.0,
        "gigpo/groups/all_failure_fraction": all_failed / group_count if group_count else 0.0,
        "gigpo/groups/all_success_fraction": all_succeeded / group_count if group_count else 0.0,
        "gigpo/groups/successful_rollouts_mean": (
            float(np.mean(successful_counts)) if successful_counts else 0.0
        ),
        "gigpo/groups/successful_rollouts_std": (
            float(np.std(successful_counts)) if successful_counts else 0.0
        ),
    }
    if valid_advantages.numel():
        metrics.update(
            {
                "gigpo/advantages/nonzero_token_fraction": float(
                    (valid_advantages != 0).float().mean().item()
                ),
                "gigpo/advantages/absolute_mean": float(valid_advantages.abs().mean().item()),
                "gigpo/advantages/std": float(valid_advantages.std(unbiased=False).item()),
            }
        )
    return metrics


def _reflection_step_group_metrics(
    task_uids: list[str],
    anchor_observations: list[Any],
    phases: list[str],
    trajectory_uids: list[str],
) -> dict[str, float]:
    """Describe reflection groups without changing GiGPO grouping or scores."""
    group_members: dict[tuple[Any, Any], list[int]] = defaultdict(list)
    reflection_keys = set()
    reflection_record_count = 0
    for index, (task_uid, anchor, phase) in enumerate(
        zip(task_uids, anchor_observations, phases, strict=True)
    ):
        key = (task_uid, _hashable(anchor))
        group_members[key].append(index)
        if phase == "reflect":
            reflection_keys.add(key)
            reflection_record_count += 1

    prefix = "gigpo/reflection_step_groups"
    if not reflection_keys:
        return {
            f"{prefix}/count": 0.0,
            f"{prefix}/records": 0.0,
            f"{prefix}/singleton_fraction": 0.0,
            f"{prefix}/size_mean": 0.0,
            f"{prefix}/size_min": 0.0,
            f"{prefix}/size_max": 0.0,
        }

    sizes = np.asarray(
        [len(group_members[key]) for key in reflection_keys], dtype=np.int64
    )
    distinct_sizes = np.asarray(
        [
            len({trajectory_uids[index] for index in group_members[key]})
            for key in reflection_keys
        ],
        dtype=np.int64,
    )
    metrics = {
        f"{prefix}/count": float(len(sizes)),
        f"{prefix}/records": float(reflection_record_count),
        f"{prefix}/singleton_fraction": float(np.mean(sizes == 1)),
        f"{prefix}/size_mean": float(sizes.mean()),
        f"{prefix}/size_min": float(sizes.min()),
        f"{prefix}/size_max": float(sizes.max()),
        f"{prefix}/distinct_trajectory_singleton_fraction": float(
            np.mean(distinct_sizes == 1)
        ),
        f"{prefix}/distinct_trajectory_size_mean": float(distinct_sizes.mean()),
        f"{prefix}/distinct_trajectory_size_min": float(distinct_sizes.min()),
        f"{prefix}/distinct_trajectory_size_max": float(distinct_sizes.max()),
    }
    for size, count in zip(*np.unique(sizes, return_counts=True)):
        metrics[f"{prefix}/size_histogram/{int(size)}"] = float(count)
    for size, count in zip(*np.unique(distinct_sizes, return_counts=True)):
        metrics[
            f"{prefix}/distinct_trajectory_size_histogram/{int(size)}"
        ] = float(count)
    return metrics


@register_trainer("alfworld_gigpo_sync")
class AlfWorldGiGPOTrainer(PPOTrainerSync):
    """Synchronous v1 trainer that restores LaMer's per-generation GiGPO."""

    def fit(self, agent_loop_manager):
        _install_live_wandb_logging()
        return super().fit(agent_loop_manager)

    def _validate(self) -> dict[str, float]:
        metrics = super()._validate()
        self._add_validation_metric_aliases(metrics)
        if self.config.trainer.get("save_best_validation", False):
            self._select_validation_checkpoint(metrics)
        return metrics

    @staticmethod
    def _add_validation_metric_aliases(metrics: dict[str, float]) -> None:
        """Expose modern VERL validation metrics under the readable LaMer names."""
        modern_prefix = "val-aux/alfworld/"
        modern_suffix = "/mean@1"
        for name, value in list(metrics.items()):
            if name.startswith(modern_prefix) and name.endswith(modern_suffix):
                metric = name[len(modern_prefix) : -len(modern_suffix)]
                metrics.setdefault(f"val/{metric}", value)

        attempts = [metrics.get(f"val/success_rate[{index}]") for index in range(3)]
        if any(value is None for value in attempts):
            return
        p_at_1, p_at_2, p_at_3 = (float(value) for value in attempts)
        metrics.update(
            {
                "val/meta_rl/p_at_1": p_at_1,
                "val/meta_rl/p_at_2": p_at_2,
                "val/meta_rl/p_at_3": p_at_3,
                "val/meta_rl/gain_attempt_2": p_at_2 - p_at_1,
                "val/meta_rl/gain_attempt_3": p_at_3 - p_at_2,
                "val/meta_rl/gain_over_first_at_2": p_at_2 - p_at_1,
                "val/meta_rl/gain_over_first_at_3": p_at_3 - p_at_1,
            }
        )

    def _select_validation_checkpoint(self, metrics: dict[str, float]) -> None:
        """Retain the best validation checkpoint and the completed run's last checkpoint.

        Selection happens after validation, including the step-zero validation.  A
        strict improvement replaces the prior best; ties retain the earlier model.
        The final step is always saved once, whether or not it is the best.
        """
        metric_name = str(self.config.trainer.best_validation_metric)
        if metric_name not in metrics:
            available = sorted(key for key in metrics if key.startswith("val-"))
            raise KeyError(
                f"Best-checkpoint metric {metric_name!r} was not produced. "
                f"Available validation metrics: {available}"
            )

        score = float(metrics[metric_name])
        if not np.isfinite(score):
            raise ValueError(
                f"Best-checkpoint metric {metric_name!r} is not finite at "
                f"step {self.global_steps}: {score}"
            )

        state = self._load_checkpoint_selection_state(metric_name)
        prior_best_step = state.get("best_step")
        prior_best_score = state.get("best_score")
        improved = prior_best_score is None or score > float(prior_best_score)
        is_last = self.global_steps >= self.total_training_steps
        save_last = bool(self.config.trainer.get("save_last_checkpoint", True))
        should_save = improved or (is_last and save_last)

        if should_save:
            self._save_checkpoint()

        if improved:
            state["best_score"] = score
            state["best_step"] = int(self.global_steps)
            if prior_best_step is not None and int(prior_best_step) != self.global_steps:
                self._remove_selected_checkpoint(int(prior_best_step))

        if is_last:
            state["last_step"] = int(self.global_steps)

        retained_steps = {state.get("best_step"), state.get("last_step")}
        state["retained_steps"] = sorted(
            int(step) for step in retained_steps if step is not None
        )
        state["last_validation_step"] = int(self.global_steps)
        state["last_validation_score"] = score
        self._write_checkpoint_selection_state(state)

        metrics["checkpoint/selection_score"] = score
        metrics["checkpoint/new_best"] = float(improved)
        metrics["checkpoint/saved"] = float(should_save)

    def _checkpoint_selection_path(self) -> Path:
        return Path(self.config.trainer.default_local_dir) / "checkpoint_selection.json"

    def _load_checkpoint_selection_state(self, metric_name: str) -> dict[str, Any]:
        path = self._checkpoint_selection_path()
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                state = json.load(handle)
            recorded_metric = state.get("selection_metric")
            if recorded_metric != metric_name:
                raise ValueError(
                    f"Checkpoint selection metric changed from {recorded_metric!r} "
                    f"to {metric_name!r} in {path}"
                )
            return state
        return {
            "selection_metric": metric_name,
            "selection_mode": "max",
            "tie_policy": "keep_earlier",
            "best_step": None,
            "best_score": None,
            "last_step": None,
            "retained_steps": [],
        }

    def _write_checkpoint_selection_state(self, state: dict[str, Any]) -> None:
        path = self._checkpoint_selection_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(".json.tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary_path.replace(path)

    def _remove_selected_checkpoint(self, step: int) -> None:
        root = Path(self.config.trainer.default_local_dir).resolve()
        checkpoint = root / f"global_step_{step}"
        if checkpoint.parent != root or checkpoint.name != f"global_step_{step}":
            raise RuntimeError(f"Refusing to remove unexpected checkpoint path: {checkpoint}")
        if checkpoint.is_symlink():
            raise RuntimeError(f"Refusing to remove symlinked checkpoint path: {checkpoint}")
        if checkpoint.exists():
            shutil.rmtree(checkpoint)

    def _compute_advantage(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        if self.config.algorithm.adv_estimator != "gigpo":
            return super()._compute_advantage(batch, metrics)
        if self.config.algorithm.use_kl_in_reward:
            raise NotImplementedError(
                "The initial modern GiGPO port requires algorithm.use_kl_in_reward=False"
            )
        rollout_correction = self.config.algorithm.get("rollout_correction")
        if rollout_correction and rollout_correction.get("bypass_mode", False):
            raise NotImplementedError(
                "ALFWorld GiGPO supports decoupled rollout correction, not bypass mode"
            )
        correction_enabled = bool(
            rollout_correction
            and (
                rollout_correction.get("rollout_is") is not None
                or rollout_correction.get("rollout_rs") is not None
            )
        )

        queued_fields = ["uid", "response_mask", "extra_fields"]
        if correction_enabled:
            queued_fields.extend(["old_log_probs", "rollout_log_probs"])
        queued = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=queued_fields,
        )
        nested_response_mask = queued.pop("response_mask")
        response_mask = nested_response_mask.to_padded_tensor(padding=0)
        rollout_is_weights = None
        if correction_enabled:
            correction_data = DataProto(
                batch=TensorDict(
                    {
                        "old_log_probs": queued.pop("old_log_probs").to_padded_tensor(
                            padding=0.0
                        ),
                        "rollout_log_probs": queued.pop(
                            "rollout_log_probs"
                        ).to_padded_tensor(padding=0.0),
                        "response_mask": response_mask,
                    },
                    batch_size=len(batch),
                )
            )
            correction_data, correction_metrics = (
                compute_rollout_correction_and_add_to_batch(
                    correction_data,
                    rollout_correction,
                )
            )
            response_mask = correction_data.batch["response_mask"]
            rollout_is_weights = correction_data.batch.get(
                "rollout_is_weights"
            )
            metrics.update(correction_metrics)
        all_task_uids = [str(value) for value in queued.pop("uid").tolist()]
        all_extra_fields = queued.pop("extra_fields").tolist()

        # Modern VERL may append synthetic rows so variable-length agent-loop
        # output is divisible by the data-parallel and mini-batch sizes.  These
        # rows deliberately have zero response/loss masks and must remain in the
        # batch for downstream partitioning, but they are not GiGPO records.
        # Excluding them before group centering is essential: padding metadata
        # is copied from a real row and would otherwise change real advantages.
        real_indices = [
            index for index, tag in enumerate(batch.tags) if not tag.get("is_padding", False)
        ]
        if not real_indices:
            raise ValueError("GiGPO batch contains no non-padding records")
        real_index_tensor = torch.tensor(
            real_indices, dtype=torch.long, device=response_mask.device
        )
        real_response_mask = response_mask.index_select(0, real_index_tensor)
        task_uids = [all_task_uids[index] for index in real_indices]
        extra_fields = [all_extra_fields[index] for index in real_indices]

        required_fields = {
            "anchor_obs",
            "traj_uid",
            "episode_reward",
            "step_return",
            "is_action_valid",
        }
        gigpo_config = self.config.algorithm.get("gigpo", {})
        future_aware_episode_credit = bool(
            gigpo_config.get("future_aware_episode_credit", False)
        )
        if future_aware_episode_credit:
            required_fields.add("future_episode_reward")
        for index, fields in zip(real_indices, extra_fields, strict=True):
            missing = required_fields - fields.keys()
            if missing:
                raise ValueError(
                    f"GiGPO record {batch.keys[index]} is missing metadata: {sorted(missing)}"
                )

        episode_rewards = [float(fields["episode_reward"]) for fields in extra_fields]
        episode_score_field = (
            "future_episode_reward"
            if future_aware_episode_credit
            else "episode_reward"
        )
        episode_scores = torch.tensor(
            [float(fields[episode_score_field]) for fields in extra_fields],
            dtype=torch.float32,
            device=response_mask.device,
        )
        valid_actions = torch.tensor(
            [bool(fields["is_action_valid"]) for fields in extra_fields],
            dtype=torch.bool,
            device=response_mask.device,
        )
        effective_actions = torch.tensor(
            [bool(fields.get("action_is_effective", False)) for fields in extra_fields],
            dtype=torch.bool,
            device=response_mask.device,
        )
        response_lengths = real_response_mask.sum(dim=-1)
        response_limit = int(self.config.data.max_response_length)
        phases = [str(fields.get("phase", "unknown")) for fields in extra_fields]
        for phase in ("play", "reflect"):
            phase_mask = torch.tensor(
                [record_phase == phase for record_phase in phases],
                dtype=torch.bool,
                device=response_mask.device,
            )
            if phase_mask.any():
                metrics[f"format/{phase}/records"] = float(phase_mask.sum().item())
                metrics[f"format/{phase}/parse_valid_ratio"] = float(
                    valid_actions[phase_mask].float().mean().item()
                )
                metrics[f"format/{phase}/effective_ratio"] = float(
                    effective_actions[phase_mask].float().mean().item()
                )
                metrics[f"format/{phase}/response_limit_ratio"] = float(
                    (response_lengths[phase_mask] >= response_limit).float().mean().item()
                )
        if gigpo_config.get("use_invalid_action_penalty", True):
            penalty = float(gigpo_config.get("invalid_action_penalty_coef", 0.5))
            episode_scores = episode_scores - (~valid_actions).float() * penalty
        metrics["valid_action_ratio"] = float(valid_actions.float().mean().item())
        metrics["gigpo/future_aware_episode_credit"] = float(
            future_aware_episode_credit
        )

        step_scores = torch.tensor(
            [float(fields["step_return"]) for fields in extra_fields],
            dtype=torch.float32,
            device=response_mask.device,
        )
        if gigpo_config.get("use_invalid_action_penalty", True):
            # Match the original LaMer path: an invalid generation is penalized
            # in both the episode-level score and the state-level return before
            # their respective within-group centering operations.
            step_scores = step_scores - (~valid_actions).float() * penalty
        anchor_observations = [fields["anchor_obs"] for fields in extra_fields]
        trajectory_uids = [str(fields["traj_uid"]) for fields in extra_fields]
        real_advantages, real_returns = compute_gigpo_advantages(
            episode_scores,
            step_scores,
            real_response_mask,
            task_uids,
            anchor_observations,
            step_advantage_weight=float(gigpo_config.get("step_advantage_w", 1.0)),
            mode=str(gigpo_config.get("mode", "mean_norm")),
        )

        advantages = torch.zeros_like(response_mask, dtype=torch.float32)
        returns = torch.zeros_like(response_mask, dtype=torch.float32)
        advantages.index_copy_(0, real_index_tensor, real_advantages)
        returns.index_copy_(0, real_index_tensor, real_returns)
        token_level_rewards = torch.zeros_like(response_mask, dtype=torch.float32)
        for real_row, batch_row in enumerate(real_indices):
            valid_positions = torch.nonzero(
                real_response_mask[real_row], as_tuple=False
            ).flatten()
            if not len(valid_positions):
                raise ValueError(
                    f"GiGPO record {batch.keys[batch_row]} has no generated response tokens"
                )
            token_level_rewards[batch_row, valid_positions[-1]] = episode_scores[real_row]

        output = TensorDict(
            {
                "advantages": response_to_nested(advantages, nested_response_mask),
                "returns": response_to_nested(returns, nested_response_mask),
                "rm_scores": response_to_nested(token_level_rewards, nested_response_mask),
                "token_level_rewards": response_to_nested(
                    token_level_rewards, nested_response_mask
                ),
            },
            batch_size=len(batch),
        )
        if correction_enabled:
            output["response_mask"] = response_to_nested(
                response_mask,
                nested_response_mask,
            )
            if rollout_is_weights is not None:
                output["rollout_is_weights"] = response_to_nested(
                    rollout_is_weights,
                    nested_response_mask,
                )
        batch = tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=output,
        )
        metrics.update(
            _gigpo_metrics(
                real_advantages,
                real_response_mask,
                task_uids,
                trajectory_uids,
                episode_rewards,
            )
        )
        metrics.update(
            _reflection_step_group_metrics(
                task_uids,
                anchor_observations,
                phases,
                trajectory_uids,
            )
        )
        return batch
