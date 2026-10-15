"""LaMer GiGPO support for VERL's v1 synchronous trainer."""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch
import transfer_queue as tq
from tensordict import TensorDict
from transfer_queue import KVBatchMeta

from verl import DataProto
from verl.trainer.distillation import is_distillation_enabled
from verl.trainer.ppo.core_algos import (
    compute_policy_loss_vanilla,
    register_policy_loss,
)
from verl.trainer.ppo.v1 import PPOTrainerSync, register_trainer
from verl.trainer.ppo.rollout_corr_helper import (
    compute_rollout_correction_and_add_to_batch,
)
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.debug import marked_timer
from verl.utils.tracking import Tracking
from verl.workers.utils.padding import response_to_nested

from agent_system.meta_trajectory_production import (
    compare_reference_and_production_token_losses,
    finalize_meta_trajectory_batch,
    meta_trajectory_metrics,
    meta_trajectory_rollout_diversity_metrics,
    ppo_record_minibatch_scale,
    prepare_lamer_form_batch,
    prepare_meta_trajectory_batch,
    scale_weighted_advantages_for_ppo_minibatches,
)
from agent_system.multi_turn_rollout.performance_instrumentation import (
    rollout_profile_metrics,
    summarize_rollout_profile,
)


@register_policy_loss("meta_vanilla")
def compute_meta_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config=None,
    rollout_is_weights: torch.Tensor | None = None,
):
    """Vanilla PPO loss plus sign-aware asymmetric clipping diagnostics."""
    loss, metrics = compute_policy_loss_vanilla(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )

    clip_low = config.clip_ratio_low
    clip_high = config.clip_ratio_high
    ratio = torch.exp(torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0))
    mask = response_mask.bool()
    denominator = mask.sum().clamp_min(1)
    upper_active = mask & (advantages > 0) & (ratio > 1.0 + clip_high)
    lower_active = mask & (advantages < 0) & (ratio < 1.0 - clip_low)
    selected_ratio = ratio[mask]
    selected_log_ratio = (log_prob - old_log_prob)[mask]
    if selected_ratio.numel():
        ratio_quantiles = torch.quantile(
            selected_ratio.float(),
            torch.tensor(
                [0.05, 0.50, 0.95],
                device=selected_ratio.device,
            ),
        )
        ratio_mean = selected_ratio.float().mean()
        mean_log_ratio = selected_log_ratio.float().mean()
        approximate_kl = (
            selected_ratio.float() - 1.0 - selected_log_ratio.float()
        ).mean()
    else:
        ratio_quantiles = torch.ones(3, device=ratio.device)
        ratio_mean = torch.ones((), device=ratio.device)
        mean_log_ratio = torch.zeros((), device=ratio.device)
        approximate_kl = torch.zeros((), device=ratio.device)
    metrics.update(
        {
            "actor/pg_clipfrac_upper_objective": float(
                upper_active.sum().div(denominator).detach()
            ),
            "actor/pg_clipfrac_lower_objective": float(
                lower_active.sum().div(denominator).detach()
            ),
            "actor/ratio_mean": float(ratio_mean.detach()),
            "actor/ratio_p05": float(ratio_quantiles[0].detach()),
            "actor/ratio_p50": float(ratio_quantiles[1].detach()),
            "actor/ratio_p95": float(ratio_quantiles[2].detach()),
            "actor/mean_log_ratio": float(mean_log_ratio.detach()),
            "actor/approx_kl_from_ratio": float(approximate_kl.detach()),
            "actor/ratio_token_count": float(selected_ratio.numel()),
        }
    )
    return loss, metrics


def _write_meta_integration_manifest(
    *,
    prepared,
    finalized,
    extra_fields: list[dict[str, Any]],
    is_padding: list[bool],
    old_log_probs: torch.Tensor | None,
    rollout_is_weights: torch.Tensor | None,
    gamma_step: float,
    gamma_traj: float,
    ppo_minibatch_record_count: int,
    ppo_minibatch_scale: float,
) -> None:
    """Persist an opt-in, pre-shuffle manifest for the one-rollout pilot."""
    audit_dir = os.environ.get("LAMER_META_INTEGRATION_AUDIT_DIR")
    if not audit_dir:
        return

    destination = Path(audit_dir)
    destination.mkdir(parents=True, exist_ok=True)
    record_by_row = {int(record.record_id): record for record in prepared.records}
    rows = []
    for row_index, padding in enumerate(is_padding):
        if padding:
            rows.append({"row_index": row_index, "is_padding": True})
            continue
        record = record_by_row[row_index]
        credit = prepared.record_credits[row_index]
        mask = finalized.response_mask[row_index].bool()
        old_values = (
            old_log_probs[row_index][mask].detach().float().cpu()
            if old_log_probs is not None
            else torch.empty(0)
        )
        rows.append(
            {
                "row_index": row_index,
                "is_padding": False,
                "task_uid": credit.task_uid,
                "trajectory_uid": credit.trajectory_uid,
                "phase": credit.phase,
                "attempt_index": credit.attempt_index,
                "turn_index": credit.turn_index,
                "environment_reward": float(record.environment_reward),
                "won": bool(extra_fields[row_index].get("won", False)),
                "task_type": str(extra_fields[row_index].get("task_type", "")),
                "projected_action": repr(record.projected_action),
                "parser_valid": credit.validity.parser_valid,
                "hard_truncated": credit.validity.hard_truncated,
                "reflection_recovered_from_truncation": (
                    credit.reflection_recovered_from_truncation
                ),
                "generated_response_length": credit.generated_response_length,
                "prompt_token_count": int(
                    extra_fields[row_index].get("prompt_token_count", 0) or 0
                ),
                "num_cached_tokens": extra_fields[row_index].get(
                    "num_cached_tokens"
                ),
                "request_start_monotonic_s": extra_fields[row_index].get(
                    "request_start_monotonic_s"
                ),
                "request_end_monotonic_s": extra_fields[row_index].get(
                    "request_end_monotonic_s"
                ),
                "request_latency_s": extra_fields[row_index].get(
                    "request_latency_s"
                ),
                "vllm_queue_latency_s": extra_fields[row_index].get(
                    "vllm_queue_latency_s"
                ),
                "vllm_prefill_latency_s": extra_fields[row_index].get(
                    "vllm_prefill_latency_s"
                ),
                "vllm_decode_latency_s": extra_fields[row_index].get(
                    "vllm_decode_latency_s"
                ),
                "rollout_worker_id": extra_fields[row_index].get(
                    "rollout_worker_id"
                ),
                "worker_process_cpu_start_s": extra_fields[row_index].get(
                    "worker_process_cpu_start_s"
                ),
                "worker_process_cpu_end_s": extra_fields[row_index].get(
                    "worker_process_cpu_end_s"
                ),
                "trainable_response_tokens": int(mask.sum().item()),
                "utility": credit.utility,
                "task_baseline": credit.task_baseline,
                "selected_baseline": credit.selected_baseline,
                "baseline_kind": credit.baseline_kind,
                "fallback_reasons": [reason.value for reason in credit.fallback_reasons],
                "base_advantage": credit.base_advantage,
                "invalid_adjustment": credit.invalid_adjustment,
                "overlong_adjustment": credit.overlong_adjustment,
                "final_unweighted_advantage": credit.final_unweighted_advantage,
                "trajectory_trainable_tokens": finalized.trajectory_token_counts[
                    credit.trajectory_key
                ],
                "trajectory_weight": finalized.trajectory_weights[
                    credit.trajectory_key
                ],
                "old_log_prob_count": int(old_values.numel()),
                "old_log_prob_sum": float(old_values.sum().item()),
            }
        )

    trajectories = []
    for (task_uid, trajectory_uid), trajectory_rows in sorted(
        prepared.trajectory_rows.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
    ):
        records = [record_by_row[index] for index in trajectory_rows]
        success_attempts = [
            int(record.attempt_index) + 1
            for record in records
            if record.phase == "play" and bool(extra_fields[int(record.record_id)].get("won", False))
        ]
        trajectories.append(
            {
                "task_uid": str(task_uid),
                "trajectory_uid": str(trajectory_uid),
                "record_rows": list(trajectory_rows),
                "attempts": sorted({int(record.attempt_index) + 1 for record in records}),
                "success_attempt": min(success_attempts) if success_attempts else None,
                "raw_play_rewards": [
                    {
                        "attempt": int(record.attempt_index) + 1,
                        "turn": int(record.turn_index) + 1,
                        "reward": float(record.environment_reward),
                    }
                    for record in records
                    if record.phase == "play"
                ],
                "utility": finalized.prepared.trajectory_utilities[(task_uid, trajectory_uid)],
                "task_baseline": finalized.prepared.task_centering.baselines[task_uid],
                "N_j": finalized.trajectory_token_counts[(task_uid, trajectory_uid)],
                "token_mass": finalized.trajectory_token_masses[(task_uid, trajectory_uid)],
            }
        )

    groups = []
    for group_index, group in enumerate(prepared.baseline_assignments.groups.values()):
        groups.append(
            {
                "group_index": group_index,
                "unique_trajectory_count": group.unique_trajectory_count,
                "distinct_valid_action_count": len(group.distinct_valid_actions),
                "distinct_valid_actions": sorted(map(repr, group.distinct_valid_actions)),
                "utility_range": group.utility_range,
                "eligible": group.eligible,
                "fallback_reasons": [reason.value for reason in group.fallback_reasons],
            }
        )

    correction = None
    if rollout_is_weights is not None:
        selected = rollout_is_weights[finalized.response_mask.bool()].detach().float().cpu()
        correction = {
            "count": int(selected.numel()),
            "mean": float(selected.mean().item()) if selected.numel() else 0.0,
            "min": float(selected.min().item()) if selected.numel() else 0.0,
            "max": float(selected.max().item()) if selected.numel() else 0.0,
            "ess": (
                float(selected.sum().square().div(selected.square().sum()).item())
                if selected.numel() and float(selected.square().sum()) > 0.0
                else 0.0
            ),
            "fraction_at_upper_boundary_2": (
                float((selected == 2.0).float().mean().item())
                if selected.numel()
                else 0.0
            ),
        }

    real_rows = set(prepared.real_indices)
    trajectory_rows = {
        row for rows_for_trajectory in prepared.trajectory_rows.values() for row in rows_for_trajectory
    }
    manifest = {
        "gamma_step": gamma_step,
        "gamma_traj": gamma_traj,
        "task_count": len(finalized.prepared.task_centering.baselines),
        "trajectory_count": len(finalized.trajectory_token_counts),
        "record_count": len(is_padding),
        "real_record_count": len(real_rows),
        "padding_record_count": sum(is_padding),
        "ppo_minibatch_record_count": ppo_minibatch_record_count,
        "ppo_minibatch_scale": ppo_minibatch_scale,
        "expected_optimizer_steps": len(is_padding) // ppo_minibatch_record_count,
        "record_coverage_exact": real_rows == trajectory_rows == set(prepared.record_credits),
        "rows": rows,
        "trajectories": trajectories,
        "state_groups": groups,
        "importance_correction": correction,
        "rollout_profile": summarize_rollout_profile(
            [
                extra_fields[index]
                for index, padding in enumerate(is_padding)
                if not padding
            ]
        ),
    }
    with (destination / "credit_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


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

    def on_sample_end(self):
        """Expose vLLM sleep time separately from the enclosing generation timer."""
        with marked_timer("sleep_replicas", self.timing_raw, color="red"):
            return super().on_sample_end()

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
        """Retain the best validation checkpoint and a rolling latest checkpoint.

        Selection happens after validation, including the step-zero validation.  A
        strict improvement replaces the prior best; ties retain the earlier model.
        When ``save_last_checkpoint`` is enabled, every validated step becomes the
        new resumable latest checkpoint and supersedes the prior non-best latest.
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
        save_last = bool(self.config.trainer.get("save_last_checkpoint", True))
        should_save = improved or save_last
        prior_retained_steps = {
            int(step) for step in state.get("retained_steps", []) if step is not None
        }
        prior_retained_steps.update(
            int(step)
            for step in (state.get("best_step"), state.get("last_step"))
            if step is not None
        )

        if should_save:
            self._save_checkpoint()

        if improved:
            state["best_score"] = score
            state["best_step"] = int(self.global_steps)

        if save_last:
            state["last_step"] = int(self.global_steps)

        retained_steps = {state.get("best_step"), state.get("last_step")}
        new_retained_steps = {
            int(step) for step in retained_steps if step is not None
        }
        for obsolete_step in sorted(prior_retained_steps - new_retained_steps):
            self._remove_selected_checkpoint(obsolete_step)
        state["retained_steps"] = sorted(new_retained_steps)
        state["last_validation_step"] = int(self.global_steps)
        state["last_validation_score"] = score
        self._write_checkpoint_selection_state(state)

        metrics["checkpoint/selection_score"] = score
        metrics["checkpoint/new_best"] = float(improved)
        metrics["checkpoint/saved"] = float(should_save)
        metrics["checkpoint/rolling_last"] = float(save_last)

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
        credit_mode = str(
            self.config.algorithm.get("credit_mode", "original_lamer")
        )
        if credit_mode in {"meta_trajectory", "lamer_form"}:
            return self._compute_meta_trajectory_advantage(batch, metrics)
        if credit_mode != "original_lamer":
            raise ValueError(f"Unknown algorithm.credit_mode: {credit_mode!r}")
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

    def _compute_meta_trajectory_advantage(
        self,
        batch: KVBatchMeta,
        metrics: dict,
    ) -> KVBatchMeta:
        """Write Checkpoint-3 meta-trajectory tensors without updating parameters."""
        if self.config.algorithm.use_kl_in_reward:
            raise NotImplementedError(
                "meta_trajectory credit currently requires algorithm.use_kl_in_reward=False"
            )
        actor_config = self.config.actor_rollout_ref.actor
        loss_mode = str(actor_config.loss_agg_mode)
        if loss_mode not in {"token-sum", "token-mean"}:
            raise ValueError(
                "ablation credit requires actor.loss_agg_mode='token-sum' or 'token-mean'"
            )
        credit_mode = str(self.config.algorithm.credit_mode)
        if credit_mode == "lamer_form" and loss_mode != "token-mean":
            raise ValueError("lamer_form credit requires token-mean loss")

        meta_config = self.config.algorithm.get("meta_trajectory")
        if meta_config is None:
            raise ValueError("algorithm.meta_trajectory configuration is required")
        state_config = meta_config.get("state_baseline")
        invalid_config = meta_config.get("invalid_penalty")
        overlong_config = meta_config.get("overlong_penalty")
        if state_config is None or invalid_config is None or overlong_config is None:
            raise ValueError(
                "meta_trajectory requires state_baseline, invalid_penalty, and "
                "overlong_penalty configuration blocks"
            )
        tau_u = state_config.get("utility_range_tolerance")
        play_cache_tokens = overlong_config.get("play_cache_tokens")
        reflection_cache_tokens = overlong_config.get("reflection_cache_tokens")
        unresolved = [
            name
            for name, value in (
                ("state_baseline.utility_range_tolerance", tau_u),
                ("overlong_penalty.play_cache_tokens", play_cache_tokens),
                ("overlong_penalty.reflection_cache_tokens", reflection_cache_tokens),
            )
            if value is None
        ]
        if unresolved:
            raise ValueError(
                "Meta-trajectory scientific configuration must be explicit: "
                + ", ".join(unresolved)
            )
        if not bool(state_config.get("enabled", True)):
            raise ValueError("Checkpoint 3 implements the frozen enabled state baseline")
        if int(state_config.get("min_unique_trajectories", 3)) != 3:
            raise ValueError("The frozen minimum unique-trajectory support is 3")
        if int(state_config.get("min_distinct_projected_actions", 2)) != 2:
            raise ValueError("The frozen minimum distinct-action support is 2")
        expected_aggregation = (
            "meta_traj_mean" if loss_mode == "token-sum" else "token_mean"
        )
        if str(meta_config.get("loss_aggregation", "meta_traj_mean")) != expected_aggregation:
            raise ValueError(
                "meta_trajectory.loss_aggregation must match actor.loss_agg_mode"
            )

        rollout_correction = self.config.algorithm.get("rollout_correction")
        if rollout_correction and rollout_correction.get("bypass_mode", False):
            raise NotImplementedError(
                "meta_trajectory credit supports decoupled rollout correction, not bypass mode"
            )
        correction_enabled = bool(
            rollout_correction
            and (
                rollout_correction.get("rollout_is") is not None
                or rollout_correction.get("rollout_rs") is not None
            )
        )

        queued_fields = ["uid", "responses", "response_mask", "extra_fields"]
        if correction_enabled:
            queued_fields.extend(["old_log_probs", "rollout_log_probs"])
        queued = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=queued_fields,
        )
        nested_responses = queued.pop("responses")
        nested_response_mask = queued.pop("response_mask")
        response_token_ids = nested_responses.to_padded_tensor(padding=0)
        ordinary_response_mask = nested_response_mask.to_padded_tensor(padding=0)
        all_task_uids = [str(value) for value in queued.pop("uid").tolist()]
        all_extra_fields = queued.pop("extra_fields").tolist()
        is_padding = [bool(tag.get("is_padding", False)) for tag in batch.tags]
        metrics.update(
            rollout_profile_metrics(
                [
                    fields
                    for fields, padding in zip(
                        all_extra_fields, is_padding, strict=True
                    )
                    if not padding
                ]
            )
        )

        prepare_credit = (
            prepare_lamer_form_batch
            if credit_mode == "lamer_form"
            else prepare_meta_trajectory_batch
        )
        preparation_kwargs = dict(
            task_uids=all_task_uids,
            extra_fields=all_extra_fields,
            response_mask=ordinary_response_mask,
            is_padding=is_padding,
            gamma_traj=float(self.config.algorithm.traj_gamma),
            gamma_step=float(self.config.algorithm.step_gamma),
            tau_u=float(tau_u),
            expected_siblings=int(self.config.actor_rollout_ref.rollout.n),
            invalid_penalty_enabled=bool(invalid_config.get("enabled", True)),
            invalid_penalty_coefficient=float(invalid_config.get("coefficient", 0.5)),
            overlong_penalty_enabled=bool(overlong_config.get("enabled", True)),
            overlong_mode=str(overlong_config.get("mode", "soft_shaping")),
            overlong_max_coefficient=float(
                overlong_config.get("max_coefficient", 0.5)
            ),
            play_cache_tokens=int(play_cache_tokens),
            reflection_cache_tokens=int(reflection_cache_tokens),
        )
        if credit_mode == "lamer_form":
            preparation_kwargs["step_advantage_weight"] = float(
                self.config.algorithm.gigpo.step_advantage_w
            )
            if str(self.config.algorithm.gigpo.mode) != "mean_norm":
                raise ValueError("lamer_form requires occurrence-weighted mean_norm")
            if bool(self.config.algorithm.gigpo.future_aware_episode_credit):
                raise ValueError("lamer_form requires attempt-local episode credit")
        prepared = prepare_credit(**preparation_kwargs)

        final_response_mask = prepared.response_mask
        rollout_is_weights = None
        old_log_probs_for_audit = None
        if correction_enabled:
            old_log_probs_for_audit = queued.pop("old_log_probs").to_padded_tensor(
                padding=0.0
            )
            correction_data = DataProto(
                batch=TensorDict(
                    {
                        "old_log_probs": old_log_probs_for_audit,
                        "rollout_log_probs": queued.pop(
                            "rollout_log_probs"
                        ).to_padded_tensor(padding=0.0),
                        "response_mask": final_response_mask,
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
            final_response_mask = correction_data.batch["response_mask"]
            rollout_is_weights = correction_data.batch.get("rollout_is_weights")
            metrics.update(correction_metrics)

        finalized = finalize_meta_trajectory_batch(
            prepared,
            final_response_mask=final_response_mask,
        )

        record_count = len(batch)
        ppo_minibatch_record_count = int(actor_config.ppo_mini_batch_size) * int(
            self.config.actor_rollout_ref.rollout.n
        )
        ppo_minibatch_scale = ppo_record_minibatch_scale(
            rollout_record_count=record_count,
            ppo_minibatch_record_count=ppo_minibatch_record_count,
        )
        optimizer_advantages = (
            scale_weighted_advantages_for_ppo_minibatches(
                finalized.weighted_advantages,
                rollout_record_count=record_count,
                ppo_minibatch_record_count=ppo_minibatch_record_count,
            )
            if loss_mode == "token-sum"
            else finalized.unweighted_advantages
        )
        device = final_response_mask.device
        utility = torch.zeros(record_count, dtype=torch.float32, device=device)
        task_baseline = torch.zeros_like(utility)
        state_baseline = torch.zeros_like(utility)
        state_baseline_available = torch.zeros(
            record_count, dtype=torch.bool, device=device
        )
        base_advantage = torch.zeros_like(utility)
        invalid_adjustment = torch.zeros_like(utility)
        overlong_adjustment = torch.zeros_like(utility)
        final_unweighted_scalar = torch.zeros_like(utility)
        trajectory_weight = torch.zeros_like(utility)
        trainable_token_count = torch.zeros(
            record_count, dtype=torch.long, device=device
        )
        token_level_rewards = torch.zeros_like(
            ordinary_response_mask,
            dtype=torch.float32,
        )
        for row_index, credit in prepared.record_credits.items():
            key = credit.trajectory_key
            utility[row_index] = credit.utility
            task_baseline[row_index] = credit.task_baseline
            if credit.candidate_state_baseline is not None:
                state_baseline[row_index] = credit.candidate_state_baseline
                state_baseline_available[row_index] = True
            base_advantage[row_index] = credit.base_advantage
            invalid_adjustment[row_index] = credit.invalid_adjustment
            overlong_adjustment[row_index] = credit.overlong_adjustment
            final_unweighted_scalar[row_index] = credit.final_unweighted_advantage
            trajectory_weight[row_index] = finalized.trajectory_weights[key]
            trainable_token_count[row_index] = finalized.trajectory_token_counts[key]
            ordinary_positions = torch.nonzero(
                ordinary_response_mask[row_index], as_tuple=False
            ).flatten()
            if len(ordinary_positions):
                token_level_rewards[row_index, ordinary_positions[-1]] = credit.utility

        output_fields: dict[str, torch.Tensor] = {
            "meta_record_row_index": torch.arange(
                record_count,
                dtype=torch.long,
                device=device,
            ),
            "advantages": response_to_nested(
                optimizer_advantages,
                nested_response_mask,
            ),
            "returns": response_to_nested(optimizer_advantages, nested_response_mask),
            "rm_scores": response_to_nested(
                token_level_rewards,
                nested_response_mask,
            ),
            "token_level_rewards": response_to_nested(
                token_level_rewards,
                nested_response_mask,
            ),
            "response_mask": response_to_nested(
                finalized.response_mask,
                nested_response_mask,
            ),
            "meta_unweighted_advantages": response_to_nested(
                finalized.unweighted_advantages,
                nested_response_mask,
            ),
            "meta_traj_loss_weight": response_to_nested(
                finalized.token_weights,
                nested_response_mask,
            ),
            "meta_trajectory_utility": utility,
            "meta_task_baseline": task_baseline,
            "meta_state_baseline": state_baseline,
            "meta_state_baseline_available": state_baseline_available,
            "meta_base_advantage": base_advantage,
            "meta_invalid_adjustment": invalid_adjustment,
            "meta_overlong_adjustment": overlong_adjustment,
            "meta_final_unweighted_advantage": final_unweighted_scalar,
            "meta_trajectory_weight": trajectory_weight,
            "meta_trainable_token_count": trainable_token_count,
            "meta_ppo_minibatch_scale": torch.full(
                (record_count,),
                ppo_minibatch_scale,
                dtype=torch.float32,
                device=device,
            ),
        }
        if rollout_is_weights is not None:
            output_fields["rollout_is_weights"] = response_to_nested(
                rollout_is_weights,
                nested_response_mask,
            )
        batch = tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=TensorDict(output_fields, batch_size=record_count),
        )

        if credit_mode == "meta_trajectory":
            metrics.update(meta_trajectory_metrics(finalized))
        else:
            metrics["lamer_form/records"] = float(len(prepared.real_indices))
        metrics.update(
            meta_trajectory_rollout_diversity_metrics(
                prepared,
                response_token_ids=response_token_ids,
                response_mask=ordinary_response_mask,
            )
        )
        if credit_mode == "meta_trajectory":
            _write_meta_integration_manifest(
                prepared=prepared,
                finalized=finalized,
                extra_fields=all_extra_fields,
                is_padding=is_padding,
                old_log_probs=old_log_probs_for_audit,
                rollout_is_weights=rollout_is_weights,
                gamma_step=float(self.config.algorithm.step_gamma),
                gamma_traj=float(self.config.algorithm.traj_gamma),
                ppo_minibatch_record_count=ppo_minibatch_record_count,
                ppo_minibatch_scale=ppo_minibatch_scale,
            )
        if loss_mode == "token-sum":
            unit_ratio_losses = -finalized.unweighted_advantages.to(torch.float64)
            reference_loss, production_loss, residual = (
                compare_reference_and_production_token_losses(
                    unit_ratio_losses,
                    finalized,
                )
            )
            metrics.update(
                {
                    "meta_credit/reference_unit_ratio_loss": reference_loss,
                    "meta_credit/production_unit_ratio_token_sum": production_loss,
                    "meta_credit/reference_token_sum_abs_residual": residual,
                    "meta_credit/reference_comparison_excludes_rollout_is": float(
                        rollout_is_weights is not None
                    ),
                }
            )
        metrics.update(
            {
                "ablation/token_mean": float(loss_mode == "token-mean"),
                "meta_update/ppo_minibatch_record_count": float(ppo_minibatch_record_count),
                "meta_update/ppo_minibatch_scale": ppo_minibatch_scale,
            }
        )
        return batch

    def _update_actor(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Run standard outer PPO record minibatches with frozen global credit.

        Complete-trajectory quantities and the ``R / b`` stochastic-estimator
        scale were attached before this method.  VERL may now partition record
        rows normally.  Each outer PPO minibatch is one optimizer boundary;
        FSDP memory microbatches remain gradient-accumulation fragments only.
        """
        credit_mode = str(
            self.config.algorithm.get("credit_mode", "original_lamer")
        )
        if credit_mode not in {"meta_trajectory", "lamer_form"}:
            return super()._update_actor(batch, metrics)

        actor_config = self.config.actor_rollout_ref.actor
        meta_config = self.config.algorithm.get("meta_trajectory")
        if meta_config is None:
            raise ValueError("algorithm.meta_trajectory configuration is required")
        if not bool(meta_config.get("require_one_step_per_ppo_minibatch", False)):
            raise ValueError(
                "meta_trajectory requires require_one_step_per_ppo_minibatch=True"
            )
        if os.environ.get("LAMER_META_PPO_MINIBATCH_UPDATE", "0") != "1":
            raise ValueError(
                "meta_trajectory PPO-minibatch updates require "
                "LAMER_META_PPO_MINIBATCH_UPDATE=1 so context cleanup does not add "
                "redundant optimizer.zero_grad calls"
            )
        if int(actor_config.ppo_epochs) != 1:
            raise ValueError(
                "meta_trajectory PPO-minibatch updates require actor.ppo_epochs=1"
            )
        if int(getattr(self, "parameter_sync_step", 1)) != 1:
            raise ValueError(
                "meta_trajectory PPO-minibatch updates require parameter_sync_step=1"
            )
        if str(actor_config.loss_agg_mode) not in {"token-sum", "token-mean"}:
            raise ValueError(
                "ablation PPO-minibatch updates require token-sum or token-mean"
            )
        if str(actor_config.policy_loss.loss_mode) != "meta_vanilla":
            raise ValueError(
                "meta_trajectory PPO-minibatch updates require "
                "actor.policy_loss.loss_mode='meta_vanilla'"
            )

        complete_record_count = len(batch)
        if complete_record_count <= 0:
            raise ValueError("Cannot update from an empty meta-trajectory rollout batch")
        ppo_minibatch_record_count = int(actor_config.ppo_mini_batch_size) * int(
            self.config.actor_rollout_ref.rollout.n
        )
        if (
            ppo_minibatch_record_count <= 0
            or complete_record_count % ppo_minibatch_record_count != 0
        ):
            raise ValueError(
                "The padded meta rollout record count must be divisible by the "
                "effective PPO minibatch record count"
            )
        ppo_minibatch_count = complete_record_count // ppo_minibatch_record_count

        calculate_entropy = actor_config.calculate_entropy or (
            actor_config.entropy_coeff != 0.0
        )
        distillation_use_topk = (
            self.distillation_config.distillation_loss.loss_settings.use_topk
            if is_distillation_enabled(self.config.get("distillation"))
            else False
        )
        distillation_only = False
        if is_distillation_enabled(self.config.get("distillation")):
            distillation_loss_cfg = self.distillation_config.distillation_loss
            distillation_only = (
                distillation_use_topk
                and not distillation_loss_cfg.use_task_rewards
                and not distillation_loss_cfg.use_policy_gradient
            )

        # Match upstream/original_lamer: ``ppo_mini_batch_size`` is expressed
        # in prompts and multiplied by rollout.n to form a global record-row
        # minibatch.  TrainingWorker's DataLoader performs the configured row
        # shuffle/partition; FSDP may then form arbitrary memory fragments.
        batch.extra_info.update(
            {
                "calculate_entropy": calculate_entropy,
                "distillation_use_topk": distillation_use_topk,
                "distillation_only": distillation_only,
                "global_batch_size": ppo_minibatch_record_count,
                "mini_batch_size": ppo_minibatch_record_count,
                "epochs": 1,
                "seed": actor_config.data_loader_seed,
                "dataloader_kwargs": {"shuffle": actor_config.shuffle},
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
            }
        )

        output: TensorDict = self.actor_rollout_wg.update_actor(batch)
        output = rename_dict(output["metrics"], "actor/")
        output["perf/mfu/actor"] = output.pop("actor/mfu")
        metrics.update(reduce_metrics(output))
        metrics.update(
            {
                "meta_update/complete_record_count": float(complete_record_count),
                "meta_update/ppo_minibatch_record_count": float(
                    ppo_minibatch_record_count
                ),
                "meta_update/ppo_minibatches": float(ppo_minibatch_count),
                "meta_update/optimizer_steps": float(ppo_minibatch_count),
                "meta_update/scheduler_steps": float(ppo_minibatch_count),
            }
        )
        return batch
