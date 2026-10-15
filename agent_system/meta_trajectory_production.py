"""Production-tensor adapter for complete-meta-trajectory credit.

The mathematical estimator lives in :mod:`agent_system.meta_trajectory_credit`.
This module only performs the auditable translation between one complete
rollout batch (record metadata plus padded response masks) and the tensors that
the VERL actor consumes.  It intentionally has no VERL or TransferQueue
dependency so the boundary can be exercised on CPU.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Literal

import torch

from agent_system.meta_trajectory_credit import (
    BaselineAssignmentResult,
    MetaCreditRecord,
    StateFallbackReason,
    StateGroupSummary,
    TaskCenteringResult,
    TrajectoryKey,
    assign_state_or_task_advantages,
    direct_meta_trajectory_utilities,
    invalid_penalty_adjustment,
    is_hard_truncated,
    mean_center_task_utilities,
    reference_complete_meta_trajectory_loss,
    reference_token_weights,
    reference_weighted_token_sum,
    response_mask_for_overlong_mode,
    soft_overlong_penalty,
)


OverlongMode = Literal["soft_shaping", "filter"]


class RecordValidityKind(str, Enum):
    """Phase-aware parse/truncation classification used by local shaping."""

    VALID_PLAY = "valid_play"
    MALFORMED_PLAY = "malformed_play"
    VALID_REFLECTION = "valid_reflection"
    MALFORMED_REFLECTION = "malformed_reflection"
    HARD_TRUNCATED_PLAY = "hard_truncated_play"
    HARD_TRUNCATED_REFLECTION = "hard_truncated_reflection"


@dataclass(frozen=True, slots=True)
class RecordValidity:
    phase: Literal["play", "reflect"]
    parser_valid: bool
    hard_truncated: bool
    kind: RecordValidityKind


@dataclass(frozen=True, slots=True)
class PreparedRecordCredit:
    """Unweighted credit and diagnostics for one real generation record."""

    row_index: int
    task_uid: str
    trajectory_uid: str
    phase: Literal["play", "reflect"]
    attempt_index: int
    turn_index: int
    utility: float
    task_baseline: float
    candidate_state_baseline: float | None
    selected_baseline: float
    baseline_kind: Literal["task", "state"]
    fallback_reasons: tuple[StateFallbackReason, ...]
    base_advantage: float
    invalid_adjustment: float
    overlong_adjustment: float
    final_unweighted_advantage: float
    generated_response_length: int
    validity: RecordValidity
    reflection_recovered_from_truncation: bool

    @property
    def trajectory_key(self) -> TrajectoryKey:
        return (self.task_uid, self.trajectory_uid)


@dataclass(frozen=True, slots=True)
class PreparedMetaTrajectoryBatch:
    """Estimator result before rollout-correction/rejection masking."""

    response_mask: torch.Tensor
    real_indices: tuple[int, ...]
    records: tuple[MetaCreditRecord, ...]
    trajectory_utilities: Mapping[TrajectoryKey, float]
    task_centering: TaskCenteringResult
    baseline_assignments: BaselineAssignmentResult
    record_credits: Mapping[int, PreparedRecordCredit]
    trajectory_rows: Mapping[TrajectoryKey, tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class FinalizedMetaTrajectoryBatch:
    """Final masks and actor tensors after every approved mask change."""

    prepared: PreparedMetaTrajectoryBatch
    response_mask: torch.Tensor
    unweighted_advantages: torch.Tensor
    weighted_advantages: torch.Tensor
    returns: torch.Tensor
    token_weights: torch.Tensor
    trajectory_token_counts: Mapping[TrajectoryKey, int]
    trajectory_weights: Mapping[TrajectoryKey, float]
    trajectory_token_masses: Mapping[TrajectoryKey, float]


def ppo_record_minibatch_scale(
    *,
    rollout_record_count: int,
    ppo_minibatch_record_count: int,
) -> float:
    """Return the unbiased record-minibatch scale ``R / b``.

    VERL samples outer PPO minibatches uniformly from the padded record rows.
    A size-``b`` subset contains each of the ``R`` rows with probability
    ``b / R``.  Multiplying its partial weighted token sum by ``R / b``
    therefore recovers the complete frozen-rollout objective in expectation.
    Synthetic padding rows are part of ``R`` but have zero loss masks, so they
    affect the sampling probability without changing the complete objective.
    """
    total = int(rollout_record_count)
    minibatch = int(ppo_minibatch_record_count)
    if total <= 0:
        raise ValueError("rollout_record_count must be positive")
    if minibatch <= 0:
        raise ValueError("ppo_minibatch_record_count must be positive")
    if minibatch > total:
        raise ValueError(
            "ppo_minibatch_record_count cannot exceed rollout_record_count"
        )
    if total % minibatch != 0:
        raise ValueError(
            "rollout_record_count must be divisible by "
            "ppo_minibatch_record_count after VERL padding"
        )
    return float(total) / float(minibatch)


def scale_weighted_advantages_for_ppo_minibatches(
    weighted_advantages: torch.Tensor,
    *,
    rollout_record_count: int,
    ppo_minibatch_record_count: int,
) -> torch.Tensor:
    """Scale frozen whole-rollout token contributions for outer PPO SGD."""
    scale = ppo_record_minibatch_scale(
        rollout_record_count=rollout_record_count,
        ppo_minibatch_record_count=ppo_minibatch_record_count,
    )
    return weighted_advantages * scale


def _finite_nonnegative(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative, got {value!r}")
    return result


def _hashable_metadata(value: Any) -> Hashable:
    """Freeze collated list/array/dict metadata without changing equality."""
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return value
    if isinstance(value, torch.Tensor):
        return tuple(_hashable_metadata(item) for item in value.detach().cpu().tolist())
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple, dict)):
        return _hashable_metadata(value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_hashable_metadata(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            sorted((str(key), _hashable_metadata(item)) for key, item in value.items())
        )
    try:
        hash(value)
    except TypeError as error:
        raise ValueError(f"Unsupported unhashable rollout metadata: {value!r}") from error
    return value


def prepare_lamer_form_batch(
    *,
    task_uids: Sequence[Any],
    extra_fields: Sequence[Mapping[str, Any]],
    response_mask: torch.Tensor,
    is_padding: Sequence[bool],
    gamma_traj: float,
    gamma_step: float,
    tau_u: float,
    expected_siblings: int | None,
    invalid_penalty_enabled: bool,
    invalid_penalty_coefficient: float,
    overlong_penalty_enabled: bool,
    overlong_mode: OverlongMode,
    overlong_max_coefficient: float,
    play_cache_tokens: int,
    reflection_cache_tokens: int,
    step_advantage_weight: float = 1.0,
) -> PreparedMetaTrajectoryBatch:
    """Use LaMer's record-weighted base credit with C's masks/local shaping.

    The rollout computes ``episode_reward`` and ``step_return`` before this
    call, using the configured discounts.  Baselines include each real record
    occurrence, including repeated visits, but never synthetic padding.
    """
    prepared = prepare_meta_trajectory_batch(
        task_uids=task_uids,
        extra_fields=extra_fields,
        response_mask=response_mask,
        is_padding=is_padding,
        gamma_traj=gamma_traj,
        gamma_step=gamma_step,
        tau_u=tau_u,
        expected_siblings=expected_siblings,
        invalid_penalty_enabled=invalid_penalty_enabled,
        invalid_penalty_coefficient=invalid_penalty_coefficient,
        overlong_penalty_enabled=overlong_penalty_enabled,
        overlong_mode=overlong_mode,
        overlong_max_coefficient=overlong_max_coefficient,
        play_cache_tokens=play_cache_tokens,
        reflection_cache_tokens=reflection_cache_tokens,
    )
    if not math.isfinite(step_advantage_weight):
        raise ValueError("step_advantage_weight must be finite")
    grouped_task: dict[Hashable, list[int]] = defaultdict(list)
    grouped_state: dict[tuple[Hashable, Hashable], list[int]] = defaultdict(list)
    episode_scores: dict[int, float] = {}
    step_returns: dict[int, float] = {}
    for row in prepared.real_indices:
        fields = extra_fields[row]
        for key in ("episode_reward", "step_return"):
            if key not in fields:
                raise ValueError(f"LaMer-form record {row} lacks {key}")
        episode_scores[row] = float(fields["episode_reward"])
        step_returns[row] = float(fields["step_return"])
        if not math.isfinite(episode_scores[row]) or not math.isfinite(step_returns[row]):
            raise ValueError(f"LaMer-form scores must be finite at row {row}")
        task = prepared.record_credits[row].task_uid
        grouped_task[task].append(row)
        grouped_state[(task, _hashable_metadata(fields["anchor_obs"]))].append(row)

    # Match the legacy float32 tensor subtraction as well as its group means.
    ordered_rows = prepared.real_indices
    position = {row: index for index, row in enumerate(ordered_rows)}
    episode_tensor = torch.tensor(
        [episode_scores[row] for row in ordered_rows], dtype=torch.float32
    )
    step_tensor = torch.tensor(
        [step_returns[row] for row in ordered_rows], dtype=torch.float32
    )
    coarse_tensor = torch.empty_like(episode_tensor)
    fine_tensor = torch.empty_like(step_tensor)
    for rows in grouped_task.values():
        indices = [position[row] for row in rows]
        coarse_tensor[indices] = episode_tensor[indices] - episode_tensor[indices].mean()
    for rows in grouped_state.values():
        indices = [position[row] for row in rows]
        fine_tensor[indices] = step_tensor[indices] - step_tensor[indices].mean()
    credits = dict(prepared.record_credits)
    for row in prepared.real_indices:
        credit = credits[row]
        index = position[row]
        base = float(coarse_tensor[index] + step_advantage_weight * fine_tensor[index])
        credits[row] = replace(
            credit,
            base_advantage=base,
            final_unweighted_advantage=(
                base + credit.invalid_adjustment + credit.overlong_adjustment
            ),
        )
    return replace(prepared, record_credits=credits)


def classify_record_validity(
    *,
    phase: str,
    parser_valid: bool,
    generated_response_length: int,
    max_response_length: int,
) -> RecordValidity:
    """Classify validity without applying play-only semantics to reflections."""
    if phase not in {"play", "reflect"}:
        raise ValueError(f"Unknown generation phase: {phase!r}")
    typed_phase: Literal["play", "reflect"] = phase
    hard_truncated = is_hard_truncated(
        generated_response_length,
        max_response_length,
    )
    if hard_truncated:
        kind = (
            RecordValidityKind.HARD_TRUNCATED_PLAY
            if phase == "play"
            else RecordValidityKind.HARD_TRUNCATED_REFLECTION
        )
    elif phase == "play":
        kind = (
            RecordValidityKind.VALID_PLAY
            if parser_valid
            else RecordValidityKind.MALFORMED_PLAY
        )
    else:
        kind = (
            RecordValidityKind.VALID_REFLECTION
            if parser_valid
            else RecordValidityKind.MALFORMED_REFLECTION
        )
    return RecordValidity(
        phase=typed_phase,
        parser_valid=bool(parser_valid),
        hard_truncated=hard_truncated,
        kind=kind,
    )


def _required_record_fields() -> frozenset[str]:
    return frozenset(
        {
            "task_uid",
            "traj_uid",
            "phase",
            "attempt_index",
            "turn_index",
            "environment_reward",
            "anchor_obs",
            "parsed_action",
            "is_action_valid",
            "response_token_count",
            "response_token_limit",
            "response_token_cap_hit",
            "reflection_recovered_from_truncation",
        }
    )


def _validate_batch_inputs(
    task_uids: Sequence[Any],
    extra_fields: Sequence[Mapping[str, Any]],
    response_mask: torch.Tensor,
    is_padding: Sequence[bool],
) -> None:
    row_count = len(task_uids)
    if len(extra_fields) != row_count or len(is_padding) != row_count:
        raise ValueError("task_uids, extra_fields, and is_padding must have equal length")
    if response_mask.ndim != 2 or response_mask.shape[0] != row_count:
        raise ValueError(
            "response_mask must have shape [record_count, response_width]"
        )
    if not torch.all((response_mask == 0) | (response_mask == 1)):
        raise ValueError("response_mask must be binary")


def prepare_meta_trajectory_batch(
    *,
    task_uids: Sequence[Any],
    extra_fields: Sequence[Mapping[str, Any]],
    response_mask: torch.Tensor,
    is_padding: Sequence[bool],
    gamma_traj: float,
    gamma_step: float,
    tau_u: float,
    expected_siblings: int | None,
    invalid_penalty_enabled: bool,
    invalid_penalty_coefficient: float,
    overlong_penalty_enabled: bool,
    overlong_mode: OverlongMode,
    overlong_max_coefficient: float,
    play_cache_tokens: int,
    reflection_cache_tokens: int,
) -> PreparedMetaTrajectoryBatch:
    """Reconstruct, center, shape, and pre-mask one complete rollout batch.

    The returned mask includes only prompt-independent response masking and the
    optional explicit hard-truncation filter.  Rollout-rejection masking is
    deliberately applied by the trainer between this function and
    :func:`finalize_meta_trajectory_batch`.
    """
    _validate_batch_inputs(task_uids, extra_fields, response_mask, is_padding)
    if expected_siblings is not None and expected_siblings <= 0:
        raise ValueError("expected_siblings must be positive when supplied")
    if overlong_mode not in {"soft_shaping", "filter"}:
        raise ValueError(f"Unknown overlong_mode: {overlong_mode!r}")
    _finite_nonnegative(
        invalid_penalty_coefficient,
        name="invalid_penalty_coefficient",
    )
    _finite_nonnegative(
        overlong_max_coefficient,
        name="overlong_max_coefficient",
    )
    if play_cache_tokens <= 0 or reflection_cache_tokens <= 0:
        raise ValueError("phase-specific overlong cache widths must be positive")

    pre_correction_mask = response_mask.clone()
    real_indices: list[int] = []
    pure_records: list[MetaCreditRecord] = []
    trajectory_rows_lists: dict[TrajectoryKey, list[int]] = defaultdict(list)
    structural_indices: dict[TrajectoryKey, set[tuple[str, int, int]]] = defaultdict(set)
    validity_by_row: dict[int, RecordValidity] = {}
    lengths_by_row: dict[int, int] = {}

    required = _required_record_fields()
    for row_index, (queued_uid, fields, padding) in enumerate(
        zip(task_uids, extra_fields, is_padding, strict=True)
    ):
        if padding:
            pre_correction_mask[row_index].zero_()
            continue
        missing = required - fields.keys()
        if missing:
            raise ValueError(
                f"Meta-trajectory record {row_index} is missing metadata: {sorted(missing)}"
            )

        task_uid = str(queued_uid)
        metadata_task_uid = str(fields["task_uid"])
        if task_uid != metadata_task_uid:
            raise ValueError(
                f"Queued uid {task_uid!r} disagrees with extra_fields.task_uid "
                f"{metadata_task_uid!r} at row {row_index}"
            )
        trajectory_uid = str(fields["traj_uid"])
        phase = str(fields["phase"])
        attempt_index = int(fields["attempt_index"])
        turn_index = int(fields["turn_index"])
        response_length = int(fields["response_token_count"])
        response_limit = int(fields["response_token_limit"])
        validity = classify_record_validity(
            phase=phase,
            parser_valid=bool(fields["is_action_valid"]),
            generated_response_length=response_length,
            max_response_length=response_limit,
        )
        if bool(fields["response_token_cap_hit"]) != validity.hard_truncated:
            raise ValueError(
                f"response_token_cap_hit disagrees with cap equality at row {row_index}"
            )
        ordinary_token_count = int(response_mask[row_index].sum().item())
        if ordinary_token_count != response_length:
            raise ValueError(
                f"response mask count {ordinary_token_count} disagrees with generated "
                f"length {response_length} at row {row_index} before rollout correction"
            )

        typed_phase: Literal["play", "reflect"] = validity.phase
        record = MetaCreditRecord(
            record_id=row_index,
            task_uid=task_uid,
            trajectory_uid=trajectory_uid,
            phase=typed_phase,
            attempt_index=attempt_index,
            turn_index=turn_index,
            environment_reward=float(fields["environment_reward"]),
            anchor_observation=_hashable_metadata(fields["anchor_obs"]),
            projected_action=(
                _hashable_metadata(fields["parsed_action"])
                if fields["parsed_action"] is not None
                else None
            ),
            is_action_valid=bool(fields["is_action_valid"]),
        )
        structural_index = (phase, attempt_index, turn_index)
        if structural_index in structural_indices[record.trajectory_key]:
            raise ValueError(
                f"Duplicate structural record index {structural_index!r} in "
                f"trajectory {record.trajectory_key!r}"
            )
        structural_indices[record.trajectory_key].add(structural_index)

        filtered = response_mask_for_overlong_mode(
            response_mask[row_index].detach().cpu().tolist(),
            response_length,
            response_limit,
            overlong_mode=overlong_mode,
        )
        pre_correction_mask[row_index] = torch.as_tensor(
            filtered,
            dtype=pre_correction_mask.dtype,
            device=pre_correction_mask.device,
        )
        real_indices.append(row_index)
        pure_records.append(record)
        trajectory_rows_lists[record.trajectory_key].append(row_index)
        validity_by_row[row_index] = validity
        lengths_by_row[row_index] = response_length

    if not real_indices:
        raise ValueError("Meta-trajectory batch contains no non-padding records")

    trajectory_utilities = direct_meta_trajectory_utilities(
        pure_records,
        gamma_traj=gamma_traj,
        gamma_step=gamma_step,
    )
    siblings_by_task: dict[str, set[str]] = defaultdict(set)
    for task_uid, trajectory_uid in trajectory_utilities:
        siblings_by_task[str(task_uid)].add(str(trajectory_uid))
    if expected_siblings is not None:
        incomplete = {
            task_uid: len(siblings)
            for task_uid, siblings in siblings_by_task.items()
            if len(siblings) != expected_siblings
        }
        if incomplete:
            raise ValueError(
                "Incomplete sibling groups: expected "
                f"{expected_siblings}, observed {incomplete}"
            )

    task_centering = mean_center_task_utilities(trajectory_utilities)
    baseline_assignments = assign_state_or_task_advantages(
        pure_records,
        trajectory_utilities,
        task_centering,
        tau_u=tau_u,
    )

    record_credits: dict[int, PreparedRecordCredit] = {}
    fields_by_row = {index: extra_fields[index] for index in real_indices}
    for record in pure_records:
        row_index = int(record.record_id)
        fields = fields_by_row[row_index]
        validity = validity_by_row[row_index]
        assignment = baseline_assignments.records[row_index]
        invalid_adjustment = (
            invalid_penalty_adjustment(
                is_action_valid=validity.parser_valid,
                hard_truncated=validity.hard_truncated,
                coefficient=invalid_penalty_coefficient,
            )
            if invalid_penalty_enabled
            else 0.0
        )
        cache_tokens = (
            play_cache_tokens if record.phase == "play" else reflection_cache_tokens
        )
        response_limit = int(fields["response_token_limit"])
        overlong_adjustment = (
            soft_overlong_penalty(
                lengths_by_row[row_index],
                response_limit,
                cache_width=cache_tokens,
                max_penalty=overlong_max_coefficient,
            )
            if overlong_penalty_enabled
            else 0.0
        )
        state_summary: StateGroupSummary | None = (
            baseline_assignments.groups.get(assignment.state_key)
            if assignment.state_key is not None
            else None
        )
        final_advantage = (
            assignment.advantage + invalid_adjustment + overlong_adjustment
        )
        record_credits[row_index] = PreparedRecordCredit(
            row_index=row_index,
            task_uid=str(record.task_uid),
            trajectory_uid=str(record.trajectory_uid),
            phase=record.phase,
            attempt_index=record.attempt_index,
            turn_index=record.turn_index,
            utility=float(trajectory_utilities[record.trajectory_key]),
            task_baseline=float(task_centering.baselines[record.task_uid]),
            candidate_state_baseline=(
                float(state_summary.candidate_state_baseline)
                if state_summary is not None
                else None
            ),
            selected_baseline=float(assignment.baseline),
            baseline_kind=assignment.baseline_kind,
            fallback_reasons=assignment.fallback_reasons,
            base_advantage=float(assignment.advantage),
            invalid_adjustment=float(invalid_adjustment),
            overlong_adjustment=float(overlong_adjustment),
            final_unweighted_advantage=float(final_advantage),
            generated_response_length=lengths_by_row[row_index],
            validity=validity,
            reflection_recovered_from_truncation=bool(
                fields["reflection_recovered_from_truncation"]
            ),
        )

    return PreparedMetaTrajectoryBatch(
        response_mask=pre_correction_mask,
        real_indices=tuple(real_indices),
        records=tuple(pure_records),
        trajectory_utilities=dict(trajectory_utilities),
        task_centering=task_centering,
        baseline_assignments=baseline_assignments,
        record_credits=record_credits,
        trajectory_rows={
            key: tuple(rows) for key, rows in trajectory_rows_lists.items()
        },
    )


def finalize_meta_trajectory_batch(
    prepared: PreparedMetaTrajectoryBatch,
    *,
    final_response_mask: torch.Tensor | None = None,
) -> FinalizedMetaTrajectoryBatch:
    """Compute post-correction ``N_j`` and broadcast exact global weights."""
    mask = (
        prepared.response_mask.clone()
        if final_response_mask is None
        else final_response_mask.clone()
    )
    if mask.shape != prepared.response_mask.shape:
        raise ValueError("final_response_mask shape changed after preparation")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("final_response_mask must be binary")
    if torch.any(mask.to(torch.bool) & ~prepared.response_mask.to(torch.bool)):
        raise ValueError("final_response_mask cannot unmask a pre-filtered token")

    token_counts: dict[TrajectoryKey, int] = {}
    nested_counts: dict[str, dict[str, int]] = defaultdict(dict)
    for trajectory_key, rows in prepared.trajectory_rows.items():
        count = int(mask[list(rows)].sum().item())
        token_counts[trajectory_key] = count
        nested_counts[str(trajectory_key[0])][str(trajectory_key[1])] = count
    nested_weights = reference_token_weights(nested_counts)
    trajectory_weights = {
        (task_uid, trajectory_uid): weight
        for task_uid, siblings in nested_weights.items()
        for trajectory_uid, weight in siblings.items()
    }

    unweighted = torch.zeros_like(mask, dtype=torch.float32)
    token_weights = torch.zeros_like(mask, dtype=torch.float32)
    for row_index, credit in prepared.record_credits.items():
        key = credit.trajectory_key
        row_mask = mask[row_index].to(torch.float32)
        unweighted[row_index] = credit.final_unweighted_advantage * row_mask
        token_weights[row_index] = trajectory_weights[key] * row_mask
    weighted = unweighted * token_weights

    trajectory_masses = {
        key: float(
            token_weights[list(rows)].sum(dtype=torch.float64).item()
        )
        for key, rows in prepared.trajectory_rows.items()
    }
    return FinalizedMetaTrajectoryBatch(
        prepared=prepared,
        response_mask=mask,
        unweighted_advantages=unweighted,
        weighted_advantages=weighted,
        returns=weighted.clone(),
        token_weights=token_weights,
        trajectory_token_counts=token_counts,
        trajectory_weights=trajectory_weights,
        trajectory_token_masses=trajectory_masses,
    )


def token_losses_by_trajectory(
    token_losses: torch.Tensor,
    finalized: FinalizedMetaTrajectoryBatch,
) -> dict[str, dict[str, list[float]]]:
    """Convert a padded token-loss matrix to the pure nested reference form."""
    if token_losses.shape != finalized.response_mask.shape:
        raise ValueError("token_losses must match the finalized response mask")
    result: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row_index, credit in finalized.prepared.record_credits.items():
        selected = token_losses[row_index].masked_select(
            finalized.response_mask[row_index].to(torch.bool)
        )
        result[credit.task_uid][credit.trajectory_uid].extend(
            float(value) for value in selected.detach().cpu().tolist()
        )
    return {
        task_uid: {trajectory_uid: list(losses) for trajectory_uid, losses in siblings.items()}
        for task_uid, siblings in result.items()
    }


def compare_reference_and_production_token_losses(
    unweighted_token_losses: torch.Tensor,
    finalized: FinalizedMetaTrajectoryBatch,
) -> tuple[float, float, float]:
    """Return reference loss, weighted token sum, and absolute residual."""
    nested_losses = token_losses_by_trajectory(unweighted_token_losses, finalized)
    reference = reference_complete_meta_trajectory_loss(nested_losses)
    nested_weights: dict[str, dict[str, float]] = defaultdict(dict)
    for (task_uid, trajectory_uid), value in finalized.trajectory_weights.items():
        nested_weights[str(task_uid)][str(trajectory_uid)] = float(value)
    production = reference_weighted_token_sum(nested_losses, nested_weights)
    return reference, production, abs(reference - production)


def meta_trajectory_metrics(
    finalized: FinalizedMetaTrajectoryBatch,
) -> dict[str, float]:
    """Compact scalar metrics safe to attach to the existing trainer logger."""
    credits = tuple(finalized.prepared.record_credits.values())
    groups = tuple(finalized.prepared.baseline_assignments.groups.values())
    fallback_counts = Counter(
        reason.value for credit in credits for reason in credit.fallback_reasons
    )
    kinds = Counter(credit.validity.kind.value for credit in credits)
    task_advantage_sums: dict[str, float] = defaultdict(float)
    for (task_uid, _), advantage in finalized.prepared.task_centering.advantages.items():
        task_advantage_sums[str(task_uid)] += float(advantage)

    metrics: dict[str, float] = {
        "meta_credit/tasks": float(len(finalized.prepared.task_centering.baselines)),
        "meta_credit/trajectories": float(len(finalized.trajectory_token_counts)),
        "meta_credit/records": float(len(credits)),
        "meta_credit/state_groups": float(len(groups)),
        "meta_credit/state_groups_eligible": float(sum(group.eligible for group in groups)),
        "meta_credit/repeated_state_visits": float(
            sum(
                max(
                    0,
                    sum(
                        1
                        for record in finalized.prepared.records
                        if record.phase == "play"
                        and record.attempt_index == 0
                        and (record.task_uid, record.anchor_observation) == group.state_key
                    )
                    - group.unique_trajectory_count,
                )
                for group in groups
            )
        ),
        "meta_credit/reflections_recovered": float(
            sum(credit.reflection_recovered_from_truncation for credit in credits)
        ),
        "meta_credit/invalid_adjusted_records": float(
            sum(credit.invalid_adjustment != 0.0 for credit in credits)
        ),
        "meta_credit/soft_overlong_adjusted_records": float(
            sum(credit.overlong_adjustment != 0.0 for credit in credits)
        ),
        "meta_credit/task_centering_max_abs_residual": max(
            (abs(value) for value in task_advantage_sums.values()),
            default=0.0,
        ),
    }
    for reason, count in fallback_counts.items():
        metrics[f"meta_credit/fallback/{reason}"] = float(count)
    for kind, count in kinds.items():
        metrics[f"meta_credit/validity/{kind}"] = float(count)
    return metrics


def meta_trajectory_rollout_diversity_metrics(
    prepared: PreparedMetaTrajectoryBatch,
    *,
    response_token_ids: torch.Tensor | None = None,
    response_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Summarize unique sibling outcomes, utilities, actions, and responses.

    Records are first collapsed by ``(task_uid, trajectory_uid)`` so repeated
    state visits never give a sibling more than one vote.  Action paths retain
    every real play decision in attempt/turn order.  When response tensors are
    supplied, response diversity uses only the first play generation from each
    complete meta-trajectory and the pre-correction rollout mask.
    """

    if (response_token_ids is None) != (response_mask is None):
        raise ValueError(
            "response_token_ids and response_mask must be supplied together"
        )
    if response_token_ids is not None:
        if response_token_ids.ndim != 2 or response_mask.ndim != 2:
            raise ValueError("response tensors must be rank two")
        if response_token_ids.shape != response_mask.shape:
            raise ValueError("response_token_ids and response_mask must have equal shape")

    records_by_trajectory: dict[
        TrajectoryKey, list[MetaCreditRecord]
    ] = defaultdict(list)
    for record in prepared.records:
        records_by_trajectory[record.trajectory_key].append(record)

    trajectories_by_task: dict[str, list[TrajectoryKey]] = defaultdict(list)
    for trajectory_key in records_by_trajectory:
        trajectories_by_task[str(trajectory_key[0])].append(trajectory_key)

    group_sizes: list[int] = []
    success_counts: list[int] = []
    utility_unique_fractions: list[float] = []
    utility_all_equal: list[bool] = []
    action_sequence_unique_fractions: list[float] = []
    fully_collapsed_action_groups: list[bool] = []
    first_action_unique_fractions: list[float] = []
    first_response_unique_fractions: list[float] = []

    for trajectory_keys in trajectories_by_task.values():
        group_size = len(trajectory_keys)
        if not group_size:
            continue
        group_sizes.append(group_size)

        successes: list[bool] = []
        utilities: list[float] = []
        action_sequences: list[tuple[Hashable | None, ...]] = []
        first_actions: list[Hashable | None] = []
        first_responses: list[tuple[int, ...]] = []

        for trajectory_key in trajectory_keys:
            trajectory_records = records_by_trajectory[trajectory_key]
            successes.append(
                any(float(record.environment_reward) > 0.0 for record in trajectory_records)
            )
            utilities.append(float(prepared.trajectory_utilities[trajectory_key]))

            play_records = sorted(
                (record for record in trajectory_records if record.phase == "play"),
                key=lambda record: (record.attempt_index, record.turn_index),
            )
            action_sequence = tuple(record.projected_action for record in play_records)
            action_sequences.append(action_sequence)
            if play_records:
                first_actions.append(play_records[0].projected_action)
                if response_token_ids is not None:
                    row_index = int(play_records[0].record_id)
                    if row_index < 0 or row_index >= response_token_ids.shape[0]:
                        raise ValueError(
                            f"response tensors have no row for record {row_index}"
                        )
                    selected = response_token_ids[row_index].masked_select(
                        response_mask[row_index].to(torch.bool)
                    )
                    first_responses.append(
                        tuple(int(token) for token in selected.detach().cpu().tolist())
                    )

        success_counts.append(sum(successes))
        unique_utilities = len(set(utilities))
        utility_unique_fractions.append(unique_utilities / group_size)
        utility_all_equal.append(unique_utilities == 1)

        unique_action_sequences = len(set(action_sequences))
        action_sequence_unique_fractions.append(unique_action_sequences / group_size)
        fully_collapsed_action_groups.append(unique_action_sequences == 1)
        if first_actions:
            first_action_unique_fractions.append(
                len(set(first_actions)) / len(first_actions)
            )
        if first_responses:
            first_response_unique_fractions.append(
                len(set(first_responses)) / len(first_responses)
            )

    prefix = "meta_diversity"
    group_count = len(group_sizes)
    if not group_count:
        return {
            f"{prefix}/groups/count": 0.0,
            f"{prefix}/groups/size_mean": 0.0,
            f"{prefix}/groups/size_min": 0.0,
            f"{prefix}/groups/size_max": 0.0,
            f"{prefix}/groups/mixed_outcome_fraction": 0.0,
            f"{prefix}/groups/all_failure_fraction": 0.0,
            f"{prefix}/groups/all_success_fraction": 0.0,
            f"{prefix}/groups/successful_trajectories_mean": 0.0,
            f"{prefix}/groups/successful_trajectories_std": 0.0,
            f"{prefix}/utility/unique_fraction_mean": 0.0,
            f"{prefix}/utility/all_equal_group_fraction": 0.0,
            f"{prefix}/action_sequence_unique_fraction_mean": 0.0,
            f"{prefix}/fully_collapsed_action_group_fraction": 0.0,
            f"{prefix}/first_action_unique_fraction_mean": 0.0,
        }

    all_failure = [count == 0 for count in success_counts]
    all_success = [
        count == size for count, size in zip(success_counts, group_sizes, strict=True)
    ]
    mixed = [
        not failed and not succeeded
        for failed, succeeded in zip(all_failure, all_success, strict=True)
    ]

    def mean(values: Sequence[float | int | bool]) -> float:
        return float(math.fsum(float(value) for value in values) / len(values))

    success_mean = mean(success_counts)
    success_std = math.sqrt(
        math.fsum((float(value) - success_mean) ** 2 for value in success_counts)
        / len(success_counts)
    )
    metrics = {
        f"{prefix}/groups/count": float(group_count),
        f"{prefix}/groups/size_mean": mean(group_sizes),
        f"{prefix}/groups/size_min": float(min(group_sizes)),
        f"{prefix}/groups/size_max": float(max(group_sizes)),
        f"{prefix}/groups/mixed_outcome_fraction": mean(mixed),
        f"{prefix}/groups/all_failure_fraction": mean(all_failure),
        f"{prefix}/groups/all_success_fraction": mean(all_success),
        f"{prefix}/groups/successful_trajectories_mean": success_mean,
        f"{prefix}/groups/successful_trajectories_std": success_std,
        f"{prefix}/utility/unique_fraction_mean": mean(utility_unique_fractions),
        f"{prefix}/utility/all_equal_group_fraction": mean(utility_all_equal),
        f"{prefix}/action_sequence_unique_fraction_mean": mean(
            action_sequence_unique_fractions
        ),
        f"{prefix}/fully_collapsed_action_group_fraction": mean(
            fully_collapsed_action_groups
        ),
        f"{prefix}/first_action_unique_fraction_mean": mean(
            first_action_unique_fractions
        ),
    }
    if first_response_unique_fractions:
        metrics[f"{prefix}/first_response_unique_fraction_mean"] = mean(
            first_response_unique_fractions
        )
    return metrics
