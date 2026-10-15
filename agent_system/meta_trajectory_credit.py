"""Pure reference helpers for complete-meta-trajectory credit assignment.

This module intentionally has no Torch, VERL, rollout, or trainer dependency.
It is the inspectable mathematical reference for the ``meta_trajectory`` credit
mode.  Production tensor code should be tested against these functions rather
than importing implementation details from the active training path.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypeAlias


TaskId: TypeAlias = Hashable
TrajectoryId: TypeAlias = Hashable
RecordId: TypeAlias = Hashable
TrajectoryKey: TypeAlias = tuple[TaskId, TrajectoryId]
StateKey: TypeAlias = tuple[TaskId, Hashable]

MIN_STATE_TRAJECTORIES = 3
MIN_DISTINCT_VALID_ACTIONS = 2


@dataclass(frozen=True, slots=True)
class MetaCreditRecord:
    """Minimal policy-record metadata needed by the pure estimator.

    ``attempt_index`` and ``turn_index`` are zero-based.  Only play records
    contribute environment reward to the nested utility.  ``is_padding`` rows
    are ignored by every batch-level helper.
    """

    record_id: RecordId
    task_uid: TaskId
    trajectory_uid: TrajectoryId
    phase: Literal["play", "reflect"]
    attempt_index: int
    turn_index: int
    environment_reward: float = 0.0
    anchor_observation: Hashable | None = None
    projected_action: Hashable | None = None
    is_action_valid: bool = False
    is_padding: bool = False

    @property
    def trajectory_key(self) -> TrajectoryKey:
        return (self.task_uid, self.trajectory_uid)


@dataclass(frozen=True, slots=True)
class TaskCenteringResult:
    """Unique-trajectory task means and mean-centered advantages."""

    baselines: Mapping[TaskId, float]
    advantages: Mapping[TrajectoryKey, float]


class StateFallbackReason(str, Enum):
    """Reason why a record retained its task-level advantage."""

    INSUFFICIENT_SUPPORT = "support"
    INSUFFICIENT_ACTION_DIVERSITY = "action_diversity"
    INSUFFICIENT_UTILITY_RANGE = "utility_range"
    NOT_ATTEMPT1_PLAY = "not_attempt1_play"


@dataclass(frozen=True, slots=True)
class StateGroupSummary:
    """Auditable statistics and eligibility decision for one raw-state group."""

    state_key: StateKey
    trajectory_utilities: Mapping[TrajectoryId, float]
    distinct_valid_actions: frozenset[Hashable]
    unique_trajectory_count: int
    utility_range: float
    candidate_state_baseline: float
    eligible: bool
    fallback_reasons: tuple[StateFallbackReason, ...]


@dataclass(frozen=True, slots=True)
class RecordBaselineAssignment:
    """The one selected base advantage for a real policy record."""

    record_id: RecordId
    trajectory_key: TrajectoryKey
    advantage: float
    baseline: float
    baseline_kind: Literal["task", "state"]
    state_key: StateKey | None
    fallback_reasons: tuple[StateFallbackReason, ...]


@dataclass(frozen=True, slots=True)
class BaselineAssignmentResult:
    """State-group diagnostics and record-level baseline selections."""

    groups: Mapping[StateKey, StateGroupSummary]
    records: Mapping[RecordId, RecordBaselineAssignment]


def _finite_float(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _validate_hashable(value: Hashable, *, name: str) -> None:
    try:
        hash(value)
    except TypeError as error:
        raise ValueError(f"{name} must be hashable, got {value!r}") from error


def _validate_record(record: MetaCreditRecord) -> None:
    _validate_hashable(record.record_id, name="record_id")
    _validate_hashable(record.task_uid, name="task_uid")
    _validate_hashable(record.trajectory_uid, name="trajectory_uid")
    if record.phase not in {"play", "reflect"}:
        raise ValueError(f"Unknown phase: {record.phase!r}")
    if record.attempt_index < 0:
        raise ValueError("attempt_index must be nonnegative")
    if record.turn_index < 0:
        raise ValueError("turn_index must be nonnegative")
    _finite_float(record.environment_reward, name="environment_reward")


def direct_meta_trajectory_utility(
    records: Iterable[MetaCreditRecord],
    *,
    gamma_traj: float,
    gamma_step: float,
) -> float:
    """Compute one trajectory's directly written nested Meta-RL utility.

    The calculation is

    ``sum_n gamma_traj**n * sum_t gamma_step**t * reward[n, t]``.

    Reflection records and padding rows do not contribute.  Record order has no
    mathematical role because attempt and turn indices carry the exponents.
    """

    gamma_traj = _finite_float(gamma_traj, name="gamma_traj")
    gamma_step = _finite_float(gamma_step, name="gamma_step")
    real_records = tuple(record for record in records if not record.is_padding)
    if not real_records:
        raise ValueError("A meta-trajectory must contain at least one real record")

    expected_key = real_records[0].trajectory_key
    terms: list[float] = []
    for record in real_records:
        _validate_record(record)
        if record.trajectory_key != expected_key:
            raise ValueError(
                "direct_meta_trajectory_utility received records from multiple "
                f"trajectories: {expected_key!r} and {record.trajectory_key!r}"
            )
        if record.phase == "play":
            terms.append(
                gamma_traj**record.attempt_index
                * gamma_step**record.turn_index
                * float(record.environment_reward)
            )
    return math.fsum(terms)


def direct_meta_trajectory_utilities(
    records: Iterable[MetaCreditRecord],
    *,
    gamma_traj: float,
    gamma_step: float,
) -> dict[TrajectoryKey, float]:
    """Group a real record batch and compute one direct utility per trajectory."""

    grouped: dict[TrajectoryKey, list[MetaCreditRecord]] = defaultdict(list)
    seen_record_ids: set[RecordId] = set()
    for record in records:
        _validate_record(record)
        if record.is_padding:
            continue
        if record.record_id in seen_record_ids:
            raise ValueError(f"Duplicate real record_id: {record.record_id!r}")
        seen_record_ids.add(record.record_id)
        grouped[record.trajectory_key].append(record)

    return {
        key: direct_meta_trajectory_utility(
            trajectory_records,
            gamma_traj=gamma_traj,
            gamma_step=gamma_step,
        )
        for key, trajectory_records in grouped.items()
    }


def mean_center_task_utilities(
    trajectory_utilities: Mapping[TrajectoryKey, float],
) -> TaskCenteringResult:
    """Give each unique sibling trajectory one vote in its task baseline."""

    if not trajectory_utilities:
        raise ValueError("trajectory_utilities must not be empty")

    by_task: dict[TaskId, dict[TrajectoryId, float]] = defaultdict(dict)
    normalized: dict[TrajectoryKey, float] = {}
    for key, utility in trajectory_utilities.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(f"Trajectory key must be (task_uid, traj_uid), got {key!r}")
        task_uid, trajectory_uid = key
        _validate_hashable(task_uid, name="task_uid")
        _validate_hashable(trajectory_uid, name="trajectory_uid")
        value = _finite_float(utility, name=f"utility[{key!r}]")
        by_task[task_uid][trajectory_uid] = value
        normalized[(task_uid, trajectory_uid)] = value

    baselines = {
        task_uid: math.fsum(siblings.values()) / len(siblings)
        for task_uid, siblings in by_task.items()
    }
    advantages = {
        key: utility - baselines[key[0]] for key, utility in normalized.items()
    }
    return TaskCenteringResult(baselines=baselines, advantages=advantages)


def assign_state_or_task_advantages(
    records: Iterable[MetaCreditRecord],
    trajectory_utilities: Mapping[TrajectoryKey, float],
    task_centering: TaskCenteringResult,
    *,
    tau_u: float,
) -> BaselineAssignmentResult:
    """Select state or task baseline for every real policy-generation record.

    Attempt-1 play means ``phase == 'play'`` and zero-based
    ``attempt_index == 0``.  Its raw-state group is eligible only with at least
    three unique trajectories, at least two distinct valid projected actions,
    and utility range strictly greater than ``tau_u``.  All other records use
    task credit with explicit fallback reasons.
    """

    tau_u = _finite_float(tau_u, name="tau_u")
    if tau_u < 0:
        raise ValueError("tau_u must be nonnegative")

    real_records: list[MetaCreditRecord] = []
    seen_record_ids: set[RecordId] = set()
    for record in records:
        _validate_record(record)
        if record.is_padding:
            continue
        if record.record_id in seen_record_ids:
            raise ValueError(f"Duplicate real record_id: {record.record_id!r}")
        seen_record_ids.add(record.record_id)
        if record.trajectory_key not in trajectory_utilities:
            raise ValueError(
                f"Missing utility for trajectory {record.trajectory_key!r}"
            )
        if record.trajectory_key not in task_centering.advantages:
            raise ValueError(
                f"Missing task advantage for trajectory {record.trajectory_key!r}"
            )
        if record.task_uid not in task_centering.baselines:
            raise ValueError(f"Missing task baseline for task {record.task_uid!r}")
        real_records.append(record)

    grouped_records: dict[StateKey, list[MetaCreditRecord]] = defaultdict(list)
    for record in real_records:
        if record.phase == "play" and record.attempt_index == 0:
            if record.anchor_observation is None:
                raise ValueError(
                    f"Attempt-1 play record {record.record_id!r} has no anchor observation"
                )
            _validate_hashable(record.anchor_observation, name="anchor_observation")
            grouped_records[(record.task_uid, record.anchor_observation)].append(record)

    group_summaries: dict[StateKey, StateGroupSummary] = {}
    for state_key, occurrences in grouped_records.items():
        utility_votes: dict[TrajectoryId, float] = {}
        valid_actions: set[Hashable] = set()
        for record in occurrences:
            utility = _finite_float(
                trajectory_utilities[record.trajectory_key],
                name=f"utility[{record.trajectory_key!r}]",
            )
            previous = utility_votes.setdefault(record.trajectory_uid, utility)
            if previous != utility:
                raise ValueError(
                    "Repeated visits from one trajectory disagree on utility: "
                    f"{record.trajectory_key!r}"
                )
            if record.is_action_valid and record.projected_action is not None:
                _validate_hashable(record.projected_action, name="projected_action")
                valid_actions.add(record.projected_action)

        unique_count = len(utility_votes)
        values = tuple(utility_votes.values())
        utility_range = max(values) - min(values)
        state_baseline = math.fsum(values) / unique_count
        reasons: list[StateFallbackReason] = []
        if unique_count < MIN_STATE_TRAJECTORIES:
            reasons.append(StateFallbackReason.INSUFFICIENT_SUPPORT)
        if len(valid_actions) < MIN_DISTINCT_VALID_ACTIONS:
            reasons.append(StateFallbackReason.INSUFFICIENT_ACTION_DIVERSITY)
        if utility_range <= tau_u:
            reasons.append(StateFallbackReason.INSUFFICIENT_UTILITY_RANGE)
        group_summaries[state_key] = StateGroupSummary(
            state_key=state_key,
            trajectory_utilities=dict(utility_votes),
            distinct_valid_actions=frozenset(valid_actions),
            unique_trajectory_count=unique_count,
            utility_range=utility_range,
            candidate_state_baseline=state_baseline,
            eligible=not reasons,
            fallback_reasons=tuple(reasons),
        )

    assignments: dict[RecordId, RecordBaselineAssignment] = {}
    for record in real_records:
        trajectory_key = record.trajectory_key
        utility = float(trajectory_utilities[trajectory_key])
        task_baseline = float(task_centering.baselines[record.task_uid])
        task_advantage = float(task_centering.advantages[trajectory_key])
        state_key: StateKey | None = None
        if record.phase == "play" and record.attempt_index == 0:
            state_key = (record.task_uid, record.anchor_observation)
            summary = group_summaries[state_key]
            if summary.eligible:
                baseline = summary.candidate_state_baseline
                advantage = utility - baseline
                baseline_kind: Literal["task", "state"] = "state"
                fallback_reasons: tuple[StateFallbackReason, ...] = ()
            else:
                baseline = task_baseline
                advantage = task_advantage
                baseline_kind = "task"
                fallback_reasons = summary.fallback_reasons
        else:
            baseline = task_baseline
            advantage = task_advantage
            baseline_kind = "task"
            fallback_reasons = (StateFallbackReason.NOT_ATTEMPT1_PLAY,)

        assignments[record.record_id] = RecordBaselineAssignment(
            record_id=record.record_id,
            trajectory_key=trajectory_key,
            advantage=advantage,
            baseline=baseline,
            baseline_kind=baseline_kind,
            state_key=state_key,
            fallback_reasons=fallback_reasons,
        )

    return BaselineAssignmentResult(groups=group_summaries, records=assignments)


def invalid_penalty_adjustment(
    *,
    is_action_valid: bool,
    hard_truncated: bool,
    coefficient: float,
) -> float:
    """Return the local invalid-output adjustment, never a utility change."""

    coefficient = _finite_float(coefficient, name="coefficient")
    if coefficient < 0:
        raise ValueError("coefficient must be nonnegative")
    if hard_truncated or is_action_valid:
        return 0.0
    return -coefficient


def is_hard_truncated(
    generated_response_length: int,
    max_response_length: int,
) -> bool:
    """Apply the frozen equality-based hard-truncation definition."""

    if max_response_length <= 0:
        raise ValueError("max_response_length must be positive")
    if generated_response_length < 0:
        raise ValueError("generated_response_length must be nonnegative")
    if generated_response_length > max_response_length:
        raise ValueError(
            "generated_response_length cannot exceed max_response_length in "
            "the capped rollout representation"
        )
    return generated_response_length == max_response_length


def soft_overlong_penalty(
    generated_response_length: int,
    max_response_length: int,
    *,
    cache_width: int,
    max_penalty: float,
) -> float:
    """Return the local linear near-cap penalty, including at the cap."""

    is_hard_truncated(
        generated_response_length,
        max_response_length,
    )
    if cache_width <= 0 or cache_width > max_response_length:
        raise ValueError("cache_width must be in [1, max_response_length]")
    max_penalty = _finite_float(max_penalty, name="max_penalty")
    if max_penalty < 0:
        raise ValueError("max_penalty must be nonnegative")

    fraction = (generated_response_length - (max_response_length - cache_width)) / cache_width
    fraction = min(max(fraction, 0.0), 1.0)
    return -max_penalty * fraction


def response_mask_for_overlong_mode(
    response_mask: Sequence[int | bool],
    generated_response_length: int,
    max_response_length: int,
    *,
    overlong_mode: Literal["soft_shaping", "filter"] = "soft_shaping",
) -> tuple[int, ...]:
    """Apply the selected overlong policy to an ordinary response mask.

    The proposed default, ``soft_shaping``, leaves capped responses trainable.
    ``filter`` is an explicit future ablation that zeros a hard-truncated
    response rather than applying the default soft-shaping estimator.
    """

    normalized = tuple(int(value) for value in response_mask)
    if any(value not in {0, 1} for value in normalized):
        raise ValueError("response_mask must be binary")
    hard_truncated = is_hard_truncated(
        generated_response_length,
        max_response_length,
    )
    if overlong_mode == "soft_shaping":
        return normalized
    if overlong_mode == "filter":
        if hard_truncated:
            return (0,) * len(normalized)
        return normalized
    raise ValueError(f"Unknown overlong_mode: {overlong_mode!r}")


TokenLossesByTask: TypeAlias = Mapping[TaskId, Mapping[TrajectoryId, Sequence[float]]]
TokenCountsByTask: TypeAlias = Mapping[TaskId, Mapping[TrajectoryId, int]]
TokenWeightsByTask: TypeAlias = dict[TaskId, dict[TrajectoryId, float]]


def reference_complete_meta_trajectory_loss(
    token_losses: TokenLossesByTask,
) -> float:
    """Token mean per trajectory, sibling mean per task, then task mean.

    Each sequence must contain only final trainable-token losses.  Empty
    sequences are rejected because the selected design has no policy yet for
    ``N_j == 0``.
    """

    if not token_losses:
        raise ValueError("token_losses must contain at least one task")
    task_losses: list[float] = []
    for task_uid, siblings in token_losses.items():
        if not siblings:
            raise ValueError(f"Task {task_uid!r} has no sibling trajectories")
        trajectory_losses: list[float] = []
        for trajectory_uid, losses in siblings.items():
            values = tuple(
                _finite_float(loss, name=f"token loss[{task_uid!r}, {trajectory_uid!r}]")
                for loss in losses
            )
            if not values:
                raise ValueError(
                    f"Trajectory {(task_uid, trajectory_uid)!r} has N_j == 0"
                )
            trajectory_losses.append(math.fsum(values) / len(values))
        task_losses.append(math.fsum(trajectory_losses) / len(trajectory_losses))
    return math.fsum(task_losses) / len(task_losses)


def reference_token_weights(
    trainable_token_counts: TokenCountsByTask,
) -> TokenWeightsByTask:
    """Return per-token ``1 / (num_tasks * K_g * N_j)`` coefficients."""

    if not trainable_token_counts:
        raise ValueError("trainable_token_counts must contain at least one task")
    num_tasks = len(trainable_token_counts)
    weights: TokenWeightsByTask = {}
    for task_uid, siblings in trainable_token_counts.items():
        if not siblings:
            raise ValueError(f"Task {task_uid!r} has no sibling trajectories")
        sibling_count = len(siblings)
        task_weights: dict[TrajectoryId, float] = {}
        for trajectory_uid, token_count in siblings.items():
            if isinstance(token_count, bool) or not isinstance(token_count, int):
                raise ValueError("trainable token counts must be integers")
            if token_count <= 0:
                raise ValueError(
                    f"Trajectory {(task_uid, trajectory_uid)!r} has N_j <= 0"
                )
            task_weights[trajectory_uid] = 1.0 / (
                num_tasks * sibling_count * token_count
            )
        weights[task_uid] = task_weights
    return weights


def reference_weighted_token_sum(
    token_losses: TokenLossesByTask,
    token_weights: Mapping[TaskId, Mapping[TrajectoryId, float]],
) -> float:
    """Evaluate the token sum induced by reference trajectory coefficients."""

    if set(token_losses) != set(token_weights):
        raise ValueError("token loss and weight task keys must match exactly")
    terms: list[float] = []
    for task_uid, siblings in token_losses.items():
        if set(siblings) != set(token_weights[task_uid]):
            raise ValueError(
                f"Token loss and weight trajectories differ for task {task_uid!r}"
            )
        for trajectory_uid, losses in siblings.items():
            weight = _finite_float(
                token_weights[task_uid][trajectory_uid],
                name=f"token weight[{task_uid!r}, {trajectory_uid!r}]",
            )
            if weight <= 0:
                raise ValueError("token weights must be positive")
            for loss in losses:
                terms.append(
                    weight
                    * _finite_float(
                        loss,
                        name=f"token loss[{task_uid!r}, {trajectory_uid!r}]",
                    )
                )
    return math.fsum(terms)
