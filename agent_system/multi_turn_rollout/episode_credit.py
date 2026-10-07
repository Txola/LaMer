"""Episode-level credit helpers shared by modern agent rollouts."""

from __future__ import annotations

from collections.abc import Sequence


def episode_scores_for_records(
    attempt_rewards: Sequence[float],
    attempt_indices: Sequence[int],
    phases: Sequence[str],
    *,
    future_aware: bool,
    step_returns: Sequence[float] | None = None,
) -> list[float]:
    """Map attempt scores to policy records, including reflection records.

    Original LaMer credit gives play records their current-attempt reward and
    gives reflections zero episode credit. In the future-aware variant, every
    record receives its attempt-entry return: the existing backward step return
    at that attempt's first play record. This derives episode credit from the
    exact ``step_gamma``/``traj_gamma`` recurrence without recomputing or
    simplifying it. A reflection's attempt index denotes the retry that it is
    generated to guide.
    """
    if len(attempt_indices) != len(phases):
        raise ValueError("attempt_indices and phases must have the same length")
    if step_returns is not None and len(step_returns) != len(phases):
        raise ValueError("step_returns and phases must have the same length")

    if future_aware:
        if step_returns is None:
            raise ValueError("step_returns are required for future-aware episode credit")
        attempt_scores: dict[int, float] = {}
        for attempt_index, phase, step_return in zip(
            attempt_indices, phases, step_returns, strict=True
        ):
            if phase == "play" and attempt_index not in attempt_scores:
                attempt_scores[attempt_index] = float(step_return)
    else:
        attempt_scores = {
            attempt_index: float(reward)
            for attempt_index, reward in enumerate(attempt_rewards)
        }

    record_scores = []
    for attempt_index, phase in zip(attempt_indices, phases, strict=True):
        if phase not in {"play", "reflect"}:
            raise ValueError(f"Unknown rollout phase: {phase!r}")
        if phase == "reflect" and not future_aware:
            record_scores.append(0.0)
        else:
            try:
                record_scores.append(attempt_scores[attempt_index])
            except KeyError as error:
                raise ValueError(
                    f"Attempt {attempt_index} has no play record from which to "
                    "derive episode credit"
                ) from error
    return record_scores
