"""Fail-fast replay-buffer polling for long ALFWorld rollouts.

VERL's v1 agent workers publish rollout completion through TransferQueue.  If
Ray kills a fire-and-forget agent actor, its prompt can otherwise remain tagged
``running`` forever.  This sampler preserves VERL's existing async-buffer
selection semantics while putting a generous upper bound on that wait.
"""

from __future__ import annotations

import time

from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync


class AlfWorldReplayBuffer(ReplayBufferAsync):
    """Raise a diagnostic error instead of waiting forever for a dead actor."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rollout_timeout_seconds = int(
            self.sampler_kwargs.get("rollout_timeout_seconds", 1800)
        )
        if self.rollout_timeout_seconds <= 0:
            raise ValueError("rollout_timeout_seconds must be a positive integer")
        self._sample_started_at: float | None = None

    def sample(self, global_steps: int, partition_id: str, batch_size: int):
        self._sample_started_at = time.monotonic()
        try:
            return super().sample(global_steps, partition_id, batch_size)
        finally:
            self._sample_started_at = None

    def _wait_for_next_poll(self, partition_id: str, last_debug_time: float) -> float:
        last_debug_time = super()._wait_for_next_poll(partition_id, last_debug_time)
        if self._sample_started_at is None:
            return last_debug_time

        elapsed = time.monotonic() - self._sample_started_at
        if elapsed <= self.rollout_timeout_seconds:
            return last_debug_time

        self._sync_metadata_from_transfer_queue()
        raise TimeoutError(
            "ALFWorld rollout batch exceeded "
            f"{self.rollout_timeout_seconds}s in partition {partition_id!r}: "
            f"pending={len(self.pending_keys[partition_id])}, "
            f"running={len(self.running_keys[partition_id])}, "
            f"finished={len(self.finished_keys[partition_id])}, "
            f"failure={len(self.failure_keys[partition_id])}. "
            "A Ray agent may have died before publishing terminal status."
        )
