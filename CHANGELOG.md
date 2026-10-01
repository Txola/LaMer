# Changelog

This changelog records the ALFWorld code changes made during the current work and
why they were needed. Commit hashes point to the repository history. The
`Unreleased` section describes changes made after the latest dated entry.

## Unreleased

### Added

- Added `results.md` to keep presentation results, training curves, runtimes,
  confidence intervals, token costs, and artifact locations in one auditable
  place.

## 2026-10-01

- `c07cbe5 add failed peer reflection evaluation`: added a frozen-policy
  evaluator for verbatim appended reflections, assisted reflection from full
  failed histories, and an extra-pass reflection merge. It also added paired
  same-task and unrelated controls, diagnostic-only retry context injection,
  stable ordered task groups, bootstrap comparisons, token accounting, and
  qualitative output so the peer experiments are reproducible without changing
  model weights.

## 2026-09-28

- `b34cedb add fsdp checkpoint merge helper`: added a reusable, validated helper
  that converts sharded FSDP checkpoints into Hugging Face models for inference.
- `06bf6ce stabilize full alfworld training`: reduced the large-machine actor
  microbatch, disabled incompatible expandable CUDA segments for vLLM sleep
  mode, and aligned step-zero validation with the production protocol after the
  earlier OOM and validation mismatch findings.
- `1981f13 add rollout diversity metrics`: added action-sequence, first-action,
  and first-response diversity measurements so collapsed rollout groups can be
  detected directly.
- `fd845be fix independent vllm rollout sampling`: stopped rollouts in the same
  task group from inheriting identical vLLM request seeds. This preserves the
  eight-rollout GiGPO grouping while allowing independent trajectories. The fix
  was applied to both full and LoRA training paths.

## 2026-09-27

- `6667d77 fix live wandb metric logging`: made step metrics flush to W&B during
  training instead of appearing only after the process finished.
- `c9d8bf2 fix large gpu alfworld training profile`: corrected the large-GPU
  memory profile after smoke-test OOMs while retaining a separate 24-GiB option.
- `7a63251 add optimized full parameter alfworld training`: added the dedicated
  full-parameter launcher, hardware profiles, run metadata capture, smoke mode,
  and production safeguards.
- `69d79ac add best and final checkpoint retention`: retained only the best
  validation checkpoint and the final checkpoint to bound storage while still
  validating every five steps.
- `1f5d42a optimize vllm weight synchronization for training`: reduced the cost
  of transferring updated FSDP actor weights into vLLM between rollouts.
- `b2ba361 fix base alfworld evaluation on new hardware`: made evaluation work on
  the larger machine, shortened Ray socket paths, added hardware profiles, and
  improved reproducibility metadata and diagnostics.

## 2026-09-25

- `6e6c364 add learning diagnostics for meta rl training`: added gradient,
  clipping, KL, advantage, group-composition, and timing metrics needed to assess
  training stability.
- `591f0bd add single gpu lora training for alfworld`: added the reproducible
  single-GPU LoRA launcher used as the comparison configuration.
- `4dd0936 fix same task rollout grouping for alfworld`: ensured that each GiGPO
  group contains eight rollouts of one task instead of unrelated tasks.
- `8434078 add balanced id validation for alfworld`: added the fixed 84-game ID
  checkpoint split across the four training task types.
- `6d020e3 add reproducible alfworld baseline evaluation`: added the base-model
  evaluation launcher, fixed environment seeding, and captured run artifacts.
- `2a70092 add detailed rollout diagnostics`: added per-action parsing,
  effectiveness, reflection, token-limit, trajectory, and validation diagnostics.

## 2026-09-24

- `e30706e add reproducible alfworld evaluation splits`: added deterministic
  split preparation and configuration support for fixed ALFWorld comparisons.
