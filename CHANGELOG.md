# Changelog

- Added an audited Qwen3-4B profile to the modern VERL ALFWorld launcher so the
  known faster model can validate the new GiGPO integration before returning
  to Qwen3.5-9B. The shared FP32-LoRA hook now checks either architecture's
  exact language projections, while model-specific presence-penalty and loss-
  aggregation defaults preserve the intended comparison.
- Split training format diagnostics into play and reflection parse-valid,
  response-limit, and environment-effectiveness metrics. This exposes whether
  Qwen3.5's aggregate invalid-action ratio is caused by truncated reasoning or
  ineffective commands without changing its rewards or policy inputs.
- Kept activation checkpointing enabled in the modern Qwen3-4B profile after
  paired one-step smokes showed that disabling it saved about seven seconds in
  the actor update but increased peak sampled GPU use from 71.2 to 85.7 GB.
  Source inspection localized PyTorch's no-gradient-input warning to the
  expected forward-only old-policy entropy calculation, not to checkpointed
  transformer blocks in the actor update.
- Matched the modern Qwen3-4B full-training profile to the successful legacy
  run's `0.001` entropy coefficient, FP32 resident actor with BF16 mixed
  precision, and `0.35` vLLM reservation. Enabled 150-update full-model runs
  with validation-best and normally-completed-final checkpoint retention.

This changelog records the ALFWorld code changes made during the current work and
why they were needed. Commit hashes point to the repository history. The
`Unreleased` section describes changes made after the latest dated entry.

## Unreleased

### Added

- Exposed the shared modern-training prompt, response, and model-context limits,
  plus the optional brief play/reflection instruction. This lets Qwen3.5 train
  and validate under the selected brief 1,536-token generation protocol instead
  of silently retaining the original 1,024-token cap.
- Added a reusable, atomic merger for modern single-GPU FSDP LoRA checkpoints.
  Base-model configuration and tokenizer metadata are now copied byte-for-byte
  so a newer merge environment cannot make the result unreadable by the pinned
  evaluation environment.
  It reconstructs the recorded frozen base revision, targets only the exact
  modules named by the adapter, verifies every tensor before merging, and
  selects the correct text or image-text Transformers class for Qwen3 and
  Qwen3.5. This produces an ordinary BF16 Hugging Face model for evaluation
  without changing the established inference protocol.
- Added an optional, default-off future-aware episode-credit variant to the
  modern ALFWorld GiGPO path. It derives each attempt score from the existing
  backward return at the attempt's first play record, preserving exact step and
  attempt-boundary discounting while leaving original LaMer credit as default.
- Added optional task-balanced, incrementally written Markdown trajectory
  traces to the Qwen3.5 ALFWorld evaluator. They expose prompts, raw model
  responses, parsed actions, environment transitions, and outcomes without
  changing the rollout protocol.
- Increased Qwen3.5 evaluation concurrency to 16 task workers on the 20-thread
  RTX PRO 6000 host and disabled unused rollout log-probability collection for
  validation. Training log probabilities and conservative vLLM memory and
  execution settings remain unchanged.
- Made the Qwen3.5 evaluation prompt, response, and combined vLLM context limits
  explicit launcher parameters. The 1,024-token comparison default is
  unchanged, while bounded longer-response diagnostics can now be labelled and
  reproduced without editing the launcher.
- Enabled vLLM prefix caching in the Qwen3.5 training launcher after validating
  it in the full prompting baseline. Same-task rollouts share long prefixes,
  and pinned VERL explicitly clears the cache after each policy-weight update;
  evaluation-only log-probability and memory shortcuts were not transferred.
- Added `results.md` to keep presentation results, training curves, runtimes,
  confidence intervals, token costs, and artifact locations in one auditable
  place.
- Added an isolated, revision-pinned Qwen3.5-9B compatibility environment and a
  modern VERL validation launcher for the fixed balanced126 ALFWorld protocol.
  The integration keeps the full multimodal model path for future LoRA
  transferability, requires chunked prefill for the hybrid recurrent model,
  and records complete run metadata.
- Added a modern ALFWorld agent loop and explicit local evaluation dataset. The
  loop preserves LaMer's fresh per-action prompts, reflections, restarts,
  seeded game identity, action projection, and cumulative per-category metrics
  without nesting extra Ray environment actors.
- Added `docs/QWEN35_ALFWORLD.md` with the compatibility rationale, staged run
  commands, verified behavior, performance tradeoffs, and the remaining GiGPO
  work required before Qwen3.5 LoRA training is valid.
- Added the modern VERL GiGPO training adapter. It retains every ALFWorld action
  and reflection as a policy sample, restores LaMer's per-attempt and
  cross-attempt returns, and matches the original GiGPO advantages numerically
  without modifying the pinned VERL checkout.
- Added an auditable Qwen3.5 training schedule that reproduces the legacy
  launcher's independently shuffled task slots, plus a conservative one-update
  full-language smoke launcher and a 20-step learning-pilot mode.
- Added a Qwen3.5-9B language-only LoRA GiGPO launcher with an audited 248-module
  target, rank-16/alpha-32 adapters, BF16 frozen weights and compute, FP32
  adapter/gradient/Adam state, adapter-only vLLM refreshes and checkpoints, and
  smoke, pilot, and full schedules. Runtime assertions stop the run if the
  pinned architecture or mixed-precision assumptions change.
- Added validation-aware LoRA checkpoint selection. Step-zero and every-fifth-
  step validation use cumulative third-attempt balanced84 success; only a
  strict validation best and the completed run's final checkpoint are retained,
  with the decision recorded in a JSON manifest.

### Changed

- Reduced the modern Qwen3-4B actor microbatch from four to two after a
  variable-length step exhausted the 95-GiB GPU during backward despite
  fitting for ten earlier updates. This changes only gradient-accumulation
  chunking, not the effective PPO batch or objective. Added checkpoint resume
  controls whose attempt logs and provenance are written separately, preserve
  the original run artifacts, reuse the original datasets, and skip redundant
  same-step validation by default.
- Stabilized Qwen3.5 LoRA GiGPO after the first long run collapsed between
  steps 14 and 22. The launcher now uses `1e-6` learning rate, `0.01` entropy
  coefficient, and per-generation `seq-mean-token-mean` loss aggregation so
  long reasoning responses cannot dominate short actions solely by token
  count.
- Corrected the presence-penalized Qwen3.5 behavior-policy mismatch with
  VERL's bounded token-level importance sampling. Training explicitly records
  vLLM post-processor log-probabilities and clips `pi_actor / pi_rollout` at
  `2.0`; launcher guards reject raw log-probabilities or disabled correction.
- Restored exact invalid-action penalty parity with the successful Qwen3-4B
  GiGPO path by penalizing both episode scores and discounted step returns
  before group centering. The modern port had previously penalized only the
  episode component, weakening the state-level signal against malformed
  actions.
- Run Qwen3.5 ALFWorld text tokenization synchronously through VERL's normal
  processor-backed continuous-token builder. This narrowly avoids a reused
  `asyncio` executor wake-up failure observed on this machine without changing
  prompt text, token IDs, model loading, sampling, or the existing Qwen3-4B
  launchers.
- Set the required per-GPU rollout log-probability microbatch size in the
  Qwen3.5 evaluator. Modern VERL validates this field even in validation-only
  mode; the setting affects batching only and does not change generation.
- Apply Qwen3.5's recommended `1.5` presence penalty to ALFWorld evaluation and
  training requests after the unchanged prompt produced recursive deliberation
  up to the response limit. The reasoning-before-action prompt is preserved,
  and `PRESENCE_PENALTY=0.0` remains available as the control condition.
- Restore the original ALFWorld GiGPO entropy coefficient and global discount
  in the Qwen3.5 training launcher, record separate training and validation
  sampling tuples, and make the one-update smoke validate again after its
  update. Use optimizer-only CPU offload so the 60-GiB host does not have to
  hold parameters, gradients, and optimizer state together while the 96-GiB
  GPU is available for the update phase. Checkpoint the large-vocabulary
  entropy calculation to exchange recomputation for lower activation memory,
  and send smoke-run diagnostics to W&B as well as the console.
- Make the modern GiGPO adapter honor VERL's synthetic batch-padding contract.
  Padding rows remain present with zero rewards, returns, and advantages for
  distributed mini-batch divisibility, but are excluded from task/state group
  centering and GiGPO metrics so they cannot alter any real training sample.
- Propagate the TransferQueue batch descriptor after writing GiGPO advantages,
  so the actor fetches the newly added fields, and expose a default-on
  `VAL_BEFORE_TRAIN` switch for faster pre-update debugging without changing
  the normal step-zero-validation protocol.
- Preserve rollout-server model-version metadata in every ALFWorld generation.
  This keeps modern VERL's post-update staleness metrics defined without
  guessing versions in the trainer and is forward-compatible with trajectories
  that may span policy versions under asynchronous execution.
- Expose and record Qwen3.5 actor gradient checkpointing independently from
  entropy checkpointing. This supports one-variable-at-a-time throughput and
  memory calibration without changing the default training computation.
- Expose and record Qwen3.5 vLLM eager execution so the supported compiled and
  CUDA-graph generation path can be benchmarked with a matched LoRA smoke before
  becoming a training default.
- Record periodic GPU utilization, memory, power, temperature, and clock data
  for Qwen3.5 training runs so combined rollout/update throughput profiles can
  be evaluated from evidence rather than memory capacity alone.
- Reduce Qwen3.5 pilot/full ALFWorld worker concurrency from eight to four after
  an eight-worker run crossed Ray's host-memory threshold. Add a bounded
  replay-buffer wait so a killed fire-and-forget actor fails diagnostically
  instead of leaving one rollout permanently marked as running.
- Reuse a bounded pool of in-process TextWorld environments in the modern
  ALFWorld agent loop. This prevents cumulative registry and allocator growth
  from eventually causing Ray to kill a rollout worker during Qwen3.5
  training. The pool preserves per-trajectory game resets, seeds, histories,
  prompts, and rewards; `ALFWORLD_ENV_POOL_SIZE` exposes the memory/concurrency
  tradeoff in the training launcher.
- Size TransferQueue for this one-node synchronous job with two storage actors
  instead of the distributed default of eight, and stop transporting duplicate
  raw prompt/response strings that the GiGPO update does not consume. This
  recovers host-memory headroom without changing generated or trained tokens.
- Release vLLM's unused pinned host-weight backup after each LoRA sleep/wake
  cycle and report the worker RSS before and after the release. This preserves
  adapter-only synchronization and rollout concurrency while preventing a
  cached BF16 base-model staging buffer from competing with ALFWorld for RAM;
  the launcher retains an explicit switch for matched diagnostics.
- Commit modern VERL W&B records at each completed step and publish readable
  `val/*` plus P@1/P@2/P@3 aliases alongside the unchanged `val-core/*` and
  `val-aux/*` source metrics.
- Document that the memory-feasible Qwen3.5 full-model profile has no FP32
  master weights and is therefore an integration probe rather than a faithful
  replacement for the FP32-master Qwen3-4B training run. Record the resulting
  small-update precision risk and the requirements for a sound LoRA path.

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
