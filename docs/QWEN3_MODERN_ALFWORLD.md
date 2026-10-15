# Qwen3-4B modern-VERL ALFWorld bridge

This experiment uses `Qwen/Qwen3-4B` as a faster diagnostic model for the
modern VERL integration developed for Qwen3.5-9B. The purpose is to separate
trainer/infrastructure errors from Qwen3.5-specific optimization and response
formatting problems.

## What is held constant

Both model profiles use the same:

- modern asynchronous ALFWorld agent loop;
- same-task rollout grouping and independently sampled group members;
- LaMer episode- and state-level GiGPO advantages;
- replay buffer and synchronous one-update-per-batch schedule;
- three attempts, reflection protocol, seeds, validation split, and metrics;
- FSDP2 actor, vLLM rollout, rollout correction, W&B logging, and checkpoint
  selection implementation.

The model-specific settings are explicit rather than inferred:

| Setting | Qwen3-4B | Qwen3.5-9B |
| --- | --- | --- |
| Hugging Face model type | `qwen3` | `qwen3_5` |
| Language layers | 36 | 32 hybrid layers |
| Audited LoRA projections | 252 | 248 |
| LoRA elements per rank | 2,064,384 | 2,704,896 |
| Default presence penalty | 0.0 | 1.5 |
| Default policy-loss aggregation | `token-mean` | `seq-mean-token-mean` |
| Default entropy coefficient | `0.001` | `0.01` |
| Resident actor parameters | FP32 | BF16 |

Qwen3-4B retains token-level loss aggregation and no presence penalty because
those settings are closest to the successful legacy 4B experiment. Applying
the Qwen3.5 stability settings here would prevent the 4B run from serving as a
clean bridge between the legacy and modern trainers. Its full-parameter
optimizer remains on the 96 GiB GPU because it fits there; CPU offload would
increase pressure on the 60 GiB host and make each diagnostic update slower.
Its FP32 resident actor parameters, BF16 mixed-precision forward/backward,
`0.001` entropy coefficient, and `0.35` vLLM reservation reproduce the
successful legacy 4B run's precision and optimization settings. The modern
trainer still differs in implementation: it uses FSDP2, asynchronous rollout,
fixed independent group seeds, and bounded rollout correction.
Activation checkpointing remains enabled for both profiles. A paired 4B smoke
without it reduced actor-update time from 48.7 to 41.7 seconds, but increased
peak sampled GPU use from 71.2 to 85.7 GB. The memory-safe setting is preferred
because the pilot contains more varied and potentially longer sequences. The
PyTorch no-gradient-input warning seen in both smokes comes from checkpointing
the old-policy entropy calculation during its intentional `torch.no_grad()`
forward-only pass; the actor update uses the same entropy function with
gradients enabled.

For LoRA, both profiles keep the frozen base in BF16 while preserving LoRA
parameters, their optimizer-facing gradients, and Adam moments in FP32. The
precision hook checks the exact layer range, projection distribution, adapter
count, and trainable-element count at runtime. It fails instead of silently
training an incomplete adapter.

## Optional future-aware episode credit

The shared modern launcher exposes
`FUTURE_AWARE_EPISODE_CREDIT`, which defaults to `False`. The default retains
original LaMer episode credit exactly: play records receive the reward from
their current attempt and reflection records receive zero episode-level
credit. For attempt rewards `[0, 0, 10]`, the episode scores are therefore
`[0, 0, 10]`.

Setting `FUTURE_AWARE_EPISODE_CREDIT=True` is a separate algorithmic variant.
It sets each attempt's episode score to the existing backward `step_return` at
that attempt's first play record. Consequently `STEP_GAMMA` discounts reward
through actions within an attempt, while `TRAJ_GAMMA` applies only where the
backward recurrence crosses an attempt boundary. All records in one attempt
receive that attempt-entry score. A reflection receives the entry score of the
retry it is generated to guide.

For example, suppose a failed attempt has two zero-reward actions and the next
attempt succeeds for reward 10 on its third action. With `STEP_GAMMA=0.95` and
`TRAJ_GAMMA=0.6`, the successful attempt's entry score is
`0.95^2 * 10 = 9.025`; the preceding attempt's entry score is
`0.95 * 0.6 * 9.025 = 5.14425`. The simplified attempt-only calculation would
instead give `6` and `10`. The existing step returns, grouping, normalization,
and invalid-action penalty are not otherwise changed.

Enable the variant with:

```bash
FUTURE_AWARE_EPISODE_CREDIT=True RUN_MODE=pilot \
bash examples/alfworld/train_qwen3_4b_modern.sh
```

Runs with this option enabled must be reported separately from original LaMer
and default modern-bridge results.

## Experimental order

### 1. Full-parameter smoke

```bash
OUTPUT_DIR=outputs/qwen3_4b_modern/full_gigpo_smoke_bridge \
bash examples/alfworld/train_qwen3_4b_modern.sh
```

This runs one fixed validation task at step zero, one training update with one
task and eight rollouts, and one post-update validation. It verifies model
loading, rollout grouping, GiGPO metadata, advantage computation, actor update,
weight synchronization, validation, and logging. It is not evidence of
learning.

### 2. Full-parameter learning pilot

Run this only after the smoke completes. It intentionally keeps the pilot short
so learning can be inspected before committing to the full run:

```bash
RUN_MODE=pilot \
OUTPUT_DIR=outputs/qwen3_4b_modern/full_gigpo_pilot20_bridge \
bash examples/alfworld/train_qwen3_4b_modern.sh
```

The pilot uses eight tasks and eight rollouts per task for 20 updates, with the
fixed balanced-84 validation protocol every five steps. Its purpose is to test
whether the modern integration reproduces the early learning behavior of the
known working 4B full-parameter trainer.

### 3. Full-parameter 150-update run

```bash
RUN_MODE=full \
OUTPUT_DIR=outputs/qwen3_4b_modern/full_gigpo_150_legacy_hparams \
bash examples/alfworld/train_qwen3_4b_modern.sh
```

This runs 150 optimizer updates, not 150 passes over a fixed dataset. Every
update samples eight training tasks and eight trajectories per task. Validation
uses the fixed balanced-84 set at step zero and every five updates. Because a
full-model checkpoint is large, the selector retains only the validation best
and the latest validated checkpoint. Each validation replaces the prior
non-best latest checkpoint, so an interrupted run can resume from at most four
completed updates behind while the independently selected best remains intact.

### 4. LoRA smoke and pilot

Only proceed if the full-parameter pilot has a healthy reward, valid-action,
group-diversity, entropy, and validation trajectory:

```bash
TRAINING_VARIANT=lora \
OUTPUT_DIR=outputs/qwen3_4b_modern/lora_gigpo_smoke_bridge \
bash examples/alfworld/train_qwen3_4b_modern.sh
```

Then use `TRAINING_VARIANT=lora RUN_MODE=pilot` for the 20-step LoRA pilot.
Comparing full and LoRA training inside the same modern stack isolates adapter
capacity and optimization from the trainer implementation.

## Interpretation boundaries

A successful 4B experiment validates the shared ALFWorld, GiGPO, modern VERL,
FSDP2, and LoRA plumbing. It does not validate Qwen3.5's hybrid attention
targets, 9B memory peak, multimodal wrapper, or tendency to exhaust the
response budget. Those remain separate 9B-specific questions.

The trainer now logs parse validity, response-limit rate, and environment
effectiveness separately for play and reflection records. These are diagnostic
metrics only; they do not change the reward or generated actions.
