# Working configurations in this fork

This document records the configurations that have completed the intended code
path in this checkout. It does not claim reproduction of the paper's reported
scores. Generated datasets, logs, checkpoints, and diagnostics belong under
`outputs/` or `diagnostics/`; both directories are ignored by Git.

## Released multi-GPU configuration

The authors' Minesweeper LaMer launcher remains at
`examples/minesweeper/lamer_minesweeper_qwen3_4b.sh`. It uses Qwen3-4B, four
GPUs, tensor parallel size two, 16 task rows per training batch, eight rollout
trajectories per task, three attempts, and 300 epochs. This exact configuration
has not been reproduced on the local single-GPU machine.

## Single-GPU LoRA training

Use `examples/minesweeper/train_lora_1gpu_overnight.sh` on the tested 24 GB GPU:

```bash
conda activate lamer
bash examples/minesweeper/train_lora_1gpu_overnight.sh
```

The defaults are:

| Setting | Value |
|---|---:|
| Base model | `Qwen/Qwen3-4B` |
| Base-model storage | BF16 |
| LoRA rank / alpha | 32 / 64 |
| LoRA targets | all linear layers |
| Learning rate | `1e-6` |
| Tasks per outer update | 1 |
| Rollout group size per task | 8 |
| Attempts per task trajectory | 3 |
| PPO minibatch size | 64 interaction records |
| PPO microbatch per GPU | 2 interaction records |
| Training steps | 80 |
| Checkpoint interval | 5 steps |
| Minesweeper backend | local/in-process |

The launcher uses vLLM for rollout generation, FSDP parameter and optimizer
offloading, gradient checkpointing, and LoRA-only checkpoints. A completed local
run reached all 80 requested updates with these settings. That run establishes
pipeline feasibility; its 32-task evaluations were too small and variable to
establish a reliable learning improvement.

The output directory defaults to
`outputs/minesweeper_lora_1gpu/batch_1_group_8_<timestamp>/`. Override settings
with environment variables, for example:

```bash
OUTPUT_DIR=/absolute/path/to/run \
TOTAL_EPOCHS=80 TRAIN_BATCH_SIZE=1 GROUP_SIZE=8 MICRO_BATCH_SIZE=2 \
LEARNING_RATE=1e-6 \
bash examples/minesweeper/train_lora_1gpu_overnight.sh
```

To resume, run the same command with the existing `OUTPUT_DIR`. The launcher
uses `trainer.resume_mode=auto`, reads the checkpoint tracker, restores LoRA
parameters plus Adam, scheduler, and RNG state, and reloads the frozen base model
from `MODEL_PATH`. Keep the model path and LoRA architecture unchanged across a
resume.

For a short capacity check, use
`examples/minesweeper/test_lora_update_1gpu.sh`. `TASK_COUNT`, `TOTAL_STEPS`,
`MICRO_BATCH_SIZE`, `LEARNING_RATE`, `LORA_RANK`, and `LORA_ALPHA` are configurable.

## Checkpoint evaluation

Evaluate a training directory with a fresh process for the baseline and each
selected checkpoint:

```bash
EVAL_TASK_COUNT=32 CHECKPOINT_STEPS=20,40,60,80 \
bash examples/minesweeper/evaluate_lora_checkpoints_1gpu.sh \
  /absolute/path/to/training_run
```

The evaluation launcher loads policy weights without restoring optimizer,
scheduler, training RNG, critic, or dataloader state. It records pass@1,
pass@2, pass@3, reward metrics, parsing/effectiveness statistics, and complete
interaction JSONL files. Results are written under the training run's
`evaluation/` directory and summarized as CSV, JSON, and Markdown.

The current launcher uses one rollout trajectory and three sequential attempts
per task. The rollout seed is set at `actor_rollout_ref.rollout.seed`, which is
the value consumed by vLLM. Increase `EVAL_TASK_COUNT` before making performance
claims; 32 tasks are suitable for debugging but give coarse success-rate steps.

## GiGPO grouping diagnostics

Capture one real optimization batch without changing GiGPO's training grouping:

```bash
bash examples/minesweeper/capture_gigpo_groups_1gpu.sh
```

See `examples/minesweeper/GROUPING_DIAGNOSTICS.md` and
`analysis/inspect_gigpo_groups.ipynb` for the stored schema and analysis. The
launcher deliberately skips the actor update through the warmup gate; it is a
diagnostic capture rather than a training run.

## Configurations that are not working baselines

Full-parameter FP32 Qwen3-4B exceeded 24 GB during backward. Full-parameter BF16
completed backward but exceeded memory when Adam created its first moment state.
The tested FSDP2 CPU-offload attempt exceeded available system RAM.
Diagnostic launchers for these failed paths are intentionally excluded from the
supported configurations.

An earlier batch-size-four LoRA run is also not a valid learning baseline: it
predated the corrected LoRA-to-vLLM synchronization and later produced non-finite
gradient norms. Use the synchronization, LoRA-only checkpoint, and local
Minesweeper backend commits together.
