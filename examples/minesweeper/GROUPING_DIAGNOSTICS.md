# Capture GiGPO groups on one GPU

From the repository root, with the `lamer` environment active:

```bash
conda activate lamer
bash examples/minesweeper/capture_gigpo_groups_1gpu.sh
```

The launcher prepares four task rows and runs one training iteration with eight
rollout trajectories per task, three attempts, and up to seven turns per attempt.
It uses Qwen/Qwen3-4B and the current single-GPU memory settings. Rewards,
advantages, grouping, and training sampling follow the normal training path.

The existing `trainer.critic_warmup=2` gate skips the actor update at step 1.
GiGPO has no critic here. The run computes advantages and writes diagnostics but
does not execute backward or update model weights. Validation, checkpoint saving,
and checkpoint resumption are disabled. These are initial-model training groups,
not groups from a model that has already learned through LaMer.

Each run creates a directory under `diagnostics/gigpo_groups_capture/` containing:

- `step_000001_groups.jsonl`: non-singleton play groups, including visible boards,
  previous reflections, parsed coordinates, parsing validity, environment
  effectiveness flags, immediate rewards, post-action boards, step returns,
  and interaction IDs.
- `step_000001_play_records.jsonl`: all play optimization records, including
  singleton groups and batching copies, independent of the detail filter.
- `step_000001_summary.json`: group counts and cross-attempt statistics.
- `run.log` and `prepare.log`: trainer and data-preparation output.
- `data/`: this run's prepared dataset, separate from other launchers' datasets.
- `hydra/`: Hydra configuration artifacts.

Reflection groups are excluded from play-state statistics. Optimization records
duplicated by `adjust_batch()` remain in the output with their original interaction
IDs. Detailed output includes both same-attempt and cross-attempt groups.

Open `analysis/inspect_gigpo_groups.ipynb` and set its groups/summary paths to a run.
It discovers the complete play-record companion from the summary, retains the
optimization view, and separately compares deduplicated structural groupings:
task/board, task/board/attempt, and task/board/attempt/trajectory. The last grouping
isolates repeated visits and is diagnostic only. Older captures load as an
explicitly incomplete subset; their missing action metadata cannot be recovered
from the old group files.

Parsing validity is not a bounds or environment-validity check. Effectiveness is
the environment's existing flag, which can compare against a stale previous-board
buffer on early returns. The notebook therefore uses actual before/after visible
boards to identify unchanged-board transitions and reports effectiveness separately.
No category is inferred from reward values alone.

To choose a directory or increase the number of tasks:

```bash
OUTPUT_DIR=diagnostics/my_capture TASK_COUNT=8 \
  bash examples/minesweeper/capture_gigpo_groups_1gpu.sh
```

Choose a fresh output directory for each capture; reusing one overwrites its files.
`MODEL_PATH` can point to a compatible Hugging Face model directory. The default
is `Qwen/Qwen3-4B`. Keep dataset size equal to training batch size and retain the
one-step limit and warmup gate when editing this capture launcher.

Inspect a run with:

```bash
cat diagnostics/my_capture/step_000001_summary.json
python - <<'PY'
import json
from pathlib import Path

path = Path("diagnostics/my_capture/step_000001_groups.jsonl")
for line in path.open():
    group = json.loads(line)
    if len(group["attempt_counts"]) > 1:
        print(json.dumps(group, indent=2, ensure_ascii=False))
        break
else:
    print("No cross-attempt play group was found in this batch.")
PY
```
