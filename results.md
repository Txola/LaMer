# ALFWorld experimental results

Last updated: 2026-09-30

This file collects the ALFWorld results discussed during development. Values are
taken from saved run artifacts whenever those artifacts are available. Results
recovered only from the conversation are explicitly marked as such.

## Conventions and evaluation sets

- `P@1`, `P@2`, and `P@3` are cumulative task success after one, two, or three
  attempts. They are not three independent attempt-success rates.
- `balanced126` contains 126 tasks: 84 in-distribution (ID) tasks from the four
  training task types and 42 out-of-distribution (OOD) tasks from the two held-out
  task types.
- Each category in `balanced126` contains 21 tasks.
- The training validation set contains 84 fixed, task-balanced ID games.
- Base model: `Qwen/Qwen3-4B`, with thinking disabled.
- Trained model: full-parameter checkpoint at training step 150 unless stated
  otherwise.
- Sampling is stochastic. Results from separately launched runs are not paired,
  even when their configuration seeds match.

## Presentation evaluation matrix: balanced126

These are the main final results. Every cell gives `successful tasks / 126
(percentage)`.

| Model | Protocol | Reflection | P@1 | P@2 | P@3 | Gain P@3 - P@1 | Time |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Base Qwen3-4B | 3x10 | No | 12/126 (9.52%) | 14/126 (11.11%) | 17/126 (13.49%) | +5 (+3.97 pp) | 5m 02s |
| Base Qwen3-4B | 3x10 | Yes | 12/126 (9.52%) | 31/126 (24.60%) | 37/126 (29.37%) | +25 (+19.84 pp) | 5m 42s |
| Base Qwen3-4B | 3x30 | No | 45/126 (35.71%) | 52/126 (41.27%) | 55/126 (43.65%) | +10 (+7.94 pp) | 13m 21s |
| Base Qwen3-4B | 3x30 | Yes | 37/126 (29.37%) | 60/126 (47.62%) | 72/126 (57.14%) | +35 (+27.78 pp) | 14m 19s |
| Checkpoint 150 | 3x10 | No | 77/126 (61.11%) | 84/126 (66.67%) | 85/126 (67.46%) | +8 (+6.35 pp) | 7m 26s |
| Checkpoint 150 | 3x10 | Yes | 77/126 (61.11%) | 88/126 (69.84%) | 91/126 (72.22%) | +14 (+11.11 pp) | 7m 46s |
| Checkpoint 150 | 3x30 | No | 92/126 (73.02%) | 104/126 (82.54%) | 107/126 (84.92%) | +15 (+11.90 pp) | 19m 53s |
| Checkpoint 150 | 3x30 | Yes | 95/126 (75.40%) | 106/126 (84.13%) | 113/126 (89.68%) | +18 (+14.29 pp) | 19m 32s |

### Important pass-one caveat

Reflection is generated only after a failed first attempt, so enabling reflection
cannot causally change P@1. The 3x30 base runs nevertheless produced 37/126 and
45/126 at P@1, while the checkpoint runs produced 95/126 and 92/126. These are
separate stochastic vLLM executions, not paired evaluations. Batch scheduling and
engine-level sampling can change generated trajectories despite matching
configuration seeds. For reflection comparisons, the most defensible values are
the within-run retry gains (`P@3 - P@1`); do not present the cross-run P@1
difference as a reflection effect.

### ID and OOD breakdown

| Model | Protocol | Reflection | ID P@1 | ID P@2 | ID P@3 | OOD P@1 | OOD P@2 | OOD P@3 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 3x10 | No | 12/84 (14.29%) | 14/84 (16.67%) | 17/84 (20.24%) | 0/42 (0.00%) | 0/42 (0.00%) | 0/42 (0.00%) |
| Base | 3x10 | Yes | 12/84 (14.29%) | 28/84 (33.33%) | 34/84 (40.48%) | 0/42 (0.00%) | 3/42 (7.14%) | 3/42 (7.14%) |
| Base | 3x30 | No | 41/84 (48.81%) | 46/84 (54.76%) | 49/84 (58.33%) | 4/42 (9.52%) | 6/42 (14.29%) | 6/42 (14.29%) |
| Base | 3x30 | Yes | 36/84 (42.86%) | 49/84 (58.33%) | 58/84 (69.05%) | 1/42 (2.38%) | 11/42 (26.19%) | 14/42 (33.33%) |
| Checkpoint 150 | 3x10 | No | 69/84 (82.14%) | 75/84 (89.29%) | 75/84 (89.29%) | 8/42 (19.05%) | 9/42 (21.43%) | 10/42 (23.81%) |
| Checkpoint 150 | 3x10 | Yes | 69/84 (82.14%) | 75/84 (89.29%) | 76/84 (90.48%) | 8/42 (19.05%) | 13/42 (30.95%) | 15/42 (35.71%) |
| Checkpoint 150 | 3x30 | No | 81/84 (96.43%) | 83/84 (98.81%) | 83/84 (98.81%) | 11/42 (26.19%) | 21/42 (50.00%) | 24/42 (57.14%) |
| Checkpoint 150 | 3x30 | Yes | 79/84 (94.05%) | 81/84 (96.43%) | 82/84 (97.62%) | 16/42 (38.10%) | 25/42 (59.52%) | 31/42 (73.81%) |

### Category breakdown: 3x30

Each cell gives cumulative `P@1 / P@2 / P@3` successes out of 21 tasks.

| Category | Base, no reflection | Base, reflection | Checkpoint 150, no reflection | Checkpoint 150, reflection |
| --- | ---: | ---: | ---: | ---: |
| Pick and place | 17 / 18 / 18 | 18 / 20 / 21 | 21 / 21 / 21 | 21 / 21 / 21 |
| Look at object in light | 13 / 16 / 16 | 10 / 11 / 12 | 21 / 21 / 21 | 21 / 21 / 21 |
| Clean then place | 6 / 7 / 8 | 6 / 11 / 13 | 19 / 21 / 21 | 18 / 19 / 19 |
| Pick two objects and place | 2 / 3 / 3 | 1 / 5 / 6 | 1 / 5 / 7 | 5 / 9 / 13 |
| Cool then place | 2 / 3 / 3 | 0 / 6 / 8 | 10 / 16 / 17 | 11 / 16 / 18 |
| Heat then place | 5 / 5 / 7 | 2 / 7 / 12 | 20 / 20 / 20 | 19 / 20 / 21 |

### Category breakdown: 3x10

Each cell gives cumulative `P@1 / P@2 / P@3` successes out of 21 tasks.

| Category | Base, no reflection | Base, reflection | Checkpoint 150, no reflection | Checkpoint 150, reflection |
| --- | ---: | ---: | ---: | ---: |
| Pick and place | 7 / 8 / 9 | 7 / 18 / 18 | 21 / 21 / 21 | 21 / 21 / 21 |
| Look at object in light | 4 / 5 / 6 | 4 / 5 / 8 | 19 / 20 / 20 | 19 / 19 / 20 |
| Clean then place | 1 / 1 / 2 | 1 / 2 / 4 | 13 / 17 / 17 | 13 / 17 / 17 |
| Pick two objects and place | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 2 / 3 |
| Cool then place | 0 / 0 / 0 | 0 / 3 / 3 | 8 / 9 / 10 | 8 / 11 / 12 |
| Heat then place | 0 / 0 / 0 | 0 / 3 / 4 | 16 / 17 / 17 | 16 / 18 / 18 |

Source directories: `outputs/presentation/*balanced126`.

## Full-parameter training

### Final corrected run

Run: `alfworld_full_qwen3_4b_seedfix_mb4_v2`

- 150 optimization steps; full-parameter Qwen3-4B training.
- Eight tasks per step and eight rollouts per task (64 trajectories).
- Actor microbatch 4, learning rate `1e-6`, three attempts of ten turns,
  reflection enabled.
- Corrected independent vLLM rollout sampling.
- Validation every five steps on the fixed 84-task ID set.
- Runtime: approximately 21h 16m wall-clock (20:18 on September 28 to 17:35
  on September 29); summed logged step time was 21.17h.
- Peak reported GPU allocation: 108.89 GiB; peak reservation: 124.48 GiB.
- Retained checkpoints: best step 85 (47 GiB) and final step 150 (55 GiB).

The best internal validation checkpoint was step 85 at P@3 = 79/84 (94.05%).
The final checkpoint reached 75/84 (89.29%). Step 150 was used for the final
balanced126 presentation matrix because it is the prespecified final model;
selecting step 85 using validation would be valid only if reported explicitly as
validation-selected.

| Step | P@1 | P@2 | P@3 |
| ---: | ---: | ---: | ---: |
| 0 | 14/84 (16.67%) | 32/84 (38.10%) | 40/84 (47.62%) |
| 5 | 17/84 (20.24%) | 34/84 (40.48%) | 39/84 (46.43%) |
| 10 | 15/84 (17.86%) | 30/84 (35.71%) | 43/84 (51.19%) |
| 15 | 26/84 (30.95%) | 36/84 (42.86%) | 46/84 (54.76%) |
| 20 | 28/84 (33.33%) | 41/84 (48.81%) | 50/84 (59.52%) |
| 25 | 39/84 (46.43%) | 48/84 (57.14%) | 54/84 (64.29%) |
| 30 | 39/84 (46.43%) | 57/84 (67.86%) | 67/84 (79.76%) |
| 35 | 45/84 (53.57%) | 63/84 (75.00%) | 70/84 (83.33%) |
| 40 | 43/84 (51.19%) | 60/84 (71.43%) | 66/84 (78.57%) |
| 45 | 44/84 (52.38%) | 57/84 (67.86%) | 65/84 (77.38%) |
| 50 | 46/84 (54.76%) | 56/84 (66.67%) | 61/84 (72.62%) |
| 55 | 55/84 (65.48%) | 65/84 (77.38%) | 68/84 (80.95%) |
| 60 | 57/84 (67.86%) | 66/84 (78.57%) | 71/84 (84.52%) |
| 65 | 56/84 (66.67%) | 67/84 (79.76%) | 73/84 (86.90%) |
| 70 | 59/84 (70.24%) | 69/84 (82.14%) | 72/84 (85.71%) |
| 75 | 63/84 (75.00%) | 70/84 (83.33%) | 75/84 (89.29%) |
| 80 | 63/84 (75.00%) | 72/84 (85.71%) | 76/84 (90.48%) |
| **85** | **65/84 (77.38%)** | **75/84 (89.29%)** | **79/84 (94.05%)** |
| 90 | 61/84 (72.62%) | 73/84 (86.90%) | 75/84 (89.29%) |
| 95 | 61/84 (72.62%) | 72/84 (85.71%) | 74/84 (88.10%) |
| 100 | 65/84 (77.38%) | 73/84 (86.90%) | 75/84 (89.29%) |
| 105 | 64/84 (76.19%) | 72/84 (85.71%) | 74/84 (88.10%) |
| 110 | 64/84 (76.19%) | 74/84 (88.10%) | 77/84 (91.67%) |
| 115 | 59/84 (70.24%) | 74/84 (88.10%) | 75/84 (89.29%) |
| 120 | 59/84 (70.24%) | 69/84 (82.14%) | 73/84 (86.90%) |
| 125 | 59/84 (70.24%) | 67/84 (79.76%) | 68/84 (80.95%) |
| 130 | 66/84 (78.57%) | 72/84 (85.71%) | 74/84 (88.10%) |
| 135 | 60/84 (71.43%) | 67/84 (79.76%) | 70/84 (83.33%) |
| 140 | 65/84 (77.38%) | 72/84 (85.71%) | 73/84 (86.90%) |
| 145 | 64/84 (76.19%) | 71/84 (84.52%) | 74/84 (88.10%) |
| **150** | **65/84 (77.38%)** | **72/84 (85.71%)** | **75/84 (89.29%)** |

Source: `outputs/alfworld_full_multi_gpu/alfworld_full_qwen3_4b_seedfix_mb4_v2`.

### Corrected-run rollout diversity

The independent-sampling fix produced distinct first responses for nearly every
rollout throughout training. As the policy improved, more task groups became
all-success and action trajectories naturally became less diverse.

| Steps | Mixed outcomes | All failed | All succeeded | Successful rollouts/group | Unique action sequences | Fully collapsed groups | Unique first responses | Nonzero-advantage tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-50 | 46.0% | 14.3% | 39.8% | 5.26/8 | 82.5% | 3.5% | 99.8% | 86.8% |
| 51-100 | 21.0% | 5.8% | 73.3% | 6.85/8 | 60.4% | 16.8% | 99.9% | 90.1% |
| 101-150 | 15.5% | 4.3% | 80.3% | 7.17/8 | 49.9% | 26.5% | 98.3% | 88.1% |
| Overall | 27.5% | 8.1% | 64.4% | 6.43/8 | 64.3% | 15.6% | 99.3% | 88.3% |

### Superseded pre-seed-fix run

The first full run was stopped at step 20 after exposing the rollout-sampling
problem. Its results are retained for diagnosis and must not be mixed with the
corrected run.

| Step | P@1 | P@2 | P@3 |
| ---: | ---: | ---: | ---: |
| 5 | 22/84 (26.19%) | 41/84 (48.81%) | 49/84 (58.33%) |
| 10 | 22/84 (26.19%) | 35/84 (41.67%) | 39/84 (46.43%) |
| 15 | 26/84 (30.95%) | 36/84 (42.86%) | 41/84 (48.81%) |
| 20 | 21/84 (25.00%) | 32/84 (38.10%) | 36/84 (42.86%) |

Source: `outputs/alfworld_full_multi_gpu/alfworld_full_qwen3_4b_main`.

### Earlier LoRA comparison (conversation-recovered)

These values came from the earlier 84-task LoRA run and are not backed by a
local output directory on this machine.

| Step | P@3 |
| ---: | ---: |
| 10 | 35.7% |
| 20 | 36.9% |
| 30 | 34.5% |
| 40 | 33.3% |
| 50 | 35.7% |
| 60 | 42.9% |
| 70 | 36.9% |
| 80 | 33.3% |
| 90 | 32.1% |
| 100 | 42.9% |
| 110 | 44.0% |

At step 110: P@1 = 14.3%, P@2 = 35.7%, P@3 = 44.0%, and P@3 - P@1 =
29.8 percentage points.

The earlier base-model reference on the same 84 fixed ID games was P@1 = 12/84
(14.29%), P@2 = 23/84 (27.38%), and P@3 = 32/84 (38.10%). This was a separate
stochastic evaluation from the later `balanced126` runs.

The previously extracted LoRA rollout composition was:

| Steps | Mixed-outcome groups | All failed | All succeeded | Nonzero-advantage tokens | Rollout success |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1-50 | 17.0% (1.36/8) | 40.5% | 42.5% | 50.6% | 49.9% |
| 51-100 | 25.8% (2.06/8) | 34.3% | 40.0% | 66.3% | 51.1% |
| 101-150 | 30.5% (2.44/8) | 29.8% | 39.8% | 74.6% | 55.0% |
| Overall | 24.4% (1.95/8) | 34.8% | 40.8% | 63.8% | 52.0% |

## Checkpoint-5 balanced126 evaluation

The early full-model checkpoint was merged and evaluated with 3x10 reflection.

| Split | P@1 | P@2 | P@3 |
| --- | ---: | ---: | ---: |
| All 126 | 28/126 (22.22%) | 43/126 (34.13%) | 50/126 (39.68%) |
| ID 84 | 23/84 (27.38%) | 36/84 (42.86%) | 42/84 (50.00%) |
| OOD 42 | 5/42 (11.90%) | 7/42 (16.67%) | 8/42 (19.05%) |

Runtime: 6m 53s. Source: `outputs/alfworld_step5_qwen_3x10_balanced126`.

## Reproducible base 3x10 reference

The repeated new-machine base evaluation produced identical aggregate values in
two runs:

| Split | P@1 | P@2 | P@3 |
| --- | ---: | ---: | ---: |
| All 126 | 11/126 (8.73%) | 28/126 (22.22%) | 37/126 (29.37%) |
| ID 84 | 11/84 (13.10%) | 26/84 (30.95%) | 32/84 (38.10%) |
| OOD 42 | 0/42 (0.00%) | 2/42 (4.76%) | 5/42 (11.90%) |

Runtimes were 6m 10s and 5m 37s. Sources:
`outputs/base_qwen_3x10_balanced126_blackwell` and
`outputs/base_qwen_3x10_balanced126_blackwell2`.

## Failed-peer assistance experiments

All three experiments use the frozen merged step-5 checkpoint. Attempt one was
run eight times per each of 84 ID tasks, producing 672 rollouts and 520 failures.
The common k=4-eligible analysis set has 492 failed recipients from 67 tasks.
Recipient retries use matched seeds within each experiment. Bootstrap confidence
intervals are task-level macro intervals with 2,000 resamples.

The self-only result is 98/492 in the assisted and merged runs, but 108/492 on the
same subset in the earlier appended-reflection run. The prompts and explicit
retry seeds matched; the different active batch composition (520 versus 492)
changed vLLM sampling. Therefore compare conditions only to the paired baseline
from the same run.

### Directly appended peer reflections

Peer reflections are appended verbatim to the recipient's normal retry context.

| Condition | Evaluated | Success | Paired pooled gain vs self | Task-macro gain (95% CI) | Mean total input tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Self only (k=0) | 520 | 126/520 (24.23%) | baseline | baseline | 7,921 |
| Same-task failed peers, k=1 | 517 | 131/517 (25.34%) | +1.35 pp | +1.81 pp (-2.18, +5.65) | 9,825 |
| Same-task failed peers, k=2 | 509 | 130/509 (25.54%) | +2.36 pp | +2.69 pp (-1.47, +6.97) | 11,379 |
| Same-task failed peers, k=4 | 492 | 140/492 (28.46%) | +6.50 pp | +6.93 pp (+2.07, +12.01) | 14,526 |
| Unrelated failed peers, k=2 | 520 | 122/520 (23.46%) | -0.77 pp | -0.58 pp (-3.97, +2.89) | 11,395 |

Runtime: 50m 10s. Source:
`outputs/alfworld_failed_peer_reflections_step5_id84_train_sampling_v2`.

### Assisted reflection from full histories

One new reflection is generated from the recipient's full failed history plus k
failed same-task histories. Only that single generated reflection is supplied to
the normal retry prompt.

| k | Success | Pooled gain vs k=0 | Rescued / harmed | Task-macro gain (95% CI) | Reflection input / output tokens | Retry input tokens | Mean total tokens |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 98/492 (19.92%) | baseline | 0 / 0 | baseline | 772 / 636 | 8,093 | 11,511 |
| 1 | 102/492 (20.73%) | +0.81 pp | 54 / 50 | +0.21 pp (-4.21, +4.57) | 1,192 / 628 | 8,162 | 12,010 |
| 2 | 117/492 (23.78%) | +3.86 pp | 57 / 38 | +3.50 pp (-0.93, +7.85) | 1,578 / 621 | 7,959 | 12,126 |
| 4 | 127/492 (25.81%) | +5.89 pp | 63 / 34 | +5.74 pp (+0.40, +11.30) | 2,348 / 618 | 7,814 | 12,784 |

The k=4 assisted condition is the strongest peer result: a positive task-level
bootstrap interval and 29 net rescues, with 11.1% more total tokens than k=0.

Runtime: 42m 57s. Source:
`outputs/alfworld_assisted_reflections_step5_id84_train_sampling`.

### Merged own and peer reflections

The recipient's normal reflection and four peer reflections are consolidated by
one extra LLM merge pass. The retry receives only the merged reflection.

| Condition | Success | Pooled gain vs self | Rescued / harmed | Task-macro gain (95% CI) | Reflection tokens in/out | Merge tokens in/out | Retry input tokens | Mean total tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Self only | 98/492 (19.92%) | baseline | 0 / 0 | baseline | 772 / 636 | 0 / 0 | 8,093 | 11,511 |
| Same-task merge, k=4 | 117/492 (23.78%) | +3.86 pp | 49 / 30 | +4.03 pp (-0.08, +8.15) | 772 / 636 | 980 / 341 | 9,646 | 14,226 |
| Unrelated-peer merge, k=4 | 95/492 (19.31%) | -0.61 pp | 18 / 21 | -0.58 pp (-3.17, +2.11) | 772 / 636 | 984 / 298 | 9,457 | 14,088 |

Same-task merging beat the unrelated control by +4.47 pooled percentage points
and +4.61 task-macro points (95% CI +0.39 to +8.91). It was nevertheless weaker
and more expensive than assisted k=4 (25.81%, 12,784 tokens).

Runtime: 32m 26s. Source:
`outputs/alfworld_merged_reflections_step5_id84_train_sampling`.

## Artifact index

- Main evaluation summaries:
  `outputs/presentation/*/validation_diagnostics/step_000000_summary.json`
- Corrected training validation curve:
  `outputs/alfworld_full_multi_gpu/alfworld_full_qwen3_4b_seedfix_mb4_v2/validation_diagnostics`
- Training configuration and logs:
  `outputs/alfworld_full_multi_gpu/alfworld_full_qwen3_4b_seedfix_mb4_v2`
- Peer experiment summaries: each peer output directory contains `summary.json`,
  `condition_summary.csv`, `task_summary.csv`, and `contrast_summary.csv`.
