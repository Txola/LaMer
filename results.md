# ALFWorld experimental results

Last updated: 2026-10-04

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

## Qwen3.5-9B integration smoke

The first end-to-end Qwen3.5-9B evaluation smoke completed on one fixed
balanced126 task using the unchanged 3x10 reflection protocol. This is an
integration diagnostic, not a baseline estimate.

| Tasks | Category | P@1 | P@2 | P@3 | Runtime |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Cool then place | 0/1 | 0/1 | 0/1 | 5m 57s |

The full 9.41B checkpoint loaded without OOM. FSDP reported 17.55 GiB allocated
and 27.32/94.96 GiB total device use immediately after model wrapping. The run
executed all three attempts, but Qwen3.5 repeatedly generated long visible
reasoning under the unchanged LaMer instruction to reason before acting. The
saved final generation reached the exact 1,024-token response cap and contained
neither `<action>` nor `</action>`; truncated text was consequently projected
as an invalid environment action. The full baseline below preserves reasoning
before action and adds Qwen3.5's recommended `1.5` presence penalty, which was
not present in the original Qwen3 protocol. This smoke must not be interpreted
as a one-task estimate of model quality.

Source: `outputs/qwen35_9b/smoke_1task_3x10_retry1`.

## Qwen3-4B modern-VERL full-parameter smoke

Two bridge smokes ran the known Qwen3-4B model through the modern VERL ALFWorld
agent loop and custom GiGPO trainer for one full-parameter update. Each used one
task, eight independently sampled same-task rollouts, three attempts of ten
turns, learning rate `1e-6`, token-mean policy loss, entropy coefficient `0.01`,
and no presence penalty. These are integration and memory diagnostics, not
evidence of learning.

Both recorded smokes used BF16 resident actor parameters and the provisional
`0.01` entropy coefficient. After the bridge was validated, the production 4B
profile was changed to the successful legacy run's FP32 resident actor with
BF16 mixed-precision compute and `0.001` entropy coefficient. The smoke timing
and memory measurements below therefore must not be treated as measurements of
the final 150-update precision profile.

| Result | Activation checkpointing | No activation checkpointing |
| --- | ---: | ---: |
| Completed updates | 1/1 | 1/1 |
| Successful rollouts | 0/8 | 0/8 |
| Play parse-valid | 238/240 (99.17%) | 239/240 (99.58%) |
| Play environment-effective | 184/240 (76.67%) | 186/240 (77.50%) |
| Play response-limit rate | 0/240 | 0/240 |
| Reflection parse-valid/effective | 16/16 | 16/16 |
| Entropy | 0.1630 | 0.1517 |
| Actor gradient norm | 0.4012 | 0.2492 |
| Rollout-correction KL | 0.00351 | 0.00286 |
| Rollout-correction effective sample size | 0.9941 | 0.9939 |
| Actor maximum allocated / reserved | 50.74 / 54.19 GiB | 78.01 / 81.65 GiB |
| Maximum sampled device use | 71,205 / 97,887 MiB | 85,721 / 97,887 MiB |
| Host memory reported by actor | 19.25 GiB | 18.54 GiB |
| Generation / actor update / full step | 64.12s / 48.67s / 123.90s | 60.09s / 41.73s / 112.64s |
| End-to-end runtime | 5m 55s | 4m 26s |

Both all-failure batches lacked task-success learning signal. Their mean scores,
`-0.00390625` and `-0.001953125`, are exactly explained by two and one invalid
records respectively receiving the `-0.5` penalty among 256 play/reflection
records. Both runs completed weight synchronization and post-update validation.
W&B runs `wjeoh77f` and `zhk3c145` synchronized successfully; the latter emitted
a non-fatal `BrokenPipeError` only while closing the already-synchronized W&B
service at process exit.

Both runs emitted PyTorch's checkpoint warning. Source inspection shows that it
comes from VERL's forward-only old-policy log-probability pass: the trainer asks
for entropy, `infer_batch` runs under `torch.no_grad()`, and entropy checkpointing
therefore receives logits that correctly do not require gradients. The actor
update is a separate non-forward-only pass and produced finite, nonzero gradient
norms. Disabling transformer activation checkpointing did not remove the warning,
which independently confirms that it was not reporting missing transformer
gradients. The checkpointed profile remains the pilot default: its seven-second
actor-update cost buys about 14.5 GB more peak device headroom for longer pilot
sequences.

Sources: `outputs/qwen3_4b_modern/full_gigpo_smoke_bridge` and
`outputs/qwen3_4b_modern/full_gigpo_smoke_no_checkpoint`.

## Qwen3.5-9B prompting baseline: balanced126

The full Qwen3.5-9B baseline used three attempts of ten turns, reflection,
thinking disabled, temperature/top-p/top-k `0.7/0.8/20`, presence penalty
`1.5`, and a 1,024-token response limit. Sixteen concurrent ALFWorld workers
were used and rollout log-probability collection was disabled because it is not
consumed by validation metrics.

| Tasks | P@1 | P@2 | P@3 | Gain P@3 - P@1 | Approx. runtime |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 126 | 17/126 (13.49%) | 36/126 (28.57%) | 49/126 (38.89%) | +32 (+25.40 pp) | 12m 02s |

The runtime is reconstructed from the first run-artifact timestamp to the
validation dump because the wrapper's `elapsed_seconds.txt` was not written.
The validation dump contains all 126 expected records and no task failures were
reported by Ray.

Each category contains 21 tasks. Values are cumulative successes.

| Category | P@1 | P@2 | P@3 | Added by attempt 2 | Added by attempt 3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Look at object in light | 8 | 9 | 11 | 1 | 2 |
| Pick and place | 6 | 15 | 18 | 9 | 3 |
| Clean then place | 1 | 5 | 7 | 4 | 2 |
| Cool then place | 0 | 0 | 0 | 0 | 0 |
| Heat then place | 0 | 0 | 2 | 0 | 2 |
| Pick two objects and place | 2 | 7 | 11 | 5 | 4 |
| **Overall** | **17** | **36** | **49** | **19** | **13** |

The final saved generation for every task was retokenized with the pinned
Qwen3.5 tokenizer. Seventeen of 126 final calls (13.49%) were exactly 1,024
tokens long; all 17 lacked a complete `<action>...</action>` pair. The six
task-balanced full trajectory samples contain 154 play/reflection calls: 21
(13.64%) reached the limit and 133 (86.36%) parsed successfully. These
trajectory-level percentages are diagnostic estimates from six deterministic
samples, not measurements over every call in all 126 tasks.

Source: `outputs/qwen35_9b/base_3x10_balanced126_presence15_fast16`.

## Qwen3.5-9B LoRA performance calibration

A realistic one-update calibration used the intended training workload of eight
tasks with eight same-task rollouts each, three attempts of ten turns, and all
GiGPO, log-probability, entropy, backward, optimizer, and vLLM adapter-refresh
stages. Validation and checkpoint writing were deliberately disabled so that
the measurement isolates a training update. Compiled vLLM execution used a
0.50 GPU-memory fraction, a 65,536-token batching budget, 64 maximum sequences,
eight agent workers, log-probability microbatch 8, and gradient plus entropy
checkpointing.

| Actor microbatch | Outcome | Step time | Generation | Old log probability | Actor update | Peak device memory |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | CUDA OOM during actor backward | n/a | completed | completed | failed | n/a |
| 4 | completed | 21m 45s | 7m 16s | 3m 30s | 10m 53s | 96,537 / 97,887 MiB |

The successful run processed 2,158,988 tokens at 1,654 tokens/s. External
telemetry measured 88.0% mean GPU utilization, with utilization at or above 90%
in 82.6% of samples. Mean power was 537.7 W and peak power was 608.9 W. The
actor reported 88.85 GiB peak allocated and 91.83 GiB peak reserved memory.
Because the device-level peak left only about 1.35 GiB free, actor microbatch 4
is a throughput result rather than a robust long-run setting; microbatch 2 is
the selected setting for the learning pilot while retaining the successful
generation and log-probability optimizations.

The update produced eight GiGPO groups: 25% mixed outcome, 50% all failure, and
25% all success. Valid-action ratio was 81.06%, all real advantage tokens were
nonzero, PPO KL was 0.000684, clip fraction was 0.432%, gradient norm was 0.361,
and mean entropy was 0.617. These are single-update diagnostics, not evidence of
learning. The run had no validation and wrote no checkpoint by design.

Sources: `outputs/qwen35_9b/lora_gigpo_perf_aggressive_fullstep` and
`outputs/qwen35_9b/lora_gigpo_perf_aggressive_mb4`.

### First full-run launch: host-memory worker loss

The first 150-step launch completed step-zero balanced84 validation and one
optimizer update, then was manually stopped after a Ray agent death left step
two permanently at seven of eight completed task groups. Step-zero validation
was P@1 = 18/84 (21.43%), P@2 = 31/84 (36.90%), and P@3 = 35/84 (41.67%).

At 18:05:53, node memory reached 59.16/62.25 GiB, just above Ray's 95%
threshold. Ray killed `agent_loop_worker_7`, which used approximately 1.12 GiB,
and classified its actor lease as non-retriable. The v1 fire-and-forget rollout
path left that prompt tagged `running`; the other seven groups finished, GPU
utilization fell to zero, and the replay buffer waited indefinitely. The
completed step-one metrics remain useful as diagnostics, but no post-update
checkpoint was written and this run is not a learning result.

The corrected launcher uses four rather than eight task workers, retains eight
rollouts inside every task group, and fails a rollout batch after a configurable
30-minute bound rather than accepting partial data or waiting forever.

Source: `outputs/qwen35_9b/lora_gigpo_full150_optimized_mb2`.

### First sustained LoRA run: policy collapse

The memory-corrected 150-step launch was stopped after 22 completed updates
because the policy had entered a clear low-entropy failure mode. It used LoRA
rank 16/alpha 32, learning rate `1e-5`, entropy coefficient `0.001`, global
`token-mean` loss aggregation, and presence penalty `1.5`. These are results
from the fixed balanced84 ID validation set, not balanced126.

| Step | P@1 | P@2 | P@3 | Actor entropy | Valid actions | Mean response tokens | Responses at 1,024-token cap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 19.05% | 33.33% | 38.10% | n/a | n/a | n/a | n/a |
| 5 | 20.24% | 35.71% | **40.48%** | 0.541 | 82.02% | 434 | 17.17% |
| 10 | 21.43% | 35.71% | 36.90% | 0.428 | 86.94% | 342 | 13.06% |
| 14 | n/a | n/a | n/a | 0.328 | **96.67%** | 187 | 3.16% |
| 15 | 15.48% | 25.00% | 29.76% | 0.296 | 88.70% | 269 | 11.02% |
| 20 | 2.38% | 7.14% | 11.90% | 0.0536 | 47.58% | 585 | 52.65% |
| 21 | n/a | n/a | n/a | 0.0260 | 24.44% | 796 | 75.46% |
| 22 | n/a | n/a | n/a | **0.0230** | **27.93%** | **758** | **71.97%** |

The entropy decrease before step 14 is not sufficient by itself to diagnose a
failure: over the same interval action validity improved and capped responses
decreased. The interpretation becomes clear after step 14, when entropy,
validity, response quality, and validation success deteriorate together and
saved validation text becomes repetitive. The best-checkpoint selector
correctly retained step 5 at P@3 = 34/84 (40.48%); no later checkpoint replaced
it. The stopped run lasted 30,531 seconds (8h 28m 51s).

Post-run inspection found two objective mismatches that make this run an
invalid test of the corrected recipe: presence-penalized vLLM rollout
probabilities were not used to correct the recomputed actor likelihoods, and
the modern GiGPO port applied the invalid-action penalty to episode scores but
not to discounted state returns. The old 4B implementation applied the latter
penalty to both. The next pilot therefore adds bounded token-level importance
sampling, restores the state-return penalty, uses equal per-generation loss
weighting, lowers the LoRA learning rate to `1e-6`, and raises the entropy
coefficient to `0.01`. Its outcome must be measured; these changes are motivated
corrections, not yet evidence that collapse is solved.

Source: `outputs/qwen35_9b/lora_gigpo_full150_memoryfix_optimized_mb2`.

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
