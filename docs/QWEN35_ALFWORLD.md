# Qwen3.5-9B ALFWorld integration

This document records the staged integration of `Qwen/Qwen3.5-9B` into the
ALFWorld workflow. The prompting baseline and GiGPO data/advantage path use the
same model-loading, chat-template, vLLM, FSDP, Ray, and environment
orchestration used by LoRA training. The modern GiGPO port and language-only
LoRA path have completed real GPU updates, adapter refreshes, and validation.
The first long pilot exposed a policy-collapse problem; the stabilization
changes and their evidence are recorded below.

## Why a separate environment is necessary

Qwen3.5 is not a drop-in model-name change for the original LaMer environment.
It is a natively multimodal model with a hybrid language backbone: three gated
linear-attention layers are interleaved with one full-attention layer. Its model
classes and recurrent cache support require a newer compatible combination of
Transformers, vLLM, PyTorch, and VERL than the working Qwen3-4B environment.

Upgrading those packages inside the existing `lamer` Conda environment would
risk breaking the reproducible Qwen3-4B results. The setup launcher therefore
creates an ignored `.compat/` environment from upstream VERL's own lockfile.
The exact pins are:

| Component | Pin |
|---|---|
| Python | `3.12.14` |
| uv | `0.12.19` |
| VERL | `fbb4b3a8bf636f290c9c59fc346f756849e9c241` |
| Qwen model revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| PyTorch | `2.13.0+cu130` |
| Transformers | `5.12.1` |
| vLLM | `0.29.0` |
| flash-attn | `2.8.3` |
| flash-linear-attention | `0.5.2` |
| PEFT | `0.19.1` |
| Ray | `2.55.1` |

The broader principle is that the model implementation, training backend, and
rollout engine form one compatibility unit. A single-package upgrade is not a
safe way to add a new architecture. This follows upstream VERL's guidance that
the Transformers version must track the inference engine and that CUDA, Torch,
and vLLM changes must be considered together.

Primary references:

- [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B)
- [Transformers Qwen3.5 architecture documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [VERL installation and dependency guidance](https://verl.readthedocs.io/en/latest/start/install.html)

## Set up the compatibility environment

Run once from the repository root:

```bash
DOWNLOAD_MODEL=1 bash examples/alfworld/setup_qwen35_compat.sh
```

This downloads a repository-local Python runtime, checks out the pinned VERL
revision under `.compat/verl-upstream`, installs its locked FSDP and vLLM stack,
adds the pinned ALFWorld dependencies, and downloads the exact model snapshot.
It does not modify the `lamer` Conda environment.

Re-running the command is intentional: it verifies the revisions and brings the
isolated environment back to the declared state.

`uv pip check` reports that Torch requests `nvidia-cusparselt-cu13==0.8.1`
while upstream VERL's x86 lock selects `0.8.0`. This is inherited from the
pinned upstream lock rather than introduced by the ALFWorld additions. Do not
upgrade cuSparseLt alone; validate a newer complete Torch/vLLM/VERL lock as a
unit if that mismatch is changed.

## Evaluation protocol

The launcher preserves the Qwen3-4B comparison protocol by default:

| Setting | Value |
|---|---:|
| Validation set | fixed balanced126 |
| Validation environment seed | `1000` |
| Rollout seed | `20` |
| Attempts per task | `3` |
| Turns per attempt | `10` |
| Reflection between attempts | enabled |
| Thinking mode | disabled |
| Temperature / top-p / top-k | `0.7 / 0.8 / 20` |
| Presence penalty | `1.5` |

These sampling values match Qwen's recommendation for general non-thinking
generation. The presence penalty was enabled after the unchanged one-task smoke
showed a 1,024-token recursive deliberation without an action tag. It discourages
revisiting tokens already used in a response while preserving reasoning before
the action. Set `PRESENCE_PENALTY=0.0` for the no-penalty control; Qwen3.5 runs
with and without this model-specific setting must be labelled separately.

Start with one task:

```bash
EVAL_TASK_COUNT=1 \
EVAL_DIR=outputs/qwen35_9b/smoke_1task_3x10 \
bash examples/alfworld/evaluate_base_qwen35_reflexion.sh
```

This is a real end-to-end GPU evaluation, not a unit test. It loads the FSDP
actor and vLLM rollout engine, runs one fixed ALFWorld task for up to three
attempts, and writes the command context, code revisions, versions, GPU details,
log, elapsed time, and validation artifacts under `EVAL_DIR`.

The evaluator also selects one trajectory per task type by default and writes
each selected trajectory incrementally to `EVAL_DIR/validation_traces/*.md`.
Each file contains the exact prompt, raw model response, parsed action or
reflection, resulting observation, token count, and outcome for every turn.
Incremental writes make a partial trace available even if a later generation
hangs or the run fails. For a one-task smoke run, inspect it while running with:

```bash
find outputs/qwen35_9b/smoke_1task_3x10/validation_traces -name '*.md' -print
tail -f outputs/qwen35_9b/smoke_1task_3x10/validation_traces/*.md
```

Set `TRAJECTORY_SAMPLES_PER_TASK=0` to disable trace files, or increase it to
retain more examples per ALFWorld category. `TRAJECTORY_SAMPLE_SEED` changes
the deterministic selection without changing rollout sampling. Tracing only
adds CPU file I/O after existing steps; it does not change model inputs,
generation settings, environment transitions, or evaluation metrics.

`MAX_RESPONSE_LENGTH` controls the per-call generation ceiling, and
`MAX_MODEL_LEN` defaults to the prompt and response limits added together. The
comparison default remains 1,024 response tokens. A longer limit is a distinct
protocol because it can rescue truncated actions while also giving repetitive
reasoning more compute. Test such a change on the one-task diagnostic first;
for example, `MAX_RESPONSE_LENGTH=1280` automatically raises the vLLM context
limit from 5,120 to 5,376 tokens.

Only after that succeeds, run all 126 tasks:

```bash
bash examples/alfworld/evaluate_base_qwen35_reflexion.sh
```

The full default output is `outputs/qwen35_9b/base_3x10_balanced126`.

On the 96-GiB RTX PRO 6000 machine, evaluation defaults to 16 ALFWorld agent
workers while Ray has 18 CPU slots. Each individual task remains sequential,
but independent tasks can then keep vLLM's GPU batch populated. Validation also
sets `CALCULATE_LOG_PROBS=False`: token log probabilities are needed for policy
optimization but not for success-rate evaluation, so omitting them avoids
unnecessary generation work without changing prompts, sampling, actions, or
metrics. Training continues to calculate rollout log probabilities.

The first full baseline retains eager vLLM execution and 50% GPU-memory
utilization. CUDA graphs may improve decode throughput, but changing execution
mode before a representative benchmark would combine a performance experiment
with the first scientific baseline. Keep `AGENT_LOOP_WORKERS=16` fixed for
later checkpoint evaluations because asynchronous request scheduling can affect
the exact samples drawn even when the engine seed is unchanged.

## Model-specific choices and tradeoffs

### Full multimodal checkpoint on a text-only task

ALFWorld supplies text only, so no image tensors enter the model. Nevertheless,
the launcher deliberately retains Qwen3.5's full processor and model class.
This validates the architecture that a future LoRA actor will actually load and
keeps Qwen3.5's three-component position-ID logic consistent between rollout
and FSDP.

Qwen and vLLM expose a language-model-only inference option that can omit the
vision encoder and free memory. That is attractive for pure serving, but using
it now would validate a different loading path from future training. It can be
benchmarked later as an explicitly labelled inference optimization after the
transferable path works. During LoRA training, the intended policy is to freeze
the vision encoder and projection and adapt only selected language modules.

### Chunked prefill is required

Qwen3.5's Gated DeltaNet layers maintain recurrent state during generation. On
this stack, vLLM failed to initialize the cache with chunked prefill disabled.
The launcher therefore sets `enable_chunked_prefill=True`. This is an
architecture requirement observed in a real full-model load, not merely a
throughput preference.

### Conservative first-run memory and execution settings

The evaluation launcher uses BF16, tensor parallel size one, 50% vLLM memory
utilization, a 5,120-token model limit, and eager execution. The 5,120-token
limit equals the 4,096-token prompt budget plus the 1,024-token response budget.
It avoids reserving KV cache for Qwen3.5's full native context window, which is
irrelevant to ALFWorld and would waste GPU memory.

Eager execution avoids CUDA-graph capture during the first integration run. It
is slower than a tuned graph configuration but gives clearer failures. After a
full baseline succeeds, memory utilization, maximum concurrent sequences, and
CUDA graphs can be tuned one variable at a time while checking that outputs and
success metrics remain comparable.

### Local ALFWorld sessions

Modern VERL already distributes asynchronous agent-loop workers through Ray.
The Qwen3.5 loop therefore runs each TextWorld game in-process inside its VERL
worker instead of creating nested Ray actors and blocking on `ray.get` for every
environment step. This changes orchestration overhead, not the selected game,
prompt templates, action projection, reward, or attempt protocol.

### Synchronous text tokenization on this machine

A machine-level diagnostic showed that a reused `asyncio` executor thread
completes its first job but is not woken for its second job. The result was the
same with the system, Conda, and repository-local Python builds, while ordinary
synchronous tokenization and fresh thread pools worked.

The custom ALFWorld loop therefore calls VERL's normal Qwen3.5 continuous-token
builder synchronously. It still uses the full `Qwen3VLProcessor`, identical chat
template, and identical token IDs. Only scheduling changes. Text tokenization
takes milliseconds, so serializing this short CPU work inside each worker is a
small cost relative to GPU generation. Upstream VERL and other agent loops are
not patched.

## What has been verified

- The pinned model loads fully in vLLM on the RTX PRO 6000 Blackwell GPU and
  generates text in BF16.
- A one-task 3x10 reflection evaluation completed end to end without OOM in
  5m57s, including all three attempts and saved validation output.
- The Gated DeltaNet prefill and decode kernels initialize on this hardware.
- The non-thinking chat template inserts the expected empty thinking boundary.
- Processor-backed and tokenizer-only rendering produce identical token IDs for
  text-only prompts.
- The seed-1000 game order matches the existing balanced126 evaluator.
- A CPU end-to-end loop executes `play -> reflect -> play`, restarts the same
  game, projects actions/reflections, and emits cumulative attempt and task-type
  metrics.
- VERL's real postprocessor accepts the final loop output and constructs the
  Qwen3.5 prompt/response masks and four-axis position IDs.
- The complete launcher composes against the pinned VERL configuration.
- The modern GiGPO estimator produces numerically identical advantages to the
  original LaMer estimator on the same episode scores, step returns, task IDs,
  and anchor observations.
- The explicit training dataset reproduces the legacy launcher's independent
  seeded task-slot schedule and records it in the run manifest.
- The modern VERL extension mechanism registers the repository-owned GiGPO
  trainer without modifying the pinned upstream checkout.

These checks validate the integration layers but do not establish task success
or training throughput. In the one-task evaluation, the unchanged instruction
to reason before acting caused Qwen3.5 to fill the 1,024-token response budget
without emitting an action tag. That is a model/prompt behavior finding rather
than an orchestration failure. Reasoning must remain before the action because
it is intended to inform action selection. The next isolated diagnostic should
therefore retain the prompt and now apply Qwen3.5's recommended presence
penalty. Only if that is insufficient should a bounded, still action-preceding
reasoning instruction be tested as a separately labelled model-specific
condition. The one-update training smoke remains the next GPU checkpoint after
the prompt protocol is selected.

## Modern GiGPO training path

LaMer treats every ALFWorld action and reflection generation as an independent
prompt/response policy sample. Modern VERL's TransferQueue can already retain a
list of outputs from one agent session, but its standard multi-trajectory
advantage path is GRPO-oriented. The repository therefore registers a small
`alfworld_gigpo_sync` trainer rather than patching the pinned VERL checkout.

For each training task, `actor_rollout_ref.rollout.n=8` creates eight
independently sampled sessions. A shared dataset UID identifies the task group,
and a distinct trajectory UID identifies each session. Every play action and
reflection records its attempt, turn, pre-action observation, validity,
immediate reward, per-attempt episode reward, and discounted step return.

The step return is computed inside the session, where chronological attempt
boundaries are known. Group normalization is computed later in the trainer,
after all eight sibling trajectories are present. This separation is required:
a single session cannot calculate a relative group advantage by itself.

The port deliberately retains two details of the original implementation for
scientific comparability:

- episode normalization is weighted by generated records rather than reducing
  to one scalar per trajectory first;
- the invalid-action penalty is applied to both the episode score and the
  discounted step return before their respective group-centering operations.

Either behavior can be studied later as an explicitly labelled algorithmic
variant, but changing it during the VERL port would confound the comparison.

The shared modern launcher now exposes one such labelled variant as
`FUTURE_AWARE_EPISODE_CREDIT=True`. Its default is `False`, preserving original
LaMer episode credit. When enabled, an attempt's episode score is the existing
backward `step_return` at its first play record. Thus reward is discounted by
`STEP_GAMMA` through actions inside later attempts before `TRAJ_GAMMA` is
applied at an attempt boundary. All records in an attempt share that entry
score, and reflections receive the entry score of the retry they guide. This
option does not otherwise alter the step-level return, grouping, normalization,
or invalid-action penalty and must be reported as a separate algorithmic
experiment.

## Training task schedule

The legacy launcher maintained eight task slots. Slot `i` independently
shuffled the training games with environment seed `seed + i`, then advanced to
its next game on every update. The modern dataset builder writes that exact
step-major schedule into `train.parquet`; each row then expands into eight model
rollouts. With data shuffling disabled, the modern dataloader reproduces the
same task curriculum and the manifest makes every selected game auditable.

The training pilot validates on `eval_id_checkpoint`, the fixed balanced84 ID
split containing the four task categories seen in training. This is different
from balanced126, which additionally includes the two OOD categories and is
used for final reporting. Selecting checkpoints on balanced84 avoids using OOD
test categories for model selection.

## Full-language training checkpoints

The launcher defaults to a one-update smoke run:

```bash
bash examples/alfworld/train_full_qwen35_9b.sh
```

It performs one-task step-zero validation, generates one group of eight
training trajectories, computes GiGPO advantages, performs one optimizer
update, refreshes the vLLM rollout weights, and validates the same task again at
step one. The post-update validation is not an estimate of generalization from
one task; it is a wiring check that the updated weights reach vLLM and can be
used for generation. Both validation points and the update diagnostics are sent
to console and W&B using the machine's existing W&B credential. Set
`TRAINER_LOGGER='[console]'` for an explicitly offline run. The smoke run does
not save a checkpoint.

Do not start the learning pilot merely because the smoke succeeds: the smoke
uses BF16 trainable parameters without FP32 master weights and therefore proves
integration, not adequate update precision. After the optimizer/adapter
precision path described below is resolved, the intended pilot form is:

```bash
RUN_MODE=pilot bash examples/alfworld/train_full_qwen35_9b.sh
```

The pilot uses eight tasks per update, eight rollouts per task, 20 updates,
step-zero balanced84 validation, and validation every five updates. It keeps the
original LaMer learning rate `1e-6` so that the port and learning-rate choice
are not changed in the same experiment. No checkpoint is saved by default;
`SAVE_FREQ` is available if a diagnostic checkpoint is explicitly wanted.

The discount settings preserve the original launcher: global gamma `0.95`,
within-attempt step gamma `0.95`, and cross-attempt trajectory gamma `0.6`.
The custom GiGPO port constructs returns with
the latter two discounts; global gamma is pinned for configuration parity and
would apply to VERL's standard return estimators, rather than being silently
left at modern VERL's default. After the first 22-step LoRA run drove actor
entropy from `0.617` to `0.023`, valid actions from `0.790` to `0.279`, and the
response-limit rate from `0.210` to `0.720`, the Qwen3.5 entropy coefficient
was raised from `0.001` to `0.01`. This remains a regularizer rather than the
primary objective, but it is large enough to exert measurable pressure against
premature determinism. These values are explicit launcher variables and are
captured in `training_parameters.txt`.
Qwen3.5's 248,320-token vocabulary makes the entropy softmax a material
activation. Entropy checkpointing discards that intermediate after the forward
calculation and recomputes it during backward, trading additional computation
for lower peak memory without changing the entropy objective.

Training and validation intentionally use different sampling distributions.
Validation uses Qwen's non-thinking general-task recommendation
(`temperature=0.7`, `top_p=0.8`, `top_k=20`) for a stable measure of task
performance. Training retains LaMer's broader behavior policy
(`temperature=1.0`, `top_p=1.0`, no top-k cutoff) so all vocabulary choices
remain available for exploration and the Qwen3-4B comparison does not also
change its RL sampling policy. The shared `1.5` presence penalty is the one
model-specific addition prompted by Qwen3.5's observed recursive responses.
Because it changes vLLM's sampling distribution but is not part of the actor
forward pass, training records vLLM `processed_logprobs` and applies clipped
token-level importance weights `pi_actor / pi_rollout` with an upper bound of
`2.0`. This is VERL's decoupled rollout-correction path: PPO retains a freshly
recomputed actor policy as its proximal anchor while importance sampling
corrects the behavior-policy gap. Sequence-level products were rejected
because hundreds of generated tokens would make their variance prohibitive.
The launcher rejects raw rollout log-probabilities or disabled correction for
this recipe.

Qwen3.5 action responses vary from a few tokens to the full 1,024-token limit,
whereas the successful 4B run usually generated about 130--230 tokens. The
actor therefore uses `seq-mean-token-mean` aggregation: every generated token
still receives its action's GiGPO advantage, including reasoning tokens, but
each action or reflection contributes equal total weight. Global `token-mean`
would allow a long repetitive response to dominate a short valid action solely
because it contains more tokens.

The first training smoke uses 35% vLLM memory so the colocated backward pass
has more headroom than inference-only evaluation. It also uses FSDP2,
microbatch one, gradient checkpointing, and optimizer-only CPU offload. This is
a diagnostic configuration, not yet a production memory guarantee.

The pinned 9B BF16 checkpoint is 19.3 GB (18.0 GiB). PyTorch AdamW preserves
the BF16 parameter dtype for both of its moment tensors in this configuration,
so parameters, gradients, and the two moments require approximately
`18 + 18 + 36 = 72 GiB` during an update, before activations and temporary
tensors. There is no separate FP32 master-weight copy: the pinned VERL loader
passes `model_dtype=bfloat16` to Transformers, explicitly converts the module
to BF16, and constructs ordinary `torch.optim.AdamW` over those BF16
parameters. On one GPU, FSDP has no second rank across which to shard them.

The machine-specific compromise is therefore:

- keep the approximately 18-GiB actor parameters on the 96-GiB GPU;
- offload only the approximately 36-GiB optimizer state to system RAM between
  updates;
- let VERL put vLLM into level-2 sleep before backpropagation, releasing its
  rollout weights and KV cache;
- use gradient checkpointing and microbatch one to leave roughly 23 GiB of GPU
  capacity beyond the static update state for activations, logits, and
  temporary buffers.

FSDP2's general CPU offload policy is disabled by default because it also puts
the 18-GiB parameters and gradients in host memory. Together with optimizer
state, Ray, Python, and ALFWorld, that is a poor fit for 60 GB RAM. The selected
phase-based placement is likely to fit, but only a completed first AdamW step,
vLLM weight refresh, and post-update validation can verify the dynamic peaks.
Initialization or rollout success alone is not sufficient evidence.

This memory-feasible profile is not numerically equivalent to the earlier 4B
full fine-tune. Direct inspection of its step-150 checkpoint shows 4.411 billion
FP32 actor parameters and 8.045 billion FP32 optimizer-state elements. In
contrast, a `1e-6` update applied directly to BF16 parameters may be smaller
than one representable BF16 increment for many weights and can round to zero.
The full-model smoke therefore validates integration and memory only; it must
not be promoted to a learning pilot until optimizer precision is resolved.

The old run was mixed precision, so “FP32 parameters” does not mean its matrix
multiplications ran in FP32. Its recorded profile is `MODEL_DTYPE=fp32_default`:
FSDP retained the original parameters as FP32 master/optimizer parameters,
temporarily cast weights to BF16 for forward and backward computation, reduced
gradients in FP32, and—because `keep_low_precision_grads` retained its default
`False`—upcast gradients to FP32 before AdamW. AdamW's first and second moments
were also FP32. Thus “BF16 working weights with FP32 master weights” is a valid
high-level description, but at the optimizer boundary the parameters,
gradients, and moments were all FP32.

A faithful 9B FP32-master configuration would need roughly 36 GiB for master
parameters, 36 GiB for gradients, and 72 GiB for the two FP32 Adam moments,
before BF16 compute copies, activations, and vLLM. Neither the 96-GiB GPU nor
the 60-GiB host can hold the relevant phase by itself. Upstream VERL supports a
TorchAO BF16 AdamW variant with stochastic rounding, which can preserve small
updates without FP32 master weights, but TorchAO is not installed in the pinned
environment and adopting it would be an explicitly different optimizer
experiment.

An 8-bit optimizer could reduce memory further, but it adds a dependency and
changes optimizer numerics, weakening comparison with the 4B experiment. The
planned language-only LoRA run remains the cleaner production path: the frozen
base weights remain, while gradients, FP32 adapter weights, and optimizer state
can be limited to the small adapters. The adapter dtype and FSDP wrapping must
be verified explicitly because the current upstream helper casts default LoRA
adapters to the BF16 base dtype.

Prefix caching is enabled for training after an end-to-end 126-task evaluation
validated it with Qwen3.5. It can reuse the identical initial prompt shared by
the eight same-task rollouts and long common prefixes in later turns. The pinned
VERL vLLM adapter clears its KV and prefix caches after every policy-weight
update, so a new optimization step cannot consume activations produced by the
previous policy. Set `ENABLE_PREFIX_CACHING=False` for an explicit control.

Other evaluation speedups are deliberately not copied blindly. Training keeps
rollout log probabilities for the initial consistency audit, frees the vLLM
cache engine before backpropagation, and retains eager execution until the
one-update smoke has established memory use and correct weight refresh. The
pilot already exposes 64 sessions concurrently (`8 tasks x 8 rollouts`), so
increasing agent-worker or sequence limits would add resource pressure without
creating more training trajectories.

Qwen3.5 is multimodal, but ALFWorld batches contain no visual tensors and skip
the visual forward path. The pinned FSDP implementation declares a
`freeze_vision_tower` option but does not consume it, so the launcher does not
claim that the tower is explicitly frozen. The first GPU smoke run must include
a gradient/parameter audit before the experiment is described as full-language
fine-tuning.

## Language-only LoRA training

The LoRA entry point is:

```bash
bash examples/alfworld/train_lora_qwen35_9b.sh
```

It is intentionally a one-update smoke by default. It performs the same
one-task step-zero validation as the training protocol, samples one task group
of eight rollouts, computes GiGPO returns and advantages over every action and
reflection generation, applies one optimizer update, synchronizes the adapter
to vLLM, and validates once more. It sends metrics to console and W&B and saves
the validation best plus the final adapter. A one-task success-rate difference
is not scientific evidence; the smoke checks wiring, precision, memory, finite
optimization, and rollout-engine refresh.

Only after that run passes should the 20-step balanced84 pilot be started:

```bash
RUN_MODE=pilot bash examples/alfworld/train_lora_qwen35_9b.sh
```

The production schedule is exposed but should not precede the pilot review:

```bash
RUN_MODE=full bash examples/alfworld/train_lora_qwen35_9b.sh
```

That mode uses eight task groups per update, eight same-task trajectories per
group, 150 updates, step-zero validation, and the fixed balanced84 validation
set every five updates. The task schedule, prompt, reflection protocol,
discounts, entropy coefficient, invalid-action penalty, and training/validation
sampling distributions are the same as the modern full-parameter launcher.
Thus the adapter parameterization is the intended independent variable.

### Adapter target and capacity

The target is an explicit regular expression rather than PEFT's convenient
`all-linear` setting. Inspection of the pinned checkpoint found 248 intended
language projections:

| Language component | Modules |
|---|---:|
| Gated DeltaNet `in_proj_a`, `in_proj_b`, `in_proj_qkv`, `in_proj_z`, `out_proj` | 120 |
| MLP `gate_proj`, `up_proj`, `down_proj` | 96 |
| Full-attention `q_proj`, `k_proj`, `v_proj`, `o_proj` | 32 |
| **Total** | **248** |

At rank 16 these modules create 496 matrices (`lora_A` and `lora_B` for every
target) containing exactly 43,278,336 trainable elements, about 0.45% of the
9.65-billion-element checkpoint. The vision tower/projector, embeddings,
language-model head, MTP modules, recurrent convolution, recurrent state
parameters, and normalization parameters remain frozen. The custom engine
fails before training if the module count, per-layer distribution, element
count, dtype, or any exclusion differs from this audited architecture. This is
preferable to silently training a partial or multimodal adapter after an
upstream model change.

The selected adapter settings are rank 16, alpha 32, zero dropout, no trained
bias, standard PEFT initialization, and no rank-stabilized LoRA. LoRA applies
the low-rank update with scale `alpha / rank`, so the initial scale is `2`.
Standard initialization randomizes `A` and initializes `B` to zero; therefore
the step-zero adapter contributes exactly zero and the initial policy equals
the frozen base model.

Alpha 32 is a controlled starting point informed by the public
[LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) recipe, not a
claim that alpha 32 is universally optimal for ALFWorld. Alpha changes the
effective update scale and interacts with learning rate. Rank 16 was chosen as
a moderate capacity/memory point; rank 32 doubles adapter and optimizer memory.
If the pilot is too weak or unstable, rank, alpha, and learning rate must be
varied as labelled experiments rather than changed together.

### Precision and optimizer path

The frozen checkpoint is stored persistently in BF16. PEFT creates the
trainable adapter parameters in FP32, while FSDP2 gathers/casts them to BF16 for
forward and backward matrix multiplication. FSDP2 then exposes FP32 gradients
to ordinary PyTorch AdamW, whose `exp_avg` and `exp_avg_sq` are also FP32. There
is no full-model FP32 master copy: only the small trainable adapter has a
high-precision persistent/update path.

This requires a repository-owned FSDP2 engine specialization because pinned
VERL's generic helper casts LoRA parameters back to the BF16 base dtype for
FSDP1 compatibility. FSDP2 keeps parameters individually as DTensors and can
retain frozen BF16 parameters alongside trainable FP32 parameters. A CPU-only
distributed diagnostic verified BF16 forward output with frozen BF16 base
parameters and FP32 adapter parameters, gradients, and Adam moments. The real
engine repeats these audits before and after FSDP wrapping and after the first
optimizer step, then aborts if an assumption is false.

The resolved model configuration also prints a nested `model.lora.rank: 0`.
That block is VERL's separate Megatron-LoRA schema; the pinned FSDP engine uses
the flat `model.lora_rank: 16`, and vLLM explicitly falls back to that flat
field. Setting the Megatron rank merely to make the printout symmetric would
risk activating backend-specific defaults that do not describe this PEFT
adapter.

The optimizer is explicit AdamW with learning rate `1e-6`, constant schedule,
zero warmup, betas `(0.9, 0.999)`, epsilon `1e-8`,
weight decay `0.01`, global gradient clipping at `1.0`, and one PPO epoch. Both
`A` and `B` belong to the same optimizer and use the same LoRA learning rate.
The initial `1e-5` 10x-LoRA heuristic improved some training-format metrics
through step 14 but then collapsed entropy, valid actions, and balanced84
success. The new default returns to LaMer's `1e-6` rate. Equal numeric rates do
not imply equal model updates: LoRA still applies its learned low-rank matrices
through the `alpha/rank = 2` scale and updates only 43.3 million parameters.
The corrected 20-step pilot is the decision point for any later rate increase.

The 43.3 million trainable values occupy about 165 MiB in FP32. Parameters,
gradients, and two Adam moments total about 660 MiB before temporary buffers.
That makes CPU optimizer offload unnecessary and avoids PCIe traffic. The
approximately 18-GiB frozen BF16 base remains on the GPU; vLLM sleeps and frees
its cache for the update phase. Gradient and entropy checkpointing trade extra
recomputation for activation memory. The first smoke keeps actor and log-prob
microbatches at one because long multi-turn records and the 248,320-token
vocabulary make activation/logit memory the uncertain term. A larger
microbatch is a later throughput experiment after measuring the real peak on
the 96-GiB GPU; it does not change the intended global PPO mini-batch or the
number of optimizer updates. `ACTOR_MICRO_BATCH_SIZE` and
`LOG_PROB_MICRO_BATCH_SIZE` control that execution-only split. General model
checkpointing is separately exposed as `GRADIENT_CHECKPOINTING`; it remains on
while microbatch size is calibrated so two independent memory/speed changes are
not mixed. Entropy checkpointing has its own existing switch because Qwen3.5's
large vocabulary makes that intermediate unusually expensive.

Generation has an independent execution switch, `ENFORCE_EAGER`. The default
remains `True` until a matched LoRA smoke verifies Qwen3.5's compiled vLLM path,
CUDA-graph capture, adapter refresh, and sleep/wake cycle together. Setting it
to `False` permits the pinned vLLM to compile this explicitly supported model
and use CUDA graphs for repeated decode shapes. Compilation adds one-time
startup cost and uses more GPU memory, but can reduce per-token kernel-launch
overhead across a long training run. It does not change the prompt or sampling
parameters, although different kernels need not produce bitwise-identical BF16
logits.

Long diagnostics also default to a lightweight GPU monitor. Every two seconds,
the launcher records utilization, device-memory use, power, temperature, and
SM/memory clocks in `gpu_timeseries.csv`. `ENABLE_GPU_MONITOR=False` disables
it, and `GPU_MONITOR_INTERVAL_SECONDS` changes the interval. This external
telemetry does not interact with CUDA or the training process; it complements
VERL's phase timings and actor peak-memory counters.

Pilot and full runs use four ALFWorld agent workers by default. Eight workers
saturated generation, but the full step-zero-validation run measured roughly
58 GiB of the 62-GiB host RAM in use. Ray then crossed its 95% memory threshold,
killed one approximately 1.1-GiB agent actor, and left its fire-and-forget
prompt marked as running. Four workers preserve the eight rollouts inside every
task group while recovering about 4.5 GiB of host-memory headroom. The manager
still dispatches all eight task groups and all 64 rollout sessions; fewer Python
actors provide less process-level CPU isolation and may modestly affect
environment-side throughput, but do not change the GiGPO batch or objective.

The repository-owned replay-buffer adapter also limits any one validation or
training rollout batch to 1,800 seconds by default. `ROLLOUT_TIMEOUT_SECONDS`
can raise that bound for unusually slow hardware. Exceeding it fails with the
pending/running/finished/failure counts rather than leaving the GPU idle in an
infinite wait. The timeout never turns an incomplete group into training data.

Modern VERL names validation aggregates under `val-core/*` and `val-aux/*`.
The custom trainer retains those source metrics and also publishes the familiar
`val/*` and `val/meta_rl/p_at_{1,2,3}` aliases. Its repository-local tracking
compatibility shim commits every explicit W&B step immediately, avoiding the
one-step live-chart delay in the pinned upstream tracker.

### Rollout synchronization and checkpoints

vLLM loads the frozen base once and enables one rank-16 adapter. On each policy
refresh, pinned VERL transfers the adapter tensors and PEFT configuration and
replaces the old vLLM adapter. It does not merge and retransmit the 9B base on
every update. Qwen3.5's vLLM implementation declares LoRA support for the
targeted standard and packed projections, but only the end-to-end smoke can
prove that the updated adapter changes this exact rollout process.

The LoRA path uses vLLM sleep level 1 while the actor updates. That mode copies
the frozen BF16 base into pinned host memory before releasing its GPU pages,
then restores the weights on wake. PyTorch may retain the now-unused pinned
block in its allocator cache even after vLLM drops the backup tensor. On this
62-GiB host, the retained block can push a later rollout over Ray's 95% RAM
threshold. `RELEASE_VLLM_HOST_CACHE_AFTER_WAKE=True` therefore returns unused
pinned blocks to the OS immediately after the synchronous weight wake and logs
the vLLM worker RSS before and after. It does not discard live weights or alter
the model, samples, or objective. The tradeoff is recreating the pinned staging
allocation at the next sleep; set the switch to `False` for a matched timing
control. The compatibility module fails explicitly if the pinned PyTorch build
does not expose the audited host-cache operation.

Validation uses cumulative third-attempt success
`val-aux/alfworld/success_rate[2]/mean@1` as the checkpoint-selection metric.
Step zero is eligible: if training never improves on the base policy, step zero
correctly remains the best. Strict improvements replace the previous best;
ties keep the earlier checkpoint. Every scheduled validation also replaces the
rolling latest checkpoint, so an interrupted run can resume from its most
recent validated update rather than from the validation best. Disk contains
one checkpoint when best and latest coincide and two otherwise. With the
unchanged five-update validation cadence, interruption can lose at most four
completed updates. `checkpoint_selection.json` records the metric, scores,
steps, tie policy, and retained directories. Native “newest N” rotation is
disabled because it cannot implement “best plus rolling latest.” Checkpoints store only adapter
weights plus optimizer/scheduler/RNG and dataloader state, not another copy of
the frozen 9B model.

The launcher records the command inputs, repository and VERL revisions,
working-tree patch, exact model revision, software versions, GPU information,
training parameters, logs, validation artifacts, and elapsed time under the
output directory. The first GPU run must establish all of the following before
the pilot is authorized:

- the before/after-FSDP target and precision audits pass;
- the first optimizer audit finds FP32 adapter gradients and Adam moments;
- gradient norm and losses are finite and some advantages are nonzero;
- rollout probability diagnostics are plausible;
- the adapter refresh reaches vLLM and post-update generation completes;
- peak GPU and CPU memory leave credible headroom;
- the step-zero and final checkpoint manifest contains the expected paths.

The optional `causal-conv1d` extension is deliberately not added for this
first run. The pinned environment already contains
`flash-linear-attention==0.5.2`, and the completed prompting baseline verified
the Gated DeltaNet GPU path. Adding another compiled kernel could improve the
small causal-convolution component, but it is a dependency and performance
change, not a correctness requirement, so it should be benchmarked separately
after the training path works.

### ALFWorld environment reuse

Modern VERL creates one agent-loop object per trajectory but keeps each Ray
agent-loop worker alive across training steps. Creating and closing a new
TextWorld environment in every trajectory retains native/Python allocations
and one global TextWorld registry entry. On the 60 GiB host this accumulated
until Ray killed a rollout worker, leaving the replay buffer waiting for the
groups assigned to that worker.

The Qwen3.5 agent loop therefore leases environments from a process-local,
bounded pool. A lease reloads the requested game and seed into an existing
TextWorld wrapper; the reflection/history manager remains trajectory-local.
Consequently no observations, actions, memories, or reflections are shared
between rollouts. `ALFWORLD_ENV_POOL_SIZE` controls the maximum number of live
environments in each Ray agent-loop worker. By default the launcher derives it
from the busiest worker's training concurrency. For the production settings it
is `16`: `TRAIN_BATCH_SIZE=8`, `GROUP_SIZE=8`, and four agent-loop workers give
`8 * 8 / 4 = 16` concurrent training trajectories per worker. Validation may
queue above that bound, which can be slightly slower but prevents it from
increasing the process memory ceiling.

Pool allocation and periodic reuse messages include the worker's peak RSS.
The existing host-wide `perf/cpu_memory_used_gb` metric remains the primary
cross-step check. A healthy pilot should reach the configured pool size during
the first rollout and then reuse it without adding TextWorld registry entries
or showing step-over-step host-memory growth.

The launcher also sets `TRANSFER_QUEUE_STORAGE_UNITS=2`. Upstream's generic
default of eight targets distributed jobs, while its recommendation is at least
twice the node count. This experiment has one node and one synchronous trainer,
so two retains the recommended redundancy and local transport concurrency while
avoiding six otherwise idle Python/Ray processes (about 2.7 GiB in the failed
run). TransferQueue still clears each sampled batch after the optimizer step.
Raw prompt, response, and next-observation strings are not transported with
training records; token IDs and all GiGPO-required state/reward metadata remain.
