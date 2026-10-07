#!/usr/bin/env bash
# Validation-only ALFWorld evaluation through modern VERL's trainable rollout path.

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
COMPAT_DIR=${COMPAT_DIR:-$REPO_ROOT/.compat}
VERL_DIR=${VERL_DIR:-$COMPAT_DIR/verl-upstream}
PYTHON_BIN=$VERL_DIR/.venv/bin/python
VERL_REVISION=fbb4b3a8bf636f290c9c59fc346f756849e9c241
MODEL_REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a
MODEL_SNAPSHOT=$COMPAT_DIR/hf-cache/models--Qwen--Qwen3.5-9B/snapshots/$MODEL_REVISION

MODEL_PATH=${MODEL_PATH:-$MODEL_SNAPSHOT}
EVAL_DIR=${EVAL_DIR:-$REPO_ROOT/outputs/qwen35_9b/base_3x10_balanced126}
EVAL_DATASET=${EVAL_DATASET:-eval_all}
EVAL_TASK_COUNT=${EVAL_TASK_COUNT:-126}
ENV_SEED=${ENV_SEED:-0}
VAL_ENV_SEED=${VAL_ENV_SEED:-1000}
ROLLOUT_SEED=${ROLLOUT_SEED:-20}
NUM_ATTEMPTS=${NUM_ATTEMPTS:-3}
MAX_TURNS_PER_ATTEMPT=${MAX_TURNS_PER_ATTEMPT:-10}
DO_REFLECTION=${DO_REFLECTION:-True}
REFLECTION_TYPE=${REFLECTION_TYPE:-reflection_only}
EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-0.7}
EVAL_TOP_P=${EVAL_TOP_P:-0.8}
EVAL_TOP_K=${EVAL_TOP_K:-20}
PRESENCE_PENALTY=${PRESENCE_PENALTY:-1.5}
BRIEF_RESPONSE_INSTRUCTION=${BRIEF_RESPONSE_INSTRUCTION:-False}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.50}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-1}
CALCULATE_LOG_PROBS=${CALCULATE_LOG_PROBS:-False}
AGENT_LOOP_WORKERS=${AGENT_LOOP_WORKERS:-16}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-18}
TRAJECTORY_SAMPLES_PER_TASK=${TRAJECTORY_SAMPLES_PER_TASK:-1}
TRAJECTORY_SAMPLE_SEED=${TRAJECTORY_SAMPLE_SEED:-0}
ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
RAY_TMPDIR=${RAY_TMPDIR:-/tmp/lamer-q35-$UID}

if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'Missing compatibility environment. Run examples/alfworld/setup_qwen35_compat.sh first.\n' >&2
    exit 2
fi
if [[ "$(git -C "$VERL_DIR" rev-parse HEAD)" != "$VERL_REVISION" ]]; then
    printf 'VERL checkout is not at the pinned revision %s.\n' "$VERL_REVISION" >&2
    exit 2
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    printf 'Missing pinned Qwen3.5-9B snapshot at %s.\n' "$MODEL_PATH" >&2
    printf 'Run DOWNLOAD_MODEL=1 examples/alfworld/setup_qwen35_compat.sh or set MODEL_PATH.\n' >&2
    exit 2
fi
if [[ ! -d "$ALFWORLD_DATA/json/valid_task_balanced126" ]]; then
    printf 'Missing prepared ALFWorld balanced126 data under %s.\n' "$ALFWORLD_DATA" >&2
    exit 2
fi
if ((MAX_MODEL_LEN < MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)); then
    printf 'MAX_MODEL_LEN=%s must be at least MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH=%s.\n' \
        "$MAX_MODEL_LEN" "$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" >&2
    exit 2
fi

mkdir -p "$EVAL_DIR/data" "$EVAL_DIR/hydra" "$EVAL_DIR/validation_traces" "$RAY_TMPDIR"
EVAL_DIR=$(cd -- "$EVAL_DIR" && pwd)
export ALFWORLD_DATA RAY_TMPDIR
export HF_HOME=$COMPAT_DIR/hf-cache
export UV_CACHE_DIR=$COMPAT_DIR/uv-cache
export PYTHONHASHSEED=$ROLLOUT_SEED
export PYTHONPATH="$VERL_DIR:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" "$REPO_ROOT/scripts/prepare_alfworld_agent_dataset.py" \
    --output-dir "$EVAL_DIR/data" \
    --eval-dataset "$EVAL_DATASET" \
    --validation-seed "$VAL_ENV_SEED" \
    --validation-size "$EVAL_TASK_COUNT"

cp -- "${BASH_SOURCE[0]}" "$EVAL_DIR/launcher.sh"
git -C "$REPO_ROOT" rev-parse HEAD > "$EVAL_DIR/code_revision.txt"
git -C "$REPO_ROOT" status --short > "$EVAL_DIR/worktree_status.txt"
git -C "$REPO_ROOT" diff --no-ext-diff > "$EVAL_DIR/working_tree.patch"
git -C "$VERL_DIR" rev-parse HEAD > "$EVAL_DIR/verl_revision.txt"
"$PYTHON_BIN" -P -c \
    'import platform, torch, transformers, vllm, verl; print("python", platform.python_version()); print("torch", torch.__version__, "cuda", torch.version.cuda); print("transformers", transformers.__version__); print("vllm", vllm.__version__); print("verl", verl.__version__)' \
    > "$EVAL_DIR/software_versions.txt"
nvidia-smi > "$EVAL_DIR/gpu_info.txt" 2>&1 || true

cat > "$EVAL_DIR/evaluation_parameters.txt" <<EOF
MODEL_PATH=$MODEL_PATH
MODEL_REVISION=$MODEL_REVISION
EVAL_DATASET=$EVAL_DATASET
EVAL_TASK_COUNT=$EVAL_TASK_COUNT
ENV_SEED=$ENV_SEED
VAL_ENV_SEED=$VAL_ENV_SEED
ROLLOUT_SEED=$ROLLOUT_SEED
NUM_ATTEMPTS=$NUM_ATTEMPTS
MAX_TURNS_PER_ATTEMPT=$MAX_TURNS_PER_ATTEMPT
DO_REFLECTION=$DO_REFLECTION
REFLECTION_TYPE=$REFLECTION_TYPE
EVAL_TEMPERATURE=$EVAL_TEMPERATURE
EVAL_TOP_P=$EVAL_TOP_P
EVAL_TOP_K=$EVAL_TOP_K
PRESENCE_PENALTY=$PRESENCE_PENALTY
BRIEF_RESPONSE_INSTRUCTION=$BRIEF_RESPONSE_INSTRUCTION
MAX_PROMPT_LENGTH=$MAX_PROMPT_LENGTH
MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH
MAX_MODEL_LEN=$MAX_MODEL_LEN
GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION
MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS
MAX_NUM_SEQS=$MAX_NUM_SEQS
LOG_PROB_MICRO_BATCH_SIZE=$LOG_PROB_MICRO_BATCH_SIZE
CALCULATE_LOG_PROBS=$CALCULATE_LOG_PROBS
AGENT_LOOP_WORKERS=$AGENT_LOOP_WORKERS
RAY_NUM_CPUS=$RAY_NUM_CPUS
TRAJECTORY_SAMPLES_PER_TASK=$TRAJECTORY_SAMPLES_PER_TASK
TRAJECTORY_SAMPLE_SEED=$TRAJECTORY_SAMPLE_SEED
RAY_TMPDIR=$RAY_TMPDIR
EOF

printf 'Evaluating Qwen3.5-9B on %s fixed balanced ALFWorld tasks.\n' "$EVAL_TASK_COUNT"
printf 'Protocol: %sx%s, reflection=%s, validation seed=%s, rollout seed=%s.\n' \
    "$NUM_ATTEMPTS" "$MAX_TURNS_PER_ATTEMPT" "$DO_REFLECTION" \
    "$VAL_ENV_SEED" "$ROLLOUT_SEED"
printf 'Response controls: brief instruction=%s, response cap=%s tokens.\n' \
    "$BRIEF_RESPONSE_INSTRUCTION" "$MAX_RESPONSE_LENGTH"
if ((TRAJECTORY_SAMPLES_PER_TASK > 0)); then
    printf 'Writing %s Markdown trajectory sample(s) per task type under %s/validation_traces.\n' \
        "$TRAJECTORY_SAMPLES_PER_TASK" "$EVAL_DIR"
fi

started=$(date +%s)
set +e
cd "$VERL_DIR"
"$PYTHON_BIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    "data.train_files=$EVAL_DIR/data/train.parquet" \
    "data.val_files=$EVAL_DIR/data/test.parquet" \
    data.train_batch_size=1 \
    "data.val_batch_size=$EVAL_TASK_COUNT" \
    "data.max_prompt_length=$MAX_PROMPT_LENGTH" \
    "data.max_response_length=$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=1 \
    data.dataloader_num_workers=0 \
    data.truncation=error \
    data.return_raw_chat=True \
    data.return_multi_modal_inputs=False \
    data.shuffle=False \
    "data.seed=$ROLLOUT_SEED" \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    "actor_rollout_ref.model.path=$MODEL_PATH" \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.data_parallel_size=1 \
    actor_rollout_ref.rollout.n=1 \
    "actor_rollout_ref.rollout.seed=$ROLLOUT_SEED" \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.load_format=dummy \
    actor_rollout_ref.rollout.layered_summon=False \
    "actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    "actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN" \
    "actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS" \
    "actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_SEQS" \
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE" \
    "actor_rollout_ref.rollout.calculate_log_probs=$CALCULATE_LOG_PROBS" \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    "actor_rollout_ref.rollout.val_kwargs.temperature=$EVAL_TEMPERATURE" \
    "actor_rollout_ref.rollout.val_kwargs.top_p=$EVAL_TOP_P" \
    "actor_rollout_ref.rollout.val_kwargs.top_k=$EVAL_TOP_K" \
    actor_rollout_ref.rollout.agent.default_agent_loop=alfworld_agent \
    "actor_rollout_ref.rollout.agent.agent_loop_config_path=$REPO_ROOT/examples/alfworld/config/agent_loop_qwen35.yaml" \
    "actor_rollout_ref.rollout.agent.num_workers=$AGENT_LOOP_WORKERS" \
    reward.num_workers=1 \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=sync \
    trainer.critic_warmup=0 \
    'trainer.logger=[console]' \
    trainer.project_name=lamer \
    trainer.experiment_name=alfworld_base_qwen35_reflexion \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.log_val_generations=0 \
    "trainer.validation_data_dir=$EVAL_DIR/validation" \
    "trainer.default_local_dir=$EVAL_DIR/checkpoints" \
    trainer.resume_mode=disable \
    +env.env_name=alfworld/AlfredTWEnv \
    "+env.seed=$ENV_SEED" \
    "+env.val_seed=$VAL_ENV_SEED" \
    +env.num_attempts=1 \
    "+env.val_num_attempts=$NUM_ATTEMPTS" \
    +env.do_reflection=False \
    "+env.val_do_reflection=$DO_REFLECTION" \
    "+env.max_turns=$MAX_TURNS_PER_ATTEMPT" \
    "+env.reflection_type=$REFLECTION_TYPE" \
    "+env.presence_penalty=$PRESENCE_PENALTY" \
    "+env.brief_response_instruction=$BRIEF_RESPONSE_INSTRUCTION" \
    "+env.alfworld.eval_dataset=$EVAL_DATASET" \
    "+env.validation_trace_dir=$EVAL_DIR/validation_traces" \
    "+env.validation_trajectory_samples_per_task=$TRAJECTORY_SAMPLES_PER_TASK" \
    "+env.validation_trajectory_sample_seed=$TRAJECTORY_SAMPLE_SEED" \
    "+env.validation_task_count=$EVAL_TASK_COUNT" \
    "ray_kwargs.ray_init.num_cpus=$RAY_NUM_CPUS" \
    "ray_kwargs.ray_init.runtime_env.py_executable=$PYTHON_BIN" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=$PYTHONPATH" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.ALFWORLD_DATA=$ALFWORLD_DATA" \
    "hydra.run.dir=$EVAL_DIR/hydra" \
    "$@" \
    2>&1 | tee "$EVAL_DIR/eval.log"
status=${PIPESTATUS[0]}
set -e
finished=$(date +%s)
printf '%s\n' "$((finished - started))" > "$EVAL_DIR/elapsed_seconds.txt"

if ((status != 0)); then
    printf 'Evaluation failed; see %s/eval.log\n' "$EVAL_DIR" >&2
    exit "$status"
fi
printf 'Evaluation complete: %s\n' "$EVAL_DIR"
if ((TRAJECTORY_SAMPLES_PER_TASK > 0)); then
    printf 'Markdown trajectory samples: %s/validation_traces/*.md\n' "$EVAL_DIR"
fi
