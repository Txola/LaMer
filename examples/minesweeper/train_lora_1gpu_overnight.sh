#!/usr/bin/env bash
# Practical single-GPU LaMer training run for one 24 GB GPU.
#
# Defaults to one Minesweeper task per outer step, eight grouped trials per
# task, and 80 steps/epochs. Resumable LoRA-only checkpoints are saved every
# five steps and none are deleted. The frozen base model is reloaded from
# MODEL_PATH when resuming instead of being duplicated in every checkpoint.
#
# Start a new run:
#   bash examples/minesweeper/train_lora_1gpu_overnight.sh
# Resume an existing run from its latest complete checkpoint:
#   OUTPUT_DIR=/absolute/path/to/run bash examples/minesweeper/train_lora_1gpu_overnight.sh
set -euo pipefail

# vLLM's sleep-mode memory pool is incompatible with PyTorch expandable
# segments. Remove that allocator option if it was inherited from the shell.
if [[ ${PYTORCH_CUDA_ALLOC_CONF:-} == *"expandable_segments:True"* ]]; then
    unset PYTORCH_CUDA_ALLOC_CONF
fi

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/outputs/minesweeper_lora_1gpu/batch_${TRAIN_BATCH_SIZE:-1}_group_${GROUP_SIZE:-8}_${RUN_ID}"}

# Reuse the original run's settings when only OUTPUT_DIR is supplied for a
# resume. Explicit environment variables still take precedence.
saved_parameter() {
    local key=$1
    local fallback=$2
    local parameters_file="$OUTPUT_DIR/run_parameters.txt"
    local value=""
    if [[ -f "$parameters_file" ]]; then
        value=$(sed -n "s/^${key}=//p" "$parameters_file" | tail -1)
    fi
    printf '%s' "${value:-$fallback}"
}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$(saved_parameter TRAIN_BATCH_SIZE 1)}
GROUP_SIZE=${GROUP_SIZE:-$(saved_parameter GROUP_SIZE 8)}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-$(saved_parameter TOTAL_EPOCHS 80)}
SAVE_FREQ=${SAVE_FREQ:-$(saved_parameter SAVE_FREQ 5)}
VAL_TASK_COUNT=${VAL_TASK_COUNT:-$(saved_parameter VAL_TASK_COUNT 1)}
# Use the stable value of two by default even if an older run recorded four;
# four failed on a long Minesweeper microbatch after seven successful steps.
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
# Match the released full-model launcher.  The earlier 1e-5 LoRA run showed
# rapidly growing gradient norms and non-finite gradients from step 26 onward.
LEARNING_RATE=${LEARNING_RATE:-$(saved_parameter LEARNING_RATE 1e-6)}
LORA_RANK=${LORA_RANK:-$(saved_parameter LORA_RANK 32)}
LORA_ALPHA=${LORA_ALPHA:-$(saved_parameter LORA_ALPHA 64)}
MODEL_PATH=${MODEL_PATH:-$(saved_parameter MODEL_PATH Qwen/Qwen3-4B)}
RESUME_MODE=${RESUME_MODE:-auto}

case "$MICRO_BATCH_SIZE" in
    1|2|4|8|16|32|64) ;;
    *) printf 'MICRO_BATCH_SIZE must divide PPO minibatch size 64.\n' >&2; exit 2 ;;
esac

if (( TOTAL_EPOCHS < 1 || SAVE_FREQ < 1 || TRAIN_BATCH_SIZE < 1 || GROUP_SIZE < 1 )); then
    printf 'TRAIN_BATCH_SIZE, GROUP_SIZE, TOTAL_EPOCHS, and SAVE_FREQ must be positive.\n' >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd -- "$OUTPUT_DIR" && pwd)
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

LATEST_CHECKPOINT_STEP=0
if [[ -f "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt" ]]; then
    LATEST_CHECKPOINT_STEP=$(<"$CHECKPOINT_DIR/latest_checkpointed_iteration.txt")
    if [[ ! "$LATEST_CHECKPOINT_STEP" =~ ^[0-9]+$ ]]; then
        printf 'Invalid checkpoint tracker: %s\n' "$LATEST_CHECKPOINT_STEP" >&2
        exit 2
    fi
fi
# The environment RNG is not part of VERL's trainer checkpoint. Offset its
# seed after a manual restart so a resumed run does not replay the first board
# sequence from seed zero.
ENV_SEED=${ENV_SEED:-$LATEST_CHECKPOINT_STEP}

# Preserve enough information to identify the exact run later.
cp -- "${BASH_SOURCE[0]}" "$OUTPUT_DIR/launcher.sh"
git rev-parse HEAD > "$OUTPUT_DIR/git_commit.txt"
git status --short > "$OUTPUT_DIR/git_status.txt"
git diff --no-ext-diff > "$OUTPUT_DIR/working_tree.patch"
python --version > "$OUTPUT_DIR/python_version.txt" 2>&1
nvidia-smi > "$OUTPUT_DIR/nvidia_smi.txt"

cat > "$OUTPUT_DIR/run_parameters.txt" <<EOF
TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE
GROUP_SIZE=$GROUP_SIZE
TOTAL_EPOCHS=$TOTAL_EPOCHS
SAVE_FREQ=$SAVE_FREQ
VAL_TASK_COUNT=$VAL_TASK_COUNT
MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE
LEARNING_RATE=$LEARNING_RATE
LORA_RANK=$LORA_RANK
LORA_ALPHA=$LORA_ALPHA
MODEL_PATH=$MODEL_PATH
RESUME_MODE=$RESUME_MODE
ENV_SEED=$ENV_SEED
LATEST_CHECKPOINT_STEP=$LATEST_CHECKPOINT_STEP
OUTPUT_DIR=$OUTPUT_DIR
CHECKPOINT_DIR=$CHECKPOINT_DIR
EOF

python3 -m examples.data_preprocess.prepare \
    --mode text \
    --local_dir "$OUTPUT_DIR/data" \
    --train_data_size "$TRAIN_BATCH_SIZE" \
    --val_data_size "$VAL_TASK_COUNT" \
    2>&1 | tee "$OUTPUT_DIR/prepare.log"

# The base model is frozen in BF16. PEFT keeps the trainable LoRA adapters in
# FP32, and vLLM preloads the base checkpoint then receives adapter updates.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    "data.train_files=$OUTPUT_DIR/data/text/train.parquet" \
    "data.val_files=$OUTPUT_DIR/data/text/test.parquet" \
    "data.train_batch_size=$TRAIN_BATCH_SIZE" \
    "data.val_batch_size=$VAL_TASK_COUNT" \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    "actor_rollout_ref.model.path=$MODEL_PATH" \
    +actor_rollout_ref.model.enable_thinking=False \
    "actor_rollout_ref.model.lora_rank=$LORA_RANK" \
    "actor_rollout_ref.model.lora_alpha=$LORA_ALPHA" \
    actor_rollout_ref.model.target_modules=all-linear \
    "actor_rollout_ref.actor.optim.lr=$LEARNING_RATE" \
    actor_rollout_ref.actor.checkpoint.save_lora_only=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.val_kwargs.seed=20 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.5 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    +algorithm.step_gamma=0.95 \
    +algorithm.traj_gamma=0.6 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=mean_norm \
    reward_model.reward_manager=episode \
    env.env_name=Minesweeper \
    "env.seed=$ENV_SEED" \
    "env.rollout.n=$GROUP_SIZE" \
    env.minesweeper.board_size=6 \
    env.minesweeper.n_mines=3 \
    env.minesweeper.board_type=board \
    env.minesweeper.mode=text \
    +env.minesweeper.execution_backend=local \
    env.num_attempts=3 \
    env.max_steps=15 \
    env.max_turns=7 \
    +env.reflection_type=reflection_only \
    trainer.critic_warmup=0 \
    'trainer.logger=[console]' \
    trainer.project_name=lamer \
    trainer.experiment_name=minesweeper_lora_batch1_group8 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    "trainer.save_freq=$SAVE_FREQ" \
    trainer.test_freq=-1 \
    "trainer.total_epochs=$TOTAL_EPOCHS" \
    "trainer.total_training_steps=$TOTAL_EPOCHS" \
    "trainer.resume_mode=$RESUME_MODE" \
    trainer.val_before_train=False \
    trainer.max_actor_ckpt_to_keep=null \
    trainer.max_critic_ckpt_to_keep=null \
    trainer.grouping_diagnostics.enabled=False \
    "trainer.default_local_dir=$CHECKPOINT_DIR" \
    "hydra.run.dir=$OUTPUT_DIR/hydra" \
    2>&1 | tee -a "$OUTPUT_DIR/train.log"

printf '\nTraining directory: %s\n' "$OUTPUT_DIR"
printf 'Checkpoints: %s\n' "$CHECKPOINT_DIR"
