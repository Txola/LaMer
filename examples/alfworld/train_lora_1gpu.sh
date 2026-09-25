#!/usr/bin/env bash
# Single-GPU LoRA Meta-RL training for ALFWorld on a 24 GB GPU.
# Each optimization step uses eight tasks with eight rollouts per task, matching
# the released full-model launcher's task and rollout batch structure.
#
# Main run:
#   bash examples/alfworld/train_lora_1gpu.sh
# Fast end-to-end check before a long run:
#   SMOKE_TEST=1 bash examples/alfworld/train_lora_1gpu.sh
# Resume the latest complete checkpoint in an existing run:
#   OUTPUT_DIR=/absolute/path/to/run bash examples/alfworld/train_lora_1gpu.sh
set -euo pipefail

if [[ ${PYTORCH_CUDA_ALLOC_CONF:-} == *"expandable_segments:True"* ]]; then
    unset PYTORCH_CUDA_ALLOC_CONF
fi

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%S)}
SMOKE_TEST=${SMOKE_TEST:-0}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/outputs/alfworld_lora_1gpu/${RUN_ID}"}

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

if [[ "$SMOKE_TEST" == "1" ]]; then
    default_total_steps=2
    default_test_freq=1
    default_val_task_count=4
    default_capture_generation_diagnostics=True
else
    default_total_steps=150
    default_test_freq=10
    default_val_task_count=84
    default_capture_generation_diagnostics=False
fi

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$(saved_parameter TRAIN_BATCH_SIZE 8)}
GROUP_SIZE=${GROUP_SIZE:-$(saved_parameter GROUP_SIZE 8)}
TOTAL_STEPS=${TOTAL_STEPS:-$(saved_parameter TOTAL_STEPS "$default_total_steps")}
TEST_FREQ=${TEST_FREQ:-$(saved_parameter TEST_FREQ "$default_test_freq")}
SAVE_FREQ=$TEST_FREQ
VAL_TASK_COUNT=${VAL_TASK_COUNT:-$(saved_parameter VAL_TASK_COUNT "$default_val_task_count")}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-$(saved_parameter MICRO_BATCH_SIZE 1)}
LEARNING_RATE=${LEARNING_RATE:-$(saved_parameter LEARNING_RATE 1e-6)}
LORA_RANK=${LORA_RANK:-$(saved_parameter LORA_RANK 32)}
LORA_ALPHA=${LORA_ALPHA:-$(saved_parameter LORA_ALPHA 64)}
MODEL_PATH=${MODEL_PATH:-$(saved_parameter MODEL_PATH Qwen/Qwen3-4B)}
NUM_ATTEMPTS=${NUM_ATTEMPTS:-$(saved_parameter NUM_ATTEMPTS 3)}
MAX_TURNS_PER_ATTEMPT=${MAX_TURNS_PER_ATTEMPT:-$(saved_parameter MAX_TURNS_PER_ATTEMPT 10)}
ENV_SEED=${ENV_SEED:-$(saved_parameter ENV_SEED 0)}
VAL_ENV_SEED=${VAL_ENV_SEED:-$(saved_parameter VAL_ENV_SEED 1000)}
ROLLOUT_SEED=${ROLLOUT_SEED:-$(saved_parameter ROLLOUT_SEED 20)}
CAPTURE_GENERATION_DIAGNOSTICS=${CAPTURE_GENERATION_DIAGNOSTICS:-$(saved_parameter CAPTURE_GENERATION_DIAGNOSTICS "$default_capture_generation_diagnostics")}
RESUME_MODE=${RESUME_MODE:-auto}
TRAINER_LOGGER=${TRAINER_LOGGER:-'[console,wandb]'}
PROJECT_NAME=${PROJECT_NAME:-$(saved_parameter PROJECT_NAME lamer)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-$(saved_parameter EXPERIMENT_NAME "alfworld_lora_1gpu_${RUN_ID}")}
WANDB_RUN_ID=${WANDB_RUN_ID:-$(saved_parameter WANDB_RUN_ID "$RUN_ID")}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
ENV_CPUS_PER_WORKER=${ENV_CPUS_PER_WORKER:-0.1}
GAMES_PER_ENV_WORKER=${GAMES_PER_ENV_WORKER:-32}

case "$MICRO_BATCH_SIZE" in
    1|2|4|8|16|32|64) ;;
    *) printf 'MICRO_BATCH_SIZE must divide PPO minibatch size 64.\n' >&2; exit 2 ;;
esac
if (( TRAIN_BATCH_SIZE < 1 || GROUP_SIZE < 2 || TOTAL_STEPS < 1 || SAVE_FREQ < 1 || TEST_FREQ < 1 )); then
    printf 'TRAIN_BATCH_SIZE, TOTAL_STEPS, SAVE_FREQ, and TEST_FREQ must be positive; GROUP_SIZE must be at least two.\n' >&2
    exit 2
fi
if (( VAL_TASK_COUNT < 1 || VAL_TASK_COUNT > 84 )); then
    printf 'VAL_TASK_COUNT must be between 1 and 84 for eval_id_checkpoint; got %s\n' "$VAL_TASK_COUNT" >&2
    exit 2
fi
if [[ "$CAPTURE_GENERATION_DIAGNOSTICS" != "True" && "$CAPTURE_GENERATION_DIAGNOSTICS" != "False" ]]; then
    printf 'CAPTURE_GENERATION_DIAGNOSTICS must be True or False; got %s\n' "$CAPTURE_GENERATION_DIAGNOSTICS" >&2
    exit 2
fi
if [[ ! -f "$ALFWORLD_DATA/json/split_manifest.json" || ! -d "$ALFWORLD_DATA/json/valid_id_task_balanced84" ]]; then
    printf 'Missing the balanced ALFWorld ID checkpoint split under %s/json. Rebuild the generated splits with:\n' "$ALFWORLD_DATA" >&2
    printf '  python scripts/prepare_alfworld_lamer_splits.py --replace\n' >&2
    exit 2
fi

export ALFWORLD_DATA PYTHONHASHSEED="$ROLLOUT_SEED"
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}
if [[ "$TRAINER_LOGGER" == *wandb* ]]; then
    export WANDB_RUN_ID WANDB_RESUME=${WANDB_RESUME:-allow}
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd -- "$OUTPUT_DIR" && pwd)
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-"$OUTPUT_DIR/tensorboard"}
mkdir -p "$CHECKPOINT_DIR" "$TENSORBOARD_DIR" "$OUTPUT_DIR/data"

LATEST_CHECKPOINT_STEP=0
if [[ -f "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt" ]]; then
    LATEST_CHECKPOINT_STEP=$(<"$CHECKPOINT_DIR/latest_checkpointed_iteration.txt")
    if [[ ! "$LATEST_CHECKPOINT_STEP" =~ ^[0-9]+$ ]]; then
        printf 'Invalid checkpoint tracker: %s\n' "$LATEST_CHECKPOINT_STEP" >&2
        exit 2
    fi
fi
# VERL restores model/optimizer/data RNG state but not the external ALFWorld
# process state. Offset only the training seed after a manual restart. The
# validation seed remains fixed across every checkpoint and resume.
TRAIN_ENV_SEED=${TRAIN_ENV_SEED:-$((ENV_SEED + LATEST_CHECKPOINT_STEP))}

python3 -c 'import alfworld, ray, textworld, torch, vllm; assert torch.cuda.is_available(), "CUDA is not available in this Python environment"'
if [[ "$TRAINER_LOGGER" == *wandb* ]]; then
    python3 -c 'import wandb'
    if [[ ${WANDB_MODE:-online} != "offline" && ${WANDB_MODE:-online} != "disabled" ]]; then
        python3 -c 'import sys, wandb; sys.exit(0 if wandb.api.api_key else "W&B is enabled but no API key is configured. Run: wandb login --verify")'
    fi
fi
if [[ "$TRAINER_LOGGER" == *tensorboard* ]]; then
    python3 -c 'import tensorboard'
fi

cp -- "${BASH_SOURCE[0]}" "$OUTPUT_DIR/launcher.sh"
cp -- "$ALFWORLD_DATA/json/split_manifest.json" "$OUTPUT_DIR/split_manifest.json"
git rev-parse HEAD > "$OUTPUT_DIR/git_commit.txt"
git status --short > "$OUTPUT_DIR/git_status.txt"
git diff --no-ext-diff > "$OUTPUT_DIR/working_tree.patch"
python --version > "$OUTPUT_DIR/python_version.txt" 2>&1
nvidia-smi > "$OUTPUT_DIR/nvidia_smi.txt" 2>&1 || true

{
    printf 'TRAIN_BATCH_SIZE=%s\n' "$TRAIN_BATCH_SIZE"
    printf 'GROUP_SIZE=%s\n' "$GROUP_SIZE"
    printf 'TOTAL_STEPS=%s\n' "$TOTAL_STEPS"
    printf 'SAVE_FREQ=%s\n' "$SAVE_FREQ"
    printf 'TEST_FREQ=%s\n' "$TEST_FREQ"
    printf 'VAL_TASK_COUNT=%s\n' "$VAL_TASK_COUNT"
    printf 'MICRO_BATCH_SIZE=%s\n' "$MICRO_BATCH_SIZE"
    printf 'LEARNING_RATE=%s\n' "$LEARNING_RATE"
    printf 'LORA_RANK=%s\n' "$LORA_RANK"
    printf 'LORA_ALPHA=%s\n' "$LORA_ALPHA"
    printf 'MODEL_PATH=%s\n' "$MODEL_PATH"
    printf 'NUM_ATTEMPTS=%s\n' "$NUM_ATTEMPTS"
    printf 'MAX_TURNS_PER_ATTEMPT=%s\n' "$MAX_TURNS_PER_ATTEMPT"
    printf 'ENV_SEED=%s\n' "$ENV_SEED"
    printf 'TRAIN_ENV_SEED=%s\n' "$TRAIN_ENV_SEED"
    printf 'VAL_ENV_SEED=%s\n' "$VAL_ENV_SEED"
    printf 'ROLLOUT_SEED=%s\n' "$ROLLOUT_SEED"
    printf 'CAPTURE_GENERATION_DIAGNOSTICS=%s\n' "$CAPTURE_GENERATION_DIAGNOSTICS"
    printf 'RESUME_MODE=%s\n' "$RESUME_MODE"
    printf 'TRAINER_LOGGER=%s\n' "$TRAINER_LOGGER"
    printf 'PROJECT_NAME=%s\n' "$PROJECT_NAME"
    printf 'EXPERIMENT_NAME=%s\n' "$EXPERIMENT_NAME"
    printf 'WANDB_RUN_ID=%s\n' "$WANDB_RUN_ID"
    printf 'LATEST_CHECKPOINT_STEP=%s\n' "$LATEST_CHECKPOINT_STEP"
    printf 'OUTPUT_DIR=%s\n' "$OUTPUT_DIR"
    printf 'CHECKPOINT_DIR=%s\n' "$CHECKPOINT_DIR"
    printf 'TENSORBOARD_DIR=%s\n' "$TENSORBOARD_DIR"
} > "$OUTPUT_DIR/run_parameters.txt"

if [[ ! -f "$OUTPUT_DIR/data/text/train.parquet" || ! -f "$OUTPUT_DIR/data/text/test.parquet" ]]; then
    python3 -m examples.data_preprocess.prepare \
        --mode text \
        --local_dir "$OUTPUT_DIR/data" \
        --train_data_size "$TRAIN_BATCH_SIZE" \
        --val_data_size "$VAL_TASK_COUNT" \
        2>&1 | tee "$OUTPUT_DIR/prepare.log"
fi

printf 'Training output: %s\n' "$OUTPUT_DIR"
printf 'Live dashboard: W&B project %s, run %s\n' "$PROJECT_NAME" "$EXPERIMENT_NAME"
if [[ "$TRAINER_LOGGER" == *tensorboard* ]]; then
    printf 'TensorBoard dashboard: tensorboard --logdir %q --port 6006\n' "$TENSORBOARD_DIR"
fi
printf 'Validation uses %s fixed ID games with seed %s; OOD tasks are excluded.\n' "$VAL_TASK_COUNT" "$VAL_ENV_SEED"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    "data.train_files=$OUTPUT_DIR/data/text/train.parquet" \
    "data.val_files=$OUTPUT_DIR/data/text/test.parquet" \
    "data.train_batch_size=$TRAIN_BATCH_SIZE" \
    "data.val_batch_size=$VAL_TASK_COUNT" \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    data.shuffle=False \
    "+data.seed=$ROLLOUT_SEED" \
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
    "+actor_rollout_ref.rollout.seed=$ROLLOUT_SEED" \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    "actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    "+actor_rollout_ref.rollout.capture_generation_diagnostics=$CAPTURE_GENERATION_DIAGNOSTICS" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    "+actor_rollout_ref.rollout.val_kwargs.seed=$ROLLOUT_SEED" \
    "actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
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
    env.env_name=alfworld/AlfredTWEnv \
    "env.seed=$TRAIN_ENV_SEED" \
    "+env.val_seed=$VAL_ENV_SEED" \
    "env.rollout.n=$GROUP_SIZE" \
    "env.num_attempts=$NUM_ATTEMPTS" \
    +env.do_reflection=True \
    env.max_steps=30 \
    "env.max_turns=$MAX_TURNS_PER_ATTEMPT" \
    +env.reflection_type=reflection_only \
    env.alfworld.eval_dataset=eval_id_checkpoint \
    "env.alfworld.games_per_worker=$GAMES_PER_ENV_WORKER" \
    "env.resources_per_worker.num_cpus=$ENV_CPUS_PER_WORKER" \
    env.resources_per_worker.num_gpus=0 \
    trainer.critic_warmup=0 \
    "trainer.logger=$TRAINER_LOGGER" \
    "trainer.project_name=$PROJECT_NAME" \
    "trainer.experiment_name=$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    "trainer.save_freq=$SAVE_FREQ" \
    "trainer.test_freq=$TEST_FREQ" \
    "trainer.total_epochs=$TOTAL_STEPS" \
    "trainer.total_training_steps=$TOTAL_STEPS" \
    "trainer.resume_mode=$RESUME_MODE" \
    trainer.val_before_train=False \
    trainer.log_val_generations=0 \
    trainer.max_actor_ckpt_to_keep=null \
    trainer.max_critic_ckpt_to_keep=null \
    trainer.validation_dump_all_interactions=False \
    trainer.validation_trajectory_samples_per_task=1 \
    trainer.validation_trajectory_sample_seed=0 \
    "trainer.validation_data_dir=$OUTPUT_DIR/validation_diagnostics" \
    trainer.grouping_diagnostics.enabled=False \
    "trainer.default_local_dir=$CHECKPOINT_DIR" \
    "hydra.run.dir=$OUTPUT_DIR/hydra" \
    2>&1 | tee -a "$OUTPUT_DIR/train.log"

printf '\nTraining directory: %s\n' "$OUTPUT_DIR"
printf 'Checkpoints: %s\n' "$CHECKPOINT_DIR"
printf 'Validation diagnostics: %s\n' "$OUTPUT_DIR/validation_diagnostics"
