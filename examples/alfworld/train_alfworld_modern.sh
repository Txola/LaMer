#!/usr/bin/env bash
# Shared modern-VERL GiGPO launcher for audited Qwen models.

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
COMPAT_DIR=${COMPAT_DIR:-$REPO_ROOT/.compat}
VERL_DIR=${VERL_DIR:-$COMPAT_DIR/verl-upstream}
PYTHON_BIN=$VERL_DIR/.venv/bin/python
VERL_REVISION=fbb4b3a8bf636f290c9c59fc346f756849e9c241

MODEL_PROFILE=${MODEL_PROFILE:-qwen35_9b}
case "$MODEL_PROFILE" in
    qwen3_4b)
        MODEL_DISPLAY_NAME=Qwen3-4B
        MODEL_REVISION=1cfa9a7208912126459214e8b04321603b3df60c
        COMPAT_MODEL_SNAPSHOT=$COMPAT_DIR/hf-cache/models--Qwen--Qwen3-4B/snapshots/$MODEL_REVISION
        EXISTING_MODEL_SNAPSHOT=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/$MODEL_REVISION
        if [[ -d "$COMPAT_MODEL_SNAPSHOT" ]]; then
            MODEL_SNAPSHOT=$COMPAT_MODEL_SNAPSHOT
        else
            MODEL_SNAPSHOT=$EXISTING_MODEL_SNAPSHOT
        fi
        OUTPUT_NAMESPACE=qwen3_4b_modern
        DEFAULT_PRESENCE_PENALTY=0.0
        PROFILE_LORA_TARGET_MODULES='^model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$'
        # Microbatch 4 reached 90.2/95.0 GiB before a 4.2-GiB backward
        # allocation on a long step.  Two preserves the effective PPO batch
        # through gradient accumulation while leaving activation headroom.
        PROFILE_ACTOR_MICRO_BATCH_SIZE=2
        PROFILE_LOG_PROB_MICRO_BATCH_SIZE=8
        PROFILE_GPU_MEMORY_UTILIZATION=0.35
        PROFILE_ENTROPY_COEFF=0.001
        PROFILE_LOSS_AGG_MODE=token-mean
        PROFILE_FULL_OPTIMIZER_OFFLOAD=False
        PROFILE_GRADIENT_CHECKPOINTING=True
        PROFILE_FULL_ACTOR_MODEL_DTYPE=fp32
        PROFILE_LORA_ACTOR_MODEL_DTYPE=bfloat16
        ;;
    qwen35_9b)
        MODEL_DISPLAY_NAME=Qwen3.5-9B
        MODEL_REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a
        MODEL_SNAPSHOT=$COMPAT_DIR/hf-cache/models--Qwen--Qwen3.5-9B/snapshots/$MODEL_REVISION
        OUTPUT_NAMESPACE=qwen35_9b
        DEFAULT_PRESENCE_PENALTY=1.5
        PROFILE_LORA_TARGET_MODULES='^model\.language_model\.layers\.\d+\.(?:linear_attn\.(?:in_proj_a|in_proj_b|in_proj_qkv|in_proj_z|out_proj)|self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$'
        PROFILE_ACTOR_MICRO_BATCH_SIZE=1
        PROFILE_LOG_PROB_MICRO_BATCH_SIZE=1
        PROFILE_GPU_MEMORY_UTILIZATION=0.35
        PROFILE_ENTROPY_COEFF=0.01
        PROFILE_LOSS_AGG_MODE=seq-mean-token-mean
        PROFILE_FULL_OPTIMIZER_OFFLOAD=True
        PROFILE_GRADIENT_CHECKPOINTING=True
        PROFILE_FULL_ACTOR_MODEL_DTYPE=bfloat16
        PROFILE_LORA_ACTOR_MODEL_DTYPE=bfloat16
        ;;
    *)
        printf 'MODEL_PROFILE must be qwen3_4b or qwen35_9b, got %s.\n' "$MODEL_PROFILE" >&2
        exit 2
        ;;
esac

RUN_MODE=${RUN_MODE:-smoke}
TRAINING_VARIANT=${TRAINING_VARIANT:-full}
DATASET_SOURCE_DIR=${DATASET_SOURCE_DIR:-}
MODEL_PATH=${MODEL_PATH:-$MODEL_SNAPSHOT}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/outputs/$OUTPUT_NAMESPACE/${TRAINING_VARIANT}_gigpo_${RUN_MODE}_${RUN_ID}}
ENTRYPOINT_SCRIPT=${ENTRYPOINT_SCRIPT:-${BASH_SOURCE[0]}}
TRAIN_ENV_SEED=${TRAIN_ENV_SEED:-0}
VAL_ENV_SEED=${VAL_ENV_SEED:-1000}
ROLLOUT_SEED=${ROLLOUT_SEED:-20}
GROUP_SIZE=${GROUP_SIZE:-8}
NUM_ATTEMPTS=${NUM_ATTEMPTS:-3}
MAX_TURNS_PER_ATTEMPT=${MAX_TURNS_PER_ATTEMPT:-10}
PROMPT_HISTORY_LENGTH=${PROMPT_HISTORY_LENGTH:-15}
PRESENCE_PENALTY=${PRESENCE_PENALTY:-$DEFAULT_PRESENCE_PENALTY}
BRIEF_RESPONSE_INSTRUCTION=${BRIEF_RESPONSE_INSTRUCTION:-False}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
ENTROPY_COEFF=${ENTROPY_COEFF:-$PROFILE_ENTROPY_COEFF}
CALCULATE_ENTROPY=${CALCULATE_ENTROPY:-True}
ENTROPY_CHECKPOINTING=${ENTROPY_CHECKPOINTING:-True}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-$PROFILE_GRADIENT_CHECKPOINTING}
USE_LIGER=${USE_LIGER:-False}
USE_FUSED_KERNELS=${USE_FUSED_KERNELS:-False}
CREDIT_MODE=${CREDIT_MODE:-original_lamer}
if [[ -z "${LOSS_AGG_MODE+x}" ]]; then
    if [[ "$CREDIT_MODE" == meta_trajectory ]]; then
        LOSS_AGG_MODE=token-sum
    else
        LOSS_AGG_MODE=$PROFILE_LOSS_AGG_MODE
    fi
fi
POLICY_LOSS_MODE=${POLICY_LOSS_MODE:-vanilla}
ALGORITHM_GAMMA=${ALGORITHM_GAMMA:-0.95}
STEP_GAMMA=${STEP_GAMMA:-0.95}
TRAJ_GAMMA=${TRAJ_GAMMA:-0.6}
META_TAU_U=${META_TAU_U:-null}
META_INVALID_PENALTY_COEFFICIENT=${META_INVALID_PENALTY_COEFFICIENT:-0.5}
META_OVERLONG_MODE=${META_OVERLONG_MODE:-soft_shaping}
META_OVERLONG_MAX_COEFFICIENT=${META_OVERLONG_MAX_COEFFICIENT:-0.5}
META_PLAY_CACHE_TOKENS=${META_PLAY_CACHE_TOKENS:-null}
META_REFLECTION_CACHE_TOKENS=${META_REFLECTION_CACHE_TOKENS:-null}
RECOVER_TRUNCATED_REFLECTION_MEMORY=${RECOVER_TRUNCATED_REFLECTION_MEMORY:-False}
LAMER_META_PPO_MINIBATCH_UPDATE=${LAMER_META_PPO_MINIBATCH_UPDATE:-0}
LAMER_META_INTEGRATION_AUDIT_DIR=${LAMER_META_INTEGRATION_AUDIT_DIR:-}
STEP_ADVANTAGE_WEIGHT=${STEP_ADVANTAGE_WEIGHT:-1.0}
GIGPO_MODE=${GIGPO_MODE:-mean_norm}
FUTURE_AWARE_EPISODE_CREDIT=${FUTURE_AWARE_EPISODE_CREDIT:-False}
OPTIMIZER_NAME=${OPTIMIZER_NAME:-AdamW}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-constant}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-0}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
ADAM_BETA1=${ADAM_BETA1:-0.9}
ADAM_BETA2=${ADAM_BETA2:-0.999}
ADAM_EPS=${ADAM_EPS:-1e-8}
GRAD_CLIP=${GRAD_CLIP:-1.0}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.20}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.20}
TRAIN_TEMPERATURE=${TRAIN_TEMPERATURE:-1.0}
TRAIN_TOP_P=${TRAIN_TOP_P:-1.0}
TRAIN_TOP_K=${TRAIN_TOP_K:--1}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.7}
VAL_TOP_P=${VAL_TOP_P:-0.8}
VAL_TOP_K=${VAL_TOP_K:-20}
ACTOR_MICRO_BATCH_SIZE=${ACTOR_MICRO_BATCH_SIZE:-$PROFILE_ACTOR_MICRO_BATCH_SIZE}
ACTOR_USE_DYNAMIC_BSZ=${ACTOR_USE_DYNAMIC_BSZ:-False}
ACTOR_MAX_TOKENS_PER_GPU=${ACTOR_MAX_TOKENS_PER_GPU:-16384}
ACTOR_SHUFFLE=${ACTOR_SHUFFLE:-False}
LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-$PROFILE_LOG_PROB_MICRO_BATCH_SIZE}
OLD_LOGPROB_MAX_TOKENS_PER_GPU=${OLD_LOGPROB_MAX_TOKENS_PER_GPU:-$ACTOR_MAX_TOKENS_PER_GPU}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-$PROFILE_GPU_MEMORY_UTILIZATION}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-True}
ENFORCE_EAGER=${ENFORCE_EAGER:-True}
VLLM_DISABLE_LOG_STATS=${VLLM_DISABLE_LOG_STATS:-True}
ROLLOUT_LOGPROBS_MODE=${ROLLOUT_LOGPROBS_MODE:-processed_logprobs}
ROLLOUT_IS=${ROLLOUT_IS:-token}
ROLLOUT_IS_THRESHOLD=${ROLLOUT_IS_THRESHOLD:-2.0}
ENABLE_GPU_MONITOR=${ENABLE_GPU_MONITOR:-True}
PROFILE_ROLLOUT=${PROFILE_ROLLOUT:-False}
GPU_MONITOR_INTERVAL_SECONDS=${GPU_MONITOR_INTERVAL_SECONDS:-2}
ROLLOUT_TIMEOUT_SECONDS=${ROLLOUT_TIMEOUT_SECONDS:-1800}
ALFWORLD_ENV_POOL_SIZE=${ALFWORLD_ENV_POOL_SIZE:-}
TRANSFER_QUEUE_STORAGE_UNITS=${TRANSFER_QUEUE_STORAGE_UNITS:-2}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-18}
ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
RAY_TMPDIR=${RAY_TMPDIR:-/tmp/lamer-modern-train-$UID}
RESUME_MODE=${RESUME_MODE:-disable}
RESUME_FROM_PATH=${RESUME_FROM_PATH:-}
if [[ -z "${VAL_BEFORE_TRAIN+x}" ]]; then
    if [[ "$RESUME_MODE" == disable ]]; then
        VAL_BEFORE_TRAIN=True
    else
        # A retained checkpoint has already passed its scheduled validation.
        # Repeating stochastic validation at the same step wastes time and can
        # incorrectly replace the score recorded for that checkpoint.
        VAL_BEFORE_TRAIN=False
    fi
fi
DRY_RUN=${DRY_RUN:-0}

case "$TRAINING_VARIANT" in
    full)
        ACTOR_MODEL_DTYPE=${ACTOR_MODEL_DTYPE:-$PROFILE_FULL_ACTOR_MODEL_DTYPE}
        LORA_RANK=0
        LORA_ALPHA=16
        LORA_TARGET_MODULES=all-linear
        LEARNING_RATE=${LEARNING_RATE:-1e-6}
        FSDP_OFFLOAD_POLICY=${FSDP_OFFLOAD_POLICY:-False}
        ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
        # The 4B profile fits its optimizer on the 96-GiB GPU and avoids slow,
        # memory-constrained host offload. The larger profile retains its
        # conservative diagnostic default.
        ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-$PROFILE_FULL_OPTIMIZER_OFFLOAD}
        ROLLOUT_LOAD_FORMAT=dummy
        LAYERED_SUMMON=False
        SAVE_LORA_ONLY=False
        SAVE_BEST_VALIDATION=False
        SAVE_LAST_CHECKPOINT=False
        MAX_ACTOR_CKPT_TO_KEEP=1
        RELEASE_VLLM_HOST_CACHE_AFTER_WAKE=${RELEASE_VLLM_HOST_CACHE_AFTER_WAKE:-False}
        EXTERNAL_MODULES=agent_system.multi_turn_rollout.alfworld_gigpo_trainer
        ;;
    lora)
        ACTOR_MODEL_DTYPE=${ACTOR_MODEL_DTYPE:-$PROFILE_LORA_ACTOR_MODEL_DTYPE}
        LORA_RANK=${LORA_RANK:-16}
        LORA_ALPHA=${LORA_ALPHA:-32}
        LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-$PROFILE_LORA_TARGET_MODULES}
        LEARNING_RATE=${LEARNING_RATE:-1e-6}
        FSDP_OFFLOAD_POLICY=${FSDP_OFFLOAD_POLICY:-False}
        ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
        ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}
        ROLLOUT_LOAD_FORMAT=safetensors
        LAYERED_SUMMON=True
        SAVE_LORA_ONLY=True
        SAVE_BEST_VALIDATION=${SAVE_BEST_VALIDATION:-True}
        SAVE_LAST_CHECKPOINT=${SAVE_LAST_CHECKPOINT:-True}
        # Native rotation cannot preserve an older validation best while also
        # rotating the latest resumable checkpoint. The custom selector does both.
        MAX_ACTOR_CKPT_TO_KEEP=null
        RELEASE_VLLM_HOST_CACHE_AFTER_WAKE=${RELEASE_VLLM_HOST_CACHE_AFTER_WAKE:-True}
        EXTERNAL_MODULES=agent_system.multi_turn_rollout.alfworld_gigpo_trainer,agent_system.multi_turn_rollout.qwen_lora_precision
        ;;
    *)
        printf 'TRAINING_VARIANT must be full or lora, got %s.\n' "$TRAINING_VARIANT" >&2
        exit 2
        ;;
esac

case "$RUN_MODE" in
    smoke)
        TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
        PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-1}
        TOTAL_STEPS=${TOTAL_STEPS:-1}
        TEST_FREQ=${TEST_FREQ:-1}
        VAL_TASK_COUNT=${VAL_TASK_COUNT:-1}
        AGENT_LOOP_WORKERS=${AGENT_LOOP_WORKERS:-1}
        TRAINER_LOGGER=${TRAINER_LOGGER:-'[console,wandb]'}
        ;;
    pilot)
        TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
        # Modern VERL multiplies this value by rollout.n. 8 * 8 preserves the
        # old launcher's effective 64-record PPO mini-batch target.
        PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
        TOTAL_STEPS=${TOTAL_STEPS:-20}
        TEST_FREQ=${TEST_FREQ:-5}
        VAL_TASK_COUNT=${VAL_TASK_COUNT:-84}
        AGENT_LOOP_WORKERS=${AGENT_LOOP_WORKERS:-4}
        TRAINER_LOGGER=${TRAINER_LOGGER:-'[console,wandb]'}
        ;;
    full)
        TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
        PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
        TOTAL_STEPS=${TOTAL_STEPS:-150}
        TEST_FREQ=${TEST_FREQ:-5}
        VAL_TASK_COUNT=${VAL_TASK_COUNT:-84}
        AGENT_LOOP_WORKERS=${AGENT_LOOP_WORKERS:-4}
        TRAINER_LOGGER=${TRAINER_LOGGER:-'[console,wandb]'}
        ;;
    *)
        printf 'RUN_MODE must be smoke, pilot, or full, got %s.\n' "$RUN_MODE" >&2
        exit 2
        ;;
esac

# Full-model production runs use the same bounded policy as LoRA: retain the
# validation best plus the latest validated checkpoint for interruption-safe
# resume. The prior non-best latest checkpoint is removed after replacement.
if [[ "$RUN_MODE" == full && "$TRAINING_VARIANT" == full ]]; then
    SAVE_BEST_VALIDATION=True
    SAVE_LAST_CHECKPOINT=True
    MAX_ACTOR_CKPT_TO_KEEP=null
fi

# Match the busiest worker's training concurrency. Validation can queue above
# this bound instead of increasing the long-lived host-memory ceiling.
if [[ -z "$ALFWORLD_ENV_POOL_SIZE" ]]; then
    ALFWORLD_ENV_POOL_SIZE=$((
        ((TRAIN_BATCH_SIZE + AGENT_LOOP_WORKERS - 1) / AGENT_LOOP_WORKERS)
        * GROUP_SIZE
    ))
fi

SAVE_FREQ=${SAVE_FREQ:--1}
PROJECT_NAME=${PROJECT_NAME:-lamer}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-alfworld_${OUTPUT_NAMESPACE}_${TRAINING_VARIANT}_gigpo_${RUN_MODE}}
EVAL_DATASET=${EVAL_DATASET:-eval_all}
BEST_VALIDATION_METRIC=${BEST_VALIDATION_METRIC:-'val-aux/alfworld/success_rate[2]/mean@1'}

if [[ "$SAVE_BEST_VALIDATION" == True && "$SAVE_FREQ" -gt 0 ]]; then
    printf 'Best/rolling-last selection uses validation cadence and requires SAVE_FREQ<=0.\n' >&2
    exit 2
fi
if [[ "$ROLLOUT_IS" != token ]]; then
    printf 'Presence-penalized training requires ROLLOUT_IS=token, got %s.\n' \
        "$ROLLOUT_IS" >&2
    exit 2
fi
if [[ "$ROLLOUT_LOGPROBS_MODE" != processed_logprobs ]]; then
    printf 'Rollout correction requires ROLLOUT_LOGPROBS_MODE=processed_logprobs, got %s.\n' \
        "$ROLLOUT_LOGPROBS_MODE" >&2
    exit 2
fi
case "$LOSS_AGG_MODE" in
    token-mean|token-sum|seq-mean-token-mean) ;;
    *)
        printf 'LOSS_AGG_MODE must be token-mean, token-sum, or seq-mean-token-mean, got %s.\n' \
            "$LOSS_AGG_MODE" >&2
        exit 2
        ;;
esac
case "$CREDIT_MODE" in
    original_lamer) ;;
    meta_trajectory|lamer_form)
        if [[ "$CREDIT_MODE" == lamer_form && "$LOSS_AGG_MODE" != token-mean ]]; then
            printf 'CREDIT_MODE=lamer_form requires LOSS_AGG_MODE=token-mean.\n' >&2
            exit 2
        fi
        if [[ "$LOSS_AGG_MODE" != token-sum && "$LOSS_AGG_MODE" != token-mean ]]; then
            printf 'Ablation credit requires LOSS_AGG_MODE=token-sum or token-mean.\n' >&2
            exit 2
        fi
        if [[ "$META_TAU_U" == null || "$META_PLAY_CACHE_TOKENS" == null || \
              "$META_REFLECTION_CACHE_TOKENS" == null ]]; then
            printf 'Meta mode requires explicit META_TAU_U, META_PLAY_CACHE_TOKENS, and META_REFLECTION_CACHE_TOKENS.\n' >&2
            exit 2
        fi
        if [[ "$LAMER_META_PPO_MINIBATCH_UPDATE" != 1 ]]; then
            printf 'Meta mode requires LAMER_META_PPO_MINIBATCH_UPDATE=1.\n' >&2
            exit 2
        fi
        if [[ "$POLICY_LOSS_MODE" != meta_vanilla ]]; then
            printf 'Meta mode requires POLICY_LOSS_MODE=meta_vanilla.\n' >&2
            exit 2
        fi
        ;;
    *)
        printf 'CREDIT_MODE must be original_lamer, lamer_form, or meta_trajectory, got %s.\n' \
            "$CREDIT_MODE" >&2
        exit 2
        ;;
esac
case "$META_OVERLONG_MODE" in
    soft_shaping|filter) ;;
    *)
        printf 'META_OVERLONG_MODE must be soft_shaping or filter, got %s.\n' \
            "$META_OVERLONG_MODE" >&2
        exit 2
        ;;
esac
case "$RECOVER_TRUNCATED_REFLECTION_MEMORY" in
    True|False) ;;
    *)
        printf 'RECOVER_TRUNCATED_REFLECTION_MEMORY must be True or False, got %s.\n' \
            "$RECOVER_TRUNCATED_REFLECTION_MEMORY" >&2
        exit 2
        ;;
esac
case "$ACTOR_USE_DYNAMIC_BSZ" in
    True|False) ;;
    *)
        printf 'ACTOR_USE_DYNAMIC_BSZ must be True or False, got %s.\n' \
            "$ACTOR_USE_DYNAMIC_BSZ" >&2
        exit 2
        ;;
esac
if [[ ! "$ACTOR_MAX_TOKENS_PER_GPU" =~ ^[1-9][0-9]*$ || \
      ! "$OLD_LOGPROB_MAX_TOKENS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Dynamic actor and old-logprob token budgets must be positive integers.\n' >&2
    exit 2
fi
if [[ "$ACTOR_USE_DYNAMIC_BSZ" == True ]]; then
    if (( ACTOR_MAX_TOKENS_PER_GPU < MAX_MODEL_LEN )); then
        printf 'ACTOR_MAX_TOKENS_PER_GPU=%s must cover one maximum-length sequence (%s).\n' \
            "$ACTOR_MAX_TOKENS_PER_GPU" "$MAX_MODEL_LEN" >&2
        exit 2
    fi
    if (( OLD_LOGPROB_MAX_TOKENS_PER_GPU < MAX_MODEL_LEN )); then
        printf 'OLD_LOGPROB_MAX_TOKENS_PER_GPU=%s must cover one maximum-length sequence (%s).\n' \
            "$OLD_LOGPROB_MAX_TOKENS_PER_GPU" "$MAX_MODEL_LEN" >&2
        exit 2
    fi
fi
case "$FUTURE_AWARE_EPISODE_CREDIT" in
    True|true) FUTURE_AWARE_EPISODE_CREDIT=True ;;
    False|false) FUTURE_AWARE_EPISODE_CREDIT=False ;;
    *)
        printf 'FUTURE_AWARE_EPISODE_CREDIT must be true or false, got %s.\n' \
            "$FUTURE_AWARE_EPISODE_CREDIT" >&2
        exit 2
        ;;
esac
case "$RESUME_MODE" in
    disable|auto|resume_path) ;;
    *)
        printf 'RESUME_MODE must be disable, auto, or resume_path, got %s.\n' \
            "$RESUME_MODE" >&2
        exit 2
        ;;
esac
if [[ "$RESUME_MODE" == resume_path && -z "$RESUME_FROM_PATH" ]]; then
    printf 'RESUME_FROM_PATH is required when RESUME_MODE=resume_path.\n' >&2
    exit 2
fi
if [[ ! "$GPU_MONITOR_INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    printf 'GPU_MONITOR_INTERVAL_SECONDS must be a positive integer, got %s.\n' \
        "$GPU_MONITOR_INTERVAL_SECONDS" >&2
    exit 2
fi
case "$BRIEF_RESPONSE_INSTRUCTION" in
    True|False) ;;
    *)
        printf 'BRIEF_RESPONSE_INSTRUCTION must be True or False, got %s.\n' \
            "$BRIEF_RESPONSE_INSTRUCTION" >&2
        exit 2
        ;;
esac
if [[ ! "$MAX_PROMPT_LENGTH" =~ ^[1-9][0-9]*$ ]] || \
   [[ ! "$MAX_RESPONSE_LENGTH" =~ ^[1-9][0-9]*$ ]] || \
   [[ ! "$MAX_MODEL_LEN" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Prompt, response, and model lengths must be positive integers.\n' >&2
    exit 2
fi
if [[ ! "$PROMPT_HISTORY_LENGTH" =~ ^[1-9][0-9]*$ ]]; then
    printf 'PROMPT_HISTORY_LENGTH must be a positive integer, got %s.\n' \
        "$PROMPT_HISTORY_LENGTH" >&2
    exit 2
fi
if ((MAX_MODEL_LEN < MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)); then
    printf 'MAX_MODEL_LEN=%s must be at least MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH=%s.\n' \
        "$MAX_MODEL_LEN" "$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" >&2
    exit 2
fi
if [[ ! "$ROLLOUT_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    printf 'ROLLOUT_TIMEOUT_SECONDS must be a positive integer, got %s.\n' \
        "$ROLLOUT_TIMEOUT_SECONDS" >&2
    exit 2
fi
if [[ ! "$ALFWORLD_ENV_POOL_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    printf 'ALFWORLD_ENV_POOL_SIZE must be a positive integer, got %s.\n' \
        "$ALFWORLD_ENV_POOL_SIZE" >&2
    exit 2
fi
if [[ ! "$TRANSFER_QUEUE_STORAGE_UNITS" =~ ^[1-9][0-9]*$ ]]; then
    printf 'TRANSFER_QUEUE_STORAGE_UNITS must be a positive integer, got %s.\n' \
        "$TRANSFER_QUEUE_STORAGE_UNITS" >&2
    exit 2
fi
case "$RELEASE_VLLM_HOST_CACHE_AFTER_WAKE" in
    True)
        VERL_RELEASE_VLLM_HOST_CACHE_AFTER_WAKE=1
        ;;
    False)
        VERL_RELEASE_VLLM_HOST_CACHE_AFTER_WAKE=0
        ;;
    *)
        printf 'RELEASE_VLLM_HOST_CACHE_AFTER_WAKE must be True or False, got %s.\n' \
            "$RELEASE_VLLM_HOST_CACHE_AFTER_WAKE" >&2
        exit 2
        ;;
esac

if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'Missing compatibility environment. Run examples/alfworld/setup_qwen35_compat.sh first.\n' >&2
    exit 2
fi
if [[ "$(git -C "$VERL_DIR" rev-parse HEAD)" != "$VERL_REVISION" ]]; then
    printf 'VERL checkout is not at the pinned revision %s.\n' "$VERL_REVISION" >&2
    exit 2
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    printf 'Missing pinned %s snapshot at %s.\n' "$MODEL_DISPLAY_NAME" "$MODEL_PATH" >&2
    exit 2
fi
if [[ ! -d "$ALFWORLD_DATA/json_2.1.1/train" ]]; then
    printf 'Missing prepared ALFWorld training data under %s.\n' "$ALFWORLD_DATA" >&2
    exit 2
fi
if [[ ! -d "$ALFWORLD_DATA/json/valid_task_balanced126" ]]; then
    printf 'Missing prepared six-category ALFWorld validation data under %s.\n' "$ALFWORLD_DATA" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR/data" "$RAY_TMPDIR"
OUTPUT_DIR=$(cd -- "$OUTPUT_DIR" && pwd)
if [[ -n "$DATASET_SOURCE_DIR" ]]; then
    DATASET_SOURCE_DIR=$(cd -- "$DATASET_SOURCE_DIR" && pwd)
    if [[ "$OUTPUT_DIR" == "$DATASET_SOURCE_DIR" || "$OUTPUT_DIR" == "$(dirname -- "$DATASET_SOURCE_DIR")" ]]; then
        printf 'Refusing to use the archived dataset directory as the output directory.\n' >&2
        exit 2
    fi
fi

RESOLVED_RESUME_PATH=
if [[ "$RESUME_MODE" == auto ]]; then
    CHECKPOINT_TRACKER=$OUTPUT_DIR/checkpoints/latest_checkpointed_iteration.txt
    if [[ ! -f "$CHECKPOINT_TRACKER" ]]; then
        printf 'RESUME_MODE=auto found no checkpoint tracker at %s.\n' \
            "$CHECKPOINT_TRACKER" >&2
        exit 2
    fi
    RESUME_ITERATION=$(<"$CHECKPOINT_TRACKER")
    if [[ ! "$RESUME_ITERATION" =~ ^[0-9]+$ ]]; then
        printf 'Invalid checkpoint iteration %q in %s.\n' \
            "$RESUME_ITERATION" "$CHECKPOINT_TRACKER" >&2
        exit 2
    fi
    RESOLVED_RESUME_PATH=$OUTPUT_DIR/checkpoints/global_step_$RESUME_ITERATION
    if [[ ! -d "$RESOLVED_RESUME_PATH" ]]; then
        printf 'Checkpoint tracker points to a missing directory: %s\n' \
            "$RESOLVED_RESUME_PATH" >&2
        exit 2
    fi
elif [[ "$RESUME_MODE" == resume_path ]]; then
    if [[ ! -d "$RESUME_FROM_PATH" ]]; then
        printf 'Resume checkpoint does not exist: %s\n' "$RESUME_FROM_PATH" >&2
        exit 2
    fi
    RESUME_FROM_PATH=$(cd -- "$RESUME_FROM_PATH" && pwd)
    if [[ ! "$(basename -- "$RESUME_FROM_PATH")" =~ ^global_step_[0-9]+$ ]]; then
        printf 'Resume checkpoint must be a global_step_<number> directory: %s\n' \
            "$RESUME_FROM_PATH" >&2
        exit 2
    fi
    RESOLVED_RESUME_PATH=$RESUME_FROM_PATH
fi

ARTIFACT_DIR=$OUTPUT_DIR
if [[ "$RESUME_MODE" != disable ]]; then
    RESUME_ATTEMPT_ID=$(date +%Y%m%d_%H%M%S)
    ARTIFACT_DIR=$OUTPUT_DIR/resume_attempts/$RESUME_ATTEMPT_ID
fi
mkdir -p "$ARTIFACT_DIR/hydra"

export ALFWORLD_DATA RAY_TMPDIR
export HF_HOME=$COMPAT_DIR/hf-cache
export UV_CACHE_DIR=$COMPAT_DIR/uv-cache
export PYTHONHASHSEED=$ROLLOUT_SEED
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONPATH="$VERL_DIR:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export VERL_USE_EXTERNAL_MODULES=$EXTERNAL_MODULES
export VERL_RELEASE_VLLM_HOST_CACHE_AFTER_WAKE

if [[ "$RESUME_MODE" == disable ]]; then
    if [[ -n "$DATASET_SOURCE_DIR" ]]; then
        for data_file in train.parquet test.parquet manifest.json; do
            if [[ ! -f "$DATASET_SOURCE_DIR/$data_file" ]]; then
                printf 'Missing archived dataset file: %s\n' "$DATASET_SOURCE_DIR/$data_file" >&2
                exit 2
            fi
            if [[ -e "$OUTPUT_DIR/data/$data_file" ]]; then
                printf 'Refusing to overwrite existing dataset file: %s\n' "$OUTPUT_DIR/data/$data_file" >&2
                exit 2
            fi
            cp -- "$DATASET_SOURCE_DIR/$data_file" "$OUTPUT_DIR/data/$data_file"
        done
        (cd -- "$OUTPUT_DIR/data" && sha256sum train.parquet test.parquet manifest.json) \
            > "$OUTPUT_DIR/data/source_sha256.txt"
    else
        "$PYTHON_BIN" "$REPO_ROOT/scripts/prepare_alfworld_agent_dataset.py" \
            --output-dir "$OUTPUT_DIR/data" \
            --eval-dataset "$EVAL_DATASET" \
            --validation-seed "$VAL_ENV_SEED" \
            --validation-size "$VAL_TASK_COUNT" \
            --training-seed "$TRAIN_ENV_SEED" \
            --training-steps "$TOTAL_STEPS" \
            --train-batch-size "$TRAIN_BATCH_SIZE"
    fi
elif [[ ! -f "$OUTPUT_DIR/data/train.parquet" || ! -f "$OUTPUT_DIR/data/test.parquet" ]]; then
    printf 'Resume requires the original train and validation datasets under %s/data.\n' \
        "$OUTPUT_DIR" >&2
    exit 2
fi

cp -- "${BASH_SOURCE[0]}" "$ARTIFACT_DIR/launcher.sh"
cp -- "$ENTRYPOINT_SCRIPT" "$ARTIFACT_DIR/entrypoint.sh"
printf 'env RUN_MODE=%q OUTPUT_DIR=%q RESUME_MODE=%q bash %q\n' \
    "$RUN_MODE" "$OUTPUT_DIR" "$RESUME_MODE" "$ENTRYPOINT_SCRIPT" \
    > "$ARTIFACT_DIR/launch_command.sh"
tar -C "$REPO_ROOT" -czf "$ARTIFACT_DIR/source_snapshot.tar.gz" \
    agent_system/meta_trajectory_credit.py \
    agent_system/meta_trajectory_production.py \
    agent_system/multi_turn_rollout/alfworld_gigpo_trainer.py \
    agent_system/multi_turn_rollout/alfworld_agent_loop.py \
    agent_system/multi_turn_rollout/episode_credit.py \
    agent_system/multi_turn_rollout/qwen_lora_precision.py \
    agent_system/environments/alfworld/envs.py \
    agent_system/environments/alfworld/env_manager.py \
    agent_system/environments/alfworld/reflection_recovery.py \
    agent_system/environments/alfworld/projection.py \
    agent_system/environments/alfworld/prompt.py \
    examples/alfworld/train_alfworld_modern.sh \
    examples/alfworld/config/agent_loop_qwen35.yaml \
    scripts/prepare_alfworld_agent_dataset.py
sha256sum "$ARTIFACT_DIR/source_snapshot.tar.gz" > "$ARTIFACT_DIR/source_snapshot.sha256"
git -C "$REPO_ROOT" rev-parse HEAD > "$ARTIFACT_DIR/code_revision.txt"
git -C "$REPO_ROOT" status --short > "$ARTIFACT_DIR/worktree_status.txt"
git -C "$REPO_ROOT" diff --no-ext-diff > "$ARTIFACT_DIR/working_tree.patch"
git -C "$VERL_DIR" rev-parse HEAD > "$ARTIFACT_DIR/verl_revision.txt"
"$PYTHON_BIN" -P -c \
    'import platform, torch, transformers, vllm, verl; print("python", platform.python_version()); print("torch", torch.__version__, "cuda", torch.version.cuda); print("transformers", transformers.__version__); print("vllm", vllm.__version__); print("verl", verl.__version__)' \
    > "$ARTIFACT_DIR/software_versions.txt"
nvidia-smi > "$ARTIFACT_DIR/gpu_info.txt" 2>&1 || true

cat > "$ARTIFACT_DIR/training_parameters.txt" <<EOF
MODEL_PROFILE=$MODEL_PROFILE
MODEL_DISPLAY_NAME=$MODEL_DISPLAY_NAME
RUN_MODE=$RUN_MODE
TRAINING_VARIANT=$TRAINING_VARIANT
MODEL_PATH=$MODEL_PATH
MODEL_REVISION=$MODEL_REVISION
LORA_RANK=$LORA_RANK
LORA_ALPHA=$LORA_ALPHA
LORA_TARGET_MODULES=$LORA_TARGET_MODULES
TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE
GROUP_SIZE=$GROUP_SIZE
PPO_MINI_BATCH_SIZE=$PPO_MINI_BATCH_SIZE
ACTOR_MICRO_BATCH_SIZE=$ACTOR_MICRO_BATCH_SIZE
ACTOR_USE_DYNAMIC_BSZ=$ACTOR_USE_DYNAMIC_BSZ
ACTOR_MAX_TOKENS_PER_GPU=$ACTOR_MAX_TOKENS_PER_GPU
OLD_LOGPROB_MAX_TOKENS_PER_GPU=$OLD_LOGPROB_MAX_TOKENS_PER_GPU
LOG_PROB_MICRO_BATCH_SIZE=$LOG_PROB_MICRO_BATCH_SIZE
TOTAL_STEPS=$TOTAL_STEPS
TEST_FREQ=$TEST_FREQ
SAVE_FREQ=$SAVE_FREQ
VAL_TASK_COUNT=$VAL_TASK_COUNT
VAL_BEFORE_TRAIN=$VAL_BEFORE_TRAIN
RESUME_MODE=$RESUME_MODE
RESUME_FROM_PATH=$RESUME_FROM_PATH
RESOLVED_RESUME_PATH=$RESOLVED_RESUME_PATH
EVAL_DATASET=$EVAL_DATASET
NUM_ATTEMPTS=$NUM_ATTEMPTS
MAX_TURNS_PER_ATTEMPT=$MAX_TURNS_PER_ATTEMPT
PROMPT_HISTORY_LENGTH=$PROMPT_HISTORY_LENGTH
PRESENCE_PENALTY=$PRESENCE_PENALTY
BRIEF_RESPONSE_INSTRUCTION=$BRIEF_RESPONSE_INSTRUCTION
MAX_PROMPT_LENGTH=$MAX_PROMPT_LENGTH
MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH
MAX_MODEL_LEN=$MAX_MODEL_LEN
LEARNING_RATE=$LEARNING_RATE
OPTIMIZER_NAME=$OPTIMIZER_NAME
LR_SCHEDULER_TYPE=$LR_SCHEDULER_TYPE
LR_WARMUP_STEPS=$LR_WARMUP_STEPS
WEIGHT_DECAY=$WEIGHT_DECAY
ADAM_BETAS=[$ADAM_BETA1,$ADAM_BETA2]
ADAM_EPS=$ADAM_EPS
GRAD_CLIP=$GRAD_CLIP
CLIP_RATIO_LOW=$CLIP_RATIO_LOW
CLIP_RATIO_HIGH=$CLIP_RATIO_HIGH
ENTROPY_COEFF=$ENTROPY_COEFF
CALCULATE_ENTROPY=$CALCULATE_ENTROPY
ENTROPY_CHECKPOINTING=$ENTROPY_CHECKPOINTING
GRADIENT_CHECKPOINTING=$GRADIENT_CHECKPOINTING
USE_LIGER=$USE_LIGER
USE_FUSED_KERNELS=$USE_FUSED_KERNELS
ACTOR_MODEL_DTYPE=$ACTOR_MODEL_DTYPE
LOSS_AGG_MODE=$LOSS_AGG_MODE
POLICY_LOSS_MODE=$POLICY_LOSS_MODE
CREDIT_MODE=$CREDIT_MODE
ALGORITHM_GAMMA=$ALGORITHM_GAMMA
STEP_GAMMA=$STEP_GAMMA
TRAJ_GAMMA=$TRAJ_GAMMA
META_TAU_U=$META_TAU_U
META_INVALID_PENALTY_COEFFICIENT=$META_INVALID_PENALTY_COEFFICIENT
META_OVERLONG_MODE=$META_OVERLONG_MODE
META_OVERLONG_MAX_COEFFICIENT=$META_OVERLONG_MAX_COEFFICIENT
META_PLAY_CACHE_TOKENS=$META_PLAY_CACHE_TOKENS
META_REFLECTION_CACHE_TOKENS=$META_REFLECTION_CACHE_TOKENS
RECOVER_TRUNCATED_REFLECTION_MEMORY=$RECOVER_TRUNCATED_REFLECTION_MEMORY
DATASET_SOURCE_DIR=$DATASET_SOURCE_DIR
LAMER_META_PPO_MINIBATCH_UPDATE=$LAMER_META_PPO_MINIBATCH_UPDATE
LAMER_META_INTEGRATION_AUDIT_DIR=$LAMER_META_INTEGRATION_AUDIT_DIR
ACTOR_SHUFFLE=$ACTOR_SHUFFLE
STEP_ADVANTAGE_WEIGHT=$STEP_ADVANTAGE_WEIGHT
GIGPO_MODE=$GIGPO_MODE
FUTURE_AWARE_EPISODE_CREDIT=$FUTURE_AWARE_EPISODE_CREDIT
TRAIN_TEMPERATURE=$TRAIN_TEMPERATURE
TRAIN_TOP_P=$TRAIN_TOP_P
TRAIN_TOP_K=$TRAIN_TOP_K
VAL_TEMPERATURE=$VAL_TEMPERATURE
VAL_TOP_P=$VAL_TOP_P
VAL_TOP_K=$VAL_TOP_K
TRAIN_ENV_SEED=$TRAIN_ENV_SEED
VAL_ENV_SEED=$VAL_ENV_SEED
ROLLOUT_SEED=$ROLLOUT_SEED
FSDP_OFFLOAD_POLICY=$FSDP_OFFLOAD_POLICY
ACTOR_PARAM_OFFLOAD=$ACTOR_PARAM_OFFLOAD
ACTOR_OPTIMIZER_OFFLOAD=$ACTOR_OPTIMIZER_OFFLOAD
ROLLOUT_LOAD_FORMAT=$ROLLOUT_LOAD_FORMAT
LAYERED_SUMMON=$LAYERED_SUMMON
RELEASE_VLLM_HOST_CACHE_AFTER_WAKE=$RELEASE_VLLM_HOST_CACHE_AFTER_WAKE
SAVE_LORA_ONLY=$SAVE_LORA_ONLY
SAVE_BEST_VALIDATION=$SAVE_BEST_VALIDATION
SAVE_LAST_CHECKPOINT=$SAVE_LAST_CHECKPOINT
BEST_VALIDATION_METRIC=$BEST_VALIDATION_METRIC
MAX_ACTOR_CKPT_TO_KEEP=$MAX_ACTOR_CKPT_TO_KEEP
GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION
MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS
MAX_NUM_SEQS=$MAX_NUM_SEQS
ENABLE_PREFIX_CACHING=$ENABLE_PREFIX_CACHING
ENFORCE_EAGER=$ENFORCE_EAGER
VLLM_DISABLE_LOG_STATS=$VLLM_DISABLE_LOG_STATS
ROLLOUT_LOGPROBS_MODE=$ROLLOUT_LOGPROBS_MODE
ROLLOUT_IS=$ROLLOUT_IS
ROLLOUT_IS_THRESHOLD=$ROLLOUT_IS_THRESHOLD
ENABLE_GPU_MONITOR=$ENABLE_GPU_MONITOR
PROFILE_ROLLOUT=$PROFILE_ROLLOUT
GPU_MONITOR_INTERVAL_SECONDS=$GPU_MONITOR_INTERVAL_SECONDS
ROLLOUT_TIMEOUT_SECONDS=$ROLLOUT_TIMEOUT_SECONDS
AGENT_LOOP_WORKERS=$AGENT_LOOP_WORKERS
ALFWORLD_ENV_POOL_SIZE=$ALFWORLD_ENV_POOL_SIZE
TRANSFER_QUEUE_STORAGE_UNITS=$TRANSFER_QUEUE_STORAGE_UNITS
RAY_NUM_CPUS=$RAY_NUM_CPUS
RAY_TMPDIR=$RAY_TMPDIR
PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF
EXTERNAL_MODULES=$EXTERNAL_MODULES
EOF

cmd=(
    "$PYTHON_BIN" -m verl.trainer.main_ppo
    algorithm.adv_estimator=gigpo
    +algorithm.credit_mode="$CREDIT_MODE"
    algorithm.use_kl_in_reward=False
    algorithm.gamma="$ALGORITHM_GAMMA"
    +algorithm.step_gamma="$STEP_GAMMA"
    +algorithm.traj_gamma="$TRAJ_GAMMA"
    +algorithm.meta_trajectory.state_baseline.enabled=True
    +algorithm.meta_trajectory.state_baseline.min_unique_trajectories=3
    +algorithm.meta_trajectory.state_baseline.min_distinct_projected_actions=2
    +algorithm.meta_trajectory.state_baseline.utility_range_tolerance="$META_TAU_U"
    +algorithm.meta_trajectory.invalid_penalty.enabled=True
    +algorithm.meta_trajectory.invalid_penalty.coefficient="$META_INVALID_PENALTY_COEFFICIENT"
    +algorithm.meta_trajectory.overlong_penalty.enabled=True
    +algorithm.meta_trajectory.overlong_penalty.mode="$META_OVERLONG_MODE"
    +algorithm.meta_trajectory.overlong_penalty.max_coefficient="$META_OVERLONG_MAX_COEFFICIENT"
    +algorithm.meta_trajectory.overlong_penalty.play_cache_tokens="$META_PLAY_CACHE_TOKENS"
    +algorithm.meta_trajectory.overlong_penalty.reflection_cache_tokens="$META_REFLECTION_CACHE_TOKENS"
    +algorithm.meta_trajectory.loss_aggregation="$(if [[ "$LOSS_AGG_MODE" == token-sum ]]; then printf meta_traj_mean; else printf token_mean; fi)"
    +algorithm.meta_trajectory.fixed_sum_denominator=null
    +algorithm.meta_trajectory.require_one_step_per_ppo_minibatch=True
    +algorithm.gigpo.step_advantage_w="$STEP_ADVANTAGE_WEIGHT"
    +algorithm.gigpo.mode="$GIGPO_MODE"
    +algorithm.gigpo.future_aware_episode_credit="$FUTURE_AWARE_EPISODE_CREDIT"
    +algorithm.gigpo.use_invalid_action_penalty=True
    +algorithm.gigpo.invalid_action_penalty_coef=0.5
    algorithm.rollout_correction.rollout_is="$ROLLOUT_IS"
    algorithm.rollout_correction.rollout_is_threshold="$ROLLOUT_IS_THRESHOLD"
    algorithm.rollout_correction.rollout_rs=null
    algorithm.rollout_correction.rollout_rs_threshold=null
    algorithm.rollout_correction.bypass_mode=False
    algorithm.rollout_correction.rollout_is_batch_normalize=False
    data.train_files="$OUTPUT_DIR/data/train.parquet"
    data.val_files="$OUTPUT_DIR/data/test.parquet"
    data.train_batch_size="$TRAIN_BATCH_SIZE"
    data.val_batch_size="$VAL_TASK_COUNT"
    data.max_prompt_length="$MAX_PROMPT_LENGTH"
    data.max_response_length="$MAX_RESPONSE_LENGTH"
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers=1
    data.dataloader_num_workers=0
    data.truncation=error
    data.return_raw_chat=True
    data.return_multi_modal_inputs=False
    data.shuffle=False
    data.validation_shuffle=False
    data.seed="$ROLLOUT_SEED"
    +data.apply_chat_template_kwargs.enable_thinking=False
    actor_rollout_ref.model.path="$MODEL_PATH"
    # Pinned VERL's FSDP backend owns the flat LoRA fields below.  The nested
    # model.lora block in the resolved config is the separate Megatron schema;
    # vLLM explicitly falls back to this flat rank for FSDP adapter serving.
    actor_rollout_ref.model.lora_rank="$LORA_RANK"
    actor_rollout_ref.model.lora_alpha="$LORA_ALPHA"
    "actor_rollout_ref.model.target_modules='$LORA_TARGET_MODULES'"
    actor_rollout_ref.model.enable_gradient_checkpointing="$GRADIENT_CHECKPOINTING"
    actor_rollout_ref.model.use_liger="$USE_LIGER"
    actor_rollout_ref.model.use_fused_kernels="$USE_FUSED_KERNELS"
    actor_rollout_ref.model.fused_kernel_options.impl_backend=torch
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.optim.optimizer="$OPTIMIZER_NAME"
    actor_rollout_ref.actor.optim.lr="$LEARNING_RATE"
    actor_rollout_ref.actor.optim.lr_scheduler_type="$LR_SCHEDULER_TYPE"
    actor_rollout_ref.actor.optim.lr_warmup_steps="$LR_WARMUP_STEPS"
    actor_rollout_ref.actor.optim.weight_decay="$WEIGHT_DECAY"
    "actor_rollout_ref.actor.optim.betas=[$ADAM_BETA1,$ADAM_BETA2]"
    "actor_rollout_ref.actor.optim.override_optimizer_config={eps:$ADAM_EPS}"
    actor_rollout_ref.actor.optim.clip_grad="$GRAD_CLIP"
    actor_rollout_ref.actor.clip_ratio_low="$CLIP_RATIO_LOW"
    actor_rollout_ref.actor.clip_ratio_high="$CLIP_RATIO_HIGH"
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE"
    actor_rollout_ref.actor.shuffle="$ACTOR_SHUFFLE"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$ACTOR_MICRO_BATCH_SIZE"
    actor_rollout_ref.actor.use_dynamic_bsz="$ACTOR_USE_DYNAMIC_BSZ"
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$ACTOR_MAX_TOKENS_PER_GPU"
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff="$ENTROPY_COEFF"
    actor_rollout_ref.actor.calculate_entropy="$CALCULATE_ENTROPY"
    actor_rollout_ref.actor.entropy_checkpointing="$ENTROPY_CHECKPOINTING"
    actor_rollout_ref.actor.loss_agg_mode="$LOSS_AGG_MODE"
    actor_rollout_ref.actor.policy_loss.loss_mode="$POLICY_LOSS_MODE"
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.fsdp_config.fsdp_size=1
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.actor.fsdp_config.offload_policy="$FSDP_OFFLOAD_POLICY"
    actor_rollout_ref.actor.fsdp_config.param_offload="$ACTOR_PARAM_OFFLOAD"
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="$ACTOR_OPTIMIZER_OFFLOAD"
    actor_rollout_ref.actor.fsdp_config.model_dtype="$ACTOR_MODEL_DTYPE"
    +actor_rollout_ref.actor.checkpoint.save_lora_only="$SAVE_LORA_ONLY"
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.data_parallel_size=1
    actor_rollout_ref.rollout.n="$GROUP_SIZE"
    actor_rollout_ref.rollout.seed="$ROLLOUT_SEED"
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.load_format="$ROLLOUT_LOAD_FORMAT"
    actor_rollout_ref.rollout.layered_summon="$LAYERED_SUMMON"
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION"
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching="$ENABLE_PREFIX_CACHING"
    actor_rollout_ref.rollout.enforce_eager="$ENFORCE_EAGER"
    actor_rollout_ref.rollout.disable_log_stats="$VLLM_DISABLE_LOG_STATS"
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.logprobs_mode="$ROLLOUT_LOGPROBS_MODE"
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN"
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS"
    actor_rollout_ref.rollout.max_num_seqs="$MAX_NUM_SEQS"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="$ACTOR_USE_DYNAMIC_BSZ"
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$OLD_LOGPROB_MAX_TOKENS_PER_GPU"
    actor_rollout_ref.rollout.temperature="$TRAIN_TEMPERATURE"
    actor_rollout_ref.rollout.top_p="$TRAIN_TOP_P"
    actor_rollout_ref.rollout.top_k="$TRAIN_TOP_K"
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.temperature="$VAL_TEMPERATURE"
    actor_rollout_ref.rollout.val_kwargs.top_p="$VAL_TOP_P"
    actor_rollout_ref.rollout.val_kwargs.top_k="$VAL_TOP_K"
    actor_rollout_ref.rollout.agent.default_agent_loop=alfworld_agent
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$REPO_ROOT/examples/alfworld/config/agent_loop_qwen35.yaml"
    actor_rollout_ref.rollout.agent.num_workers="$AGENT_LOOP_WORKERS"
    reward.num_workers=1
    trainer.use_v1=True
    trainer.v1.trainer_mode=alfworld_gigpo_sync
    +trainer.v1.alfworld_gigpo_sync.parameter_sync_step=1
    "trainer.v1.sampler.custom_sampler.path='pkg://agent_system.multi_turn_rollout.alfworld_replay_buffer'"
    trainer.v1.sampler.custom_sampler.name=AlfWorldReplayBuffer
    +trainer.v1.sampler.sampler_kwargs.rollout_timeout_seconds="$ROLLOUT_TIMEOUT_SECONDS"
    transfer_queue.backend.SimpleStorage.num_data_storage_units="$TRANSFER_QUEUE_STORAGE_UNITS"
    trainer.critic_warmup=0
    trainer.logger="$TRAINER_LOGGER"
    trainer.project_name="$PROJECT_NAME"
    trainer.experiment_name="$EXPERIMENT_NAME"
    trainer.n_gpus_per_node=1
    trainer.nnodes=1
    trainer.balance_batch=False
    trainer.save_freq="$SAVE_FREQ"
    trainer.test_freq="$TEST_FREQ"
    trainer.total_epochs=1
    trainer.total_training_steps="$TOTAL_STEPS"
    trainer.val_before_train="$VAL_BEFORE_TRAIN"
    trainer.val_only=False
    trainer.log_val_generations=0
    trainer.max_actor_ckpt_to_keep="$MAX_ACTOR_CKPT_TO_KEEP"
    +trainer.save_best_validation="$SAVE_BEST_VALIDATION"
    "+trainer.best_validation_metric='$BEST_VALIDATION_METRIC'"
    +trainer.save_last_checkpoint="$SAVE_LAST_CHECKPOINT"
    trainer.default_local_dir="$OUTPUT_DIR/checkpoints"
    trainer.validation_data_dir="$OUTPUT_DIR/validation"
    trainer.resume_mode="$RESUME_MODE"
    +env.env_name=alfworld/AlfredTWEnv
    +env.seed="$TRAIN_ENV_SEED"
    +env.val_seed="$VAL_ENV_SEED"
    +env.num_attempts="$NUM_ATTEMPTS"
    +env.val_num_attempts="$NUM_ATTEMPTS"
    +env.do_reflection=True
    +env.val_do_reflection=True
    +env.max_turns="$MAX_TURNS_PER_ATTEMPT"
    +env.alfworld.prompt_history_length="$PROMPT_HISTORY_LENGTH"
    +env.reflection_type=reflection_only
    +env.presence_penalty="$PRESENCE_PENALTY"
    +env.brief_response_instruction="$BRIEF_RESPONSE_INSTRUCTION"
    +env.recover_truncated_reflection_memory="$RECOVER_TRUNCATED_REFLECTION_MEMORY"
    +env.alfworld.eval_dataset="$EVAL_DATASET"
    +env.alfworld.local_env_pool_size="$ALFWORLD_ENV_POOL_SIZE"
    +env.alfworld.profile_rollout="$PROFILE_ROLLOUT"
    ray_kwargs.ray_init.num_cpus="$RAY_NUM_CPUS"
    ray_kwargs.ray_init.runtime_env.py_executable="$PYTHON_BIN"
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="$PYTHONPATH"
    +ray_kwargs.ray_init.runtime_env.env_vars.ALFWORLD_DATA="$ALFWORLD_DATA"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES='$VERL_USE_EXTERNAL_MODULES'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_RELEASE_VLLM_HOST_CACHE_AFTER_WAKE='$VERL_RELEASE_VLLM_HOST_CACHE_AFTER_WAKE'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.LAMER_META_PPO_MINIBATCH_UPDATE='$LAMER_META_PPO_MINIBATCH_UPDATE'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.LAMER_META_INTEGRATION_AUDIT_DIR='$LAMER_META_INTEGRATION_AUDIT_DIR'"
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
    hydra.run.dir="$ARTIFACT_DIR/hydra"
)

if [[ "$RESUME_MODE" == resume_path ]]; then
    cmd+=(trainer.resume_from_path="$RESUME_FROM_PATH")
fi

if [[ "$DRY_RUN" == 1 ]]; then
    cmd+=(--cfg job)
fi

printf '%s %s GiGPO %s: %s task(s)/step, %s rollouts/task, %s step(s).\n' \
    "$MODEL_DISPLAY_NAME" "$TRAINING_VARIANT" "$RUN_MODE" "$TRAIN_BATCH_SIZE" "$GROUP_SIZE" "$TOTAL_STEPS"
printf 'Host-memory bounds: %s ALFWorld environments/worker, %s TransferQueue storage actors.\n' \
    "$ALFWORLD_ENV_POOL_SIZE" "$TRANSFER_QUEUE_STORAGE_UNITS"
printf 'Release unused vLLM pinned host cache after weight wake: %s.\n' \
    "$RELEASE_VLLM_HOST_CACHE_AFTER_WAKE"
printf 'Generation protocol: brief=%s, prompt=%s, response=%s, context=%s.\n' \
    "$BRIEF_RESPONSE_INSTRUCTION" "$MAX_PROMPT_LENGTH" \
    "$MAX_RESPONSE_LENGTH" "$MAX_MODEL_LEN"
if [[ "$RESUME_MODE" != disable ]]; then
    printf 'Resume mode: %s from %s; attempt artifacts: %s.\n' \
        "$RESUME_MODE" "$RESOLVED_RESUME_PATH" "$ARTIFACT_DIR"
fi
if [[ "$VAL_BEFORE_TRAIN" == True ]]; then
    printf 'Step-zero validation: %s fixed %s tasks with seed %s.\n' \
        "$VAL_TASK_COUNT" "$EVAL_DATASET" "$VAL_ENV_SEED"
else
    if ((TEST_FREQ > 0)); then
        if [[ "$RESUME_MODE" == disable ]]; then
            printf 'Step-zero validation disabled; post-update validation remains configured.\n'
        else
            printf 'Already-completed checkpoint validation will not be repeated; scheduled validation remains configured.\n'
        fi
    else
        printf 'Validation disabled for this diagnostic run.\n'
    fi
fi

GPU_MONITOR_PID=
stop_gpu_monitor() {
    if [[ -n "$GPU_MONITOR_PID" ]]; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
        GPU_MONITOR_PID=
    fi
}

if [[ "$DRY_RUN" != 1 && "$ENABLE_GPU_MONITOR" == True ]] && \
    command -v nvidia-smi >/dev/null 2>&1 && \
    nvidia-smi --query-gpu=index --format=csv,noheader >/dev/null 2>&1; then
    nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory \
        --format=csv \
        -l "$GPU_MONITOR_INTERVAL_SECONDS" \
        > "$ARTIFACT_DIR/gpu_timeseries.csv" \
        2> "$ARTIFACT_DIR/gpu_monitor.log" &
    GPU_MONITOR_PID=$!
    trap stop_gpu_monitor EXIT
    printf 'Recording GPU telemetry every %s seconds in %s/gpu_timeseries.csv.\n' \
        "$GPU_MONITOR_INTERVAL_SECONDS" "$ARTIFACT_DIR"
fi

started=$(date +%s)
set +e
cd "$VERL_DIR"
"${cmd[@]}" "$@" 2>&1 | tee "$ARTIFACT_DIR/train.log"
status=${PIPESTATUS[0]}
set -e
finished=$(date +%s)
stop_gpu_monitor
trap - EXIT
printf '%s\n' "$((finished - started))" > "$ARTIFACT_DIR/elapsed_seconds.txt"

if ((status != 0)); then
    printf 'Training failed; see %s/train.log\n' "$ARTIFACT_DIR" >&2
    exit "$status"
fi
if [[ "$DRY_RUN" == 1 ]]; then
    printf 'Configuration validation complete: %s\n' "$ARTIFACT_DIR"
    exit 0
fi
printf 'Training complete: %s\n' "$OUTPUT_DIR"
