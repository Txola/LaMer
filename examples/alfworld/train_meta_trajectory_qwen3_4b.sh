#!/usr/bin/env bash
# Frozen first-experiment protocol for Qwen3-4B LoRA meta-trajectory credit.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export MODEL_PROFILE=qwen3_4b
export TRAINING_VARIANT=lora
export CREDIT_MODE=meta_trajectory
export LAMER_META_PPO_MINIBATCH_UPDATE=1


export NUM_ATTEMPTS=3
export MAX_TURNS_PER_ATTEMPT=15
export PROMPT_HISTORY_LENGTH=15
export MAX_RESPONSE_LENGTH=1024
export PRESENCE_PENALTY=0.0

export TRAIN_TEMPERATURE=1.0
export TRAIN_TOP_P=1.0
export TRAIN_TOP_K=-1
export VAL_TEMPERATURE=0.7
export VAL_TOP_P=0.8
export VAL_TOP_K=20

export STEP_GAMMA=0.9675
export TRAJ_GAMMA=0.6
export META_TAU_U=1e-6
export META_INVALID_PENALTY_COEFFICIENT=0.5
export META_OVERLONG_MODE=soft_shaping
export META_OVERLONG_MAX_COEFFICIENT=0.5
export META_PLAY_CACHE_TOKENS=150
export META_REFLECTION_CACHE_TOKENS=150
export RECOVER_TRUNCATED_REFLECTION_MEMORY=True

export LOSS_AGG_MODE=token-sum
export POLICY_LOSS_MODE=meta_vanilla
export ENTROPY_COEFF=0
export CALCULATE_ENTROPY=False
export ACTOR_MICRO_BATCH_SIZE=1
export ACTOR_USE_DYNAMIC_BSZ=${ACTOR_USE_DYNAMIC_BSZ:-True}
export ACTOR_MAX_TOKENS_PER_GPU=${ACTOR_MAX_TOKENS_PER_GPU:-6144}
# Pinned VERL requires old-logprob and backward to agree on whether dynamic
# batching is enabled, but permits this inference-only budget to differ.
export OLD_LOGPROB_MAX_TOKENS_PER_GPU=${OLD_LOGPROB_MAX_TOKENS_PER_GPU:-6144}
export ACTOR_SHUFFLE=True
export CLIP_RATIO_LOW=0.20
export CLIP_RATIO_HIGH=0.28
export LEARNING_RATE=2e-5

export ENTRYPOINT_SCRIPT=${BASH_SOURCE[0]}
exec "$SCRIPT_DIR/train_alfworld_modern.sh" "$@"
