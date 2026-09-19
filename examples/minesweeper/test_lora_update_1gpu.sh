#!/usr/bin/env bash
# Test one real GiGPO LoRA actor update on one 24GB GPU.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

TASK_COUNT=${TASK_COUNT:-16}
VAL_TASK_COUNT=${VAL_TASK_COUNT:-1}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/diagnostics/lora_update_test/tasks_${TASK_COUNT}_micro_${MICRO_BATCH_SIZE}_$(date -u +%Y%m%dT%H%M%S_%N)"}

case "$MICRO_BATCH_SIZE" in
    1|2|4|8|16|32|64) ;;
    *) printf 'MICRO_BATCH_SIZE must divide PPO minibatch size 64.\n' >&2; exit 2 ;;
esac

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd -- "$OUTPUT_DIR" && pwd)

python3 -m examples.data_preprocess.prepare \
    --mode text \
    --local_dir "$OUTPUT_DIR/data" \
    --train_data_size "$TASK_COUNT" \
    --val_data_size "$VAL_TASK_COUNT" \
    2>&1 | tee "$OUTPUT_DIR/prepare.log"

# The base model is frozen in BF16. PEFT keeps the trainable LoRA adapters in
# FP32, and vLLM preloads the base checkpoint then receives adapter updates.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    "data.train_files=$OUTPUT_DIR/data/text/train.parquet" \
    "data.val_files=$OUTPUT_DIR/data/text/test.parquet" \
    "data.train_batch_size=$TASK_COUNT" \
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
    actor_rollout_ref.actor.optim.lr=1e-5 \
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
    env.seed=0 \
    env.rollout.n=8 \
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
    trainer.experiment_name=minesweeper_lora_update_test \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.resume_mode=disable \
    trainer.val_before_train=False \
    trainer.grouping_diagnostics.enabled=False \
    "hydra.run.dir=$OUTPUT_DIR/hydra" \
    2>&1 | tee "$OUTPUT_DIR/run.log"

printf '\nLoRA update-test directory: %s\n' "$OUTPUT_DIR"
