#!/usr/bin/env bash
# Capture one GiGPO training batch on one 24GB GPU, without updating weights.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

TASK_COUNT=${TASK_COUNT:-4}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/diagnostics/gigpo_groups_capture/$(date -u +%Y%m%dT%H%M%S_%N)"}
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd -- "$OUTPUT_DIR" && pwd)

# Keep capture data separate from the published launcher's prepared data.
python3 -m examples.data_preprocess.prepare \
    --mode text \
    --local_dir "$OUTPUT_DIR/data" \
    --train_data_size "$TASK_COUNT" \
    --val_data_size 4 \
    2>&1 | tee "$OUTPUT_DIR/prepare.log"

# Dataset size equals batch size, so one epoch contains one training iteration.
# Step 1 is below critic_warmup=2: the existing trainer gate skips update_actor.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    "data.train_files=$OUTPUT_DIR/data/text/train.parquet" \
    "data.val_files=$OUTPUT_DIR/data/text/test.parquet" \
    "data.train_batch_size=$TASK_COUNT" \
    data.val_batch_size=4 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    "actor_rollout_ref.model.path=$MODEL_PATH" \
    +actor_rollout_ref.model.enable_thinking=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
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
    env.num_attempts=3 \
    env.max_steps=15 \
    env.max_turns=7 \
    +env.reflection_type=reflection_only \
    trainer.critic_warmup=2 \
    'trainer.logger=[console]' \
    trainer.project_name=lamer \
    trainer.experiment_name=minesweeper_group_capture \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.resume_mode=disable \
    trainer.val_before_train=False \
    trainer.grouping_diagnostics.enabled=True \
    "trainer.grouping_diagnostics.output_dir=$OUTPUT_DIR" \
    trainer.grouping_diagnostics.cross_attempt_only=False \
    "hydra.run.dir=$OUTPUT_DIR/hydra" \
    2>&1 | tee "$OUTPUT_DIR/run.log"

printf '\nCapture directory: %s\n' "$OUTPUT_DIR"
