#!/usr/bin/env bash
# Evaluate Minesweeper LoRA checkpoints sequentially on one 24 GB GPU.
# Each checkpoint gets a fresh Python/Ray/vLLM process so CUDA memory is fully
# released and the stochastic evaluation starts from the same seeds.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

TRAINING_RUN_DIR=${1:?Usage: $0 /absolute/path/to/training_run}
TRAINING_RUN_DIR=$(cd -- "$TRAINING_RUN_DIR" && pwd)
CHECKPOINT_ROOT="$TRAINING_RUN_DIR/checkpoints"
PARAMETERS_FILE="$TRAINING_RUN_DIR/run_parameters.txt"

if [[ ! -d "$CHECKPOINT_ROOT" || ! -f "$PARAMETERS_FILE" ]]; then
    printf 'Expected checkpoints/ and run_parameters.txt under %s\n' "$TRAINING_RUN_DIR" >&2
    exit 2
fi

run_parameter() {
    local key=$1
    local fallback=$2
    local value
    value=$(sed -n "s/^${key}=//p" "$PARAMETERS_FILE" | tail -1)
    printf '%s' "${value:-$fallback}"
}

MODEL_PATH=${MODEL_PATH:-$(run_parameter MODEL_PATH Qwen/Qwen3-4B)}
LORA_RANK=${LORA_RANK:-$(run_parameter LORA_RANK 32)}
LORA_ALPHA=${LORA_ALPHA:-$(run_parameter LORA_ALPHA 64)}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-$(run_parameter MICRO_BATCH_SIZE 4)}
EVAL_TASK_COUNT=${EVAL_TASK_COUNT:-32}
EVAL_SEED=${EVAL_SEED:-0}
ROLLOUT_SEED=${ROLLOUT_SEED:-20}
EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-0.7}
EVAL_TOP_P=${EVAL_TOP_P:-0.8}
EVAL_TOP_K=${EVAL_TOP_K:-20}
INCLUDE_BASELINE=${INCLUDE_BASELINE:-1}
CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-all}
FORCE_REEVALUATE=${FORCE_REEVALUATE:-0}
VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}
EVAL_DIR=${EVAL_DIR:-"$TRAINING_RUN_DIR/evaluation/minesweeper_${EVAL_TASK_COUNT}tasks_env${EVAL_SEED}_rollout${ROLLOUT_SEED}"}
export VERL_LOGGING_LEVEL

mkdir -p "$EVAL_DIR/data"
EVAL_DIR=$(cd -- "$EVAL_DIR" && pwd)

python3 -m examples.data_preprocess.prepare \
    --mode text \
    --local_dir "$EVAL_DIR/data" \
    --train_data_size 1 \
    --val_data_size "$EVAL_TASK_COUNT" \
    2>&1 | tee "$EVAL_DIR/prepare.log"

cat > "$EVAL_DIR/evaluation_parameters.txt" <<EOF
TRAINING_RUN_DIR=$TRAINING_RUN_DIR
MODEL_PATH=$MODEL_PATH
LORA_RANK=$LORA_RANK
LORA_ALPHA=$LORA_ALPHA
EVAL_TASK_COUNT=$EVAL_TASK_COUNT
EVAL_SEED=$EVAL_SEED
ROLLOUT_SEED=$ROLLOUT_SEED
EVAL_TEMPERATURE=$EVAL_TEMPERATURE
EVAL_TOP_P=$EVAL_TOP_P
EVAL_TOP_K=$EVAL_TOP_K
INCLUDE_BASELINE=$INCLUDE_BASELINE
CHECKPOINT_STEPS=$CHECKPOINT_STEPS
FORCE_REEVALUATE=$FORCE_REEVALUATE
VERL_LOGGING_LEVEL=$VERL_LOGGING_LEVEL
VALIDATION_INTERACTION_DIAGNOSTICS=1
EOF

step_selected() {
    local step=$1
    [[ "$CHECKPOINT_STEPS" == "all" || ",$CHECKPOINT_STEPS," == *",$step,"* ]]
}

evaluate_one() {
    local label=$1
    local checkpoint_path=$2
    local result_dir="$EVAL_DIR/$label"
    mkdir -p "$result_dir"

    if [[ "$FORCE_REEVALUATE" != "1" && -f "$result_dir/metrics.json" ]] && rg -q '"status": "complete"' "$result_dir/metrics.json"; then
        printf 'Skipping completed evaluation: %s\n' "$label"
        return
    fi

    local resume_args=(trainer.resume_mode=disable)
    if [[ -n "$checkpoint_path" ]]; then
        resume_args=(trainer.resume_mode=resume_path "trainer.resume_from_path=$checkpoint_path")
        printf '%s\n' "$checkpoint_path" > "$result_dir/checkpoint_path.txt"
    else
        printf 'base_model_with_zero_initialized_lora\n' > "$result_dir/checkpoint_path.txt"
    fi

    printf 'Evaluating %s on %s fixed Minesweeper tasks\n' "$label" "$EVAL_TASK_COUNT"
    local started finished status
    started=$(date +%s)
    set +e
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=gigpo \
        "data.train_files=$EVAL_DIR/data/text/train.parquet" \
        "data.val_files=$EVAL_DIR/data/text/test.parquet" \
        data.train_batch_size=1 \
        "data.val_batch_size=$EVAL_TASK_COUNT" \
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
        actor_rollout_ref.actor.checkpoint.save_lora_only=True \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE" \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.strategy=fsdp \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        "actor_rollout_ref.rollout.seed=$ROLLOUT_SEED" \
        actor_rollout_ref.rollout.load_format=safetensors \
        actor_rollout_ref.rollout.layered_summon=True \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.free_cache_engine=False \
        "actor_rollout_ref.rollout.val_kwargs.temperature=$EVAL_TEMPERATURE" \
        "actor_rollout_ref.rollout.val_kwargs.top_p=$EVAL_TOP_P" \
        "actor_rollout_ref.rollout.val_kwargs.top_k=$EVAL_TOP_K" \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        algorithm.gamma=0.95 \
        +algorithm.step_gamma=0.95 \
        +algorithm.traj_gamma=0.6 \
        algorithm.gigpo.step_advantage_w=1.0 \
        algorithm.gigpo.mode=mean_norm \
        reward_model.reward_manager=episode \
        env.env_name=Minesweeper \
        "env.seed=$EVAL_SEED" \
        env.rollout.n=1 \
        env.minesweeper.board_size=6 \
        env.minesweeper.n_mines=3 \
        env.minesweeper.board_type=board \
        env.minesweeper.mode=text \
        +env.minesweeper.execution_backend=local \
        env.num_attempts=3 \
        +env.val_num_attempts=3 \
        +env.do_reflection=True \
        +env.val_do_reflection=True \
        env.max_steps=15 \
        env.max_turns=7 \
        +env.reflection_type=reflection_only \
        trainer.critic_warmup=0 \
        'trainer.logger=[console]' \
        trainer.project_name=lamer \
        trainer.experiment_name=minesweeper_checkpoint_evaluation \
        trainer.n_gpus_per_node=1 \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.total_epochs=1 \
        trainer.total_training_steps=1 \
        trainer.val_before_train=True \
        +trainer.val_only=True \
        trainer.log_val_generations=1 \
        trainer.grouping_diagnostics.enabled=True \
        "trainer.validation_data_dir=$result_dir/validation_diagnostics" \
        "trainer.default_local_dir=$CHECKPOINT_ROOT" \
        "${resume_args[@]}" \
        "hydra.run.dir=$result_dir/hydra" \
        2>&1 | tee "$result_dir/eval.log"
    status=${PIPESTATUS[0]}
    set -e
    finished=$(date +%s)
    printf '%s\n' "$((finished - started))" > "$result_dir/elapsed_seconds.txt"

    python3 analysis/summarize_minesweeper_evaluations.py --eval-dir "$EVAL_DIR"
    if (( status != 0 )); then
        printf 'Evaluation failed for %s; see %s\n' "$label" "$result_dir/eval.log" >&2
        return "$status"
    fi
}

if [[ "$INCLUDE_BASELINE" == "1" ]]; then
    evaluate_one baseline ""
fi

mapfile -t CHECKPOINT_DIRS < <(find "$CHECKPOINT_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'global_step_*' | sort -V)
if (( ${#CHECKPOINT_DIRS[@]} == 0 )); then
    printf 'No checkpoints found under %s\n' "$CHECKPOINT_ROOT" >&2
    exit 2
fi

for checkpoint_dir in "${CHECKPOINT_DIRS[@]}"; do
    step=${checkpoint_dir##*global_step_}
    if step_selected "$step"; then
        if [[ ! -f "$checkpoint_dir/data.pt" ]] || \
           [[ ! -f "$checkpoint_dir/actor/lora_model_world_size_1_rank_0.pt" && \
              ! -f "$checkpoint_dir/actor/model_world_size_1_rank_0.pt" ]]; then
            printf 'Skipping incomplete checkpoint: %s\n' "$checkpoint_dir" >&2
            continue
        fi
        evaluate_one "$(printf 'step_%06d' "$step")" "$checkpoint_dir"
    fi
done

python3 analysis/summarize_minesweeper_evaluations.py --eval-dir "$EVAL_DIR"
printf '\nEvaluation table: %s\n' "$EVAL_DIR/summary.md"
