#!/usr/bin/env bash
# Validation-only ALFWorld evaluation of an untrained base model with Reflexion.
# Run each rollout seed in a fresh process so vLLM starts from the requested seed.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

DEFAULT_PYTHON_BIN=python3
if [[ -x "$HOME/miniconda3/envs/lamer/bin/python" ]]; then
    DEFAULT_PYTHON_BIN="$HOME/miniconda3/envs/lamer/bin/python"
fi
PYTHON_BIN=${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
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
TRAJECTORY_SAMPLES_PER_TASK=${TRAJECTORY_SAMPLES_PER_TASK:-1}
TRAJECTORY_SAMPLE_SEED=${TRAJECTORY_SAMPLE_SEED:-0}
DUMP_ALL_INTERACTIONS=${DUMP_ALL_INTERACTIONS:-False}
HARDWARE_PROFILE=${HARDWARE_PROFILE:-small_24gb}
case "$HARDWARE_PROFILE" in
    small_24gb)
        PROFILE_VLLM_ATTENTION_BACKEND=XFORMERS
        PROFILE_GPU_MEMORY_UTILIZATION=0.6
        PROFILE_MAX_NUM_BATCHED_TOKENS=16384
        PROFILE_RAY_NUM_CPUS=$(nproc)
        PROFILE_ENV_CPUS_PER_WORKER=0.1
        PROFILE_GAMES_PER_ENV_WORKER=32
        ;;
    large_96gb)
        # vLLM 0.10.2 bundles its own Flash-Attention backend. Using it avoids
        # importing xformers, whose 0.0.32.post1 release rejects the separate
        # flash-attn 2.8.3 package required by this Blackwell environment.
        PROFILE_VLLM_ATTENTION_BACKEND=FLASH_ATTN
        PROFILE_GPU_MEMORY_UTILIZATION=0.8
        PROFILE_MAX_NUM_BATCHED_TOKENS=32768
        PROFILE_RAY_NUM_CPUS=18
        PROFILE_ENV_CPUS_PER_WORKER=1
        # 13 validation actors + one training actor + TaskRunner leave three
        # Ray CPU slots for the GPU worker and control tasks on the 20-core host.
        PROFILE_GAMES_PER_ENV_WORKER=10
        ;;
    *)
        printf 'HARDWARE_PROFILE must be small_24gb or large_96gb; got %s\n' \
            "$HARDWARE_PROFILE" >&2
        exit 2
        ;;
esac
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-$PROFILE_VLLM_ATTENTION_BACKEND}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-$PROFILE_GPU_MEMORY_UTILIZATION}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-$PROFILE_MAX_NUM_BATCHED_TOKENS}
ENFORCE_EAGER=${ENFORCE_EAGER:-True}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-$PROFILE_RAY_NUM_CPUS}
ENV_CPUS_PER_WORKER=${ENV_CPUS_PER_WORKER:-$PROFILE_ENV_CPUS_PER_WORKER}
GAMES_PER_ENV_WORKER=${GAMES_PER_ENV_WORKER:-$PROFILE_GAMES_PER_ENV_WORKER}
VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}
FORCE_REEVALUATE=${FORCE_REEVALUATE:-0}
ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
EVAL_DIR=${EVAL_DIR:-"$REPO_ROOT/outputs/alfworld_base_qwen_reflexion/balanced${EVAL_TASK_COUNT}_env${VAL_ENV_SEED}_rollout${ROLLOUT_SEED}"}
export ALFWORLD_DATA PYTHONHASHSEED="$ROLLOUT_SEED" VERL_LOGGING_LEVEL
export VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=${VLLM_USE_V1:-0}

if [[ "$EVAL_DATASET" != eval_all ]]; then
    printf 'This full balanced baseline requires EVAL_DATASET=eval_all; got %s\n' "$EVAL_DATASET" >&2
    exit 2
fi
if (( EVAL_TASK_COUNT < 1 || EVAL_TASK_COUNT > 126 )); then
    printf 'EVAL_TASK_COUNT must be between 1 and 126 for eval_all; got %s\n' "$EVAL_TASK_COUNT" >&2
    exit 2
fi
if (( NUM_ATTEMPTS < 1 )); then
    printf 'NUM_ATTEMPTS must be positive; got %s\n' "$NUM_ATTEMPTS" >&2
    exit 2
fi
if (( MAX_TURNS_PER_ATTEMPT < 1 )); then
    printf 'MAX_TURNS_PER_ATTEMPT must be positive; got %s\n' "$MAX_TURNS_PER_ATTEMPT" >&2
    exit 2
fi
if [[ "$DO_REFLECTION" != "True" && "$DO_REFLECTION" != "False" ]]; then
    printf 'DO_REFLECTION must be True or False; got %s\n' "$DO_REFLECTION" >&2
    exit 2
fi
if [[ "$DUMP_ALL_INTERACTIONS" != "True" && "$DUMP_ALL_INTERACTIONS" != "False" ]]; then
    printf 'DUMP_ALL_INTERACTIONS must be True or False; got %s\n' "$DUMP_ALL_INTERACTIONS" >&2
    exit 2
fi
if [[ "$ENFORCE_EAGER" != "True" && "$ENFORCE_EAGER" != "False" ]]; then
    printf 'ENFORCE_EAGER must be True or False; got %s\n' "$ENFORCE_EAGER" >&2
    exit 2
fi

"$PYTHON_BIN" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] <= (3, 12)):
    raise SystemExit(
        f"Python {sys.version.split()[0]} is unsupported for ALFWorld; "
        "use Python 3.10-3.12."
    )
PY
"$PYTHON_BIN" - <<'PY'
import importlib
import importlib.util
from packaging.version import Version
required = (
    "alfworld", "datasets", "flash_attn", "hydra", "peft", "ray",
    "textworld", "torch", "transformers", "vllm",
)
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing evaluation packages: " + ", ".join(missing))
torch = importlib.import_module("torch")
transformers = importlib.import_module("transformers")
if Version(transformers.__version__).major >= 5:
    raise SystemExit(
        f"Transformers {transformers.__version__} is incompatible with this "
        "LaMer/vLLM adapter; install transformers==4.55.2, "
        "tokenizers==0.21.4, and huggingface-hub==0.34.4."
    )
try:
    importlib.import_module("flash_attn")
except (ImportError, OSError) as exc:
    raise SystemExit(
        "flash-attn cannot load with torch " + torch.__version__ + ". "
        "Reinstall flash-attn against this PyTorch build before launching Ray. "
        f"Original error: {exc}"
    ) from exc
assert torch.cuda.is_available(), "CUDA is not available to PyTorch"
if torch.cuda.get_device_capability(0)[0] >= 12:
    cuda_major_minor = tuple(map(int, torch.version.cuda.split(".")[:2]))
    assert cuda_major_minor >= (12, 8), (
        f"Blackwell requires a CUDA >=12.8 PyTorch build; found {torch.version.cuda}"
    )
    assert "sm_120" in torch.cuda.get_arch_list(), (
        f"PyTorch wheel lacks sm_120 support: {torch.cuda.get_arch_list()}"
    )
PY

if [[ ! -f "$ALFWORLD_DATA/json/split_manifest.json" || ! -d "$ALFWORLD_DATA/json/valid_task_balanced126" ]]; then
    if [[ -d "$ALFWORLD_DATA/json_2.1.1/train" && -d "$ALFWORLD_DATA/json_2.1.1/valid_train" && -d "$ALFWORLD_DATA/json_2.1.1/valid_seen" ]]; then
        "$PYTHON_BIN" scripts/prepare_alfworld_lamer_splits.py --replace
    else
        printf 'Missing official ALFWorld data and prepared split under %s.\n' "$ALFWORLD_DATA" >&2
        exit 2
    fi
fi

mkdir -p "$EVAL_DIR/data"
EVAL_DIR=$(cd -- "$EVAL_DIR" && pwd)
# Ray nests long session and socket names below its temp directory. Keeping this
# outside EVAL_DIR avoids Linux's 107-byte AF_UNIX socket-path limit when the
# evaluation output path is descriptive.
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/lamer-ray-$UID}
mkdir -p "$RAY_TMPDIR"
if [[ "$FORCE_REEVALUATE" != "1" && -f "$EVAL_DIR/validation_diagnostics/step_000000_metrics.json" ]]; then
    printf 'A completed evaluation already exists under %s\n' "$EVAL_DIR" >&2
    printf 'Set FORCE_REEVALUATE=1 to replace its generated evaluation files.\n' >&2
    exit 2
fi
cp "$ALFWORLD_DATA/json/split_manifest.json" "$EVAL_DIR/split_manifest.json"
cp -- "${BASH_SOURCE[0]}" "$EVAL_DIR/launcher.sh"
git rev-parse HEAD > "$EVAL_DIR/code_revision.txt"
git status --short > "$EVAL_DIR/worktree_status.txt"
git diff --no-ext-diff > "$EVAL_DIR/working_tree.patch"
"$PYTHON_BIN" -c 'import platform, alfworld, ray, textworld, torch, transformers, vllm; print(platform.python_version()); print("torch", torch.__version__, "cuda", torch.version.cuda, "arch", torch.cuda.get_arch_list()); print("transformers", transformers.__version__); print("ray", ray.__version__); print("vllm", vllm.__version__); print("alfworld", getattr(alfworld, "__version__", "unknown")); print("textworld", getattr(textworld, "__version__", "unknown"))' > "$EVAL_DIR/software_versions.txt"
nvidia-smi > "$EVAL_DIR/gpu_info.txt" 2>&1 || true

if [[ ! -f "$EVAL_DIR/data/text/train.parquet" || ! -f "$EVAL_DIR/data/text/test.parquet" ]]; then
    "$PYTHON_BIN" -m examples.data_preprocess.prepare \
        --mode text \
        --local_dir "$EVAL_DIR/data" \
        --train_data_size 1 \
        --val_data_size "$EVAL_TASK_COUNT" \
        2>&1 | tee "$EVAL_DIR/prepare.log"
fi

cat > "$EVAL_DIR/evaluation_parameters.txt" <<EOF
MODEL_PATH=$MODEL_PATH
EVAL_DATASET=$EVAL_DATASET
EVAL_TASK_COUNT=$EVAL_TASK_COUNT
SPLIT_MANIFEST=$EVAL_DIR/split_manifest.json
ENV_SEED=$ENV_SEED
VAL_ENV_SEED=$VAL_ENV_SEED
ROLLOUT_SEED=$ROLLOUT_SEED
PYTHONHASHSEED=$PYTHONHASHSEED
NUM_ATTEMPTS=$NUM_ATTEMPTS
DO_REFLECTION=$DO_REFLECTION
REFLECTION_TYPE=$REFLECTION_TYPE
MAX_TURNS_PER_ATTEMPT=$MAX_TURNS_PER_ATTEMPT
EVAL_TEMPERATURE=$EVAL_TEMPERATURE
EVAL_TOP_P=$EVAL_TOP_P
EVAL_TOP_K=$EVAL_TOP_K
ENABLE_THINKING=False
TRAJECTORY_SAMPLES_PER_TASK=$TRAJECTORY_SAMPLES_PER_TASK
TRAJECTORY_SAMPLE_SEED=$TRAJECTORY_SAMPLE_SEED
DUMP_ALL_INTERACTIONS=$DUMP_ALL_INTERACTIONS
HARDWARE_PROFILE=$HARDWARE_PROFILE
VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND
GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION
MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS
ENFORCE_EAGER=$ENFORCE_EAGER
RAY_NUM_CPUS=$RAY_NUM_CPUS
ENV_CPUS_PER_WORKER=$ENV_CPUS_PER_WORKER
GAMES_PER_ENV_WORKER=$GAMES_PER_ENV_WORKER
RAY_TMPDIR=$RAY_TMPDIR
FORCE_REEVALUATE=$FORCE_REEVALUATE
EOF

printf 'Evaluating %s on %s fixed balanced ALFWorld tasks (validation seed %s, rollout seed %s, profile %s)\n' \
    "$MODEL_PATH" "$EVAL_TASK_COUNT" "$VAL_ENV_SEED" "$ROLLOUT_SEED" "$HARDWARE_PROFILE"
started=$(date +%s)
set +e
"$PYTHON_BIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    "data.train_files=$EVAL_DIR/data/text/train.parquet" \
    "data.val_files=$EVAL_DIR/data/text/test.parquet" \
    data.train_batch_size=1 \
    "data.val_batch_size=$EVAL_TASK_COUNT" \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    data.shuffle=False \
    "+data.seed=$ROLLOUT_SEED" \
    "actor_rollout_ref.model.path=$MODEL_PATH" \
    +actor_rollout_ref.model.enable_thinking=False \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
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
    "actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER" \
    actor_rollout_ref.rollout.free_cache_engine=False \
    +actor_rollout_ref.rollout.capture_generation_diagnostics=True \
    "actor_rollout_ref.rollout.val_kwargs.temperature=$EVAL_TEMPERATURE" \
    "actor_rollout_ref.rollout.val_kwargs.top_p=$EVAL_TOP_P" \
    "actor_rollout_ref.rollout.val_kwargs.top_k=$EVAL_TOP_K" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    "actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS" \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    +algorithm.step_gamma=0.95 \
    +algorithm.traj_gamma=0.6 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=mean_norm \
    reward_model.reward_manager=episode \
    env.env_name=alfworld/AlfredTWEnv \
    "env.seed=$ENV_SEED" \
    "+env.val_seed=$VAL_ENV_SEED" \
    env.rollout.n=1 \
    env.num_attempts=1 \
    "+env.val_num_attempts=$NUM_ATTEMPTS" \
    +env.do_reflection=False \
    "+env.val_do_reflection=$DO_REFLECTION" \
    env.max_steps=30 \
    "env.max_turns=$MAX_TURNS_PER_ATTEMPT" \
    "+env.reflection_type=$REFLECTION_TYPE" \
    "env.alfworld.eval_dataset=$EVAL_DATASET" \
    "env.alfworld.games_per_worker=$GAMES_PER_ENV_WORKER" \
    "env.resources_per_worker.num_cpus=$ENV_CPUS_PER_WORKER" \
    env.resources_per_worker.num_gpus=0 \
    trainer.critic_warmup=0 \
    'trainer.logger=[console]' \
    trainer.project_name=lamer \
    trainer.experiment_name=alfworld_base_qwen_reflexion \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.val_before_train=True \
    +trainer.val_only=True \
    trainer.log_val_generations=0 \
    "trainer.validation_dump_all_interactions=$DUMP_ALL_INTERACTIONS" \
    "trainer.validation_trajectory_samples_per_task=$TRAJECTORY_SAMPLES_PER_TASK" \
    "trainer.validation_trajectory_sample_seed=$TRAJECTORY_SAMPLE_SEED" \
    "trainer.validation_data_dir=$EVAL_DIR/validation_diagnostics" \
    "trainer.default_local_dir=$EVAL_DIR/checkpoints" \
    trainer.resume_mode=disable \
    "ray_init.num_cpus=$RAY_NUM_CPUS" \
    +ray_init.include_dashboard=False \
    "hydra.run.dir=$EVAL_DIR/hydra" \
    2>&1 | tee "$EVAL_DIR/eval.log"
status=${PIPESTATUS[0]}
set -e
finished=$(date +%s)
printf '%s\n' "$((finished - started))" > "$EVAL_DIR/elapsed_seconds.txt"

if (( status != 0 )); then
    printf 'Evaluation failed; see %s/eval.log\n' "$EVAL_DIR" >&2
    exit "$status"
fi

printf 'Evaluation complete. Results:\n'
printf '  metrics: %s/validation_diagnostics/step_000000_metrics.json\n' "$EVAL_DIR"
printf '  summary: %s/validation_diagnostics/step_000000_summary.json\n' "$EVAL_DIR"
printf '  sampled trajectories: %s/validation_diagnostics/step_000000_trajectory_samples.md\n' "$EVAL_DIR"
if [[ "$DUMP_ALL_INTERACTIONS" == "True" ]]; then
    printf '  all interactions: %s/validation_diagnostics/step_000000_interactions.jsonl\n' "$EVAL_DIR"
fi
