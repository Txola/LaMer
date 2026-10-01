#!/usr/bin/env bash
# Frozen-policy diagnostic for appended, assisted, or merged peer reflection.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

DEFAULT_PYTHON_BIN=python3
if [[ -x "$HOME/miniconda3/envs/lamer/bin/python" ]]; then
    DEFAULT_PYTHON_BIN="$HOME/miniconda3/envs/lamer/bin/python"
fi
PYTHON_BIN=${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}
: "${MODEL_PATH:?Set MODEL_PATH to a Hugging Face model or merged checkpoint directory}"

HARDWARE_PROFILE=${HARDWARE_PROFILE:-small_24gb}
case "$HARDWARE_PROFILE" in
    small_24gb)
        PROFILE_ATTENTION_BACKEND=XFORMERS
        PROFILE_GPU_MEMORY_UTILIZATION=0.6
        PROFILE_MAX_NUM_BATCHED_TOKENS=16384
        PROFILE_RAY_NUM_CPUS=$(nproc)
        PROFILE_ENV_CPUS_PER_WORKER=0.1
        PROFILE_GAMES_PER_WORKER=32
        PROFILE_NUM_TASKS=8
        ;;
    large_96gb)
        PROFILE_ATTENTION_BACKEND=FLASH_ATTN
        PROFILE_GPU_MEMORY_UTILIZATION=0.8
        PROFILE_MAX_NUM_BATCHED_TOKENS=32768
        PROFILE_RAY_NUM_CPUS=18
        PROFILE_ENV_CPUS_PER_WORKER=1
        PROFILE_GAMES_PER_WORKER=16
        PROFILE_NUM_TASKS=24
        ;;
    *)
        printf 'HARDWARE_PROFILE must be small_24gb or large_96gb; got %s\n' \
            "$HARDWARE_PROFILE" >&2
        exit 2
        ;;
esac

ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/alfworld_failed_peer_reflections}
EXPERIMENT_VARIANT=${EXPERIMENT_VARIANT:-appended_reflections}
EVAL_DATASET=${EVAL_DATASET:-eval_all}
NUM_TASKS=${NUM_TASKS:-$PROFILE_NUM_TASKS}
GROUP_SIZE=${GROUP_SIZE:-8}
MAX_FAILED_RECIPIENTS=${MAX_FAILED_RECIPIENTS:-160}
MAX_TURNS=${MAX_TURNS:-10}
MAX_RESPONSE_TOKENS=${MAX_RESPONSE_TOKENS:-1024}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
VAL_ENV_SEED=${VAL_ENV_SEED:-1000}
ENGINE_SEED=${ENGINE_SEED:-20}
DONOR_SEED=${DONOR_SEED:-22}
RECIPIENT_SEED=${RECIPIENT_SEED:-23}
RETRY_SEEDS=${RETRY_SEEDS:-24}
BOOTSTRAP_SEED=${BOOTSTRAP_SEED:-25}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-2000}
QUALITATIVE_EXAMPLES=${QUALITATIVE_EXAMPLES:-6}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.8}
TOP_K=${TOP_K:-20}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-$PROFILE_GPU_MEMORY_UTILIZATION}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-$PROFILE_MAX_NUM_BATCHED_TOKENS}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-$PROFILE_RAY_NUM_CPUS}
ENV_CPUS_PER_WORKER=${ENV_CPUS_PER_WORKER:-$PROFILE_ENV_CPUS_PER_WORKER}
GAMES_PER_ENV_WORKER=${GAMES_PER_ENV_WORKER:-$PROFILE_GAMES_PER_WORKER}
RAY_TMPDIR=${RAY_TMPDIR:-/tmp/lamer-peer-ray-$UID}
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-$PROFILE_ATTENTION_BACKEND}

export ALFWORLD_DATA VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=${VLLM_USE_V1:-0}
export PYTHONHASHSEED=$ENGINE_SEED
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

if [[ ! -d "$ALFWORLD_DATA/json/valid_task_balanced126" ]]; then
    printf 'Missing prepared balanced126 split under %s/json.\n' "$ALFWORLD_DATA" >&2
    exit 2
fi
if [[ -d "$MODEL_PATH" ]]; then
    MODEL_PATH=$(cd -- "$MODEL_PATH" && pwd -P)
fi
mkdir -p "$OUTPUT_DIR" "$RAY_TMPDIR"
OUTPUT_DIR=$(cd -- "$OUTPUT_DIR" && pwd -P)

git rev-parse HEAD > "$OUTPUT_DIR/code_revision.txt"
git status --short > "$OUTPUT_DIR/worktree_status.txt"
git diff --no-ext-diff > "$OUTPUT_DIR/working_tree.patch"
cp -- "${BASH_SOURCE[0]}" "$OUTPUT_DIR/launcher.sh"
cp -- "$ALFWORLD_DATA/json/split_manifest.json" "$OUTPUT_DIR/split_manifest.json"
mkdir -p "$OUTPUT_DIR/source_snapshot"
cp --parents \
    scripts/evaluate_alfworld_peer_reflections.py \
    agent_system/environments/alfworld/prompt.py \
    agent_system/environments/alfworld/env_manager.py \
    agent_system/environments/alfworld/envs.py \
    "$OUTPUT_DIR/source_snapshot"
"$PYTHON_BIN" -c 'import platform, ray, torch, transformers, vllm; print(platform.python_version()); print("torch", torch.__version__, "cuda", torch.version.cuda); print("transformers", transformers.__version__); print("ray", ray.__version__); print("vllm", vllm.__version__)' \
    > "$OUTPUT_DIR/software_versions.txt"
nvidia-smi > "$OUTPUT_DIR/gpu_info.txt" 2>&1 || true

printf 'Running frozen failed-peer reflection diagnostic\n'
printf '  variant: %s\n  model: %s\n  tasks: %s\n  group size: %s\n  output: %s\n' \
    "$EXPERIMENT_VARIANT" "$MODEL_PATH" "$NUM_TASKS" "$GROUP_SIZE" "$OUTPUT_DIR"

"$PYTHON_BIN" scripts/evaluate_alfworld_peer_reflections.py \
    --experiment-variant "$EXPERIMENT_VARIANT" \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --alfworld-data "$ALFWORLD_DATA" \
    --eval-dataset "$EVAL_DATASET" \
    --num-tasks "$NUM_TASKS" \
    --group-size "$GROUP_SIZE" \
    --max-failed-recipients "$MAX_FAILED_RECIPIENTS" \
    --max-turns "$MAX_TURNS" \
    --max-response-tokens "$MAX_RESPONSE_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --validation-seed "$VAL_ENV_SEED" \
    --donor-seed "$DONOR_SEED" \
    --recipient-seed "$RECIPIENT_SEED" \
    --retry-seeds "$RETRY_SEEDS" \
    --engine-seed "$ENGINE_SEED" \
    --bootstrap-seed "$BOOTSTRAP_SEED" \
    --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
    --qualitative-examples "$QUALITATIVE_EXAMPLES" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --ray-num-cpus "$RAY_NUM_CPUS" \
    --env-cpus-per-worker "$ENV_CPUS_PER_WORKER" \
    --games-per-worker "$GAMES_PER_ENV_WORKER" \
    --ray-tmpdir "$RAY_TMPDIR" \
    2>&1 | tee "$OUTPUT_DIR/eval.log"
