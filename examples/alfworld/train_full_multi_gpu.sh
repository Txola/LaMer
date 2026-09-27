#!/usr/bin/env bash
# Full-parameter Qwen3-4B Meta-RL training for the fixed ALFWorld ID split.
# Defaults to the single 96 GB Blackwell host. Set HARDWARE_PROFILE=small_24gb
# for the released four-GPU, 24 GB-per-GPU layout.
#
# Smoke test (required before production):
#   SMOKE_TEST=1 bash examples/alfworld/train_full_multi_gpu.sh
# Production (run only after reviewing a successful smoke test):
#   ALLOW_PRODUCTION=1 bash examples/alfworld/train_full_multi_gpu.sh
# Resume: rerun the same command with the same OUTPUT_DIR. Immutable experiment
# parameters are checked against run_parameters.txt before Ray is started.
set -Eeuo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

DEFAULT_PYTHON_BIN=python3
if [[ -x "$HOME/miniconda3/envs/lamer/bin/python" ]]; then
    DEFAULT_PYTHON_BIN="$HOME/miniconda3/envs/lamer/bin/python"
fi
PYTHON_BIN=${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}
SMOKE_TEST=${SMOKE_TEST:-0}
ALLOW_PRODUCTION=${ALLOW_PRODUCTION:-0}
DRY_RUN=${DRY_RUN:-0}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}

case "$SMOKE_TEST" in 0|1) ;; *) printf 'SMOKE_TEST must be 0 or 1.\n' >&2; exit 2 ;; esac
case "$ALLOW_PRODUCTION" in 0|1) ;; *) printf 'ALLOW_PRODUCTION must be 0 or 1.\n' >&2; exit 2 ;; esac
case "$DRY_RUN" in 0|1) ;; *) printf 'DRY_RUN must be 0 or 1.\n' >&2; exit 2 ;; esac

if [[ "$SMOKE_TEST" == 1 ]]; then
    RUN_MODE=smoke
    DEFAULT_TOTAL_STEPS=2
    DEFAULT_TEST_FREQ=1
    DEFAULT_VAL_TASK_COUNT=4
    DEFAULT_CAPTURE_GENERATION_DIAGNOSTICS=True
    DEFAULT_GROUPING_DIAGNOSTICS=True
    DEFAULT_EXPERIMENT_NAME="alfworld_full_qwen3_4b_smoke_${RUN_ID}"
    DEFAULT_OUTPUT_DIR="$REPO_ROOT/outputs/alfworld_full_multi_gpu/smoke_${RUN_ID}"
else
    RUN_MODE=production
    DEFAULT_TOTAL_STEPS=150
    DEFAULT_TEST_FREQ=5
    DEFAULT_VAL_TASK_COUNT=84
    DEFAULT_CAPTURE_GENERATION_DIAGNOSTICS=True
    DEFAULT_GROUPING_DIAGNOSTICS=True
    DEFAULT_EXPERIMENT_NAME=alfworld_full_qwen3_4b_main
    DEFAULT_OUTPUT_DIR="$REPO_ROOT/outputs/alfworld_full_multi_gpu/alfworld_full_qwen3_4b_main"
    if [[ "$ALLOW_PRODUCTION" != 1 && "$DRY_RUN" != 1 ]]; then
        printf 'Production is guarded. Run a successful smoke test, then set ALLOW_PRODUCTION=1.\n' >&2
        exit 2
    fi
fi

OUTPUT_DIR=${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd -- "$OUTPUT_DIR" && pwd)
PARAMETERS_FILE="$OUTPUT_DIR/run_parameters.txt"

saved_parameter() {
    local key=$1
    local fallback=$2
    local value=""
    if [[ -f "$PARAMETERS_FILE" ]]; then
        value=$(sed -n "s/^${key}=//p" "$PARAMETERS_FILE" | tail -1)
    fi
    printf '%s' "${value:-$fallback}"
}

assert_resume_parameter() {
    local key=$1
    local current=$2
    local saved=""
    if [[ -f "$PARAMETERS_FILE" ]]; then
        saved=$(sed -n "s/^${key}=//p" "$PARAMETERS_FILE" | tail -1)
        if [[ -n "$saved" && "$saved" != "$current" ]]; then
            printf 'Refusing unsafe resume: %s was %q, requested %q. Use a new OUTPUT_DIR.\n' \
                "$key" "$saved" "$current" >&2
            exit 2
        fi
    fi
}

# Scientific controls. These defaults match the comparison LoRA run.
MODEL_PATH=${MODEL_PATH:-$(saved_parameter MODEL_PATH Qwen/Qwen3-4B)}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$(saved_parameter TRAIN_BATCH_SIZE 8)}
GROUP_SIZE=${GROUP_SIZE:-$(saved_parameter GROUP_SIZE 8)}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$(saved_parameter PPO_MINI_BATCH_SIZE 64)}
TOTAL_STEPS=${TOTAL_STEPS:-$(saved_parameter TOTAL_STEPS "$DEFAULT_TOTAL_STEPS")}
TEST_FREQ=${TEST_FREQ:-$(saved_parameter TEST_FREQ "$DEFAULT_TEST_FREQ")}
# Interval saving is disabled. Validation drives the best-and-final policy.
SAVE_FREQ=-1
CHECKPOINT_POLICY=${CHECKPOINT_POLICY:-$(saved_parameter CHECKPOINT_POLICY best_and_last)}
CHECKPOINT_METRIC=${CHECKPOINT_METRIC:-$(saved_parameter CHECKPOINT_METRIC val/meta_rl/p_at_3)}
CHECKPOINT_METRIC_MODE=${CHECKPOINT_METRIC_MODE:-$(saved_parameter CHECKPOINT_METRIC_MODE max)}
VAL_TASK_COUNT=${VAL_TASK_COUNT:-$(saved_parameter VAL_TASK_COUNT "$DEFAULT_VAL_TASK_COUNT")}
NUM_ATTEMPTS=${NUM_ATTEMPTS:-$(saved_parameter NUM_ATTEMPTS 3)}
MAX_TURNS_PER_ATTEMPT=${MAX_TURNS_PER_ATTEMPT:-$(saved_parameter MAX_TURNS_PER_ATTEMPT 10)}
LEARNING_RATE=${LEARNING_RATE:-$(saved_parameter LEARNING_RATE 1e-6)}
ENV_SEED=${ENV_SEED:-$(saved_parameter ENV_SEED 0)}
VAL_ENV_SEED=${VAL_ENV_SEED:-$(saved_parameter VAL_ENV_SEED 1000)}
ROLLOUT_SEED=${ROLLOUT_SEED:-$(saved_parameter ROLLOUT_SEED 20)}

# Hardware settings do not alter the scientific batch structure. Full FP32
# training on the small profile requires the original four 24 GB GPUs.
HARDWARE_PROFILE=${HARDWARE_PROFILE:-$(saved_parameter HARDWARE_PROFILE large_96gb)}
case "$HARDWARE_PROFILE" in
    large_96gb)
        PROFILE_N_GPUS=1
        PROFILE_TENSOR_PARALLEL_SIZE=1
        PROFILE_ACTOR_MICRO_BATCH_SIZE=16
        PROFILE_LOG_PROB_MICRO_BATCH_SIZE=32
        PROFILE_ACTOR_PARAM_OFFLOAD=False
        # Keep Adam on the 96 GB GPU to avoid a roughly 32 GiB PCIe transfer
        # each update. The smaller vLLM budget leaves room during generation.
        PROFILE_ACTOR_OPTIMIZER_OFFLOAD=False
        PROFILE_GPU_MEMORY_UTILIZATION=0.4
        PROFILE_MAX_NUM_BATCHED_TOKENS=32768
        PROFILE_VLLM_ATTENTION_BACKEND=FLASH_ATTN
        PROFILE_RAY_NUM_CPUS=18
        PROFILE_ENV_CPUS_PER_WORKER=1
        PROFILE_GAMES_PER_ENV_WORKER=32
        ;;
    small_24gb)
        PROFILE_N_GPUS=4
        PROFILE_TENSOR_PARALLEL_SIZE=2
        PROFILE_ACTOR_MICRO_BATCH_SIZE=8
        PROFILE_LOG_PROB_MICRO_BATCH_SIZE=16
        PROFILE_ACTOR_PARAM_OFFLOAD=True
        PROFILE_ACTOR_OPTIMIZER_OFFLOAD=True
        PROFILE_GPU_MEMORY_UTILIZATION=0.6
        PROFILE_MAX_NUM_BATCHED_TOKENS=16384
        PROFILE_VLLM_ATTENTION_BACKEND=XFORMERS
        PROFILE_RAY_NUM_CPUS=$(nproc)
        PROFILE_ENV_CPUS_PER_WORKER=0.1
        PROFILE_GAMES_PER_ENV_WORKER=32
        ;;
    *)
        printf 'HARDWARE_PROFILE must be large_96gb or small_24gb; got %s.\n' \
            "$HARDWARE_PROFILE" >&2
        exit 2
        ;;
esac
N_GPUS=${N_GPUS:-$(saved_parameter N_GPUS "$PROFILE_N_GPUS")}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-$(saved_parameter TENSOR_PARALLEL_SIZE "$PROFILE_TENSOR_PARALLEL_SIZE")}
ACTOR_MICRO_BATCH_SIZE=${ACTOR_MICRO_BATCH_SIZE:-$(saved_parameter ACTOR_MICRO_BATCH_SIZE "$PROFILE_ACTOR_MICRO_BATCH_SIZE")}
LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-$(saved_parameter LOG_PROB_MICRO_BATCH_SIZE "$PROFILE_LOG_PROB_MICRO_BATCH_SIZE")}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-$(saved_parameter ACTOR_PARAM_OFFLOAD "$PROFILE_ACTOR_PARAM_OFFLOAD")}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-$(saved_parameter ACTOR_OPTIMIZER_OFFLOAD "$PROFILE_ACTOR_OPTIMIZER_OFFLOAD")}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-$(saved_parameter REF_PARAM_OFFLOAD True)}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-$(saved_parameter GPU_MEMORY_UTILIZATION "$PROFILE_GPU_MEMORY_UTILIZATION")}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-$(saved_parameter MAX_NUM_BATCHED_TOKENS "$PROFILE_MAX_NUM_BATCHED_TOKENS")}
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-$(saved_parameter VLLM_ATTENTION_BACKEND "$PROFILE_VLLM_ATTENTION_BACKEND")}

CAPTURE_GENERATION_DIAGNOSTICS=${CAPTURE_GENERATION_DIAGNOSTICS:-$(saved_parameter CAPTURE_GENERATION_DIAGNOSTICS "$DEFAULT_CAPTURE_GENERATION_DIAGNOSTICS")}
GROUPING_DIAGNOSTICS=${GROUPING_DIAGNOSTICS:-$(saved_parameter GROUPING_DIAGNOSTICS "$DEFAULT_GROUPING_DIAGNOSTICS")}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-$(saved_parameter MAX_ACTOR_CKPT_TO_KEEP 2)}
EXPECTED_CKPT_GIB=${EXPECTED_CKPT_GIB:-50}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-$(saved_parameter RAY_NUM_CPUS "$PROFILE_RAY_NUM_CPUS")}
ENV_CPUS_PER_WORKER=${ENV_CPUS_PER_WORKER:-$(saved_parameter ENV_CPUS_PER_WORKER "$PROFILE_ENV_CPUS_PER_WORKER")}
GAMES_PER_ENV_WORKER=${GAMES_PER_ENV_WORKER:-$(saved_parameter GAMES_PER_ENV_WORKER "$PROFILE_GAMES_PER_ENV_WORKER")}
ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
RESUME_MODE=${RESUME_MODE:-auto}
TRAINER_LOGGER=${TRAINER_LOGGER:-'[console,wandb]'}
PROJECT_NAME=${PROJECT_NAME:-$(saved_parameter PROJECT_NAME lamer)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-$(saved_parameter EXPERIMENT_NAME "$DEFAULT_EXPERIMENT_NAME")}
WANDB_RUN_ID=${WANDB_RUN_ID:-$(saved_parameter WANDB_RUN_ID "$EXPERIMENT_NAME")}

for boolean_name in ACTOR_PARAM_OFFLOAD ACTOR_OPTIMIZER_OFFLOAD REF_PARAM_OFFLOAD CAPTURE_GENERATION_DIAGNOSTICS GROUPING_DIAGNOSTICS; do
    boolean_value=${!boolean_name}
    if [[ "$boolean_value" != True && "$boolean_value" != False ]]; then
        printf '%s must be True or False; got %s.\n' "$boolean_name" "$boolean_value" >&2
        exit 2
    fi
done
for positive_name in TRAIN_BATCH_SIZE GROUP_SIZE PPO_MINI_BATCH_SIZE TOTAL_STEPS TEST_FREQ VAL_TASK_COUNT NUM_ATTEMPTS MAX_TURNS_PER_ATTEMPT N_GPUS TENSOR_PARALLEL_SIZE ACTOR_MICRO_BATCH_SIZE LOG_PROB_MICRO_BATCH_SIZE MAX_ACTOR_CKPT_TO_KEEP RAY_NUM_CPUS GAMES_PER_ENV_WORKER; do
    positive_value=${!positive_name}
    if [[ ! "$positive_value" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s must be a positive integer; got %s.\n' "$positive_name" "$positive_value" >&2
        exit 2
    fi
done
if (( VAL_TASK_COUNT > 84 )); then
    printf 'VAL_TASK_COUNT cannot exceed the 84-game fixed ID split.\n' >&2; exit 2
fi
if [[ "$CHECKPOINT_POLICY" != best_and_last ]]; then
    printf 'This launcher requires CHECKPOINT_POLICY=best_and_last.\n' >&2; exit 2
fi
if [[ "$CHECKPOINT_METRIC_MODE" != max && "$CHECKPOINT_METRIC_MODE" != min ]]; then
    printf 'CHECKPOINT_METRIC_MODE must be max or min.\n' >&2; exit 2
fi
if (( MAX_ACTOR_CKPT_TO_KEEP != 2 )); then
    printf 'MAX_ACTOR_CKPT_TO_KEEP must be 2 for best-and-final retention.\n' >&2; exit 2
fi
if (( TRAIN_BATCH_SIZE != 8 || GROUP_SIZE != 8 || PPO_MINI_BATCH_SIZE != 64 )); then
    printf 'The fair comparison requires 8 tasks, 8 rollouts/task, and PPO minibatch 64.\n' >&2; exit 2
fi
if [[ "$MODEL_PATH" != Qwen/Qwen3-4B || "$NUM_ATTEMPTS" != 3 || "$MAX_TURNS_PER_ATTEMPT" != 10 || "$LEARNING_RATE" != 1e-6 || "$ENV_SEED" != 0 || "$VAL_ENV_SEED" != 1000 || "$ROLLOUT_SEED" != 20 ]]; then
    printf 'The fair comparison requires Qwen3-4B, 3x10, lr=1e-6, and seeds 0/1000/20.\n' >&2
    exit 2
fi
if [[ "$HARDWARE_PROFILE" == large_96gb && ( "$N_GPUS" != 1 || "$TENSOR_PARALLEL_SIZE" != 1 ) ]]; then
    printf 'large_96gb requires one GPU and tensor parallel size 1.\n' >&2; exit 2
fi
if [[ "$HARDWARE_PROFILE" == small_24gb && ( "$N_GPUS" != 4 || "$TENSOR_PARALLEL_SIZE" != 2 ) ]]; then
    printf 'small_24gb requires four GPUs and tensor parallel size 2.\n' >&2; exit 2
fi
if (( N_GPUS % TENSOR_PARALLEL_SIZE != 0 )); then
    printf 'N_GPUS (%s) must be divisible by TENSOR_PARALLEL_SIZE (%s).\n' "$N_GPUS" "$TENSOR_PARALLEL_SIZE" >&2; exit 2
fi
if (( (PPO_MINI_BATCH_SIZE / N_GPUS) % ACTOR_MICRO_BATCH_SIZE != 0 )); then
    printf 'Per-rank PPO minibatch (%s) must be divisible by actor microbatch (%s).\n' \
        "$((PPO_MINI_BATCH_SIZE / N_GPUS))" "$ACTOR_MICRO_BATCH_SIZE" >&2
    exit 2
fi
if [[ "$RUN_MODE" == smoke && ( "$TOTAL_STEPS" != 2 || "$TEST_FREQ" != 1 || "$VAL_TASK_COUNT" != 4 ) ]]; then
    printf 'Smoke mode requires 2 steps, validation each step, and 4 validation tasks.\n' >&2; exit 2
fi
if [[ "$RUN_MODE" == production && ( "$TOTAL_STEPS" != 150 || "$TEST_FREQ" != 5 || "$VAL_TASK_COUNT" != 84 ) ]]; then
    printf 'Production mode requires 150 steps, validation every 5 steps, and 84 validation tasks.\n' >&2; exit 2
fi

for immutable_name in RUN_MODE MODEL_PATH TRAIN_BATCH_SIZE GROUP_SIZE PPO_MINI_BATCH_SIZE TOTAL_STEPS TEST_FREQ SAVE_FREQ CHECKPOINT_POLICY CHECKPOINT_METRIC CHECKPOINT_METRIC_MODE VAL_TASK_COUNT NUM_ATTEMPTS MAX_TURNS_PER_ATTEMPT LEARNING_RATE ENV_SEED VAL_ENV_SEED ROLLOUT_SEED HARDWARE_PROFILE N_GPUS TENSOR_PARALLEL_SIZE ACTOR_MICRO_BATCH_SIZE LOG_PROB_MICRO_BATCH_SIZE ACTOR_PARAM_OFFLOAD ACTOR_OPTIMIZER_OFFLOAD REF_PARAM_OFFLOAD GPU_MEMORY_UTILIZATION MAX_NUM_BATCHED_TOKENS VLLM_ATTENTION_BACKEND CAPTURE_GENERATION_DIAGNOSTICS GROUPING_DIAGNOSTICS RAY_NUM_CPUS ENV_CPUS_PER_WORKER GAMES_PER_ENV_WORKER PROJECT_NAME EXPERIMENT_NAME WANDB_RUN_ID; do
    assert_resume_parameter "$immutable_name" "${!immutable_name}"
done

CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
TENSORBOARD_DIR=${TENSORBOARD_DIR:-$OUTPUT_DIR/tensorboard}
# Ray's AF_UNIX socket path is capped at 107 bytes. Keep its session root short;
# checkpoints and diagnostics remain under OUTPUT_DIR.
RAY_TMPDIR=${RAY_TMPDIR:-/tmp/lamer-ray-${UID}-full}
mkdir -p "$CHECKPOINT_DIR" "$TENSORBOARD_DIR" "$RAY_TMPDIR" \
    "$OUTPUT_DIR/data" "$OUTPUT_DIR/validation_diagnostics" \
    "$OUTPUT_DIR/grouping_diagnostics"

LATEST_CHECKPOINT_STEP=0
if [[ -f "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt" ]]; then
    LATEST_CHECKPOINT_STEP=$(<"$CHECKPOINT_DIR/latest_checkpointed_iteration.txt")
    if [[ ! "$LATEST_CHECKPOINT_STEP" =~ ^[0-9]+$ ]]; then
        printf 'Invalid checkpoint tracker: %s\n' "$LATEST_CHECKPOINT_STEP" >&2; exit 2
    fi
fi
# VERL restores model, optimizer, scheduler, dataloader, and training RNG. The
# external ALFWorld process is rebuilt, so advance only its training seed.
TRAIN_ENV_SEED=${TRAIN_ENV_SEED:-$((ENV_SEED + LATEST_CHECKPOINT_STEP))}

export ALFWORLD_DATA RAY_TMPDIR
export PYTHONHASHSEED="$ROLLOUT_SEED"
export VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=${VLLM_USE_V1:-0}
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}
export TENSORBOARD_DIR
if [[ ${PYTORCH_CUDA_ALLOC_CONF:-} == *"expandable_segments:True"* ]]; then
    unset PYTORCH_CUDA_ALLOC_CONF
fi
if [[ "$TRAINER_LOGGER" == *wandb* ]]; then
    export WANDB_RUN_ID WANDB_RESUME=${WANDB_RESUME:-allow}
fi

validation_count=$(( (TOTAL_STEPS + TEST_FREQ - 1) / TEST_FREQ ))
max_checkpoint_writes=$validation_count
all_checkpoint_gib=$((max_checkpoint_writes * EXPECTED_CKPT_GIB))
retained_checkpoint_gib=$((MAX_ACTOR_CKPT_TO_KEEP * EXPECTED_CKPT_GIB))
required_free_gib=$((retained_checkpoint_gib + EXPECTED_CKPT_GIB + 32))
available_disk_kib=$(df -Pk "$OUTPUT_DIR" | awk 'NR==2 {print $4}')
available_disk_gib=$((available_disk_kib / 1024 / 1024))

printf 'Run mode: %s\nHardware profile: %s\nOutput: %s\n' "$RUN_MODE" "$HARDWARE_PROFILE" "$OUTPUT_DIR"
printf 'Distributed layout: %s GPUs, FSDP, tensor parallel %s, actor microbatch %s/GPU, log-prob microbatch %s/GPU.\n' \
    "$N_GPUS" "$TENSOR_PARALLEL_SIZE" "$ACTOR_MICRO_BATCH_SIZE" "$LOG_PROB_MICRO_BATCH_SIZE"
printf 'Checkpoint plan: %s validations; at most %s writes including final, ~%s GiB each, ~%s GiB without pruning.\n' \
    "$validation_count" "$max_checkpoint_writes" "$EXPECTED_CKPT_GIB" "$all_checkpoint_gib"
printf 'Retention: strict best %s plus final/latest; at most %s checkpoints (~%s GiB).\n' \
    "$CHECKPOINT_METRIC" "$MAX_ACTOR_CKPT_TO_KEEP" "$retained_checkpoint_gib"
printf 'Disk available: %s GiB; conservative launch requirement: %s GiB. Ray temp: %s\n' \
    "$available_disk_gib" "$required_free_gib" "$RAY_TMPDIR"
if (( available_disk_gib < required_free_gib )); then
    printf 'Insufficient free disk for checkpoint retention plus one in-progress checkpoint.\n' >&2; exit 2
fi

if [[ "$DRY_RUN" != 1 ]]; then
    "$PYTHON_BIN" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] <= (3, 12)):
    raise SystemExit(
        f"Python {sys.version.split()[0]} is unsupported for this ALFWorld stack; "
        "use Python 3.10-3.12 (3.12 preferred)."
    )
PY
    command -v nvidia-smi >/dev/null || { printf 'nvidia-smi is unavailable.\n' >&2; exit 2; }
    gpu_rows=$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits) || {
        printf 'nvidia-smi cannot communicate with the NVIDIA driver.\n' >&2; exit 2;
    }
    visible_gpu_count=$(printf '%s\n' "$gpu_rows" | sed '/^[[:space:]]*$/d' | wc -l)
    if (( visible_gpu_count < N_GPUS )); then
        printf 'Need %s visible GPUs, but nvidia-smi reports %s.\n' "$N_GPUS" "$visible_gpu_count" >&2; exit 2
    fi
    "$PYTHON_BIN" - <<PY
import importlib
import importlib.util
from packaging.version import Version
required = (
    "alfworld", "datasets", "flash_attn", "hydra", "peft", "ray",
    "textworld", "torch", "transformers", "vllm", "wandb",
)
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing training packages: " + ", ".join(missing))
torch = importlib.import_module("torch")
transformers = importlib.import_module("transformers")
if Version(transformers.__version__).major >= 5:
    raise SystemExit(
        f"Transformers {transformers.__version__} is incompatible with this "
        "LaMer/vLLM adapter; install a compatible Transformers 4 release."
    )
try:
    importlib.import_module("flash_attn")
except (ImportError, OSError) as exc:
    raise SystemExit(
        "flash-attn cannot load with torch " + torch.__version__ + ". "
        f"Original error: {exc}"
    ) from exc
assert torch.cuda.is_available(), "CUDA is not available to PyTorch"
assert torch.cuda.device_count() >= $N_GPUS, (
    f"PyTorch sees {torch.cuda.device_count()} GPUs; $N_GPUS required"
)
if torch.cuda.get_device_capability(0)[0] >= 12:
    cuda_major_minor = tuple(map(int, torch.version.cuda.split(".")[:2]))
    assert cuda_major_minor >= (12, 8), (
        f"Blackwell requires a CUDA >=12.8 PyTorch build; found {torch.version.cuda}"
    )
    assert "sm_120" in torch.cuda.get_arch_list(), (
        f"PyTorch wheel lacks sm_120 support: {torch.cuda.get_arch_list()}"
    )
PY
    if [[ "$TRAINER_LOGGER" == *wandb* && ${WANDB_MODE:-online} != offline && ${WANDB_MODE:-online} != disabled ]]; then
        "$PYTHON_BIN" -c 'import sys, wandb; sys.exit(0 if wandb.api.api_key else "W&B is enabled but no API key is configured")'
    fi

    if [[ ! -f "$ALFWORLD_DATA/json/split_manifest.json" || ! -d "$ALFWORLD_DATA/json/valid_id_task_balanced84" ]]; then
        if [[ -d "$ALFWORLD_DATA/json_2.1.1/train" && -d "$ALFWORLD_DATA/json_2.1.1/valid_train" && -d "$ALFWORLD_DATA/json_2.1.1/valid_seen" ]]; then
            "$PYTHON_BIN" scripts/prepare_alfworld_lamer_splits.py --replace
        else
            printf 'Missing official ALFWorld data and prepared split under %s. Install/download ALFWorld data first.\n' "$ALFWORLD_DATA" >&2
            exit 2
        fi
    fi
    "$PYTHON_BIN" - "$ALFWORLD_DATA/json/split_manifest.json" <<'PY'
import json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
expected = {
    "pick_and_place_simple": 21,
    "look_at_obj_in_light": 21,
    "pick_clean_then_place_in_recep": 21,
    "pick_heat_then_place_in_recep": 21,
}
checkpoint = manifest["checkpoint_evaluation"]
assert checkpoint["id_split"] == "valid_id_task_balanced84", checkpoint
assert checkpoint["id_count"] == 84, checkpoint
actual = manifest["splits"]["valid_id_task_balanced84"]["task_counts"]
assert actual == expected, (actual, expected)
sources = {pathlib.Path(path).name for path in manifest["validation_sources"]}
assert sources == {"valid_train", "valid_seen"}, sources
PY
fi

cp -- "${BASH_SOURCE[0]}" "$OUTPUT_DIR/launcher.sh"
cp -- "$ALFWORLD_DATA/json/split_manifest.json" "$OUTPUT_DIR/split_manifest.json" 2>/dev/null || true
git rev-parse HEAD > "$OUTPUT_DIR/git_commit.txt"
git status --short > "$OUTPUT_DIR/git_status.txt"
git diff --no-ext-diff > "$OUTPUT_DIR/working_tree.patch"
"$PYTHON_BIN" --version > "$OUTPUT_DIR/python_version.txt" 2>&1 || true
nvidia-smi > "$OUTPUT_DIR/nvidia_smi.txt" 2>&1 || true
{
    "$PYTHON_BIN" - <<'PY' 2>&1 || true
import importlib.metadata as md
import torch
for name in ("alfworld", "textworld", "ray", "torch", "transformers", "vllm", "wandb"):
    try: print(f"{name}={md.version(name)}")
    except md.PackageNotFoundError: print(f"{name}=MISSING")
print(f"torch_cuda={getattr(torch.version, 'cuda', None)}")
PY
    command -v nvcc >/dev/null && nvcc --version || true
} > "$OUTPUT_DIR/software_versions.txt"

{
    for name in RUN_MODE HARDWARE_PROFILE MODEL_PATH TRAIN_BATCH_SIZE GROUP_SIZE PPO_MINI_BATCH_SIZE TOTAL_STEPS SAVE_FREQ TEST_FREQ CHECKPOINT_POLICY CHECKPOINT_METRIC CHECKPOINT_METRIC_MODE VAL_TASK_COUNT NUM_ATTEMPTS MAX_TURNS_PER_ATTEMPT LEARNING_RATE ENV_SEED TRAIN_ENV_SEED VAL_ENV_SEED ROLLOUT_SEED N_GPUS TENSOR_PARALLEL_SIZE ACTOR_MICRO_BATCH_SIZE LOG_PROB_MICRO_BATCH_SIZE ACTOR_PARAM_OFFLOAD ACTOR_OPTIMIZER_OFFLOAD REF_PARAM_OFFLOAD GPU_MEMORY_UTILIZATION MAX_NUM_BATCHED_TOKENS VLLM_ATTENTION_BACKEND CAPTURE_GENERATION_DIAGNOSTICS GROUPING_DIAGNOSTICS MAX_ACTOR_CKPT_TO_KEEP EXPECTED_CKPT_GIB RESUME_MODE TRAINER_LOGGER PROJECT_NAME EXPERIMENT_NAME WANDB_RUN_ID LATEST_CHECKPOINT_STEP OUTPUT_DIR CHECKPOINT_DIR TENSORBOARD_DIR RAY_TMPDIR; do
        printf '%s=%s\n' "$name" "${!name}"
    done
    printf 'LORA_RANK=0\n'
    printf 'SAVE_LORA_ONLY=False\n'
    printf 'MODEL_DTYPE=fp32_default\n'
    printf 'VALIDATIONS_PLANNED=%s\n' "$validation_count"
    printf 'CHECKPOINT_WRITES_MAX=%s\n' "$max_checkpoint_writes"
    printf 'CHECKPOINT_STORAGE_ALL_GIB=%s\n' "$all_checkpoint_gib"
    printf 'CHECKPOINT_STORAGE_RETAINED_GIB=%s\n' "$retained_checkpoint_gib"
} > "$PARAMETERS_FILE"

if [[ "$DRY_RUN" != 1 && ( ! -f "$OUTPUT_DIR/data/text/train.parquet" || ! -f "$OUTPUT_DIR/data/text/test.parquet" ) ]]; then
    "$PYTHON_BIN" -m examples.data_preprocess.prepare \
        --mode text \
        --local_dir "$OUTPUT_DIR/data" \
        --train_data_size "$TRAIN_BATCH_SIZE" \
        --val_data_size "$VAL_TASK_COUNT" \
        2>&1 | tee "$OUTPUT_DIR/prepare.log"
fi

cmd=(
    "$PYTHON_BIN" -m verl.trainer.main_ppo
    algorithm.adv_estimator=gigpo
    "data.train_files=$OUTPUT_DIR/data/text/train.parquet"
    "data.val_files=$OUTPUT_DIR/data/text/test.parquet"
    "data.train_batch_size=$TRAIN_BATCH_SIZE"
    "data.val_batch_size=$VAL_TASK_COUNT"
    data.max_prompt_length=4096
    data.max_response_length=1024
    data.filter_overlong_prompts=True
    data.truncation=error
    data.return_raw_chat=True
    data.shuffle=False
    "+data.seed=$ROLLOUT_SEED"
    "actor_rollout_ref.model.path=$MODEL_PATH"
    +actor_rollout_ref.model.enable_thinking=False
    actor_rollout_ref.model.lora_rank=0
    "actor_rollout_ref.actor.optim.lr=$LEARNING_RATE"
    actor_rollout_ref.actor.checkpoint.save_lora_only=False
    actor_rollout_ref.model.use_remove_padding=True
    "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ACTOR_MICRO_BATCH_SIZE"
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.strategy=fsdp
    "actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAM_OFFLOAD"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL_SIZE"
    actor_rollout_ref.rollout.name=vllm
    "+actor_rollout_ref.rollout.seed=$ROLLOUT_SEED"
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=True
    "actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=False
    "+actor_rollout_ref.rollout.capture_generation_diagnostics=$CAPTURE_GENERATION_DIAGNOSTICS"
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8
    actor_rollout_ref.rollout.val_kwargs.top_k=20
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    "+actor_rollout_ref.rollout.val_kwargs.seed=$ROLLOUT_SEED"
    "actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE"
    "actor_rollout_ref.ref.fsdp_config.param_offload=$REF_PARAM_OFFLOAD"
    actor_rollout_ref.actor.use_invalid_action_penalty=True
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.5
    algorithm.use_kl_in_reward=False
    algorithm.gamma=0.95
    +algorithm.step_gamma=0.95
    +algorithm.traj_gamma=0.6
    algorithm.gigpo.step_advantage_w=1.0
    algorithm.gigpo.mode=mean_norm
    reward_model.reward_manager=episode
    env.env_name=alfworld/AlfredTWEnv
    "env.seed=$TRAIN_ENV_SEED"
    "+env.val_seed=$VAL_ENV_SEED"
    "env.rollout.n=$GROUP_SIZE"
    "env.num_attempts=$NUM_ATTEMPTS"
    +env.do_reflection=True
    env.max_steps=30
    "env.max_turns=$MAX_TURNS_PER_ATTEMPT"
    +env.reflection_type=reflection_only
    env.alfworld.eval_dataset=eval_id_checkpoint
    "env.alfworld.games_per_worker=$GAMES_PER_ENV_WORKER"
    "env.resources_per_worker.num_cpus=$ENV_CPUS_PER_WORKER"
    env.resources_per_worker.num_gpus=0
    trainer.critic_warmup=0
    "trainer.logger=$TRAINER_LOGGER"
    "trainer.project_name=$PROJECT_NAME"
    "trainer.experiment_name=$EXPERIMENT_NAME"
    "trainer.n_gpus_per_node=$N_GPUS"
    trainer.nnodes=1
    "trainer.save_freq=$SAVE_FREQ"
    "trainer.test_freq=$TEST_FREQ"
    "trainer.checkpoint_policy=$CHECKPOINT_POLICY"
    "trainer.checkpoint_metric=$CHECKPOINT_METRIC"
    "trainer.checkpoint_metric_mode=$CHECKPOINT_METRIC_MODE"
    "trainer.total_epochs=$TOTAL_STEPS"
    "trainer.total_training_steps=$TOTAL_STEPS"
    "trainer.resume_mode=$RESUME_MODE"
    trainer.val_before_train=False
    trainer.log_val_generations=0
    "trainer.max_actor_ckpt_to_keep=$MAX_ACTOR_CKPT_TO_KEEP"
    trainer.max_critic_ckpt_to_keep=null
    trainer.validation_dump_all_interactions=False
    trainer.validation_trajectory_samples_per_task=1
    trainer.validation_trajectory_sample_seed=0
    "trainer.validation_data_dir=$OUTPUT_DIR/validation_diagnostics"
    "trainer.grouping_diagnostics.enabled=$GROUPING_DIAGNOSTICS"
    "trainer.grouping_diagnostics.output_dir=$OUTPUT_DIR/grouping_diagnostics"
    "trainer.default_local_dir=$CHECKPOINT_DIR"
    "ray_init.num_cpus=$RAY_NUM_CPUS"
    +ray_init.include_dashboard=False
    "hydra.run.dir=$OUTPUT_DIR/hydra"
)

printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'
if [[ "$DRY_RUN" == 1 ]]; then
    exit 0
fi

"${cmd[@]}" 2>&1 | tee -a "$OUTPUT_DIR/train.log"

printf 'Training directory: %s\n' "$OUTPUT_DIR"
printf 'Checkpoints: %s\n' "$CHECKPOINT_DIR"
printf 'Validation diagnostics: %s\n' "$OUTPUT_DIR/validation_diagnostics"
