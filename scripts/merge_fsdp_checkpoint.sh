#!/usr/bin/env bash
# Safely convert a verl FSDP actor checkpoint into Hugging Face format.
set -Eeuo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DEFAULT_PYTHON_BIN=python3
if [[ -x "$HOME/miniconda3/envs/lamer/bin/python" ]]; then
    DEFAULT_PYTHON_BIN="$HOME/miniconda3/envs/lamer/bin/python"
fi
PYTHON_BIN=${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}
FORCE=0
TARGET_INPUT=""
CHECKPOINT_INPUT=""

usage() {
    cat >&2 <<'EOF'
Usage:
  scripts/merge_fsdp_checkpoint.sh [--target-dir DIR] [--force] CHECKPOINT

CHECKPOINT may be either a verl global_step_N directory or its actor
subdirectory. The default output is global_step_N/hf_model.

The script prints only the absolute merged-model path on stdout, allowing:

  MODEL_PATH=$(bash scripts/merge_fsdp_checkpoint.sh path/to/global_step_5)

Options:
  --target-dir DIR  Write the Hugging Face model to DIR.
  --force           Replace an existing target after a new merge succeeds.
  -h, --help        Show this help text.

Environment:
  PYTHON_BIN         Python interpreter used to run scripts/model_merger.py.
EOF
}

die() {
    printf 'merge_fsdp_checkpoint: %s\n' "$*" >&2
    exit 2
}

while (( $# > 0 )); do
    case "$1" in
        --target-dir)
            (( $# >= 2 )) || die "--target-dir requires a value"
            TARGET_INPUT=$2
            shift 2
            ;;
        --force)
            FORCE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            (( $# == 1 )) || die "expected exactly one checkpoint path after --"
            CHECKPOINT_INPUT=$1
            shift
            ;;
        -*)
            die "unknown option: $1"
            ;;
        *)
            [[ -z "$CHECKPOINT_INPUT" ]] || die "expected exactly one checkpoint path"
            CHECKPOINT_INPUT=$1
            shift
            ;;
    esac
done

[[ -n "$CHECKPOINT_INPUT" ]] || {
    usage
    exit 2
}
[[ -x "$PYTHON_BIN" ]] || command -v "$PYTHON_BIN" >/dev/null 2>&1 || \
    die "Python interpreter not found: $PYTHON_BIN"
[[ -f "$REPO_ROOT/scripts/model_merger.py" ]] || \
    die "missing merger: $REPO_ROOT/scripts/model_merger.py"
[[ -d "$CHECKPOINT_INPUT" ]] || die "checkpoint directory does not exist: $CHECKPOINT_INPUT"

CHECKPOINT_INPUT=$(cd -- "$CHECKPOINT_INPUT" && pwd -P)
if [[ -d "$CHECKPOINT_INPUT/actor" ]]; then
    CHECKPOINT_ROOT=$CHECKPOINT_INPUT
    ACTOR_DIR="$CHECKPOINT_INPUT/actor"
else
    ACTOR_DIR=$CHECKPOINT_INPUT
    CHECKPOINT_ROOT=$(cd -- "$ACTOR_DIR/.." && pwd -P)
fi

[[ -f "$ACTOR_DIR/config.json" ]] || die "missing actor config: $ACTOR_DIR/config.json"

shopt -s nullglob
MODEL_SHARDS=("$ACTOR_DIR"/model_world_size_*_rank_*.pt)
shopt -u nullglob
(( ${#MODEL_SHARDS[@]} > 0 )) || \
    die "no model_world_size_N_rank_R.pt shards found under $ACTOR_DIR"

WORLD_SIZE=""
declare -A SEEN_RANKS=()
for shard in "${MODEL_SHARDS[@]}"; do
    filename=${shard##*/}
    if [[ ! "$filename" =~ ^model_world_size_([0-9]+)_rank_([0-9]+)\.pt$ ]]; then
        die "unexpected model shard name: $filename"
    fi
    shard_world_size=${BASH_REMATCH[1]}
    shard_rank=${BASH_REMATCH[2]}
    if [[ -z "$WORLD_SIZE" ]]; then
        WORLD_SIZE=$shard_world_size
    elif [[ "$WORLD_SIZE" != "$shard_world_size" ]]; then
        die "model shards declare inconsistent world sizes"
    fi
    (( shard_rank < WORLD_SIZE )) || \
        die "rank $shard_rank is outside declared world size $WORLD_SIZE"
    SEEN_RANKS[$shard_rank]=1
done

(( ${#SEEN_RANKS[@]} == WORLD_SIZE )) || \
    die "expected $WORLD_SIZE distinct model shards, found ${#SEEN_RANKS[@]}"
for (( rank = 0; rank < WORLD_SIZE; rank++ )); do
    [[ -n ${SEEN_RANKS[$rank]+x} ]] || die "missing model shard for rank $rank"
done

if [[ -z "$TARGET_INPUT" ]]; then
    TARGET_INPUT="$CHECKPOINT_ROOT/hf_model"
fi
target_name=$(basename -- "$TARGET_INPUT")
[[ -n "$target_name" && "$target_name" != "." && "$target_name" != ".." ]] || \
    die "target must name a model directory"
target_parent_input=$(dirname -- "$TARGET_INPUT")
mkdir -p -- "$target_parent_input"
TARGET_PARENT=$(cd -- "$target_parent_input" && pwd -P)
TARGET_DIR="$TARGET_PARENT/$target_name"
MARKER_NAME=.merge_complete.json

has_merged_model() {
    local directory=$1
    [[ -f "$directory/config.json" ]] || return 1
    [[ -f "$directory/tokenizer_config.json" ]] || return 1
    compgen -G "$directory/*.safetensors" >/dev/null || \
        compgen -G "$directory/pytorch_model*.bin" >/dev/null
}

marker_matches_source() {
    local marker=$1
    "$PYTHON_BIN" - "$marker" "$ACTOR_DIR" <<'PY'
import json
import pathlib
import sys

marker = pathlib.Path(sys.argv[1])
source = pathlib.Path(sys.argv[2]).resolve()
try:
    metadata = json.loads(marker.read_text(encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if pathlib.Path(metadata.get("source_actor_dir", "")).resolve() == source else 1)
PY
}

if [[ -L "$TARGET_DIR" ]]; then
    die "refusing symbolic-link target: $TARGET_DIR"
fi
if [[ -e "$TARGET_DIR" ]]; then
    [[ -d "$TARGET_DIR" ]] || die "target exists and is not a directory: $TARGET_DIR"
    if (( FORCE == 0 )) && \
            has_merged_model "$TARGET_DIR" && \
            [[ -f "$TARGET_DIR/$MARKER_NAME" ]] && \
            marker_matches_source "$TARGET_DIR/$MARKER_NAME"; then
        printf 'Reusing completed merge: %s\n' "$TARGET_DIR" >&2
        printf '%s\n' "$TARGET_DIR"
        exit 0
    fi
    (( FORCE == 1 )) || die \
        "target exists but is incomplete or from another checkpoint: $TARGET_DIR (use --force to replace it)"
fi

case "$TARGET_DIR" in
    /|"$HOME"|"$REPO_ROOT"|"$CHECKPOINT_ROOT"|"$ACTOR_DIR")
        die "refusing unsafe target path: $TARGET_DIR"
        ;;
esac

source_bytes=0
for shard in "${MODEL_SHARDS[@]}"; do
    shard_bytes=$(stat -c '%s' -- "$shard")
    source_bytes=$((source_bytes + shard_bytes))
done
required_bytes=$((source_bytes + source_bytes / 10 + 536870912))
available_bytes=$(df --output=avail -B1 "$TARGET_PARENT" | tail -n 1 | tr -d ' ')
[[ "$available_bytes" =~ ^[0-9]+$ ]] || die "could not determine free disk space"
(( available_bytes >= required_bytes )) || die \
    "insufficient disk space: need about $required_bytes bytes, have $available_bytes bytes"

mem_available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
if [[ "$mem_available_kib" =~ ^[0-9]+$ ]] && \
        (( mem_available_kib * 1024 < source_bytes * 2 )); then
    printf 'Warning: available RAM is less than twice the model-shard size; merging may use swap.\n' >&2
fi

TMP_DIR=$(mktemp -d "$TARGET_PARENT/.${target_name}.merge-tmp.XXXXXX")
cleanup() {
    if [[ -n ${TMP_DIR:-} && -d "$TMP_DIR" ]]; then
        rm -rf -- "$TMP_DIR"
    fi
}
trap cleanup EXIT

printf 'Merging FSDP actor checkpoint:\n  source: %s\n  target: %s\n  world size: %s\n' \
    "$ACTOR_DIR" "$TARGET_DIR" "$WORLD_SIZE" >&2
PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" "$REPO_ROOT/scripts/model_merger.py" merge \
    --backend fsdp \
    --local_dir "$ACTOR_DIR" \
    --target_dir "$TMP_DIR" >&2

has_merged_model "$TMP_DIR" || die "merger exited successfully but produced no complete Hugging Face model"

git_revision=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || printf 'unknown')
"$PYTHON_BIN" - "$TMP_DIR/$MARKER_NAME" "$ACTOR_DIR" "$WORLD_SIZE" \
        "$source_bytes" "$git_revision" <<'PY'
import datetime
import json
import pathlib
import sys

marker, source, world_size, source_bytes, git_revision = sys.argv[1:]
metadata = {
    "source_actor_dir": str(pathlib.Path(source).resolve()),
    "source_world_size": int(world_size),
    "source_model_bytes": int(source_bytes),
    "repository_git_revision": git_revision,
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "merger": "scripts/model_merger.py",
}
pathlib.Path(marker).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
PY

if [[ -e "$TARGET_DIR" ]]; then
    rm -rf -- "$TARGET_DIR"
fi
mv -- "$TMP_DIR" "$TARGET_DIR"
TMP_DIR=""
trap - EXIT

printf 'Merge complete: %s\n' "$TARGET_DIR" >&2
printf '%s\n' "$TARGET_DIR"
