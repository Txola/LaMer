#!/usr/bin/env bash
# Merge a modern single-GPU VERL LoRA checkpoint into its frozen HF base.
set -Eeuo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
COMPAT_DIR=${COMPAT_DIR:-$REPO_ROOT/.compat}
VERL_DIR=${VERL_DIR:-$COMPAT_DIR/verl-upstream}
PYTHON_BIN=${PYTHON_BIN:-$VERL_DIR/.venv/bin/python}

if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'merge_fsdp_lora_checkpoint: missing modern VERL Python: %s\n' \
        "$PYTHON_BIN" >&2
    printf 'Run examples/alfworld/setup_qwen35_compat.sh first or set PYTHON_BIN.\n' >&2
    exit 2
fi

exec "$PYTHON_BIN" "$REPO_ROOT/scripts/merge_fsdp_lora_checkpoint.py" "$@"
