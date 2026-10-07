#!/usr/bin/env bash
# Qwen3.5-9B full-parameter diagnostic entry point for modern VERL.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export MODEL_PROFILE=qwen35_9b
export TRAINING_VARIANT=${TRAINING_VARIANT:-full}
export ENTRYPOINT_SCRIPT=$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")

exec bash "$SCRIPT_DIR/train_alfworld_modern.sh" "$@"
