#!/usr/bin/env bash
# Qwen3-4B entry point for the modern VERL ALFWorld GiGPO integration.
#
# Full-parameter integration smoke (default):
#   bash examples/alfworld/train_qwen3_4b_modern.sh
# Short full-parameter learning pilot:
#   RUN_MODE=pilot bash examples/alfworld/train_qwen3_4b_modern.sh
# LoRA smoke/pilot after full-parameter learning is verified:
#   TRAINING_VARIANT=lora RUN_MODE=smoke bash examples/alfworld/train_qwen3_4b_modern.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export MODEL_PROFILE=qwen3_4b
export TRAINING_VARIANT=${TRAINING_VARIANT:-full}
export ENTRYPOINT_SCRIPT=$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")

exec bash "$SCRIPT_DIR/train_alfworld_modern.sh" "$@"
