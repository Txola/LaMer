#!/usr/bin/env bash
# Qwen3.5-9B language-only LoRA GiGPO entry point.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export ENTRYPOINT_SCRIPT=$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")
export TRAINING_VARIANT=lora

exec bash "$SCRIPT_DIR/train_full_qwen35_9b.sh" "$@"
