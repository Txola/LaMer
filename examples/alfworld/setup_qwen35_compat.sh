#!/usr/bin/env bash
# Build an isolated, pinned Qwen3.5/modern-VERL environment without changing lamer.

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
COMPAT_DIR=${COMPAT_DIR:-$REPO_ROOT/.compat}
PYTHON_BIN=${PYTHON_BIN:-python}
UV_VERSION=0.12.19
PYTHON_VERSION=3.12.14
VERL_REVISION=fbb4b3a8bf636f290c9c59fc346f756849e9c241
MODEL_REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a
UV_ENV=$COMPAT_DIR/uv-bootstrap
UV_PYTHON_DIR=$COMPAT_DIR/python
VERL_DIR=$COMPAT_DIR/verl-upstream
UV_CACHE_DIR=${UV_CACHE_DIR:-$COMPAT_DIR/uv-cache}
HF_CACHE_DIR=${HF_CACHE_DIR:-$COMPAT_DIR/hf-cache}

mkdir -p "$COMPAT_DIR" "$UV_CACHE_DIR" "$HF_CACHE_DIR"
if [[ ! -x "$UV_ENV/bin/uv" ]]; then
    "$PYTHON_BIN" -m venv "$UV_ENV"
    "$UV_ENV/bin/python" -m pip install "uv==$UV_VERSION"
fi

# Do not inherit whichever Python happens to own the active shell.  Keep an
# exact patched runtime inside the ignored compatibility directory along with
# the rest of this reproducible environment.
UV_CACHE_DIR="$UV_CACHE_DIR" UV_PYTHON_INSTALL_DIR="$UV_PYTHON_DIR" \
    "$UV_ENV/bin/uv" python install --managed-python "$PYTHON_VERSION"
MANAGED_PYTHON=$(
    UV_CACHE_DIR="$UV_CACHE_DIR" UV_PYTHON_INSTALL_DIR="$UV_PYTHON_DIR" \
        "$UV_ENV/bin/uv" python find --managed-python "$PYTHON_VERSION"
)

if [[ ! -d "$VERL_DIR/.git" ]]; then
    git clone https://github.com/verl-project/verl.git "$VERL_DIR"
elif [[ -n "$(git -C "$VERL_DIR" status --porcelain --untracked-files=no)" ]]; then
    printf 'Refusing to replace tracked changes in %s.\n' "$VERL_DIR" >&2
    exit 2
fi

git -C "$VERL_DIR" fetch origin "$VERL_REVISION"
git -C "$VERL_DIR" checkout --detach "$VERL_REVISION"

UV_CACHE_DIR="$UV_CACHE_DIR" "$UV_ENV/bin/uv" sync \
    --project "$VERL_DIR" \
    --frozen \
    --extra fsdp \
    --extra vllm \
    --python "$MANAGED_PYTHON"

# These are the only LaMer/ALFWorld additions to upstream VERL's locked stack.
# NumPy 2.2.6 satisfies both TextWorld's tested environment and modern
# mistral-common/numba constraints; upstream's broad override otherwise chooses
# NumPy 2.4.x.
UV_CACHE_DIR="$UV_CACHE_DIR" "$UV_ENV/bin/uv" pip install \
    --no-config \
    --python "$VERL_DIR/.venv/bin/python" \
    'numpy==2.2.6' \
    'gymnasium==0.29.1' \
    'textworld==1.7.0' \
    'alfworld==0.4.2'

if [[ "${DOWNLOAD_MODEL:-0}" == 1 ]]; then
    "$VERL_DIR/.venv/bin/hf" download Qwen/Qwen3.5-9B \
        --revision "$MODEL_REVISION" \
        --cache-dir "$HF_CACHE_DIR"
fi

PYTHONPATH="$VERL_DIR:$REPO_ROOT" "$VERL_DIR/.venv/bin/python" -P - <<'PY'
import alfworld
import gymnasium
import numpy
import platform
import textworld
import torch
import transformers
import verl
import vllm
from agent_system.environments.alfworld.env_manager import AlfWorldEnvironmentManager

print("python", platform.python_version())
print("verl", verl.__version__, verl.__file__)
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
print("numpy", numpy.__version__)
print("gymnasium", gymnasium.__version__)
print("textworld", textworld.__version__)
print("alfworld", getattr(alfworld, "__version__", "unknown"))
print("LaMer ALFWorld import OK")
PY

printf 'Compatibility environment ready at %s\n' "$VERL_DIR/.venv"
if [[ "${DOWNLOAD_MODEL:-0}" != 1 ]]; then
    printf 'Model download skipped. Run again with DOWNLOAD_MODEL=1 when needed.\n'
fi
