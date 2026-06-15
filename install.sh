#!/usr/bin/env bash
# One-shot install: creates conda env "fox" with Python 3.10, installs all deps.
# Usage: ./install.sh [env_name]
set -euo pipefail

ENV_NAME="${1:-fox}"
PYTHON_VERSION="3.11"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found. Install Miniconda/Anaconda first." >&2
    exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Env '$ENV_NAME' already exists. Updating packages..."
else
    echo "Creating conda env '$ENV_NAME' (python=$PYTHON_VERSION)..."
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Done. Activate with: conda activate $ENV_NAME"
