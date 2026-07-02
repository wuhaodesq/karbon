#!/usr/bin/env bash
# devagi · run a stage's training
# 用法: bash scripts/cloud/run_stage.sh <stage> <preset> [extra flags to src.train]
#
# Example:
#   bash scripts/cloud/run_stage.sh 0 cloud_24g
#   bash scripts/cloud/run_stage.sh 1 cloud_24g --resume checkpoints/ckpt_stage0_XXX.pt
#   bash scripts/cloud/run_stage.sh 0 cloud_24g --smoke-only

set -euo pipefail

if [[ $# -lt 2 ]]; then
    cat <<EOF >&2
usage: bash scripts/cloud/run_stage.sh <stage> <preset> [extra args to src.train]

  stage:   0 | 1 | 2 | 3 | 4 | 5 | 6
  preset:  local_smoke | cloud_24g | home_64g

Example:
  bash scripts/cloud/run_stage.sh 0 cloud_24g
EOF
    exit 2
fi

STAGE="$1"; shift
PRESET="$1"; shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"

if [[ ! -d "${VENV_PATH}" ]]; then
    echo "ERROR: venv missing. Run scripts/cloud/setup_env.sh first." >&2
    exit 1
fi
# shellcheck disable=SC1090,SC1091
source "${VENV_PATH}/bin/activate"

# Fragmentation mitigation (Axiom 5)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd "${PROJECT_ROOT}"

echo "==> Running Stage ${STAGE} with preset ${PRESET}"
echo "    extra args: $*"

python -m src.train --stage "${STAGE}" --preset "${PRESET}" "$@"
