#!/usr/bin/env bash
set -euo pipefail
cd /root/karbon

LATEST=$(ls -t checkpoints/ckpt_stage11_*.pt 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    exec .venv/bin/python -m src.train --stage 11 --preset home_64g --config stage11_hierarchical.yaml --resume "$LATEST" > logs/stage11.log 2>&1
else
    exec .venv/bin/python -m src.train --stage 11 --preset home_64g --config stage11_hierarchical.yaml > logs/stage11.log 2>&1
fi
