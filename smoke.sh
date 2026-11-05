#!/usr/bin/env bash
set -eu
cd -- "$(dirname -- "$0")"
source remote-env.sh
.venv/bin/python -u train.py \
  --editor ../../Qwen3-4B-Instruct-2507 \
  --target ../../Qwen3.6-27B \
  --b-gpus 1,2 --group 2 --steps 1 --per-suite 128 \
  --eval-tokens 4096 --batch-size 2 --out runs/check-128
