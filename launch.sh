#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
source remote-env.sh
run=$(mktemp -d "$WE_ROOT/runs/multiround-$(date +%Y%m%d-%H%M%S)-XXXX")
git rev-parse HEAD > "$run/commit.txt"
nohup .venv/bin/python -u train.py \
  --editor ../../Qwen3-4B-Instruct-2507 --target ../../Qwen3.6-27B \
  --a-gpu 0 --b-gpus 1,2,3,4,5,6,7 --group 7 --steps 10 \
  --reads 12 --trials 3 --per-suite 0 --eval-tokens 4096 --batch-size 4 \
  --out "$run" > "$run/train.log" 2>&1 < /dev/null &
echo "$!" > "$run/train.pid"
printf 'Run: %s\nPID: %s\n' "$run" "$(cat "$run/train.pid")"
