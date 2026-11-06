#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
source remote-env.sh
# Eight resident replicas; edits are always scored before acceptance and next round.
if pgrep -f '[d]3-agent.py.*--rounds' >/dev/null; then
    echo 'A D3 search is already running; refusing duplicate launch.'
    exit 1
fi
nohup .d3-venv/bin/python -u d3-agent.py --direct --rounds 200 --per-suite 200 --verify-parallel --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7 "$@" > runs/d3-search.log 2>&1 < /dev/null &
echo "D3 search PID $!; 200 rounds; 8 GPUs; fixed 800 questions per edit; log runs/d3-search.log"
