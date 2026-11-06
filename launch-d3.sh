#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
source remote-env.sh
# One resident checkpoint; edits are always scored before acceptance and next round.
if pgrep -f '[d]3-agent.py.*--rounds' >/dev/null; then
    echo 'A D3 search is already running; refusing duplicate launch.'
    exit 1
fi
nohup .d3-venv/bin/python -u d3-agent.py --direct --rounds 200 --per-suite 200 > runs/d3-search.log 2>&1 < /dev/null &
echo "D3 search PID $!; 200 rounds; fixed 800 questions per edit; log runs/d3-search.log"
