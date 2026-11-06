#!/usr/bin/env bash
set -eu
cd -- "$(dirname -- "$0")"
source remote-env.sh
pgrep -af 'prepare-d3.py|setup-d3.sh|train.py|multiprocessing.spawn' || true
du -sh models/d3-ultra data/d3 2>/dev/null || true
tail -n 12 runs/d3-setup.log 2>/dev/null || true
