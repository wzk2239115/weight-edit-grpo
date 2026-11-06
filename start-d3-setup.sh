#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
source remote-env.sh
nohup bash setup-d3.sh > runs/d3-setup.log 2>&1 < /dev/null &
echo "D3 setup PID $!; log: runs/d3-setup.log"
