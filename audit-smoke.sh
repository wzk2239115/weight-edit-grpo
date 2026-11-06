#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
source remote-env.sh
.d3-venv/bin/python test-numeric-runner.py
.d3-venv/bin/python -u d3-agent.py --direct --smoke
latest_run=$(ls -dt runs/d3-smoke-* | head -n 1)
.d3-venv/bin/python audit-d3.py "$latest_run"
