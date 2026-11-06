#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
source remote-env.sh
.d3-venv/bin/python -m pip install --no-build-isolation --no-deps 'causal-conv1d==1.7.0'
.d3-venv/bin/python -u check-d3.py
