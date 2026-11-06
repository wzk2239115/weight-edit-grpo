#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
source remote-env.sh
python3 -m venv --system-site-packages .d3-venv
.d3-venv/bin/python -m pip install 'transformers==5.17.0' 'flash-linear-attention==0.5.2' 'fla-core==0.5.2' 'huggingface_hub>=1' datasets accelerate pillow
.d3-venv/bin/python -u prepare-d3.py
