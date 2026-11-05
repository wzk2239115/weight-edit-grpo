#!/usr/bin/env bash
set -eu
cd -- "$(dirname -- "$0")"
source remote-env.sh
# Reuse the existing PyTorch installation without changing its files.
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python tests.py
.venv/bin/python prepare.py
