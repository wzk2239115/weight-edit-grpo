#!/usr/bin/env bash
# Install only in our venv; retain the container's NVIDIA PyTorch build.
set -euo pipefail
cd -- "$(dirname -- "$0")"
source remote-env.sh
export MAX_JOBS=8
export TORCH_CUDA_ARCH_LIST=9.0
.venv/bin/python -c 'import torch, triton; print("torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__)'
nvcc --version
.venv/bin/python -m pip install ninja packaging einops
.venv/bin/python -m pip install --no-deps fla-core==0.6.0 flash-linear-attention==0.6.0
.venv/bin/python -m pip install --no-build-isolation --no-deps causal-conv1d==1.7.0
.venv/bin/python acceleration-check.py "$@"
