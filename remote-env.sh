#!/usr/bin/env bash
# Source this from the cloned repository, inside wangzekai/tmp.
WE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
case "$WE_ROOT" in
  /home/jovyan/h800fast/wangzekai/tmp/*) ;;
  *) echo 'Clone this repository below /home/jovyan/h800fast/wangzekai/tmp first.' >&2; return 1 ;;
esac
umask 077
mkdir -p "$WE_ROOT"/{data,runs,cache/tmp,cache/hf,cache/torch,cache/nltk}
export http_proxy='http://public-proxy.qihoo.net:3128'
export https_proxy="$http_proxy" HTTP_PROXY="$http_proxy" HTTPS_PROXY="$http_proxy"
export HF_ENDPOINT='https://hf-mirror.com'
export HF_HOME="$WE_ROOT/cache/hf" HF_DATASETS_CACHE="$WE_ROOT/cache/hf/datasets"
export HF_HUB_CACHE="$WE_ROOT/cache/hf/hub" HF_XET_CACHE="$WE_ROOT/cache/hf/xet"
export HF_HUB_DISABLE_XET=1
export TMPDIR="$WE_ROOT/cache/tmp" PIP_CACHE_DIR="$WE_ROOT/cache/pip"
export XDG_CACHE_HOME="$WE_ROOT/cache" TORCH_HOME="$WE_ROOT/cache/torch"
export TRITON_CACHE_DIR="$WE_ROOT/cache/triton" CUDA_CACHE_PATH="$WE_ROOT/cache/cuda"
export NLTK_DATA="$WE_ROOT/cache/nltk" PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$WE_ROOT/vendor${PYTHONPATH:+:$PYTHONPATH}"
export HISTFILE="$WE_ROOT/cache/shell_history"
