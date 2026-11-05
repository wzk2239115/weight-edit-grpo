# Validation (2026-11-05)

Local Mac, Python 3.12; this is **not** a GPU training result.

- Six unit checks: numeric edit validation, preservation gate, JSON-only action
  parsing, BF16 no-op rejection/restoration after an evaluation exception,
  Unicode JSONL loading, and a real clipped policy-gradient update on a tiny
  randomly initialized language model. Equal group rewards leave weights unchanged.
- `prepare.py` completed with actual dataset downloads and official IFEval checkers.
- Reward counts: GSM8K 5,925; MATH 6,013; MMLU-Pro 7,244; IFEval 323.
- Dev counts: GSM8K 1,548; MATH 1,487; MMLU-Pro 2,376; IFEval 104.
- Test counts: GSM8K 1,319; MATH-500 500; MMLU-Pro 2,412; IFEval 114.
- Scorers accepted sampled reference math/knowledge answers; all 114 held-out
  IFEval items executed their checkers on dummy output without dependency errors.

GPU follow-up through JupyterLab:

- Repository cloned under the agreed `wangzekai/tmp` directory.
- Setup completed, six tests passed in the GPU container, four suites downloaded
  through the configured proxy/HF mirror. Split checksums matched the local files.
- `smoke.sh` launched: two 27B B replicas and the 4B A loaded successfully.
- The initial small smoke completed, but equal negative group rewards skipped the
  A update. This established execution only, not learning or capability gains.
- Inference dependencies subsequently installed inside the project `.venv`:
  causal-conv1d 1.7.0, fla-core 0.6.0, flash-linear-attention 0.6.0. The existing
  NVIDIA PyTorch 2.7.0a0 and Triton 3.2.0 were retained. FLA warns that Triton 3.3+
  is recommended; compatibility beyond the checked workloads is unverified.
- BF16 CUDA convolution matched the PyTorch reference within the check's
  tolerance; chunk and recurrent GDN outputs/final states agreed. Transformers
  selected the external optimized implementations for all four kernel functions.
- Real B generated twice, including cached decoding, and all four optimized
  functions were called. The first 32-token generation took 43.68s including
  compilation; the second took 1.76s (18.14 tokens/s). This short, single-prompt
  check is not full benchmark throughput or a measured speedup against fallback.

Still required: measure full evaluation throughput and run the revised multi-round
editor training with substantial evaluation and informative group rewards.
No capability improvement established.
