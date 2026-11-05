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
- B reported missing `causal-conv1d` and `flash-linear-attention`; generation uses
  the slower reference implementation. Large-scale throughput is not yet validated.
- The group was still running when UI control returned `noWindowsAvailable`.
  Its outcome and any actual A update remain unconfirmed.

Still required: read the completed smoke result, measure latency/memory, address
the B inference bottleneck, and run a real editor training group with informative
rewards. No capability improvement established.
