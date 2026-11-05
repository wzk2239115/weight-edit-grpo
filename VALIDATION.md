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

Still required: GPU environment/data setup, actual A/B loading, latency/memory
measurement, and a real editor training group. No capability improvement established.
