# weight-edit-grpo

Minimal experiment: a small editor **A** reads selected blocks of a larger model **B**,
chooses a layer/matrix, and emits actual floating-point replacement values. A fixed
executor copies those values; it does not compute an optimizer update or a preset
projection. Benchmark outcomes train A with group-relative, clipped policy gradients.

**Priority: demonstrate one real training update and measure whether edits help.**
No scheduler, service framework, arbitrary Python execution, or distributed RL stack.

## What is implemented

- An architecture-aware diagnostic prompt in `prompts.py`, reward-only baseline
  failure examples, a global layer survey and optional selected-layer matrix statistics.
  Statistics suggest hypotheses; they do not establish neuron semantics or causality.
- Up to 12 inspection actions and 3 evaluated trial edits per episode (`--reads`,
  `--trials`). A sees each trial's scores, preservation failures and sample outputs,
  can inspect again and revise, then explicitly selects a trial with `finish` or
  withdraws with `trial=-1` (zero reward). Each proposal is one rectangular block
  of up to 256 scalar values, evaluated against the original B; trials do not accumulate.
  The whole read/edit/feedback/selection trajectory receives the final group reward.
  Search episodes are interleaved so B replicas can evaluate different trials concurrently.
  The policy must explain its evidence, obtain the survey, and read actual blocks from
  at least three distinct layers and two matrix families before editing. The policy
  chooses all layers, coordinates, block sizes and numerical values; layer zero is
  permitted when justified. Fixed zero-coordinate sample patches have been removed.
  This is an initial action budget, not a claim that 256 values suffice to improve a 27B model.
- BF16 writes checked for finite values and actual changes after rounding.
- Each candidate starts from the same B; modified entries are restored in `finally`.
- Persistent independent B replicas on different GPUs evaluate candidates concurrently.
- GSM8K and MATH target accuracy; MMLU-Pro and original strict IFEval preservation.
- Reward = mean target accuracy gain, unless either preservation score drops more
  than the tolerance (default **2 percentage points**), in which case reward is negative.
  Invalid/no-op edits receive -2. All equal rewards explicitly skip the update.
- A trains with LoRA, group-normalized rewards and token-level PPO clipping across
  its read/edit trajectory. No reference KL term in this first implementation.
- Candidate text, exact patches, item-level answers, scores and update status saved locally.

This is a pilot, not evidence of successful editing. Random numerical edits may produce
zero measurable gains. Report the invalid rate, no-op rate, reward variance and actual
update count; a running loop alone does not validate the hypothesis.

## GPU machine setup

Everything created remotely stays under `/home/jovyan/h800fast/wangzekai/tmp`.
Existing model files are read-only inputs. Do not install into an existing environment.

```bash
cd /home/jovyan/h800fast/wangzekai/tmp
git clone https://github.com/wzk2239115/weight-edit-grpo.git
cd weight-edit-grpo
source remote-env.sh
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python prepare.py
.venv/bin/python tests.py
```

Or run `bash setup.sh` after cloning to perform those steps together.

Before evaluating Qwen3.6-27B, run `bash accelerate.sh`. It installs pinned
`causal-conv1d` and FLA kernels into `.venv`, retaining the container's PyTorch.
The check executes BF16 CUDA convolution and Gated DeltaNet kernels, verifies
Transformers dispatch, and generates twice with the actual B (including cached
decoding). The first pass can include Triton compilation; the printed token rate
is a short inference check, not a benchmark throughput estimate. All build and
runtime caches stay below this repository. Recheck without reinstalling with
`source remote-env.sh; .venv/bin/python acceleration-check.py`.
The current NVIDIA container supplies Triton 3.2.0; FLA recommends 3.3+, so it
prints a version warning. CUDA numerical checks and real B prefill/cached decoding
passed on this installed stack; other shapes and workloads are not covered by
that short check. No PyTorch/Triton upgrade is performed by the installer.

`remote-env.sh` sets the corporate proxy, `HF_ENDPOINT=https://hf-mirror.com`, and
explicit local cache/temp paths. The mirror must be reachable; the IFEval checker
also requires access to raw.githubusercontent.com and NLTK's download host. Downloads
fail loudly if blocked. No credentials are needed for the selected public datasets.

## Smoke run, then spend compute on evaluation

First verify loading, tool output, patches, evaluation and policy gradients with
two B replicas and 128 questions **per suite** (or all available if fewer).
This limited run is still a plumbing check, not evidence of an ability improvement.
`bash smoke.sh` runs the following with batch size 2.

```bash
.venv/bin/python train.py \
  --editor ../../Qwen3-4B-Instruct-2507 --target ../../Qwen3.6-27B \
  --b-gpus 1,2 --group 2 --steps 1 --per-suite 128 \
  --eval-tokens 4096 --out runs/check-128
```

Then use all eight GPUs: A on GPU 0, seven B replicas on GPUs 1–7. A uses LoRA so
the other cards can be spent on evaluation. Each candidate sees the same question
batch in its group. The default `--per-suite 0` evaluates the entire reward split
for every edit: GSM8K 5,925; MATH 6,013; MMLU-Pro 7,244; IFEval 323
(19,505 questions per candidate with the prepared manifest). The complete-set order
is fixed across groups, and unchanged B baselines are reused within each worker.
Reward-set results guide training; final claims require the untouched test split.

Baseline batches are distributed across the seven workers once, merged in original
question order, and shared with all replicas. Original batch boundaries are retained.
Identical reward sets reuse this baseline across steps. Waiting polls worker liveness
every minute instead of aborting a healthy evaluation after two hours.

Before raising the batch size, run `source remote-env.sh; .venv/bin/python -u throughput.py`.
This uses reward prompts only, compares batches 4/16/32, and tests 4096 decode tokens
at the largest batch that passed the short probe. It does not train A. Check memory
on long prompts before selecting `WE_BATCH_SIZE=16 bash launch.sh` (or another tested
size); `bash launch.sh` retains batch 4 until a larger batch is validated.

```bash
.venv/bin/python train.py \
  --editor ../../Qwen3-4B-Instruct-2507 --target ../../Qwen3.6-27B \
  --group 7 --steps 10 --per-suite 0 --eval-tokens 4096 \
  --out runs/first
```

27B BF16 weights are roughly 54GB before runtime memory. Single-card B replicas,
batch size and sequence length **must be load-tested**; reduce batch size on OOM.
Current first version uses Transformers generation for transparent in-memory edits;
it has not been optimized to vLLM throughput. No inference settings change between
baseline and edits. Log truncation and increase token budgets if needed.

## TensorBoard

```bash
source remote-env.sh
.venv/bin/python -m pip install tensorboard
.venv/bin/python tensorboard_logs.py
.venv/bin/tensorboard --logdir "$PWD/runs" --host 127.0.0.1 --port 6006
```

For JupyterLab's TensorBoard launcher, enter the absolute `runs` directory as
the log directory. New runs write scalars after each completed group; the backfill
command imports JSON records from runs started before TensorBoard was added.
An empty dashboard before the first group completes is expected. `train/update_count`
counts actual updates; equal rewards skip updates. UI access depends on the platform's
TensorBoard or authenticated port-proxy support.

## Held-out evaluation commands

```bash
.venv/bin/python train.py \
  --editor ../../Qwen3-4B-Instruct-2507 --target ../../Qwen3.6-27B \
  --b-gpus 1 --evaluate-patch runs/first/step-0000/patch-0.json \
  --split dev --per-suite 0 --out runs/check-dev
```

Select on dev, then run once with `--split test` into a different output directory.
GSM8K train is split into reward/dev, with official test held out. MATH training
questions are reward/dev and MATH-500 is held out. MMLU-Pro and IFEval have custom,
deterministic prompt-hash reward/dev/test partitions; their held-out results are
**subset results, not the official full-suite score**. Preparation removes exact
prompt overlap across splits and saves file checksums. Repeated reward use is training.
Public benchmarks can already occur in base-model pretraining; this split does not
establish absence of pretraining contamination.

Code execution benchmarks are intentionally not implemented yet. Four suites above
are the executable first experiment; do not claim code capability is preserved.
Likewise finite suites cannot prove that *all* other capabilities are unchanged.

## Limits to keep visible

- Model directory names are local labels. Verify source/revision and actual tensor
  dtype before making model-specific claims; the local 27B config says `qwen3_5`.
- No automatic checkpoint resume yet. Adapters, optimizer state and records are saved.
- No teacher demonstrations yet. If the policy never emits useful edits, examine the
  trajectory and cold-start bottleneck before increasing training duration.
- Single 27B model/one target goal first; later test transfer to another checkpoint
  and multiple goals before claiming a general editing skill.
- The tolerance is a pilot threshold, not a statistical guarantee. Use enough held-out
  questions and paired uncertainty estimates for conclusions.
- Keep datasets, model weights, tokens and private machine paths out of Git commits.

Inspired by [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
and its [released experiment logs](https://github.com/RikiyaT/lm_understand-logs).

## D3 weight search (API agent)

This separate experiment keeps the decision model and evaluator fixed. DeepSeek
inspects named tensors in multiple turns and emits explicit scalar assignments;
only those assignments can change the resident checkpoint. Rejected candidates
are restored. Accepted candidates accumulate from the best checkpoint, with
per-suite regression checks. BF16 no-op assignments are rejected. Accuracy and
NLL gains are logged separately; completing 200 trials does not imply improvement.

On the GPU, all writes remain under this clone within `wangzekai/tmp`:

```bash
source remote-env.sh
bash setup-d3.sh
# Supply the key without recording it in shell history or displaying it:
read -r -s -p '360 API key: ' DEEPSEEK_API_KEY; echo
export DEEPSEEK_API_KEY
.d3-venv/bin/python d3-agent.py --api-check
.d3-venv/bin/python -u d3-agent.py --rounds 200 --per-suite 200
```

Defaults: `https://api.360.cn/v1`, `deepseek/deepseek-v4.1-flash`.
Override with `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL`. On the Mac, the API client
can read the existing `360-proxy` provider in `~/.config/opencode/opencode.json`.
No key is included in code, logs or Git.

Public alternatives to the private SR benchmarks: ARC-Challenge, MMLU-Pro,
HellaSwag and BoolQ, pinned to dataset revisions and deduplicated across splits.
Default search uses a fixed 800 questions (200 per suite); dev is evaluated only
after the search budget. `data/d3/test.jsonl` remains held out and is never sent
to the agent. These tests measure public choice-task transfer, not reproduction
of the model card's private SR score. Full pools, split hashes and revisions are
stored in `data/d3/manifest.json`.

Outputs stay in `runs/d3-search-*`: original baseline, actual edit counts,
accepted/rejected trials, tensor catalog, question IDs and `best-patch.json`.
The source checkpoint is never overwritten. This is a specialized weight tool
agent, with no shell access to alter the runtime or scorer.

GPU setup verified on 2026-11-06: 52.2 GB checkpoint downloaded, all four public
benchmark pools prepared, real checkpoint load and eight-question integration
inference passed. The old GRPO training tree was stopped; its logs were retained.
In this company network, the API must connect directly (`--direct`) while HF
continues through the download proxy. Hidden credential setup is available via
`python3 configure-d3.py`; `secrets/d3-api.json` stays outside Git.

Run `bash accelerate-d3.sh` to install the convolution kernel in the separate
D3 environment. Check the whole API/edit/evaluate loop with:

```bash
.d3-venv/bin/python -u d3-agent.py --direct --smoke
```

This smoke mode makes one proposal on eight questions and skips dev. It verifies
integration only. The research run remains `--direct --rounds 200 --per-suite 200`.

### Coding agent boundary and trajectory audit

The current agent writes bounded Python numerical programs (`tool: code`), rather
than only a JSON list. Programs can inspect previously requested tensor slices,
use loops/conditions/arithmetic, and emit arbitrary coordinates and scalar values.
They run in a separate resource-limited process through an AST interpreter; agent
source is never passed to Python exec/eval. This is a numerical coding agent,
not OpenCode or an unrestricted terminal agent.

It receives no checkpoint objects, filesystem handles, labels, or evaluation code.
Only the trusted evaluator applies emitted assignments. Framework, runtime code,
config and benchmark hashes are checked before/after evaluation. Runs record full
assistant responses, tool results, program source/hash/output and realized
before/after values after checkpoint dtype conversion. Files stay under runs/.

`python audit-d3.py runs/NAME` verifies recorded boundary hashes and replays the
numerical programs to reproduce the evaluated patches. Legacy runs without this
full evidence are explicitly marked incomplete. This audit does not establish
capability gains or rule out benchmark overfitting; fresh-checkpoint replay and
held-out testing are required for a research result. Do not pull or modify code
while an audited run is active: that intentionally invalidates its file hashes.
