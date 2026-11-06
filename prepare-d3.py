"""Pinned decision checkpoint + public choice tasks; no private SR benchmarks."""
import argparse
import hashlib
import json
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download

root = Path(__file__).resolve().parent
destination = root / 'models/d3-ultra'
destination.mkdir(parents=True, exist_ok=True)
pin = destination / 'revision.txt'
revision = pin.read_text().strip() if pin.exists() else HfApi().model_info('vllm-sr/d3-ultra').sha
pin.write_text(revision)
parser = argparse.ArgumentParser()
parser.add_argument('--data-only', action='store_true')
args = parser.parse_args()
if not args.data_only:
    print('Downloading d3-ultra revision', revision, flush=True)
    snapshot_download('vllm-sr/d3-ultra', revision=revision, local_dir=destination, max_workers=4,
                      ignore_patterns=['assets/*', '.eval_results/*'])

manifest_path = root / 'data/d3/manifest.json'
dataset_pins = json.loads(manifest_path.read_text())['datasets'] if manifest_path.exists() else {}
def dataset(repo, config=None, split=None):
    if repo not in dataset_pins:
        dataset_pins[repo] = HfApi().dataset_info(repo).sha
    return load_dataset(repo, config, split=split, revision=dataset_pins[repo])

rows = []
def add(suite, split, state, question, options, answer):
    identity = hashlib.sha256((state.strip() + '\n' + question.strip()).encode()).hexdigest()
    # Labelled official final partitions stay untouched by the search.
    bucket = int(identity[:8], 16) % 10
    part = 'test' if split == 'test' else ('dev' if split == 'dev' or bucket == 0 else 'reward')
    rows.append(dict(id=suite + ':' + identity, split=part, suite=suite, state=state,
                     question=question, options=options, answer=answer, identity=identity))

for split, part in [('train', 'reward'), ('validation', 'dev'), ('test', 'test')]:
    for x in dataset('allenai/ai2_arc', 'ARC-Challenge', split=split):
        choices = x['choices']
        add('arc', part, '', x['question'], choices['text'], choices['label'].index(x['answerKey']))
for x in dataset('cais/mmlu_pro', split='test'):
    h = int(hashlib.sha256(x['question'].strip().encode()).hexdigest()[:8], 16) % 10
    part = 'test' if h < 2 else ('dev' if h < 4 else 'reward')
    add('mmlu', part, '', x['question'], x['options'], x['answer_index'])
for split, part in [('train', 'reward'), ('validation', 'test')]:
    for x in dataset('Rowan/hellaswag', split=split):
        add('hellaswag', part, x['ctx'], 'Which continuation best completes the passage?',
            x['endings'], int(x['label']))
    for x in dataset('google/boolq', split=split):
        add('boolq', part, x['passage'], x['question'], ['No', 'Yes'], int(x['answer']))

# A duplicated prompt is retained only in its highest-priority partition.
unique = {}
for row in sorted(rows, key=lambda r: {'test': 0, 'dev': 1, 'reward': 2}[r['split']]):
    unique.setdefault(row['identity'], row)
out = root / 'data/d3'
out.mkdir(parents=True, exist_ok=True)
for part in ('reward', 'dev', 'test'):
    selected = [r for r in unique.values() if r['split'] == part]
    with (out / f'{part}.jsonl').open('w') as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(part, {s: sum(r['suite'] == s for r in selected)
                 for s in ('arc', 'mmlu', 'hellaswag', 'boolq')}, flush=True)

(out / 'manifest.json').write_text(json.dumps(dict(model_revision=revision, datasets=dataset_pins,
    files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.jsonl')}), indent=2))
