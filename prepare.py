"""Download four small benchmark suites; deterministic, disjoint reward/dev/test files."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import zipfile

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data'


def partition(rows, suite, source_split, official_test=False):
    result = []
    for i, row in enumerate(rows):
        row = dict(row)
        key = hashlib.sha256(row['prompt'].strip().encode()).hexdigest()
        bucket = int(key[:8], 16) % 10
        split = 'test' if official_test else ('dev' if bucket < 2 else 'reward')
        # Suites without a training split reserve 20% for final evaluation as well.
        if source_split == 'partitioned_test':
            split = 'test' if bucket < 2 else ('dev' if bucket < 4 else 'reward')
        row.update(id=f'{suite}:{source_split}:{i}', suite=suite, split=split)
        result.append(row)
    return result


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    for split in ('train', 'test'):
        ds = load_dataset('openai/gsm8k', 'main', split=split)
        rows += partition([dict(prompt=x['question'], answer=x['answer'].split('####')[-1].strip())
                           for x in ds], 'gsm8k', split, split == 'test')
    for config in ('algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra',
                   'number_theory', 'prealgebra', 'precalculus'):
        ds = load_dataset('EleutherAI/hendrycks_math', config, split='train')
        rows += partition([dict(prompt=x['problem'], answer=x['solution']) for x in ds],
                          'math', 'train_' + config)
    ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
    rows += partition([dict(prompt=x['problem'], answer=x['answer']) for x in ds],
                      'math', 'math500', True)
    ds = load_dataset('TIGER-Lab/MMLU-Pro', split='test')
    rows += partition([dict(prompt=x['question'] + '\n' + '\n'.join(
        f'{chr(65+i)}. {v}' for i, v in enumerate(x['options'])), answer=x['answer'],
        category=x['category']) for x in ds], 'mmlu', 'partitioned_test')
    # Fetch the original IFEval implementation, not a substitute keyword scorer.
    vendor = ROOT / 'vendor' / 'instruction_following_eval'
    vendor.mkdir(parents=True, exist_ok=True)
    (vendor / '__init__.py').touch()
    base = 'https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval/'
    for name in ('instructions.py', 'instructions_registry.py', 'instructions_util.py',
                 'evaluation_lib.py', 'data/input_data.jsonl'):
        path = vendor / name
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(base + name, path)
    ds = [json.loads(x) for x in (vendor / 'data/input_data.jsonl').read_text().splitlines()]
    rows += partition([dict(prompt=x['prompt'], answer='', instruction_id_list=x['instruction_id_list'],
                            kwargs=x['kwargs'], key=x['key']) for x in ds], 'ifeval', 'partitioned_test')
    # Fixed official resource URLs also work behind explicit corporate proxies.
    nltk_dir = Path(os.environ.get('NLTK_DATA', str(ROOT / 'cache/nltk')))
    (nltk_dir / 'tokenizers').mkdir(parents=True, exist_ok=True)
    for name in ('punkt', 'punkt_tab'):
        archive = nltk_dir / 'tokenizers' / f'{name}.zip'
        urllib.request.urlretrieve(
            f'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/{name}.zip',
            archive)
        destination = (nltk_dir / 'tokenizers').resolve()
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.namelist():
                if not (destination / member).resolve().is_relative_to(destination):
                    raise ValueError('Unsafe archive path')
            bundle.extractall(destination)
    # Remove cross-split prompt overlaps, giving held-out tests precedence.
    priority = {'test': 0, 'dev': 1, 'reward': 2}
    unique = {}
    for row in sorted(rows, key=lambda x: priority[x['split']]):
        unique.setdefault((row['suite'], row['prompt'].strip()), row)
    manifest = {}
    for split in ('reward', 'dev', 'test'):
        selected = [x for x in unique.values() if x['split'] == split]
        path = OUT / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in selected))
        manifest[split] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              counts={s: sum(x['suite'] == s for x in selected)
                                      for s in ('gsm8k', 'math', 'mmlu', 'ifeval')})
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
