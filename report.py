"""Print compact evidence from a completed group, without dumping generations."""
import argparse
import json
import hashlib
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('directory', nargs='?', default='runs/smoke/step-0000')
args = parser.parse_args()
directory = Path(args.directory)
print('UPDATE', (directory / 'update.json').read_text())
for record in json.loads((directory / 'candidates.json').read_text()):
    action = record.get('action') or {}
    summary = {k: record.get(k) for k in ('candidate', 'error', 'changed', 'base', 'edited',
                                        'reward', 'target_gain', 'drops', 'passed')}
    summary['location'] = {k: action.get(k) for k in ('tensor', 'row', 'col')}
    values = action.get('values') or []
    summary['shape'] = [len(values), len(values[0]) if values else 0]
    summary['patch_hash'] = hashlib.sha256(json.dumps(action, sort_keys=True).encode()).hexdigest()[:16]
    summary['read_requests'] = sum('"read"' in x['content'] for x in record.get('messages', [])
                                   if x['role'] == 'assistant')
    print(json.dumps(summary))
