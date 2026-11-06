"""Compact public-safe terminal status; do not dump code, credentials or question text."""
import json
from pathlib import Path
runs=sorted(Path('runs').glob('d3-search-*/history.jsonl'),key=lambda p:p.stat().st_mtime)
if not runs:
    raise SystemExit('No search history yet')
p=runs[-1]
print(p.parent)
records=[json.loads(x) for x in p.open()]
for r in records:
    if r['event'] in ('baseline','round_complete','round_audit','final_dev'):
        print(json.dumps(r))
last=records[-1]
print(json.dumps({k:v for k,v in last.items() if k in ('event','round','turn','tool','tensor')}))
