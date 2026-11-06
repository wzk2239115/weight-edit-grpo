"""Offline trajectory audit. This checks recorded evidence, not held-out performance."""
import hashlib
import json
from pathlib import Path
import sys
from numeric_runner import NumericalProgram

run = Path(sys.argv[1]).resolve()
records = [json.loads(line) for line in (run/'history.jsonl').open()]
if not (run/'boundary-hashes.json').exists():
    raise SystemExit('INCOMPLETE: legacy run has no complete trajectory/boundary evidence')
root = Path(__file__).resolve().parent
hashes = json.loads((run/'boundary-hashes.json').read_text())
for path, expected in hashes.items():
    if hashlib.sha256((root/path).read_bytes()).hexdigest() != expected:
        raise SystemExit('FAIL: boundary file differs: '+path)
catalog = json.loads((run/'tensor-catalog.json').read_text())
inspected, output, candidates = {}, None, []
last_code = None
for r in records:
    if r['event']=='agent_response':
        try:
            obj=json.loads(r['content'])
            if obj.get('tool')=='code':
                last_code=obj['code']
                source_path=run / f"round-{r['round']}-turn-{r['turn']}.py"
        except (ValueError,KeyError,TypeError):
            pass
    if r['event']=='tool_result' and isinstance(r['result'],dict) and 'values' in r['result']:
        inspected[r['result']['name']]=r['result']
    if r['event']=='claude_patch':
        source=(run/r['path']).read_text()
        if hashlib.sha256(source.encode()).hexdigest()!=r['source_sha256'] or json.loads(source)!=r['changes']:
            raise SystemExit('FAIL: Claude Code patch artifact differs from evaluated edits')
        output=r['changes']
    if r['event']=='program_output':
        if not source_path.exists() or source_path.read_text()!=last_code:
            raise SystemExit('FAIL: saved program source differs from agent response')
        if last_code is None or hashlib.sha256(last_code.encode()).hexdigest()!=r['source_sha256']:
            raise SystemExit('FAIL: source hash does not match agent response')
        if NumericalProgram(inspected,catalog).run(last_code)!=r['changes']:
            raise SystemExit('FAIL: numerical code does not reproduce patch')
        output=r['changes']
    if r['event']=='candidate':
        if output!=r['assignments']:
            raise SystemExit('FAIL: evaluated patch differs from program output')
        realized=r['realized']
        if r['actual_changed']!=sum(x['before']!=x['after'] for x in realized):
            raise SystemExit('FAIL: inconsistent actual edit count')
        if len(realized)!=len(output) or [(x['name'],x['index']) for x in realized]!=[(x['name'],x['index']) for x in output]:
            raise SystemExit('FAIL: inconsistent edited coordinates')
        candidates.append(dict(round=r['round'], actual_changed=r['actual_changed'],accepted=r['accepted']))
        output=None
if not candidates:
    raise SystemExit('INCOMPLETE: no evaluated coding-agent candidate')
print(json.dumps(dict(status='PASS', boundary_files=len(hashes), candidates=candidates,
    limitation='Recorded trajectory and file integrity verified; fresh-checkpoint replay and held-out validation remain separate.')))
