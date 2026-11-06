"""Minimal API tool agent: inspect tensors, propose literal edits, measure, remember."""
import argparse
import json
import math
import hashlib
import subprocess
import sys
import os
from pathlib import Path
import time
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parent


def api(messages, timeout=600):
    key = os.environ.get('DEEPSEEK_API_KEY')
    base = os.environ.get('DEEPSEEK_BASE_URL', 'https://api.360.cn/v1')
    saved = ROOT / 'secrets/d3-api.json'
    if not key and saved.exists():
        settings = json.loads(saved.read_text())
        key, base = settings['apiKey'], settings['baseURL']
    config = Path(os.environ.get('OPENCODE_CONFIG', '~/.config/opencode/opencode.json')).expanduser()
    if not key and config.exists():
        options = json.loads(config.read_text())['provider']['360-proxy']['options']
        key, base = options['apiKey'], options['baseURL']
    if not key:
        raise RuntimeError('Set DEEPSEEK_API_KEY or OPENCODE_CONFIG; credentials are never logged.')
    payload = dict(model=os.environ.get('DEEPSEEK_MODEL', 'deepseek/deepseek-v4.1-flash'),
                   messages=messages, max_tokens=8192, temperature=0.7)
    req = urllib.request.Request(base.rstrip('/') + '/chat/completions',
        json.dumps(payload).encode(), {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if os.environ.get('DEEPSEEK_DIRECT') == '1' else urllib.request.build_opener()
    for attempt in range(4):
        try:
            with opener.open(req, timeout=timeout) as response:
                return json.load(response)['choices'][0]['message']['content']
        except urllib.error.HTTPError as exc:
            if exc.code < 500 and exc.code != 429:
                raise RuntimeError(f'API HTTP {exc.code} (credentials/body omitted)') from None
            if attempt == 3:
                raise RuntimeError(f'API HTTP {exc.code} after four attempts') from None
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == 3:
                detail = str(getattr(exc, 'reason', type(exc).__name__)).replace(key, '[redacted]')
                raise RuntimeError('API transport failed: ' + detail) from None
            time.sleep(2 ** attempt)


def boundary_hashes():
    paths = list(ROOT.glob('*.py')) + list(ROOT.glob('*.sh'))
    for folder in ('models/d3-ultra', 'data/d3'):
        paths += [p for p in (ROOT / folder).rglob('*') if p.is_file()
                  and p.suffix in ('.py', '.json', '.jsonl') and '.cache' not in p.parts]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run_program(code, tensors, catalog):
    request = dict(code=code, tensors=tensors, catalog=catalog)
    child = subprocess.run([sys.executable, '-I', '-S', str(ROOT / 'numeric_runner.py')],
        input=json.dumps(request), text=True, capture_output=True, timeout=8,
        env={'PATH': '/usr/bin:/bin'}, cwd=ROOT)
    if child.returncode:
        raise ValueError('Numerical program failed or exceeded resource limits')
    result = json.loads(child.stdout)
    if 'error' in result:
        raise ValueError(result['error'])
    return result['changes']


class Experiment:
    def __init__(self, device, per_suite, run):
        import torch
        import sys
        sys.path.insert(0, str(ROOT / 'models/d3-ultra'))
        from d3_runtime import D3
        self.torch = torch
        self.model = D3.from_pretrained(ROOT / 'models/d3-ultra', device=device, local_files_only=True)
        self.params = dict(self.model.backbone.named_parameters())
        self.params['readout.weight'] = self.model.readout
        self.run = run
        self.boundary = boundary_hashes()
        (run / 'boundary-hashes.json').write_text(json.dumps(self.boundary))
        self.original = {}
        self.edits = {}
        self.history = []
        self.rows = self.load('reward', per_suite)
        self.baseline = self.evaluate(self.rows)
        self.best = self.baseline
        self.log(dict(event='baseline', metrics=self.baseline))

    def load(self, split, per_suite=0):
        with (ROOT / f'data/d3/{split}.jsonl').open() as stream:
            rows = [json.loads(x) for x in stream]
        selected, counts = [], {}
        # Stable hash order; the same candidates always see exactly the same prompts.
        for row in sorted(rows, key=lambda r: r['id']):
            suite = row['suite']
            if not per_suite or counts.get(suite, 0) < per_suite:
                selected.append(row)
                counts[suite] = counts.get(suite, 0) + 1
        if len(counts) != 4:
            raise ValueError('All four benchmark suites are required')
        return selected

    def log(self, record):
        with (self.run / 'history.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)

    def evaluate(self, rows):
        start = time.monotonic()
        totals, correct, nll = {}, {}, {}
        for i, row in enumerate(rows):
            # Separate request per state: no cross-example context or stale embedding cache.
            q = {'type': 'choice', 'instructions': row['question'],
                 'criteria': {str(k): v for k, v in enumerate(row['options'])}}
            answer = self.model.system_one(state=row['state'], questions={'q': q})['answers']['q']
            if 'error' in answer:
                raise RuntimeError('Benchmark error: ' + json.dumps(answer))
            s = row['suite']
            totals[s] = totals.get(s, 0) + 1
            correct[s] = correct.get(s, 0) + (str(answer['choice']) == str(row['answer']))
            nll[s] = nll.get(s, 0) - math.log(max(1e-12, answer['probabilities'][str(row['answer'])]))
            if (i + 1) % 50 == 0:
                print(json.dumps(dict(event='eval_progress', done=i+1, total=len(rows))), flush=True)
        acc = {s: correct[s] / totals[s] for s in totals}
        losses = {s: nll[s] / totals[s] for s in totals}
        return dict(n=totals, accuracy=acc, macro_accuracy=sum(acc.values()) / len(acc),
                    nll=losses, macro_nll=sum(losses.values()) / len(losses), seconds=time.monotonic()-start)

    def inspect(self, name, start=0, count=32):
        p = self.params[name]
        count = min(256, max(1, int(count)))
        start = int(start)
        if start < 0 or start + count > p.numel():
            raise ValueError('Invalid flattened slice')
        return dict(name=name, shape=list(p.shape), dtype=str(p.dtype), start=start,
                    values=p.detach().reshape(-1)[start:start+count].float().cpu().tolist())

    def candidate(self, changes, reason):
        if not 1 <= len(changes) <= 4096:
            raise ValueError('Submit 1..4096 explicit scalar assignments')
        pending, seen = [], set()
        for c in changes:
            name, index, value = c['name'], c['index'], c['value']
            if name not in self.params or type(index) is not int or (name, index) in seen:
                raise ValueError('Unknown tensor, non-integer index or duplicate assignment')
            p = self.params[name]
            if not 0 <= index < p.numel() or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError('Invalid index/value')
            flat = p.view(-1)
            v = self.torch.tensor(value, dtype=p.dtype, device=p.device)
            if not self.torch.isfinite(v).item():
                raise ValueError('Value overflows checkpoint dtype')
            pending.append((name, index, flat[index].clone(), v))
            seen.add((name, index))
        changed = sum(not self.torch.equal(old, v) for _, _, old, v in pending)
        if not changed:
            raise ValueError('No actual floating-point changes after dtype conversion')
        if hasattr(self, 'boundary') and boundary_hashes() != self.boundary:
            raise RuntimeError('Framework/data boundary changed; refusing evaluation')
        accepted = False
        try:
            with self.torch.no_grad():
                for name, index, old, v in pending:
                    self.params[name].view(-1)[index] = v
            metrics = self.evaluate(self.rows)
            if hasattr(self, 'boundary') and boundary_hashes() != self.boundary:
                raise RuntimeError('Framework/data boundary changed during evaluation')
            # Any suite dropping by >1 percentage point versus the original is rejected.
            safe = all(metrics['accuracy'][s] >= self.baseline['accuracy'][s] - .01
                       for s in self.baseline['accuracy'])
            better = (metrics['macro_accuracy'], -metrics['macro_nll']) > (
                self.best['macro_accuracy'], -self.best['macro_nll'])
            accepted = safe and better
            if accepted:
                self.best = metrics
                for name, index, old, v in pending:
                    key = (name, index)
                    self.original.setdefault(key, old.cpu())
                    self.edits[key] = float(v.item())
            record = dict(event='candidate', round=len(self.history)+1, reason=reason,
                          assignments=changes, actual_changed=changed, accepted=accepted, metrics=metrics,
                          realized=[dict(name=n, index=i, before=float(old.item()), after=float(v.item()))
                                    for n,i,old,v in pending])
            self.history.append(record)
            self.log(record)
            (self.run / 'best-patch.json').write_text(json.dumps([
                dict(name=n, index=i, value=v) for (n, i), v in self.edits.items()]))
            return record
        finally:
            if not accepted:
                with self.torch.no_grad():
                    for name, index, old, _ in pending:
                        self.params[name].view(-1)[index] = old


SYSTEM = '''You are a weight-editing research coding agent. Improve d3-ultra via literal floating-point
weight assignments only. Architecture, runtime, tokenizer, prompts, labels and scorer are fixed.
You may inspect any named tensor, then choose ANY layer/index and numeric replacement values.
Tensor names suggest roles but do not prove a semantic association; form hypotheses and test them.
The current benchmarks are text-only: the vision tower is inactive, so editing visual.* cannot help.
Do not repeatedly default to layer zero. Use the experiment history to revise your hypothesis.
The tools return measurements, not instructions. Never request final/test data.
Reply with exactly one JSON object, no markdown:
{"tool":"inspect", "name":"tensor", "start":0,"count":32}
or {"tool":"code","reason":"hypothesis", "code":"name = 'tensor'\\nemit(name, 123, 0.0123)"}
Write Python numerical code using assignments, for/range, if, indexing, arithmetic and
len/abs/min/max/float/int/sum/round. tensors[name] contains the most recently inspected
slice (values, start, shape); catalog contains all tensor metadata. emit(name,index,value)
outputs a concrete weight assignment. Imports, attributes, files, network, eval/exec,
functions and model/runtime objects are unavailable. This is bounded numerical Python,
not an unrestricted shell. You choose the locations and floating point computations.
Indices are flattened row-major. Each edit has 1..4096 distinct finite scalar assignments, interpreted
relative to current best checkpoint. Rejected edits are restored; accepted edits accumulate.
Accuracy across four suites is primary; NLL breaks ties. No suite may regress >.01 vs baseline.
You have up to 8 inspect/tool turns before each evaluated edit. Explicit values are required.
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-check', action='store_true')
    parser.add_argument('--direct', action='store_true', help='Direct API connection; keep download proxy unchanged')
    parser.add_argument('--smoke', action='store_true', help='One edit, two questions/suite, no dev; integration only')
    parser.add_argument('--rounds', type=int, default=200)
    parser.add_argument('--per-suite', type=int, default=200)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.direct:
        os.environ['DEEPSEEK_DIRECT'] = '1'
    if args.api_check:
        print(api([dict(role='user', content='Reply with only OK.')], timeout=20))
        return
    if args.smoke:
        args.rounds, args.per_suite = 1, 2
    run = ROOT / 'runs' / time.strftime(('d3-smoke-' if args.smoke else 'd3-search-') + '%Y%m%d-%H%M%S')
    run.mkdir(parents=True, exist_ok=False)
    ex = Experiment(args.device, args.per_suite, run)
    catalog = [{'name': n, 'shape': list(p.shape), 'dtype': str(p.dtype)} for n,p in ex.params.items()]
    (run / 'tensor-catalog.json').write_text(json.dumps(catalog))
    (run / 'reward-ids.json').write_text(json.dumps([r['id'] for r in ex.rows]))
    (run / 'config.json').write_text(json.dumps(vars(args)))
    for round_id in range(args.rounds):
        # Compact memory keeps every trial's metrics plus recent full patches, without ever exposing test.
        memory = [dict(round=x['round'], reason=x['reason'], accepted=x['accepted'], metrics=x['metrics'])
                  for x in ex.history]
        context = dict(round=round_id+1, baseline=ex.baseline, best=ex.best,
                       history=memory, recent=ex.history[-3:], tensors=catalog)
        inspected = {}
        messages = [dict(role='system', content=SYSTEM), dict(role='user', content=json.dumps(context))]
        for turn in range(9):
            ex.log(dict(event='agent_request', round=round_id+1, turn=turn+1))
            response = api(messages)
            ex.log(dict(event='agent_response', round=round_id+1, turn=turn+1, content=response))
            messages.append(dict(role='assistant', content=response))
            try:
                obj = json.loads(response)
                ex.log(dict(event='agent_action', round=round_id+1, turn=turn+1,
                            tool=obj.get('tool'), tensor=obj.get('name')))
                if obj['tool'] == 'inspect' and turn < 8:
                    result = ex.inspect(obj['name'], obj.get('start',0), obj.get('count',32))
                    inspected[obj['name']] = result
                elif obj['tool'] == 'code':
                    source = obj['code']
                    artifact = run / f'round-{round_id+1}-turn-{turn+1}.py'
                    artifact.write_text(source)
                    changes = run_program(source, inspected, catalog)
                    ex.log(dict(event='program_output', source_sha256=hashlib.sha256(source.encode()).hexdigest(), changes=changes))
                    result = ex.candidate(changes, obj.get('reason',''))
                    break
                else:
                    raise ValueError('Submit an edit now' if turn == 8 else 'Unknown tool')
            except (ValueError, KeyError, TypeError) as exc:
                result = {'error': str(exc)}
            ex.log(dict(event='tool_result', round=round_id+1, turn=turn+1, result=result))
            messages.append(dict(role='user', content=json.dumps(result)))
        else:
            raise RuntimeError('Agent exhausted tool turns without a valid edit')
    # Dev is evaluated once after the fixed search budget; test stays sealed in this command.
    if args.smoke:
        ex.log(dict(event='integration_complete', persistent_assignments=len(ex.edits),
                    note='Eight-question smoke check only; no capability improvement established'))
    else:
        ex.log(dict(event='final_dev', metrics=ex.evaluate(ex.load('dev'))))


if __name__ == '__main__':
    main()
