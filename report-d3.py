"""Summarize audited search; optionally compare base and replayed edits on dev."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--compare-dev', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('agent', root/'d3-agent.py')
    a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a)
    run = args.run.resolve()
    history = a.completed_history(run)
    records = [json.loads(x) for x in (run/'history.jsonl').read_text().splitlines()]
    baseline = next(r['metrics'] for r in records if r['event']=='baseline')
    accepted = [r for r in history if r['accepted']]
    best = accepted[-1]['metrics'] if accepted else baseline
    edits, originals = {}, {}
    for record in accepted:
        for c in record['realized']:
            key = (c['name'], c['index'])
            originals.setdefault(key, c['before'])
            edits[key] = c['after']
    actual = {k:v for k,v in edits.items() if v != originals[k]}
    result = dict(run=str(run), audited_rounds=len(history), accepted=len(accepted),
                  rejected=len(history)-len(accepted), baseline_reward=baseline,
                  best_reward=best, best_round=accepted[-1]['round'] if accepted else 0,
                  persistent_changed=len(actual), changed_tensors=dict(Counter(k[0] for k in actual)),
                  accuracy_gains=[dict(round=r['round'],accuracy=r['metrics']['macro_accuracy'])
                                 for i,r in enumerate(history) if r['accepted'] and
                                 r['metrics']['macro_accuracy'] >
                                 max([baseline['macro_accuracy']]+[p['metrics']['macro_accuracy'] for p in history[:i] if p['accepted']])],
                  final_dev=next((r['metrics'] for r in records if r['event']=='final_dev'),None),
                  test_evaluated=False)
    if args.compare_dev:
        out = root/'runs'/time.strftime('d3-dev-check-%Y%m%d-%H%M%S')
        out.mkdir()
        devices = [f'cuda:{i}' for i in range(8)]
        ex = a.Experiment(devices[0], 200, out, devices)
        try:
            rows = ex.load('dev')
            result['base_dev'] = ex.evaluate(rows)
            ex.resume(run)
            result['replayed_dev'] = ex.evaluate(rows)
            result['dev_accuracy_delta'] = {s:result['replayed_dev']['accuracy'][s]-v
                                           for s,v in result['base_dev']['accuracy'].items()}
            result['dev_macro_delta'] = result['replayed_dev']['macro_accuracy']-result['base_dev']['macro_accuracy']
            (out/'report.json').write_text(json.dumps(result,indent=2))
        finally:
            if ex.replicas: ex.replicas.close()
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
