"""CPU checks for the actual-edit/rollback boundary; no checkpoint required."""
import importlib.util
import tempfile
from pathlib import Path
import torch

spec = importlib.util.spec_from_file_location('agent', 'd3-agent.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)

def make(directory):
    ex = a.Experiment.__new__(a.Experiment)
    ex.torch = torch
    ex.params = {'layer.weight': torch.tensor([[1., 2.]], dtype=torch.bfloat16)}
    ex.run = Path(directory)
    ex.rows, ex.original, ex.edits, ex.history = [], {}, {}, []
    ex.baseline = ex.best = dict(accuracy={'a':.5,'b':.5}, macro_accuracy=.5, macro_nll=1.)
    return ex

with tempfile.TemporaryDirectory() as directory:
    ex = make(directory)
    ex.evaluate = lambda rows: dict(accuracy={'a':.6,'b':.5}, macro_accuracy=.55, macro_nll=.9)
    result = ex.candidate([dict(name='layer.weight', index=1, value=3.)], 'test accepted')
    assert result['accepted'] and result['actual_changed'] == 1
    assert ex.params['layer.weight'][0,1] == 3
    import json
    diff = json.loads((ex.run / 'accepted-diffs/0001.json').read_text())
    assert diff['changes'][0]['before'] == 2. and diff['changes'][0]['after'] == 3.
    assert diff['changes'][0]['delta'] == 1. and diff['parent_version'] == 0
    ex.evaluate = lambda rows: dict(accuracy={'a':.9,'b':.4}, macro_accuracy=.65, macro_nll=.8)
    assert not ex.candidate([dict(name='layer.weight', index=1, value=4.)], 'regression')['accepted']
    assert ex.params['layer.weight'][0,1] == 3
    assert len(list((ex.run/'accepted-diffs').glob('*.json'))) == 1
    def fail(rows):
        raise RuntimeError('simulated scoring failure')
    ex.evaluate = fail
    try:
        ex.candidate([dict(name='layer.weight', index=1, value=5.)], 'exception')
    except RuntimeError:
        pass
    assert ex.params['layer.weight'][0,1] == 3
    for change in [dict(name='layer.weight',index=1,value=3.000001),
                   dict(name='layer.weight',index=1,value=float('nan')),
                   dict(name='layer.weight',index=-1,value=1.)]:
        try:
            ex.candidate([change], 'invalid')
        except ValueError:
            pass
        else:
            raise AssertionError('invalid/no-op patch accepted')
    assert ex.inspect('layer.weight', 0, 2)['values'] == [1.,3.]
print('PASS: real BF16 assignments, no-op rejection, regression rollback, exception rollback')

# Resuming replays accepted audited rounds only, without inheriting unfinished edits.
with tempfile.TemporaryDirectory() as directory:
    import json
    parent = Path(directory)/'runs/parent'
    parent.mkdir(parents=True)
    ex = make(directory)
    ex.rows = [dict(id=str(i)) for i in range(8)]
    (parent/'config.json').write_text(json.dumps(dict(per_suite=2)))
    (parent/'reward-ids.json').write_text(json.dumps([r['id'] for r in ex.rows]))
    record = dict(event='candidate', round=1, accepted=True, reason='prior',
                  realized=[dict(name='layer.weight', index=1, before=2., after=3.)])
    records = [record, dict(event='round_audit',round=1,result=dict(status='PASS')),
               dict(event='round_complete',round=1),
               dict(record,round=2,realized=[dict(name='layer.weight',index=1,before=3.,after=4.)])]
    (parent/'history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    record['metrics'] = ex.baseline
    record['actual_changed'] = 1
    records[0] = record
    (parent/'history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    old_root = a.ROOT
    a.ROOT = Path(directory).resolve()
    ex.evaluate = lambda rows: ex.baseline
    try:
        ex.resume(parent)
        summary = ex.memory_summary()[0]
        assert summary['changed_tensors']['layer.weight']['examples'][0]['before']==2.
        assert summary['delta_macro_nll']==0.
        assert ex.recall(1)['changes'][0]['after']==3.
        assert ex.current_edit_summary()['layer.weight']['examples'][0]['current']==3.
        child = Path(directory)/'runs/child'
        child.mkdir()
        link = dict(event='resume', parent='runs/parent',
                    parent_history_sha256=a.hashlib.sha256((parent/'history.jsonl').read_bytes()).hexdigest())
        second = dict(record, round=2, accepted=False)
        (child/'history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in
            [link, second, dict(event='round_audit',round=2,result=dict(status='PASS')),
             dict(event='round_complete',round=2)]))
        assert [r['round'] for r in a.completed_history(child)]==[1,2]
    finally:
        a.ROOT = old_root
    assert ex.params['layer.weight'][0,1].item()==3.
    assert len(ex.history)==1 and ex.original[('layer.weight',1)].item()==2.
print('PASS: audited resume preserves accepted edits and excludes unfinished round')
