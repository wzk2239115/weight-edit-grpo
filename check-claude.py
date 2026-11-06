"""Real Claude Code + MCP integration, CPU fixture only; no capability claim."""
import importlib.util
import json
from pathlib import Path
import time
import torch
from claude_backend import run_round
spec=importlib.util.spec_from_file_location('agent',Path(__file__).with_name('d3-agent.py'))
a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
run=a.ROOT/'runs'/time.strftime('claude-check-%Y%m%d-%H%M%S'); run.mkdir()
ex=a.Experiment.__new__(a.Experiment)
ex.run=run; ex.torch=torch; ex.params={'layer.weight':torch.tensor([[1.,2.]],dtype=torch.bfloat16)}
ex.rows=[]; ex.history=[]; ex.original={}; ex.edits={}
ex.boundary=a.boundary_hashes()
(run/'boundary-hashes.json').write_text(json.dumps(ex.boundary))
ex.baseline=ex.best=dict(accuracy={'fixture':.5},macro_accuracy=.5,macro_nll=1.)
ex.evaluate=lambda rows:dict(accuracy={'fixture':.6},macro_accuracy=.6,macro_nll=.9)
catalog=[dict(name='layer.weight',shape=[1,2],dtype='torch.bfloat16')]
(run/'tensor-catalog.json').write_text(json.dumps(catalog))
run_round(ex,dict(round=1,baseline=ex.baseline,best=ex.best,tensors=catalog,history=[],current_edits={},recent=[]))
assert len(ex.history)==1 and ex.history[0]['actual_changed']>0
print('PASS: real Claude Code native tools + MCP inspect/submit + actual BF16 update + independent audit; CPU fixture only')
print(run)
