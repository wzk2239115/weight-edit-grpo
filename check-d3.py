"""Load the real checkpoint and measure four public tasks without API calls."""
import importlib.util
import json
import time
from pathlib import Path
spec = importlib.util.spec_from_file_location('agent', Path(__file__).with_name('d3-agent.py'))
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
run = a.ROOT / 'runs' / time.strftime('d3-check-%Y%m%d-%H%M%S')
run.mkdir(parents=True)
ex = a.Experiment('cuda:0', 2, run)
name = next(n for n,p in ex.params.items() if p.ndim == 2 and 'mlp' in n and not n.startswith('visual.'))
old = ex.inspect(name, 0, 1)['values'][0]
# Verify assignment on the checkpoint, then exact restoration. This is an integration
# check, not an agent-proposed improvement and not part of search statistics.
p = ex.params[name].view(-1)
with ex.torch.no_grad():
    saved = p[0].clone()
    try:
        p[0] = old + .03125
        assert not ex.torch.equal(p[0], saved)
    finally:
        p[0] = saved
assert p[0].item() == old
print(json.dumps(dict(event='checkpoint_edit_restore_pass', tensor=name,
                      note='Integration probe only; zero persistent edits; no search run yet')), flush=True)
