"""Check sharded metric reduction and replica rollback without checkpoint downloads."""
import math
import torch
from d3_parallel import score_rows, merge_scores, temporary_score

class Model:
    def __init__(self):
        self.p = torch.tensor([1.], dtype=torch.bfloat16)
    def system_one(self, state, questions):
        if state == 'fail':
            raise RuntimeError('scoring failed')
        return {'answers': {'q': {'choice': str(state%2), 'probabilities': {'0':.75, '1':.25}}}}

model=Model()
rows=[dict(suite='a' if i<9 else 'b',state=i,question='q', options=['x','y'],answer=str(i%3%2)) for i in range(19)]
whole=merge_scores([score_rows(model,rows)], 0)
parts=merge_scores([score_rows(model,rows[i::8]) for i in range(8)], 0)
assert whole['n']==parts['n'] and whole['accuracy']==parts['accuracy']
assert math.isclose(whole['macro_nll'],parts['macro_nll'],abs_tol=1e-12)
params={'x':model.p}
changes=[dict(name='x', index=0,value=2.)]
assert temporary_score(model,params,rows,changes)==score_rows(model,rows)
assert model.p.item()==1.
try:
    temporary_score(model,params,[dict(rows[0],state='fail')],changes)
except RuntimeError:
    pass
else:
    raise AssertionError('failure was hidden')
assert model.p.item()==1.
try:
    temporary_score(model,params,rows,[dict(name='x',index=0,value=1.000001)])
except RuntimeError:
    pass
else:
    raise AssertionError('replica dtype mismatch was hidden')
assert model.p.item()==1.
print('PASS: uneven 8-way reduction, identical counts/accuracy/NLL, replica success/error rollback')
