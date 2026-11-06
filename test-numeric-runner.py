from numeric_runner import NumericalProgram
p=NumericalProgram({'x':dict(start=10, values=[.5, -.25])},[])
code="s = tensors['x']\nfor i in range(len(s['values'])):\n    emit('x', s['start'] + i, s['values'][i] + 0.125)"
assert p.run(code)==[dict(name='x',index=10,value=.625),dict(name='x',index=11,value=-.125)]
for code in ["import os", "open('data/d3/test.jsonl')", "tensors.__class__", "exec('print(1)')",
             "eval('1')", "__import__('os')", "emit = 3", "for tensors in range(1):\n    emit('x',0,1)",
             "tensors['x'] = 3", "while True:\n    pass"]:
    try:
        NumericalProgram({},[]).run(code)
    except (ValueError,KeyError):
        pass
    else:
        raise AssertionError('Forbidden code accepted: '+code)
print('PASS: numerical loops and computed floats; file/import/exec/object access blocked')
