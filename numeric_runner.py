"""Bounded Python numerical interpreter. Never exec/eval agent source."""
import ast
import json
import operator
import sys

class NumericalProgram:
    def __init__(self, tensors, catalog):
        self.env = dict(tensors=tensors, catalog=catalog)
        self.edits = []
        self.steps = 0
        self.ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
        self.calls = dict(range=lambda *a: range(*a), len=len, abs=abs, min=min, max=max,
                          float=float, int=int, sum=sum, round=round, emit=self.emit)

    def tick(self):
        self.steps += 1
        if self.steps > 100000:
            raise ValueError('Program operation budget exceeded')

    def emit(self, name, index, value):
        if len(self.edits) >= 4096:
            raise ValueError('Assignment budget exceeded')
        self.edits.append(dict(name=name, index=index, value=value))

    def expr(self, n):
        self.tick()
        if isinstance(n, ast.Constant) and type(n.value) in (str, int, float, bool, type(None)):
            return n.value
        if isinstance(n, ast.Name) and n.id in self.env:
            return self.env[n.id]
        if isinstance(n, (ast.List, ast.Tuple)):
            return [self.expr(x) for x in n.elts]
        if isinstance(n, ast.Subscript):
            return self.expr(n.value)[self.expr(n.slice)]
        if isinstance(n, ast.BinOp) and type(n.op) in self.ops:
            left, right = self.expr(n.left), self.expr(n.right)
            if type(left) not in (int,float) or type(right) not in (int,float):
                raise ValueError('Arithmetic requires numbers')
            return self.ops[type(n.op)](left, right)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd, ast.Not)):
            value = self.expr(n.operand)
            return not value if isinstance(n.op, ast.Not) else (-value if isinstance(n.op, ast.USub) else +value)
        if isinstance(n, ast.Compare) and len(n.ops) == 1:
            ops = {ast.Lt:operator.lt, ast.LtE:operator.le, ast.Gt:operator.gt,
                   ast.GtE:operator.ge, ast.Eq:operator.eq, ast.NotEq:operator.ne}
            if type(n.ops[0]) in ops:
                return ops[type(n.ops[0])](self.expr(n.left), self.expr(n.comparators[0]))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in self.calls and not n.keywords:
            return self.calls[n.func.id](*[self.expr(x) for x in n.args])
        raise ValueError('Forbidden expression: ' + type(n).__name__)

    def block(self, statements):
        for n in statements:
            self.tick()
            if isinstance(n, ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0], ast.Name):
                name = n.targets[0].id
                if name.startswith('_') or name in self.calls or name in ('tensors','catalog'):
                    raise ValueError('Reserved variable')
                self.env[name] = self.expr(n.value)
            elif isinstance(n, ast.Expr):
                self.expr(n.value)
            elif isinstance(n, ast.For) and isinstance(n.target, ast.Name) and not n.orelse:
                if n.target.id.startswith('_') or n.target.id in self.calls or n.target.id in ('tensors','catalog'):
                    raise ValueError('Reserved loop variable')
                for value in self.expr(n.iter):
                    self.tick()
                    self.env[n.target.id] = value
                    self.block(n.body)
            elif isinstance(n, ast.If):
                self.block(n.body if self.expr(n.test) else n.orelse)
            else:
                raise ValueError('Forbidden statement: ' + type(n).__name__)

    def run(self, code):
        if len(code) > 64000:
            raise ValueError('Source too long')
        self.block(ast.parse(code).body)
        return self.edits

if __name__ == '__main__':
    # Linux child: also bound resource exhaustion, separately from the language boundary.
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (3,3))
    resource.setrlimit(resource.RLIMIT_AS, (256*1024**2,256*1024**2))
    request = json.load(sys.stdin)
    try:
        result = NumericalProgram(request['tensors'], request['catalog']).run(request['code'])
        print(json.dumps(dict(changes=result), allow_nan=False))
    except Exception as exc:
        print(json.dumps(dict(error=type(exc).__name__ + ': ' + str(exc)[:300])))
