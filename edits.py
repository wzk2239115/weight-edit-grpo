"""The executor only reads/writes selected floating-point matrix entries."""
import json
import math
import re


def parse_action(text):
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    action = json.loads(text)
    if not isinstance(action, dict) or action.get('op') not in ('read', 'edit'):
        raise ValueError('Expected one JSON object, op=read or edit')
    return action


def rectangle(action, shape, budget=256):
    row, col = action['row'], action['col']
    if action['op'] == 'edit':
        values = action['values']
        if not isinstance(values, list) or not values or not isinstance(values[0], list):
            raise ValueError('values must be a nonempty rectangular 2D list')
        nr, nc = len(values), len(values[0])
        if not nc or any(not isinstance(v, list) or len(v) != nc for v in values):
            raise ValueError('Ragged matrix')
        if any(type(v) not in (float, int) or not math.isfinite(v) for line in values for v in line):
            raise ValueError('Only finite floating-point values are accepted')
    else:
        nr, nc = action['rows'], action['cols']
    if any(type(v) is not int for v in (row, col, nr, nc)):
        raise ValueError('Indices and dimensions must be integers')
    if min(row, col) < 0 or min(nr, nc) <= 0 or nr * nc > budget:
        raise ValueError('Invalid bounds or edit/read budget exceeded')
    if row + nr > shape[0] or col + nc > shape[1]:
        raise ValueError('Slice outside matrix')
    return slice(row, row + nr), slice(col, col + nc)


def reward(base, edited, tolerance):
    target = ('gsm8k', 'math')
    gain = sum(edited[s] - base[s] for s in target) / len(target)
    drops = {s: max(0., base[s] - edited[s]) for s in ('mmlu', 'ifeval')}
    violations = {s: d for s, d in drops.items() if d > tolerance}
    # No amount of target gain can compensate a failed preservation gate.
    value = -1. - sum(violations.values()) if violations else gain
    return dict(reward=value, target_gain=gain, drops=drops, passed=not violations)
