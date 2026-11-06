"""Resident per-GPU replicas; only numeric assignments cross the worker boundary."""
import hashlib
import json
import math
import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def score_rows(model, rows):
    totals, correct, nll = {}, {}, {}
    for row in rows:
        q = dict(type='choice', instructions=row['question'],
                 criteria={str(k): v for k, v in enumerate(row['options'])})
        answer = model.system_one(state=row['state'], questions={'q': q})['answers']['q']
        if 'error' in answer:
            raise RuntimeError('Benchmark returned an error')
        s = row['suite']
        totals[s] = totals.get(s, 0) + 1
        correct[s] = correct.get(s, 0) + (str(answer['choice']) == str(row['answer']))
        nll[s] = nll.get(s, 0) - math.log(max(1e-12, answer['probabilities'][str(row['answer'])]))
    return dict(n=totals, correct=correct, nll_sum=nll)


def merge_scores(parts, seconds):
    sums = {key: {} for key in ('n', 'correct', 'nll_sum')}
    for part in parts:
        for key in sums:
            for suite, value in part[key].items():
                sums[key][suite] = sums[key].get(suite, 0) + value
    totals = sums['n']
    acc = {s: sums['correct'][s] / n for s, n in totals.items()}
    losses = {s: sums['nll_sum'][s] / n for s, n in totals.items()}
    return dict(n=totals, accuracy=acc, macro_accuracy=sum(acc.values())/len(acc),
                nll=losses, macro_nll=sum(losses.values())/len(losses), seconds=seconds)


def patch_hash(changes):
    return hashlib.sha256(json.dumps(changes, sort_keys=True).encode()).hexdigest()


def temporary_score(model, params, rows, changes):
    import torch
    saved = []
    try:
        with torch.no_grad():
            for c in changes:
                flat = params[c['name']].view(-1)
                index = c['index']
                old = flat[index].clone()
                saved.append((flat, index, old))
                flat[index] = c['value']
                if float(flat[index].item()) != c['value']:
                    raise RuntimeError('Replica dtype/value mismatch')
        return score_rows(model, rows)
    finally:
        with torch.no_grad():
            for flat, index, old in reversed(saved):
                flat[index] = old


def worker(connection, device, root):
    import sys
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0, str(Path(root)/'models/d3-ultra'))
    try:
        from d3_runtime import D3
        model = D3.from_pretrained(Path(root)/'models/d3-ultra', device=device, local_files_only=True)
        params = dict(model.backbone.named_parameters())
        params['readout.weight'] = model.readout
        connection.send(dict(ready=device))
        while True:
            request = connection.recv()
            if request is None:
                return
            stats = temporary_score(model, params, request['rows'], request['changes'])
            connection.send(dict(stats=stats, patch_sha256=patch_hash(request['changes']), device=device))
    except BaseException as exc:
        try:
            connection.send(dict(error=f'{device}: {type(exc).__name__}: {exc}'))
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


class Replicas:
    def __init__(self, devices, root):
        self.connections, self.processes, self.devices = [], [], devices
        ctx = mp.get_context('spawn')
        try:
            for device in devices:
                parent, child = ctx.Pipe()
                proc = ctx.Process(target=worker, args=(child, device, str(root)), daemon=True)
                proc.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(proc)
            for connection in self.connections:
                result = self.receive(connection, 1800)
                if 'ready' not in result:
                    raise RuntimeError('Replica failed to initialize')
        except BaseException:
            self.close()
            raise

    @staticmethod
    def receive(connection, timeout=1800):
        if not connection.poll(timeout):
            raise RuntimeError('GPU replica timed out; refusing partial evaluation')
        result = connection.recv()
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def evaluate(self, model, rows, changes):
        start = time.monotonic()
        count = 1 + len(self.connections)
        shards = [rows[i::count] for i in range(count)]
        expected = patch_hash(changes)
        for connection, shard in zip(self.connections, shards[1:]):
            connection.send(dict(rows=shard, changes=changes))
        # The controller scores GPU 0 while the seven processes score their shards.
        with ThreadPoolExecutor(max_workers=1) as executor:
            local = executor.submit(score_rows, model, shards[0])
            parts = []
            for connection in self.connections:
                reply = self.receive(connection)
                if reply['patch_sha256'] != expected:
                    raise RuntimeError('Replica patch mismatch; refusing evaluation')
                parts.append(reply['stats'])
            parts.append(local.result())
        result = merge_scores(parts, time.monotonic()-start)
        if sum(result['n'].values()) != len(rows):
            raise RuntimeError('Incomplete multi-GPU evaluation')
        print(json.dumps(dict(event='parallel_eval', devices=count, questions=len(rows),
            shard_sizes=[len(s) for s in shards], patch_sha256=expected,
            seconds=result['seconds'])), flush=True)
        return result

    def close(self):
        for proc in self.processes:
            if proc.is_alive():
                proc.terminate()
        for proc in self.processes:
            proc.join(timeout=5)
        for connection in self.connections:
            connection.close()
