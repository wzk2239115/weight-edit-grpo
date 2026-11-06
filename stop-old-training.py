"""Inspect/stop only this clone's old train.py process tree; keep its artifacts."""
import os
from pathlib import Path
import signal
import sys
import time
root = Path(__file__).resolve().parent
processes = {}
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit():
        continue
    try:
        args = (entry / 'cmdline').read_bytes().split(b'\0')
        parent = int(next(x for x in (entry/'status').read_text().splitlines() if x.startswith('PPid:')).split()[1])
        cwd = (entry/'cwd').resolve()
    except (OSError, StopIteration):
        continue
    if cwd == root:
        processes[int(entry.name)] = (parent, [x.decode(errors='replace') for x in args if x])
parents = {pid for pid, (_,args) in processes.items()
           if any(Path(arg).name == 'train.py' for arg in args[:3])}
targets = set(parents)
while True:
    more = {pid for pid,(parent,_) in processes.items() if parent in targets}
    if more <= targets:
        break
    targets |= more
for pid in sorted(targets):
    print(pid, ' '.join(processes[pid][1]))
if '--stop' in sys.argv:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in targets:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        time.sleep(1)
    print('Stopped old training tree; logs retained')
