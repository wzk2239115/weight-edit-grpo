"""Install the official Claude Code native npm artifact under this repo only."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import urllib.request
root=Path(__file__).resolve().parent
if not str(root).startswith('/home/jovyan/h800fast/wangzekai/tmp/'):
    raise SystemExit('Run this installer in the remote tmp repository')
version='2.1.311'
folder=root/'tools'; folder.mkdir(exist_ok=True)
url=f'https://registry.npmjs.org/@anthropic-ai/claude-code-linux-x64/{version}'
metadata=json.load(urllib.request.urlopen(url,timeout=60))
archive=folder/'claude-code.tgz'
with urllib.request.urlopen(metadata['dist']['tarball'],timeout=180) as src, archive.open('wb') as out:
    while chunk:=src.read(1024*1024): out.write(chunk)
algorithm,digest=metadata['dist']['integrity'].split('-',1)
assert algorithm=='sha512'
assert base64.b64encode(hashlib.sha512(archive.read_bytes()).digest()).decode()==digest
with tarfile.open(archive) as tar:
    member=next(m for m in tar.getmembers() if Path(m.name).name=='claude' and m.isfile())
    binary=folder/'claude'; binary.write_bytes(tar.extractfile(member).read()); binary.chmod(0o700)
subprocess.run([str(binary),'--version'],check=True)
