"""Interactive credential entry; never echo or commit the key."""
import getpass
import json
import os
import sys
from pathlib import Path
root = Path(__file__).resolve().parent
os.umask(0o077)
key = getpass.getpass('360 API key (hidden): ').strip()
if '--hex' in sys.argv:
    key = bytes.fromhex(key).decode()
if not key:
    raise SystemExit('No key saved')
folder = root / 'secrets'
folder.mkdir(exist_ok=True)
path = folder / 'd3-api.json'
path.write_text(json.dumps(dict(apiKey=key, baseURL='https://api.360.cn/v1')))
path.chmod(0o600)
print('Credential saved privately. Run d3-agent.py --api-check to verify.')
