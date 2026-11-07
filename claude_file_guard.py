"""Claude Code file tools may access only the experiment scratch workspace."""
import json
import os
from pathlib import Path
import sys
r=json.load(sys.stdin)
name=r.get('tool_name','')
if name == 'Bash' or name.startswith('mcp__weights__'):
    sys.exit(0)
if name not in ('Read','Write','Edit'):
    print('Only workspace file tools and weight-evaluator MCP tools are permitted.',file=sys.stderr)
    sys.exit(2)
p=Path(r.get('tool_input',{}).get('file_path','')).resolve()
workspace=Path(os.environ['WE_AGENT_WORKSPACE']).resolve()
if (name in ('Write','Edit') and p == workspace/'context.json') or not p.is_relative_to(workspace) or any(x.startswith('.') for x in p.relative_to(workspace).parts):
    print('Access outside the agent scratch workspace is denied.',file=sys.stderr)
    sys.exit(2)
