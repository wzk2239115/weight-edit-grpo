"""Claude Code owns the agent loop. Python owns only the immutable evaluation boundary."""
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import secrets
import shlex
import shutil
import subprocess
import sys
import threading

PROMPT = '''You are the weight-editing coding agent. Use your native Bash/Read/Write/Edit tools to
maintain notes, write/run arbitrary Python analysis and generate patch JSON files in this workspace.
Bash is unsandboxed. ALL writes (including caches/temp files) MUST stay inside this workspace
under /home/jovyan/h800fast/wangzekai/tmp. Never modify the base model, framework, scorer,
benchmark files, run evidence, or credentials; never read secrets or process environments.
Use the Python interpreter named in context.json. Weight updates are applied ONLY by MCP.
Do not launch extra GPU model replicas or interfere with the evaluator processes. The weights MCP provides current
weights, exact prior experiments, and independent evaluation. Choose ANY layer and actual
floating-point replacements; architecture/runtime/scorer/prompts/data cannot be changed.
Read context.json first. It contains all previous round summaries and current accepted edits.
Use history to recover older exact changes; rejected edits are absent from the current model.
Write a JSON list [{"name":"tensor name","index":123,"value":0.0123}, ...] to patch.json.
Submit it with evaluate_patch and a hypothesis. Submit exactly one valid candidate this round.
Inspect and revise freely before submission. After measured feedback, update notes.md and finish.
Accuracy is primary; lower NLL alone is not evidence of answering more questions correctly.
Do not access benchmark contents, labels or credentials, or modify framework files. Never fabricate scores.
'''


def run_round(ex, context):
    root=Path(__file__).resolve().parent
    workspace=ex.run/'claude-workspace'
    workspace.mkdir(exist_ok=True)
    (workspace/'context.json').write_text(json.dumps(context))
    token=secrets.token_hex(32)
    outcome=[]
    audit_failed=[]
    round_id=context['round']
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_POST(self):
            if self.headers.get('Authorization') != 'Bearer '+token:
                self.send_error(403); return
            try:
                request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                name,args=request['name'],request.get('arguments',{})
                ex.log(dict(event='claude_tool',round=round_id,tool=name,arguments=args))
                if name=='inspect': result=ex.inspect(**args)
                elif name=='history': result=ex.recall(args['round'],args.get('offset',0),args.get('count',256))
                elif name=='evaluate_patch':
                    if outcome: raise ValueError('This round already evaluated a candidate')
                    path=(workspace/args['path']).resolve()
                    if not path.is_relative_to(workspace.resolve()) or path.is_symlink():
                        raise ValueError('Patch must be inside workspace')
                    source=path.read_text()
                    changes=json.loads(source)
                    artifact=ex.run/f'round-{round_id}-claude-patch-{hashlib.sha256(source.encode()).hexdigest()[:12]}.json'
                    artifact.write_text(source)
                    ex.log(dict(event='claude_patch',round=round_id,path=artifact.name,
                                source_sha256=hashlib.sha256(source.encode()).hexdigest(),changes=changes))
                    result=ex.candidate(changes,args['reason'])
                    outcome.append(result)
                    audit=subprocess.run([sys.executable,str(root/'audit-d3.py'),str(ex.run)],capture_output=True,text=True,check=True)
                    ex.log(dict(event='round_audit',round=round_id,result=json.loads(audit.stdout)))
                    ex.log(dict(event='round_complete',round=round_id,accepted=result['accepted'],actual_changed=result['actual_changed'],metrics=result['metrics']))
                else: raise ValueError('Unknown weight tool')
                ex.log(dict(event='claude_tool_result',round=round_id,tool=name,result=result))
            except Exception as exc:
                if outcome and isinstance(exc, subprocess.CalledProcessError): audit_failed.append(True)
                result=dict(error=f'{type(exc).__name__}: {exc}')
            body=json.dumps(result).encode()
            self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(body)
    server=HTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    try:
        settings=json.loads((root/'secrets/d3-api.json').read_text()) if (root/'secrets/d3-api.json').exists() else json.loads((Path.home()/'.config/opencode/opencode.json').read_text())['provider']['360-proxy']['options']
        env=dict(os.environ,ANTHROPIC_AUTH_TOKEN=settings['apiKey'],ANTHROPIC_BASE_URL=settings['baseURL'].removesuffix('/v1'),
                 CLAUDE_CONFIG_DIR=str(ex.run/'claude-config'),CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC='1',DISABLE_AUTOUPDATER='1',
                 WE_AGENT_WORKSPACE=str(workspace),WE_TOOL_URL=f'http://127.0.0.1:{server.server_port}',WE_TOOL_TOKEN=token)
        for k in ('ANTHROPIC_API_KEY','CLAUDECODE','http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','all_proxy'):
            env.pop(k,None)
        mcp=json.dumps(dict(mcpServers=dict(weights=dict(command=sys.executable,args=[str(root/'claude_weight_tools.py')]))))
        guard=shlex.join([sys.executable,str(root/'claude_file_guard.py')])
        hooks=json.dumps(dict(sandbox=dict(enabled=False),hooks=dict(PreToolUse=[dict(matcher='.*',hooks=[dict(type='command',command=guard)])])))
        cli=os.environ.get('CLAUDE_BIN') or shutil.which('claude') or str(root/'tools/claude')
        args=[cli,'-p',PROMPT,'--model',os.environ.get('DEEPSEEK_MODEL','deepseek/deepseek-v4.1-flash'),
              '--output-format','stream-json','--verbose','--tools','Bash,Read,Write,Edit','--allowedTools','Bash,Read,Write,Edit,mcp__weights__*',
              '--permission-mode','dontAsk','--setting-sources','','--settings',hooks,'--strict-mcp-config','--mcp-config',mcp]
        session_file=ex.run/'claude-session.txt'
        if session_file.exists(): args += ['--resume',session_file.read_text().strip()]
        trajectory=ex.run/f'round-{round_id}-claude.jsonl'
        with trajectory.open('w') as stream, (ex.run/f'round-{round_id}-claude.stderr').open('w') as errors:
            proc=subprocess.run(args,cwd=workspace,env=env,stdout=stream,stderr=errors,timeout=1800)
        for line in trajectory.open():
            event=json.loads(line)
            if event.get('session_id'): session_file.write_text(event['session_id'])
        if proc.returncode or not outcome or audit_failed:
            raise RuntimeError(f'Claude Code failed or submitted no evaluated patch; inspect {trajectory.name}')
    finally:
        server.shutdown(); server.server_close()
