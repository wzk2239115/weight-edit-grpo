"""MCP stdio transport to the trusted resident evaluator (no model/data files exposed)."""
import json
import os
import sys
import urllib.request

TOOLS = [
 dict(name='inspect',description='Read CURRENT flattened weight values, at most 256.',inputSchema=dict(type='object',properties=dict(name=dict(type='string'),start=dict(type='integer'),count=dict(type='integer')),required=['name'])),
 dict(name='history',description='Read exact previous before/after edits and measured results. Paginated.',inputSchema=dict(type='object',properties=dict(round=dict(type='integer'),offset=dict(type='integer'),count=dict(type='integer')),required=['round'])),
 dict(name='evaluate_patch',description='Submit a JSON file you wrote: list of {name,index,value}, 1..4096 literal scalar replacements. Independently evaluates 800 questions, audits and accepts or rolls back. Only ONE evaluated candidate per round.',inputSchema=dict(type='object',properties=dict(path=dict(type='string'),reason=dict(type='string')),required=['path','reason']))]

for line in sys.stdin:
    req=json.loads(line)
    if 'id' not in req:
        continue
    try:
        method=req['method']
        if method=='initialize':
            result=dict(protocolVersion=req['params']['protocolVersion'],capabilities=dict(tools={}),serverInfo=dict(name='weight-evaluator',version='1.0'))
        elif method=='tools/list': result=dict(tools=TOOLS)
        elif method=='ping': result={}
        elif method=='tools/call':
            body=json.dumps(req['params']).encode()
            request=urllib.request.Request(os.environ['WE_TOOL_URL'],body,{'Authorization':'Bearer '+os.environ['WE_TOOL_TOKEN'],'Content-Type':'application/json'})
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request,timeout=1800) as response: data=json.load(response)
            result=dict(content=[dict(type='text',text=json.dumps(data))],isError='error' in data)
        else: raise ValueError('Unsupported MCP method')
        reply=dict(jsonrpc='2.0',id=req['id'],result=result)
    except Exception as exc:
        reply=dict(jsonrpc='2.0',id=req['id'],error=dict(code=-32603,message=type(exc).__name__))
    print(json.dumps(reply),flush=True)
