"""Reward-only batch sizing probe; leaves B weights unchanged."""
import json
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from worker import load_model

target = '../../Qwen3.6-27B'
model = load_model(target, 'cuda:0')
tokenizer = AutoTokenizer.from_pretrained(target, local_files_only=True, padding_side='left')
tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
rows = [json.loads(line) for line in Path('data/reward.jsonl').open()]
# Include prompts across suites; the long run also allocates the full decode cache.
prompts = [tokenizer.apply_chat_template([dict(role='user', content=r['prompt'])],
            tokenize=False, add_generation_prompt=True) for r in rows[::max(1, len(rows)//32)][:32]]
recommendation = None
with torch.inference_mode():
    for size in (4, 16, 32):
        inputs = tokenizer(prompts[:size], padding=True, return_tensors='pt').to(model.device)
        try:
            model.generate(**inputs, max_new_tokens=2, do_sample=False)
            torch.cuda.synchronize()
            start = time.monotonic()
            output = model.generate(**inputs, min_new_tokens=128, max_new_tokens=128, do_sample=False)
            torch.cuda.synchronize()
            seconds = time.monotonic() - start
            print(json.dumps(dict(batch_size=size, seconds=seconds,
                  tokens_per_second=size*128/seconds,
                  peak_gb=torch.cuda.max_memory_allocated()/1e9)), flush=True)
            del output
            recommendation = size
        except torch.OutOfMemoryError:
            print(json.dumps(dict(batch_size=size, error='oom')), flush=True)
        del inputs
        torch.cuda.empty_cache()
    assert recommendation is not None
    inputs = tokenizer(prompts[:recommendation], padding=True, return_tensors='pt').to(model.device)
    start = time.monotonic()
    output = model.generate(**inputs, min_new_tokens=4096, max_new_tokens=4096, do_sample=False)
    torch.cuda.synchronize()
    print(json.dumps(dict(event='full_decode_pass', batch_size=recommendation,
                         seconds=time.monotonic()-start)), flush=True)
    print('Probe only: validate longest prompts before adopting this batch size.', flush=True)
