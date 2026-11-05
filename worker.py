"""Persistent B replicas: read matrices, temporarily patch, evaluate, restore."""
import json
import re
import traceback
from contextlib import contextmanager

from edits import rectangle


@contextmanager
def temporary_edit(tensor, action):
    import torch
    region = rectangle(action, tensor.shape)
    old = tensor[region].detach().clone()
    new = torch.tensor(action['values'], dtype=tensor.dtype, device=tensor.device)
    if not torch.isfinite(new).all():
        raise ValueError('Nonfinite values after dtype conversion')
    changed = int((old != new).sum().item())
    if changed == 0:
        raise ValueError('No weight changes survive dtype rounding')
    try:
        with torch.no_grad():
            tensor[region] = new
        yield changed
    finally:
        with torch.no_grad():
            tensor[region] = old


def load_model(path, device):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    cls = AutoModelForImageTextToText if hasattr(config, 'vision_config') else AutoModelForCausalLM
    return cls.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16,
                               device_map={'': device}).eval()


def score(row, text):
    if row['suite'] in ('math', 'gsm8k'):
        from math_verify import parse, verify
        gold = row['answer']
        if '\\boxed' not in gold:
            gold = '\\boxed{' + gold + '}'
        # Only score the final answer, excluding the reasoning section.
        text = text.split('</think>')[-1]
        return float(verify(parse(gold), parse(text)))
    if row['suite'] == 'mmlu':
        text = text.split('</think>')[-1]
        matches = re.findall(r'(?:answer\s*(?:is|:)?\s*|\\boxed\{)([A-J])', text, re.I)
        if not matches:
            matches = re.findall(r'^\s*([A-J])[.)\s]*$', text)
        return float(bool(matches) and matches[-1].upper() == row['answer'])
    from instruction_following_eval import evaluation_lib as lib
    inp = lib.InputExample(key=row['key'], instruction_id_list=row['instruction_id_list'],
                           prompt=row['prompt'], kwargs=row['kwargs'])
    result = lib.test_instruction_following_strict(inp, {row['prompt']: text.split('</think>')[-1]})
    return float(result.follow_all_instructions)


def evaluate(model, tokenizer, rows, batch_size, max_tokens, seed):
    import torch
    torch.manual_seed(seed)
    totals, results = {}, []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompts = []
        for row in batch:
            suffix = '\nGive your final answer in \\boxed{}.' if row['suite'] in ('math', 'gsm8k') else ''
            if row['suite'] == 'mmlu':
                suffix = '\nEnd with Answer: followed by the answer letter.'
            prompts.append(tokenizer.apply_chat_template(
                [dict(role='user', content=row['prompt'] + suffix)], tokenize=False,
                add_generation_prompt=True))
        inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id)
        for row, ids in zip(batch, output[:, inputs.input_ids.shape[1]:]):
            eos = (ids == tokenizer.eos_token_id).nonzero()
            if len(eos):
                ids = ids[:int(eos[0].item()) + 1]
            text = tokenizer.decode(ids, skip_special_tokens=False)
            for special in tokenizer.all_special_tokens:
                if special not in ('<think>', '</think>'):
                    text = text.replace(special, '')
            value = score(row, text)
            suite = row['suite']
            totals.setdefault(suite, []).append(value)
            results.append(dict(id=row['id'], suite=suite, category=row.get('category'), correct=value, response=text,
                                tokens=len(ids), truncated=bool(ids[-1] != tokenizer.eos_token_id)))
    return dict(scores={k: sum(v) / len(v) for k, v in totals.items()}, results=results)


def serve(model_path, gpu, incoming, outgoing, options):
    try:
        import torch
        from transformers import AutoTokenizer
        model = load_model(model_path, f'cuda:{gpu}')
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side='left')
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        matrices = {n: p for n, p in model.named_parameters() if p.ndim == 2 and '.layers.' in n}
        if not matrices:
            raise RuntimeError('No editable layer matrices found')
        catalog = {}
        for name, tensor in matrices.items():
            template = re.sub(r'(?<=layers\.)\d+', '{layer}', name)
            catalog[template] = list(tensor.shape)
        layer_ids = sorted({int(re.search(r'\.layers\.(\d+)\.', n)[1]) for n in matrices})
        outgoing.put(('ready', dict(templates=catalog, layers=layer_ids)))
        baseline = None
        rows = None
        while True:
            job = incoming.get()
            if job['op'] == 'stop':
                return
            try:
                if job['op'] == 'prepare':
                    if baseline is not None and rows == job['rows']:
                        outgoing.put(('ok', baseline))
                        continue
                    rows = job['rows']
                    baseline = evaluate(model, tokenizer, rows, **options)
                    outgoing.put(('ok', baseline))
                    continue
                action = job['action']
                if action['tensor'] not in matrices:
                    raise ValueError('Unknown or non-layer matrix')
                tensor = matrices[action['tensor']]
                region = rectangle(action, tensor.shape)
                if job['op'] == 'read':
                    values = tensor[region].detach().float().cpu().tolist()
                    outgoing.put(('ok', dict(values=values, dtype=str(tensor.dtype))))
                else:
                    if baseline is None:
                        raise RuntimeError('prepare must run first')
                    with temporary_edit(tensor, action) as changed:
                        edited = evaluate(model, tokenizer, rows, **options)
                        outgoing.put(('ok', dict(base=baseline['scores'], edited=edited['scores'],
                                                 changed=changed, results=edited['results'])))
            except (ValueError, KeyError, TypeError, IndexError):
                outgoing.put(('error', traceback.format_exc()))
            except Exception:
                outgoing.put(('fatal', traceback.format_exc()))
                return
    except Exception:
        outgoing.put(('fatal', traceback.format_exc()))
