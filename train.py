"""Minimal tool-using GRPO. One A GPU; persistent independent B evaluation GPUs."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor'))
from edits import parse_action, reward
from worker import serve

SYSTEM = '''You edit another neural network's floating-point weights to improve math while
preserving knowledge and instruction following. Select layers and matrix entries yourself.
Reply with exactly one JSON object. To inspect weights:
{"op":"read","tensor":"exact name","row":0,"col":0,"rows":4,"cols":4}
To submit your final edit (absolute NEW floating-point values, not deltas):
{"op":"edit","tensor":"exact name","row":0,"col":0,"values":[[0.012,-0.034]]}
You may read several blocks, then submit ONE rectangular edit with at most 256 elements.
No gradients, optimizers, predefined matrix transformations, or edits to inference settings.
The executor only copies the numbers you supply. A no-op edit is invalid.'''


class WorkerFailure(RuntimeError):
    pass


def receive(worker):
    tag, data = worker[1].get(timeout=7200)
    if tag == 'fatal':
        raise WorkerFailure(data)
    if tag not in ('ok', 'ready'):
        raise RuntimeError(data)
    return data


def select_rows(path, limit, seed):
    with path.open() as stream:
        rows = [json.loads(line) for line in stream]
    rng = random.Random(seed)
    rng.shuffle(rows)
    selected = []
    for suite in ('gsm8k', 'math', 'mmlu', 'ifeval'):
        subset = [r for r in rows if r['suite'] == suite]
        if not subset:
            raise ValueError(f'Missing suite: {suite}')
        selected.extend(subset if limit == 0 else subset[:limit])
    return selected


def token_logps(model, prefix, completion):
    import torch
    ids = torch.cat([prefix, completion], dim=1)
    logits = model(ids).logits[:, prefix.shape[1]-1:-1, :].float()
    return logits.log_softmax(-1).gather(-1, completion.unsqueeze(-1)).squeeze(-1)


def episode(model, tokenizer, catalog, worker, args):
    import torch
    messages = [dict(role='system', content=SYSTEM), dict(role='user', content=
        'Goal: improve mathematical reasoning without losing other abilities.\n'
        + 'Available layer indices and matrix templates:\n' + json.dumps(catalog))]
    trajectory = []
    for step in range(args.reads + 1):
        if step == args.reads:
            messages.append(dict(role='user', content='Read budget exhausted. Submit your final edit now.'))
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prefix = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids.to(model.device)
        if prefix.shape[1] > args.prompt_tokens:
            return trajectory, None, 'Prompt budget exceeded', messages
        with torch.no_grad():
            ids = model.generate(prefix, attention_mask=torch.ones_like(prefix), do_sample=True,
                                 temperature=args.temperature, top_p=.95,
                                 max_new_tokens=args.edit_tokens, pad_token_id=tokenizer.pad_token_id,
                                 use_cache=True)
            completion = ids[:, prefix.shape[1]:]
            old = token_logps(model, prefix, completion).detach()
        text = tokenizer.decode(completion[0], skip_special_tokens=True)
        trajectory.append((prefix.detach().cpu(), completion.detach().cpu(), old.cpu()))
        messages.append(dict(role='assistant', content=text))
        try:
            action = parse_action(text)
            if action['op'] == 'edit':
                return trajectory, action, None, messages
            if step == args.reads:
                raise ValueError('Read budget exhausted')
            worker[0].put(dict(op='read', action=action))
            result = receive(worker)
        except WorkerFailure:
            raise
        except Exception as exc:
            result = dict(error=str(exc))
        # User role keeps this compatible with models without tool-role templates.
        messages.append(dict(role='user', content='Weight reader result: ' + json.dumps(result)))
    return trajectory, None, 'No edit submitted', messages


def update(model, optimizer, trajectories, rewards, args):
    import torch
    values = torch.tensor(rewards, dtype=torch.float32)
    advantages = (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)
    if float(values.std(unbiased=False)) < 1e-8:
        return dict(updated=False, reason='equal_group_rewards', loss=0.)
    total_loss = 0.
    for epoch in range(args.update_epochs):
        optimizer.zero_grad(set_to_none=True)
        for trajectory, advantage in zip(trajectories, advantages):
            count = sum(t[1].numel() for t in trajectory)
            for prefix, completion, old in trajectory:
                logps = token_logps(model, prefix.to(model.device), completion.to(model.device))
                ratio = (logps - old.to(model.device)).exp()
                a = advantage.to(model.device)
                unclipped = ratio * a
                clipped = ratio.clamp(1-args.clip, 1+args.clip) * a
                loss = -torch.minimum(unclipped, clipped).sum() / max(1, count) / len(trajectories)
                loss.backward()
                total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    return dict(updated=True, loss=total_loss / args.update_epochs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--editor', required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--b-gpus', default='1,2,3,4,5,6,7')
    p.add_argument('--a-gpu', type=int, default=0)
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--group', type=int, default=7)
    p.add_argument('--reads', type=int, default=4)
    p.add_argument('--per-suite', type=int, default=128, help='0 = all rows in split')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--eval-tokens', type=int, default=2048)
    p.add_argument('--edit-tokens', type=int, default=2048)
    p.add_argument('--prompt-tokens', type=int, default=12000)
    p.add_argument('--temperature', type=float, default=.8)
    p.add_argument('--tolerance', type=float, default=.02)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--clip', type=float, default=.2)
    p.add_argument('--update-epochs', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default='runs/first')
    p.add_argument('--evaluate-patch', help='Evaluate one saved action JSON; no A training')
    p.add_argument('--split', choices=['dev', 'test'], default='dev')
    args = p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    if args.group < 2 or args.update_epochs < 1:
        p.error('group >= 2 and update-epochs >= 1 required')
    out = Path(args.out).resolve()
    if ROOT not in out.parents:
        p.error('Outputs must be inside this repository')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(vars(args), indent=2))
    torch.manual_seed(args.seed)
    ctx = mp.get_context('spawn')
    workers, processes = [], []
    try:
        gpus = [int(x) for x in args.b_gpus.split(',')]
        if len(gpus) != len(set(gpus)) or args.a_gpu in gpus:
            p.error('B GPU ids must be unique and separate from A')
        if args.evaluate_patch:
            gpus = gpus[:1]
        for gpu in gpus:
            qin, qout = ctx.Queue(), ctx.Queue()
            proc = ctx.Process(target=serve, args=(args.target, gpu, qin, qout,
                dict(batch_size=args.batch_size, max_tokens=args.eval_tokens, seed=args.seed)))
            proc.start()
            workers.append((qin, qout))
            processes.append(proc)
        catalogs = [receive(w) for w in workers]
        if any(c != catalogs[0] for c in catalogs):
            raise RuntimeError('B workers disagree on model structure')
        (out / 'catalog.json').write_text(json.dumps(catalogs[0], indent=2))
        if args.evaluate_patch:
            rows = select_rows(ROOT / f'data/{args.split}.jsonl', args.per_suite, args.seed)
            workers[0][0].put(dict(op='prepare', rows=rows))
            base = receive(workers[0])
            action = json.loads(Path(args.evaluate_patch).read_text())
            workers[0][0].put(dict(op='edit', action=action))
            result = receive(workers[0])
            (out / f'{args.split}.json').write_text(json.dumps(dict(baseline=base, edited=result)))
            print(json.dumps(result['edited']), flush=True)
            return
        tokenizer = AutoTokenizer.from_pretrained(args.editor, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(args.editor, local_files_only=True,
            dtype=torch.bfloat16, device_map={'': f'cuda:{args.a_gpu}'})
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            task_type='CAUSAL_LM'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
        model.config.use_cache = False
        model.eval()  # Disable dropout; gradients still flow through LoRA during updates.
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        for step in range(args.steps):
            directory = out / f'step-{step:04d}'
            directory.mkdir(exist_ok=True)
            rows = select_rows(ROOT / 'data/reward.jsonl', args.per_suite, args.seed + step)
            (directory / 'rows.json').write_text(json.dumps([x['id'] for x in rows]))
            for w in workers:
                w[0].put(dict(op='prepare', rows=rows))
            baselines = [receive(w) for w in workers]
            (directory / 'baseline.json').write_text(json.dumps(baselines[0]))
            # Independent replicas should yield identical deterministic baseline scores.
            if any(b['scores'] != baselines[0]['scores'] for b in baselines):
                raise RuntimeError('B worker baselines disagree; do not compare noisy groups')
            trajectories, records = [], []
            for start in range(0, args.group, len(workers)):
                pending = []
                for i in range(start, min(start + len(workers), args.group)):
                    w = workers[i - start]
                    trajectory, action, error, messages = episode(model, tokenizer, catalogs[0], w, args)
                    trajectories.append(trajectory)
                    record = dict(candidate=i, action=action, error=error, messages=messages)
                    records.append(record)
                    if action is not None:
                        w[0].put(dict(op='edit', action=action))
                        pending.append((w, record))
                    else:
                        record['reward'] = -2.
                for w, record in pending:
                    try:
                        result = receive(w)
                        record.update(result)
                        record.update(reward(result['base'], result['edited'], args.tolerance))
                    except WorkerFailure:
                        raise
                    except Exception as exc:
                        record.update(error=str(exc), reward=-2.)
            rewards = [r['reward'] for r in records]
            # Persist candidates before the update so failed runs retain their evidence.
            (directory / 'candidates.json').write_text(json.dumps(records))
            for record in records:
                if record['action'] is not None:
                    (directory / f'patch-{record["candidate"]}.json').write_text(json.dumps(record['action']))
            update_result = update(model, optimizer, trajectories, rewards, args)
            summary = dict(step=step, rewards=rewards, **update_result)
            (directory / 'update.json').write_text(json.dumps(summary))
            from tensorboard_logs import write_run
            write_run(out)
            print(json.dumps(summary), flush=True)
            model.save_pretrained(directory / 'editor-adapter')
            torch.save(optimizer.state_dict(), directory / 'optimizer.pt')
    finally:
        for w in workers:
            w[0].put(dict(op='stop'))
        for proc in processes:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join()


if __name__ == '__main__':
    main()
