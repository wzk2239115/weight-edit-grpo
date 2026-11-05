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
from prompts import SYSTEM, diagnostics

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


def episode(model, tokenizer, catalog, args, feedback, rows):
    """Yield tool requests so independent search episodes can share A and parallel B replicas."""
    import torch
    import re
    messages = [dict(role='system', content=SYSTEM), dict(role='user', content=
        'Goal: improve mathematical reasoning without losing other abilities.\n'
        + 'Available layer indices and matrix templates:\n' + json.dumps(catalog)
        + '\nBaseline observations (reward data only):\n' + json.dumps(feedback)
        + f'\nBudget: {args.reads} inspections, {args.trials} trial edits, then finish.')]
    trajectory, trials = [], []
    surveyed, layers_read, families_read = False, set(), set()
    inspections, attempts = 0, 0
    for step in range(args.reads + args.trials + 1):
        if step == args.reads + args.trials:
            messages.append(dict(role='user', content='Action budget exhausted. Use finish to select a measured trial, or trial=-1 to withdraw.'))
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prefix = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids.to(model.device)
        if prefix.shape[1] > args.prompt_tokens:
            break
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
            if not isinstance(action.get('analysis'), str) or not action['analysis'].strip():
                raise ValueError('Include your evidence and decision in the analysis field')
            if action['op'] == 'finish':
                selected = action.get('trial')
                if type(selected) is not int or not -1 <= selected < len(trials):
                    raise ValueError('Choose a measured trial id or -1 to withdraw')
                if selected == -1:
                    result = dict(base=feedback['scores'], edited=feedback['scores'], changed=0,
                                  results=[], reward=0., target_gain=0., drops={}, passed=True)
                    patch = None
                else:
                    chosen = trials[selected]
                    patch, result = chosen['action'], chosen['result']
                return trajectory, dict(action=patch, error=None, messages=messages,
                                         trials=trials, selected_trial=selected, **result)
            if step == args.reads + args.trials:
                raise ValueError('Only finish is allowed now')
            if action['op'] == 'edit':
                if not surveyed or len(layers_read) < 3 or len(families_read) < 2:
                    raise ValueError('Inspect the global survey, three distinct layers and two matrix families first')
                if attempts >= args.trials:
                    raise ValueError('Trial budget exhausted; inspect or finish')
                attempts += 1
            else:
                if inspections >= args.reads:
                    raise ValueError('Inspection budget exhausted; trial an edit or finish')
                inspections += 1
            result = yield dict(op=action['op'], action=action)
            if 'error' in result:
                raise ValueError(result['error'])
            if action['op'] == 'edit':
                result.update(reward(result['base'], result['edited'], args.tolerance))
                trial_id = len(trials)
                trials.append(dict(id=trial_id, action=action, result=result))
                result = dict(trial=trial_id, changed=result['changed'],
                              **reward(result['base'], result['edited'], args.tolerance),
                              observations=diagnostics(rows, dict(scores=result['edited'], results=result['results'])))
            elif action['op'] == 'survey':
                surveyed = True
            else:
                layers_read.add(int(re.search(r'\.layers\.(\d+)\.', action['tensor'])[1]))
                families_read.add(action['tensor'].split('.layers.')[1].split('.', 1)[1])
        except Exception as exc:
            result = dict(error=str(exc))
        messages.append(dict(role='user', content='Tool observation: ' + json.dumps(result)))
    return trajectory, dict(action=None, error='No valid finish before context/action budget',
                             messages=messages, trials=trials, reward=-2.)


def run_searches(generators, workers):
    """Round-robin A turns; dispatch B tool requests before waiting for their feedback."""
    active = dict(enumerate(generators))
    replies, finished = {}, {}
    while active:
        pending = []
        for index, generator in list(active.items()):
            try:
                request = generator.send(replies.pop(index, None))
                workers[index][0].put(request)
                pending.append(index)
            except StopIteration as done:
                finished[index] = done.value
                del active[index]
        for index in pending:
            try:
                replies[index] = receive(workers[index])
            except WorkerFailure:
                raise
            except Exception as exc:
                replies[index] = dict(error=str(exc))
    return [finished[i] for i in range(len(generators))]


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
    p.add_argument('--trials', type=int, default=3, help='Maximum evaluated edits per episode')
    p.add_argument('--reads', type=int, default=12, help='Maximum inspection actions, including survey')
    p.add_argument('--per-suite', type=int, default=0, help='0 (default) = complete split; positive limits are debug only')
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
    if args.group < 2 or args.update_epochs < 1 or args.reads < 4 or args.trials < 1:
        p.error('group >= 2, update-epochs >= 1 and reads >= 4, trials >= 1 required')
    if args.per_suite < 0:
        p.error('per-suite must be nonnegative')
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
            # Full-set comparisons keep identical ordering/batching across groups.
            row_seed = args.seed if args.per_suite == 0 else args.seed + step
            rows = select_rows(ROOT / 'data/reward.jsonl', args.per_suite, row_seed)
            print(json.dumps(dict(step=step, evaluation_questions=len(rows), counts={
                suite: sum(r['suite'] == suite for r in rows)
                for suite in ('gsm8k', 'math', 'mmlu', 'ifeval')})), flush=True)
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
                indices = range(start, min(start + len(workers), args.group))
                searches = [episode(model, tokenizer, catalogs[0], args,
                                    diagnostics(rows, baselines[0]), rows) for i in indices]
                for i, (trajectory, record) in zip(indices, run_searches(searches, workers)):
                    trajectories.append(trajectory)
                    records.append(dict(candidate=i, **record))
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
