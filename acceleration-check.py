"""GPU execution checks, then a short inference using the real read-only B."""
import argparse
import importlib.metadata as metadata
import inspect
import time

import torch
import triton
from causal_conv1d import causal_conv1d_fn
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule


def external_implementation(fn):
    # Transformers wraps the selected package implementation in a closure.
    seen = set()
    while callable(fn) and id(fn) not in seen:
        seen.add(id(fn))
        variables = inspect.getclosurevars(fn).nonlocals if inspect.isfunction(fn) else {}
        if 'implementation' in variables:
            return variables['implementation']
        fn = getattr(fn, '__wrapped__', None)
    raise RuntimeError('Cannot verify Transformers kernel dispatch')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='../../Qwen3.6-27B')
    args = parser.parse_args()
    assert torch.cuda.is_available()
    print('GPU:', torch.cuda.get_device_name(0), 'torch:', torch.__version__, flush=True)
    print('triton:', triton.__version__, flush=True)
    for package in ('fla-core', 'flash-linear-attention', 'causal-conv1d'):
        print(package, metadata.version(package), flush=True)
    torch.manual_seed(42)
    x = torch.randn(2, 128, 128, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(128, 4, device='cuda', dtype=torch.bfloat16)
    actual = causal_conv1d_fn(x, w, activation='silu')
    expected = torch.nn.functional.silu(torch.nn.functional.conv1d(
        x, w[:, None], padding=3, groups=128)[..., :128])
    torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.04)
    shape = (2, 128, 4, 128)
    q, k, v = [torch.randn(shape, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    g = -torch.rand(shape[:-1], device='cuda')
    beta = torch.rand(shape[:-1], device='cuda', dtype=torch.bfloat16)
    options = dict(q=q, k=k, v=v, g=g, beta=beta, output_final_state=True,
                   use_qk_l2norm_in_kernel=True)
    chunk, chunk_state = chunk_gated_delta_rule(**options)
    recurrent, recurrent_state = fused_recurrent_gated_delta_rule(**options)
    torch.testing.assert_close(chunk, recurrent, atol=0.03, rtol=0.03)
    torch.testing.assert_close(chunk_state, recurrent_state, atol=0.03, rtol=0.03)
    print('CUDA convolution + chunk/recurrent GDN checks PASS', flush=True)

    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    names = ('causal_conv1d_fn', 'causal_conv1d_update',
             'torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule')
    counts = dict.fromkeys(names, 0)
    for name in names:
        original = getattr(modeling, name)
        implementation = external_implementation(original)
        print(name, '->', implementation.__module__, flush=True)
        assert implementation.__module__.startswith(('fla.', 'causal_conv1d.'))

        def counted(*a, _fn=original, _name=name, **kw):
            counts[_name] += 1
            return _fn(*a, **kw)

        setattr(modeling, name, counted)

    from transformers import AutoTokenizer
    from worker import load_model
    model = load_model(args.model, 'cuda:0')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Compute 17 * 23.'}],
                                          tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda:0')
    with torch.inference_mode():
        for trial in range(2):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = model.generate(**inputs, min_new_tokens=32, max_new_tokens=32,
                                    do_sample=False, use_cache=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            tokens = output.shape[1] - inputs.input_ids.shape[1]
            print(f'B inference {trial}: {tokens} tokens, {elapsed:.2f}s, '
                  f'{tokens / elapsed:.2f} tokens/s', flush=True)
    assert all(counts.values()), counts
    print('Real B optimized kernel calls:', counts, flush=True)
    print('ACCELERATION CHECK PASS', flush=True)


if __name__ == '__main__':
    main()
