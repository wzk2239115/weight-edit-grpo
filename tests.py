import unittest
from edits import parse_action, rectangle, reward


class EditTests(unittest.TestCase):
    def test_only_numeric_bounded_rectangles(self):
        action = dict(op='edit', row=1, col=2, values=[[.1, -.2], [.3, .4]])
        self.assertEqual(rectangle(action, (4, 5)), (slice(1, 3), slice(2, 4)))
        for values in ([[float('nan')]], [[1], [2, 3]], [['0.1']], [[True]]):
            with self.assertRaises(ValueError):
                rectangle(dict(action, values=values), (4, 5))
        with self.assertRaises(ValueError):
            rectangle(dict(action, row=4), (4, 5))

    def test_gate_cannot_be_bought_by_target_gain(self):
        base = dict(gsm8k=.2, math=.1, mmlu=.8, ifeval=.8)
        edited = dict(gsm8k=1., math=1., mmlu=.7, ifeval=.8)
        self.assertLess(reward(base, edited, .02)['reward'], 0.)
        self.assertEqual(reward(base, base, .02)['reward'], 0.)

    def test_json_does_not_execute_code(self):
        with self.assertRaises(ValueError):
            parse_action('__import__("os").system("true")')

    def test_bf16_rounding_and_restore_after_failure(self):
        import torch
        from worker import temporary_edit
        tensor = torch.ones(4, 4, dtype=torch.bfloat16)
        action = dict(op='edit', row=1, col=1, values=[[1.00001]])
        with self.assertRaises(ValueError):
            with temporary_edit(tensor, action):
                pass
        action['values'] = [[.5]]
        with self.assertRaises(RuntimeError):
            with temporary_edit(tensor, action) as changed:
                self.assertEqual(changed, 1)
                self.assertEqual(tensor[1, 1], .5)
                raise RuntimeError('Simulated evaluation failure')
        self.assertTrue(torch.equal(tensor, torch.ones_like(tensor)))

    def test_grpo_changes_editor_and_equal_rewards_skip(self):
        import torch
        from types import SimpleNamespace
        from transformers import GPT2Config, GPT2LMHeadModel
        from train import token_logps, update
        torch.manual_seed(7)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=32, n_embd=16,
                                         n_layer=1, n_head=2, bos_token_id=1, eos_token_id=2)).eval()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        args = SimpleNamespace(update_epochs=2, clip=.2)
        trajectories = []
        for sequence in ([4, 5, 6], [7, 8, 9]):
            prefix, completion = torch.tensor([[1, 2, 3]]), torch.tensor([sequence])
            with torch.no_grad():
                old = token_logps(model, prefix, completion)
            trajectories.append([(prefix, completion, old)])
        before = model.transformer.wte.weight.detach().clone()
        self.assertFalse(update(model, optimizer, trajectories, [0., 0.], args)['updated'])
        self.assertTrue(torch.equal(before, model.transformer.wte.weight))
        result = update(model, optimizer, trajectories, [1., -1.], args)
        self.assertTrue(result['updated'])
        self.assertFalse(torch.equal(before, model.transformer.wte.weight))


if __name__ == '__main__':
    unittest.main()
