import unittest

import numpy as np
import torch

from cx_tta.geometry_readout import GeometryReadout
from cx_tta.structured_action_adapter import (
    StructuredActionAdapter, action_direction_loss,
)


def observations(n=2, channels=4, dtype=torch.float32):
    return dict(
        visual=torch.randn(n, 3, 2, 2, channels, dtype=dtype),
        text=torch.randn(n, 2, channels, dtype=dtype),
        state=torch.randn(n, 14, dtype=dtype),
        past_visual=torch.randn(n, 3, 2, 2, channels, dtype=dtype),
        past_text=torch.randn(n, 2, channels, dtype=dtype),
        past_state=torch.randn(n, 14, dtype=dtype),
        has_history=torch.ones(n, 1, dtype=dtype),
        ee_pose=torch.randn(n, 2, 9, dtype=dtype),
        past_ee_pose=torch.randn(n, 2, 9, dtype=dtype),
    )


class StructuredActionContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def test_initial_zero_residual_and_exact_bypass_after_nonzero_head(self):
        model = StructuredActionAdapter(4, np.zeros(24))
        x = observations()
        base = torch.randn(2, 32, 14)
        output = model(x, base)
        self.assertTrue(torch.equal(output['delta'], torch.zeros_like(base)))
        self.assertTrue(torch.equal(output['actions'], base))
        with torch.no_grad():
            model.action_head[-1].bias.copy_(torch.linspace(-10, 10, 14))
        corrected = model(x, base)
        self.assertTrue(torch.all(corrected['delta'].abs() <= model.caps))
        self.assertTrue(torch.any(corrected['delta'] != 0))
        bypass = model(x, base, enabled=False)
        self.assertIs(bypass['actions'], base)
        self.assertTrue(torch.equal(bypass['delta'], torch.zeros_like(base)))
        self.assertTrue(torch.equal(corrected['geometry'], bypass['geometry']))

    def test_gt_unknown_fields_bad_shapes_and_nonfinite_input_rejected(self):
        model = StructuredActionAdapter(4, np.zeros(24))
        x = observations()
        base = torch.zeros(2, 32, 14)
        for key in ('contact_world', 'task', 'asset_id', 'regression_valid'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                model(dict(x, **{key: torch.ones(2)}), base)
        with self.assertRaises(ValueError):
            model(x, base[:, :31])
        bad = dict(x, state=x['state'].clone())
        bad['state'][0, 0] = float('nan')
        with self.assertRaises(ValueError):
            model(bad, base)

    def test_geometry_matches_original_readout_with_same_shared_weights(self):
        mean = np.linspace(-.5, .5, 24, dtype=np.float32)
        original = GeometryReadout(4, mean)
        with torch.no_grad():
            original.head.weight.normal_()
            original.head.bias.normal_()
        model = StructuredActionAdapter(4, mean)
        copied = model.state_dict()
        for key, value in original.state_dict().items():
            copied[key.replace('head.', 'geometry_head.') if key.startswith('head.') else key] = value
        model.load_state_dict(copied)
        x = observations()
        actual = model(x, torch.zeros(2, 32, 14))
        expected = original(x)
        for key in ('current_attention', 'past_attention'):
            self.assertTrue(torch.equal(actual[key], expected[key]))
        self.assertTrue(torch.equal(actual['geometry'], expected['regression']))

    def test_dtype_preserved_and_arm_normalization_only_in_action_features(self):
        model = StructuredActionAdapter(4, np.zeros(24)).double()
        x = observations(dtype=torch.float64)
        base = torch.full((2, 32, 14), torch.pi, dtype=torch.float64)
        base[..., (6, 13)] = .7
        saved = []
        handle = model.action_head[0].register_forward_pre_hook(lambda _, args: saved.append(args[0].detach().clone()))
        output = model(x, base)
        handle.remove()
        self.assertEqual(output['actions'].dtype, torch.float64)
        context = saved[0][..., 64:78]
        self.assertTrue(torch.equal(context[..., (6, 13)], base[..., (6, 13)]))
        self.assertTrue(torch.allclose(context[..., (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)], torch.ones(2, 32, 12, dtype=torch.float64)))
        self.assertTrue(torch.equal(base[..., (6, 13)], torch.full((2, 32, 2), .7, dtype=torch.float64)))

    def test_direction_loss_true_valid_length_zero_weights_and_clipping(self):
        model = StructuredActionAdapter(4, np.zeros(24))
        base = torch.zeros(3, 32, 14)
        delta = torch.zeros_like(base, requires_grad=True)
        target = torch.full_like(base, float('nan'))
        valid = torch.zeros(3, 32, dtype=torch.bool)
        valid[0, :2] = True
        valid[1] = True
        target[0, 0] = model.caps * 2
        target[0, 1] = model.caps * .5
        weights = torch.tensor([.3, 0., .7])
        loss = action_direction_loss(delta, base, target, valid, weights, model.caps)
        self.assertAlmostEqual(float(loss), .3 * (.5 + .125) / 2, places=7)
        loss.backward()
        self.assertTrue(torch.all(delta.grad[0, :2] != 0))
        self.assertTrue(torch.equal(delta.grad[0, 2:], torch.zeros_like(delta.grad[0, 2:])))
        self.assertTrue(torch.equal(delta.grad[1:], torch.zeros_like(delta.grad[1:])))
        changed = target.clone()
        changed[1:] = 1e9
        changed[0, 2:] = -1e9
        self.assertEqual(float(action_direction_loss(delta, base, changed, valid, weights, model.caps)), float(loss))


if __name__ == '__main__':
    unittest.main()
