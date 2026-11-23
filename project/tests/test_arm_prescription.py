import unittest
import numpy as np
from cx_tta.arm_prescription_policy import compose, CONDITIONS
from cx_tta.causal_arm_gate import OneShotClosingGate


class PrescriptionContract(unittest.TestCase):
    def setUp(self):
        self.base = np.zeros((32, 14), np.float32)
        self.base[:, 6] = -.3  # Original gripper command must not be projected.
        self.base[:, 13] = 1.2
        self.base[:, 0] = -0.
        self.raw = np.full_like(self.base, .07)
        self.raw[:, 0] = 0.

    def test_null_gripper_and_signed_zero_identity(self):
        for c in CONDITIONS:
            a, d, _ = compose(self.base, self.raw, c, 1, [1, 3])
            off = d == 0
            self.assertEqual(a[off].tobytes(), self.base[off].tobytes())
            self.assertEqual(a[:, [6, 13]].tobytes(), self.base[:, [6, 13]].tobytes())
        self.assertEqual(compose(self.base, self.raw, 'Z_zero', 1, [1, 3])[0].tobytes(), self.base.tobytes())

    def test_shared_first_block_then_stop_or_continue(self):
        for q in (0, 1):
            x = compose(self.base, self.raw, 'F_first_full', q, [1, 3])
            y = compose(self.base, self.raw, 'B_two_full', q, [1, 3])
            self.assertEqual(x[0].tobytes(), y[0].tobytes())
        self.assertFalse(compose(self.base, self.raw, 'F_first_full', 2, [1, 3])[2]['active'])
        self.assertTrue(compose(self.base, self.raw, 'B_two_full', 2, [1, 3])[2]['active'])

    def test_half_scale_finite_horizon_and_no_trigger(self):
        for c in CONDITIONS:
            for q in (0, 3, 9):
                self.assertEqual(compose(self.base, self.raw, c, q, [1, 3])[0].tobytes(), self.base.tobytes())
            self.assertEqual(compose(self.base, self.raw, c, 2, None)[0].tobytes(), self.base.tobytes())
        full = compose(self.base, self.raw, 'B_two_full', 1, [1, 3])[1]
        half = compose(self.base, self.raw, 'H_two_half', 1, [1, 3])[1]
        np.testing.assert_array_equal(half, full * np.float32(.5))

    def test_future_independence_and_reset(self):
        state = np.ones(14, np.float32)
        g = OneShotClosingGate()
        first = g.step(0, state, self.base)
        self.assertEqual(first['window'], [1, 3])
        for q in range(1, 5):
            self.assertEqual(g.step(q, state, np.ones_like(self.base))['window'], first['window'])
        g.reset()
        self.assertIsNone(g.step(0, state, np.ones_like(self.base))['window'])

    def test_invalid_candidate_and_window_rejected(self):
        bad = self.raw.copy(); bad[0, 1] = .11
        with self.assertRaises(ValueError): compose(self.base, bad, 'H_two_half', 1, [1, 3])
        with self.assertRaises(ValueError): compose(self.base, self.raw, 'F_first_full', 1, [1, 4])
        with self.assertRaises(ValueError): compose(self.base.astype(np.float64), self.raw, 'Z_zero', 1, [1, 3])


if __name__ == '__main__':
    unittest.main()
