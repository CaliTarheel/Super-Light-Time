"""Independent force, chain, power and incompatible-geometry oracles."""
import unittest
import numpy as np

import constrained_plate_motion as constrained


class Quadratic:
    def __init__(self, drive):
        self.torque = np.asarray(drive, float)
        self.size = len(self.torque)
        self.stiffness = np.eye(self.size)

    def _evaluate(self, x, delta):
        self._force_reference = np.abs(x)+np.abs(self.torque)
        return .5*x@x-self.torque@x, x-self.torque, self.stiffness, {}, {}


class ConstrainedPlateMotionTests(unittest.TestCase):
    def test_retreating_trench_pulls_upstream_and_reacts_on_upper_plate(self):
        model = Quadratic([0., 2.])
        x, reaction, report = constrained.solve(model, [[1., -1.]], [.1])
        np.testing.assert_allclose(x, [1.05, .95], atol=1e-10)
        np.testing.assert_allclose(reaction, [1.05], atol=1e-10)
        self.assertAlmostEqual(report['reaction_source_power_w'], .105)
        self.assertAlmostEqual(x@x, model.torque@x+report['reaction_source_power_w'])

    def test_chain_enforces_both_intakes_without_locking_a_previous_rate(self):
        rows = np.array([[1., -1., 0.], [0., 1., -1.]])
        x, _, _ = constrained.solve(Quadratic([0., 0., 3.]), rows, [.1, .1])
        np.testing.assert_allclose(x, [1.1, 1., .9], atol=1e-10)
        faster, reactions, _ = constrained.solve(Quadratic([5., 1., 0.]), rows, [.1, .1])
        np.testing.assert_allclose(faster, [5., 1., 0.], atol=1e-10)
        np.testing.assert_array_equal(reactions, 0.)
        np.testing.assert_allclose(rows@faster, [4., 1.], atol=1e-10)

    def test_opposed_geometric_demands_reject_instead_of_dropping_a_trench(self):
        with self.assertRaises(constrained.IncompatibleIntake):
            constrained.solve(Quadratic([1., 0.]), [[1., -1.], [-1., 1.]], [.1, .1])

    def test_duplicate_constraints_do_not_add_force_or_change_motion(self):
        model = Quadratic([0., 2.])
        rows = np.tile([1., -1.], (10, 1))
        x, reaction, _ = constrained.solve(model, rows, np.full(10, .1))
        np.testing.assert_allclose(x, [1.05, .95], atol=1e-10)
        self.assertAlmostEqual(float(reaction.sum()), 1.05)

    def test_daughter_opening_can_resolve_a_rigid_geometric_conflict(self):
        # The third coordinate is fault opening; its work costs 0.2 per unit.
        model = Quadratic([0., 0., 0.])
        rows = [[1., -1., 1.], [-1., 1., 1.], [0., 0., 1.]]
        x, reaction, report = constrained.solve(model, rows, [.1, .1, 0.],
                                                linear_cost=[0., 0., .2])
        np.testing.assert_allclose(x, [0., 0., .1], atol=1e-10)
        self.assertAlmostEqual(x@x+.2*x[-1], report['reaction_source_power_w'])
        self.assertGreater(reaction[:2].sum(), 0.)


if __name__ == '__main__':
    unittest.main()
