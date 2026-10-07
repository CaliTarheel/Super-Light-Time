"""Independent expanded-slab force and work oracles for the local force reader."""
from copy import deepcopy
import math
import unittest

import numpy as np

import plate_balance as pb
import plate_limit_analysis as limit
import slab_tether_forces as forces
from tests.test_slab_tether_forces import fixture
from tests.test_moving_hinge_tether import moving_fixture


def opposed_fixture():
    """One incoming plate attached to equal opposing slabs under two uppers.

    This is deliberately a force-only fixture, not an initialized world. Its
    two current local trace directions give exactly opposite incoming Euler
    torque rows; no speed or asymmetric driver is prescribed.
    """
    state = moving_fixture(edges=(1000., 1000.))
    state.xyz = np.tile(np.vstack((np.eye(3), -np.eye(3))), (3, 1))
    state.cell_area = np.full(18, 500.)
    state.plate = np.repeat(np.arange(3), 6)
    state.crust = np.zeros(18, int)
    state.age = np.full(18, 150.)
    state.active = np.ones(3, bool)
    state.plate_uid = np.array([11, 22, 33])
    state.omega = np.zeros((3, 3))
    state.bp, state.bq = np.array([0, 0]), np.array([1, 2])
    state.ba, state.bb = np.array([0, 1]), np.array([6, 12])
    angle = .3
    state.bmid = np.array([[math.cos(angle), math.sin(angle), 0.],
                          [math.cos(angle), -math.sin(angle), 0.]])
    state.bn = np.array([[-math.sin(angle), math.cos(angle), 0.],
                        [-math.sin(angle), -math.cos(angle), 0.]])
    second = deepcopy(state.trench_systems[0])
    second['id'], second['overriding_plate_uid'] = 2, 33
    state.trench_systems.append(second)
    state.trench_id = np.array([1, 2])
    return state


def expanded_oracle(channel, x):
    """Differentiate the uncondensed physical dissipation at its minimizing u.

    Original: E = Cs/2*((h+c*u)^2+(s*u)^2) + Cn/2*(u-q)^2 - W*u.
    Horizontal: replace the neck slip by (c*u-q); q remains horizontal intake.
    The physical plate force is minus E's derivative at fixed independent u.
    """
    qrows, hrows = channel['inlet_rows'], channel['hinge_rows']
    q, h = qrows@x*pb.CM_YR_M_S, hrows@x*pb.CM_YR_M_S
    cs, cn, weight = (channel[k] for k in
                     ('mantle_drag_n_s_m', 'neck_drag_n_s_m', 'weight_n'))
    c, s = channel['dip_cosine'], channel['dip_sine']
    neck_projection = c if channel.get('horizontal_intake', False) else 1.
    u = (weight+cn*neck_projection*q-cs*c*h)/(cs+cn*neck_projection**2)
    slip = neck_projection*u-q
    net = channel['edge_shares'][:, None]*pb.CM_YR_M_S*(
        (cn*slip)[:, None]*qrows-(cs*(h+c*u))[:, None]*hrows)
    physical_work = channel['edge_shares']@(weight*u
        -cs*((h+c*u)**2+(s*u)**2)-cn*slip**2)
    return net, float(physical_work)


class SlabForceLedgerTests(unittest.TestCase):
    def assemblies(self):
        for label, state in (('fixed', fixture()), ('moving', moving_fixture())):
            state.bmid[1] = [0., 1., 0.]
            state.bn[1] = [0., 0., 1.]
            model = pb.Balance(state, 1., slab_tethers=True)
            yield label, model.tether_assembly
            if label == 'moving':
                yield 'horizontal', forces.prepare_horizontal(model)

    def test_arbitrary_motion_matches_expanded_force_condensed_operator_and_power(self):
        rng = np.random.default_rng(20261001)
        for label, assembly in self.assemblies():
            with self.subTest(law=label):
                size = len(assembly['drive'])
                for x in rng.normal(size=(30, size))*12.:
                    ledger = forces.edge_force_ledger(assembly, x)
                    net = ledger['net_generalized_forces']
                    oracle, power = [], 0.
                    for channel in assembly['channels']:
                        local, work = expanded_oracle(channel, x)
                        oracle.append(local)
                        power += work
                    np.testing.assert_allclose(net, np.vstack(oracle), rtol=2e-12, atol=1e-5)
                    np.testing.assert_allclose(net.sum(axis=0),
                        assembly['drive']-assembly['stiffness']@x, rtol=2e-12, atol=1e-5)
                    report = forces.audit(assembly, x)
                    scale = max(abs(power), abs(report['weight_power_w']), 1.)
                    self.assertLess(abs(float(net.sum(axis=0)@x)-power)/scale, 2e-12)
                    self.assertLess(abs(float(net.sum(axis=0)@x)-report['plate_power_w'])/scale, 2e-12)

    def test_edge_subdivision_and_storage_reversal_preserve_plate_force(self):
        x = np.array([2., -3., 7., -5., 9., -4.])
        original = pb.Balance(moving_fixture(edges=(1000.,)), 1., slab_tethers=True)
        wanted = forces.edge_force_ledger(original.tether_assembly, x)['net_generalized_forces'].sum(axis=0)
        for count in (2, 10):
            state = moving_fixture(edges=(1000./count,)*count)
            state.bp[::2], state.bq[::2] = state.bq[::2].copy(), state.bp[::2].copy()
            state.ba[::2], state.bb[::2] = state.bb[::2].copy(), state.ba[::2].copy()
            state.bn[::2] *= -1.
            model = pb.Balance(state, 1., slab_tethers=True)
            ledger = forces.edge_force_ledger(model.tether_assembly, x)
            np.testing.assert_allclose(ledger['net_generalized_forces'].sum(axis=0), wanted,
                                       rtol=2e-12, atol=1e-5)
            np.testing.assert_array_equal(ledger['edges'], np.arange(count))

    def test_reader_is_nonmutating_and_rejects_invalid_motion(self):
        model = pb.Balance(moving_fixture(), 1., slab_tethers=True)
        assembly = model.tether_assembly
        before = deepcopy(assembly)
        x = np.arange(model.size, dtype=float)
        ledger = forces.edge_force_ledger(assembly, x)
        np.testing.assert_array_equal(x, np.arange(model.size, dtype=float))
        for key in ('drive', 'stiffness', 'down_drive', 'over_drive'):
            np.testing.assert_array_equal(assembly[key], before[key])
        for got, old in zip(assembly['channels'], before['channels']):
            for key in ('edges', 'edge_shares', 'inlet_rows', 'hinge_rows'):
                np.testing.assert_array_equal(got[key], old[key])
        ledger['edges'][:] = -1
        np.testing.assert_array_equal(assembly['channels'][0]['edges'], before['channels'][0]['edges'])
        for invalid in (np.zeros(model.size+1), np.zeros((1, model.size)),
                        np.full(model.size, np.nan), np.full(model.size, np.inf)):
            with self.subTest(shape=np.shape(invalid)), self.assertRaises(ValueError):
                forces.edge_force_ledger(assembly, invalid)

    def test_empty_attachment_has_no_local_force(self):
        state = moving_fixture()
        state.trench_id[:] = 0
        model = pb.Balance(state, 1., slab_tethers=True)
        ledger = forces.edge_force_ledger(model.tether_assembly, np.zeros(model.size))
        self.assertEqual(ledger['net_generalized_forces'].shape, (0, model.size))
        self.assertEqual(ledger['edges'].shape, (0,))

    def test_opposed_slabs_have_zero_rigid_drive_but_finite_separating_work(self):
        state = opposed_fixture()
        model = pb.Balance(state, 1., slab_tethers=True)
        model.solve()
        assembly = model.tether_assembly
        # The incoming plate's own motion is solved, not imposed. Its slab
        # drives cancel while the two overriding plates may retreat normally.
        np.testing.assert_array_equal(assembly['down_drive'][:3], 0.)
        self.assertLess(np.linalg.norm(model.x[:3]), 1e-10)
        ledger = forces.edge_force_ledger(assembly, model.x)
        local = ledger['net_generalized_forces'][:, :3]
        gross = np.linalg.norm(local, axis=1).sum()
        self.assertGreater(gross, 0.)
        self.assertLess(np.linalg.norm(local.sum(axis=0))/gross, 1e-12)

        # A minimal spherical strip places each exact local incoming moment at
        # its own rim. The one internal connection crosses the strip at x=1.
        # Opposite differential rotations separate its two cells. With a unit
        # z-axis rotation the known yield work is L*R*S, so P/D = traction/(L*S).
        points = state.bmid[::-1].copy()
        torques = local[::-1]*pb.RADIUS_M/pb.CM_YR_M_S
        separating = np.array([[0., 0., -.5], [0., 0., .5]])
        work = float(np.sum(torques*separating))
        self.assertGreater(work, 0.)
        length_m = 1e6
        line_force = work/(length_m*pb.RADIUS_M)
        self.assertGreater(line_force, 0.)
        for strength_factor, expected_ratio in ((2., .5), (.5, 2.)):
            with self.subTest(strength_factor=strength_factor):
                strength = np.array([strength_factor*line_force])
                found = limit.analyse(points, torques, np.array([[0, 1]]),
                    np.array([[1., 0., 0.]]), np.array([length_m]),
                    (strength, 3.*strength, strength/math.sqrt(3.)),
                    axes=np.array([[0., 0., 1.]]))
                self.assertAlmostEqual(found['ratio'], expected_ratio, places=12)
                np.testing.assert_array_equal(found['cut'], [True])
                self.assertEqual(found['ratio'] >= 1., strength_factor < 1.)
        # This is an onset/virtual-work regression. It does not prescribe a
        # finite opening speed, seed a ridge, or commit a topology change.


if __name__ == '__main__':
    unittest.main()
