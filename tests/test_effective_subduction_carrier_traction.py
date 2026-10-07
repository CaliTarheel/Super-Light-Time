"""Independent finite-port force/work and future-only policy regressions."""
from copy import deepcopy
from pathlib import Path
import json
import math
import pickle
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import balance_force_ledger as ledger
import checkpoint
import effective_subduction as effective
import effective_subduction_carrier_traction as carrier
import native_engine
import plate_balance as pb
from tests.test_effective_subduction_forces import fixture, uniform_basis

FORCE = 3e13


def world(edges=(1000.,), plate_count=2):
    state = fixture(edges, plate_count)
    state.t, state.steps = 25., 100
    state.config['effective_subduction']['force_n_per_m'] = FORCE
    state.effective_subduction_settings['force_n_per_m'] = FORCE
    state.effective_subduction_diagnostics = {}
    for row in state.trench_systems:
        row['effective_subduction']['force_n_per_m'] = FORCE
    state.rng = np.random.default_rng(34)
    return state


def configure(state, alpha=.1):
    carrier.upgrade(state, dict(enabled=True, alpha=alpha))
    return state


def opposed_carrier():
    state = world((1000., 1000.), 3)
    angle = .3
    state.bmid = np.array([[math.cos(angle), math.sin(angle), 0.],
                          [math.cos(angle), -math.sin(angle), 0.]])
    state.bn = np.array([[-math.sin(angle), math.cos(angle), 0.],
                        [-math.sin(angle), -math.cos(angle), 0.]])
    state.bp[:], state.bq[:] = [1, 2], 0
    state.trench_id[:] = [1, 2]
    row = state.trench_systems[0]
    row.update(downgoing_plate_uid=22, overriding_plate_uid=11)
    state.trench_systems.append(dict(deepcopy(row), id=2, downgoing_plate_uid=33))
    points = []
    for center in state.bmid:
        tangent = np.cross([0., 0., 1.], center)
        for d in (-.006, 0., .006):
            for z in (-.004, .004):
                value = center+d*tangent+[0., 0., z]
                points.append(value/np.linalg.norm(value))
    state.xyz = np.vstack((points, np.eye(3), -np.eye(3), np.eye(3), -np.eye(3)))
    state.plate = np.r_[np.zeros(12, int), np.ones(6, int), np.full(6, 2, int)]
    state.cell_area = np.full(24, 1e6)
    state.crust = np.zeros(24, int)
    state.age = np.full(24, 100.)
    state.ba, state.bb = np.array([12, 18]), np.array([0, 6])
    return configure(state)


class CarrierTractionTests(unittest.TestCase):
    def test_normalization_is_explicit_and_disabled_operator_is_unchanged(self):
        state = world()
        before = pb.Balance(state, 1.)
        original = deepcopy(state.config)
        self.assertNotIn('carrier_traction', effective.normalize(state.config['effective_subduction']))
        self.assertEqual(carrier.snapshot(state), {})
        self.assertEqual(state.config, original)
        state.config['effective_subduction']['carrier_traction'] = carrier.normalize()
        after = pb.Balance(state, 1.)
        np.testing.assert_array_equal(after.torque, before.torque)
        np.testing.assert_array_equal(after.stiffness, before.stiffness)
        self.assertEqual(carrier.snapshot(state), {})
        for controls in ({'version': True}, {'version': 2}, {'enabled': 1}, {'alpha': True},
                         {'alpha': -.1}, {'alpha': 1.1}, {'alpha': float('nan')},
                         {'alpha': float('inf')}, {'unknown': 1}):
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                carrier.normalize(controls)
        state.config['effective_subduction']['carrier_traction']['enabled'] = True
        with self.assertRaisesRegex(ValueError, 'activation metadata'):
            pb.Balance(state, 1.)

    def test_upgrade_preserves_all_inherited_state_and_creates_zero_future_work(self):
        state = world()
        before = {k: pickle.dumps(v, protocol=5) for k, v in vars(state).items() if k != 'config'}
        config = deepcopy(state.config)
        result = carrier.upgrade(state)
        self.assertEqual(set(vars(state)), set(before)|{'config', carrier.FIELD})
        self.assertEqual({k: pickle.dumps(getattr(state, k), protocol=5) for k in before}, before)
        nested = deepcopy(state.config)
        nested['effective_subduction'].pop('carrier_traction')
        self.assertEqual(nested, config)
        self.assertEqual(result['activation_myr'], 25.)
        self.assertEqual(result['retrospective_work_j'], 0.)
        self.assertEqual(result['parameters']['alpha'], .1)
        first = pickle.dumps(vars(state), protocol=5)
        self.assertEqual(carrier.upgrade(state), result)
        self.assertEqual(pickle.dumps(vars(state), protocol=5), first)
        with self.assertRaisesRegex(ValueError, 'reparameterized'):
            carrier.upgrade(state, dict(enabled=True, alpha=.2))
        self.assertEqual(pickle.dumps(vars(state), protocol=5), first)

    def test_actual_balance_matches_independent_torque_source_power_and_derivative(self):
        state = configure(world())
        model = pb.Balance(state, 1.)
        baseline = world()
        old = pb.Balance(baseline, 1.)
        scale = pb.CM_YR_M_S*FORCE*1e6
        expected = np.array([0., 0., scale, 0., 0., -.1*scale])
        np.testing.assert_allclose(model.drivers['slab'], expected, rtol=2e-15)
        np.testing.assert_array_equal(model.slab_down_drive, old.slab_down_drive)
        np.testing.assert_array_equal(model.stiffness, old.stiffness)
        np.testing.assert_array_equal(model.hinge_coefficient, old.hinge_coefficient)
        np.testing.assert_array_equal(model.hinge, old.hinge)
        delta = pb.HUBER_CONTINUATION_KM_MYR[-1]*pb.KM_MYR_CM_YR
        for x in (np.array([2., -3., 5., 7., 4., -6.]), np.tile([1., -2., 3.], 2)):
            independent_power = scale*(x[2]-.1*x[5])
            self.assertAlmostEqual(float(model.drivers['slab']@x)/independent_power, 1., places=14)
            # The source difference is independently linear; passive Hessian is identical.
            found = model._evaluate(x, delta)
            prior = old._evaluate(x, delta)
            np.testing.assert_allclose(found[1]-prior[1], -(expected-old.drivers['slab']), rtol=1e-12, atol=.001)
            np.testing.assert_array_equal(found[2], prior[2])
            direction = np.array([1., .5, -.4, -.3, -.2, .7])
            epsilon = 1e-4
            difference = lambda y: model._evaluate(y, delta)[0]-old._evaluate(y, delta)[0]
            derivative = (difference(x+epsilon*direction)-difference(x-epsilon*direction))/(2*epsilon)
            oracle = -float((expected-old.drivers['slab'])@direction)
            self.assertLess(abs(derivative-oracle)/abs(oracle), 2e-10)
        model.solve()
        report = model.diagnostics(model.rotation())
        power = report['effective_subduction_carrier_power']
        self.assertAlmostEqual(power['represented_source_power_w']/float(expected@model.x), 1., places=13)
        self.assertEqual(power['reservoir_exchange_power_w'], -power['represented_source_power_w'])
        self.assertLess(abs(power['power_partition_residual_w'])/abs(power['represented_source_power_w']), 1e-14)
        self.assertEqual(report['negative_resisting_work_elements'], 0)
        self.assertLessEqual(report['scaled_force_residual'], pb.FORCE_RELATIVE_TOLERANCE)
        self.assertFalse(power['passive_operators_changed'])
        self.assertFalse(power['accumulated_work_reported'])

    def test_subdivision_reversal_and_zero_fraction_preserve_the_selected_law(self):
        base = pb.Balance(configure(world()), 1.)
        for count in (2, 10):
            state = configure(world((1000./count,)*count))
            state.bp[::2], state.bq[::2] = state.bq[::2].copy(), state.bp[::2].copy()
            state.ba[::2], state.bb[::2] = state.bb[::2].copy(), state.ba[::2].copy()
            state.bn[::2] *= -1.
            model = pb.Balance(state, 1.)
            np.testing.assert_allclose(model.torque, base.torque, rtol=2e-14)
            np.testing.assert_allclose(model.stiffness, base.stiffness, rtol=2e-14)
        zero = pb.Balance(configure(world(), 0.), 1.)
        old = pb.Balance(world(), 1.)
        np.testing.assert_array_equal(zero.torque, old.torque)
        np.testing.assert_array_equal(zero.stiffness, old.stiffness)

    def test_force_ledger_keeps_complete_carrier_and_prospective_daughter_work(self):
        state = opposed_carrier()
        model = pb.Balance(state, 1.)
        rng = np.random.default_rng(48)
        for x in rng.normal(size=(3, model.size))*5.:
            result = ledger.export(model, x, band_km=100.)
            expected = model._evaluate(x, result['delta'])
            compiled = ledger.compile_mode(result, uniform_basis(model))
            for actual, wanted in zip(compiled.evaluate(x), expected[:3]):
                np.testing.assert_allclose(actual, wanted, rtol=1e-11, atol=.1)
            np.testing.assert_allclose(result['generalized_force'], -expected[1], rtol=1e-11, atol=.1)
        result = ledger.export(model, np.zeros(model.size), band_km=100.)
        local = result['plates'][0]['localcomponents_n_m']['effective_subduction']
        self.assertLess(np.linalg.norm(local.sum(axis=0))/np.linalg.norm(local), 1e-12)
        self.assertGreater(np.linalg.norm(local), 0.)
        basis = uniform_basis(model, 1)
        basis[0][:6, 2, -1] = -.5
        basis[0][6:12, 2, -1] = .5
        compiled = ledger.compile_mode(result, basis)
        cells = result['plates'][0]['cells']
        independent = float(np.sum(result['plates'][0]['cell_generalized_force']*basis[0][cells, :, -1]))
        self.assertGreater(independent, 0.)
        self.assertAlmostEqual(-compiled.evaluate(np.zeros(model.size+1))[1][-1]/independent, 1., places=12)
        self.assertFalse(result['diagnostics']['equilibrium_reaction_added'])
        state.config['effective_subduction']['carrier_traction']['alpha'] = .2
        getattr(state, carrier.FIELD)['parameters']['alpha'] = .2
        with self.assertRaisesRegex(ledger.UnsupportedForceLedger, 'frozen balance'):
            ledger.export(model, np.zeros(model.size))

    def test_dry_unmatched_pending_and_untrained_ports_do_not_load_carrier(self):
        for mode in ('dry', 'unmatched', 'pending', 'untrained'):
            state = configure(world())
            if mode == 'dry': state.crust[0] = 1
            elif mode == 'unmatched': state.trench_id[:] = 99
            elif mode == 'pending': state.trench_systems[0]['effective_subduction']['activation_pending'] = True
            else: state.trench_systems[0]['effective_subduction']['initiation_finite_support'] = True
            with self.subTest(mode=mode), patch('effective_subduction_initiation.supported_parents',
                                              return_value=np.array([False])):
                model = pb.Balance(state, 1.)
                np.testing.assert_array_equal(model.slab_down_drive, 0.)
                np.testing.assert_array_equal(model.slab_over_drive, 0.)
                np.testing.assert_array_equal(model.effective_carrier_owner, -1)
        state = configure(world((400., 600.)))
        with patch('native_subduction.enabled', return_value=True), patch(
                'native_subduction.edge_ocean_fraction', return_value=np.array([0., .25])):
            model = pb.Balance(state, 1.)
        scale = FORCE*600.e3*.25*pb.CM_YR_M_S
        self.assertAlmostEqual(model.slab_down_drive[2]/scale, 1., places=14)
        self.assertAlmostEqual(model.slab_over_drive[5]/(-.1*scale), 1., places=14)
        np.testing.assert_array_equal(model.trench_edges, [1])

    def test_owner_transfer_uses_real_carrier_and_corruption_fails_closed(self):
        state = configure(world(plate_count=3))
        state.trench_systems[0]['overriding_plate_uid'] = 33
        state.bq[:] = 2
        state.bb[:] = 12
        model = pb.Balance(state, 1.)
        np.testing.assert_array_equal(model.effective_carrier_owner, 2)
        np.testing.assert_array_equal(model.slab_over_drive[3:6], 0.)
        self.assertNotEqual(model.slab_over_drive[8], 0.)
        for kind in ('uid', 'epoch', 'work', 'normal', 'configuration'):
            bad = deepcopy(state)
            if kind == 'uid': bad.plate_uid[2] = bad.plate_uid[1]
            elif kind == 'epoch': getattr(bad, carrier.FIELD)['activation_myr'] = 26.
            elif kind == 'work': getattr(bad, carrier.FIELD)['retrospective_work_j'] = 1.
            elif kind == 'normal': bad.bn[0] = bad.bmid[0]
            else: bad.config['effective_subduction']['carrier_traction']['alpha'] = .2
            before = pickle.dumps(vars(bad), protocol=5)
            with self.subTest(kind=kind), self.assertRaises(ValueError): pb.Balance(bad, 1.)
            self.assertEqual(pickle.dumps(vars(bad), protocol=5), before)

    def test_observed_arc_host_motion_uses_only_exact_admitted_finite_support(self):
        state = configure(world((100., 900.), 3))
        state.trench_systems[0]['overriding_plate_uid'] = 33
        state.bq[:], state.bb[:] = 2, 12
        state.trench_id[1] = 999  # nearby halo must not enter the observation
        state.backarc_basins = [dict(id=4, trench_id=1, phase='rifting',
            parent_plate_uid=22, arc_plate_uid=33, downgoing_plate_uid=11)]
        angle = 100./6371.
        a = np.array([math.cos(-angle/2), 0., math.sin(-angle/2)])
        b = np.array([math.cos(angle/2), 0., math.sin(angle/2)])
        state.native_boundary_geometry = dict(segments_start=np.array([a, a]), segments_end=np.array([b, b]),
            segment_normals=np.tile([0., 1., 0.], (2, 1)), contact_index=np.array([0, 1]))
        state.omega[1] = [0., 0., 2./6371.]
        state.omega[2] = [0., 0., -8./6371.]
        before = pickle.dumps(vars(state), protocol=5)
        result = carrier.snapshot(state)
        row = result[f'{carrier.FIELD}_diagnostics']['measured_arc_host_motion'][0]
        self.assertEqual(row['finite_piece_count'], 1)
        self.assertAlmostEqual(row['supported_length_km'], 100., places=10)
        self.assertAlmostEqual(row['mean_retreat_relative_host_km_myr'], 10., places=12)
        self.assertFalse(row['source_or_loading_added'])
        self.assertFalse(result[f'{carrier.FIELD}_diagnostics']['independent_within_host_hinge_motion'])
        self.assertEqual(pickle.dumps(vars(state), protocol=5), before)
        frame = dict(result, effective_subduction_version=1, time_myr=state.t, config=state.config)
        self.assertTrue(effective.validate_frame(json.loads(json.dumps(frame))))
        for key, value in (('activation_myr', 26.), ('retrospective_work_j', 1.),
                           ('independent_within_host_hinge_motion', True)):
            bad = deepcopy(frame)
            bad[f'{carrier.FIELD}_diagnostics'][key] = value
            with self.assertRaises(ValueError): effective.validate_frame(bad)
        incomplete = deepcopy(frame)
        incomplete.pop(f'{carrier.FIELD}_parameters')
        with self.assertRaises(ValueError): effective.validate_frame(incomplete)
        self.assertFalse(effective.validate_frame({'time_myr': 25.}))
        state.t = 25.0000001
        rounded = dict(carrier.snapshot(state), effective_subduction_version=1,
                       time_myr=round(state.t, 6), config=state.config)
        self.assertTrue(effective.validate_frame(rounded))

    def test_native_fresh_policy_snapshot_and_typed_checkpoint_keep_force_and_epoch(self):
        state = native_engine.Simulation(dict(width=48, height=24, mesh_level=2, coast_geometry_level=2,
            plate_count=4, mechanics_nodes=128, seed=37, physics_profile='reviewed_v1',
            primordial_subduction={'enabled': True}, effective_subduction=dict(enabled=True,
                force_n_per_m=FORCE, carrier_traction=dict(enabled=True, alpha=.1))))
        self.assertEqual(getattr(state, carrier.FIELD)['activation_myr'], 0.)
        snapshot = state.snapshot()
        self.assertTrue(effective.validate_frame(snapshot))
        before = pb.Balance(state, 1.)
        self.assertEqual(state.effective_subduction_settings['force_n_per_m'], FORCE)
        self.assertTrue(np.any(before.slab_over_drive))
        self.assertEqual(state.process_totals['ocean_consumed_km2'], 0.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.npz'
            checkpoint.write_checkpoint(path, state, {'config': state.config}, {})
            restored, _ = checkpoint.read_checkpoint(path, {}, type(state))
        self.assertEqual(getattr(restored, carrier.FIELD), getattr(state, carrier.FIELD))
        self.assertTrue(effective.validate_frame(restored.snapshot()))
        after = pb.Balance(restored, 1.)
        np.testing.assert_array_equal(after.torque, before.torque)
        np.testing.assert_array_equal(after.stiffness, before.stiffness)
        np.testing.assert_array_equal(restored.rng.bit_generator.state, state.rng.bit_generator.state)


if __name__ == '__main__':
    unittest.main()
