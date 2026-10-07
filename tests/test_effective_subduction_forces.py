"""Independent torque, work and passive-resistance oracles for trench traction."""
from copy import deepcopy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import balance_force_ledger as ledger
import breakup_mode
import effective_subduction as effective
import plate_balance as pb
import plate_limit_analysis as limit
import slab_memory


FORCE_N_PER_M = 5e12


def fixture(edges=(1000.,), plate_count=2):
    """A finite declared trench with no slab, imposed velocity, or material GPE.

    Axis-aligned controls give each plate the independent basal metric 4*I.
    They are not a geographical world; native initialization is qualified by
    the separate experiment and lifecycle tests.
    """
    n = len(edges)
    points = np.tile(np.vstack((np.eye(3), -np.eye(3))), (plate_count, 1))
    rows = [dict(id=1, phase='mature', maturity=1.,
                 downgoing_plate_uid=11, overriding_plate_uid=22,
                 effective_subduction=dict(version=1, force_n_per_m=FORCE_N_PER_M))]
    return SimpleNamespace(
        config=dict(effective_subduction=dict(enabled=True, force_n_per_m=FORCE_N_PER_M)),
        effective_subduction_version=1,
        effective_subduction_settings=dict(enabled=True, force_n_per_m=FORCE_N_PER_M),
        slab_memory_version=0, subduction_response_version=0, plate_resistance_version=1,
        active=np.ones(plate_count, bool), plate_uid=np.arange(1, plate_count+1)*11,
        names=[f'Plate {p}' for p in range(plate_count)],
        xyz=points, cell_area=np.full(len(points), 1e6),
        plate=np.repeat(np.arange(plate_count), 6), crust=np.zeros(len(points), int),
        age=np.full(len(points), 100.), omega=np.zeros((plate_count, 3)),
        bp=np.zeros(n, int), bq=np.ones(n, int),
        ba=np.zeros(n, int), bb=np.full(n, 6, int),
        bl=np.asarray(edges, float), bmid=np.tile([1., 0., 0.], (n, 1)),
        bn=np.tile([0., 1., 0.], (n, 1)), trench_id=np.ones(n, int),
        normal_speed=np.zeros(n), bcode=np.zeros(n, np.uint8), down=np.full(n, -1, int),
        trench_systems=rows)


def uniform_basis(model, extra=0):
    result = {}
    for owner in model.plates:
        values = np.zeros((len(model.s.xyz), 3, model.size+extra))
        values[:, :, 3*model.slot[owner]:3*model.slot[owner]+3] = np.eye(3)
        result[owner] = values
    return result


def opposed_fixture():
    state = fixture(edges=(1000., 1000.), plate_count=3)
    angle = .3
    state.bmid = np.array([[math.cos(angle), math.sin(angle), 0.],
                          [math.cos(angle), -math.sin(angle), 0.]])
    state.bn = np.array([[-math.sin(angle), math.cos(angle), 0.],
                        [-math.sin(angle), -math.cos(angle), 0.]])
    state.bq[1] = 2
    state.bb[1] = 12
    state.trench_id[1] = 2
    second = deepcopy(state.trench_systems[0])
    second.update(id=2, overriding_plate_uid=33)
    state.trench_systems.append(second)
    return state


class EffectiveSubductionForceTests(unittest.TestCase):
    def test_single_trace_has_analytic_incoming_torque_and_retained_basal_drag(self):
        state = fixture()
        model = pb.Balance(state, 1.)
        # Physical torque is R * (r cross F), with F = line force * length.
        expected_torque = np.array([0., 0., pb.RADIUS_M*FORCE_N_PER_M*1e6])
        np.testing.assert_allclose(model.drivers['slab'][:3]*pb.RADIUS_M/pb.CM_YR_M_S,
                                   expected_torque, rtol=1e-14)
        np.testing.assert_array_equal(model.drivers['slab'][3:], 0.)
        np.testing.assert_array_equal(model.slab_over_drive, 0.)
        basal = 4e12*pb.ASTHENOSPHERE_DRAG_PA_S_M*pb.CM_YR_M_S**2*np.eye(3)
        np.testing.assert_allclose(model.basal[0], basal, rtol=1e-14)
        self.assertGreater(np.linalg.eigvalsh(model.stiffness).min(), 0.)
        self.assertEqual(len(model.slab_stokes_coefficient), 0)
        self.assertEqual(len(model.slab_anchor_coefficient), 0)
        self.assertEqual(len(model.hinge_coefficient), 1)
        self.assertTrue(any(item['kind'] == 'megathrust' for item in model.elements))

    def test_subdivision_and_boundary_storage_reversal_preserve_operator(self):
        base = pb.Balance(fixture(), 1.)
        base.solve()
        for count in (2, 10):
            state = fixture((1000./count,)*count)
            state.bp[::2], state.bq[::2] = state.bq[::2].copy(), state.bp[::2].copy()
            state.ba[::2], state.bb[::2] = state.bb[::2].copy(), state.ba[::2].copy()
            state.bn[::2] *= -1.
            model = pb.Balance(state, 1.)
            np.testing.assert_allclose(model.torque, base.torque, rtol=2e-14)
            np.testing.assert_allclose(model.stiffness, base.stiffness, rtol=2e-14)
            model.solve()
            np.testing.assert_allclose(model.rotation(), base.rotation(), rtol=1e-9, atol=1e-12)

    def test_start_from_rest_and_ignore_slab_inventory_dip_and_previous_velocity(self):
        state = fixture()
        baseline = pb.Balance(state, 1.)
        baseline.solve()
        self.assertGreater(np.linalg.norm(baseline.rotation()[0]), 0.)
        self.assertTrue(np.all(state.omega == 0.))
        for dip in (1., 89.):
            changed = deepcopy(state)
            changed.omega[:] = np.array([.04, -.09, .02])
            changed.trench_systems[0].update(slab_retained_excess_mass_kg=1e30,
                slab_retained_area_km2=1e20, slab_attachment=0., slab_line_load_kg_per_m=1e30)
            with (patch.object(slab_memory, 'line_load', side_effect=AssertionError('Detailed slab source accessed')),
                  patch.object(slab_memory, 'SUBDUCTION_DIP_DEG', dip)):
                model = pb.Balance(changed, 1.)
                model.solve()
            np.testing.assert_array_equal(model.torque, baseline.torque)
            np.testing.assert_array_equal(model.stiffness, baseline.stiffness)
            np.testing.assert_allclose(model.rotation(), baseline.rotation(), rtol=1e-9, atol=1e-12)

    def test_trace_declaration_has_force_through_opening_and_color_changes(self):
        state = fixture()
        expected = effective.line_state(state)
        for speed, code in ((0., 0), (20., 1), (-20., 2), (1e-3, 3)):
            state.normal_speed[:] = speed
            state.bcode[:] = code
            state.down[:] = -1
            found = effective.line_state(state)
            for actual, wanted in zip(found, expected):
                np.testing.assert_array_equal(actual, wanted)
        state.trench_id[:] = 0
        model = pb.Balance(state, 1.)
        model.solve()
        np.testing.assert_array_equal(model.x, 0.)

    def test_work_ledger_matches_full_force_and_operator_for_arbitrary_motion(self):
        state = fixture(edges=(400., 600.))
        state.bmid[1] = [0., 1., 0.]
        state.bn[1] = [0., 0., 1.]
        model = pb.Balance(state, 1.)
        rng = np.random.default_rng(20261002)
        for x in rng.normal(size=(8, model.size))*12.:
            report = ledger.export(model, x)
            expected = model._evaluate(x, report['delta'])
            np.testing.assert_allclose(report['generalized_force'], -expected[1], rtol=2e-11, atol=1e-4)
            compiled = ledger.compile_mode(report, uniform_basis(model))
            for actual, wanted in zip(compiled.evaluate(x), expected[:3]):
                np.testing.assert_allclose(actual, wanted, rtol=2e-11, atol=.01)
            self.assertEqual(compiled._resistance_work(x, report['delta']),
                             model._resistance_work(x, report['delta']))
            work, passivity = model._resistance_work(x, report['delta'])
            self.assertGreaterEqual(sum(work.values()), 0.)
            self.assertEqual(passivity['negative_resisting_work_elements'], 0)
            self.assertIn('effective_subduction', report['plates'][0]['localcomponents_n_m'])
            self.assertNotIn('attached_slab', report['plates'][0]['localcomponents_n_m'])
            self.assertFalse(report['diagnostics']['equilibrium_reaction_added'])
        model.solve()
        report = model.diagnostics(model.rotation())
        self.assertLessEqual(report['scaled_force_residual'], pb.FORCE_RELATIVE_TOLERANCE)
        self.assertLess(abs(report['power_balance_error_w']), 1e-8*abs(report['driver_work_w']))
        self.assertEqual(report['slab_stokes_dissipation_w'], 0.)
        self.assertEqual(report['slab_anchor_dissipation_w'], 0.)

    def test_opposed_pulls_cancel_whole_plate_but_have_finite_internal_tension(self):
        state = opposed_fixture()
        model = pb.Balance(state, 1.)
        model.solve()
        np.testing.assert_array_equal(model.drivers['slab'][:3], 0.)
        np.testing.assert_array_equal(model.x, 0.)
        # An exact two-port strip exposes separating work even though the
        # rigid input plate is stationary. Independent yield work is L*R*S.
        physical = FORCE_N_PER_M*1e6*pb.RADIUS_M
        moments = np.array([[0., 0., -physical], [0., 0., physical]])
        separating = np.array([[0., 0., -.5], [0., 0., .5]])
        self.assertAlmostEqual(float(np.sum(moments*separating))/physical, 1., places=14)
        for factor, ratio in ((2., .5), (.5, 2.)):
            strength = np.array([factor*FORCE_N_PER_M])
            found = limit.analyse(state.bmid[::-1], moments, np.array([[0, 1]]),
                np.array([[1., 0., 0.]]), np.array([1e6]),
                (strength, 3.*strength, strength/math.sqrt(3.)),
                axes=np.array([[0., 0., 1.]]))
            self.assertAlmostEqual(found['ratio'], ratio, places=12)
            np.testing.assert_array_equal(found['cut'], [True])
        # Place well-conditioned control neighborhoods at the actual ports so
        # the shared ledger also preserves the separating virtual work.
        points = []
        for center in state.bmid:
            tangent = np.cross([0., 0., 1.], center)
            for d in (-.006, 0., .006):
                for z in (-.004, .004):
                    point = center+d*tangent+np.array([0., 0., z])
                    points.append(point/np.linalg.norm(point))
        state.xyz = np.vstack((points, np.eye(3), -np.eye(3), np.eye(3), -np.eye(3)))
        state.plate = np.r_[np.zeros(12, int), np.ones(6, int), np.full(6, 2, int)]
        state.cell_area = np.full(24, 1e6)
        state.crust = np.zeros(24, int)
        state.age = np.full(24, 100.)
        state.ba, state.bb = np.array([0, 6]), np.array([12, 18])
        model = pb.Balance(state, 1.)
        report = ledger.export(model, np.zeros(model.size), band_km=100.)
        local = report['plates'][0]['localcomponents_n_m']['effective_subduction']
        self.assertLess(np.linalg.norm(local.sum(axis=0))/np.linalg.norm(local), 1e-12)
        basis = uniform_basis(model, 1)
        basis[0][:6, 2, -1] = .5
        basis[0][6:12, 2, -1] = -.5
        compiled = ledger.compile_mode(report, basis)
        cells = report['plates'][0]['cells']
        exact = float(np.sum(report['plates'][0]['cell_generalized_force']*basis[0][cells, :, -1]))
        self.assertGreater(exact, 0.)
        self.assertAlmostEqual(-compiled.evaluate(np.zeros(model.size+1))[1][-1]/exact, 1., places=12)

    def test_buoyant_arrival_removes_only_its_local_pull_and_keeps_collision_resistance(self):
        state = opposed_fixture()
        state.ba[:] = [0, 1]
        state.crust[0] = 1
        owners, forces, live = effective.line_state(state)
        np.testing.assert_array_equal(owners, [0, 0])
        np.testing.assert_array_equal(live, [True, True])
        np.testing.assert_array_equal(forces, [0., FORCE_N_PER_M])
        model = pb.Balance(state, 1.)
        np.testing.assert_array_equal(model.trench_edges, [1])
        expected = -FORCE_N_PER_M*1e6*pb.CM_YR_M_S
        self.assertAlmostEqual(model.drivers['slab'][2]/expected, 1., places=14)
        compression = next(item for item in model.elements if item['kind'] == 'compression')
        np.testing.assert_array_equal(compression['edges'], [0])
        np.testing.assert_array_equal(model.drivers['slab'][3:], 0.)
        state.trench_systems[1]['phase'] = 'shutdown'
        model = pb.Balance(state, 1.)
        model.solve()
        np.testing.assert_array_equal(model.x, 0.)

    def test_finite_daughter_response_uses_effective_ledger_and_closes_cut_work(self):
        state = opposed_fixture()
        for group, sign in ((slice(0, 3), 1.), (slice(3, 6), -1.)):
            points = np.array([[math.cos(sign*.3+d), math.sin(sign*.3+d), z]
                               for d, z in ((-.025, -.04), (0., .04), (.025, 0.))])
            state.xyz[group] = points/np.linalg.norm(points, axis=1)[:, None]
        state.ba = np.array([0, 3])
        model = pb.Balance(state, 1.)
        model.solve()
        state.omega = model.rotation()
        cells = np.arange(6)
        piece = np.array([True, True, True, False, False, False])
        axis = np.array([0., 0., 1.])
        cut = dict(edges=np.array([[0, 3]]), mid=np.array([[1., 0., 0.]]),
                   length_m=np.array([1e6]), cut=np.array([True]))
        results = []
        for factor in (2., .5):
            strength = tuple(np.array([factor*FORCE_N_PER_M*v])
                             for v in (1., 3., 1./math.sqrt(3.)))
            result = breakup_mode.solve(state, 0, cells, piece, axis, cut, strength, model)
            self.assertTrue(result['opening_mode_supported'], result['opening_mode_reason'])
            self.assertLess(result['mode_work']['closure_relative'], 1e-8)
            self.assertLess(result['common_restriction_relative_error'], 1e-9)
            self.assertLess(result['differential_virtual_work_relative_error'], 1e-9)
            results.append(result)
        self.assertEqual(results[0]['predicted_opening_cm_yr_mean'], 0.)
        self.assertGreater(results[1]['predicted_opening_cm_yr_mean'], 0.)
        self.assertGreater(results[1]['mode_work']['cut_power_w'], 0.)
        self.assertEqual(results[1]['mode_passivity']['negative_resisting_work_elements'], 0)


if __name__ == '__main__':
    unittest.main()
