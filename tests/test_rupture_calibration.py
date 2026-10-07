"""Mode-specific rupture gates; numerical checks, not geological calibration."""
from contextlib import ExitStack
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import enhanced_rifting
import progressive_rifting as rifts
from tests.test_progressive_rifting import loaded_continent, load_step


def link_world(enhanced=False, length=400.):
    angle = length/rifts.RADIUS_KM
    xyz = np.array([[1., 0., 0.], [np.cos(angle), np.sin(angle), 0.]])
    mesh = dict(xyz=xyz, edges=np.array([[0, 1]]), bases=np.array([10, 11]),
                owner_uids=np.array([7, 7]), owners=np.zeros(2, int), area=np.ones(2),
                thickness=np.full(2, 35.), reference_thickness=np.full(2, 35.),
                craton=np.zeros(2), suture=np.zeros(2), heat=np.zeros(2),
                parcel_node=np.arange(2), trace_node=np.empty(0, int))
    state = SimpleNamespace(config=dict(mechanics_nodes=128, enhanced_rifting={'enabled': enhanced}),
                            rift_bonds={}, t=0., n=2, kind=np.ones(2, int), mass=np.ones(2),
                            parcel_cell=np.arange(2), trace_kind=np.empty(0, int),
                            trace_xyz=np.empty((0, 3)), _indices=lambda xyz: np.empty(0, int),
                            geometric_log_area=np.zeros(2), omega=np.zeros((1, 3)),
                            parcel_rift_realized_tension=np.zeros(2),
                            parcel_rift_realized_compression=np.zeros(2), active=np.array([True, False]))
    return state, mesh


def prescribed_update(state, mesh, dt, speed=1.):
    """Exercise update with a prescribed local response, omitting scar display."""
    a, b = mesh['edges'].T
    length = rifts.RADIUS_KM*np.arctan2(
        np.linalg.norm(np.cross(mesh['xyz'][a], mesh['xyz'][b]), axis=1),
        np.sum(mesh['xyz'][a]*mesh['xyz'][b], axis=1))
    response = dict(edge_length_km=length, edge_extension=np.array([speed]),
                    velocity=np.zeros((2, 3)), iterations=1, relative_residual=0.,
                    converged=True, component_count=1)
    with ExitStack() as stack:
        stack.enter_context(patch.object(rifts.rift_material, 'refresh', return_value=mesh))
        stack.enter_context(patch.object(rifts, 'boundary_loading', return_value=(None, None)))
        stack.enter_context(patch.object(rifts.rift_mechanics, 'solve_loading', return_value=response))
        stack.enter_context(patch.object(enhanced_rifting, 'material_viscosity', return_value=np.ones(2)))
        stack.enter_context(patch.object(rifts, '_inherited_alignment', return_value=np.zeros(1)))
        for name in ('_track', '_seed_scars', '_save_properties'):
            stack.enter_context(patch.object(rifts, name))
        rifts.update(state, dt, np.zeros(2))
    state.t += dt


def observed_cut_candidates(state, mesh):
    """Inspect the real cut algorithm's input; two nodes cannot form daughters."""
    with (patch.object(rifts.rift_material, 'refresh', return_value=mesh),
          patch.object(rifts.rift_mesh, 'coherent_cut', wraps=rifts.rift_mesh.coherent_cut) as cut,
          patch('backarc.protected_hosts', return_value=set())):
        assert not rifts.commit(state)
    return None if not cut.called else cut.call_args.args[2]


class RuptureCalibrationTests(unittest.TestCase):
    def test_update_and_commit_keep_proxy_and_realized_thresholds_distinct(self):
        for enhanced, threshold in ((False, .15), (True, .35)):
            with self.subTest(enhanced=enhanced):
                state, mesh = link_world(enhanced)
                state.rift_bonds[(10, 11, 7)] = dict(damage=1., strain=.2, extension_km=320.)
                if enhanced:
                    state.rift_realized_motion = dict(
                        **{k: mesh[k].copy() for k in ('bases', 'owner_uids', 'edges')},
                        strain=np.array([.001]), edge_extension=np.array([1.]),
                        velocity=np.zeros((2, 3)), dt_myr=1.)
                prescribed_update(state, mesh, 1.)
                self.assertEqual(state.rift_mechanics['rupture_strain_threshold'], threshold)
                self.assertEqual(state.rift_pending['damage_strain_measure'],
                                 state.rift_mechanics['damage_strain_measure'])
                candidates = observed_cut_candidates(state, mesh)
                if enhanced:
                    self.assertIsNone(candidates)
                    self.assertAlmostEqual(state.rift_bonds[(10, 11, 7)]['strain'], .201)
                    state.rift_pending['strain'][:] = .35
                    np.testing.assert_array_equal(observed_cut_candidates(state, mesh), [True])
                else:
                    np.testing.assert_array_equal(candidates, [True])
                    self.assertAlmostEqual(state.rift_bonds[(10, 11, 7)]['strain'], .200625)
                    state.rift_pending['strain'][:] = np.nextafter(.15, 0.)
                    self.assertIsNone(observed_cut_candidates(state, mesh))

    def test_historical_pending_without_annotation_uses_recorded_config(self):
        for enhanced in (False, True):
            state, mesh = link_world(enhanced)
            state.rift_pending = dict(mesh=mesh, reliable=True, damage=np.ones(1),
                                      strain=np.array([.2]), edge_extension=np.ones(1),
                                      velocity=np.zeros((2, 3)))
            candidates = observed_cut_candidates(state, mesh)
            self.assertEqual(candidates is None, enhanced)

    def test_mode_change_does_not_reinterpret_new_pending_work(self):
        state, mesh = link_world()
        state.rift_bonds[(10, 11, 7)] = dict(damage=1., strain=.8, extension_km=1280.)
        prescribed_update(state, mesh, 1.)
        old_bonds = deepcopy(state.rift_bonds)
        state.config['enhanced_rifting']['enabled'] = True
        with patch.object(rifts.rift_material, 'refresh', side_effect=AssertionError('stale measure')):
            self.assertFalse(rifts.commit(state))
        self.assertEqual(state.rift_bonds, old_bonds)

    def test_current_opening_and_damage_still_gate_both_modes(self):
        for enhanced in (False, True):
            for speed, damage in ((-2., 1.), (0., 1.), (.02, 1.), (2., .949)):
                with self.subTest(enhanced=enhanced, speed=speed, damage=damage):
                    state, mesh = link_world(enhanced)
                    state.rift_pending = dict(mesh=mesh, reliable=True, damage=np.array([damage]),
                                              strain=np.array([2.]), edge_extension=np.array([speed]))
                    self.assertIsNone(observed_cut_candidates(state, mesh))

    def test_fixed_rate_proxy_exposure_is_additive_across_step_partitions(self):
        for speed, expected in ((2., .005), (400., .32)):
            with self.subTest(speed=speed):
                whole, mesh = link_world()
                divided = deepcopy(whole)
                prescribed_update(whole, mesh, 4., speed=speed)
                for _ in range(8):
                    prescribed_update(divided, mesh, .5, speed=speed)
                for state in (whole, divided):
                    self.assertAlmostEqual(state.rift_bonds[(10, 11, 7)]['strain'], expected, places=15)
                    self.assertEqual(state.rift_bonds[(10, 11, 7)]['extension_km'], 4.*speed)
        # Damage integration is deliberately not asserted timestep invariant.

    def test_uniform_scaled_links_keep_same_proxy_strain_without_node_scaling(self):
        outcomes = []
        for length, speed, budget in ((400., 2., 128), (800., 4., 8192)):
            state, mesh = link_world(length=length)
            state.config['mechanics_nodes'] = budget
            prescribed_update(state, mesh, 2., speed=speed)
            outcomes.append(state.rift_bonds[(10, 11, 7)]['strain'])
            self.assertEqual(state.rift_mechanics['rupture_strain_threshold'], .15)
        np.testing.assert_allclose(outcomes, [.0025, .0025], rtol=0., atol=1e-17)

    def test_realized_monotonic_log_history_and_compression_have_distinct_roles(self):
        state, mesh = link_world(True)
        for factor in (1.2, 1.25, .8):
            before = {key: mesh[key].copy() for key in ('xyz', 'bases', 'owner_uids', 'edges', 'owners')}
            angle = np.arctan2(mesh['xyz'][1, 1], mesh['xyz'][1, 0])*factor
            mesh['xyz'][1] = [np.cos(angle), np.sin(angle), 0.]
            with patch.object(rifts.rift_material, 'refresh', return_value=mesh):
                enhanced_rifting.finish_motion(state, before, 1.)
            prescribed_update(state, mesh, 1.)
            if factor == 1.25:
                self.assertAlmostEqual(state.rift_bonds[(10, 11, 7)]['strain'], np.log(1.5), places=14)
                prior_damage = state.rift_bonds[(10, 11, 7)]['damage']
        row = state.rift_bonds[(10, 11, 7)]
        self.assertAlmostEqual(row['strain'], np.log(1.5), places=14)
        self.assertLess(row['damage'], prior_damage)
        self.assertIsNone(observed_cut_candidates(state, mesh))

    def test_a_run_keeps_the_realized_threshold_it_recorded(self):
        # SEP21T was started at .30; the reviewed .35 applies only when a run
        # records nothing, and the proxy mode ignores the realized value.
        for recorded, strain, breaks in ((None, .32, False), (.30, .32, True), (.30, np.nextafter(.30, 0.), False)):
            with self.subTest(recorded=recorded, strain=strain):
                state, mesh = link_world(True)
                if recorded is not None:
                    state.realized_rupture_strain_threshold = recorded
                self.assertEqual(rifts._rupture_calibration(state)['rupture_strain_threshold'],
                                 .35 if recorded is None else recorded)
                state.rift_pending = dict(mesh=mesh, reliable=True, damage=np.ones(1),
                                          strain=np.array([strain]), edge_extension=np.ones(1),
                                          velocity=np.zeros((2, 3)),
                                          damage_strain_measure=rifts._rupture_calibration(state)['damage_strain_measure'])
                candidates = observed_cut_candidates(state, mesh)
                self.assertEqual(candidates is not None, breaks)
        proxy, _ = link_world(False)
        proxy.realized_rupture_strain_threshold = .30
        self.assertEqual(rifts._rupture_calibration(proxy)['rupture_strain_threshold'], .15)

    def test_a_recorded_realized_threshold_must_be_a_fraction(self):
        for bad in (True, 0., 1., -.1, 1.5, '0.3', np.nan):
            with self.subTest(bad=bad):
                state, _ = link_world(True)
                state.realized_rupture_strain_threshold = bad
                with self.assertRaisesRegex(ValueError, 'realized rupture strain threshold'):
                    rifts._rupture_calibration(state)

    def test_inherited_proxy_history_is_copied_without_conversion(self):
        row = dict(damage=.73, strain=.18, extension_km=12., last_seen_myr=10.)
        state = SimpleNamespace(rift_bonds={(10, 11, 7): deepcopy(row)},
                                rift_material={'base_parent': {12: 10}})
        rifts._inherit_bonds(state, [(11, 12, 8)])
        inherited = state.rift_bonds[(11, 12, 8)]
        for key, value in row.items():
            self.assertEqual(inherited[key], value)
        self.assertIsNot(inherited, state.rift_bonds[(10, 11, 7)])

    def test_loaded_continent_breaks_through_before_old_proxy_gate(self):
        state = loaded_continent()
        material = {name: getattr(state, name).copy()
                    for name in ('mass', 'pos', 'kind', 'parcel_patch', 'trace_xyz', 'trace_id')}
        complete = False
        with patch.object(state, '_choose_fracture', side_effect=AssertionError('No new arbitrary crack')):
            for _ in range(150):
                load_step(state)
                with patch.object(rifts.rift_mesh, 'coherent_cut', wraps=rifts.rift_mesh.coherent_cut) as cut:
                    complete = rifts.commit(state)
                if complete:
                    break
        self.assertTrue(complete, state.rift_mechanics)
        failed = cut.call_args.args[2]
        self.assertTrue(np.any(state.rift_pending['strain'][failed] < .35))
        self.assertTrue(np.all(state.rift_pending['strain'][failed] >= .15))
        self.assertTrue(np.all(state.rift_pending['damage'][failed] >= .95))
        for name, value in material.items():
            np.testing.assert_array_equal(getattr(state, name), value, err_msg=name)
        self.assertEqual(sum(row['phase'] == 'broken_through' for row in state.rift_systems), 1)


if __name__ == '__main__':
    unittest.main()
