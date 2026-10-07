"""Progressive rifts require repeated local deformation, not breakup clocks.

The integrated fixture prescribes opposing ocean-boundary velocities while
holding geometry fixed, like a loading experiment. The real local membrane,
column, scar, damage, history and final material-splitting code all execute.
This is a controlled worldbuilding closure test, not an Earth calibration.
"""
from copy import deepcopy
from types import SimpleNamespace
import json
import unittest
from unittest.mock import Mock, patch

import numpy as np

import progressive_rifting as rifts
from ridge_geometry import rotate
import structure_engine as structure
from raster_engine import Simulation


def loaded_continent():
    width, height = 96, 48
    lon = (np.arange(width)+.5)*2*np.pi/width-np.pi
    lat = np.pi/2-(np.arange(height)+.5)*np.pi/height
    lo, la = np.meshgrid(lon, lat)
    land = (np.abs(lo) < .6) & (np.abs(la) < .35)
    crust = np.where(land, np.where(np.abs(lo) > .16, 2, 1), 0).astype(np.uint8)
    s = Simulation(dict(width=width, height=height, plate_count=4, seed=12,
                        erosion=0., mechanics_nodes=1024),
                   dict(width=width, height=height, crust=crust.ravel()))
    # A single continental block between two prescribed oceanic drivers. The
    # two cold cratonic flanks are separated by initially undamaged mobile crust.
    s.active[:] = False
    s.active[:3] = True
    s.plate[:] = np.where(land.ravel(), 0, np.where(lo.ravel() < 0, 1, 2))
    s.parcel_plate[:] = 0
    s.trace_plate[:] = 0
    s.support[:] = 0
    s.support[s.plate, np.arange(s.n)] = 1
    s.omega[:] = 0
    s.omega[1, 2], s.omega[2, 2] = -.008, .008
    s.rift_id[:] = s.trace_rift_id[:] = -1
    s.rift_extension_m[:] = s.trace_rift_extension_m[:] = 0
    s.rift_records, s.rift_pair_ids, s.next_rift_id = [], {}, 1
    s.born[:] = 0
    s.t, s.steps = 0., 0
    s.initial_ocean_plate_uid = -1
    s.ridge_episodes = []
    # Initial random plate identities have been replaced by this prescribed
    # experiment, so initialize its reference graph before its first loading.
    del s.rift_material
    rifts.initialize(s)
    s._boundaries()
    return s


def load_step(s):
    s.t += 2.
    s.steps += 1
    zero = np.zeros(s.n)
    # Time-dependent cooling executes once. The separate local response owns
    # the prescribed extension, so the boundary extension grid remains zero.
    structure.deform(s, zero, zero, zero, 2.)
    rifts.update(s, 2., zero)


def material_ledger(s):
    return s.trace_uplift_m-s.trace_extension_m-s.trace_erosion_m+s.trace_adjustment_m


class ProgressiveRiftingTests(unittest.TestCase):
    def test_opposite_strike_vector_signs_retain_the_same_inherited_weakness(self):
        xyz = np.array([[1., -.03, 0.], [1., .03, 0.]])
        xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
        mesh = dict(xyz=xyz, edges=np.array([[0, 1]]), parcel_node=np.array([0, 0, 1, 1]))
        s = SimpleNamespace(rift_id=np.ones(4, int), mass=np.ones(4),
                            rift_tangent=np.tile([0., 0., 1.], (4, 1)))
        reference = rifts._inherited_alignment(s, mesh)
        s.rift_tangent[[1, 3]] *= -1.
        reversed_signs = rifts._inherited_alignment(s, mesh)
        np.testing.assert_allclose(reference, 1., atol=1e-12)
        np.testing.assert_allclose(reversed_signs, reference, atol=1e-12)

    def test_neighboring_parallel_cross_rift_links_form_one_tracked_belt(self):
        xyz = np.array([[1., -.03, -.05], [1., .03, -.05],
                        [1., -.03, 0.], [1., .03, 0.],
                        [1., -.03, .05], [1., .03, .05]])
        xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
        edges = np.array([[0, 1], [2, 3], [4, 5], [0, 2], [2, 4], [1, 3], [3, 5]])
        mesh = dict(xyz=xyz, edges=edges, owners=np.zeros(6, int), owner_uids=np.full(6, 7),
                    area=np.ones(6), bases=np.arange(6))
        s = SimpleNamespace(rift_systems=[], next_rift_system_id=1, t=12.,
                            active=np.array([True]), plate_uid=np.array([7]), _record=Mock())
        damage = np.array([.2, .2, .2, 0., 0., 0., 0.])
        rifts._track(s, mesh, rifts._edge_keys(mesh), damage, damage*.1,
                     np.ones(7)*5., np.ones(6))
        self.assertEqual(len(s.rift_systems), 1)
        self.assertEqual(s.rift_systems[0]['_bonds'], {(0, 1, 7), (2, 3, 7), (4, 5, 7)})
        self.assertEqual(s.rift_systems[0]['phase'], 'incipient')

    def test_partial_transfer_copies_then_independently_evolves_material_bonds(self):
        mesh = dict(bases=np.array([4, 5, 4, 5]), owner_uids=np.array([10, 10, 11, 11]),
                    edges=np.array([[0, 1], [2, 3]]))
        keys = rifts._edge_keys(mesh)
        self.assertEqual(keys, [(4, 5, 10), (4, 5, 11)])
        s = SimpleNamespace(rift_bonds={keys[0]: dict(damage=.4, strain=.2,
                                                     extension_km=80., last_seen_myr=20.)})
        rifts._inherit_bonds(s, keys+[(4, 5, 12)])
        self.assertEqual(s.rift_bonds[keys[1]]['damage'], .4)
        self.assertEqual(s.rift_bonds[(4, 5, 12)]['strain'], .2)
        self.assertIsNot(s.rift_bonds[keys[0]], s.rift_bonds[keys[1]])
        s.rift_bonds[keys[1]].update(damage=.8, strain=.6, last_seen_myr=30.)
        s.rift_bonds[keys[0]]['damage'] = .1
        rifts._inherit_bonds(s, keys+[(4, 5, 12), (4, 5, 13)])
        self.assertEqual(s.rift_bonds[keys[1]]['damage'], .8)
        self.assertEqual(s.rift_bonds[(4, 5, 12)]['damage'], .4)
        self.assertEqual(s.rift_bonds[(4, 5, 13)]['damage'], .8)
        self.assertEqual(s.rift_bonds[(4, 5, 13)]['inherited_from'], keys[1])

    def test_column_budget_reports_net_bounded_thinning_for_boundary_deduplication(self):
        s = loaded_continent()
        zero = np.zeros(s.n)
        equal = structure.deform(s, np.full(s.n, 30.), np.full(s.n, 30.), zero, 2.)
        np.testing.assert_array_equal(equal['parcel']['extension_strain'], 0.)
        np.testing.assert_array_equal(equal['trace']['extension_strain'], 0.)
        net = structure.deform(s, np.full(s.n, 30.), np.full(s.n, 50.), zero, 2.)
        expected = .25*np.where(s.kind == 2, .35, 1.)*(50.-30.)/400.*2.
        np.testing.assert_allclose(net['parcel']['extension_strain'], expected, atol=1e-14)
        limited = structure.deform(s, zero, np.full(s.n, 1e8), zero, 2.)
        np.testing.assert_allclose(limited['parcel']['extension_strain'], .16, atol=1e-14)

    def test_cancelled_boundary_extension_does_not_suppress_real_interior_thinning(self):
        first = loaded_continent()
        second = deepcopy(first)
        actual = (np.zeros(len(first.mass)), np.zeros(len(first.trace_id)))
        # A requested boundary extension can be cancelled by simultaneous
        # shortening. Both cases have the same actual boundary thinning (zero).
        rifts.update(first, 2., np.zeros(first.n), realized_extension=actual)
        rifts.update(second, 2., np.full(second.n, 1000.), realized_extension=actual)
        self.assertTrue(np.any(first.structure['thickness_km'] < first.structure['reference_thickness_km']))
        for name in ('structure', 'trace_structure'):
            for field in structure.FIELDS:
                np.testing.assert_array_equal(getattr(first, name)[field], getattr(second, name)[field], err_msg=field)
        np.testing.assert_array_equal(first.relief, second.relief)
        np.testing.assert_array_equal(first.trace_relief_m, second.trace_relief_m)

    def test_historical_damage_does_not_break_currently_compressing_links(self):
        s = SimpleNamespace(active=np.array([True, False]),
                            rift_pending=dict(reliable=True, mesh={}, damage=np.ones(3),
                                              strain=np.ones(3), edge_extension=np.array([-2., -.1, 0.])))
        # Even a historical fully damaged belt is not a current rupture when
        # its actual resolved cross-belt motion is closing or stationary.
        with patch.object(rifts.rift_material, 'refresh', side_effect=AssertionError('No active extension')):
            self.assertFalse(rifts.commit(s))

    def test_relative_strength_distinguishes_cold_cratons_thinning_heat_and_damage(self):
        cold = rifts.relative_strength(35., 35., 0., 0., 0., 0.)
        craton = rifts.relative_strength(42., 42., 1., 0., 0., 0.)
        self.assertGreater(craton, cold*4)
        for weak in (rifts.relative_strength(20., 35., 0., 0., 0., 0.),
                     rifts.relative_strength(35., 35., 0., 1., 0., 0.),
                     rifts.relative_strength(35., 35., 0., 0., 1000., 0.),
                     rifts.relative_strength(35., 35., 0., 0., 0., .8)):
            self.assertLess(weak, cold)
            self.assertGreaterEqual(weak, .06)

    def test_compression_and_quiet_cannot_grow_damage_but_extension_can(self):
        old = np.array([0., .2, .75, 1.])
        quiet = rifts.evolve_damage(old, np.zeros(4), np.ones(4), 2.)
        compressed = rifts.evolve_damage(old, np.full(4, -.2), np.ones(4), 2., compression=.15)
        extended = rifts.evolve_damage(old, np.full(4, .02), np.ones(4), 2.)
        self.assertTrue(np.all(quiet <= old))
        self.assertTrue(np.all(compressed <= quiet))
        self.assertEqual(compressed[0], 0.)
        self.assertTrue(np.all(extended[:3] > old[:3]))
        self.assertTrue(np.all(extended <= 1.))
        weak = rifts.evolve_damage(old, np.full(4, .02), np.full(4, .3), 2.)
        self.assertTrue(np.all(weak >= extended))

    def test_common_plate_motion_produces_no_interior_damage_or_new_basin(self):
        s = loaded_continent()
        s.omega[:] = [.002, -.003, .007]
        s._boundaries()
        old_relief = s.relief.copy()
        old_trace = s.trace_relief_m.copy()
        for _ in range(10):
            load_step(s)
            self.assertFalse(rifts.commit(s))
        self.assertEqual(s.rift_systems, [])
        self.assertFalse(any(value['damage'] or value['strain'] or value['extension_km']
                             for value in s.rift_bonds.values()))
        np.testing.assert_array_equal(s.relief, old_relief)
        np.testing.assert_array_equal(s.trace_relief_m, old_trace)
        self.assertTrue(s.rift_mechanics['converged'])

    def test_unloaded_daughters_have_no_invented_opening_or_rigid_bias(self):
        xyz = np.array([[1., -.3, .1], [1., -.2, -.15], [1., -.05, .2],
                        [1., .05, -.2], [1., .2, .15], [1., .3, -.1]])
        xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
        area = np.array([3., 2., 4., 7., 2., 1.])*1e5
        positive = xyz[:, 1] > 0
        zero = rifts.daughter_rotations(xyz, area, np.zeros_like(xyz), positive)
        np.testing.assert_array_equal(zero, 0.)
        common = np.array([.01, -.004, .006])
        rigid = rifts.daughter_rotations(xyz, area, np.cross(common, xyz)*rifts.RADIUS_KM, positive)
        np.testing.assert_allclose(rigid, 0., atol=1e-14)

    def test_daughter_fit_conserves_common_motion_and_rotates_with_globe(self):
        rng = np.random.default_rng(822)
        xyz = rng.normal(size=(50, 3))
        xyz[:, 0] += 4
        xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
        area = rng.uniform(.5, 4., len(xyz))*1e5
        positive = xyz[:, 1] > 0
        drive = rng.normal(size=xyz.shape)*12.
        drive -= xyz*np.sum(drive*xyz, axis=1)[:, None]
        result = rifts.daughter_rotations(xyz, area, drive, positive)
        matrices = np.eye(3)[None, :, :]-xyz[:, :, None]*xyz[:, None, :]
        torque = np.zeros(3)
        for select, omega in zip((~positive, positive), result):
            torque += np.sum(matrices[select]*area[select, None, None], axis=0)@omega
        np.testing.assert_allclose(torque, 0., atol=1e-9)
        rotation = np.array([.3, 1.6, -.8])
        moved = rifts.daughter_rotations(rotate(xyz, rotation), area, rotate(drive, rotation), positive)
        np.testing.assert_allclose(moved, rotate(result, rotation), atol=1e-13)

    def test_a_loaded_basin_can_fail_then_reactivate_with_its_inherited_identity(self):
        s = loaded_continent()
        birth = s.trace_relief_m.copy()
        volume = s.structure['thickness_km']*s.structure['area_factor']
        for _ in range(20):
            load_step(s)
        self.assertFalse(any(row['phase'] == 'broken_through' for row in s.rift_systems))
        active = [row for row in s.rift_systems if row['phase'] == 'active']
        self.assertTrue(active)
        selected_id = max(active, key=lambda row: row['peak_damage'])['id']
        accumulated = {key: value['strain'] for key, value in s.rift_bonds.items()}
        driving = s.omega.copy()
        s.omega[:] = 0.
        s._boundaries()
        for _ in range(12):
            load_step(s)
            self.assertFalse(rifts.commit(s))
        selected = next(row for row in s.rift_systems if row['id'] == selected_id)
        self.assertEqual(selected['phase'], 'failed')
        for key, strain in accumulated.items():
            self.assertEqual(s.rift_bonds[key]['strain'], strain)
        self.assertGreater(selected['peak_damage'], .35)
        s.omega[:] = driving
        s._boundaries()
        for _ in range(4):
            load_step(s)
        self.assertEqual(selected['phase'], 'active')
        phases = [item['phase'] for item in selected['history']]
        self.assertIn('failed', phases)
        self.assertEqual(phases[-1], 'active')
        self.assertTrue(any(event['type'] == 'rift_reactivated' for event in s.events))
        np.testing.assert_allclose(s.structure['thickness_km']*s.structure['area_factor'], volume, atol=2e-11)
        np.testing.assert_allclose(s.trace_relief_m-birth, material_ledger(s), atol=1e-9)
        json.dumps(rifts.snapshot(s), allow_nan=False)

    def test_repeated_local_loading_reaches_actual_breakthrough_without_opening_kick(self):
        s = loaded_continent()
        birth = s.trace_relief_m.copy()
        material = {key: getattr(s, key).copy() for key in ('mass', 'kind', 'pos', 'parcel_patch', 'trace_id', 'trace_xyz')}
        mantle = s.mantle.copy()
        q = int(np.flatnonzero(~s.active)[0])
        complete = False
        with (patch.object(s, '_choose_fracture', side_effect=AssertionError('A developed rift must not choose a new random cut.')),
              patch.object(structure, 'cut_rift', side_effect=AssertionError('A developed rift must not add a fixed height cut.'))):
            for _ in range(200):
                load_step(s)
                before_relief = s.relief.copy()
                before_trace = s.trace_relief_m.copy()
                before_omega = s.omega[0].copy()
                with patch.object(s, '_split', wraps=s._split) as split:
                    complete = rifts.commit(s)
                    if complete:
                        chosen = split.call_args.kwargs['chosen']
                        self.assertTrue(split.call_args.kwargs['progressive'])
                        break
        self.assertTrue(complete, (s.t, s.rift_mechanics, rifts.snapshot(s)))
        self.assertGreater(s.t, 20.)
        self.assertTrue(s.active[q])
        for key, value in material.items():
            np.testing.assert_array_equal(getattr(s, key), value, err_msg=key)
        np.testing.assert_array_equal(s.relief, before_relief)
        np.testing.assert_array_equal(s.trace_relief_m, before_trace)
        np.testing.assert_array_equal(s.mantle[0], mantle[0])
        np.testing.assert_array_equal(s.mantle[q], mantle[0])
        np.testing.assert_allclose(s.omega[0], before_omega+chosen['rotations'][0], atol=1e-15)
        np.testing.assert_allclose(s.omega[q], before_omega+chosen['rotations'][1], atol=1e-15)
        self.assertGreater(np.linalg.norm(s.omega[q]-s.omega[0]), 0.)
        for group in np.unique(s.parcel_craton[s.parcel_craton >= 0]):
            self.assertEqual(len(np.unique(s.parcel_plate[s.parcel_craton == group])), 1)
        broken = [row for row in s.rift_systems if row['phase'] == 'broken_through']
        self.assertEqual(len(broken), 1)
        self.assertGreater(len(broken[0]['history']), 2)
        self.assertGreater(broken[0]['extension_km'], 0.)
        event = next(event for event in reversed(s.events) if event['type'] == 'rift')
        self.assertEqual(event['details']['rift_belt_relief_reduction_m'], 0)
        np.testing.assert_allclose(s.trace_relief_m-birth, material_ledger(s), atol=2e-9)


if __name__ == '__main__':
    unittest.main()
