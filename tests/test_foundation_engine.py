"""Late geological integration: conservative maps, local identity and saved evidence."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from crust_transport import deposit, footprints
import geology_snapshot
from history import STRUCTURE_TRACE_FIELDS
import local_accretion
from server import SimulationManager, write_json
import structure_engine
from raster_engine import Simulation, _rotate
from tests.test_local_accretion import world, mature


class FoundationEngineTests(unittest.TestCase):
    def test_ambiguous_coalesced_contact_anchor_cannot_inherit_a_mature_debt(self):
        for anchor in ('p_patch', 'q_patch'):
            with self.subTest(anchor=anchor):
                s = world()
                plan = mature(s)
                old = s.local_accretion_contacts[0]
                old_id, old_patch = old['id'], old[anchor]
                original_loading = old['loading']
                self.assertGreaterEqual(original_loading, local_accretion.THRESHOLD)
                # One formerly shared ID now describes two real coalesced
                # columns. Neither descendant may arbitrarily inherit its debt.
                parcels = np.flatnonzero(s.parcel_patch == old_patch)
                replacements = np.where(np.arange(len(parcels)) % 2, 90001, 90000)
                s.parcel_patch[parcels] = replacements
                s.trace_patch[s.trace_patch == old_patch] = 90000
                local_accretion.remap_contact_patches(s, np.full(len(parcels), old_patch), replacements)
                self.assertEqual(old['loading'], 0.)
                self.assertTrue(old['completed'])
                self.assertEqual(old[anchor], old_patch, 'ambiguous identity is not relabelled to one arbitrary child')
                self.assertFalse(local_accretion.apply_accretion(s, plan))
                s.t += 2.
                self.assertEqual(local_accretion.plan_accretions(s, 2.), [])
                current = [row for row in s.local_accretion_contacts if row['last_seen_myr'] == s.t]
                self.assertEqual(len(current), 1)
                self.assertNotEqual(current[0]['id'], old_id)
                self.assertLess(current[0]['loading'], local_accretion.THRESHOLD)

    def test_repeated_unambiguous_coalescence_preserves_contact_identity_and_loading(self):
        s = world()
        for _ in range(4):
            s.t += 2.
            local_accretion.plan_accretions(s, 2.)
        old = deepcopy(s.local_accretion_contacts[0])
        local_accretion.remap_contact_patches(s, np.array([old['p_patch']]*4), np.full(4, 90000))
        result = s.local_accretion_contacts[0]
        self.assertEqual(result['p_patch'], 90000)
        self.assertIn(90000,result['p_patches'])
        self.assertNotIn(old['p_patch'],result['p_patches'])
        for key in old.keys()-{'p_patch','p_patches'}:
            self.assertEqual(result[key], old[key], key)

    def test_extra_properties_follow_full_spherical_footprints_and_conserve_integrals(self):
        s = world(width=64, height=32)
        # Finite equatorial footprints transported exactly onto both poles,
        # plus an antimeridian footprint. No centre-cell-only projection works.
        position = np.array([[1., 0., 0.]]*3)
        east, extent = footprints(position, s.w, s.h)
        extent *= 3.
        for i, rotation in enumerate(([0., -np.pi/2, 0.], [0., np.pi/2, 0.], [0., 0., np.pi])):
            position[i:i+1] = _rotate(position[i:i+1], np.asarray(rotation))
            east[i:i+1] = _rotate(east[i:i+1], np.asarray(rotation))
        mass = np.array([100., 300., 200.])
        fields = dict(thickness=np.array([32., 64., 21.]), unit=np.ones(3), signed=np.array([-1., 40., 0.]))
        args = (position, east, extent, mass, np.array([1, 2, 3]), np.array([100., 800., 500.]),
                np.zeros(3), np.array([0, 1, 0]), s.w, s.h, s.xyz, s.cell_area, s._sample_coordinates)
        basic = deposit(*args)
        projected = deposit(*args, extra_fields=fields, chunk_size=1)
        for actual, expected in zip(projected[:7], basic):
            np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-11)
        amount, extra = projected[0], projected[-1]
        self.assertAlmostEqual(float(amount.sum()), float(mass.sum()), places=10)
        np.testing.assert_allclose(extra['unit'], amount, atol=1e-12)
        for name, source in fields.items():
            self.assertAlmostEqual(float(extra[name].sum()), float(mass @ source), places=8)
        for cells, expected in ((np.flatnonzero(s.xyz[:, 2] > .9), 32.),
                                (np.flatnonzero(s.xyz[:, 2] < -.9), 64.)):
            cells = cells[amount[cells] > 0]
            self.assertGreater(len(np.unique(cells % s.w)), s.w//2)
            np.testing.assert_allclose(extra['thickness'][cells]/amount[cells], expected, atol=1e-12)
        seam = amount.reshape(s.h, s.w)
        self.assertGreater(seam[:, 0].sum(), 0.)
        self.assertGreater(seam[:, -1].sum(), 0.)

    def test_mixed_properties_use_reference_mass_not_class_or_owner(self):
        s = world()
        position = np.repeat(s.xyz[[8*s.w+12]], 2, axis=0)
        east, extent = footprints(position, s.w, s.h)
        result = deposit(position, east, extent*2, np.array([10., 30.]), np.array([1, 3]),
                         np.zeros(2), np.zeros(2), np.array([0, 1]), s.w, s.h, s.xyz,
                         s.cell_area, s._sample_coordinates,
                         extra_fields={'column': np.array([32., 64.])})
        visible = result[0] > 0
        self.assertGreater(visible.sum(), 1)
        np.testing.assert_allclose(result[-1]['column'][visible]/result[0][visible], 56., atol=1e-12)

    def test_mixed_arc_accretion_preserves_raster_full_height_when_class_changes(self):
        s = world()
        incoming = s.parcel_plate == 0
        # Three arc samples and one continental sample in each coherent patch.
        # Arc -> continent changes the majority map class, but not full height.
        juvenile = incoming & (np.arange(len(s.kind)) % 4 != 0)
        s.kind[juvenile] = 3
        s.trace_kind[s.trace_plate == 0] = 3
        s.trace_origin_kind[:] = s.trace_kind
        structure_engine.initialize_parcels(s)
        s.trace_structure = structure_engine._new(s.trace_kind, s.trace_relief_m)
        s.age = np.zeros(s.n)
        Simulation._rasterize(s)
        plan = mature(s)
        initial_height = s.land_height.copy()
        initial_class = s.crust.copy()
        initial_mass = s.land_mass.copy()
        columns = deepcopy(s.structure)
        full_material_height = structure_engine.material_height(s.kind, s.relief)
        selected = plan['parcel_indices']
        remote = incoming & ~np.isin(np.arange(len(s.kind)), selected)
        remote_kind, remote_relief = s.kind[remote].copy(), s.relief[remote].copy()
        self.assertTrue(np.any(initial_class[plan['surface_cells']] == 3))
        self.assertTrue(local_accretion.apply_accretion(s, plan))
        Simulation._rasterize(s)
        self.assertTrue(np.any(initial_class != s.crust), 'fixture must cross the mixed-cell class threshold')
        np.testing.assert_allclose(s.land_height, initial_height, rtol=1e-14, atol=1e-10)
        np.testing.assert_array_equal(s.land_mass, initial_mass)
        np.testing.assert_array_equal(structure_engine.material_height(s.kind, s.relief), full_material_height)
        land = s.crust > 0
        np.testing.assert_allclose(structure_engine.material_height(s.crust[land], s.land_relief[land]),
                                   s.land_height[land], atol=1e-10)
        np.testing.assert_array_equal(s.kind[remote], remote_kind)
        np.testing.assert_array_equal(s.relief[remote], remote_relief)
        for name, original in columns.items():
            np.testing.assert_array_equal(s.structure[name], original, err_msg=name)


class FoundationSavedHistoryTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1]/'tmp'
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        s = Simulation(dict(width=64, height=32, plate_count=4, seed=12, duration_myr=4,
                            dt_myr=2, snapshot_myr=2, erosion=1.))
        self.frames = [s.snapshot()]
        # Real column evolution and snapshot code; positions stay fixed so the
        # saved geometric field and followed column can be checked independently.
        s.bcode[:] = 3
        zeros = np.zeros(s.n)
        for extension in (30., 0.):
            s.t += 2.
            structure_engine.deform(s, zeros, np.full(s.n, extension), zeros, 2.)
            s._rasterize()
            self.frames.append(s.snapshot())
        manager = SimulationManager(self.temp.name)
        run_id = 'foundation-fixture'
        folder = manager.path(run_id)
        folder.mkdir()
        manager.current = dict(state='complete', run_id=run_id, config=s.config, frame_count=3,
            time_myr=4., duration_myr=4., frames=[dict(index=i, time_myr=2.*i, stats=frame['stats'])
                                               for i, frame in enumerate(self.frames)])
        write_json(folder/'config.json', s.config)
        write_json(folder/'initial.json', dict(width=s.w, height=s.h, crust=s.initial_crust))
        for index, frame in enumerate(self.frames):
            manager.save_frame(index, frame)
        manager.persist()
        self.manager = SimulationManager(self.temp.name)  # actual closed-file reload
        self.run_id = run_id

    def test_real_snapshots_reload_all_maps_and_follow_actual_material_diagnostics(self):
        self.assertEqual(self.manager.status()['run_id'], self.run_id)
        for index, expected in enumerate(self.frames):
            actual = self.manager.frame(index)
            self.assertEqual(actual['structure_version'], 1)
            for name in (*geology_snapshot.GRID_FIELDS, 'trench'):
                np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
            self.assertEqual(actual['trench_systems'], expected['trench_systems'])
        # Select a real mapped continental cell; the history reader independently
        # identifies its closest same-owner marker, then follows that stable ID.
        cell = int(np.flatnonzero(self.frames[-1]['crust'] > 0)[len(np.flatnonzero(self.frames[-1]['crust'] > 0))//2])
        result = self.manager.history(-1, cell, self.run_id)
        self.assertEqual(result['mode'], 'material')
        self.assertEqual([point['time_myr'] for point in result['points']], [0., 2., 4.])
        for point, frame in zip(result['points'], self.frames):
            marker = int(np.flatnonzero(frame['trace_id'] == result['trace_id'])[0])
            self.assertEqual(point['structure_version'], 1)
            for name in STRUCTURE_TRACE_FIELDS:
                self.assertEqual(point[name], float(frame['trace_'+name][marker]), name)
            self.assertAlmostEqual(point['denudation_m']-point['rebound_m'], point['erosion_m'], places=9)
        self.assertGreater(np.max(self.frames[-1]['trace_denudation_m']), 0.)
        self.assertGreater(np.max(self.frames[-1]['trace_thermal_subsidence_m']), 0.)

    def test_ocean_site_history_retains_defaults_without_inventing_material_counters(self):
        cell = int(np.flatnonzero(self.frames[-1]['crust'] == 0)[0])
        result = self.manager.history(-1, cell, self.run_id)
        self.assertEqual(result['mode'], 'location')
        for point in result['points']:
            self.assertEqual(point['structure_version'], 1)
            self.assertEqual(point['crustal_thickness_km'], 7.)
            self.assertEqual(point['crustal_root_km'], 0.)
            self.assertEqual(point['rift_cooling_age_myr'], -1.)
            self.assertEqual(point['rift_thermal_support_m'], 0.)
            self.assertEqual(point['foreland_deflection_m'], 0.)
            self.assertNotIn('denudation_m', point)
            self.assertNotIn('rebound_m', point)
            self.assertNotIn('thermal_subsidence_m', point)


if __name__ == '__main__':
    unittest.main()
