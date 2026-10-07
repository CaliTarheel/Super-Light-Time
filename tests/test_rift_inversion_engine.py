"""Inherited continental rifts retain identity and a finite uplift ledger."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from checkpoint import read_checkpoint, write_checkpoint
from plate_domains import PersistentDomainTracker
from raster_engine import Simulation, _rotate, _unit


def compressional_world(width=128, *, scar=True, erosion=0.):
    """A real convergent continental contact loads an old rift 1,000 km inland.

    The far-side divergent contact is disabled to isolate continental inversion;
    contact geometry and near-side compression still come from real boundaries.
    """
    height = width//2
    s = Simulation(dict(width=width, height=height, seed=11, plate_count=4,
                        duration_myr=200, dt_myr=2, snapshot_myr=2, erosion=erosion),
                   dict(width=width, height=height, crust=np.ones(width*height, np.uint8)))
    s.plate = (s.lon.ravel() > 0).astype(np.int16)
    s.active[:] = False
    s.active[:2] = True
    s.count = 2
    s.plate_uid[:] = 0
    s.plate_uid[:2] = [1, 2]
    s.next_plate_uid = 3
    s.names[:2] = ['Inherited-rift continent', 'Approaching continent']
    s.omega[:] = 0
    s.omega[0], s.omega[1] = [0., 0., .003], [0., 0., -.003]
    s.mantle[:] = 0
    s.support[:] = 0
    s.support[s.plate, np.arange(s.n)] = 1
    s.parcel_plate[:] = (np.arctan2(s.pos[:, 1], s.pos[:, 0]) > 0)
    s.trace_plate[:] = (np.arctan2(s.trace_xyz[:, 1], s.trace_xyz[:, 0]) > 0)
    s.t, s.steps = 100., 0
    s.born[:] = s.t
    s.events, s.event_keys, s.rift_records = [], set(), []
    s.rift_pair_ids, s.next_rift_id = {}, 42
    s.process_totals = {key: 0. for key in s.process_totals}
    s.ridge_episodes = []
    for prefix, points in (('', s.pos), ('trace_', s.trace_xyz)):
        getattr(s, prefix+'rift_id')[:] = -1
        getattr(s, prefix+'rift_birth_myr')[:] = -1
        getattr(s, prefix+'rift_tangent')[:] = 0
        getattr(s, prefix+'rift_extension_m')[:] = 0
        getattr(s, prefix+'inversion_uplift_m')[:] = 0
        if scar:
            lon = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
            lat = np.degrees(np.arcsin(points[:, 2]))
            belt = (lon > -12) & (lon < -7) & (np.abs(lat) < 20)
            getattr(s, prefix+'rift_id')[belt] = 42
            getattr(s, prefix+'rift_birth_myr')[belt] = 10.
            getattr(s, prefix+'rift_extension_m')[belt] = 100.
            strike = np.array([0., 0., 1.])-points[belt]*points[belt, 2, None]
            getattr(s, prefix+'rift_tangent')[belt] = _unit(strike)
    if scar:
        center = np.array([np.cos(np.radians(-10)), np.sin(np.radians(-10)), 0.])
        s.t = 10.
        s._new_rift_record('controlled inherited rift', (0,), center)
        s.t = 100.
    s._rasterize()
    s._boundaries()
    s.bcode[s.bcode != 4] = 3
    s.domain_tracker = PersistentDomainTracker(s.w, s.h)
    s._update_surface_domains()
    return s


class RiftInversionEngineTests(unittest.TestCase):
    def test_initial_accepted_breakup_seeds_structural_scars_without_fictitious_extension(self):
        s = Simulation(dict(width=64, height=32, plate_count=4, seed=12))
        self.assertTrue(s.initial_fractures)
        scar = s.rift_id >= 0
        self.assertTrue(np.any(scar))
        self.assertTrue(np.all(np.isin(s.kind[scar], [1, 2])))
        np.testing.assert_array_equal(s.rift_birth_myr[scar], 0.)
        self.assertFalse(np.any(s.rift_extension_m))
        self.assertFalse(np.any(s.inversion_uplift_m))
        self.assertFalse(np.any(s.trace_extension_m))
        np.testing.assert_allclose(np.linalg.norm(s.rift_tangent[scar], axis=1), 1., atol=1e-10)
        np.testing.assert_allclose(np.sum(s.rift_tangent[scar]*s.pos[scar], axis=1), 0., atol=1e-10)
        # Initial source-cell patches keep a coherent scar identity even though
        # the four finite material samples have different subcell positions.
        for patch in np.unique(s.parcel_patch[scar]):
            self.assertEqual(len(np.unique(s.rift_id[s.parcel_patch == patch])), 1)
        self.assertTrue(set(np.unique(s.rift_id[scar])).issubset({row['id'] for row in s.rift_records}))

    def test_preexisting_separate_islands_and_plain_collision_do_not_invent_rifts(self):
        width, height = 64, 32
        lon, lat = np.meshgrid((np.arange(width)+.5)*2*np.pi/width-np.pi,
                              np.pi/2-(np.arange(height)+.5)*np.pi/height)
        xyz = np.column_stack((np.cos(lat).ravel()*np.cos(lon).ravel(),
                               np.cos(lat).ravel()*np.sin(lon).ravel(), np.sin(lat).ravel()))
        centers = np.array([[1., 0., 0.], [-.5, np.sqrt(3)/2, 0.], [-.5, -np.sqrt(3)/2, 0.]])
        crust = (np.max(xyz@centers.T, axis=1) > np.cos(np.radians(18))).astype(np.uint8)
        s = Simulation(dict(width=width, height=height, plate_count=4),
                       dict(width=width, height=height, crust=crust))
        self.assertEqual(s.initial_fractures, [])
        self.assertTrue(np.all(s.rift_id == -1))
        c = compressional_world(scar=False)
        self.assertTrue(np.any(c.bcode == 4))
        c.t += 2.
        c._deform_and_accrete(2.)
        self.assertTrue(np.all(c.rift_id == -1))
        self.assertFalse(np.any(c.inversion_uplift_m))
        self.assertTrue(np.any(c.trace_uplift_m > 0))

    def test_inland_inversion_adds_realized_gain_once_without_changing_material(self):
        s = compressional_world(erosion=1.)
        # Isolate the inversion contribution. Otherwise its changed thickness
        # legitimately feeds back into the separate interior mechanics solve.
        s.config['rift_strength'] = 0.
        baseline = deepcopy(s)
        baseline.rift_extension_m[:] = 0
        baseline.trace_rift_extension_m[:] = 0
        old_mass, old_patch, old_relief = s.mass.copy(), s.parcel_patch.copy(), s.trace_relief_m.copy()
        for world in (s, baseline):
            world.t += 2.
            world._deform_and_accrete(2.)
            world._rasterize()
        selected = s.trace_rift_id == 42
        self.assertTrue(np.any(s.trace_inversion_uplift_m[selected] > 0),
                        'Current plate-boundary compression must reach the inherited inland rift')
        self.assertTrue(np.all(s.trace_inversion_uplift_m <= s.trace_uplift_m+1e-12))
        self.assertFalse(np.any(baseline.trace_inversion_uplift_m),
                         'An identified scar without recorded extension supplies no uplift budget')
        ledger = s.trace_uplift_m-s.trace_extension_m-s.trace_erosion_m+s.trace_adjustment_m
        np.testing.assert_allclose(s.trace_relief_m-old_relief, ledger, atol=1e-10)
        np.testing.assert_allclose(s.trace_uplift_m-baseline.trace_uplift_m,
                                   s.trace_inversion_uplift_m, atol=1e-10)
        np.testing.assert_array_equal(s.mass, old_mass)
        np.testing.assert_array_equal(s.parcel_patch, old_patch)
        self.assertLessEqual(float(s.inversion_uplift_m.max()), 300.)
        self.assertEqual(s.process_totals['arc_added_km2'], 0.)

    def test_scar_identity_and_tangent_follow_advection_then_survive_welding(self):
        s = compressional_world()
        scar = s.rift_id >= 0
        expected = _rotate(s.rift_tangent[scar], s.omega[s.parcel_plate[scar]]*2.)
        ids, birth, extension, patches, mass = (getattr(s, name).copy() for name in
            ('rift_id', 'rift_birth_myr', 'rift_extension_m', 'parcel_patch', 'mass'))
        s._advect(2.)
        np.testing.assert_allclose(s.rift_tangent[scar], expected, atol=1e-12)
        np.testing.assert_allclose(np.sum(s.rift_tangent[scar]*s.pos[scar], axis=1), 0., atol=1e-10)
        s._rasterize()
        s._weld(0, 1)
        for name, value in zip(('rift_id', 'rift_birth_myr', 'rift_extension_m', 'parcel_patch', 'mass'),
                               (ids, birth, extension, patches, mass)):
            np.testing.assert_array_equal(getattr(s, name), value, err_msg=name)
        self.assertTrue(np.all(s.parcel_plate[scar] == 1))

    def test_observed_opening_records_actual_extension_before_compressional_inversion(self):
        s = compressional_world(scar=False)
        # This experiment accounts only for observed boundary opening followed
        # by inversion, not additional interior basins away from that boundary.
        s.config['rift_strength'] = 0.
        s.omega *= -1
        s._boundaries()
        s.bcode[s.bcode != 5] = 3
        self.assertTrue(np.any(s.bcode == 5))
        mass, initial_relief = s.mass.copy(), s.trace_relief_m.copy()
        s.t += 2.
        s._deform_and_accrete(2.)
        extended = s.trace_rift_extension_m > 0
        self.assertTrue(np.any(extended))
        self.assertTrue(np.all(s.trace_rift_id[extended] > 0))
        np.testing.assert_allclose(s.trace_rift_extension_m, s.trace_extension_m, atol=1e-10)
        # Opening now also records temporary thermal support. Mechanical rift
        # lowering remains explicit, while the full height ledger includes it.
        np.testing.assert_allclose(s.trace_relief_m-initial_relief,
                                   s.trace_uplift_m-s.trace_extension_m, atol=1e-10)
        self.assertFalse(np.any(s.trace_inversion_uplift_m))
        recorded_ids, extension = s.trace_rift_id.copy(), s.trace_rift_extension_m.copy()
        s.omega *= -1
        s._boundaries()
        s.bcode[s.bcode != 4] = 3
        s.t += 2.
        s._deform_and_accrete(2.)
        self.assertTrue(np.any(s.trace_inversion_uplift_m[extended] > 0))
        np.testing.assert_array_equal(s.trace_rift_id, recorded_ids)
        np.testing.assert_array_equal(s.trace_rift_extension_m, extension)
        np.testing.assert_array_equal(s.mass, mass)
        self.assertTrue(any(event['type'] == 'rift_inversion' for event in s.events))

    def test_real_later_split_records_250_m_once_and_retains_earlier_scar_identity(self):
        s = compressional_world()
        old_ids, old_birth = s.rift_id.copy(), s.rift_birth_myr.copy()
        old_relief, old_extension = s.relief.copy(), s.rift_extension_m.copy()
        old_trace, old_trace_extension = s.trace_relief_m.copy(), s.trace_rift_extension_m.copy()
        old_mass, old_patch = s.mass.copy(), s.parcel_patch.copy()
        self.assertTrue(s._split(0), 'The controlled uncratonic parent must accept a physical fracture')
        added = s.rift_extension_m-old_extension
        self.assertTrue(np.any(np.isclose(added, 250., atol=1e-9, rtol=0)))
        self.assertTrue(np.all(np.isclose(added, 0., atol=1e-9, rtol=0)
                               | np.isclose(added, 250., atol=1e-9, rtol=0)))
        np.testing.assert_allclose(s.relief-old_relief, -added, atol=1e-10)
        np.testing.assert_allclose(s.trace_relief_m-old_trace,
                                   -(s.trace_rift_extension_m-old_trace_extension), atol=1e-10)
        old = old_ids > 0
        np.testing.assert_array_equal(s.rift_id[old], old_ids[old])
        np.testing.assert_array_equal(s.rift_birth_myr[old], old_birth[old])
        np.testing.assert_array_equal(s.mass, old_mass)
        np.testing.assert_array_equal(s.parcel_patch, old_patch)

    def test_new_arc_material_and_coalesced_arcs_have_no_continental_rift_history(self):
        s = Simulation(dict(width=64, height=32, plate_count=4),
                       dict(width=64, height=32, crust=np.zeros(64*32, np.uint8)))
        s.steps = 1
        s._add_arc_crust(np.array([1000, 1001]), np.array([100., 200.]))
        self.assertEqual(len(s.mass), 2)
        ids = s.trace_id.copy()
        s.pos[1] = s.pos[0]
        mass = float(s.mass.sum())
        s._coalesce_arcs()
        self.assertEqual(len(s.mass), 1)
        self.assertAlmostEqual(float(s.mass.sum()), mass)
        np.testing.assert_array_equal(s.trace_id, ids)
        for prefix in ('', 'trace_'):
            self.assertTrue(np.all(getattr(s, prefix+'rift_id') == -1))
            self.assertTrue(np.all(getattr(s, prefix+'rift_birth_myr') == -1))
            self.assertFalse(np.any(getattr(s, prefix+'rift_tangent')))
            self.assertFalse(np.any(getattr(s, prefix+'rift_extension_m')))
            self.assertFalse(np.any(getattr(s, prefix+'inversion_uplift_m')))

    def test_initial_juvenile_arc_crust_does_not_receive_continental_scar_identity(self):
        s = Simulation(dict(width=64, height=32, plate_count=4),
                       dict(width=64, height=32, crust=np.full(64*32, 3, np.uint8)))
        self.assertTrue(s.initial_fractures, 'Fixture must exercise accepted initial cuts')
        self.assertTrue(np.all(s.kind == 3))
        self.assertTrue(np.all(s.rift_id == -1))
        self.assertTrue(np.all(s.trace_rift_id == -1))
        self.assertFalse(np.any(s.rift_tangent))

    def test_inverting_rift_checkpoint_preserves_exact_normal_step_continuation(self):
        from server import SimulationManager
        s = compressional_world()
        s.t += 2.
        s._deform_and_accrete(2.)
        s._rasterize()
        self.assertTrue(np.any(s.inversion_uplift_m > 0))
        scratch = Path(__file__).resolve().parents[1]/'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path = Path(directory)/'checkpoint.npz'
            compatibility = SimulationManager.compatibility()
            write_checkpoint(path, s, dict(config=s.config, time_myr=s.t), compatibility)
            restored, _ = read_checkpoint(path, compatibility, Simulation)
        self.assertEqual(restored.rift_records, s.rift_records)
        for prefix in ('', 'trace_'):
            for name in ('rift_id', 'rift_birth_myr', 'rift_tangent', 'rift_extension_m', 'inversion_uplift_m'):
                np.testing.assert_array_equal(getattr(restored, prefix+name), getattr(s, prefix+name))
        s.step(2.)
        restored.step(2.)
        expected, actual = s.snapshot(), restored.snapshot()
        for key, value in expected.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(actual[key], value, err_msg=key)
            else:
                self.assertEqual(actual[key], value, key)

    def test_onset_and_quiet_events_follow_real_compression_without_erasing_scar(self):
        s = compressional_world()
        saved_omega = s.omega.copy()
        s.t += 2.
        s._deform_and_accrete(2.)
        self.assertEqual(sum(e['type'] == 'rift_inversion' for e in s.events), 1)
        active_frame = s.snapshot()
        ids, budget = s.rift_id.copy(), s.rift_extension_m.copy()
        consumed = s.inversion_uplift_m.copy()
        s.omega[:] = 0
        s._boundaries()
        for _ in range(5):
            s.t += 2.
            s._deform_and_accrete(2.)
        self.assertEqual(sum(e['type'] == 'rift_inversion_quiet' for e in s.events), 1)
        quiet = next(e for e in s.events if e['type'] == 'rift_inversion_quiet')
        self.assertIn(1, quiet['plate_uids'],
                      'Host-plate history filtering must retain the quiet phase')
        from history import read_history
        frames = [active_frame, s.snapshot()]
        scratch = Path(__file__).resolve().parents[1]/'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path = Path(directory)
            for index, frame in enumerate(frames):
                arrays = {key: value for key, value in frame.items() if isinstance(value, np.ndarray)}
                metadata = {key: value for key, value in frame.items() if key not in arrays}
                np.savez_compressed(path/f'frame_{index:04d}.npz', **arrays)
                (path/f'frame_{index:04d}.json').write_text(json.dumps(metadata))
            marker = int(np.flatnonzero(s.trace_rift_id == 42)[0])
            cell = int(s._indices(s.trace_xyz[[marker]])[0])
            manifest = dict(run_id='quiet-inversion', frame_count=2,
                            frames=[dict(index=i, time_myr=f['time_myr']) for i, f in enumerate(frames)])
            regional = read_history(path, manifest, 1, cell)
            self.assertTrue(any(e['type'] == 'rift_inversion_quiet' for e in regional['events']))
        np.testing.assert_array_equal(s.inversion_uplift_m, consumed)
        np.testing.assert_array_equal(s.rift_id, ids)
        np.testing.assert_array_equal(s.rift_extension_m, budget)
        s.omega[:] = saved_omega
        s._boundaries()
        s.bcode[s.bcode != 4] = 3
        s.t += 2.
        s._deform_and_accrete(2.)
        self.assertEqual(sum(e['type'] == 'rift_inversion' for e in s.events), 2)
        self.assertTrue(np.any(s.inversion_uplift_m > consumed))
        self.assertTrue(np.all(s.inversion_uplift_m <= np.minimum(2000., 3*s.rift_extension_m)))

    def test_gospl_geometric_forcing_counts_inversion_uplift_only_once(self):
        from gospl_export import icosphere, interval_forcing
        from tests.test_gospl_export import synthetic_frame
        # An independently prescribed +100 m inversion and -20 m relaxation
        # leaves +80 m in the saved surface. The subset label adds no new term.
        first = synthetic_frame(height=1000., loss=500.)
        second = synthetic_frame(2., height=1080., loss=520., turn=1)
        first['trace_inversion_uplift_m'] = np.zeros(len(first['trace_id']))
        second['trace_inversion_uplift_m'] = np.full(len(second['trace_id']), 100.)
        first['trace_uplift_m'] = first['trace_inversion_uplift_m'].copy()
        second['trace_uplift_m'] = second['trace_inversion_uplift_m'].copy()
        vertices, _ = icosphere(1)
        fields, _ = interval_forcing(vertices, first, second, dict(dt_myr=2., erosion=1.))
        np.testing.assert_allclose(fields['geometric_rate']*2_000_000., 80., atol=1e-5)
        np.testing.assert_allclose(fields['relaxation_correction']*2_000_000., 20., atol=1e-5)
        np.testing.assert_allclose(fields['upsub']*2_000_000., 100., atol=1e-5)


if __name__ == '__main__':
    unittest.main()
