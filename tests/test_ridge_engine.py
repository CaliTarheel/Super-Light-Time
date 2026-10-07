"""A resolved three-plate junction drives finite, accounted ridge episodes."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from checkpoint import read_checkpoint, write_checkpoint
from plate_domains import PersistentDomainTracker
from ridge_interaction import detect_ridge_trench_contacts, sample_ridge_effects
from raster_engine import Simulation


def prescribed_world(width=128, erosion=0.):
    """Actual T-junction: two incoming ocean plates, one continental overrider.

    Boundary geometry/classification is real. The returned simulation still
    has its normal methods; callers must explicitly freeze forces/topology if
    their experiment needs prescribed motions rather than full evolution.
    """
    height = width//2
    longitude = (np.arange(width)+.5)*2*np.pi/width-np.pi
    crust = np.tile((longitude > 0).astype(np.uint8), height)
    s = Simulation(dict(width=width, height=height, seed=9, plate_count=4,
                        duration_myr=64, dt_myr=2, snapshot_myr=2, erosion=erosion),
                   dict(width=width, height=height, crust=crust))
    s.plate = np.where(s.lon.ravel() > 0, 2, np.where(s.lat.ravel() > 0, 0, 1)).astype(np.int16)
    s.active[:] = False
    s.active[:3] = True
    s.count = 3
    s.plate_uid[:] = 0
    s.plate_uid[:3] = [1, 2, 3]
    s.next_plate_uid = 4
    s.names[:3] = ['Northern incoming plate', 'Southern incoming plate', 'Continental overrider']
    s.omega[:] = 0
    s.omega[0] = [0., -.002, .005]
    s.omega[1] = [0., .002, .005]
    s.mantle[:] = 0
    s.support[:] = 0
    s.support[s.plate, np.arange(s.n)] = 1
    s.parcel_plate[:] = 2
    s.trace_plate[:] = 2
    s.age[:] = 2.
    s.polarity[:] = -1
    s.events, s.event_keys = [], set()
    s.process_totals = {key: 0. for key in s.process_totals}
    s.ridge_episodes, s.next_ridge_episode_id = [], 1
    s._rasterize()
    s._boundaries()
    s.domain_tracker = PersistentDomainTracker(s.w, s.h)
    s._update_surface_domains()
    return s


class RidgeEngineTests(unittest.TestCase):
    def test_actual_boundaries_supply_evidence_and_start_a_single_local_episode(self):
        s = prescribed_world()
        contacts = detect_ridge_trench_contacts(s)
        self.assertTrue(contacts, 'The real grid must resolve an approaching ridge/trench junction')
        for contact in contacts:
            evidence = contact['evidence']
            ridge, trench = evidence['ridge_boundary_index'], evidence['trench_boundary_index']
            self.assertEqual(s.bcode[ridge], 1)
            self.assertEqual(s.bcode[trench], 2)
            self.assertEqual(contact['incoming_plate_uids'], [1, 2])
            self.assertEqual(contact['overriding_plate_uid'], 3)
            self.assertGreater(evidence['axis_approach_km_myr'], .5)
            self.assertLessEqual(evidence['ridge_trench_distance_km'], 350.)
        s._update_ridge_windows(0.)
        self.assertEqual(len(s.ridge_episodes), 1)
        self.assertEqual(s.process_totals['ridge_subduction_events'], 1)
        self.assertEqual([event['type'] for event in s.events], ['ridge_subduction'])
        s._update_ridge_windows(0.)
        self.assertEqual(len(s.events), 1, 'Continuous contact must not retrigger or refill a pulse')

    def test_thermal_peak_fades_while_finite_volcanic_relief_and_history_remain(self):
        s = prescribed_world()
        s._update_ridge_windows(0.)
        original_mass, original_patch = s.mass.copy(), s.parcel_patch.copy()
        original_relief, original_trace = s.relief.copy(), s.trace_relief_m.copy()
        # Isolate the episode after genuine detection. Hold geometry stationary
        # and disable ordinary boundary deformation, not the pulse processes.
        s.bcode[:] = 3
        # Relative prescribed boundary velocities still drive the independent
        # interior mechanics, even when the boundary class is held transform.
        # Disable that separate process for this isolated ridge-pulse budget.
        s.config['rift_strength'] = 0.
        frames = {0: s.snapshot()}
        for epoch in range(2, 41, 2):
            s.t, s.steps = float(epoch), epoch//2
            s._update_ridge_windows(2.)
            s._deform_and_accrete(2.)
            s._rasterize()
            if epoch in (8, 16, 32, 40):
                frames[epoch] = s.snapshot()
                with patch.object(s, 'ridge_episodes', []):
                    without_thermal = s.snapshot()['elevation']
                expected = sample_ridge_effects(s.ridge_episodes, s.xyz,
                                                s.plate_uid[s.plate], s.t)['thermal_support_m']
                np.testing.assert_allclose(frames[epoch]['elevation']-without_thermal,
                                           expected, atol=.001)
        self.assertEqual(frames[0]['ridge_episodes'][0]['phase'], 'heating')
        self.assertEqual(frames[8]['ridge_episodes'][0]['phase'], 'cooling')
        self.assertGreater(float(frames[8]['trace_ridge_thermal_m'].max()), 300.)
        self.assertGreater(float(frames[8]['trace_ridge_thermal_m'].max()),
                           float(frames[16]['trace_ridge_thermal_m'].max()))
        self.assertEqual(frames[32]['ridge_episodes'], [])
        self.assertFalse(np.any(frames[40]['trace_ridge_thermal_m']))
        np.testing.assert_array_equal(s.mass, original_mass)
        np.testing.assert_array_equal(s.parcel_patch, original_patch)
        self.assertGreater(float(np.max(s.relief-original_relief)), 100.)
        self.assertLessEqual(float(np.max(s.relief-original_relief)), 200.)
        np.testing.assert_allclose(s.trace_relief_m-original_trace, s.trace_uplift_m, atol=1e-12)
        np.testing.assert_array_equal(s.trace_uplift_m, s.trace_ridge_uplift_m)
        np.testing.assert_array_equal(frames[32]['trace_ridge_uplift_m'], frames[40]['trace_ridge_uplift_m'])
        self.assertEqual(s.process_totals['arc_added_km2'], 0.)
        self.assertEqual([event['type'] for event in s.events],
                         ['ridge_subduction', 'slab_window_peak', 'slab_window_cooling'])

    def test_real_weld_and_selected_fracture_host_transfer_preserve_episode_budget(self):
        s = prescribed_world()
        s._update_ridge_windows(0.)
        original = deepcopy(s.ridge_episodes[0])
        mass, patches = s.mass.copy(), s.parcel_patch.copy()
        s._weld(2, 0)
        self.assertEqual(s.ridge_episodes[0]['overriding_plate_uid'], 1)
        self.assertEqual(s.ridge_episodes[0]['started_myr'], original['started_myr'])
        self.assertEqual(s.ridge_episodes[0]['volcanic_budget_m'], original['volcanic_budget_m'])
        np.testing.assert_array_equal(s.mass, mass)
        np.testing.assert_array_equal(s.parcel_patch, patches)
        # This is the actual transfer hook called by both fracture paths after
        # assigning a center to the new plate. An unrelated center stays put.
        other = deepcopy(s.ridge_episodes[0])
        other['id'], other['center'] = 2, s.xyz[s._indices(np.array([[-1., 0., 0.]]))[0]].tolist()
        s.ridge_episodes.append(other)
        center_cell = int(s._indices(np.asarray([original['center']]))[0])
        other_cell = int(s._indices(np.asarray([other['center']]))[0])
        s.plate_uid[3] = 44
        s.plate[center_cell], s.plate[other_cell] = 3, 0
        s._transfer_ridge_hosts(0, 3)
        self.assertEqual([row['overriding_plate_uid'] for row in s.ridge_episodes], [44, 1])
        self.assertEqual([row['volcanic_budget_m'] for row in s.ridge_episodes], [200., 200.])

    def test_new_arc_markers_start_at_zero_and_coalescing_preserves_ridge_history(self):
        s = prescribed_world()
        s._update_ridge_windows(0.)
        s.t = 8.
        import structure_engine
        structure_engine.deform(s, np.zeros(s.n), np.zeros(s.n), np.zeros(s.n), 8.)
        old_ids, old_ridge = s.trace_id.copy(), s.trace_ridge_uplift_m.copy()
        self.assertTrue(np.any(old_ridge > 0))
        cell = int(np.flatnonzero((s.plate == 0) & (s.xyz[:, 0] < -.5))[0])
        s.steps = 1
        s._add_arc_crust(np.array([cell, cell+1]), np.array([100., 200.]))
        new = ~np.isin(s.trace_id, old_ids)
        self.assertTrue(np.any(new))
        self.assertFalse(np.any(s.trace_ridge_uplift_m[new]))
        np.testing.assert_array_equal(s.trace_ridge_uplift_m[:len(old_ids)], old_ridge)
        arc = np.flatnonzero(s.kind == 3)
        s.pos[arc[1]] = s.pos[arc[0]]
        ids, counters, mass = s.trace_id.copy(), s.trace_ridge_uplift_m.copy(), float(s.mass.sum())
        s._coalesce_arcs()
        np.testing.assert_array_equal(s.trace_id, ids)
        np.testing.assert_array_equal(s.trace_ridge_uplift_m, counters)
        self.assertAlmostEqual(float(s.mass.sum()), mass)

    def test_active_episode_checkpoint_restores_exact_continuation(self):
        from server import SimulationManager
        s = prescribed_world()
        s._update_ridge_windows(0.)
        s.step(2.)
        self.assertTrue(s.ridge_episodes)
        scratch = Path(__file__).resolve().parents[1]/'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path = Path(directory)/'checkpoint.npz'
            compatibility = SimulationManager.compatibility()
            write_checkpoint(path, s, dict(config=s.config, time_myr=s.t), compatibility)
            restored, _ = read_checkpoint(path, compatibility, Simulation)
        self.assertEqual(restored.ridge_episodes, s.ridge_episodes)
        s.step(2.)
        restored.step(2.)
        expected, actual = s.snapshot(), restored.snapshot()
        for key, value in expected.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(actual[key], value, err_msg=key)
            else:
                self.assertEqual(actual[key], value, key)


if __name__ == '__main__':
    unittest.main()
