"""Back-arc opening from actual retreat, with conserved material and history."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import backarc
from checkpoint import read_checkpoint, write_checkpoint
from plate_domains import PersistentDomainTracker
from raster_engine import Simulation


def arc_world(width=128, continental=False):
    height = width//2
    lon, lat = np.meshgrid((np.arange(width)+.5)*2*np.pi/width-np.pi,
                          np.pi/2-(np.arange(height)+.5)*np.pi/height)
    crust = np.zeros(width*height, np.uint8)
    belt = (lon.ravel() > .025) & (lon.ravel() < .075) & (np.abs(lat.ravel()) < .38)
    crust[belt] = 1 if continental else 3
    s = Simulation(dict(width=width, height=height, seed=37, plate_count=4,
                        duration_myr=160., dt_myr=2., snapshot_myr=2.),
                   dict(width=width, height=height, crust=crust))
    s.plate = (s.lon.ravel() > 0).astype(np.int16)
    s.active[:] = False
    s.active[:2] = True
    s.count = 2
    s.plate_uid[:] = 0
    s.plate_uid[:2] = [1, 2]
    s.next_plate_uid = 3
    s.names[:2] = ['Incoming ocean', 'Overriding arc margin']
    s.support[:] = 0.
    s.support[s.plate, np.arange(s.n)] = 1.
    s.parcel_plate[:] = s.trace_plate[:] = 1
    s.omega[:] = 0.
    s.omega[0] = [0., 0., .008]
    s.mantle[:] = 0.
    s.polarity[:] = -1
    s.polarity[0, 1] = s.polarity[1, 0] = 0
    s.age[:] = 100.
    s.age[s.crust > 0] = 0.
    s.events, s.event_keys = [], set()
    s.backarc_basins = []
    s.next_backarc_id = 1
    s.born[:] = 0.
    s._rasterize()
    s._boundaries()
    s.domain_tracker = PersistentDomainTracker(s.w, s.h)
    s._update_surface_domains()
    return s


def advance_prescribed(s, steps):
    # Forces are held fixed for the mechanism experiment; the production
    # back-arc correction, transport, arc growth and deformation remain active.
    with patch.object(Simulation, '_forces', lambda self, dt: None):
        for _ in range(steps):
            s.step(2.)


class BackarcEngineTests(unittest.TestCase):
    def test_retreat_loads_then_separates_arc_and_opens_real_ocean_boundary(self):
        s = arc_world(continental=True)
        original_mass = s.mass.copy()
        original_patches = s.parcel_patch.copy()
        original_count = len(s.mass)
        advance_prescribed(s, 60)
        ruptured = [r for r in s.backarc_basins if r['arc_plate_uid'] is not None]
        self.assertTrue(ruptured, [(r['phase'], r['loading_km']) for r in s.backarc_basins])
        row = ruptured[0]
        self.assertGreater(row['rupture_myr'], row['started_myr'])
        self.assertGreater(row['opening_km'], 20.)
        self.assertTrue(any(e['type'] == 'backarc_spreading' for e in s.events))
        np.testing.assert_array_equal(s.mass[:original_count], original_mass)
        np.testing.assert_array_equal(s.parcel_patch[:original_count], original_patches)
        self.assertTrue(np.isfinite(s.snapshot()['elevation']).all())
        self.assertAlmostEqual(s.mass.sum()-s.process_totals['arc_added_km2'], original_mass.sum(), delta=1e-5)
        # Every original four-sample patch still has one owner.
        order = np.argsort(s.parcel_patch)
        ids, owner = s.parcel_patch[order], s.parcel_plate[order]
        self.assertFalse(np.any((ids[1:] == ids[:-1]) & (owner[1:] != owner[:-1])))
        self.assertEqual(s.snapshot()['backarc_basins'], s.backarc_basins)

    def test_common_rotation_cannot_accumulate_retreat_loading(self):
        s = arc_world()
        s.omega[:] = [0., .001, .003]
        advance_prescribed(s, 8)
        self.assertEqual(s.backarc_basins, [])
        self.assertEqual(s.process_totals['backarc_ruptures'], 0.)

    def test_disabled_rifting_never_creates_a_basin(self):
        s = arc_world()
        s.config['rift_strength'] = 0.
        advance_prescribed(s, 8)
        self.assertEqual(s.backarc_basins, [])

    def test_loading_decays_without_a_trench_instead_of_rupturing_on_age(self):
        s = arc_world()
        advance_prescribed(s, 4)
        self.assertTrue(s.backarc_basins)
        before = s.backarc_basins[0]['loading_km']
        s.omega[:] = 0.
        advance_prescribed(s, 12)
        self.assertLess(s.backarc_basins[0]['loading_km'], before)
        self.assertIsNone(s.backarc_basins[0]['arc_plate_uid'])
        self.assertEqual(s.backarc_basins[0]['phase'], 'quiet')

    def test_active_basin_checkpoint_continues_exactly(self):
        s = arc_world()
        advance_prescribed(s, 24)
        self.assertTrue(any(r['arc_plate_uid'] is not None for r in s.backarc_basins))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.npz'
            write_checkpoint(path, s, {'config': s.config}, {})
            restored, _ = read_checkpoint(path, {}, Simulation)
        advance_prescribed(s, 3)
        advance_prescribed(restored, 3)
        self.assertEqual(s.backarc_basins, restored.backarc_basins)
        self.assertEqual(s.events, restored.events)
        for name in ('pos', 'mass', 'support', 'omega', 'trace_relief_m', 'trace_extension_m'):
            np.testing.assert_array_equal(getattr(s, name), getattr(restored, name))


if __name__ == '__main__':
    unittest.main()
