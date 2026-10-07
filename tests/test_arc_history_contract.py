"""Arc identity and basal geometry survive saved and rotated histories."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import gospl_export
import mesh_history
import server
try:
    from .test_deforming_history_contract import saved_surface
except ImportError:
    from test_deforming_history_contract import saved_surface


def arc_frame():
    frame = saved_surface()
    n = len(frame['material_faces'])
    frame.update(arc_material_version=1, arc_surface_version=1,
        material_arc_id=np.arange(1, n+1, dtype=np.int64),
        material_arc_basal_m=np.linspace(-6000., -5000., n))
    return frame


class ArcHistoryContractTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.manager = server.SimulationManager(Path(temp.name)/'runs')
        self.manager.current = dict(run_id='arc-contract', frame_count=1,
            frames=[dict(time_myr=0.)], config={})
        self.manager.path().mkdir(parents=True)

    def test_arc_fields_are_persisted_exactly_and_are_scalars_under_rotation(self):
        frame = arc_frame(); self.manager.save_frame(0, frame)
        for stored in (self.manager.native_frame(0),
                       self.manager.native_frame(0, dict(yaw=42, pitch=65, roll=-17)),
                       gospl_export.load_frame(self.manager.path(), 0)):
            self.assertEqual(stored['arc_material_version'], 1)
            self.assertEqual(stored['arc_surface_version'], 1)
            for key in mesh_history.ARC_DTYPES:
                np.testing.assert_array_equal(stored[key], frame[key])
        lean = self.manager.frame(0)
        self.assertFalse(set(mesh_history.ARC_DTYPES).intersection(lean))

    def test_old_history_remains_unversioned(self):
        self.manager.save_frame(0, saved_surface())
        restored = self.manager.native_frame(0)
        self.assertFalse(set(mesh_history.ARC_DTYPES).intersection(restored))
        self.assertNotIn('arc_surface_version', restored)

    def test_incomplete_or_invalid_arc_fields_fail_before_overwrite(self):
        source = arc_frame(); self.manager.save_frame(0, source)
        target = self.manager.path()/'frame_0000.npz'; original = target.read_bytes()
        invalid = []
        for missing in ('material_arc_id', 'material_arc_basal_m', 'arc_material_version'):
            bad = deepcopy(source); del bad[missing]; invalid.append(bad)
        for key, value in (('arc_material_version', 2), ('arc_surface_version', 3),
                           ('material_arc_id', np.array([-1, 2])),
                           ('material_arc_basal_m', np.array([np.nan, -5000.]))):
            bad = deepcopy(source); bad[key] = value; invalid.append(bad)
        for bad in invalid:
            with self.assertRaises(ValueError): self.manager.save_frame(0, bad)
            self.assertEqual(target.read_bytes(), original)


if __name__ == '__main__': unittest.main()
