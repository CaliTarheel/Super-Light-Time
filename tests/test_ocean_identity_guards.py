"""Initial ocean protection follows every seeded UID, never recycled slots."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import progressive_rifting as rifts
from tests.test_stress_breakup import ring_world


class InitialOceanIdentityTests(unittest.TestCase):
    def test_all_seeded_oceans_keep_timer_protection_until_identity_changes(self):
        s = ring_world()
        s.initial_ocean_plate_uid = int(s.plate_uid[0])
        s.initial_ocean_plate_uids = s.plate_uid[:2].copy()
        s.omega[1], s.omega[2] = [0., 0., .012], [0., 0., -.012]
        s._boundaries()
        with patch.object(s, '_split'), patch.object(s, '_split_ocean'):
            s._topology(2.)
            self.assertTrue(s.stress_diagnostics[0]['initial_ocean_protected'])
            self.assertTrue(s.stress_diagnostics[1]['initial_ocean_protected'])
            self.assertFalse(s.stress_diagnostics[2]['initial_ocean_protected'])
            s.plate_uid[1] += 1000
            s._topology(2.)
            self.assertFalse(s.stress_diagnostics[1]['initial_ocean_protected'])

    def test_continental_breakthrough_guard_covers_all_initial_ocean_uids(self):
        mesh = dict(xyz=np.array([[1.,0.,0.],[0.,1.,0.]]), bases=np.array([0,1]),
                    edges=np.array([[0,1]]), owners=np.array([0,1]),
                    owner_uids=np.array([11,22]), area=np.ones(2))
        s = SimpleNamespace(active=np.array([True,True,False]), plate_uid=np.array([11,22,33]),
            initial_ocean_plate_uid=11, initial_ocean_plate_uids=np.array([11,22]),
            rift_pending=dict(reliable=True, mesh=mesh, damage=np.ones(1),
                              strain=np.ones(1), edge_extension=np.ones(1)))
        cuts = [dict(owner=p, components=[np.array([0]),np.array([1])]) for p in (0,1)]
        with patch.object(rifts.rift_material, 'refresh', return_value=mesh), \
             patch.object(rifts.rift_mesh, 'coherent_cut', return_value=cuts), \
             patch('backarc.protected_hosts', return_value=set()):
            self.assertFalse(rifts.commit(s))
        np.testing.assert_array_equal(s.plate_uid, [11,22,33])


if __name__ == '__main__':
    unittest.main()
