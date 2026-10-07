"""Visible-domain history follows persistent plates across midstep allocation."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from checkpoint import read_checkpoint, write_checkpoint
from plate_domains import PersistentDomainTracker
from raster_engine import Simulation, _rotate


def ocean_world():
    width, height = 48, 24
    s = Simulation(dict(width=width, height=height, plate_count=4, seed=37),
                   dict(width=width, height=height,
                        crust=np.zeros(width*height, np.uint8)))
    s.active[:] = False
    s.active[0] = True
    s.plate[:] = 0
    s.support[:] = 0.
    s.support[0] = 1.
    s.omega[:] = 0.
    s.omega[0] = [0., 0., .07]
    # A retired slot retains a very different old motion and identity.
    s.omega[1] = [0., -.12, 0.]
    s.names[0] = 'Moving parent'
    s.domain_tracker = PersistentDomainTracker(width, height)
    s._rasterize()
    s._boundaries()
    s._update_surface_domains()
    return s


class SurfaceDomainMotionTests(unittest.TestCase):
    def assert_same(self, left, right, path='state'):
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right, err_msg=path)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys(), path)
            for name in left:
                self.assert_same(left[name], right[name], path+'.'+str(name))
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right), path)
            for index, value in enumerate(left):
                self.assert_same(value, right[index], path+f'[{index}]')
        else:
            self.assertEqual(left, right, path)

    def split_during_real_step(self, *, reuse=False, grandchild=False):
        s = ocean_world()
        old_capacity = s.capacity
        old_parent_domain = int(s.domain[0])
        parent_uid = int(s.plate_uid[0])
        motion = s.omega[0].copy()
        slot = 1 if reuse else old_capacity
        old_slot_uid = int(s.plate_uid[1])
        def topology(dt):
            s._ensure_plate_capacity(slot+1+int(grandchild))
            s._new_plate_identity(slot, 0)
            s.active[slot] = True
            s.names[slot] = 'Inherited rift child'
            s.plate[s.xyz[:, 0] > 0] = slot
            s.omega[slot] = [.19, 0., 0.]
            # Parent recoil and child opening occur after material transport.
            s.omega[0] = [0., .11, 0.]
            if grandchild:
                s._new_plate_identity(slot+1, slot)
                s.active[slot+1] = True
                s.names[slot+1] = 'Same-step descendant'
                s.plate[(s.plate == slot) & (s.xyz[:, 2] > 0)] = slot+1
                s.omega[slot+1] = [0., 0., -.16]
            s.support[:] = 0.
            s.support[s.plate, np.arange(s.n)] = 1.
        with patch.object(s, '_forces'), patch.object(s, '_topology', side_effect=topology), \
             patch('raster_engine.backarc.apply_motion'), \
             patch.object(s.domain_tracker, 'update', wraps=s.domain_tracker.update) as track:
            s.step(2.)
            previous_cells = track.call_args.kwargs['previous_cells'].copy()
        expected = s._indices(_rotate(s.xyz, -motion*2.))
        np.testing.assert_array_equal(previous_cells, expected)
        self.assertTrue(np.any(expected != np.arange(s.n)), 'fixture must actually move')
        self.assertEqual(s.plate_parent_uid[slot], parent_uid)
        self.assertNotEqual(s.plate_uid[slot], old_slot_uid)
        self.assertEqual(s.capacity, old_capacity if reuse else old_capacity+8)
        for row in s.domains:
            if row['plate_id'] == 0:
                self.assertEqual(row['uid'], old_parent_domain)
                self.assertEqual(row['name'], 'Moving parent')
            else:
                self.assertEqual(row['name'], s.names[row['plate_id']])
                self.assertNotEqual(row['uid'], old_parent_domain)
                self.assertEqual(row['parent_plate_uid'], s.plate_uid[row['plate_id']])
        return s

    def test_capacity_growth_after_advection_uses_parent_motion(self):
        self.split_during_real_step()

    def test_reused_slot_cannot_inherit_retired_occupants_motion(self):
        self.split_during_real_step(reuse=True)

    def test_same_step_descendant_resolves_to_original_moving_ancestor(self):
        self.split_during_real_step(grandchild=True)

    def test_unknown_or_cyclic_ancestry_has_no_invented_correspondence(self):
        s = ocean_world()
        old_uid, old_omega = s.plate_uid.copy(), s.omega.copy()
        s.plate_uid[0] = s.next_plate_uid
        s.plate_parent_uid[0] = s.plate_uid[0]
        with patch.object(s.domain_tracker, 'update', wraps=s.domain_tracker.update) as track:
            s._update_surface_domains(2., old_omega, old_uid)
        np.testing.assert_array_equal(track.call_args.kwargs['previous_cells'], np.full(s.n, -1))

    def test_full_checkpoint_after_growth_and_exact_unpatched_continuation(self):
        s = self.split_during_real_step(grandchild=True)
        compatibility = dict(engine_sha256='surface-domain-motion-test',
                             auxiliary_sources_sha256={}, numpy_version=np.__version__)
        scratch = Path(__file__).resolve().parents[1]/'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            path = Path(folder)/'checkpoint.npz'
            write_checkpoint(path, s, dict(config=s.config, time_myr=s.t), compatibility)
            restored, _ = read_checkpoint(path, compatibility, Simulation)
        self.assert_same(s.snapshot(), restored.snapshot())
        for name in ('plate_uid', 'plate_parent_uid', 'omega', 'support'):
            self.assert_same(getattr(s, name), getattr(restored, name), name)
        for _ in range(2):
            s.step(2.)
            restored.step(2.)
            self.assert_same(s.snapshot(), restored.snapshot())
            self.assert_same(vars(s.domain_tracker), vars(restored.domain_tracker))


if __name__ == '__main__':
    unittest.main()
