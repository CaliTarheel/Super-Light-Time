"""Pin contour-based trench classification across contact ownership flips."""
import unittest

import numpy as np

import effective_subduction as effective
import effective_subduction_initiation as initiation
import trench_history
from tests.test_effective_subduction_initiation import motion, set_pieces, tick, world


def shared_trench_scene():
    """Use the finite-trench and shared-contact fixture from PR #27."""
    s = world()
    initiation.upgrade(s, dict(version=2, enabled=True))
    tick(s)
    for _ in range(10):
        tick(s)
    if not initiation.activate_pending(s):
        raise AssertionError('The finite trench fixture did not activate.')
    set_pieces(s, np.array([-.03, 0., .06]), parents=[0, 1])
    motion(s, 12., 12.)
    full = s.bl.copy()
    for key in ('ba', 'bb', 'bp', 'bq', 'bcode', 'down', 'normal_speed',
                'trench_id', 'trench_maturity'):
        setattr(s, key, np.append(getattr(s, key), getattr(s, key)[:1]))
    s.bmid = np.vstack((s.bmid, s.bmid[:1]))
    s.bn = np.vstack((s.bn, s.bn[:1]))
    s.bl = np.array([full[0]*2/3, full[1]*2/3, (full[0]+full[1])/3])
    s.native_boundary_geometry.update(
        share_segment=np.array([0, 0, 1, 1], np.int32),
        share_contact=np.array([0, 2, 1, 2], np.int32),
        share_weight=np.array([2/3, 1/3, 2/3, 1/3]),
    )
    s.contact_share_version = 1
    effective.prepare(s)
    return s


def move_contact_midpoint_off_trace(s):
    row = next(r for r in s.trench_systems if r['phase'] != 'shutdown')
    trench_id = row['id']
    owners, force, live = effective.line_state(s)
    candidates = np.flatnonzero((s.trench_id == trench_id) & live & (force > 0.))
    if not len(candidates):
        raise AssertionError('Fixture has no live trench force to test.')
    edge = int(candidates[0])
    anchors = np.asarray(row['geometry_xyz'], float)
    center = anchors[len(anchors)//2]
    angle = 130./6371.
    s.bmid[edge] = np.cos(angle)*center + np.sin(angle)*np.array([0., 1., 0.])
    return edge, trench_id, force[edge]


class ContactShareForceTransitionTests(unittest.TestCase):
    def test_v1_uses_shared_contour_when_contact_midpoint_crosses_match_radius(self):
        s = shared_trench_scene()
        edge, trench_id, before = move_contact_midpoint_off_trace(s)
        s.trench_id = trench_history._match(s)
        _, after, _ = effective.line_state(s)
        self.assertEqual(s.trench_id[edge], trench_id)
        self.assertGreater(before, 0.)
        self.assertAlmostEqual(after[edge], before, places=12)

    def test_v0_retains_historical_midpoint_classification(self):
        s = shared_trench_scene()
        s.contact_share_version = 0
        edge, trench_id, before = move_contact_midpoint_off_trace(s)
        s.trench_id = trench_history._match(s)
        _, after, _ = effective.line_state(s)
        self.assertEqual(s.trench_id[edge], 0)
        self.assertGreater(before, 0.)
        self.assertEqual(after[edge], 0.)

    def test_spherical_segment_match_uses_interior_projection(self):
        start = np.array([[np.cos(-.03), 0., np.sin(-.03)]])
        end = np.array([[np.cos(.03), 0., np.sin(.03)]])
        anchor = np.array([[1., 0., 0.]])
        np.testing.assert_allclose(trench_history._segment_nearest(start, end, anchor), [0.], atol=1e-8)


if __name__ == '__main__':
    unittest.main()
