"""Material loading follows finite interfaces independently of force grouping."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np

import deforming_regions
import native_material_evolution as evolution
from ridge_geometry import rotate
from tests.test_deforming_regions import fixture, unit, RADIUS


def representations():
    surface, omega, _ = fixture(-20.)
    points = unit(np.array([[1., -.10, -.16], [1., -.035, 0.], [1., -.10, .16]]))
    first, last = points[:-1], points[1:]
    normal = unit(np.cross(first, last))
    normal *= np.where(normal[:, 1] < 0., -1., 1.)[:, None]
    length = RADIUS*np.arctan2(np.linalg.norm(np.cross(first, last), axis=1), np.sum(first*last, axis=1))
    states = []
    for grouped in (True, False):
        middle = unit(first+last)
        bn = normal
        bl = length
        if grouped:
            middle = unit(np.sum(middle*length[:, None], axis=0, keepdims=True))
            bn = unit(np.sum(normal*length[:, None], axis=0, keepdims=True))
            bl = np.array([length.sum()])
        count = len(bl)
        state = SimpleNamespace(material_surface=surface, omega=omega,
            bmid=middle, bn=bn, bl=bl, bp=np.zeros(count, int), bq=np.ones(count, int),
            native_boundary_geometry=dict(segments_start=first, segments_end=last,
                segment_normals=normal, contact_index=np.zeros(2, int) if grouped else np.arange(2)))
        state._valid_loading_edges = lambda count=count: np.ones(count, bool)
        states.append(state)
    return states


class MaterialLoadingGeometryTests(unittest.TestCase):
    def test_regrouping_force_quadrature_cannot_change_material_strain(self):
        coarse, fine = representations()
        mesh = coarse.material_surface
        old = [evolution.material_loading_boundaries(s, 0)[0] for s in (coarse, fine)]
        targets = [deforming_regions._targets(mesh['vertices'], mesh['vertex_owner'], coarse.omega,
                   front, RADIUS, 400., 1.)[0] for front in old]
        self.assertGreater(np.max(np.linalg.norm(targets[0]-targets[1], axis=1)), .1)
        boundaries = [evolution.material_loading_boundaries(s, 1)[0] for s in (coarse, fine)]
        result = [deforming_regions.deform(mesh, coarse.omega, front, 2., mechanics_version=1,
                    iterations=1024, tolerance=1e-9) for front in boundaries]
        self.assertTrue(all(r['diagnostics']['solver']['converged'] for r in result))
        for name in ('vertices', 'area_ratio', 'commanded_residual_velocity_km_myr', 'rigid_mask'):
            np.testing.assert_array_equal(result[0][name], result[1][name])

    def test_stale_parent_admission_applies_to_all_its_finite_pieces(self):
        coarse, _ = representations()
        coarse._valid_loading_edges = lambda: np.array([False])
        front, diagnostic = evolution.material_loading_boundaries(coarse, 1)
        self.assertEqual(diagnostic['invalid_segments'], 2)
        mesh = coarse.material_surface
        target, weight, _ = deforming_regions._targets(mesh['vertices'], mesh['vertex_owner'],
            coarse.omega, front, RADIUS, 400., 1.)
        np.testing.assert_array_equal(target, 0.)
        np.testing.assert_array_equal(weight, 0.)

    def test_parent_owner_reversal_and_rotation_preserve_the_physical_front(self):
        _, state = representations()
        baseline, _ = evolution.material_loading_boundaries(state, 1)
        turned = deepcopy(state)
        axis = np.array([.4, -1.2, .3])
        for name in ('segments_start', 'segments_end', 'segment_normals'):
            turned.native_boundary_geometry[name] = rotate(turned.native_boundary_geometry[name], axis)
        turned.bp, turned.bq = state.bq.copy(), state.bp.copy()
        turned.native_boundary_geometry['segment_normals'] *= -1.
        front, _ = evolution.material_loading_boundaries(turned, 1)
        np.testing.assert_allclose(front['bmid'], rotate(baseline['bmid'], axis), atol=2e-15)
        np.testing.assert_allclose(front['bn'], -rotate(baseline['bn'], axis), atol=2e-15)
        np.testing.assert_allclose(front['bl'], baseline['bl'], rtol=2e-15)
        mesh = state.material_surface
        a = deforming_regions._targets(mesh['vertices'], mesh['vertex_owner'], state.omega,
            baseline, RADIUS, 400., 1.)[0]
        b = deforming_regions._targets(rotate(mesh['vertices'], axis), mesh['vertex_owner'],
            rotate(state.omega, axis), front, RADIUS, 400., 1.)[0]
        np.testing.assert_allclose(b, rotate(a, axis), atol=1e-12)

    def test_no_segment_is_invented_for_an_unrepresented_control_contact(self):
        coarse, _ = representations()
        coarse.native_boundary_geometry = dict(segments_start=np.empty((0, 3)),
            segments_end=np.empty((0, 3)), segment_normals=np.empty((0, 3)), contact_index=np.empty(0, int))
        front, diagnostic = evolution.material_loading_boundaries(coarse, 1)
        self.assertEqual(len(front['bl']), 0)
        self.assertEqual(diagnostic['total_loading_fronts'], 0)


if __name__ == '__main__':
    unittest.main()
