"""Contact time cannot be discarded when a constrained response fails."""
from copy import deepcopy
import unittest
from unittest import mock

import numpy as np

import bounded_gravity
import contact_response
import deforming_regions
import viscous_sheet
from material_surface import spherical_face_areas
from tests.test_deforming_regions import fixture
from tests.test_gravitational_relaxation import sphere_polygon_area


def constrained_patch():
    x, y = np.meshgrid(np.linspace(-.025, .025, 3), np.linspace(-.025, .025, 3))
    points = np.column_stack((np.ones(x.size), x.ravel(), y.ravel()))
    points /= np.linalg.norm(points, axis=1)[:, None]
    faces = np.array([(a, a+1, a+4) if k == 0 else (a, a+4, a+3)
                      for a in (0, 1, 3, 4) for k in (0, 1)])
    area = np.array([sphere_polygon_area(triangle) for triangle in points[faces]])
    velocity = np.random.default_rng(7).normal(size=points.shape)*20.
    velocity -= points*np.sum(points*velocity, axis=1)[:, None]
    return points, faces, area, velocity, np.zeros(len(points), bool)


class ContactConstraintTests(unittest.TestCase):
    def test_damped_newton_progress_keeps_physical_bounds_and_actual_stationarity(self):
        # The old line search repeatedly returned toward the infeasible
        # unconstrained velocity once a later Newton iteration needed damping.
        points, faces, area, preferred, rigid = constrained_patch()
        before = [x.copy() for x in (points, faces, area, preferred, rigid)]
        minimum, maximum = area*.85, area*1.15
        endpoint, velocity, solver, detail = contact_response.redistribute(
            points, faces, area, preferred, rigid, minimum, maximum, 8.)
        actual = np.array([sphere_polygon_area(triangle) for triangle in endpoint[faces]])
        self.assertTrue(np.all(actual >= minimum*(1-1e-12)))
        self.assertTrue(np.all(actual <= maximum*(1+1e-12)))
        triangles = endpoint[faces]
        self.assertTrue(np.all(np.einsum('ij,ij->i', triangles[:, 0],
            np.cross(triangles[:, 1], triangles[:, 2])) > 0.))
        # The column-volume update uses measured geometry, with no thickness
        # clipping and no geometry correction after the constrained solve.
        volume = area*35.
        np.testing.assert_allclose(actual*(volume/actual), volume, rtol=2e-16)
        self.assertTrue(solver['converged'])
        self.assertLessEqual(detail['sqp_iterations'], 32)
        # Completing every linearized inequality can make the full Newton
        # step feasible on this former damping regression. Damping remains a
        # search option; needing it is not part of the physical contract.
        self.assertTrue(all(0. < row['step_scale'] <= 1. for row in detail['history']))
        worst = [max(row['maximum_lower_relative_violation'], row['maximum_upper_relative_violation'])
                 for row in detail['history']]
        self.assertTrue(all(after <= max(before, bounded_gravity.FEASIBILITY_TOLERANCE)*1.0001
                            for before, after in zip(worst, worst[1:])))
        self.assertFalse(detail['speed_scaled_after_constraints'])
        self.assertFalse(detail['nonlinear_geometry_correction_applied'])
        self.assertTrue(detail['original_physical_bounds_unchanged'])
        # Cached inverse columns are accurate to the inner force tolerance,
        # which is looser than this contact's area tolerance. Symmetrizing
        # their represented Schur map left a repeatable constraint bias;
        # solving that map must resolve the physical bounds without padding.
        self.assertEqual(detail['inward_roundoff_padding_km2'],[0.]*len(detail['active_faces']))
        self.assertEqual(detail['area_roundoff_corrections'],[])
        # Independently recompute the stationarity residual from the returned
        # velocity, force and physical reaction, not the reported scalar.
        context = viscous_sheet.prepare(points, faces, area, 100.)
        rhs = viscous_sheet.apply(context, preferred)
        jacobian = bounded_gravity.endpoint_area_jacobian(points, velocity, faces, 8., 6371.)/8.
        reaction = np.zeros_like(points)
        for face, side, multiplier, norm in zip(detail['active_faces'], detail['active_sides'],
                detail['multipliers_normalized'], detail['constraint_jacobian_norm_km']):
            np.add.at(reaction, faces[face], (1 if side == 'minimum' else -1)*jacobian[face]*multiplier/norm)
        residual = viscous_sheet.apply(context, velocity)-rhs-reaction
        residual -= points*np.sum(points*residual, axis=1)[:, None]
        self.assertLess(np.linalg.norm(residual)/np.linalg.norm(rhs), 1e-8)
        self.assertLess(detail['normalized_complementarity'], 1e-8)
        for original, current in zip(before, (points, faces, area, preferred, rigid)):
            np.testing.assert_array_equal(current, original)

    def test_precise_inverses_get_a_proportionate_budget_and_fail_as_geometry_errors(self):
        # The inverses are the caller's operator at 100 times its tolerance, so
        # they need more CG iterations than the caller's own solve.
        points, faces, area, preferred, rigid = constrained_patch()
        original = viscous_sheet.solve
        budgets = []
        def record(*args, **kwargs):
            budgets.append((kwargs['iterations'], kwargs['tolerance']))
            return original(*args, **kwargs)
        with mock.patch('viscous_sheet.solve', side_effect=record):
            contact_response.redistribute(points, faces, area, preferred, rigid,
                area*.85, area*1.15, 8., iterations=300)
        self.assertTrue(budgets)
        self.assertEqual(set(budgets), {(bounded_gravity.PRECISE_ITERATION_FACTOR*300, 1e-8*.01)})
        def starve(*args, **kwargs):
            return original(*args, **dict(kwargs, iterations=1))
        with mock.patch('viscous_sheet.solve', side_effect=starve):
            with self.assertRaisesRegex(bounded_gravity.InverseSolveError,
                    'stricter internal stationarity gate: 1/1 iterations.*current geometry'):
                contact_response.redistribute(points, faces, area, preferred, rigid,
                    area*.85, area*1.15, 8.)
        # Contact callers still see an ordinary constraint failure.
        self.assertTrue(issubclass(bounded_gravity.InverseSolveError, bounded_gravity.ConstraintSolveError))

    def test_constrained_speed_bound_is_rejected_without_scaling(self):
        points, faces, area, velocity, rigid = constrained_patch()
        with self.assertRaises(bounded_gravity.SpeedBoundError):
            contact_response.redistribute(points, faces, area, velocity, rigid,
                area*.85, area*1.15, 8., max_speed_km_myr=.1)

    def test_strict_contact_failure_rejects_the_whole_uncommitted_result(self):
        mesh, omega, fronts = fixture(-90.)
        before = deepcopy(mesh)
        controls = dict(mechanics_version=1, iterations=1024, tolerance=1e-8,
            face_min_area_ratio=np.full(len(mesh['faces']), .98),
            face_max_area_ratio=np.full(len(mesh['faces']), 1.01),
            redistribute_limited_contact=True)
        with mock.patch('contact_response.redistribute', side_effect=bounded_gravity.ConstraintSolveError('forced bounded failure')):
            with self.assertRaisesRegex(deforming_regions.IncompleteContactStepError,
                    'complete coupled timestep must be rejected.*forced bounded failure'):
                deforming_regions.deform(mesh, omega, fronts, 2., require_complete_contact=True, **controls)
            # Older checkpoints retain their explicitly reported partial
            # behavior unless the reviewed profile opts into complete steps.
            legacy = deforming_regions.deform(mesh, omega, fronts, 2., **controls)
        self.assertLess(legacy['diagnostics']['accepted_residual_fraction'], 1.)
        for name, value in before.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(mesh[name], value)

    def test_strict_linear_failure_is_rejected_and_complete_response_is_accepted(self):
        mesh, omega, fronts = fixture()
        with self.assertRaisesRegex(deforming_regions.IncompleteContactStepError, 'stationarity gate'):
            deforming_regions.deform(mesh, omega, fronts, 2., iterations=1,
                tolerance=1e-13, require_complete_contact=True)
        result = deforming_regions.deform(mesh, omega, fronts, 2.,
            mechanics_version=1, iterations=1024, require_complete_contact=True,
            redistribute_limited_contact=True)
        self.assertEqual(result['diagnostics']['accepted_residual_fraction'], 1.)
        self.assertEqual(result['diagnostics']['rigid_plate_motion_fraction'], 1.)


if __name__ == '__main__':
    unittest.main()
