"""Local traction/work oracles for connected, rotating and contained contacts."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np

import collision_contacts as contacts
import material_surface
import plate_balance as balance
import weld_geometry
from tests.test_collision_force_continuity import weld_model
from tests.test_collision_local_fronts import patch, refine
from tests.test_suture_weld import collided
from ridge_geometry import rotate


def fixture(*, refine_top=False, refine_under=False, contained=False):
    top = patch([-900., 900.], [-500., 200.], 0)
    lower = patch([-650., 650.], [-100., 600.], 1)
    if contained:
        top = patch([-1100., 1100.], [-900., 900.], 0)
        lower = patch([-400., 400.], [-250., 250.], 1)
    s = collided(refine(top) if refine_top else top, refine(lower) if refine_under else lower)
    s.omega[:] = [[0., 0., 0.], [.02, 0., 0.]]
    contacts.refresh(s)
    return s


def model(s):
    s.suture_weld_coordinate_version = 1
    result = weld_model(s)
    result.stiffness = np.zeros((result.size, result.size))
    result.torque = np.zeros(result.size)
    result.hinge_coefficient = np.empty(0)
    return result


def x_from_omega(s):
    return s.omega.ravel()/(balance.CM_YR_M_S/balance.RADIUS_M*balance.SECONDS_PER_MYR)


class LocalWeldTests(unittest.TestCase):
    def test_connected_rotating_patch_cannot_cancel_local_peeling(self):
        s = fixture()
        self.assertEqual(len(s.collision_contacts[0]['local_fronts']), 1)
        old = weld_model(s)
        old_rate = old.elements[0]['rows'][0]@x_from_omega(s)
        self.assertLess(abs(float(old_rate.sum())), 1e-9)
        local = model(s)
        element = local.elements[0]
        opening = sum(weld_geometry.integrate(op, x_from_omega(s), a, b, .01)[3]
                      for op, (a, b) in zip(element['rows'][0], element['bounds']))
        self.assertGreater(opening, .01)
        _, gradient, _, _, _ = local._evaluate(x_from_omega(s), .0001)
        self.assertGreater(float(gradient@x_from_omega(s)), 1e8)

    def test_contained_sheet_has_resistance_without_inventing_motion_front(self):
        s = fixture(contained=True)
        front = s.collision_contacts[0]['local_fronts'][0]
        self.assertFalse(front['direction_resolved'])

        self.assertEqual(len(weld_model(s).weld_rows), 0)
        s.omega[1] = [0., .002, .003]
        local = model(s)
        self.assertGreater(len(local.weld_rows), 0)
        _, gradient, _, _, _ = local._evaluate(x_from_omega(s), .0001)
        self.assertGreater(float(gradient@x_from_omega(s)), 1e8)
        self.assertFalse(front['direction_resolved'])

    def test_plastic_assembly_keeps_material_weld_without_control_edges(self):
        local = model(fixture(contained=True))
        local.elements = []
        local.live = np.empty(0, int)
        local._plastic()
        self.assertGreater(len(local.weld_rows), 0)
        self.assertEqual(local.elements[0]['shape'], 'arc_opening')

    def test_internal_mesh_edges_do_not_add_work_force_or_torque(self):
        reference = model(fixture())
        x = np.array([.2, -.3, .4, -.7, .9, -.6])
        expected = reference._evaluate(x, .01)
        for top, under in ((True, False), (False, True), (True, True)):
            local = model(fixture(refine_top=top, refine_under=under))
            self.assertEqual(len(local.weld_rows), len(reference.weld_rows))
            actual = local._evaluate(x, .01)
            for value, want in zip(actual[:3], expected[:3]):
                np.testing.assert_allclose(value, want, rtol=2e-9, atol=1e-3)

    def test_equal_opposite_torque_and_zero_common_rotation(self):
        local = model(fixture())
        operator = local.elements[0]['rows'][0]
        np.testing.assert_array_equal(operator[:, :, :3]+operator[:, :, 3:], 0.)
        for spin in ([.2, -.4, .7], [-1., 2., -.1]):
            x = np.tile(spin, 2)
            _, gradient, _, _, _ = local._evaluate(x, .001)
            np.testing.assert_allclose(gradient[:3]+gradient[3:], 0., atol=1e-5)
            self.assertLess(abs(float(gradient@x)), 1e-4)

    def test_energy_derivative_hessian_and_dissipation(self):
        local = model(fixture())
        x = np.array([.15, .4, -.7, -.3, .1, .5])
        direction = np.array([.7, -.1, .2, -.5, .8, .4])
        value, gradient, hessian, _, _ = local._evaluate(x, .03)
        h = 1e-5
        plus = local._evaluate(x+h*direction, .03)
        minus = local._evaluate(x-h*direction, .03)
        self.assertAlmostEqual((plus[0]-minus[0])/(2*h)/(gradient@direction), 1., places=8)
        np.testing.assert_allclose((plus[1]-minus[1])/(2*h), hessian@direction, rtol=3e-7, atol=.1)
        self.assertGreaterEqual(np.linalg.eigvalsh(hessian).min(), -np.linalg.norm(hessian)*1e-13)
        self.assertGreater(float(gradient@x), 0.)

    def test_common_rotation_preserves_power_and_rotates_torque(self):
        s = fixture(); local = model(s)
        x = x_from_omega(s)
        value, gradient, _, _, _ = local._evaluate(x, .01)
        turn = np.array([.4, -.6, .2])
        rotated = deepcopy(s)
        rotated.material_surface['vertices'] = rotate(rotated.material_surface['vertices'], turn)
        rotated.pos = rotate(rotated.pos, turn)
        rotated.omega = rotate(rotated.omega, turn)
        material_surface.refresh_geometry(rotated.material_surface); contacts.refresh(rotated)
        next_model = model(rotated)
        next_value, next_gradient, _, _, _ = next_model._evaluate(x_from_omega(rotated), .01)
        self.assertAlmostEqual(next_value/value, 1., places=10)
        rotated_gradient = rotate(gradient.reshape(-1, 3), turn)
        np.testing.assert_allclose(next_gradient.reshape(-1, 3), rotated_gradient, rtol=2e-10, atol=1e-3)

    def test_analytic_integration_matches_independent_dense_integral(self):
        operator = np.eye(2); x = np.array([.7, -.9]); delta = .1
        a, b = -.7, 4.2
        theta = np.linspace(a, b, 200001)
        basis = np.column_stack((np.cos(theta), np.sin(theta)))
        speed = basis@x
        density = np.where(speed >= 0., 0., np.where(speed <= -delta, -speed-delta/2., speed**2/(2.*delta)))
        slope = np.clip(speed/delta, -1., 0.)
        trapezoid = np.full(len(theta), (b-a)/(len(theta)-1)); trapezoid[[0, -1]] *= .5
        actual = weld_geometry.integrate(operator, x, a, b, delta)
        self.assertAlmostEqual(actual[0], float(trapezoid@density), places=9)
        np.testing.assert_allclose(actual[1], (trapezoid*slope)@basis, atol=2e-9)
        # Splitting an arc at arbitrary geometric locations changes no work.
        pieces = [weld_geometry.integrate(operator, x, l, r, delta)
                  for l, r in zip([a, .23, 1.31], [.23, 1.31, b])]
        for index in range(4):
            np.testing.assert_allclose(sum(p[index] for p in pieces), actual[index], atol=1e-13)

    def test_closing_and_rest_have_exactly_zero_traction_and_work(self):
        for x in (np.zeros(2), np.array([1., 0.])):
            value, gradient, hessian, opening = weld_geometry.integrate(np.eye(2), x, -.5, .5, .1)
            self.assertEqual(value, 0.); self.assertEqual(opening, 0.)
            np.testing.assert_array_equal(gradient, 0.)
            np.testing.assert_array_equal(hessian, 0.)

    def test_actual_newton_solve_balances_local_resistance(self):
        local = model(fixture())
        local.stiffness = np.eye(local.size)*1e9
        local.torque = np.array([0., 2., -1., .4, -2., 1.])*1e9
        local.solve()
        gradient = local._evaluate(local.x, .01)[1]
        self.assertLessEqual(local._relative_residual(gradient), balance.FORCE_RELATIVE_TOLERANCE)

    def test_tangential_slip_is_not_counted_as_normal_opening(self):
        s = fixture(); front = s.collision_contacts[0]['local_fronts'][0]
        overlap = s._collision_overlap
        arcs = weld_geometry.arcs(s.material_surface, s.parcel_collision_sheet, overlap,
            front, s.collision_contacts[0]['top_sheet'], overlap['_front_geometry'])
        for arc in arcs:
            # Rotation about this edge's normal moves material along the edge.
            operator = np.cross(np.array([arc['basis'], arc['tangent']]), arc['normal'])
            value, gradient, _, opening = weld_geometry.integrate(operator, arc['normal'], arc['start'], arc['end'], .01)
            self.assertLess(abs(value), 1e-25)
            self.assertLess(abs(opening), 1e-15)

    def test_migration_and_checkpoint_preserve_local_law_and_material(self):
        import checkpoint
        s = fixture(); before = deepcopy(s.structure)
        history = deepcopy(s.collision_contacts)
        report = weld_geometry.upgrade(s)
        self.assertEqual(weld_geometry.upgrade(s), report)
        self.assertEqual(s.collision_contacts, history)
        for name in before: np.testing.assert_array_equal(s.structure[name], before[name])
        s.rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.npz'
            checkpoint.write_checkpoint(path, s, dict(config=s.config), {})
            restored, _ = checkpoint.read_checkpoint(path, {}, SimpleNamespace)
        self.assertEqual(restored.suture_weld_coordinate_version, 1)
        frame = contacts.snapshot_fields(restored)
        self.assertEqual(frame['suture_weld_coordinate_version'], 1)
        contacts._validate_weld_frame(frame, frame['collision_contacts'])
        for invalid in (True, 2, -1):
            with self.assertRaises(ValueError):
                contacts._validate_weld_frame(dict(frame, suture_weld_coordinate_version=invalid), frame['collision_contacts'])
        expected = model(s)._evaluate(x_from_omega(s), .01)[0]
        actual = model(restored)._evaluate(x_from_omega(restored), .01)[0]
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
