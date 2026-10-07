"""Area, torque and work oracles for physical adjacent-interface shear."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

import collision_interface as interface
import collision_contacts as contacts
import material_surface
import plate_balance as balance
import checkpoint
from mesh_geometry import icosphere
from ridge_geometry import rotate
from tests.test_local_weld_traction import fixture
from tests.test_collision_surface_acceptance import triangle, triangle_area, divide, unit
from tests.test_burial_depth import prepare


def independent_patch_work(polygon, omega):
    """Positive quadrature of |omega x r|^2 on projected planar triangles.

    This integrates the work itself, without the tested boundary-moment
    identity or subtraction of nearly equal second moments. Decimal computes
    only the thin triangle's geometric Jacobian from the represented inputs.
    """
    from decimal import Decimal, localcontext
    with localcontext() as context:
        context.prec = 70
        points = []
        for row in polygon:
            point = [Decimal.from_float(float(v)) for v in row]
            length = sum(v*v for v in point).sqrt()
            points.append([v/length for v in point])
        triangles = []
        a = points[0]
        for b, c in zip(points[1:-1], points[2:]):
            u, v = [[right-left for left, right in zip(a, p)] for p in (b, c)]
            cross = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
            determinant = float(abs(sum(x*y for x, y in zip(a, cross))))
            triangles.append((np.array([a, b, c], float), determinant))
    nodes, weights = np.polynomial.legendre.leggauss(16)
    nodes = (nodes+1)/2
    weights /= 2
    work = 0.
    for (a, b, c), determinant in triangles:
        for u, wu in zip(nodes, weights):
            q = a+u*(b-a)+(1-u)*nodes[:, None]*(c-a)
            norm = np.linalg.norm(q, axis=1)
            speed_squared = np.sum(np.cross(omega, q/norm[:, None])**2, axis=1)
            work += wu*float(weights@(speed_squared*determinant*(1-u)/norm**3))
    return work


def model(s, enable=True):
    if enable:
        interface.upgrade(s)
    result = balance.Balance.__new__(balance.Balance)
    result.s = s; result.plates = list(range(len(s.plate_uid)))
    result.size = 3*len(result.plates); result.slot = {plate: plate for plate in result.plates}
    result.stiffness = np.zeros((result.size, result.size)); result.notes = {}
    interface.assemble(result)
    return result


class InterfaceGeometryTests(unittest.TestCase):
    def test_production_overlap_at_six_myr_preserves_tiny_spin_resistance(self):
        # Exact represented clipping triangle from the reviewed seed-12 run.
        # The old double-precision edge cancellation made its smallest metric
        # eigenvalue -2.416e-5 m2, although the physical work is positive.
        polygon = np.array([
            [0.8468084999286877, -0.3704785799758462, 0.38165558587240195],
            [0.846809359955314, -0.3704744330573402, 0.3816577031096237],
            [0.8468029482024216, -0.37047381058571105, 0.38167253318229427]])
        radius_m = 6371000.
        metric = interface.rotation_metric(polygon, radius_m/1000.)
        self.assertGreater(np.linalg.eigvalsh(metric).min(), 0.)
        for omega in (*np.eye(3), unit(polygon.sum(axis=0))):
            expected = independent_patch_work(polygon, omega)*radius_m**2
            actual = float(omega@metric@omega)
            # Near the centroid axis, a double-precision 3x3 representation
            # subtracts O(area) terms to resolve O(area*angular_size^2) work.
            # Bound only that final representation error, not physical work.
            tolerance = 32*np.finfo(float).eps*np.linalg.norm(metric)*float(omega@omega)
            self.assertGreater(actual, 0.)
            self.assertAlmostEqual(actual, expected, delta=tolerance)
        self.assertAlmostEqual(float(np.trace(metric)/2), 1442.324496699554, delta=1e-9)

    def test_rotated_thin_patch_work_matches_positive_surface_quadrature(self):
        for width in (1e-7, 1e-10, 1e-14):
            polygon = rotate(unit([[1., -.03, 0.], [1., .03, 0.],
                                  [1., .03, width], [1., -.03, width]]), [.7, 1.2, -.3])
            metric = interface.rotation_metric(polygon, .001)
            for omega in (np.array([.2, -.7, .3]), unit(polygon.sum(axis=0)),
                          np.array([1., 0., 0.])):
                expected = independent_patch_work(polygon, omega)
                actual = float(omega@metric@omega)
                self.assertGreater(actual, 0.)
                self.assertAlmostEqual(actual/expected, 1., delta=3e-11)

    def test_precision_fallback_does_not_change_invalid_geometry_into_drag(self):
        polygon = unit([[1., -.03, 0.], [1., .03, 0.], [1., .03, 1e-14], [1., -.03, 1e-14]])
        with self.assertRaisesRegex(ValueError, 'winding'):
            interface.rotation_metric(polygon[::-1], .001)
        with self.assertRaisesRegex(ValueError, 'winding'):
            interface.rotation_metric(polygon[[0, 2, 1, 3]], .001)
        invalid = polygon.copy()
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            interface.rotation_metric(invalid, .001)

    def test_whole_sphere_metric_is_two_thirds_area_identity(self):
        mesh = icosphere(0)
        actual = sum(interface.rotation_metric(mesh['vertices'][face], 1.) for face in mesh['faces'])
        np.testing.assert_allclose(actual, np.eye(3)*(8.*np.pi/3.)*1e6, rtol=2e-15, atol=1e-8)

    def test_octant_has_independent_spherical_moment_oracle(self):
        # For x,y,z >= 0: area=pi/2, each diagonal second moment=pi/6,
        # and integral(x*y) dA=1/3, from direct spherical polar integration.
        expected = np.full((3, 3), -1./3.)
        np.fill_diagonal(expected, np.pi/3.)
        actual = interface.rotation_metric(np.eye(3), .001)  # physical radius 1 m
        np.testing.assert_allclose(actual, expected, atol=3e-16)
        self.assertTrue(np.all(np.linalg.eigvalsh(actual) > 0.))

    def test_polygon_metric_is_additive_under_refinement(self):
        parent = triangle(); vertices, faces = divide(parent)
        actual = sum(interface.rotation_metric(vertices[face], 6371.) for face in faces)
        np.testing.assert_allclose(actual, interface.rotation_metric(parent, 6371.), rtol=2e-12, atol=.01)

    def test_triple_stack_couples_only_adjacent_layers(self):
        s = prepare([(triangle(), 30., 0., False)]*3, [(1, 2), (2, 3)])
        assembled = model(s)
        rows = interface.regions(s)
        self.assertEqual({(r['top_owner'], r['under_owner']) for r in rows}, {(0, 1), (1, 2)})
        self.assertAlmostEqual(sum(r['area_km2'] for r in rows)/triangle_area(triangle()), 2., places=12)
        np.testing.assert_array_equal(assembled.stiffness[:3, 6:9], 0.)

    def test_partial_middle_sheet_screens_only_its_actual_footprint(self):
        a, b, c = triangle(); half = np.array([a, unit(a+b), c])
        s = prepare([(triangle(), 30., 0., False), (half, 30., 0., False),
                     (triangle(), 30., 0., False)], [(1, 2), (2, 3)])
        rows = interface.regions(s)
        pairs = {}
        for row in rows:
            key = row['top_owner'], row['under_owner']
            pairs[key] = pairs.get(key, 0.)+row['area_km2']
        self.assertEqual(set(pairs), {(0, 1), (1, 2), (0, 2)})
        for value in pairs.values():
            self.assertAlmostEqual(value/triangle_area(triangle()), .5, places=12)


class InterfaceMechanicsTests(unittest.TestCase):
    def test_contained_relative_spin_has_positive_work_and_zero_common_rotation(self):
        s = fixture(contained=True); assembled = model(s)
        x = np.array([0., 0., 0., 1., 0., 0.])  # spin about the centre of the patch
        force = assembled.stiffness@x
        self.assertGreater(float(x@force), 0.)
        np.testing.assert_allclose(force[:3]+force[3:], 0., atol=1e-8)
        common = np.tile([.3, -.6, .9], 2)
        np.testing.assert_allclose(assembled.stiffness@common, 0., atol=1e-6)
        self.assertGreaterEqual(np.linalg.eigvalsh(assembled.stiffness).min(), -np.linalg.norm(assembled.stiffness)*1e-14)

    def test_either_mesh_refinement_preserves_force_and_work(self):
        baseline = model(fixture(contained=True)).stiffness
        for top, under in ((True, False), (False, True), (True, True)):
            actual = model(fixture(contained=True, refine_top=top, refine_under=under)).stiffness
            np.testing.assert_allclose(actual, baseline, rtol=5e-11, atol=1e-5)

    def test_rotating_geometry_preserves_mechanical_work(self):
        s = fixture(); first = model(s)
        velocity = np.array([[.3, -.1, .6], [.1, .5, -.4]])
        expected = float(velocity.ravel()@first.stiffness@velocity.ravel())
        rotation = [.4, -.6, .2]
        s.material_surface['vertices'] = rotate(s.material_surface['vertices'], rotation)
        material_surface.refresh_geometry(s.material_surface)
        s.pos = material_surface.face_centres(s.material_surface); contacts.refresh(s)
        second = model(s); changed = rotate(velocity, rotation).ravel()
        self.assertAlmostEqual(float(changed@second.stiffness@changed)/expected, 1., places=11)

    def test_linear_shear_units_and_parameter_scaling(self):
        s = fixture(contained=True); assembled = model(s)
        metric = sum(row['metric_m2'] for row in interface.regions(s))
        expected = 1e20/13200.*balance.CM_YR_M_S**2*metric
        np.testing.assert_allclose(assembled.stiffness[:3, :3], expected, rtol=2e-15, atol=1e-6)
        s.collision_interface_parameters['viscosity_pa_s'] *= 3.
        s.collision_interface_parameters['thickness_m'] *= 2.
        changed = model(s)
        np.testing.assert_allclose(changed.stiffness, 1.5*assembled.stiffness, rtol=2e-15, atol=1e-6)

    def test_actual_viscous_assembly_runs_even_without_trench_edges(self):
        s = fixture(contained=True)
        control = icosphere(0); n = len(control['faces'])
        s.cell_area = control['area_km2']; s.xyz = control['xyz']
        s.crust = np.zeros(n, int); s.plate = np.zeros(n, int)
        s.support = np.full((2, n), .5)
        baseline = model(s, enable=False)
        for result in (baseline,):
            result.bp = np.empty(0, int); result.live = np.empty(0, int); result.trench = np.empty(0, bool)
            result._viscous()
        updated = model(s)
        updated.bp = np.empty(0, int); updated.live = np.empty(0, int); updated.trench = np.empty(0, bool)
        updated._viscous()
        np.testing.assert_allclose(updated.stiffness-baseline.stiffness, updated.interface_stiffness, rtol=3e-8, atol=1e-4)
        self.assertGreater(np.linalg.norm(updated.interface_stiffness), 0.)

    def test_explicit_migration_restart_schema_and_legacy_noop(self):
        s = fixture(); old = deepcopy(s.structure)
        untouched = model(s, enable=False)
        np.testing.assert_array_equal(untouched.stiffness, 0.)
        report = interface.upgrade(s)
        self.assertEqual(interface.upgrade(s), report)
        for name in old: np.testing.assert_array_equal(s.structure[name], old[name])
        s.rng = np.random.default_rng(8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.npz'
            checkpoint.write_checkpoint(path, s, dict(config=s.config), {})
            restored, _ = checkpoint.read_checkpoint(path, {}, SimpleNamespace)
        np.testing.assert_array_equal(model(restored).stiffness, model(s).stiffness)
        frame = dict(collision_contact_version=1, **interface.snapshot(restored))
        interface.validate_frame(frame)
        for bad in (True, 2, -1):
            with self.assertRaises(ValueError): interface.validate_frame(dict(frame, collision_interface_version=bad))
        frame['collision_interface_parameters']['thickness_m'] = 0.
        with self.assertRaises(ValueError): interface.validate_frame(frame)
        for bad in (True, 0., -1., np.nan, np.inf):
            fresh = fixture()
            with self.assertRaises(ValueError): interface.upgrade(fresh, viscosity_pa_s=bad)
            self.assertFalse(hasattr(fresh, 'collision_interface_version'))

    def test_force_solve_slows_relative_spin_and_balances_power(self):
        s = fixture(contained=True)
        coupled = model(s)
        drag = np.eye(coupled.size)*1e8  # specified positive basal resistance
        torque = np.array([1., 0., 0., -1., 0., 0.])*1e8
        uncoupled = np.linalg.solve(drag, torque)
        coupled.stiffness += drag
        coupled.torque = torque; coupled.elements = []
        coupled.hinge_coefficient = np.empty(0)
        coupled.solve()
        self.assertLess(abs(coupled.x[0]-coupled.x[3]), abs(uncoupled[0]-uncoupled[3]))
        gradient = coupled._evaluate(coupled.x, .01)[1]
        self.assertLessEqual(coupled._relative_residual(gradient), balance.FORCE_RELATIVE_TOLERANCE)
        supplied = float(torque@coupled.x)
        dissipated = float(coupled.x@drag@coupled.x+coupled.x@coupled.interface_stiffness@coupled.x)
        self.assertAlmostEqual(supplied/dissipated, 1., places=12)


if __name__ == '__main__':
    unittest.main()
