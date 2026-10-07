"""Conservative detached reconstruction, exact nonpenetration and rollback."""
from copy import deepcopy
from decimal import Decimal, localcontext
from types import SimpleNamespace
import unittest

import numpy as np

import historical_geometry_repair as repair
import material_surface
import mesh_coverage
from crustal_structure import initialize_structure


def fixture():
    # Two boundary fans share A, but D has crossed the other fan's A--B edge.
    xy = np.array([[0., 0.], [4., 0.], [0., -3.], [2., -.1], [3., 2.]])
    vertices = np.column_stack((np.ones(len(xy)), xy / 6371.))
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    faces = np.array([[1, 0, 2], [3, 4, 0]], np.int64)
    surface = material_surface.initialize_surface(vertices, faces, np.zeros(2, int),
        np.ones(2, np.uint8), face_id=np.array([101, 102]))
    points = material_surface.face_centres(surface)
    tangent = np.broadcast_to([0., 1., 0.], points.shape).copy()
    tangent -= points * np.sum(points * tangent, axis=1)[:, None]
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    structure = initialize_structure(np.ones(2, np.uint8))
    return SimpleNamespace(material_surface=surface, parcel_patch=surface['face_id'].copy(),
        parcel_collision_sheet=np.full(2, 3, np.int64), parcel_plate=np.zeros(2, int),
        mass=surface['reference_area_km2'].copy(), structure=structure,
        trace_structure=deepcopy(structure), pos=points.copy(), trace_xyz=points.copy(),
        trace_patch=surface['face_id'].copy(), trace_id=np.array([501, 502]),
        trace_plate=np.zeros(2, int), rift_tangent=tangent.copy(),
        trace_rift_tangent=tangent.copy(), relief=np.array([220., 220.]),
        trace_relief_m=np.array([220., 220.]), trace_adjustment_m=np.zeros(2),
        t=146., steps=73, rng=np.random.default_rng(37),
        process_totals={'erosion': 123.}, collision_contacts=[],
        trench_systems=[{'id': 7, 'slab_retained_area_km2': 100.}])


def indices(state):
    # initialize_surface compacts the original vertex order deterministically.
    faces = state.material_surface['faces']
    return dict(vertex_id=int(faces[1, 0]), separating_edge=tuple(faces[0, :2]),
                excluded_face=0, max_displacement_km=1.)


def legacy_metric_fixture():
    # Represented coordinates of the independently diagnosed 146 Myr face7927.
    triangle = np.array([
        [0.25605841893280173, -0.8401569454219101, 0.478090360865976],
        [0.25604501616583375, -0.8401024514675464, 0.4781932880487313],
        [0.25596900779920223, -0.8401538733744679, 0.47814363542786026]])
    state = fixture()
    for name, value in list(vars(state).items()):
        if isinstance(value, np.ndarray):
            setattr(state, name, value[:1].copy())
    for name in ('structure', 'trace_structure'):
        setattr(state, name, {key: value[:1].copy() for key, value in getattr(state, name).items()})
    surface = material_surface.initialize_surface(triangle, np.array([[0, 1, 2]]),
        np.zeros(1, int), np.ones(1, np.uint8), face_id=np.array([101]))
    # Preserve the exact checkpoint vectors, including their represented norms.
    surface['vertices'] = triangle.copy()
    a, b, c = triangle
    legacy = 2*np.arctan2(abs(a@np.cross(b, c)), 1+a@b+b@c+c@a)*6371.**2
    surface['area_km2'][:] = legacy
    state.material_surface = surface
    return state


def independent_decimal_area(triangle):
    with localcontext() as context:
        context.prec = 80
        a, b, c = [[Decimal.from_float(float(v)) for v in point] for point in triangle]
        def dot(u, v):
            return sum(x*y for x, y in zip(u, v))
        cross = [b[1]*c[2]-b[2]*c[1], b[2]*c[0]-b[0]*c[2], b[0]*c[1]-b[1]*c[0]]
        return 2*np.arctan2(float(abs(dot(a, cross))), float(1+dot(a, b)+dot(b, c)+dot(c, a)))*6371.**2


class HistoricalGeometryRepairTests(unittest.TestCase):
    def test_detached_repair_closes_exact_overlap_and_preserves_inventories(self):
        original = fixture()
        before = deepcopy(original)
        state, report = repair.prepare(original, **indices(original))
        self.assertEqual(report['final_overlap_pairs'], 0)
        self.assertEqual(len(mesh_coverage.same_sheet_overlaps(
            state.material_surface['vertices'], state.material_surface['faces'],
            state.parcel_collision_sheet)['first']), 0)
        self.assertGreater(report['geometric_projection']['displacement_km'], .09)
        self.assertLess(report['geometric_projection']['displacement_km'], .11)
        np.testing.assert_array_equal(original.material_surface['vertices'], before.material_surface['vertices'])
        np.testing.assert_array_equal(original.structure['thickness_km'], before.structure['thickness_km'])
        for name in ('mass', 'parcel_patch', 'parcel_plate', 'parcel_collision_sheet',
                     'trace_id', 'trace_patch', 'trace_plate'):
            np.testing.assert_array_equal(getattr(state, name), getattr(original, name))
        for name in ('faces', 'face_id', 'face_owner', 'face_region', 'reference_area_km2'):
            np.testing.assert_array_equal(state.material_surface[name], original.material_surface[name])
        # The excluded triangle itself and all its columns are not edited.
        np.testing.assert_array_equal(state.material_surface['vertices'][state.material_surface['faces'][0]],
                                      original.material_surface['vertices'][original.material_surface['faces'][0]])
        for name in original.structure:
            np.testing.assert_array_equal(state.structure[name][0], original.structure[name][0])
        np.testing.assert_allclose(state.material_surface['area_km2'] * state.structure['thickness_km'],
            original.material_surface['area_km2'] * original.structure['thickness_km'], rtol=2e-12)
        for old, new in ((original.structure, state.structure), (original.trace_structure, state.trace_structure)):
            np.testing.assert_allclose(old['thickness_km'] * old['area_factor'],
                                       new['thickness_km'] * new['area_factor'], rtol=2e-12)
            for name in old:
                if name not in ('thickness_km', 'area_factor'):
                    np.testing.assert_array_equal(old[name], new[name])
        self.assertEqual(state.t, original.t)
        self.assertEqual(state.steps, original.steps)
        self.assertEqual(state.rng.bit_generator.state, original.rng.bit_generator.state)
        self.assertEqual(state.process_totals, original.process_totals)
        self.assertEqual(state.trench_systems, original.trench_systems)
        self.assertLess(report['trace_barycentric_max_error'], 1e-11)
        np.testing.assert_allclose(state.trace_relief_m-original.trace_relief_m,
                                   state.trace_adjustment_m-original.trace_adjustment_m)
        self.assertFalse(report['physical_evolution'])
        self.assertNotEqual(report['reconstruction_energy_change_j'], 0.)
        self.assertNotEqual(id(state), id(original))

    def test_rejects_displacement_cap_without_mutating_source(self):
        state = fixture()
        original = deepcopy(state)
        arguments = dict(indices(state), max_displacement_km=.01)
        with self.assertRaisesRegex(ValueError, 'displacement cap'):
            repair.prepare(state, **arguments)
        np.testing.assert_array_equal(state.material_surface['vertices'], original.material_surface['vertices'])
        np.testing.assert_array_equal(state.structure['thickness_km'], original.structure['thickness_km'])
        self.assertFalse(hasattr(state, 'historical_geometry_repair'))

    def test_rejects_column_bound_violation_without_clipping(self):
        state = fixture()
        state.structure['thickness_km'][1] = 75.
        state.trace_structure['thickness_km'][1] = 75.
        original = deepcopy(state)
        with self.assertRaisesRegex(ValueError, 'column state'):
            repair.prepare(state, **indices(state))
        np.testing.assert_array_equal(state.structure['thickness_km'], original.structure['thickness_km'])
        np.testing.assert_array_equal(state.material_surface['vertices'], original.material_surface['vertices'])

    def test_existing_phase_or_entry_state_cannot_be_reinterpreted(self):
        state = fixture()
        state.retained_dense_crust_version = 1
        with self.assertRaisesRegex(ValueError, 'before thermal'):
            repair.prepare(state, **indices(state))
        del state.retained_dense_crust_version
        state.continental_entry_regions = {'version': 1}
        with self.assertRaisesRegex(ValueError, 'entry energy'):
            repair.prepare(state, **indices(state))

    def test_repair_cannot_repeat_or_ignore_other_penetration(self):
        state = fixture()
        repaired, _ = repair.prepare(state, **indices(state))
        with self.assertRaisesRegex(ValueError, 'already recorded'):
            repair.prepare(repaired, **indices(repaired))
        # A duplicate of the excluded face leaves an independent penetration.
        state.material_surface['faces'] = np.vstack((state.material_surface['faces'], state.material_surface['faces'][0]))
        state.material_surface['face_id'] = np.r_[state.material_surface['face_id'], 103]
        state.parcel_patch = state.material_surface['face_id'].copy()
        state.parcel_collision_sheet = np.r_[state.parcel_collision_sheet, 3]
        with self.assertRaisesRegex(ValueError, 'genuine one-face'):
            repair.prepare(state, **indices(state))

    def test_missing_marker_identity_fails_before_commit(self):
        state = fixture()
        state.trace_patch[1] = 999
        before = state.material_surface['vertices'].copy()
        with self.assertRaisesRegex(ValueError, 'identified containing'):
            repair.prepare(state, **indices(state))
        np.testing.assert_array_equal(state.material_surface['vertices'], before)

    def test_projection_and_exact_acceptance_are_rotation_invariant(self):
        base = fixture()
        reference, reference_report = repair.prepare(base, **indices(base))
        axis = np.array([.2, -.7, .5])
        axis /= np.linalg.norm(axis)
        x, y, z = axis
        cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        angle = .731
        rotation = np.eye(3)*np.cos(angle)+(1-np.cos(angle))*np.outer(axis, axis)+np.sin(angle)*cross
        rotated = deepcopy(base)
        rotated.material_surface['vertices'] = base.material_surface['vertices'] @ rotation.T
        for name in ('pos', 'trace_xyz', 'rift_tangent', 'trace_rift_tangent'):
            setattr(rotated, name, getattr(base, name) @ rotation.T)
        result, report = repair.prepare(rotated, **indices(rotated))
        np.testing.assert_allclose(result.material_surface['vertices'],
                                   reference.material_surface['vertices'] @ rotation.T, rtol=0., atol=2e-14)
        self.assertEqual(report['final_overlap_pairs'], 0)
        self.assertAlmostEqual(report['geometric_projection']['displacement_km'],
                               reference_report['geometric_projection']['displacement_km'], places=8)

    def test_metric_normalization_matches_decimal_and_retains_inventory(self):
        import burial_depth
        original = legacy_metric_fixture()
        original_vertices = original.material_surface['vertices'].copy()
        old_area = original.material_surface['area_km2'].copy()
        state, report = repair.normalize_area_metrics(original)
        oracle = independent_decimal_area(original_vertices)
        self.assertAlmostEqual(oracle, .2050606461348463, places=15)
        np.testing.assert_allclose(state.material_surface['area_km2'], [oracle], rtol=2e-15)
        self.assertGreater(report['maximum_relative_correction'], 1e-9)
        self.assertLess(report['maximum_relative_correction'], 2e-9)
        np.testing.assert_array_equal(original.material_surface['area_km2'], old_area)
        np.testing.assert_array_equal(state.material_surface['vertices'], original_vertices)
        np.testing.assert_array_equal(state.trace_xyz, original.trace_xyz)
        np.testing.assert_array_equal(state.trace_rift_tangent, original.trace_rift_tangent)
        for name in ('mass', 'parcel_patch', 'parcel_plate', 'parcel_collision_sheet', 'trace_id', 'trace_patch'):
            np.testing.assert_array_equal(getattr(state, name), getattr(original, name))
        np.testing.assert_allclose(state.material_surface['area_km2']*state.structure['thickness_km'],
                                   old_area*original.structure['thickness_km'], rtol=2e-15)
        for old, new in ((original.structure, state.structure), (original.trace_structure, state.trace_structure)):
            np.testing.assert_allclose(old['thickness_km']*old['area_factor'],
                                      new['thickness_km']*new['area_factor'], rtol=2e-15)
            for key in old:
                if key not in ('thickness_km', 'area_factor'):
                    np.testing.assert_array_equal(old[key], new[key])
        self.assertEqual(state.steps, original.steps)
        self.assertEqual(state.t, original.t)
        self.assertEqual(state.process_totals, original.process_totals)
        self.assertEqual(state.rng.bit_generator.state, original.rng.bit_generator.state)
        triangle = state.material_surface['vertices'][state.material_surface['faces']]
        _, areas, error = burial_depth.partition_face(triangle, 0, np.array([], int),
            np.array([], int), state.material_surface['area_km2'][0], 6371.)
        self.assertLess(error, 2e-10)
        self.assertEqual(len(areas), 1)

    def test_metric_normalization_rejects_corruption_without_mutation(self):
        state = legacy_metric_fixture()
        state.material_surface['area_km2'] *= 1.000001
        before = deepcopy(state)
        with self.assertRaisesRegex(ValueError, 'legacy-roundoff bound'):
            repair.normalize_area_metrics(state)
        np.testing.assert_array_equal(state.material_surface['area_km2'], before.material_surface['area_km2'])
        np.testing.assert_array_equal(state.structure['thickness_km'], before.structure['thickness_km'])
        self.assertFalse(hasattr(state, 'historical_area_metric_normalization'))

    def test_metric_normalization_preserves_true_column_bounds(self):
        state = legacy_metric_fixture()
        state.structure['thickness_km'][:] = 75.
        with self.assertRaisesRegex(ValueError, 'column state'):
            repair.normalize_area_metrics(state)
        self.assertEqual(state.structure['thickness_km'][0], 75.)

    def test_metric_normalization_rejects_repetition_and_phase_state(self):
        state = legacy_metric_fixture()
        normalized, _ = repair.normalize_area_metrics(state)
        with self.assertRaisesRegex(ValueError, 'already normalized'):
            repair.normalize_area_metrics(normalized)
        state.retained_dense_crust_version = 1
        with self.assertRaisesRegex(ValueError, 'before thermal'):
            repair.normalize_area_metrics(state)


if __name__ == '__main__':
    unittest.main()
