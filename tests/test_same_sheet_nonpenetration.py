"""Exact local regression for transient same-sheet material penetration."""
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import deforming_regions
import material_surface
import mesh_coverage
import native_material_evolution as evolution
from crustal_structure import initialize_structure


FAN_VERTICES = np.array([
    [0.9509694911738107, -0.17362529745308053, 0.2559517199413844],
    [0.9387545571258786, -0.19547686686809657, 0.2837757494833344],
    [0.9387137328645062, -0.19559238619285263, 0.2838311931324157],
    [0.9470019224210534, -0.17166860295370667, 0.27150920737748824],
    [0.9404226024506100, -0.18159629464478990, 0.2874510646550202],
    [0.9448730988401854, -0.18471034727685345, 0.2703644109290447],
    [0.9450092050801574, -0.18502794796312996, 0.2696706524379739],
])
FAN_FACES = np.array([
    [0, 3, 5],
    [1, 5, 4],
    [3, 4, 5],
    [0, 5, 6],
    [2, 6, 5],
], dtype=np.int64)
OVERLAP_FACES = np.array([1, 4])
OVERLAP_VERTICES = np.array([1, 2, 4, 5, 6])


def prospective_overlap():
    points = FAN_VERTICES.copy()
    points[1] = 2*points[2]-points[1]
    points[1] /= np.linalg.norm(points[1])
    return points


def rotation(angle=.731, axis=(.2, -.7, .5)):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return (np.eye(3)*np.cos(angle)+(1-np.cos(angle))*np.outer(axis, axis)
            +np.sin(angle)*cross)


def tiny_triangle(scale=2e-7):
    centre = np.array([.3, -.4, .8])
    centre /= np.linalg.norm(centre)
    first = np.cross(centre, [0., 0., 1.])
    first /= np.linalg.norm(first)
    second = np.cross(centre, first)
    points = centre+np.array([[0., 0.], [scale, 0.], [0., scale]])@np.stack((first, second))
    return points/np.linalg.norm(points, axis=1)[:, None]


def fan_state():
    kinds = np.ones(len(FAN_FACES), np.uint8)
    ids = 7100+np.arange(len(FAN_FACES), dtype=np.int64)
    mesh = material_surface.initialize_surface(FAN_VERTICES, FAN_FACES,
        np.zeros(len(FAN_FACES), int), kinds, face_id=ids)
    centres = material_surface.face_centres(mesh)
    tangent = np.broadcast_to([0., 0., 1.], centres.shape).copy()
    tangent -= centres*np.sum(tangent*centres, axis=1)[:, None]
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    return SimpleNamespace(material_surface=mesh, parcel_patch=ids.copy(),
        parcel_collision_sheet=np.full(len(FAN_FACES), 31, np.int64),
        mass=mesh['reference_area_km2'].copy(), parcel_plate=np.zeros(len(FAN_FACES), int),
        pos=centres.copy(), structure=initialize_structure(kinds),
        config=dict(deforming_regions=1, deformation_width_km=400.),
        material_mechanics_version=0, same_sheet_nonpenetration_version=1,
        omega=np.zeros((1, 3)), t=0.,
        collision_contacts=[], next_collision_contact_id=1,
        rift_tangent=tangent.copy(), trace_xyz=centres.copy(), trace_patch=ids.copy(),
        trace_rift_tangent=tangent.copy())


def deformation_result(surface, vertices, protected):
    area = material_surface.spherical_face_areas(vertices, surface['faces'], surface['radius_km'])
    strain = np.log(area/surface['area_km2'])
    rigid = np.zeros(len(vertices), bool) if protected is None else np.asarray(protected, bool).copy()
    return dict(vertices=vertices.copy(), areal_strain=strain,
        rigid_mask=rigid, region_weight=np.zeros(len(vertices)),
        source_boundary=np.full(len(vertices), -1, np.int32),
        limited_face_mask=np.zeros(len(surface['faces']), bool),
        diagnostics=dict(model='mocked exact fan response', solver=dict(converged=True),
            minimum_area_ratio=float((area/surface['area_km2']).min()),
            maximum_area_ratio=float((area/surface['area_km2']).max())))


def inventory(state):
    mesh = state.material_surface
    return dict(faces=mesh['faces'].copy(), face_id=mesh['face_id'].copy(),
        reference_area_km2=mesh['reference_area_km2'].copy(),
        parcel_patch=state.parcel_patch.copy(), mass=state.mass.copy(),
        structure={name: value.copy() for name, value in state.structure.items()})


class SameSheetOverlapOracleTests(unittest.TestCase):
    def test_exact_fan_detects_only_nonedge_overlap_and_respects_sheet_identity(self):
        sheets = np.full(len(FAN_FACES), 31, np.int64)
        source = mesh_coverage.same_sheet_overlaps(FAN_VERTICES, FAN_FACES, sheets)
        self.assertEqual(len(source['first']), 0)
        self.assertEqual(len(source['area_km2']), 0)

        points = prospective_overlap()
        result = mesh_coverage.same_sheet_overlaps(points, FAN_FACES, sheets)
        np.testing.assert_array_equal(result['first'], [1])
        np.testing.assert_array_equal(result['second'], [4])
        np.testing.assert_allclose(result['area_km2'], [18.83873686224433], rtol=2e-12)
        self.assertEqual(len(np.intersect1d(FAN_FACES[1], FAN_FACES[4])), 1)

        sheets[4] = 32
        separate = mesh_coverage.same_sheet_overlaps(points, FAN_FACES, sheets)
        self.assertEqual(len(separate['first']), 0)
        self.assertEqual(len(separate['area_km2']), 0)

    def test_positive_area_edge_pair_and_duplicate_are_not_topology_exempt(self):
        points = prospective_overlap()
        points = np.vstack((points, points[4]))
        faces = np.array([[1, 5, 4], [1, 5, 7]], dtype=np.int64)
        # This same-directed shared edge is a positive-area coincidence rather
        # than an ordinary oppositely directed manifold neighbour.  Topology
        # alone must not exempt it from the geometric invariant.
        geometric = mesh_coverage.material_overlaps(points, faces, np.array([1, 2]))
        self.assertGreater(geometric['area_km2'][0], 4000.)
        guarded = mesh_coverage.same_sheet_overlaps(points, faces, np.array([9, 9]))
        np.testing.assert_array_equal(guarded['first'], [0])
        np.testing.assert_array_equal(guarded['second'], [1])
        np.testing.assert_allclose(guarded['area_km2'], geometric['area_km2'])

        # Two distinct face records sharing all three indices are duplicates,
        # not an ordinary edge-neighbour pair, and remain reportable.
        duplicates = np.array([[1, 5, 4], [5, 4, 1]], dtype=np.int64)
        repeated = mesh_coverage.same_sheet_overlaps(points, duplicates, np.array([9, 9]))
        np.testing.assert_array_equal(repeated['first'], [0])
        np.testing.assert_array_equal(repeated['second'], [1])
        self.assertGreater(repeated['area_km2'][0], 4000.)

    def test_raw_positive_rule_detects_a_sub_square_metre_face_overlap(self):
        triangle = tiny_triangle()
        points = np.vstack((triangle, triangle))
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        result = mesh_coverage.same_sheet_overlaps(points, faces, np.array([4, 4]))
        np.testing.assert_array_equal(result['first'], [0])
        np.testing.assert_array_equal(result['second'], [1])
        self.assertGreater(result['area_km2'][0], 0.)
        self.assertLess(result['area_km2'][0], 1e-6)

    def test_rotation_face_order_corner_cycle_and_vertex_relabel_are_deterministic(self):
        points = prospective_overlap()
        sheets = np.full(len(FAN_FACES), 31, np.int64)
        reference = mesh_coverage.same_sheet_overlaps(points, FAN_FACES, sheets)

        face_order = np.array([4, 2, 0, 3, 1])
        vertex_order = np.array([6, 2, 5, 0, 4, 1, 3])
        inverse = np.empty(len(vertex_order), int)
        inverse[vertex_order] = np.arange(len(vertex_order))
        faces = inverse[FAN_FACES[face_order]]
        for row, shift in enumerate([1, 2, 0, 1, 2]):
            faces[row] = np.roll(faces[row], shift)
        moved = points[vertex_order]@rotation().T
        result = mesh_coverage.same_sheet_overlaps(moved, faces, sheets[face_order])
        mapped = np.sort(np.column_stack((face_order[result['first']],
                                          face_order[result['second']])), axis=1)
        np.testing.assert_array_equal(mapped, [OVERLAP_FACES])
        np.testing.assert_allclose(result['area_km2'], reference['area_km2'], rtol=2e-12, atol=1e-10)


class SameSheetProtectedRerunTests(unittest.TestCase):
    @staticmethod
    def prepared_loading():
        return ({}, dict(representation='empty exact-fan test loading'))

    def assert_inventory_equal(self, state, before):
        after = inventory(state)
        for name in ('faces', 'face_id', 'reference_area_km2', 'parcel_patch', 'mass'):
            np.testing.assert_array_equal(after[name], before[name])
        self.assertEqual(set(after['structure']), set(before['structure']))
        for name in before['structure']:
            np.testing.assert_array_equal(after['structure'][name], before['structure'][name])

    def test_historical_checkpoint_does_not_claim_a_zero_overlap_postcondition(self):
        state = fan_state()
        state.same_sheet_nonpenetration_version = 0
        safe = state.material_surface['vertices'].copy()

        def deform(surface, omega, boundaries, dt, **kwargs):
            return deformation_result(surface, safe, kwargs.get('vertex_protected'))

        with mock.patch('deforming_regions.deform', side_effect=deform), \
                mock.patch('mesh_coverage.same_sheet_overlaps', side_effect=AssertionError('legacy guard ran')):
            evolution.advect(state, 1., prepared_loading=self.prepared_loading())

        guard = state.deformation_diagnostics['same_sheet_nonpenetration']
        self.assertEqual(guard['version'], 0)
        self.assertIsNone(guard['final_overlap_pairs'])
        self.assertIn('not resolved', guard['limitation'])

    def test_overlap_expands_protection_and_clean_rerun_commits_without_inventory_change(self):
        state = fan_state()
        before = inventory(state)
        safe = state.material_surface['vertices'].copy()
        overlapping = prospective_overlap()
        protected_calls = []

        def deform(surface, omega, boundaries, dt, **kwargs):
            protected = kwargs.get('vertex_protected')
            protected = (np.zeros(len(safe), bool) if protected is None
                         else np.asarray(protected, bool).copy())
            protected_calls.append(protected)
            resolved = np.all(protected[OVERLAP_VERTICES])
            return deformation_result(surface, safe if resolved else overlapping, protected)

        with mock.patch('deforming_regions.deform', side_effect=deform):
            evolution.advect(state, 1., prepared_loading=self.prepared_loading())

        self.assertEqual(len(protected_calls), 2)
        np.testing.assert_array_equal(protected_calls[0], False)
        np.testing.assert_array_equal(np.flatnonzero(protected_calls[1]), OVERLAP_VERTICES)
        np.testing.assert_allclose(state.material_surface['vertices'], safe, rtol=0., atol=0.)
        self.assert_inventory_equal(state, before)
        guard = state.deformation_diagnostics['same_sheet_nonpenetration']
        self.assertEqual(guard['passes'], 2)
        self.assertEqual(guard['reruns'], 1)
        self.assertEqual(guard['final_overlap_pairs'], 0)
        self.assertEqual(len(guard['events']), 1)
        np.testing.assert_array_equal(guard['events'][0]['faces'], OVERLAP_FACES)
        np.testing.assert_array_equal(guard['events'][0]['vertices'], OVERLAP_VERTICES)

    def test_overlap_that_survives_the_explicit_protection_bound_fails_closed(self):
        state = fan_state()
        before = inventory(state)
        start = state.material_surface['vertices'].copy()
        overlapping = prospective_overlap()
        protected_calls = []

        def deform(surface, omega, boundaries, dt, **kwargs):
            protected = kwargs.get('vertex_protected')
            protected = (np.zeros(len(start), bool) if protected is None
                         else np.asarray(protected, bool).copy())
            protected_calls.append(protected)
            return deformation_result(surface, overlapping, protected)

        with mock.patch('deforming_regions.deform', side_effect=deform):
            with self.assertRaisesRegex(deforming_regions.IncompleteContactStepError,
                    'Same-sheet nonpenetration active set'):
                evolution.advect(state, 1., prepared_loading=self.prepared_loading())

        self.assertGreaterEqual(len(protected_calls), 2)
        np.testing.assert_array_equal(np.flatnonzero(protected_calls[-1]), OVERLAP_VERTICES)
        np.testing.assert_array_equal(state.material_surface['vertices'], start)
        self.assert_inventory_equal(state, before)

    def test_gravity_overlap_reruns_with_rigid_clamp_and_publishes_only_acceptance(self):
        state = fan_state()
        state.material_mechanics_version = 1
        before = inventory(state)
        safe = state.material_surface['vertices'].copy()
        overlapping = prospective_overlap()
        original_area = state.material_surface['area_km2'].copy()
        original_trace = state.trace_xyz.copy()
        original_trace_patch = state.trace_patch.copy()
        original_sheets = state.parcel_collision_sheet.copy()
        gravity_rigid = []

        def deform(surface, omega, boundaries, dt, **kwargs):
            return deformation_result(surface, safe, kwargs.get('vertex_protected'))

        def relax(points, faces, volumes, reference, sheets, dt, **kwargs):
            gravity_rigid.append(np.asarray(kwargs['rigid_mask'], bool).copy())
            vertices = overlapping if len(gravity_rigid) == 1 else safe
            return vertices.copy(), dict(completed_dt_myr=dt,
                energy_before_km4=1., energy_after_km4=.5,
                maximum_displacement_km=1.)

        class Admission:
            instances = []
            def __init__(self, simulation, volumes):
                self.profile = dict(dense_fraction=np.zeros(len(volumes)), sheet_order={})
                self.published = False
                self.__class__.instances.append(self)
            def prepare(self, points):
                return object()
            def accept(self, candidate):
                self.candidate = candidate
            def publish(self, simulation):
                self.published = True
                simulation.accepted_admission = id(self)

        with mock.patch('deforming_regions.deform', side_effect=deform), \
             mock.patch('gravitational_relaxation.reference_energy', return_value=0.), \
             mock.patch('gravitational_relaxation.relax', side_effect=relax), \
             mock.patch('column_density.options', return_value={'density_profile': {}}), \
             mock.patch('column_density.ContactAdmission', Admission):
            evolution.advect(state, .25, prepared_loading=self.prepared_loading())

        self.assertEqual(len(gravity_rigid), 2)
        np.testing.assert_array_equal(np.flatnonzero(gravity_rigid[0]), [])
        np.testing.assert_array_equal(np.flatnonzero(gravity_rigid[1]), OVERLAP_VERTICES)
        self.assertEqual(len(Admission.instances), 2)
        self.assertFalse(Admission.instances[0].published)
        self.assertTrue(Admission.instances[1].published)
        self.assertEqual(state.accepted_admission, id(Admission.instances[1]))
        np.testing.assert_array_equal(state.parcel_collision_sheet, original_sheets)
        np.testing.assert_array_equal(state.trace_patch, original_trace_patch)
        np.testing.assert_allclose(state.trace_xyz, original_trace, rtol=0., atol=2e-15)
        final_area = material_surface.spherical_face_areas(
            state.material_surface['vertices'], state.material_surface['faces'])
        np.testing.assert_allclose(state.geometric_log_area,
            np.log(final_area/original_area), rtol=0., atol=2e-15)
        self.assertLess(abs(state.material_geometry_volume_residual_km3), 2e-9)
        self.assert_inventory_equal(state, before)
        guard = state.deformation_diagnostics['same_sheet_nonpenetration']
        self.assertEqual(guard['events'][0]['stage'], 'gravity')
        np.testing.assert_array_equal(guard['events'][0]['patch_ids'],
            state.parcel_patch[OVERLAP_FACES])


if __name__ == '__main__':
    unittest.main(verbosity=2)
