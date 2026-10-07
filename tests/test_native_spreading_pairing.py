"""Actual paired-production checks with independent rigid-strip area oracles."""
from types import SimpleNamespace
import unittest
import numpy as np
import mesh_geometry
import mesh_transport
import native_spreading
from tests._native_spreading_oracle import polygon_area, material_rectangles, RADIUS


def unit(points):
    points = np.asarray(points, float)
    return points/np.linalg.norm(points, axis=-1, keepdims=True)


def scene(*, split=1, corridor=False, extent=120., level=3):
    mesh = mesh_geometry.icosphere(level)
    n = len(mesh['faces'])
    plate = (mesh['xyz'][:, 1] >= 0.).astype(np.int16)
    edges = np.asarray(mesh['edge_faces'])
    crossing = np.flatnonzero(plate[edges[:, 0]] != plate[edges[:, 1]])
    chosen = crossing[np.argmax(mesh['edge_mid'][crossing, 0])]
    a, b = edges[chosen]
    if plate[a] != 0: a, b = b, a
    points = unit([[1., 0., z/RADIUS] for z in np.linspace(-extent, extent, split+1)])
    covered = [(-500., -30., -extent-200., extent+200.),
               (30., 500., -extent-200., extent+200.)] if corridor else []
    vertices, faces = material_rectangles(covered)
    owners = np.repeat(np.arange(len(covered)), 4).astype(np.int16)
    surface = dict(vertices=vertices, faces=faces, vertex_owner=owners)
    support = np.zeros((3, n)); support[plate, np.arange(n)] = 1.
    locator = mesh_geometry.build_locator(mesh['vertices'], mesh['faces'])
    stencil = mesh_transport.face_vertex_stencil(mesh)
    s = SimpleNamespace(n=n, native_mesh=mesh, material_surface=surface, xyz=mesh['xyz'],
        native_locator=locator, cell_area=mesh['area_km2'], plate=plate,
        ba=np.array([a]), bb=np.array([b]), bp=np.array([0]), bq=np.array([1]),
        active=np.ones(3, bool), plate_uid=np.array([10, 11, 12]), support=support,
        omega=np.array([[0., 0., -.001], [0., 0., .001], [0., 0., 0.]]),
        process_totals=dict(ocean_created_km2=0.),
        native_boundary_geometry=dict(segments_start=points[:-1], segments_end=points[1:],
            segment_normals=np.tile([0., 1., 0.], (split, 1)),
            contact_index=np.zeros(split, np.int32)))
    s._sample_coordinates = lambda at: mesh_transport.sample_coordinates(mesh, locator, stencil, at)
    return s, points[[0, -1]]


def expected_area(axis, dt):
    # Explicit rotation about z, independent of the production Euler helper.
    angle = .001*dt
    matrix = np.array([[np.cos(angle), -np.sin(angle), 0.],
                       [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    shore = axis @ matrix.T
    return 2.*polygon_area(np.array([axis[0], shore[0], shore[1], axis[1]]))


class NativeSpreadingPairingTests(unittest.TestCase):
    def advance(self, s, dt=2.):
        before = {name: np.asarray(value).copy() for name, value in s.material_surface.items()}
        transported = s.support.copy()
        birth = native_spreading.advance(s, transported, dt)
        area = float(birth @ s.cell_area)
        report = s.spreading_diagnostics
        self.assertAlmostEqual(area, s.process_totals['ocean_created_km2'], delta=1e-6)
        self.assertAlmostEqual(report['side_p_area_km2'], report['side_q_area_km2'], delta=1e-6)
        self.assertAlmostEqual(area, report['side_p_area_km2']+report['side_q_area_km2'], delta=1e-6)
        np.testing.assert_allclose(transported.sum(axis=0), s.support.sum(axis=0), atol=1e-12)
        self.assertTrue(np.all((birth >= 0.) & (birth <= 1.)))
        for name, value in before.items(): np.testing.assert_array_equal(s.material_surface[name], value)
        return area, birth, transported

    def test_real_paired_strip_matches_independent_spherical_area(self):
        for dt in (1., 2.):
            s, axis = scene()
            area, _, _ = self.advance(s, dt)
            self.assertAlmostEqual(area, expected_area(axis, dt), delta=1e-6)

    def test_refining_the_same_finite_front_does_not_create_extra_ocean(self):
        values = []
        for split in (1, 2, 4):
            s, axis = scene(split=split)
            area, _, _ = self.advance(s)
            self.assertAlmostEqual(area, expected_area(axis, 2.), delta=1e-6)
            values.append(area)
        self.assertLess(max(values)-min(values), 1e-6)

    def test_partial_ocean_cells_produce_paired_area_without_changing_continents(self):
        for level in (3, 4):
            s, axis = scene(corridor=True, level=level)
            area, _, _ = self.advance(s)
            self.assertAlmostEqual(area, expected_area(axis, 2.), delta=1e-6)
            self.assertGreater(s.spreading_diagnostics['partly_ocean_receiving_cells'], 0)

    def test_true_third_owner_and_its_guarded_neighbors_receive_no_birth(self):
        baseline, _ = scene(extent=600., level=4)
        _, birth, _ = self.advance(baseline)
        receivers = np.flatnonzero(birth > 0.)
        candidates = receivers[~np.isin(receivers, np.r_[baseline.ba, baseline.bb])]
        self.assertGreater(len(candidates), 0)
        forbidden = int(candidates[np.argmin(baseline.xyz[candidates, 2])])
        s, _ = scene(extent=600., level=4)
        s.plate[forbidden] = 2
        s.support[:, forbidden] = 0.; s.support[2, forbidden] = 1.
        _, birth, transported = self.advance(s)
        guarded = np.any(s.plate[s.native_mesh['face_neighbors']] == 2, axis=1) | (s.plate == 2)
        np.testing.assert_array_equal(birth[guarded], 0.)
        np.testing.assert_array_equal(transported[:, forbidden], s.support[:, forbidden])
        self.assertGreater(s.spreading_diagnostics['rejected_foreign_claims'], 0)


if __name__ == '__main__': unittest.main()
