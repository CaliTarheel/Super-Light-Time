"""Column thinning follows signed area change, including on a rotated globe."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from rift_mechanics import areal_strain_rate, RADIUS_KM
from ridge_geometry import rotate
import structure_engine as structure
import progressive_rifting as rifts
from tests.test_interior_structure import material


def star(directions=(0., np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4), radius=.02):
    angles = np.asarray(directions)
    tangent = np.column_stack((np.zeros(len(angles)), np.cos(angles), np.sin(angles)))
    centre = np.array([1., 0., 0.])
    xyz = np.vstack((centre, np.cos(radius)*centre+np.sin(radius)*tangent))
    edges = np.column_stack((np.zeros(len(angles), int), np.arange(1, len(angles)+1)))
    return xyz, edges, tangent


def extension_from_velocity(xyz, edges, velocity):
    a, b = edges.T
    cosine = np.sum(xyz[a]*xyz[b], axis=1)
    sine = np.linalg.norm(np.cross(xyz[a], xyz[b]), axis=1)
    forward_a = (xyz[b]-cosine[:, None]*xyz[a])/sine[:, None]
    forward_b = (cosine[:, None]*xyz[b]-xyz[a])/sine[:, None]
    return np.sum(velocity[b]*forward_b, axis=1)-np.sum(velocity[a]*forward_a, axis=1)


class ArealRiftStrainTests(unittest.TestCase):
    def test_isotropic_extension_returns_sum_of_two_principal_strains(self):
        xyz, edges, _ = star()
        speed = np.full(len(edges), .006*.02*RADIUS_KM)
        result = areal_strain_rate(xyz, edges, speed)
        self.assertTrue(result['supported'][0])
        self.assertAlmostEqual(result['rate'][0], .012, places=13)
        self.assertTrue(np.isfinite(result['rate']).all())

    def test_area_preserving_extension_and_compression_do_not_thin_columns(self):
        xyz, edges, tangent = star()
        tensor = np.diag([.007, -.007])
        axial = np.einsum('ni,ij,nj->n', tangent[:, 1:], tensor, tangent[:, 1:])
        self.assertTrue(np.any(axial > 0) and np.any(axial < 0))
        result = areal_strain_rate(xyz, edges, axial*.02*RADIUS_KM)
        self.assertAlmostEqual(result['rate'][0], 0., places=13)
        s = material()
        old = s.structure['thickness_km'].copy()
        strain = np.full(4, max(result['rate'][0], 0.)*.25*2.)
        structure.extend_interior(s, strain, strain)
        np.testing.assert_allclose(s.structure['thickness_km'], old, atol=1e-12)

    def test_simple_shear_has_tensile_links_but_zero_area_change(self):
        xyz, edges, tangent = star()
        # At the centre use v_y = gamma*z, v_z = 0. Parallel transport to the
        # ring gives exactly the tangent affine velocity samples of this field.
        local_velocity = np.column_stack((np.zeros(len(tangent)), .009*tangent[:, 2]*.02*RADIUS_KM,
                                          np.zeros(len(tangent))))
        velocity = np.zeros_like(xyz)
        for i, (direction, local) in enumerate(zip(tangent, local_velocity)):
            velocity[i+1] = rotate(local, np.cross(xyz[0], direction)*.02)
        extension = extension_from_velocity(xyz, edges, velocity)
        self.assertGreater(extension.max(), 0.)
        self.assertLess(extension.min(), 0.)
        result = areal_strain_rate(xyz, edges, extension)
        self.assertAlmostEqual(result['rate'][0], 0., places=13)

    def test_finite_euler_rotation_never_creates_area_change(self):
        xyz, edges, _ = star(radius=.3)
        velocity = np.cross([.007, -.003, .01], xyz)*RADIUS_KM
        result = areal_strain_rate(xyz, edges, extension_from_velocity(xyz, edges, velocity))
        np.testing.assert_allclose(result['rate'], 0., atol=1e-14)

    def test_orthogonal_rank_two_support_identifies_trace_without_shear(self):
        xyz, edges, _ = star((0., np.pi/2, np.pi, 3*np.pi/2))
        rates = np.array([.004, -.001, .004, -.001])
        result = areal_strain_rate(xyz, edges, rates*.02*RADIUS_KM)
        self.assertEqual(result['tensor_rank'][0], 2)
        self.assertTrue(result['supported'][0])
        self.assertAlmostEqual(result['rate'][0], .003, places=13)

    def test_one_dimensional_link_does_not_invent_transverse_area_change(self):
        xyz, edges, _ = star((0., np.pi))
        result = areal_strain_rate(xyz, edges, np.ones(2)*10.)
        self.assertFalse(np.any(result['supported']))
        np.testing.assert_array_equal(result['rate'], 0.)

    def test_globe_rotation_through_pole_and_edge_reversal_preserve_scalar_rate(self):
        xyz, edges, tangent = star()
        tensor = np.array([[.004, -.002], [-.002, -.001]])
        axial = np.einsum('ni,ij,nj->n', tangent[:, 1:], tensor, tangent[:, 1:])
        extension = axial*.02*RADIUS_KM
        first = areal_strain_rate(xyz, edges, extension)
        for rotation in ([0., np.pi/2, 0.], [.8, -1.5, .2], [0., 0., np.pi],
                         [0., np.arcsin(.89999), 0.], [0., np.arcsin(.90001), 0.]):
            result = areal_strain_rate(rotate(xyz, rotation), edges[:, ::-1], extension)
            np.testing.assert_allclose(result['rate'], first['rate'], atol=1e-12)
            np.testing.assert_array_equal(result['supported'], first['supported'])

    def test_irregular_directions_recover_the_signed_trace(self):
        xyz, edges, tangent = star((.11, .77, 1.81, 2.3, 3.6, 4.42, 5.02, 5.93))
        tensor = np.array([[.005, .004], [.004, -.006]])
        axial = np.einsum('ni,ij,nj->n', tangent[:, 1:], tensor, tangent[:, 1:])
        result = areal_strain_rate(xyz, edges, axial*.02*RADIUS_KM)
        self.assertTrue(result['supported'][0])
        self.assertAlmostEqual(result['rate'][0], -.001, places=13)

    def test_positive_area_change_conserves_volume_and_closes_actual_column_ledger(self):
        xyz, edges, _ = star()
        result = areal_strain_rate(xyz, edges, np.full(len(edges), .01*.02*RADIUS_KM))
        s = material()
        volume = s.structure['thickness_km']*s.structure['area_factor']
        birth = s.trace_relief_m.copy()
        strain = np.full(4, result['rate'][0]*.25*2.)
        structure.extend_interior(s, strain, strain)
        np.testing.assert_allclose(s.structure['thickness_km']*s.structure['area_factor'], volume, atol=1e-12)
        self.assertTrue(np.all(s.structure['thickness_km'] < s.structure['reference_thickness_km']))
        np.testing.assert_allclose(s.trace_relief_m-birth, s.trace_uplift_m-s.trace_extension_m, atol=1e-11)

    def test_progressive_update_can_damage_tensile_links_without_thinning_shear_material(self):
        xyz, edges, tangent = star()
        count = len(xyz)
        s = SimpleNamespace(kind=np.ones(count, np.uint8), relief=np.full(count, 100.),
                            pos=xyz, mass=np.ones(count), trace_kind=np.ones(count, np.uint8),
                            trace_relief_m=np.full(count, 100.), trace_xyz=xyz.copy(),
                            parcel_patch=np.arange(count), parcel_cell=np.arange(count),
                            parcel_plate=np.zeros(count, int), trace_plate=np.zeros(count, int),
                            rift_id=np.full(count, -1), trace_rift_id=np.full(count, -1),
                            rift_tangent=np.zeros_like(xyz), rift_bonds={}, rift_systems=[],
                            next_rift_system_id=1, plate_uid=np.array([7]), active=np.array([True]),
                            config=dict(mechanics_nodes=128, rift_strength=1.), t=2.,
                            _indices=lambda points: np.arange(len(points)),
                            _remember_rift_extension=Mock(), _record=Mock())
        for name in ('uplift', 'extension', 'erosion', 'adjustment'):
            setattr(s, 'trace_'+name+'_m', np.zeros(count))
        structure.initialize_parcels(s)
        structure.initialize_traces(s, np.arange(count))
        mesh = dict(xyz=xyz, edges=edges, bases=np.arange(count), owners=np.zeros(count, int),
                    owner_uids=np.full(count, 7), area=np.ones(count), craton=np.zeros(count),
                    suture=np.zeros(count), thickness=np.full(count, 35.),
                    reference_thickness=np.full(count, 35.), heat=np.zeros(count),
                    parcel_node=np.arange(count), trace_node=np.arange(count))
        # A prescribed resolved pure-shear response has genuine positive fault
        # extension in one direction and balancing shortening perpendicular to
        # it. The real update path must retain damage but use its areal trace.
        axial = .007*(tangent[:, 1]**2-tangent[:, 2]**2)
        response = dict(velocity=np.zeros_like(xyz), edge_extension=axial*.02*RADIUS_KM,
                        edge_shear=np.zeros(len(edges)), edge_length_km=np.full(len(edges), .02*RADIUS_KM),
                        iterations=1, converged=True, relative_residual=0., component_count=1)
        with (patch.object(rifts.rift_material, 'refresh', return_value=mesh),
              patch.object(rifts, 'boundary_loading', return_value=(np.zeros_like(xyz), np.ones(count))),
              patch.object(rifts.rift_mechanics, 'solve_loading', return_value=response)):
            rifts.update(s, 2., np.zeros(count))
        self.assertTrue(any(row['damage'] > 0 for row in s.rift_bonds.values()))
        self.assertTrue(any(row['strain'] > 0 for row in s.rift_bonds.values()))
        np.testing.assert_allclose(s.structure['thickness_km'], 35., atol=1e-12)
        np.testing.assert_allclose(s.trace_structure['thickness_km'], 35., atol=1e-12)
        np.testing.assert_allclose(s.trace_extension_m, 0., atol=1e-12)
        np.testing.assert_allclose(s.relief, 100., atol=1e-12)
        self.assertEqual(s.rift_mechanics['areal_strain_supported_nodes'], 1)

    def test_large_sparse_graph_uses_local_matrices_and_remains_finite(self):
        # 8,192 nodes spread around a sphere: six bounded index-neighbours per
        # node, with deliberately nonuniform angles and no dense global solve.
        n = 8192
        i = np.arange(n)
        z = 1-2*(i+.5)/n
        angle = i*np.pi*(3-np.sqrt(5.))
        xyz = np.column_stack((np.sqrt(1-z*z)*np.cos(angle), np.sqrt(1-z*z)*np.sin(angle), z))
        edges = np.concatenate([np.column_stack((i, (i+offset) % n)) for offset in (34, 55, 89)])
        result = areal_strain_rate(xyz, edges, np.zeros(len(edges)))
        self.assertEqual(result['rate'].shape, (n,))
        np.testing.assert_array_equal(result['rate'], 0.)
        self.assertTrue(np.all(result['supported']))


if __name__ == '__main__':
    unittest.main()
