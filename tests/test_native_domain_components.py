"""Native domain identity follows real material and control connectivity."""
import unittest
import numpy as np

from mesh_geometry import icosphere,geometry
from material_surface import initialize_surface
from mesh_coverage import intersections
from native_domain_components import classify


def coverage(control,material,area=None):
    return dict(control_index=np.asarray(control,int),material_index=np.asarray(material,int),
                area_km2=np.ones(len(control)) if area is None else np.asarray(area,float))


def graph(edges):
    return dict(edge_faces=np.asarray(edges,int).reshape(-1,2))


def material(faces,owners):
    faces=np.asarray(faces,int).reshape(-1,3)
    count=int(faces.max()+1) if faces.size else 0
    return dict(vertices=np.zeros((count,3)),faces=faces,face_owner=np.asarray(owners,int))


class NativeDomainComponentTests(unittest.TestCase):
    def test_tiny_detached_material_cannot_alias_distant_same_owner_control(self):
        # Cell zero is the distant owner's territory; the island only overlaps
        # cell two, controlled by another plate. Nearest-owner fallback would
        # incorrectly label the island as part of cell zero's domain.
        mesh=graph([[0,1],[1,2]])
        surface=material([[0,1,2]],[0])
        result=classify(mesh,np.array([0,1,1]),surface,coverage([2],[0]))
        self.assertNotEqual(result['material_labels'][0],result['control_labels'][0])
        self.assertNotIn(result['material_labels'][0],result['control_labels'])
        self.assertEqual(result['count'],3)

    def test_connected_material_bridge_joins_separate_control_components(self):
        mesh=graph([[0,1],[1,2]])
        surface=material([[0,1,2],[2,1,3]],[0,0])
        result=classify(mesh,np.array([0,1,0]),surface,coverage([0,2],[0,1]))
        self.assertEqual(result['control_labels'][0],result['control_labels'][2])
        self.assertEqual(result['material_labels'][0],result['material_labels'][1])
        self.assertEqual(result['count'],2)

    def test_shared_edge_never_joins_different_material_owners(self):
        mesh=graph([[0,1]])
        surface=material([[0,1,2],[2,1,3]],[0,1])
        result=classify(mesh,np.array([0,1]),surface,coverage([0,1],[0,1]))
        self.assertNotEqual(result['material_labels'][0],result['material_labels'][1])
        np.testing.assert_array_equal(result['material_labels'],result['control_labels'])

    def test_same_vertex_or_geometric_overlap_is_not_a_material_bridge(self):
        mesh=graph([[0,1]])
        for faces in ([[0,1,2],[2,3,4]],[[0,1,2],[3,4,5]]):
            surface=material(faces,[0,0])
            # Neither sheet touches owner-zero control terrain. Geometric
            # overlap within a foreign cell supplies no stitching operation.
            result=classify(mesh,np.array([1,1]),surface,coverage([1,1],[0,1]))
            self.assertNotEqual(result['material_labels'][0],result['material_labels'][1])
            self.assertEqual(result['count'],3)

    def test_same_owner_control_terrain_can_join_separate_material_sheets(self):
        mesh=graph([[0,1]])
        surface=material([[0,1,2],[3,4,5]],[0,0])
        result=classify(mesh,np.array([0,0]),surface,coverage([0,1],[0,1]))
        self.assertEqual(result['count'],1)

    def test_zero_area_touch_cannot_attach_material(self):
        mesh=graph([[0,1]])
        surface=material([[0,1,2]],[0])
        result=classify(mesh,np.array([0,0]),surface,coverage([0],[0],[0.]))
        self.assertEqual(result['count'],2)

    def test_empty_material_and_disconnected_ocean_owner_zero(self):
        mesh=graph([[0,1],[1,2]])
        result=classify(mesh,np.array([0,1,0]),material([],[]),coverage([],[]))
        np.testing.assert_array_equal(result['control_labels'],[0,1,2])
        self.assertEqual(result['material_labels'].shape,(0,))
        self.assertEqual(result['count'],3)

    def test_real_spherical_coverage_and_domains_survive_polar_rotation(self):
        mesh=icosphere(2)
        owner=(mesh['xyz'][:,0]>0).astype(int)
        kind=np.where(mesh['xyz'][:,2]>.35,1,0).astype(np.uint8)
        surface=initialize_surface(mesh['vertices'],mesh['faces'],owner,kind)
        hit=intersections(mesh,surface['vertices'],surface['faces'])
        first=classify(mesh,owner,surface,hit)
        q,_=np.linalg.qr(np.random.default_rng(91).normal(size=(3,3)))
        q[:,0]*=np.linalg.det(q)
        moved=geometry(mesh['vertices']@q,mesh['faces'])
        shifted=dict(surface,vertices=surface['vertices']@q)
        hit=intersections(moved,shifted['vertices'],shifted['faces'])
        after=classify(moved,owner,shifted,hit)
        np.testing.assert_array_equal(first['control_labels'],after['control_labels'])
        np.testing.assert_array_equal(first['material_labels'],after['material_labels'])
        self.assertEqual(first['count'],after['count'])

    def test_classification_does_not_mutate_inputs(self):
        mesh=icosphere(1)
        owner=np.zeros(len(mesh['faces']),int)
        surface=initialize_surface(mesh['vertices'],mesh['faces'],owner,np.ones(len(owner),np.uint8))
        hit=intersections(mesh,surface['vertices'],surface['faces'])
        edges=mesh['edge_faces'].copy();faces=surface['faces'].copy();areas=hit['area_km2'].copy()
        result=classify(mesh,owner,surface,hit)
        self.assertEqual(result['count'],1)
        np.testing.assert_array_equal(mesh['edge_faces'],edges)
        np.testing.assert_array_equal(surface['faces'],faces)
        np.testing.assert_array_equal(hit['area_km2'],areas)


if __name__ == '__main__':
    unittest.main()
