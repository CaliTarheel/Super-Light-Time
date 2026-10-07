"""Spherical polygon coverage conserves true material area at coarse resolution."""
import math
import unittest
import numpy as np

from mesh_geometry import icosphere,geometry,build_locator,spherical_area
from mesh_coverage import intersections,clip_triangle,_polygon_area


def rotation(angle=.23,axis=(.2,.7,.4)):
    axis=np.asarray(axis,float);axis/=np.linalg.norm(axis)
    x,y,z=axis
    skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)*np.cos(angle)+(1-np.cos(angle))*np.outer(axis,axis)+np.sin(angle)*skew


def small_triangle(centre,radius):
    centre=np.asarray(centre,float);centre/=np.linalg.norm(centre)
    east=np.cross(centre,[.2,.4,1.]);east/=np.linalg.norm(east)
    north=np.cross(centre,east)
    angle=np.arange(3)*2*np.pi/3
    return np.cos(radius)*centre+np.sin(radius)*(np.cos(angle)[:,None]*east+np.sin(angle)[:,None]*north)


class MeshCoverageTests(unittest.TestCase):
    def test_identical_mesh_is_exact_cell_coverage_without_edge_area(self):
        mesh=icosphere(2)
        hits=intersections(mesh,mesh['vertices'],mesh['faces'])
        np.testing.assert_array_equal(hits['control_index'],np.arange(len(mesh['faces'])))
        np.testing.assert_array_equal(hits['material_index'],np.arange(len(mesh['faces'])))
        np.testing.assert_allclose(hits['area_km2'],mesh['area_km2'],rtol=3e-14)

    def test_known_partial_octant_has_exact_spherical_area(self):
        mesh=geometry(np.eye(3),np.array([[0,1,2]]),radius_km=1.,require_closed=False)
        source=np.eye(3)@rotation(np.pi/4.,(0,0,1)).T
        hits=intersections(mesh,source,np.array([[0,1,2]]))
        self.assertEqual(len(hits['area_km2']),1)
        self.assertAlmostEqual(hits['area_km2'][0],math.pi/4.,places=14)

    def test_tiny_triangle_straddling_edge_keeps_its_actual_area(self):
        mesh=icosphere(2)
        triangle=small_triangle(mesh['edge_mid'][6],.00004)
        faces=np.array([[0,1,2]])
        expected=spherical_area(triangle,faces)[0]
        hits=intersections(mesh,triangle,faces)
        self.assertGreaterEqual(len(hits['area_km2']),2)
        self.assertAlmostEqual(hits['area_km2'].sum()/expected,1.,places=6)
        self.assertLess(hits['area_km2'].sum(),mesh['area_km2'].min()*1e-5)
        self.assertTrue(np.all(hits['area_km2']>0))

    def test_rotated_material_conserves_area_and_leaves_holes_uncovered(self):
        mesh=icosphere(2)
        source=icosphere(1)
        chosen=np.arange(len(source['faces']))[::3]
        vertices=source['vertices']@rotation().T
        faces=source['faces'][chosen]
        hits=intersections(mesh,vertices,faces)
        actual=np.bincount(hits['material_index'],weights=hits['area_km2'],minlength=len(faces))
        np.testing.assert_allclose(actual,spherical_area(vertices,faces),rtol=2e-13,atol=2e-7)
        self.assertLess(hits['area_km2'].sum(),mesh['area_km2'].sum()*.5)
        self.assertLess(len(np.unique(hits['control_index'])),len(mesh['faces']))

    def test_rotation_through_poles_preserves_intersections_and_area(self):
        mesh=icosphere(2)
        source=icosphere(1)
        faces=source['faces'][::4]
        vertices=source['vertices']@rotation(.19).T
        original=intersections(mesh,vertices,faces)
        q=rotation(1.72,(-.1,.7,.4))
        moved=geometry(mesh['vertices']@q.T,mesh['faces'])
        after=intersections(moved,vertices@q.T,faces)
        np.testing.assert_array_equal(original['control_index'],after['control_index'])
        np.testing.assert_array_equal(original['material_index'],after['material_index'])
        np.testing.assert_allclose(original['area_km2'],after['area_km2'],rtol=2e-10,atol=1e-7)

    def test_overlapping_material_sheets_remain_separate(self):
        mesh=icosphere(1)
        faces=np.r_[mesh['faces'][[0]],mesh['faces'][[0]]]
        hits=intersections(mesh,mesh['vertices'],faces)
        np.testing.assert_array_equal(hits['control_index'],[0,0])
        np.testing.assert_array_equal(hits['material_index'],[0,1])
        np.testing.assert_allclose(hits['area_km2'],mesh['area_km2'][0],rtol=3e-14)
        self.assertAlmostEqual(hits['area_km2'].sum()/mesh['area_km2'][0],2.,places=14)

    def test_candidate_index_matches_brute_clipping_including_large_faces(self):
        mesh=icosphere(1)
        source=icosphere(0)
        faces=source['faces'][[0,3,7,13]]
        vertices=source['vertices']@rotation(.73).T
        hits=intersections(mesh,vertices,faces)
        expected=[]
        for i,triangle in enumerate(vertices[faces]):
            for j,control in enumerate(mesh['vertices'][mesh['faces']]):
                area=_polygon_area(clip_triangle(triangle,control),6371.)
                if area>1e-10:
                    expected.append((j,i,area))
        np.testing.assert_array_equal(hits['control_index'],[row[0] for row in expected])
        np.testing.assert_array_equal(hits['material_index'],[row[1] for row in expected])
        np.testing.assert_allclose(hits['area_km2'],[row[2] for row in expected],rtol=2e-13,atol=2e-8)
        fine=build_locator(mesh['vertices'],mesh['faces'],bin_resolution=127)
        after=intersections(mesh,vertices,faces,locator=fine)
        np.testing.assert_array_equal(hits['control_index'],after['control_index'])
        np.testing.assert_allclose(hits['area_km2'],after['area_km2'],rtol=0.,atol=0.)

    def test_empty_material_and_stale_locator(self):
        mesh=icosphere(1)
        hits=intersections(mesh,np.empty((0,3)),np.empty((0,3),int))
        self.assertEqual(len(hits['area_km2']),0)
        moved=geometry(mesh['vertices']@rotation().T,mesh['faces'])
        with self.assertRaises(ValueError):
            intersections(moved,mesh['vertices'],mesh['faces'],
                          locator=build_locator(mesh['vertices'],mesh['faces']))


if __name__ == '__main__':
    unittest.main()
