"""Independent metric and work-coordinate checks for projected entry material."""

import math
import unittest

import numpy as np

import entry_footprint as footprint
import slab_memory
from ridge_geometry import rotate
from tests.test_continental_entry import geometry


NORMAL = np.array([0., 0., 1.])


def point(latitude, longitude):
    return np.array([math.cos(latitude)*math.cos(longitude),
                     math.cos(latitude)*math.sin(longitude), math.sin(latitude)])


class EntryFootprintTests(unittest.TestCase):
    def test_material_distance_depth_and_horizontal_footprint_agree(self):
        dip=50.
        material=point(100./6371., .2)
        projected=footprint.project_points(material,NORMAL,dip)
        np.testing.assert_allclose(footprint.unproject_points(projected,NORMAL,dip),
                                   material,rtol=0.,atol=2e-16)
        physical_offset=6371.*math.asin(float(projected@NORMAL))
        self.assertAlmostEqual(physical_offset,100.*math.cos(math.radians(dip)),places=10)
        self.assertAlmostEqual(float(footprint.depth_m(material,NORMAL,dip))/1000.,
                               100.*math.sin(math.radians(dip)),places=10)
        self.assertAlmostEqual(physical_offset*math.tan(math.radians(dip)),
                               float(footprint.depth_m(material,NORMAL,dip))/1000.,places=10)
        self.assertEqual(slab_memory.slab_length_km(dict(slab_retained_area_km2=1000.),10.),100.)
        outside=point(-.02,.4)
        np.testing.assert_array_equal(footprint.project_points(outside,NORMAL,dip),outside)
        self.assertEqual(float(footprint.projected_area_ratio(outside,NORMAL,dip)),1.)

    def test_projected_area_jacobian_matches_independent_tangent_cross_product(self):
        dip=50.
        step=1e-6
        for latitude in (.001,.04,.25,-.04):
            longitude=.23
            material=point(latitude,longitude)
            def local_area(mapped):
                a=(mapped(point(latitude+step,longitude))-
                   mapped(point(latitude-step,longitude)))/(2.*step)
                b=(mapped(point(latitude,longitude+step))-
                   mapped(point(latitude,longitude-step)))/(2.*step)
                return float(np.linalg.norm(np.cross(a,b)))
            ratio=local_area(lambda p:footprint.project_points(p,NORMAL,dip))/local_area(lambda p:p)
            self.assertAlmostEqual(ratio,float(footprint.projected_area_ratio(material,NORMAL,dip)),
                                   places=10)

    def test_material_strip_and_projected_strip_have_distinct_areas(self):
        dip=50.;factor=math.cos(math.radians(dip))
        near,far=20./6371.,100./6371.
        nodes,weights=np.polynomial.legendre.leggauss(32)
        angles=(near+far)/2.+(far-near)*nodes/2.
        sites=np.array([point(float(angle),.1) for angle in angles])
        numerical=(far-near)/2.*float(weights@(
            footprint.projected_area_ratio(sites,NORMAL,dip)*np.cos(angles)))
        exact=math.sin(factor*far)-math.sin(factor*near)
        material=math.sin(far)-math.sin(near)
        self.assertAlmostEqual(numerical,exact,places=14)
        self.assertAlmostEqual(exact/material,factor,delta=3e-5)
        self.assertGreater(material/exact,1.55)

    def test_projection_rotates_with_hinge_and_keeps_finite_arc_coordinate(self):
        from finite_entry_arc import endpoint_planes

        normal=np.array([0.,1.,0.]);middle=np.array([1.,0.,0.])
        left,right,_,_=endpoint_planes(normal,middle,200.,6371.)
        material=np.array([point(.02,.01),point(-.02,.01)])
        # Rotate the fixture so latitude uses the selected hinge normal.
        material=rotate(material,[-math.pi/2.,0.,0.])
        projected=footprint.project_points(material,normal,50.)
        for plane in (left,right):
            np.testing.assert_array_equal(np.sign(material@plane),np.sign(projected@plane))
        axis=np.array([.14,-.21,.07])
        rotated=footprint.project_points(rotate(material,axis),rotate(normal,axis),50.)
        np.testing.assert_allclose(rotated,rotate(projected,axis),rtol=0.,atol=5e-16)
        np.testing.assert_allclose(footprint.projected_area_ratio(rotate(material,axis),
            rotate(normal,axis),50.),footprint.projected_area_ratio(material,normal,50.),
            rtol=2e-14)

    def test_reference_area_ratio_includes_both_material_map_and_dip_projection(self):
        from entry_overlap_geometry import projective_area_ratio

        triangle=geometry()[0]
        weights=np.array([.2,.3,.5])
        direction=weights@triangle
        point_on_face=direction/np.linalg.norm(direction)
        material=projective_area_ratio(triangle,weights)
        projected=footprint.reference_to_footprint_ratio(triangle,weights,[0.,1.,0.],50.)
        self.assertAlmostEqual(float(projected/material),
                               float(footprint.projected_area_ratio(point_on_face,[0.,1.,0.],50.)),
                               places=14)

    def test_rejects_singular_or_unphysical_geometry(self):
        with self.assertRaisesRegex(ValueError,'unit points'):
            footprint.project_points([2.,0.,0.],NORMAL,50.)
        with self.assertRaisesRegex(ValueError,'dip'):
            footprint.project_points([1.,0.,0.],NORMAL,90.)
        with self.assertRaisesRegex(ValueError,'pole'):
            footprint.projected_area_ratio([0.,0.,1.],NORMAL,50.)
        with self.assertRaisesRegex(ValueError,'radius'):
            footprint.depth_m([1.,0.,0.],NORMAL,50.,radius_km=-1.)


if __name__=='__main__':
    unittest.main()
