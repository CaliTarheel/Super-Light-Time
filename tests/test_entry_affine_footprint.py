"""Independent metric and inverse checks for the native affine entry coordinate."""

import unittest

import numpy as np

from entry_affine_footprint import AffineConveyorFootprint
from ridge_geometry import rotate
from tests.test_channel_entry_zones import paired
from tests.test_entry_footprint import point


class AffineFootprintTests(unittest.TestCase):
    def fixture(self):
        surface,_,_,_,_,spec=paired()
        lower=surface['vertices'][surface['faces'][0]]
        return lower,spec['hinge_normals'][0]

    def test_affine_entry_coordinate_and_inverse_agree_inside_face(self):
        lower,normal=self.fixture()
        mapped=AffineConveyorFootprint(lower,normal,50.)
        bary=np.array([[.2,.3,.5],[.1,.7,.2],[.7,.1,.2]])
        physical=mapped.project(bary)
        np.testing.assert_allclose(mapped.unproject(physical),bary,rtol=0.,atol=2e-13)
        expected=((bary@mapped.q)*np.where(bary@mapped.q>0.,mapped.factor,1.)/
                  mapped.radius_m)
        np.testing.assert_allclose(np.arcsin(physical@normal),expected,
                                   rtol=0.,atol=3e-16)
        np.testing.assert_allclose(mapped.project(np.eye(3))[0],lower[0],atol=2e-16)
        with self.assertRaisesRegex(ValueError,'barycentric'):
            mapped.project([.2,.3,.4])

    def test_area_jacobian_matches_independent_barycentric_tangents(self):
        lower,normal=self.fixture()
        mapped=AffineConveyorFootprint(lower,normal,50.)
        step=1e-6
        du=np.array([-1.,1.,0.]);dv=np.array([-1.,0.,1.])
        for bary in (np.array([.2,.3,.5]),np.array([.7,.1,.2])):
            pdu=(mapped.project(bary+step*du)-mapped.project(bary-step*du))/(2*step)
            pdv=(mapped.project(bary+step*dv)-mapped.project(bary-step*dv))/(2*step)
            measured=(mapped.radius_m**2*np.linalg.norm(np.cross(pdu,pdv)) /
                      (2.*mapped.area_km2*1e6))
            np.testing.assert_allclose(mapped.reference_to_footprint_ratio(bary),
                                       measured,rtol=5e-10)

    def test_zero_entry_is_a_continuous_physical_hinge(self):
        lower,normal=self.fixture()
        mapped=AffineConveyorFootprint(lower,normal,50.)
        fraction=-mapped.q[0]/(mapped.q[1]-mapped.q[0])
        on=np.array([1.-fraction,fraction,0.])
        self.assertAlmostEqual(float(mapped.project(on)@normal),0.,places=14)
        for step in (1e-5,1e-7):
            left=on+step*np.array([1.,-1.,0.])
            right=on-step*np.array([1.,-1.,0.])
            self.assertLess(np.linalg.norm(mapped.project(left)-mapped.project(right)),
                            5.*step)

    def test_common_rotation_preserves_projection_and_local_area(self):
        lower,normal=self.fixture()
        old=AffineConveyorFootprint(lower,normal,50.)
        axis=[.2,-.1,.07]
        new=AffineConveyorFootprint(rotate(lower,axis),rotate(normal,axis),50.)
        bary=np.array([[.2,.3,.5],[.7,.1,.2]])
        np.testing.assert_allclose(new.project(bary),rotate(old.project(bary),axis),
                                   rtol=0.,atol=7e-16)
        np.testing.assert_allclose(new.reference_to_footprint_ratio(bary),
                                   old.reference_to_footprint_ratio(bary),rtol=2e-12)

    def test_affine_constant_distance_face_fails_closed_on_fold(self):
        face=np.array([point(.02,angle) for angle in (0.,.02,.04)])
        with self.assertRaisesRegex(ValueError,'folds or degenerates'):
            AffineConveyorFootprint(face,[0.,0.,1.],50.)


if __name__=='__main__':
    unittest.main()
