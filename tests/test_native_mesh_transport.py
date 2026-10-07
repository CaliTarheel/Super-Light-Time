"""Conservation and spherical motion tests for native finite-volume transport."""
import unittest

import numpy as np

from mesh_geometry import icosphere, build_locator
from mesh_transport import advect, edge_flux, face_vertex_stencil, sample_coordinates


class NativeMeshTransportTests(unittest.TestCase):
    def test_closed_euler_flux_has_zero_cell_divergence(self):
        mesh = icosphere(3)
        a, b = mesh['edge_faces'].T
        for omega in np.random.default_rng(19).normal(0,.01,(6,3)):
            flux = edge_flux(mesh, omega)
            divergence = np.bincount(a, weights=flux, minlength=len(mesh['faces']))
            divergence -= np.bincount(b, weights=flux, minlength=len(mesh['faces']))
            self.assertLess(np.max(np.abs(divergence))/np.max(np.abs(flux)), 6e-15)

    def test_flux_matches_independent_velocity_line_integral(self):
        mesh = icosphere(0)
        omega = np.array([.004,-.006,.003])
        a, b = mesh['vertices'][mesh['edge_vertices']].transpose(1,0,2)
        cosine = np.einsum('ij,ij->i', a, b)
        angle = np.arccos(cosine)
        tangent = b-a*cosine[:,None]
        tangent /= np.linalg.norm(tangent,axis=1)[:,None]
        nodes, weights = np.polynomial.legendre.leggauss(12)
        theta = angle[:,None]*(nodes+1.)*.5
        points = a[:,None,:]*np.cos(theta)[:,:,None] + tangent[:,None,:]*np.sin(theta)[:,:,None]
        velocity = np.cross(omega, points)*mesh['radius_km']
        normal_velocity = np.einsum('eqi,ei->eq', velocity, mesh['edge_normal'])
        integral = np.sum(normal_velocity*weights[None,:],axis=1)*angle*.5*mesh['radius_km']
        np.testing.assert_allclose(edge_flux(mesh, omega), integral, rtol=1e-14, atol=2e-10)

    def test_uniform_fields_and_partition_of_unity(self):
        mesh = icosphere(2)
        field = np.ones((len(mesh['faces']),3))*np.array([1.,7.,-2.])
        result = advect(mesh, field, [.03,-.02,.01], 30.)
        np.testing.assert_allclose(result, field, atol=4e-14, rtol=0.)
        parts = np.random.default_rng(2).uniform(size=(len(field),4))
        parts /= parts.sum(axis=1)[:,None]
        result = advect(mesh,parts,[.02,.03,-.01],50.)
        np.testing.assert_allclose(result.sum(axis=1),1.,atol=2e-15)
        self.assertTrue(np.all(result >= 0.))

    def test_area_integrals_and_bounds_survive_large_timestep(self):
        mesh = icosphere(3)
        field = np.random.default_rng(37).uniform(-2.,5.,size=(len(mesh['faces']),3))
        result = advect(mesh,field,[.04,-.06,.03],40.)
        np.testing.assert_allclose(mesh['area_km2']@result,mesh['area_km2']@field,rtol=4e-15)
        self.assertTrue(np.all(result >= field.min(axis=0)-1e-13))
        self.assertTrue(np.all(result <= field.max(axis=0)+1e-13))

    def test_long_motion_crosses_pole_without_mass_loss(self):
        mesh = icosphere(3)
        centre = np.array([0.,-.4,np.sqrt(.84)])
        original = np.exp(3.*(mesh['xyz']@centre-1.))
        result = original.copy()
        omega = np.array([.006,0.,0.])
        for _ in range(250):
            result = advect(mesh,result,omega,2.)
        np.testing.assert_allclose(mesh['area_km2']@result,mesh['area_km2']@original,rtol=2e-15)
        moment = np.sum(mesh['xyz']*(result*mesh['area_km2'])[:,None],axis=0)
        moment /= np.linalg.norm(moment)
        angle = 3.
        expected = np.array([centre[0],centre[1]*np.cos(angle)-centre[2]*np.sin(angle),
                             centre[1]*np.sin(angle)+centre[2]*np.cos(angle)])
        # Upwind surface transport diffuses a tracer but should move its broad
        # centre through the south pole, not tear or stop at a map coordinate.
        self.assertLess(np.rad2deg(np.arccos(np.clip(moment@expected,-1.,1.))),1.)
        self.assertGreaterEqual(result.min(),original.min()-1e-13)
        self.assertLessEqual(result.max(),original.max()+1e-13)

    def test_smaller_timesteps_converge_to_semidiscrete_solution(self):
        mesh = icosphere(3)
        initial = np.sin(mesh['xyz'][:,0])*np.cos(mesh['xyz'][:,1])
        omega = np.array([.004,.003,-.002])
        def run(dt):
            result = initial.copy()
            for _ in range(round(20./dt)):
                result = advect(mesh,result,omega,dt)
            return result
        reference = run(.03125)
        errors = [np.sqrt(np.average((run(dt)-reference)**2,weights=mesh['area_km2'])) for dt in (2.,1.,.5,.25)]
        self.assertTrue(all(b < .55*a for a,b in zip(errors[:-1],errors[1:])))
        # This verifies temporal convergence, not disappearance of the spatial
        # diffusion inherent in first-order upwind transport.

    def test_common_rotation_of_mesh_and_euler_vector_is_invariant(self):
        from mesh_geometry import geometry
        mesh = icosphere(2)
        q,_ = np.linalg.qr(np.random.default_rng(8).normal(size=(3,3)))
        q[:,0] *= np.linalg.det(q)
        moved = geometry(mesh['vertices']@q,mesh['faces'])
        omega = np.array([.002,-.007,.003])
        values = mesh['xyz'][:,2]**2+.3*mesh['xyz'][:,0]
        np.testing.assert_allclose(edge_flux(mesh,omega),edge_flux(moved,omega@q),rtol=3e-13,atol=1e-10)
        np.testing.assert_allclose(advect(mesh,values,omega,15.),advect(moved,values,omega@q,15.),atol=2e-15)

    def test_native_sampling_stencil_has_unit_weight(self):
        mesh = icosphere(2)
        stencil = face_vertex_stencil(mesh)
        locator = build_locator(mesh['vertices'],mesh['faces'])
        points = np.r_[np.eye(3),-np.eye(3),mesh['edge_mid']]
        coordinates = sample_coordinates(mesh,locator,stencil,points)
        total = sum(weight for _,weight in coordinates)
        np.testing.assert_allclose(total,1.,atol=7e-16)
        self.assertTrue(all(np.all(weight >= 0.) for _,weight in coordinates))
        self.assertTrue(all(np.all((index >= 0)&(index < len(mesh['faces']))) for index,_ in coordinates))


if __name__ == '__main__':
    unittest.main()
