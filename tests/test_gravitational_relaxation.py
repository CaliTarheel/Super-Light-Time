"""Independent scalar-area, clipping and virtual-work gravity checks."""
import importlib.util
import math
from pathlib import Path
import sys
import unittest
from unittest import mock
import numpy as np

import bounded_gravity
import gravitational_relaxation as gravity


def unit(value):
    value=np.asarray(value,float)
    return value/np.linalg.norm(value,axis=-1,keepdims=True)


def sphere_polygon_area(points,radius=6371.):
    """Scalar spherical excess; no production geometry or clipping calls."""
    if len(points)<3:return 0.
    a=points[0];total=0.
    for i in range(1,len(points)-1):
        b,c=points[i:i+2]
        determinant=(a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0]))
        denominator=1.+sum(a[j]*b[j]+a[j]*c[j]+b[j]*c[j] for j in range(3))
        total+=2.*math.atan2(abs(determinant),denominator)*radius**2
    return total


def planar_clip(subject,clipper):
    """Scalar gnomonic 2D line clipping, independent of 3D production AD."""
    result=[np.array(p,float) for p in subject]
    def side(a,b,p):return float((b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0]))
    for a,b in zip(clipper,np.roll(clipper,-1,axis=0)):
        if not result:break
        source=result;result=[];previous=source[-1];before=side(a,b,previous)
        for point in source:
            after=side(a,b,point)
            if (before>=0)!=(after>=0):
                result.append(previous+(point-previous)*before/(before-after))
            if after>=0:result.append(point)
            previous,before=point,after
    return np.asarray(result,float).reshape(-1,2)


def independent_energy(points,faces,volumes,reference,sheets,radius=6371.):
    triangles=points[faces]
    areas=np.array([sphere_polygon_area(tri,radius) for tri in triangles])
    height=volumes/areas
    energy=.5*sum(float(a*(h-r)**2) for a,h,r in zip(areas,height,reference))
    for i in range(len(faces)):
        for j in range(i+1,len(faces)):
            if sheets[i]==sheets[j]:continue
            aa=triangles[i,:,:2]/triangles[i,:,2,None]
            bb=triangles[j,:,:2]/triangles[j,:,2,None]
            polygon=planar_clip(aa,bb)
            if len(polygon):
                spherical=unit(np.column_stack((polygon,np.ones(len(polygon)))))
                energy+=sphere_polygon_area(spherical,radius)*height[i]*height[j]
    return float(energy)


def fixture(overlap=True):
    xy=np.array([[-.08,-.06],[.08,-.06],[-.04,.08],[-.04,-.02],[.11,.01],[0,.09]])
    if not overlap:xy[3:,0]+=.3
    points=unit(np.column_stack((xy,np.ones(len(xy)))))
    faces=np.array([[0,1,2],[3,4,5]],int)
    return points,faces,np.array([38.,24.]),np.array([30.,25.]),np.array([2,7],int)


def rotation(angle=.8,axis=(.2,.5,-.3)):
    axis=unit(axis);x,y,z=axis;skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)*np.cos(angle)+(1-np.cos(angle))*np.outer(axis,axis)+np.sin(angle)*skew


class GravitationalRelaxationTests(unittest.TestCase):
    def test_fixed_volume_virtual_work_matches_independent_energy(self):
        points,faces,height,reference,sheets=fixture()
        areas=np.array([sphere_polygon_area(tri) for tri in points[faces]])
        volumes=areas*height
        value,gradient,report=gravity.energy_gradient(points,faces,height,reference,sheets)
        expected=independent_energy(points,faces,volumes,reference,sheets)
        self.assertAlmostEqual(value/expected,1.,places=13)
        rng=np.random.default_rng(718)
        for k in range(4):
            motion=rng.normal(size=points.shape)
            motion-=points*np.sum(points*motion,axis=1)[:,None]
            analytic=float(np.sum(gradient*motion));errors=[]
            for step in (3e-6,1e-6,3e-7):
                plus=independent_energy(unit(points+step*motion),faces,volumes,reference,sheets)
                minus=independent_energy(unit(points-step*motion),faces,volumes,reference,sheets)
                finite_difference=(plus-minus)/(2*step)
                errors.append(abs(finite_difference-analytic)/max(abs(analytic),1.))
            self.assertLess(max(errors[-2:]),2e-7,(k,errors,analytic))
        self.assertGreater(report['overlap_pairs'],0)

    def test_reference_equilibrium_is_zero_without_overlaps(self):
        points,faces,height,reference,sheets=fixture(overlap=False)
        value,gradient,report=gravity.energy_gradient(points,faces,reference,reference,sheets)
        self.assertEqual(value,0.);np.testing.assert_array_equal(gradient,0.)
        self.assertEqual(report['overlap_pairs'],0)

    def test_spherical_rotation_energy_covariance_and_zero_net_torque(self):
        points,faces,height,reference,sheets=fixture()
        value,gradient,_=gravity.energy_gradient(points,faces,height,reference,sheets)
        torque=np.cross(points,gradient).sum(axis=0)
        self.assertLess(np.linalg.norm(torque)/np.linalg.norm(gradient),3e-13)
        for angle,axis in ((1.57,[0,1,0]),(2.1,[.2,.7,.3])):
            q=rotation(angle,axis)
            moved,other,_=gravity.energy_gradient(points@q.T,faces,height,reference,sheets)
            np.testing.assert_allclose(moved,value,rtol=3e-13)
            np.testing.assert_allclose(other,gradient@q.T,rtol=2e-11,atol=2e-6)

    def test_reciprocal_sheet_exchange_and_relabelling_preserve_force(self):
        points,faces,height,reference,sheets=fixture()
        value,gradient,_=gravity.energy_gradient(points,faces,height,reference,sheets,chunk_pairs=1)
        other,exchanged,_=gravity.energy_gradient(points,faces[::-1],height[::-1],reference[::-1],sheets[::-1],chunk_pairs=9)
        np.testing.assert_allclose(other,value,rtol=3e-14,atol=0.)
        np.testing.assert_allclose(exchanged,gradient,rtol=2e-13,atol=1e-6)
        _,relabeled,_=gravity.energy_gradient(points,faces,height,reference,np.array([98,1]))
        np.testing.assert_array_equal(relabeled,gradient)
        # Suppressing only the inter-sheet term must recover the self-energy.
        self_energy,_,same=gravity.energy_gradient(points,faces,height,reference,np.ones(2,int))
        independent_self=sum(.5*sphere_polygon_area(t)*(h-r)**2 for t,h,r in zip(points[faces],height,reference))
        np.testing.assert_allclose(self_energy,independent_self,rtol=2e-14)
        self.assertEqual(same['overlap_pairs'],0)

    def test_area_gradient_near_corner_and_thin_face(self):
        for narrow in (1e-3,1e-5,1e-7):
            points=unit(np.array([[0.,0.,1.],[.03,0.,1.],[narrow,narrow,1.]]))
            faces=np.array([[0,1,2]])
            gradients=gravity.face_area_gradients(points,faces)[0]
            direction=np.array([[.13,.21,0.],[-.17,.08,0.],[.04,-.11,0.]])
            direction-=points*np.sum(points*direction,axis=1)[:,None]
            step=narrow*1e-3
            measured=(sphere_polygon_area(unit(points+step*direction))-sphere_polygon_area(unit(points-step*direction)))/(2*step)
            prediction=float(np.sum(gradients*direction))
            np.testing.assert_allclose(prediction,measured,rtol=1e-7,atol=2e-5)

    def test_coincident_boundaries_have_finite_exchange_symmetric_subgradient(self):
        points,faces,height,reference,sheets=fixture()
        points[3:]=points[:3]
        value,gradient,report=gravity.energy_gradient(points,faces,height,reference,sheets)
        other,reverse,_=gravity.energy_gradient(points,faces[::-1],height[::-1],reference[::-1],sheets[::-1])
        self.assertTrue(np.isfinite(gradient).all());self.assertEqual(value,other)
        self.assertGreater(report['clipping_near_edge_observations'],0)
        np.testing.assert_allclose(reverse,gradient,rtol=3e-13,atol=2e-6)
        self.assertLess(np.linalg.norm(np.cross(points,gradient).sum(axis=0))/np.linalg.norm(gradient),3e-13)

    def test_physical_unit_normalization_matches_si_force_drag_balance(self):
        points,faces,_,_,_=fixture(overlap=False)
        areas=np.array([12.,18.]);gradient=np.arange(18,dtype=float).reshape(6,3)-8.
        gradient-=points*np.sum(points*gradient,axis=1)[:,None]
        eta,H,L,radius=1.3e23,95.,350.,6371.
        force,report=gravity.normalized_body_force(points,faces,areas,gradient,
            viscosity_pa_s=eta,lithosphere_thickness_km=H,length_km=L,radius=radius)
        K=2800.*9.81*(1.-2800./3300.)
        energy_gradient_J=K*gradient*1000.**4
        force_N=-energy_gradient_J/(radius*1000.)
        node_area_m2=np.repeat(areas/3.,3)*1000.**2
        basal_drag=2.*eta*(H*1000.)/(L*1000.)**2
        expected_m_s=force_N/(node_area_m2[:,None]*basal_drag)
        expected_km_myr=expected_m_s*(365.25*86400.*1e6)/1000.
        np.testing.assert_allclose(force,expected_km_myr,rtol=3e-15,atol=0.)
        self.assertAlmostEqual(report['basal_drag_pa_s_m']/basal_drag,1.,places=15)
        for kwargs,multiplier in (({'viscosity_pa_s':2*eta},.5),({'length_km':2*L},4.),({'lithosphere_thickness_km':2*H},.5)):
            controls=dict(viscosity_pa_s=eta,lithosphere_thickness_km=H,length_km=L);controls.update(kwargs)
            scaled,_=gravity.normalized_body_force(points,faces,areas,gradient,**controls)
            np.testing.assert_allclose(scaled,force*multiplier,rtol=2e-15,atol=0.)


class GravitationalEvolutionTests(unittest.TestCase):
    def single(self,height=60.):
        points=unit(np.array([[-.005,-.003,1.],[.005,-.003,1.],[0,.006,1.]]))
        faces=np.array([[0,1,2]],int)
        areas=np.array([sphere_polygon_area(points)])
        return points,faces,areas*height,np.array([30.]),np.array([1]),areas

    def advance(self,points,faces,volumes,reference,sheets,areas,dt,**kwargs):
        return gravity.relax(points,faces,volumes,reference,sheets,dt,
            rigid_mask=kwargs.pop('rigid_mask',np.zeros(len(points),bool)),
            minimum_area_km2=areas*.2,maximum_area_km2=areas*5.,tolerance=1e-10,**kwargs)

    def test_isolated_reference_equilibrium_and_zero_interval_are_unchanged(self):
        points,faces,volumes,reference,sheets,areas=self.single(height=30.)
        result,info=self.advance(points,faces,volumes,reference,sheets,areas,8.)
        np.testing.assert_array_equal(result,points)
        self.assertEqual(info['completed_dt_myr'],8.)
        self.assertLess(info['energy_after_km4'],1e-20)
        zero,report=self.advance(points,faces,volumes*2,reference,sheets,areas,0.)
        np.testing.assert_array_equal(zero,points)
        self.assertEqual(report['completed_dt_myr'],0.)

    def test_compressed_column_expands_reduces_energy_and_closes_fixed_volume(self):
        points,faces,volumes,reference,sheets,areas=self.single()
        before=[v.copy() for v in (points,faces,volumes,reference,sheets,areas)]
        result,info=self.advance(points,faces,volumes,reference,sheets,areas,4.)
        final_area=np.array([sphere_polygon_area(result)])
        self.assertGreater(final_area[0],areas[0])
        self.assertLess((volumes/final_area)[0],60.)
        self.assertLess(info['energy_after_km4'],info['energy_before_km4'])
        independent=independent_energy(result,faces,volumes,reference,sheets)
        np.testing.assert_allclose(info['energy_after_km4'],independent,rtol=3e-14)
        np.testing.assert_allclose(final_area*(volumes/final_area),volumes,rtol=0,atol=1e-10)
        self.assertEqual(info['completed_dt_myr'],4.)
        for saved,current in zip(before,(points,faces,volumes,reference,sheets,areas)):np.testing.assert_array_equal(saved,current)

    def test_unequal_overlapping_sheets_reduce_energy_reciprocally_and_craton_stays_exact(self):
        points,faces,height,reference,sheets=fixture()
        areas=np.array([sphere_polygon_area(tri) for tri in points[faces]]);volumes=areas*height
        result,info=self.advance(points,faces,volumes,reference,sheets,areas,1.)
        self.assertLess(info['energy_after_km4'],info['energy_before_km4'])
        self.assertGreater(np.linalg.norm(result[:3]-points[:3]),0.)
        self.assertGreater(np.linalg.norm(result[3:]-points[3:]),0.)
        exchanged,other=self.advance(points,faces[::-1],volumes[::-1],reference[::-1],sheets[::-1],areas[::-1],1.)
        np.testing.assert_allclose(exchanged,result,rtol=0,atol=2e-13)
        np.testing.assert_allclose(other['energy_after_km4'],info['energy_after_km4'],rtol=3e-13)
        rigid=np.array([True,True,True,False,False,False])
        held,diagnostic=self.advance(points,faces,volumes,reference,sheets,areas,1.,rigid_mask=rigid)
        np.testing.assert_array_equal(held[rigid],points[rigid])
        self.assertGreater(np.linalg.norm(held[~rigid]-points[~rigid]),0.)
        self.assertLess(diagnostic['energy_after_km4'],diagnostic['energy_before_km4'])

    def test_outer_timestep_refinement_converges_for_same_full_interval(self):
        points,faces,volumes,reference,sheets,areas=self.single()
        outputs={}
        for divisions in (1,2,4,16):
            current=points.copy();completed=0.
            for _ in range(divisions):
                current,report=self.advance(current,faces,volumes,reference,sheets,areas,4./divisions)
                completed+=report['completed_dt_myr']
            self.assertEqual(completed,4.)
            outputs[divisions]=current
        errors=[np.linalg.norm(outputs[n]-outputs[16]) for n in (1,2,4)]
        self.assertLess(errors[1],.6*errors[0],errors)
        self.assertLess(errors[2],.6*errors[1],errors)

    def test_solver_or_internal_budget_failure_never_mutates_input_geometry(self):
        points,faces,volumes,reference,sheets,areas=self.single()
        before=[v.copy() for v in (points,faces,volumes,reference,sheets,areas)]
        with mock.patch('viscous_sheet.solve',return_value=(np.zeros_like(points),dict(converged=False,failure_reason='forced independent failure'))):
            with self.assertRaisesRegex(ValueError,'stationarity gate'):
                self.advance(points,faces,volumes,reference,sheets,areas,4.)
        with self.assertRaisesRegex(ValueError,'internal step budget'):
            self.advance(points,faces,volumes,reference,sheets,areas,1000.,max_substeps=1)
        with self.assertRaisesRegex(ValueError,'could not advance'):
            self.advance(points,faces,volumes,reference,sheets,areas,1000.,max_backtracks=1)
        for saved,current in zip(before,(points,faces,volumes,reference,sheets,areas)):np.testing.assert_array_equal(saved,current)

    def at_upper_bound(self,points,faces,volumes,reference,sheets,areas,dt,**kwargs):
        # A compressed column already at its maximum area, like a column on the
        # 8 km floor: every expanding trial needs the constrained endpoint.
        return gravity.relax(points,faces,volumes,reference,sheets,dt,
            rigid_mask=np.zeros(len(points),bool),minimum_area_km2=areas*.2,
            maximum_area_km2=areas.copy(),tolerance=1e-10,constraint_version=1,**kwargs)

    def test_geometry_only_inverse_failure_is_not_retried_over_shorter_intervals(self):
        # At 122 Myr an unconverged inverse failed identically in all 32 trials
        # and was then reported as a timestep problem.
        points,faces,volumes,reference,sheets,areas=self.single()
        failure=bounded_gravity.InverseSolveError('forced unconverged inverse')
        with mock.patch('bounded_gravity.solve_endpoint',side_effect=failure) as endpoint:
            with self.assertRaisesRegex(bounded_gravity.InverseSolveError,'forced unconverged inverse'):
                self.at_upper_bound(points,faces,volumes,reference,sheets,areas,4.)
        self.assertEqual(endpoint.call_count,1)

    def test_infeasible_interval_is_still_halved_and_named_in_the_failure(self):
        points,faces,volumes,reference,sheets,areas=self.single()
        failure=bounded_gravity.ConstraintSolveError('forced infeasible interval')
        with mock.patch('bounded_gravity.solve_endpoint',side_effect=failure) as endpoint:
            with self.assertRaisesRegex(ValueError,'could not advance.*after 5 trials.*forced infeasible interval'):
                self.at_upper_bound(points,faces,volumes,reference,sheets,areas,4.,max_backtracks=5)
        self.assertEqual([call.args[9] for call in endpoint.call_args_list],[4./2**k for k in range(5)])

    def test_entry_bound_violations_beyond_the_budget_are_not_retried(self):
        # This mock pins the proposed gravity exception routing. It does not
        # reproduce or validate the separate contact/adaptive failure path.
        points,faces,volumes,reference,sheets,areas=self.single()
        failure=bounded_gravity.ActiveSetBudgetError('forced entry-geometry budget exhaustion')
        with mock.patch('bounded_gravity.solve_endpoint',side_effect=failure) as endpoint:
            with self.assertRaisesRegex(bounded_gravity.ActiveSetBudgetError,'forced entry-geometry'):
                self.at_upper_bound(points,faces,volumes,reference,sheets,areas,4.)
        self.assertEqual(endpoint.call_count,1)

    def test_motion_driven_active_set_exhaustion_is_still_halved(self):
        # Only the step-independent case escapes. A budget exceeded because the
        # motion drove faces onto their bounds may well fit in a shorter step.
        points,faces,volumes,reference,sheets,areas=self.single()
        failure=bounded_gravity.ConstraintSolveError('exceeded its explicit active-set budget')
        with mock.patch('bounded_gravity.solve_endpoint',side_effect=failure) as endpoint:
            with self.assertRaisesRegex(ValueError,'could not advance.*active-set budget'):
                self.at_upper_bound(points,faces,volumes,reference,sheets,areas,4.,max_backtracks=4)
        self.assertEqual([call.args[9] for call in endpoint.call_args_list],[4./2**k for k in range(4)])

    def test_active_set_budget_scales_with_the_faces_being_solved(self):
        # Check the proposed default and reported capacity on this fixture;
        # contact_response now defers to it (test_contact_accommodation_budget).
        points,faces,volumes,reference,sheets,areas=self.single()
        _,info=self.at_upper_bound(points,faces,volumes,reference,sheets,areas,4.)
        expected=max(bounded_gravity.MINIMUM_ACTIVE_BUDGET,
                     int(bounded_gravity.ACTIVE_SET_FACE_FRACTION*len(faces)))
        self.assertEqual(info['active_set_budget'],expected)
        self.assertGreaterEqual(info['maximum_active_bounds'],1)
        self.assertLessEqual(info['maximum_active_bounds'],info['active_set_budget'])


if __name__=='__main__':unittest.main()
