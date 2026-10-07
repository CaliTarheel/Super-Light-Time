"""Dependent inequalities may have slack bounds despite positive pinv reactions."""
from itertools import permutations
import json
from pathlib import Path
import unittest

import numpy as np

import bounded_gravity
import contact_response
import deforming_regions
from material_surface import spherical_face_areas
import viscous_sheet

FIXTURE=Path(__file__).parent/'fixtures'/'contact_asein_lite_10x_0myr.npz'


class PhysicalNullPivotTests(unittest.TestCase):
    def test_dependent_inequalities_have_the_independent_minimum_for_every_order(self):
        # min .5*(x*x+y*y), x>=1, y>=1, x+y>=3. The exact answer
        # is (1.5,1.5): only the third bound binds. The inconsistent three-row
        # equality pseudoinverse has positive multipliers for every row.
        rows=np.array([[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]])/np.array([[1.],[1.],[np.sqrt(2.)]])
        target=np.array([1.,1.,3./np.sqrt(2.)])
        for order in permutations(range(3)):
            with self.subTest(order=order):
                selected=list(order)
                for _ in range(3):
                    r=rows[selected];d=target[selected];schur=r@r.T
                    multipliers=np.linalg.lstsq(schur,d,rcond=1e-12)[0]
                    negative=np.flatnonzero(multipliers < -64*np.finfo(float).eps*max(np.max(np.abs(multipliers)),1e-30))
                    if len(negative):
                        del selected[int(negative[np.argmin(multipliers[negative])])]
                        continue
                    if np.linalg.norm(schur@multipliers-d)<=1e-12:break
                    self.assertTrue(np.all(multipliers>0.))
                    pivot=bounded_gravity._dependent_inequality_pivot(r[:,None,:],d,multipliers)
                    self.assertIsNotNone(pivot)
                    at,proof=pivot
                    self.assertLessEqual(proof['normalized_null_reaction'],64*np.finfo(float).eps)
                    del selected[at]
                else:self.fail('The certified pivots did not reach a consistent working set.')
                answer=multipliers@r
                np.testing.assert_allclose(answer,[1.5,1.5,0.],rtol=0.,atol=2e-15)
                self.assertTrue(np.all(rows@answer>=target-2e-15))
                self.assertTrue(np.all(multipliers>=0.))
                self.assertLessEqual(np.max(np.abs(multipliers*(r@answer-d))),2e-15)

    def test_dependency_pivot_uses_the_actual_rows_with_a_nonsymmetric_inverse_map(self):
        rows=np.array([[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]])/np.array([[1.],[1.],[np.sqrt(2.)]])
        target=np.array([1.,1.,3./np.sqrt(2.)])
        inverse=np.eye(3);inverse[0,1]=1e-10
        schur=rows@inverse@rows.T
        self.assertGreater(np.linalg.norm(schur-schur.T),0.)
        multipliers=np.linalg.lstsq(schur,target,rcond=1e-12)[0]
        pivot=bounded_gravity._dependent_inequality_pivot(rows[:,None,:],target,multipliers)
        self.assertIsNotNone(pivot)
        at,proof=pivot
        direction=np.array([-1.,-1.,np.sqrt(2.)])
        self.assertLessEqual(np.linalg.norm(direction@rows),2e-15)
        self.assertLessEqual(np.linalg.norm(schur@direction),2e-15)
        self.assertIn(at,(0,1))
        self.assertLessEqual(proof['normalized_null_reaction'],64*np.finfo(float).eps)

    def test_viscous_ill_conditioning_cannot_retire_an_independent_physical_row(self):
        rows=np.array([[1.,0.,0.],[0.,1.,0.]])
        schur=np.diag([1.,1e-14]);target=np.ones(2)
        multipliers=np.linalg.lstsq(schur,target,rcond=1e-12)[0]
        self.assertGreater(np.linalg.norm(schur@multipliers-target),.9)
        self.assertIsNone(bounded_gravity._dependent_inequality_pivot(rows[:,None,:],target,multipliers))
        # A genuinely distinct geometric direction also cannot be called null.
        rows=np.array([[1.,0.,0.],[1.,1e-10,0.]])
        self.assertIsNone(bounded_gravity._dependent_inequality_pivot(rows[:,None,:],target,multipliers))

    def test_arithmetic_zero_reaction_never_produces_a_negative_dual_step(self):
        rows=np.array([[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]])/np.array([[1.],[1.],[np.sqrt(2.)]])
        target=np.array([1.,1.,3./np.sqrt(2.)])
        multipliers=np.array([-1e-15,.625,.625*np.sqrt(2.)])
        # This reaction is inside the caller's existing arithmetic negative
        # threshold; it is already treated as zero after working-set resolution.
        self.assertGreaterEqual(multipliers[0],-64*np.finfo(float).eps*np.max(np.abs(multipliers)))
        at,proof=bounded_gravity._dependent_inequality_pivot(rows[:,None,:],target,multipliers)
        self.assertEqual(at,0)
        self.assertEqual(proof['step_to_zero'],0.)

    def test_nonnegative_null_infeasibility_cannot_discard_a_physical_bound(self):
        # x>=1 and -x>=1 have a positive left-null Farkas witness. No
        # nonnegative-multiplier descent can reach a row to retire.
        rows=np.array([[1.,0.,0.],[-1.,0.,0.]])
        self.assertIsNone(bounded_gravity._dependent_inequality_pivot(
            rows[:,None,:],np.ones(2),np.zeros(2)))

    def test_captured_startup_passes_every_original_geometry_and_force_gate(self):
        with np.load(FIXTURE) as data:
            args=[data[name].copy() for name in ('points','faces','area','preferred','rigid','minimum','maximum')]
            dt=float(data['dt']);kwargs=json.loads(str(data['kwargs_json']))
        before=[value.copy() for value in args]
        points,faces,area,preferred,rigid,minimum,maximum=args
        self.assertEqual(points.shape,(658,3));self.assertEqual(len(faces),986)
        self.assertTrue(np.all(area>=minimum));self.assertTrue(np.all(area<=maximum))
        endpoint,velocity,solver,detail=contact_response.redistribute(*args,dt,**kwargs)
        actual=spherical_face_areas(endpoint,faces)
        self.assertTrue(np.all(actual>=minimum*(1-1e-12)))
        self.assertTrue(np.all(actual<=maximum*(1+1e-12)))
        signed,quality=deforming_regions._quality(endpoint,faces)
        _,original_quality=deforming_regions._quality(points,faces)
        self.assertTrue(np.all(signed>1e-15))
        self.assertTrue(np.all(quality>=np.minimum(.15,original_quality*.5)))
        np.testing.assert_array_equal(endpoint[rigid],points[rigid])
        self.assertTrue(solver['converged'])
        self.assertLessEqual(np.max(np.linalg.norm(velocity,axis=1)),kwargs['max_speed_km_myr'])
        self.assertFalse(detail['speed_scaled_after_constraints'])
        self.assertFalse(detail['nonlinear_geometry_correction_applied'])
        self.assertTrue(detail['original_physical_bounds_unchanged'])
        self.assertTrue(detail['dependent_inequality_pivots'])
        self.assertLessEqual(detail['normalized_complementarity'],kwargs['tolerance'])
        # Independent endpoint reactions and the unchanged start-stage viscous
        # operator must balance the original preferred load, not a reduced force.
        context=viscous_sheet.prepare(points,faces,area,kwargs['length_km'])
        rhs=viscous_sheet.apply(context,preferred);rhs[rigid]=0.
        jacobian=bounded_gravity.endpoint_area_jacobian(points,velocity,faces,dt,6371.)/dt
        reaction=np.zeros_like(points)
        for face,side,multiplier,norm in zip(detail['active_faces'],detail['active_sides'],
                detail['multipliers_normalized'],detail['constraint_jacobian_norm_km']):
            np.add.at(reaction,faces[face],(1 if side=='minimum' else -1)*jacobian[face]*multiplier/norm)
        residual=viscous_sheet.apply(context,velocity)-rhs-reaction
        residual-=points*np.sum(points*residual,axis=1)[:,None];residual[rigid]=0.
        self.assertLessEqual(np.linalg.norm(residual)/np.linalg.norm(rhs),kwargs['tolerance'])
        for old,value in zip(before,args):np.testing.assert_array_equal(old,value)


if __name__=='__main__':unittest.main()
