"""Complete inequality QPs precede unchanged nonlinear physical acceptance."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import bounded_gravity as bg
import contact_response
import deforming_regions
import viscous_sheet as sheet
from material_surface import spherical_face_areas

FIXTURE=Path(__file__).parent/'fixtures/contact_asein_lite_43myr.npz'


class OmittedBoundsTests(unittest.TestCase):
    def test_captured43myr_omitted_upper_bounds_obey_full_original_contract(self):
        with np.load(FIXTURE) as z:
            args=[z[k].copy() for k in ('points','faces','area','preferred','rigid','minimum','maximum')]
            kwargs=json.loads(str(z['kwargs_json']));dt=float(z['dt'])
        points,faces,area,preferred,rigid,minimum,maximum=args
        before=[v.copy() for v in args]
        q,v,solver,detail=contact_response.redistribute(*args,dt,**kwargs)
        actual=spherical_face_areas(q,faces)
        self.assertTrue(np.all(actual>=minimum*(1-1e-12)))
        self.assertTrue(np.all(actual<=maximum*(1+1e-12)))
        signed,quality=deforming_regions._quality(q,faces)
        _,old_quality=deforming_regions._quality(points,faces)
        self.assertTrue(np.all(signed>1e-15))
        self.assertTrue(np.all(quality>=np.minimum(.15,old_quality*.5)))
        np.testing.assert_array_equal(q[rigid],points[rigid])
        self.assertLessEqual(float(np.linalg.norm(v,axis=1).max()),kwargs['max_speed_km_myr'])
        self.assertTrue(solver['converged'])
        self.assertLessEqual(detail['sqp_iterations'],32)
        self.assertLessEqual(detail['normalized_complementarity'],kwargs['tolerance'])
        self.assertFalse(detail['speed_scaled_after_constraints'])
        self.assertFalse(detail['nonlinear_geometry_correction_applied'])
        self.assertTrue(detail['original_physical_bounds_unchanged'])
        bindings=set(zip(detail['active_faces'],detail['active_sides']))
        self.assertEqual(bindings,{(6,'minimum'),(0,'maximum'),(5,'minimum')})
        self.assertTrue(any(p['face']==0 and p['kind']=='omitted_linearized_bound'
            for p in detail['linearized_inequality_pivots']))
        # The omitted upper row initially has a negative equality-pseudoinverse
        # reaction because the existing lower pair shares two free DOFs. It
        # must replace a lower reaction along a real null row, not be deleted
        # immediately and added again in an endless working-set cycle.
        self.assertTrue(any(p['step_to_zero']>0 for p in detail['dependent_inequality_pivots']))
        context=sheet.prepare(points,faces,area,kwargs['length_km'])
        rhs=sheet.apply(context,preferred);rhs[rigid]=0.
        jac=bg.endpoint_area_jacobian(points,v,faces,dt,kwargs['radius'])/dt
        reaction=np.zeros_like(points)
        for face,side,multiplier,norm in zip(detail['active_faces'],detail['active_sides'],
                detail['multipliers_normalized'],detail['constraint_jacobian_norm_km']):
            self.assertGreaterEqual(multiplier,0.)
            np.add.at(reaction,faces[face],(1 if side=='minimum' else -1)*jac[face]*multiplier/norm)
        residual=sheet.apply(context,v)-rhs-reaction
        residual-=points*np.sum(points*residual,axis=1)[:,None];residual[rigid]=0.
        self.assertLessEqual(np.linalg.norm(residual)/np.linalg.norm(rhs),kwargs['tolerance'])
        for a,b in zip(before,args):np.testing.assert_array_equal(a,b)

    def test_inactive_blocker_and_dependent_remove_readd_cycle_have_exact_qp_minimum(self):
        # min .5||v-(.1,.1)||²; x+y>=2, x-y>=.2, y<=.4.
        # The initial predictor violates only the first two bounds. Solving
        # them gives(1.1,.9), which violates the previously inactive upper row.
        # The full independent answer is(1.6,.4), with x-y slack; dropping the
        # new row by its inconsistent equality pseudoinverse signs cycles.
        points=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        faces=np.tile([0,1,2],(3,1));rigid=np.array([False,True,True])
        rows=np.array([[1.,1.],[1.,-1.],[0.,1.]])
        preferred=np.zeros_like(points);preferred[0,1:]=.1
        minimum=np.array([12.,10.2,1.]);maximum=np.array([100.,100.,10.4])
        def endpoint(p,v,dt,radius):
            q=p.copy();q[0,1:]=v[0,1:];return q
        def areas(q,f,radius):return 10.+rows@q[0,1:]
        def jacobian(p,v,f,dt,radius):
            j=np.zeros((3,3,3));j[:,0,1:]=rows;return j
        def solve(p,f,a,boundary,rigid,driven,length_km,**kw):
            return kw['body_force'].copy(),dict(iterations=1,maximum_iterations=kw['iterations'],
                relative_residual=0.,tolerance=kw['tolerance'],converged=True)
        with patch.object(bg,'endpoint',endpoint),patch.object(bg,'spherical_face_areas',areas),\
                patch.object(bg,'endpoint_area_jacobian',jacobian),\
                patch.object(sheet,'prepare',return_value={'data':np.ones(3),'face_scale':np.ones(3)}),\
                patch.object(sheet,'apply',side_effect=lambda c,v:v),patch.object(sheet,'solve',solve),\
                patch.object(sheet,'strain_rate',return_value={'D':np.zeros((3,2,2)),'divergence':np.zeros(3)}):
            q,v,info,detail=bg.solve_endpoint(points,faces,np.full(3,10.),preferred,preferred,
                {'converged':True},rigid,minimum,maximum,1.,feasibility_tolerance=1e-12)
        np.testing.assert_allclose(v[0,1:],[1.6,.4],rtol=0.,atol=3e-14)
        self.assertEqual(set(zip(detail['active_faces'],detail['active_sides'])),{(0,'minimum'),(2,'maximum')})
        self.assertTrue(np.all(areas(q,faces,6371.)>=minimum-3e-14))
        self.assertTrue(np.all(areas(q,faces,6371.)<=maximum+3e-14))
        self.assertLessEqual(info['relative_residual'],1e-8)
        self.assertLessEqual(detail['normalized_complementarity'],1e-8)
        self.assertTrue(detail['dependent_inequality_pivots'])

    def test_inconsistent_dependent_pivot_keeps_feasible_nonnegative_dual_path(self):
        r=np.array([[1.,1.,0.],[1.,-1.,0.],[0.,-1.,0.]])/np.array([[np.sqrt(2.)],[np.sqrt(2.)],[1.]])
        target=np.array([1.8/np.sqrt(2.),.2/np.sqrt(2.),-.3])
        current=np.array([.9*np.sqrt(2.),.1*np.sqrt(2.),0.])
        pinv=np.linalg.lstsq(r@r.T,target,rcond=1e-12)[0]
        self.assertLess(pinv[-1],0.)  # its signs are not a restricted optimum
        at,proof,direction=bg._dependent_inequality_pivot(r[:,None,:],target,current,with_direction=True)
        updated=current+proof['step_to_zero']*direction
        self.assertEqual(at,1)
        self.assertGreater(proof['step_to_zero'],0.)
        self.assertTrue(np.all(updated>=-2e-15))
        self.assertAlmostEqual(updated[at],0.,delta=2e-15)
        np.testing.assert_allclose(updated@r,current@r,rtol=0.,atol=2e-15)
        self.assertGreater(direction@target,0.)

if __name__=='__main__':unittest.main()
