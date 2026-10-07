"""A real 204 Myr contact failure: three area inequalities, two free DOFs."""
from itertools import permutations
from pathlib import Path
import unittest
import numpy as np
import bounded_gravity
import contact_response
import deforming_regions
import viscous_sheet
from material_surface import spherical_face_areas

ROOT=Path(__file__).parent/'fixtures'/'contact204'


def captured(depth):
    with np.load(ROOT/f'depth-{depth}.npz') as data:
        args=[data[k] for k in ('points','faces','area','preferred','rigid','minimum','maximum')]
        return args,float(data['dt']),data['viscosity_weights']


class DependentBoundTests(unittest.TestCase):
    def test_all_six_rejected_intervals_pass_original_nonlinear_and_force_checks(self):
        for depth in range(2,8):
            with self.subTest(depth=depth):
                args,dt,weights=captured(depth)
                before=[v.copy() for v in args]
                self.assertEqual(len(args[1]),3)
                self.assertEqual(np.count_nonzero(~args[4]),1)
                q,v,solver,detail=contact_response.redistribute(*args,dt,
                    iterations=1024,tolerance=1e-8,viscosity_weights=weights)
                area=spherical_face_areas(q,args[1])
                self.assertTrue(np.all(area>=args[5]*(1-1e-12)))
                self.assertTrue(np.all(area<=args[6]*(1+1e-12)))
                triangles=q[args[1]]
                self.assertTrue(np.all(np.einsum('ij,ij->i',triangles[:,0],
                    np.cross(triangles[:,1],triangles[:,2]))>0.))
                np.testing.assert_array_equal(q[args[4]],args[0][args[4]])
                self.assertTrue(solver['converged'])
                self.assertLessEqual(solver['relative_residual'],1e-8)
                self.assertLessEqual(detail['normalized_complementarity'],1e-8)
                self.assertTrue(np.all(np.array(detail['multipliers_normalized'])>=0.))
                self.assertEqual(len(detail['active_faces']),1)
                self.assertLessEqual(detail['sqp_iterations'],3)
                self.assertTrue(detail['original_physical_bounds_unchanged'])
                for original,after in zip(before,args):
                    np.testing.assert_array_equal(original,after)

    def test_face_order_does_not_change_which_physical_solution_is_accepted(self):
        args,dt,weights=captured(2)
        original=None
        for order in permutations(range(3)):
            idx=np.array(order)
            reordered=[a.copy() for a in args]
            for at in (1,2,5,6):
                reordered[at]=args[at][idx]
            q,_,solver,_=contact_response.redistribute(*reordered,dt,
                iterations=1024,tolerance=1e-8,viscosity_weights=weights[idx])
            self.assertTrue(solver['converged'])
            if original is None:original=q
            else:np.testing.assert_allclose(q,original,rtol=0.,atol=2e-12)

    def test_discrepancies_do_not_cancel_and_report_physical_units(self):
        current=np.array([6.,14.,10.])
        minimum=np.array([8.,8.,8.]);maximum=np.array([12.,12.,12.])
        self.assertEqual(bounded_gravity.area_violation_summary(current,minimum,maximum),
            dict(total_area_violation_km2=4.,maximum_face_area_violation_km2=2.,violated_faces=2))
        message=str(bounded_gravity._area_failure('rejected',current,minimum,maximum))
        self.assertIn('4 km2 total absolute',message)
        self.assertIn('not lost crust',message)

    def test_large_captured_trials_stay_outward_and_satisfy_full_contact_contract(self):
        for depth in (0,1):
            with self.subTest(depth=depth):
                args,dt,weights=captured(depth)
                points,faces,area,preferred,rigid,minimum,maximum=args
                q,velocity,solver,detail=contact_response.redistribute(*args,dt,
                    iterations=1024,tolerance=1e-8,viscosity_weights=weights)
                actual=spherical_face_areas(q,faces)
                self.assertTrue(np.all(actual>=minimum*(1-1e-12)))
                self.assertTrue(np.all(actual<=maximum*(1+1e-12)))
                signed,quality=deforming_regions._quality(q,faces)
                _,old_quality=deforming_regions._quality(points,faces)
                self.assertTrue(np.all(signed>1e-15))
                self.assertTrue(np.all(quality>=np.minimum(.15,old_quality*.5)))
                np.testing.assert_array_equal(q[rigid],points[rigid])
                self.assertTrue(detail['outward_orientation_preserved'])
                self.assertFalse(detail['speed_scaled_after_constraints'])
                self.assertLessEqual(float(np.linalg.norm(velocity,axis=1).max()),100.)
                self.assertLessEqual(detail['normalized_complementarity'],1e-8)
                if depth==0:self.assertGreater(detail['predictor_orientation_backtracks'],0)
                # Recompute the full returned force residual independently of
                # the solver's diagnostic, including the endpoint reactions.
                context=viscous_sheet.prepare(points,faces,area,100.,viscosity_weights=weights)
                rhs=viscous_sheet.apply(context,preferred);rhs[rigid]=0.
                jac=bounded_gravity.endpoint_area_jacobian(points,velocity,faces,dt,6371.)/dt
                reaction=np.zeros_like(points)
                for face,side,multiplier,norm in zip(detail['active_faces'],detail['active_sides'],
                        detail['multipliers_normalized'],detail['constraint_jacobian_norm_km']):
                    np.add.at(reaction,faces[face],(1 if side=='minimum' else -1)*jac[face]*multiplier/norm)
                residual=viscous_sheet.apply(context,velocity)-rhs-reaction
                residual-=points*np.sum(points*residual,axis=1)[:,None];residual[rigid]=0.
                self.assertLessEqual(np.linalg.norm(residual)/np.linalg.norm(rhs),1e-8)


if __name__=='__main__':unittest.main()
