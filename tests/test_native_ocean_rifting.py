"""Native ocean rupture needs current extension and connected mature damage."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np

import ocean_rifting as ocean
from ridge_geometry import rotate


def experiment(sign=1., age=80., weak=True):
    # Two three-face ocean lobes joined by one ocean neck, with external
    # boundary owners at either end. Every edge is explicit native adjacency.
    lon=np.array([-.18,-.12,-.06,.06,.12,.18,-.24,.24])
    lat=np.array([0.,.025,0.,0.,.025,0.,0.,0.])
    xyz=np.column_stack((np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)))
    edges=np.array([[0,1],[1,2],[2,0],[3,4],[4,5],[5,3],[2,3],[6,0],[5,7]])
    mesh=dict(xyz=xyz,area=np.full(8,100000.),edge_faces=edges)
    material=dict(owner=np.array([0,0,0,0,0,0,1,2]),ocean=np.ones(8,bool),age_myr=np.full(8,age))
    plates=dict(omega=np.array([[0.,0.,0.],[0.,0.,-.008*sign],[0.,0.,.008*sign]]),
                uids=np.array([1,2,3]),active=np.ones(3,bool),mantle=np.full((3,3),.002))
    pairs=np.array([[6,0],[5,7]])
    mid=xyz[pairs[:,0]]+xyz[pairs[:,1]];mid/=np.linalg.norm(mid,axis=1)[:,None]
    normal=xyz[pairs[:,1]]-xyz[pairs[:,0]];normal/=np.linalg.norm(normal,axis=1)[:,None]
    boundaries=dict(ba=pairs[:,0],bb=pairs[:,1],bp=material['owner'][pairs[:,0]],bq=material['owner'][pairs[:,1]],
                    bmid=mid,bn=normal,bl=np.full(2,400.))
    weakness=np.array([0.,0.,1.,1.,0.,0.,0.,0.]) if weak else np.zeros(8)
    return mesh,material,plates,boundaries,ocean.initialize(8,weakness=weakness)


class NativeOceanRiftingTests(unittest.TestCase):
    def test_cooling_strength_is_bounded_and_inherited_faults_weaken_old_ocean(self):
        age=np.array([0.,10.,40.,80.,160.,1000.])
        strength=ocean.cooling_strength(age)
        self.assertTrue(np.all(np.diff(strength)>=0))
        self.assertTrue(np.all((strength>=.45)&(strength<=1.8)))
        self.assertLess(ocean.cooling_strength(160.,1.),ocean.cooling_strength(10.,0.))

    def test_shared_rigid_rotation_never_accumulates_damage_or_splits(self):
        mesh,material,plates,boundaries,state=experiment()
        plates['omega'][:]=[.004,-.006,.003]
        state['damage'][:]=1.;state['tensile_strain'][:]=2.
        for _ in range(15):
            result=ocean.update(mesh,material,plates,boundaries,state,2.)
            state=result['history']
            self.assertEqual(result['proposals'],[])
            np.testing.assert_allclose(result['response']['velocity'],0.,atol=1e-13)
        np.testing.assert_array_equal(state['tensile_strain'],2.)
        self.assertTrue(np.all(state['damage']<1.))

    def test_compression_cannot_grow_tensile_damage_or_make_an_ocean_ridge(self):
        mesh,material,plates,boundaries,state=experiment(sign=-1.)
        for _ in range(40):
            result=ocean.update(mesh,material,plates,boundaries,state,2.)
            state=result['history']
            self.assertEqual(result['proposals'],[])
        self.assertLess(float(state['tensile_strain'].max()),1e-6)
        self.assertLess(float(state['damage'].max()),1e-5)

    def test_loaded_primordial_ocean_develops_connected_rupture_without_instant_clock(self):
        mesh,material,plates,boundaries,state=experiment()
        # UID1 is deliberately the primordial identity; no continental parcels
        # exist in this experiment and no minimum plate age is supplied.
        before=deepcopy((mesh,material,plates,state))
        first=ocean.update(mesh,material,plates,boundaries,state,2.)
        self.assertEqual(first['proposals'],[])
        self.assertTrue(np.any(first['history']['tensile_strain']>0))
        state=first['history']; proposal=None
        for _ in range(250):
            result=ocean.update(mesh,material,plates,boundaries,state,2.)
            state=result['history']
            if result['proposals']:
                proposal=result['proposals'][0];break
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal['parent_uid'],1)
        self.assertEqual(len(proposal['child_faces']),3)
        self.assertEqual(len(proposal['parent_faces']),6)
        np.testing.assert_array_equal(proposal['failed_edge_indices'],[6])
        self.assertGreater(proposal['mean_opening_km_myr'],.01)
        # Proposal fitting cannot change plate motion, mantle or any material.
        for actual,original in zip((mesh,material,plates),before[:3]):
            for key in original:np.testing.assert_array_equal(actual[key],original[key])
        self.assertFalse(np.any(before[3]['damage']))
        child=np.isin(proposal['parent_faces'],proposal['child_faces'])
        points=mesh['xyz'][proposal['parent_faces']];area=mesh['area'][proposal['parent_faces']]
        matrices=[]
        for selected in (~child,child):
            matrices.append(np.eye(3)*area[selected].sum()-np.einsum('n,ni,nj->ij',area[selected],points[selected],points[selected]))
        residual=matrices[0]@proposal['delta_omega'][0]+matrices[1]@proposal['delta_omega'][1]
        np.testing.assert_allclose(residual,0.,atol=1e-10)

    def test_no_current_opening_keeps_mature_belt_uncommitted(self):
        mesh,material,plates,boundaries,state=experiment(sign=-1.)
        state['damage'][:]=1.;state['tensile_strain'][:]=1.
        result=ocean.update(mesh,material,plates,boundaries,state,2.)
        self.assertEqual(result['proposals'],[])

    def test_inherited_weakness_localizes_and_accelerates_the_same_loading(self):
        steps=[]
        for weak in (True,False):
            mesh,material,plates,boundaries,state=experiment(weak=weak)
            for step in range(100):
                result=ocean.update(mesh,material,plates,boundaries,state,2.)
                state=result['history']
                if result['proposals']:break
            self.assertTrue(result['proposals'])
            steps.append(step+1)
        self.assertGreater(steps[0],1)
        self.assertLess(steps[0],steps[1])

    def test_damage_that_leaves_an_intact_alternative_path_is_not_a_split(self):
        mesh,material,plates,boundaries,state=experiment()
        # A strong continental link still joins the two ocean lobes. Damage
        # cannot discard that material to make the ocean cut look connected.
        mesh['edge_faces']=np.vstack((mesh['edge_faces'],[1,4]))
        material['ocean'][[1,4]]=False
        state['damage'][:]=1.;state['tensile_strain'][:]=1.
        result=ocean.update(mesh,material,plates,boundaries,state,2.)
        self.assertEqual(result['proposals'],[])

    def test_global_rotation_preserves_response_and_damage(self):
        mesh,material,plates,boundaries,state=experiment()
        baseline=ocean.update(mesh,material,plates,boundaries,state,2.)
        rotation=np.array([.7,-1.1,2.])
        moved_mesh={**mesh,'xyz':rotate(mesh['xyz'],rotation)}
        moved_plates={**plates,'omega':rotate(plates['omega'],rotation)}
        moved_boundary={**boundaries,'bmid':rotate(boundaries['bmid'],rotation),'bn':rotate(boundaries['bn'],rotation)}
        changed=ocean.update(moved_mesh,material,moved_plates,moved_boundary,state,2.)
        np.testing.assert_allclose(changed['response']['velocity'],rotate(baseline['response']['velocity'],rotation),atol=2e-9)
        np.testing.assert_allclose(changed['history']['damage'],baseline['history']['damage'],atol=1e-11)

    def test_stale_boundary_owner_and_disabled_strength_cannot_supply_loading(self):
        mesh,material,plates,boundaries,state=experiment()
        boundaries['bp'][:]=boundaries['bq']
        result=ocean.update(mesh,material,plates,boundaries,state,2.)
        np.testing.assert_array_equal(result['history']['damage'],0.)
        mesh,material,plates,boundaries,state=experiment()
        result=ocean.update(mesh,material,plates,boundaries,state,2.,strength_scale=0.)
        np.testing.assert_array_equal(result['history']['damage'],0.)

    def test_continental_neck_is_never_an_ocean_cut(self):
        mesh,material,plates,boundaries,state=experiment()
        material['ocean'][[2,3]]=False
        state['damage'][:]=1.;state['tensile_strain'][:]=1.
        result=ocean.update(mesh,material,plates,boundaries,state,2.)
        self.assertEqual(result['proposals'],[])

    def test_failed_solver_cannot_acquire_damage_or_split(self):
        mesh,material,plates,boundaries,state=experiment()
        real=ocean.rift_mechanics.solve_loading
        def inaccurate(*args,**kwargs):
            response=real(*args,**kwargs)
            return {**response,'converged':False,'relative_residual':.1}
        with patch.object(ocean.rift_mechanics,'solve_loading',side_effect=inaccurate):
            result=ocean.update(mesh,material,plates,boundaries,state,2.)
        np.testing.assert_array_equal(result['history']['damage'],0.)
        self.assertEqual(result['proposals'],[])
        self.assertFalse(result['diagnostics']['solver_reliable'])

    def test_invalid_history_is_rejected_without_mutating_inputs(self):
        args=list(experiment());args[-1]['damage'][0]=float('nan')
        with self.assertRaises(ValueError):ocean.update(*args,2.)


if __name__=='__main__':unittest.main()
