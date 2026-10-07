"""Independent area integration, virtual work, refinement and force oracles."""
import pickle
import unittest
import numpy as np
import continental_entry as entry
import material_surface
import plate_balance as pb
from ridge_geometry import rotate
from tests.test_slab_tether_forces import fixture as force_fixture
from tests.test_slab_tether_evolution import linear_forces


def unit(x):
    x=np.asarray(x,float);return x/np.linalg.norm(x,axis=-1,keepdims=True)


def geometry():
    return unit([[1.,-.02,-.02],[1.,.03,-.01],[1.,.01,.04]]),np.array([[0,1,2]])


def native_fixture():
    s=force_fixture();s.plate_resistance_version=1
    p,f=geometry();area=material_surface.spherical_face_areas(p,f)
    s.material_surface=dict(vertices=p,faces=f,face_id=np.array([917]),face_owner=np.array([0]),
        vertex_owner=np.zeros(3,int),area_km2=area,reference_area_km2=area.copy())
    s.structure=dict(thickness_km=np.array([35.]),area_factor=np.ones(1))
    s.parcel_plate=np.array([0]);s.mass=area.copy();s.collision_contacts=[]
    return s


def domain(s):
    return dict(face_ids=[917],hinge_normals=[[0.,1.,0.]],dip_degrees=[50.],
                overriding_plate_uids=[int(s.plate_uid[1])])


def planar_oracle(q):
    """Clip a 2-D reference triangle, then integrate via its area and centroid."""
    vertices=[np.array(v,float) for v in ((0.,0.),(1.,0.),(0.,1.))]
    value=lambda p:q[0]+(q[1]-q[0])*p[0]+(q[2]-q[0])*p[1]
    clipped=[];old=vertices[-1];before=value(old)
    for point in vertices:
        after=value(point)
        if (before>0)!=(after>0):clipped.append(old+(point-old)*before/(before-after))
        if after>0:clipped.append(point)
        old,before=point,after
    if len(clipped)<3:return 0.,np.zeros(3)
    p=np.asarray(clipped);next=np.roll(p,-1,axis=0)
    cross=p[:,0]*next[:,1]-p[:,1]*next[:,0];twice_area=cross.sum()
    center=((p+next)*cross[:,None]).sum(axis=0)/(3.*twice_area)
    weights=twice_area*np.array([1.-center.sum(),*center])
    return twice_area*value(center),weights


class ContinentalEntryTests(unittest.TestCase):
    def test_horizontal_surface_candidate_keeps_reciprocal_entry_work(self):
        p,f=geometry();normal=np.array([[0.,1.,0.]])
        args=([1000.],[2800.e12],normal,[50.])
        bounds=dict(finite_midpoints=np.array([[1.,0.,0.]]),
                    finite_half_lengths_km=np.array([200.]))
        old=entry.evaluate(p,f,*args,**bounds)
        new=entry.evaluate(p,f,*args,**bounds,
                           coordinate_mode='horizontal_surface')
        ratio=1./np.cos(np.deg2rad(50.))
        for name in ('energy_j','vertex_gradient_j',
                     'hinge_normal_gradient_j','entering_potential_torque_n_m',
                     'hinge_potential_torque_n_m','mean_entry_depth_m',
                     'corner_entry_depth_m'):
            np.testing.assert_allclose(new[name],old[name]*ratio,rtol=3e-15,
                                       atol=1e-8)
        np.testing.assert_array_equal(new['entered_reference_fraction'],
                                      old['entered_reference_fraction'])
        self.assertIn('horizontal',new['coordinate'])
        axis=unit([.2,-.1,.3]);step=1e-7
        plus=p.copy();minus=p.copy()
        plus[1]=rotate(p[[1]],axis*step)[0]
        minus[1]=rotate(p[[1]],-axis*step)[0]
        measured=(entry.evaluate(plus,f,*args,**bounds,
                   coordinate_mode='horizontal_surface')['energy_j']
                  -entry.evaluate(minus,f,*args,**bounds,
                   coordinate_mode='horizontal_surface')['energy_j'])/(2.*step)
        predicted=np.cross(p[1],new['vertex_gradient_j'][1])@axis
        np.testing.assert_allclose(measured,predicted,rtol=3e-8)
        with self.assertRaisesRegex(ValueError,'supported dip coordinate'):
            entry.evaluate(p,f,*args,coordinate_mode='unspecified')

    def test_frozen_torque_assembly_preserves_reaction_and_does_not_mutate_inputs(self):
        incoming=np.array([2,2]);overriding=np.array([0,1])
        torque=np.array([[3.,-2.,5.],[-7.,11.,13.]])
        before=torque.copy()
        drive=entry._plate_drive(9,{0:0,1:1,2:2},incoming,overriding,
                                 torque,-torque,2.)
        np.testing.assert_array_equal(drive[:3],-2.*(-torque[0]))
        np.testing.assert_array_equal(drive[3:6],-2.*(-torque[1]))
        np.testing.assert_array_equal(drive[6:],-2.*torque.sum(axis=0))
        np.testing.assert_allclose(drive.reshape(3,3).sum(axis=0),0.,atol=0.)
        np.testing.assert_array_equal(torque,before)

    def test_tiny_entered_sliver_is_not_lost_by_complement_subtraction(self):
        from decimal import Decimal,localcontext
        from itertools import permutations
        p,n=1e-12,1e12
        with localcontext() as context:
            context.prec=90
            positive=Decimal.from_float(p);negative=Decimal.from_float(n)
            end=positive/(positive+negative)
            # Integrate 2(1-lambda)*(p-(p+n)*lambda) over the entered strip.
            expected=float(2*positive*end-(2*positive+negative)*end**2
                           +Decimal(2)/3*(positive+negative)*end**3)
        for q in set(permutations((-n,p,p))):
            mean,w=entry.positive_linear_average([q])
            self.assertAlmostEqual(mean[0]/expected,1.,places=14)
            self.assertTrue(np.all(w>=0.))

    def test_local_positive_depth_matches_independent_clipped_area_integral(self):
        rng=np.random.default_rng(731)
        q=np.vstack(([-2.,1.,1.],[-1.,-1.,1.],[0.,0.,0.],rng.normal(size=(40,3))))
        mean,weights=entry.positive_linear_average(q)
        for i,v in enumerate(q):
            expected,w=planar_oracle(v)
            self.assertAlmostEqual(mean[i],expected,places=13)
            np.testing.assert_allclose(weights[i],w,atol=2e-15)
        self.assertEqual(max(float(q[0].mean()),0.),0.)
        self.assertAlmostEqual(mean[0],8./27.,places=14)

    def test_reference_subdivision_preserves_entry_energy_and_gradient(self):
        maps=np.array([[[1,0,0],[.5,.5,0],[.5,0,.5]],
                       [[.5,.5,0],[0,1,0],[0,.5,.5]],
                       [[.5,0,.5],[0,.5,.5],[0,0,1]],
                       [[.5,.5,0],[0,.5,.5],[.5,0,.5]]])
        for q in ([-2.,1.,1.],[-1.,-1.,1.],[1.,2.,3.],[-3.,-2.,-1.]):
            q=np.array(q);whole,gradient=entry.positive_linear_average(q[None])
            children,w=entry.positive_linear_average(maps@q)
            self.assertAlmostEqual(children.mean(),whole[0],places=14)
            np.testing.assert_allclose(np.einsum('fvi,fv->i',maps,w)/4.,gradient[0],atol=2e-15)

    def test_spherical_virtual_work_and_overrider_reaction_match_energy_differences(self):
        p,f=geometry();n=np.array([[0.,1.,0.]])
        args=([1000.],[2800.*1000.*1e9],n,[50.])
        result=entry.evaluate(p,f,*args)
        rng=np.random.default_rng(41);step=1e-7
        for k in range(3):
            axis=unit(rng.normal(size=3));plus=p.copy();minus=p.copy()
            plus[k]=rotate(p[[k]],axis*step)[0];minus[k]=rotate(p[[k]],-axis*step)[0]
            measured=(entry.evaluate(plus,f,*args)['energy_j']-entry.evaluate(minus,f,*args)['energy_j'])/(2*step)
            predicted=np.cross(p[k],result['vertex_gradient_j'][k])@axis
            np.testing.assert_allclose(predicted,measured,rtol=2e-8)
        axis=unit([.1,.3,.7])
        plus=entry.evaluate(p,f,args[0],args[1],rotate(n,axis*step),args[3])['energy_j']
        minus=entry.evaluate(p,f,args[0],args[1],rotate(n,-axis*step),args[3])['energy_j']
        np.testing.assert_allclose((plus-minus)/(2*step),result['hinge_potential_torque_n_m'][0]@axis,rtol=2e-8)
        np.testing.assert_allclose(result['entering_potential_torque_n_m']+result['hinge_potential_torque_n_m'],0.,
                                   atol=np.linalg.norm(result['entering_potential_torque_n_m'])*3e-15)

    def test_common_rotation_and_signed_density_do_not_manufacture_work(self):
        p,f=geometry();normal=np.array([[0.,1.,0.]])
        ordinary=entry.evaluate(p,f,[1000.],[2800.e12],normal,[50.])
        spin=np.array([.7,-.2,.3])
        moved=entry.evaluate(rotate(p,spin),f,[1000.],[2800.e12],rotate(normal,spin),[50.])
        self.assertAlmostEqual(moved['energy_j']/ordinary['energy_j'],1.,places=13)
        for density,ratio in ((3300.,0.),(3450.,-.3)):
            state=entry.evaluate(p,f,[1000.],[density*1e12],normal,[50.])
            np.testing.assert_allclose(state['vertex_gradient_j'],ordinary['vertex_gradient_j']*ratio,rtol=2e-15,atol=1e-10)
            self.assertAlmostEqual(state['energy_j']/ordinary['energy_j'],ratio,places=14)

    def test_spherical_refinement_converges_to_independent_reference_area_quadrature(self):
        nodes,weights=np.polynomial.legendre.leggauss(60)
        x=(nodes+1.)*.015;y=nodes*.02
        xx,yy=np.meshgrid(x,y,indexing='ij')
        depth=6371e3*np.arcsin(xx/np.sqrt(1.+xx*xx+yy*yy))
        expected=float(np.sum(depth*weights[:,None]*weights[None,:])*.015*.02/(.04*.04))
        errors=[]
        for count in (2,4,8,16):
            points=unit([[1.,a,b] for a in np.linspace(-.01,.03,count+1) for b in np.linspace(-.02,.02,count+1)])
            faces=[]
            for i in range(count):
                for j in range(count):
                    a=i*(count+1)+j;b=a+count+1
                    faces.extend(([a,b,a+1],[b,b+1,a+1]))
            faces=np.array(faces);number=len(faces)
            result=entry.evaluate(points,faces,np.full(number,1000./number),np.full(number,2800.e12/number),
                                  np.tile([0.,1.,0.],(number,1)),np.full(number,50.))
            mean=result['energy_j']/(9.81*500.e12*np.sin(np.deg2rad(50.)))
            errors.append(abs(mean-expected))
        self.assertTrue(all(b<a*.4 for a,b in zip(errors,errors[1:])),errors)

    def test_native_mass_uses_existing_phase_volumes_without_changing_state(self):
        s=native_fixture();s.retained_dense_crust_version=1
        s.structure['dense_crust_km_per_reference_km2']=np.array([21.])
        before=pickle.dumps(s);volume,mass=entry.native_inventory(s)
        expected_volume=s.material_surface['area_km2']*35.
        np.testing.assert_allclose(volume,expected_volume,rtol=0.,atol=0.)
        np.testing.assert_allclose(mass,expected_volume*1e9*(.4*2800.+.6*3450.),rtol=2e-15)
        self.assertEqual(pickle.dumps(s),before)
        s.structure['area_factor']*=.5
        with self.assertRaisesRegex(ValueError,'physical native'):entry.native_inventory(s)

    def test_actual_native_phase_checkpoint_repeats_frozen_entry_force(self):
        from pathlib import Path
        import tempfile,checkpoint,phase_evolution
        from native_engine import Simulation
        from tests.test_native_processes import ocean_fixture
        s=ocean_fixture(two=True)
        s._add_arc_crust(np.array([12]),np.array([1000.]))
        s._rasterize();s._boundaries()
        s.plate_balance_version=1;s.plate_resistance_version=1
        phase_evolution.upgrade(s,1000.)
        owner=int(s.parcel_plate[0]);other=1-owner
        center=unit(s.material_surface['vertices'][s.material_surface['faces'][0]].sum(axis=0))
        normal=unit(np.cross(center,[.3,.4,.7])+.02*center)
        spec=dict(face_ids=s.material_surface['face_id'][:1],hinge_normals=normal[None],dip_degrees=[50.],
                  overriding_plate_uids=[int(s.plate_uid[other])])
        model=pb.Balance(s,1.);report=entry.apply_to_balance(model,**spec);model.solve()
        self.assertGreater(report['energy_j'],0.)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'world.npz'
            checkpoint.write_checkpoint(path,s,dict(config=s.config),{})
            resumed,_=checkpoint.read_checkpoint(path,None,Simulation)
        again=pb.Balance(resumed,1.);copy=entry.apply_to_balance(again,**spec);again.solve()
        self.assertEqual(copy['energy_j'],report['energy_j'])
        np.testing.assert_array_equal(again.x,model.x)
        np.testing.assert_array_equal(entry.native_inventory(resumed)[1],entry.native_inventory(s)[1])

    def test_existing_material_stack_cannot_also_receive_oceanic_ramp_energy(self):
        s=native_fixture();mesh=s.material_surface
        mesh['vertices']=np.vstack((mesh['vertices'],mesh['vertices']))
        mesh['faces']=np.array([[0,1,2],[3,4,5]])
        mesh['face_id']=np.array([917,918]);s.parcel_plate=np.array([0,1])
        model=pb.Balance(s,1.,slab_tethers=True)
        before=model.torque.copy()
        with self.assertRaisesRegex(ValueError,'material stack'):entry.apply_to_balance(model,**domain(s))
        np.testing.assert_array_equal(model.torque,before)

    def test_actual_slab_balance_slows_stalls_and_rebounds_as_same_continent_enters(self):
        s=native_fixture();points=s.material_surface['vertices'].copy()
        volume,mass=entry.native_inventory(s)
        with linear_forces():
            def motion(angle):
                s.material_surface['vertices']=rotate(points,[0.,0.,angle])
                model=pb.Balance(s,1.,slab_tethers=True)
                before=pickle.dumps(s);result=entry.apply_to_balance(model,**domain(s));model.solve()
                self.assertEqual(pickle.dumps(s),before)
                np.testing.assert_allclose(result['volumes_km3'],volume,rtol=3e-14)
                np.testing.assert_allclose(result['masses_kg'],mass,rtol=3e-14)
                np.testing.assert_allclose(result['drive'][:3]+result['drive'][3:],0.,atol=max(np.linalg.norm(result['drive'])*1e-14,1e-20))
                oracle=np.linalg.solve(model.stiffness,model.torque)
                np.testing.assert_allclose(model.x,oracle,rtol=2e-13,atol=1e-14)
                return model.x[2]-model.x[5]
            before,partial,after=[motion(angle) for angle in (-.05,0.,.05)]
            self.assertGreater(before,partial);self.assertGreater(partial,0.);self.assertLess(after,0.)
            low,high=0.,.05
            for _ in range(42):
                middle=.5*(low+high)
                if motion(middle)>0.:low=middle
                else:high=middle
            self.assertAlmostEqual(motion(.5*(low+high))/before,0.,places=11)

    def test_identity_reuse_double_application_and_invalid_geometry_are_rejected(self):
        s=native_fixture();model=pb.Balance(s,1.,slab_tethers=True)
        wrong=domain(s);wrong['face_ids']=[999]
        with self.assertRaisesRegex(ValueError,'identity is absent'):entry.apply_to_balance(model,**wrong)
        entry.apply_to_balance(model,**domain(s))
        with self.assertRaisesRegex(ValueError,'already'):entry.apply_to_balance(model,**domain(s))
        p,f=geometry()
        with self.assertRaises(ValueError):entry.evaluate(p,f,[-1.],[1.],[[0.,1.,0.]],[50.])
        with self.assertRaises(ValueError):entry.evaluate(p,f,[1.],[1.],[[0.,1.,0.]],[90.])
        with self.assertRaisesRegex(ValueError,'pole'):entry.evaluate(p,f,[1.],[1.],p[[0]],[50.])

    def test_entry_buoyancy_increases_local_neck_extension_in_coupled_damage_solve(self):
        import slab_tether_local,slab_tether_history,slab_tether_evolution
        s=native_fixture();row=s.trench_systems[0]
        slab_tether_local.localize(row,s.bmid[:1],[1.],support_radius_km=1000.)
        before=pickle.dumps(s)
        with linear_forces():
            free,a=slab_tether_evolution.integrate_frozen_geometry(s,.01)
            loaded,b=slab_tether_evolution.integrate_frozen_geometry(s,.01,continental_entry_domain=domain(s))
        self.assertEqual(pickle.dumps(s),before)
        field=slab_tether_history.FIELD
        self.assertGreater(loaded.trench_systems[0][field][0]['neck']['damage'],
                           free.trench_systems[0][field][0]['neck']['damage'])
        self.assertLess(loaded.omega[0,2]-loaded.omega[1,2],free.omega[0,2]-free.omega[1,2])
        self.assertTrue(b['continental_entry_domain_supplied']);self.assertFalse(a['continental_entry_domain_supplied'])
        self.assertGreater(b['frozen_continental_entry_energy_j'],0.)
        self.assertLessEqual(b['maximum_scaled_force_residual'],pb.FORCE_RELATIVE_TOLERANCE)


if __name__=='__main__':unittest.main()
