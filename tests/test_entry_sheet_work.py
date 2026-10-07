"""Independent physical units, energy/work and evolution checks for entry flow."""
import pickle
import unittest
from unittest import mock
import numpy as np
import continental_entry as entry
import gravitational_relaxation as gravity
import material_surface
from ridge_geometry import rotate
from tests.test_continental_entry import geometry,planar_oracle,native_fixture


def setup(density=2800.):
    points,faces=geometry()
    area=material_surface.spherical_face_areas(points,faces)
    volume=area*35.;sheets=np.array([1]);owner=np.zeros(len(points),int)
    normal=np.array([[0.,1.,0.]])
    potential=entry.FrozenPotential(points,faces,volume,volume*1e9*density,sheets,
        np.array([0]),normal,[50.])
    profile=dict(dense_fraction=np.array([(density-2800.)/650.]),sheet_order={})
    return points,faces,area,volume,sheets,owner,potential,profile


def independent_energy(points,faces,volume,potential,density):
    # Existing column reference energy from its vertical mass moment, plus
    # entry depth independently clipped/integrated in the material triangle.
    area=material_surface.spherical_face_areas(points,faces)
    dense=(density-2800.)/650.;h=volume/area
    bottom=-h*density/3300.;mid=bottom+h*dense;top=bottom+h
    moment=.5*(3300.*bottom**2+3450.*(mid**2-bottom**2)+2800.*(top**2-mid**2))
    column=float(np.sum(area*moment*(h-35.)**2/h**2))*9.81e12
    q=6371e3*np.arcsin(points[faces[0]]@potential.normals[0])
    depth=planar_oracle(q)[0]*np.sin(np.deg2rad(50.))
    return column+9.81*(3300.-density)*volume[0]*1e9*depth


def independent_partition(points,faces,area,gradient):
    # Weighted least squares in an explicit 3N x 3 rigid-mode matrix.
    modes=np.stack([np.cross(axis,points).ravel() for axis in np.eye(3)],axis=1)
    weights=np.repeat(np.bincount(faces.ravel(),weights=np.repeat(area/3.,3),minlength=len(points)),3)
    rigid=weights*(modes@np.linalg.solve(modes.T@(weights[:,None]*modes),modes.T@gradient.ravel()))
    return gradient-rigid.reshape(points.shape),rigid.reshape(points.shape),weights.reshape(-1,3)[:,0]


def advance(data,dt,**kwargs):
    p,f,a,v,s,o,potential,profile=data
    return gravity.relax(p,f,v,np.full(len(f),35.),s,dt,
        rigid_mask=kwargs.pop('rigid_mask',np.zeros(len(p),bool)),
        minimum_area_km2=a*.2,maximum_area_km2=a*5.,
        vertex_owner=o,density_profile=profile,entry_potential=potential,
        tolerance=1e-10,**kwargs)


class EntrySheetWorkTests(unittest.TestCase):
    def test_drag_only_motion_matches_newtons_force_and_rigid_work_partition(self):
        data=setup();p,f,a,v,s,o,potential,profile=data
        g=potential.evaluate(p,f,v,s,radius=6371.)['vertex_gradient_j']
        residual,rigid,lumped=independent_partition(p,f,a,g)
        drag=2.*1e23*100e3/(100e3)**2
        expected_velocity=-residual/(6371e3*drag*(lumped*1e6)[:,None])
        expected_velocity*=365.25*86400.*1e6/1000.
        dt=.001
        expected=rotate(p,np.cross(p,expected_velocity)*dt/6371.)
        result,report=advance(data,dt,viscosity_weights=np.zeros(len(f)))
        np.testing.assert_allclose(result,expected,rtol=0.,atol=3e-15)
        self.assertEqual(report['substeps'],1)
        self.assertEqual(report['internal_steps'][0]['velocity_safety_scale'],1.)
        self.assertGreater(np.linalg.norm(result-p),1e-9)
        physical_after=independent_energy(result,f,v,potential,2800.)
        self.assertAlmostEqual(report['energy_after_km4']*gravity.ENERGY_UNIT_J/physical_after,1.,places=12)
        self.assertGreater(report['column_energy_after_km4'],report['column_energy_before_km4'])
        self.assertLess(report['energy_after_km4'],report['energy_before_km4'])
        expected_power=float(np.sum(g*expected_velocity)/6371.)
        self.assertAlmostEqual(report['internal_steps'][0]['virtual_power_km4_myr']*gravity.ENERGY_UNIT_J/expected_power,1.,places=12)
        self.assertLess(report['rigid_mode_partition']['action_reaction_residual'],1e-12)
        # The companion rigid force is exactly the force already supplied to
        # the actual plate balance, not another independently chosen driver.
        import plate_balance as pb
        world=native_fixture();model=pb.Balance(world,1.,slab_tethers=True)
        rigid_load=entry.apply_to_balance(model,face_ids=[917],hinge_normals=[[0.,1.,0.]],
            dip_degrees=[50.],overriding_plate_uids=[int(world.plate_uid[1])])
        for actual,drive in ((report['rigid_mode_partition']['owner_torque_n_m'][0],rigid_load['drive'][:3]),
                (report['rigid_mode_partition']['external_hinge_reaction_n_m'],rigid_load['drive'][3:])):
            expected=drive*pb.RADIUS_M/pb.CM_YR_M_S
            self.assertLess(np.linalg.norm(np.asarray(actual)-expected)/np.linalg.norm(expected),3e-14)

    def test_entry_column_gradient_and_partition_close_arbitrary_virtual_work(self):
        data=setup();p,f,a,v,s,o,potential,profile=data
        p=rotate(p,np.array([[.002,0.,0.],[0.,-.001,0.],[0.,0.,.001]]))
        a=material_surface.spherical_face_areas(p,f)
        value,gradient,_=gravity.energy_gradient(p,f,v/a,np.array([35.]),s,density_profile=profile)
        gradient=gradient*gravity.ENERGY_UNIT_J+potential.evaluate(p,f,v,s,radius=6371.)['vertex_gradient_j']
        rng=np.random.default_rng(237);motion=rng.normal(size=p.shape)
        motion-=p*np.sum(p*motion,axis=1)[:,None]
        h=1e-7
        plus=rotate(p,np.cross(p,motion)*h);minus=rotate(p,-np.cross(p,motion)*h)
        measured=(independent_energy(plus,f,v,potential,2800.)-independent_energy(minus,f,v,potential,2800.))/(2*h)
        self.assertAlmostEqual(float(np.sum(gradient*motion))/measured,1.,places=7)
        part=gravity.rigid_mode_partition(p,f,a,gradient/gravity.ENERGY_UNIT_J,o)
        residual=part['residual']*gravity.ENERGY_UNIT_J
        oracle,rigid,weights=independent_partition(p,f,a,gradient)
        np.testing.assert_allclose(residual,oracle,rtol=2e-11)
        self.assertAlmostEqual(float(np.sum((residual+rigid)*motion))/measured,1.,places=7)
        np.testing.assert_allclose(np.cross(p,residual).sum(axis=0),0.,atol=np.linalg.norm(gradient)*2e-13)

    def test_neutral_and_dense_entry_keep_signed_physical_work(self):
        results={}
        for density in (2800.,3300.,3450.):
            data=setup(density);p,f,a,v,s,o,potential,profile=data
            result,report=advance(data,.001,viscosity_weights=np.zeros(len(f)))
            physical=independent_energy(result,f,v,potential,density)
            np.testing.assert_allclose(report['energy_after_km4']*gravity.ENERGY_UNIT_J,physical,rtol=2e-13,atol=1.)
            if density==3300.:
                np.testing.assert_array_equal(result,p)
            else:
                self.assertLess(report['energy_after_km4'],report['energy_before_km4'])
            results[density]=result-p
        self.assertLess(float(np.sum(results[2800.]*results[3450.])),0.)

    def test_frozen_mass_and_phase_cannot_disagree_and_source_is_not_mutated(self):
        s=native_fixture();before=pickle.dumps(s)
        potential=entry.FrozenPotential.from_state(s,face_ids=[917],hinge_normals=[[0.,1.,0.]],dip_degrees=[50.])
        self.assertEqual(pickle.dumps(s),before)
        self.assertFalse(potential.masses.flags.writeable)
        with self.assertRaisesRegex(ValueError,'different conserved mass'):
            potential.validate_density(dict(dense_fraction=np.array([.5])))
        with self.assertRaisesRegex(ValueError,'inventory'):
            potential.evaluate(s.material_surface['vertices'],potential.faces,potential.volumes*1.01,potential.sheets,radius=6371.)
        with self.assertRaisesRegex(ValueError,'identity is absent'):
            entry.FrozenPotential.from_state(s,face_ids=[999],hinge_normals=[[0.,1.,0.]],dip_degrees=[50.])
        s.structure['thickness_km']*=2.
        np.testing.assert_allclose(potential.volumes,s.mass*35.,rtol=0.,atol=0.)

    def test_new_stack_intersection_rejects_trial_before_spending_entry_energy(self):
        p,f,a,v,s,o,potential,profile=setup()
        points=np.vstack((p,rotate(p,[0.,0.,.2])))
        faces=np.vstack((f,f+3));volumes=np.repeat(v,2);sheets=np.array([1,2])
        potential=entry.FrozenPotential(points,faces,volumes,volumes*2800e9,sheets,np.array([0]),[[0.,1.,0.]],[50.])
        trial=points.copy();trial[:3]=trial[3:]
        before=points.copy()
        with self.assertRaisesRegex(entry.EntryGeometryError,'material stack'):
            potential.evaluate(trial,faces,volumes,sheets,radius=6371.)
        np.testing.assert_array_equal(points,before)

    def test_full_energy_work_gate_catches_false_descent_of_projected_force(self):
        data=setup();p,f,a,v,s,o,potential,profile=data
        g=potential.evaluate(p,f,v,s,radius=6371.)['vertex_gradient_j']
        residual,rigid,lumped=independent_partition(p,f,a,g)
        ratio=2.*np.sum(residual**2/lumped[:,None])/np.sum(rigid**2/lumped[:,None])
        velocity=(-residual+ratio*rigid)/lumped[:,None]
        velocity/=np.linalg.norm(velocity)
        self.assertLess(np.sum(residual*velocity),0.)
        self.assertGreater(np.sum(g*velocity),0.)
        before=pickle.dumps(data)
        with mock.patch('viscous_sheet.solve',return_value=(velocity,dict(converged=True))):
            with self.assertRaisesRegex(ValueError,'negative virtual-work'):
                advance(data,1.)
        self.assertEqual(pickle.dumps(data),before)

    def test_full_interval_refinement_and_restart_at_fixed_hinge(self):
        data=setup();p,f,a,v,s,o,potential,profile=data
        endpoints={}
        for count in (1,2,4,32):
            current=p.copy();time=0.
            for step in range(count):
                local=(current,f,a,v,s,o,potential,profile)
                current,report=advance(local,.4/count)
                time+=report['completed_dt_myr']
            self.assertAlmostEqual(time,.4,places=14)
            endpoints[count]=current
        errors=[np.linalg.norm(endpoints[n]-endpoints[32]) for n in (1,2,4)]
        self.assertTrue(all(b<.65*a for a,b in zip(errors,errors[1:])),errors)
        halfway,_=advance(data,.2)
        resumed=entry.FrozenPotential(halfway,f,v,v*2800e9,s,np.array([0]),potential.normals,potential.dips)
        actual,_=advance((halfway,f,a,v,s,o,resumed,profile),.2)
        np.testing.assert_array_equal(actual,endpoints[2])

    def test_column_only_work_gate_also_checks_actual_motion(self):
        from tests.test_gravitational_relaxation import fixture
        p,f,h,reference,sheets=fixture()
        area=material_surface.spherical_face_areas(p,f);volume=area*h
        owners=np.repeat([0,1],3)
        _,gradient,_=gravity.energy_gradient(p,f,h,reference,sheets)
        residual=np.empty_like(gradient);rigid=np.empty_like(gradient);lumped=np.empty(len(p))
        for index in range(2):
            selection=slice(3*index,3*index+3)
            residual[selection],rigid[selection],lumped[selection]=independent_partition(
                p[selection],np.array([[0,1,2]]),area[index:index+1],gradient[selection])
        ratio=2.*np.sum(residual**2/lumped[:,None])/np.sum(rigid**2/lumped[:,None])
        velocity=(-residual+ratio*rigid)/lumped[:,None]
        velocity/=np.linalg.norm(velocity)
        self.assertLess(np.sum(residual*velocity),0.)
        self.assertGreater(np.sum(gradient*velocity),0.)
        with mock.patch('viscous_sheet.solve',return_value=(velocity,dict(converged=True))):
            with self.assertRaisesRegex(ValueError,'negative virtual-work'):
                gravity.relax(p,f,volume,reference,sheets,1.,rigid_mask=np.zeros(len(p),bool),
                    minimum_area_km2=area*.2,maximum_area_km2=area*5.,vertex_owner=owners)

    def test_large_interval_backtracks_on_combined_energy_and_completes_time(self):
        data=setup();p,f,a,v,s,o,potential,profile=data
        before=pickle.dumps(data)
        result,report=advance(data,1000.)
        self.assertEqual(pickle.dumps(data),before)
        self.assertEqual(report['completed_dt_myr'],1000.)
        self.assertGreater(report['backtracks'],0)
        self.assertGreater(report['substeps'],1)
        for row in report['internal_steps']:
            self.assertLess(row['energy_after_km4'],row['energy_before_km4'])
            self.assertLess(row['virtual_power_km4_myr'],0.)
        self.assertAlmostEqual(report['energy_after_km4']*gravity.ENERGY_UNIT_J/
            independent_energy(result,f,v,potential,2800.),1.,places=12)

    def test_sheet_evolution_covaries_when_material_and_hinge_rotate_together(self):
        data=setup();p,f,a,v,s,o,potential,profile=data
        result,report=advance(data,.4)
        spin=np.array([.7,-.2,.5]);moved=rotate(p,spin)
        other=entry.FrozenPotential(moved,f,v,potential.masses,s,potential.indices,
            rotate(potential.normals,spin),potential.dips)
        changed,again=advance((moved,f,a,v,s,o,other,profile),.4)
        np.testing.assert_allclose(changed,rotate(result,spin),rtol=0.,atol=3e-13)
        self.assertAlmostEqual(again['energy_after_km4']/report['energy_after_km4'],1.,places=12)

    def test_short_entry_stage_resolves_active_physical_area_bound(self):
        # A fixed 64-epsilon inward solver target left a finite reaction gap:
        # normalized complementarity grew as 1/dt despite exact force balance.
        # These stages must solve at the physical bound, without loosening the
        # force/complementarity tolerances or dropping requested physical time.
        p,f,a,v,s,o,potential,profile=setup()
        for dt in (1e-5,.001,.1):
            result,report=gravity.relax(p,f,v,np.array([35.]),s,dt,
                rigid_mask=np.zeros(3,bool),minimum_area_km2=a.copy(),maximum_area_km2=a*5.,
                vertex_owner=o,density_profile=profile,entry_potential=potential,
                constraint_version=1,tolerance=1e-10)
            self.assertEqual(report['completed_dt_myr'],dt)
            self.assertLess(report['energy_after_km4'],report['energy_before_km4'])
            self.assertGreater(np.linalg.norm(result-p),0.)
            bound=report['internal_steps'][0]['bound_constraints']
            self.assertEqual(bound['active_sides'],['minimum'])
            self.assertLessEqual(bound['kkt_relative_residual'],1e-10)
            self.assertLessEqual(bound['normalized_complementarity'],1e-10)
            self.assertTrue(bound['original_physical_bounds_unchanged'])
            self.assertEqual(bound['inward_roundoff_padding_km2'],[0.])
            actual=material_surface.spherical_face_areas(result,f)
            np.testing.assert_allclose(actual,a,rtol=3e-14,atol=0.)


if __name__=='__main__':unittest.main()
