"""Surface-gated erosion: physical requests, conservation and saved policy."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import crustal_structure as columns
import collision_surface
import mesh_geometry
import mesh_transport
import native_frame_sampling as sampling
import structure_engine
import surface_erosion as erosion
from tests.test_collision_surface_acceptance import world, saved, triangle, unit, divide, triangle_area
from tests.test_collision_erosion_timing import fixture


def live_ocean(s):
    f = saved(s)
    s.native_mesh = dict(vertices=f['mesh_vertices'], faces=f['mesh_faces'], area_km2=f['mesh_area_km2'])
    s.native_locator = mesh_geometry.build_locator(f['mesh_vertices'], f['mesh_faces'])
    s.native_stencil = mesh_transport.face_vertex_stencil(s.native_mesh)
    s.plate = f['mesh_plate']; s.crust = f['mesh_crust']; s.age = f['mesh_age_myr']
    s.ocean_relief = f['mesh_ocean_relief_m']
    s.support = np.zeros((len(s.plate_uid), len(s.plate)))
    s.support[0] = 1.
    s.ridge_episodes = []
    s.collision_coast_version = 2
    erosion.upgrade(s)
    return s


def sample_saved(s):
    f=saved(s);ctx=sampling.prepare(f)
    return lambda p: sampling.sample_frame(f,p,prepared=ctx)


class SurfaceErosionTests(unittest.TestCase):
    def test_area_quadrature_positive_normalized_rotation_and_refinement(self):
        tri=triangle();faces=np.array([[0,1,2]])
        p,w=erosion.quadrature(tri,faces,2)
        self.assertTrue(np.all(w>0));self.assertAlmostEqual(float(w.sum()),1.)
        # Analytic constant exposed field; exact independently of subdivision.
        def constant(p):return dict(raw_elevation_m=np.full(len(p),123.),material_face=np.zeros(len(p),int))
        h,d=erosion.integrate(tri,faces,constant)
        np.testing.assert_allclose(h,123.);np.testing.assert_allclose(d,1.)
        from orientation import rotation_matrix
        r=rotation_matrix(dict(yaw=33.,pitch=-47.,roll=19.))
        pr,wr=erosion.quadrature(tri@r.T,faces,2)
        np.testing.assert_allclose(pr,p@r.T,atol=1e-14)
        np.testing.assert_allclose(wr,w,atol=1e-13)
        v,f=divide(tri);pp,ww=erosion.quadrature(v,f,1)
        weights=np.array([triangle_area(v[x]) for x in f]);weights/=weights.sum()
        np.testing.assert_allclose(pp.reshape(-1,3),p.reshape(-1,3),atol=1e-14)
        np.testing.assert_allclose((ww*weights[:,None]).ravel(),w.ravel(),atol=1e-13)
        skinny=unit([[1.,0.,0.],[1.,.001,0.],[1.,.0005,1e-8]])@r.T
        _,thin_weights=erosion.quadrature(skinny,faces)
        self.assertTrue(np.isfinite(thin_weights).all())
        self.assertTrue(np.all(thin_weights>0))
        self.assertAlmostEqual(float(thin_weights.sum()),1.)

    def test_wet_and_buried_samples_never_contribute_and_partial_face_is_integrated(self):
        tri=triangle();faces=np.array([[0,1,2]])
        def half(p):
            dry=p[:,1]>0.
            return dict(raw_elevation_m=np.where(dry,1000.,-1000.),material_face=np.zeros(len(p),int))
        h,d=erosion.integrate(tri,faces,half,level=3)
        self.assertGreater(h[0],300.);self.assertLess(h[0],700.)
        np.testing.assert_allclose(h,1000*d)
        for height,face in ((-1000.,0),(5000.,1)):
            def sample(p):return dict(raw_elevation_m=np.full(len(p),height),material_face=np.full(len(p),face))
            h,d=erosion.integrate(tri,faces,sample)
            np.testing.assert_array_equal(h,0.);np.testing.assert_array_equal(d,0.)

    def test_real_shared_surface_submerged_shelf_and_uplifted_emergence(self):
        # A wholly submerged neutral footprint contributes no erosion request.
        tri=unit([[1.,-.002,-.001],[1.,.002,-.001],[1.,0.,.002]])
        s=world([(tri,35.,-100.,False)])
        collision_surface.refresh(s);live_ocean(s)
        h=erosion.prepare(s)
        np.testing.assert_array_equal(h,0.)
        h_saved,d=erosion.integrate(s.material_surface['vertices'],s.material_surface['faces'],sample_saved(s))
        np.testing.assert_allclose(h,h_saved,atol=1e-8)
        # A wider footprint has both dry interior and wet coast; uplift is
        # applied to columns, not to the coastal datum or the erosion routine.
        wide=unit([[1.,-.24,-.15],[1.,.24,-.15],[1.,0.,.24]])
        s=world([(wide,35.,1000.,False)])
        collision_surface.refresh(s);live_ocean(s)
        base=erosion.prepare(s)
        self.assertGreater(base[0],0.)
        self.assertLess(s.surface_erosion_diagnostics['dry_area_km2'],s.mass.sum())
        s.relief+=3000.
        uplifted=erosion.prepare(s)
        self.assertGreater(uplifted[0],base[0])
        expected,_=erosion.integrate(s.material_surface['vertices'],s.material_surface['faces'],sample_saved(s))
        np.testing.assert_allclose(uplifted,expected,atol=1e-8)
        stacked=world([(triangle(),35.,1000.,False),(triangle(),40.,5000.,False)])
        collision_surface.refresh(stacked);live_ocean(stacked)
        stacked_height=erosion.prepare(stacked)
        self.assertGreater(stacked_height[0],0.)
        self.assertEqual(stacked_height[1],0.)

    def test_positive_column_at_submarine_shelf_requests_zero(self):
        from tests.test_coastal_deformation import CoastalDeformationTests
        s,frame,point=CoastalDeformationTests().shelf_fixture()
        ctx=sampling.prepare(frame)
        result=sampling.sample_frame(frame,point,prepared=ctx)
        self.assertAlmostEqual(result['elevation'][0],-129.375,places=6)
        self.assertGreater(columns.elevation(s.structure)[0],0.)
        # A small integration cell wholly inside that shelf. The shared
        # sampler still uses the complete continent and its actual coastline.
        from tests.test_native_gospl_sampling import triangle as cell
        v=cell(point[0],radius=.00001)
        h,d=erosion.integrate(v,np.array([[0,1,2]]),
            lambda p:sampling.sample_frame(frame,p,prepared=ctx))
        np.testing.assert_array_equal(h,0.);np.testing.assert_array_equal(d,0.)
        s.config=dict(erosion=1.)
        old=structure_engine._erosion_plan(s,s.structure,s.kind,columns.elevation(s.structure),2.,np.ones(1))
        new=structure_engine._erosion_plan(s,s.structure,s.kind,h,2.,np.ones(1))
        self.assertGreater(old['denudation_m'][0],0.)
        np.testing.assert_array_equal(new['denudation_m'],0.)

    def test_real_deform_shared_timing_and_volume_rebound_closure(self):
        s=live_ocean(fixture());s.erosion_relief_version=1
        old=deepcopy(s.structure);zero=np.zeros(s.n)
        seen=[];prepare=erosion.prepare
        def capture(s):
            h=prepare(s);seen.append(h.copy());return h
        with patch.object(erosion,'prepare',side_effect=capture):
            result=structure_engine.deform(s,zero,zero,zero,2.)
        self.assertEqual(len(seen),1)
        self.assertGreater(seen[0][0],0.)
        self.assertEqual(s.structure['denudation_m'][0],s.trace_structure['denudation_m'][0])
        self.assertEqual(result['parcel']['net_erosion_m'][0],result['trace']['net_erosion_m'][0])
        self.assertFalse(any(k.startswith('_') for b in result.values() for k in b))
        removed=(old['thickness_km']-s.structure['thickness_km'])*1000.
        loss=columns.elevation(old)-columns.elevation(s.structure)
        np.testing.assert_allclose(removed,s.structure['denudation_m'],atol=1e-9)
        np.testing.assert_allclose(removed-loss,s.structure['rebound_m'],atol=1e-9)
        np.testing.assert_allclose(loss,seen[0]*(-np.expm1(-2./180.)),atol=1e-8)
        self.assertAlmostEqual(float(s.material_surface['area_km2']@removed/1000.),
                               float(s.material_surface['area_km2']@s.structure['denudation_m']/1000.),places=7)

    def test_zero_erosion_skips_sampling_and_preserves_columns(self):
        s=live_ocean(fixture(0.));before=s.structure['thickness_km'].copy()
        with patch.object(erosion,'prepare',side_effect=AssertionError('zero erosion needs no sampling')):
            structure_engine.deform(s,np.zeros(s.n),np.zeros(s.n),np.zeros(s.n),2.)
        np.testing.assert_array_equal(before,s.structure['thickness_km'])
        self.assertTrue(s.surface_erosion_diagnostics['sampling_skipped'])

    def test_explicit_policy_validation_and_checkpoint_roundtrip(self):
        s=fixture();before=deepcopy(s.structure)
        self.assertEqual(erosion.version(s),0)
        with self.assertRaises(ValueError):erosion.upgrade(s)
        s.collision_coast_version=2
        receipt=erosion.upgrade(s);self.assertEqual(erosion.upgrade(s),receipt)
        for k in before:np.testing.assert_array_equal(s.structure[k],before[k])
        import checkpoint
        arrays={};encoded=checkpoint._encode(erosion.snapshot(s),arrays)
        restored=checkpoint._decode(encoded,arrays)
        self.assertEqual(restored[erosion.FIELD],1)
        for value in (-1,2,True,1.,'1'):
            s.surface_erosion_version=value
            with self.assertRaises(ValueError):erosion.version(s)
        s.surface_erosion_version=1;s.collision_coast_version=1
        with self.assertRaises(ValueError):structure_engine.deform(s,np.zeros(s.n),np.zeros(s.n),np.zeros(s.n),2.)
        for k in before:np.testing.assert_array_equal(s.structure[k],before[k])

    def test_trace_mapping_and_export_policy_boundary(self):
        s=fixture();s.trace_patch=s.parcel_patch[[1,0,1]]
        np.testing.assert_array_equal(erosion.trace_values(s,[12.,35.]),[35.,12.,35.])
        s.trace_patch=np.array([99999])
        with self.assertRaises(ValueError):erosion.trace_values(s,[12.,35.])
        s=live_ocean(fixture());a=saved(s);b=deepcopy(a)
        a[erosion.FIELD]=0;b[erosion.FIELD]=1;b['time_myr']=2.
        import gospl_export
        with self.assertRaisesRegex(ValueError,'mix surface erosion'):
            gospl_export.interval_forcing(s.pos,a,b,{})

    def test_native_phase_step_checkpoint_restart_and_saved_surface(self):
        import checkpoint, native_engine, dense_crust, mesh_history, tempfile
        from pathlib import Path
        s=native_engine.Simulation(dict(width=48,height=24,mesh_level=2,coast_geometry_level=2,
            plate_count=4,mechanics_nodes=128,seed=37,physics_profile='reviewed_v1',
            primordial_subduction={'enabled':True},subduction_response='moving_hinge_v1',
            retained_phases='thermal_v1'))
        s.collision_coast_version=2;erosion.upgrade(s)
        self.assertEqual(s.erosion_relief_version,2)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'surface-erosion.npz'
            checkpoint.write_checkpoint(path,s,dict(config=s.config),{})
            resumed,_=checkpoint.read_checkpoint(path,{},native_engine.Simulation)
        self.assertEqual(erosion.snapshot(resumed),erosion.snapshot(s))
        for state in (s,resumed):
            state.step(.02)
            dense_crust.validate(state.structure);dense_crust.validate(state.trace_structure)
            budget=state.material_column_budget
            self.assertLess(abs(budget['phase_mass_residual_kg'])/budget['after_columns_mass_kg'],1e-11)
            self.assertLess(abs(budget['residual_km3'])/budget['after_columns_volume_km3'],1e-11)
            frame=state.snapshot()
            mesh_history.arrays(frame)
            self.assertEqual(frame[erosion.FIELD],1)
            self.assertEqual(frame['surface_erosion_diagnostics']['quadrature_samples_per_face'],64)
            # Same post-step geometry gives the same field through the saved
            # native sampler and the live physical erosion preparation.
            ctx=sampling.prepare(frame)
            h,_=erosion.integrate(state.material_surface['vertices'],state.material_surface['faces'],
                lambda p:sampling.sample_frame(frame,p,prepared=ctx))
            np.testing.assert_allclose(erosion.prepare(state),h,atol=1e-7)
        for field in ('structure','trace_structure'):
            for key in getattr(s,field):
                np.testing.assert_array_equal(getattr(s,field)[key],getattr(resumed,field)[key])


if __name__=='__main__':unittest.main()
