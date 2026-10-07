"""Evolved columns must retain signed relief across an inherited shelf."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
import collision_coast
import collision_surface
import continental_margin
import native_frame_sampling as sampling
from tests import test_collision_layer_emergence as emergence
from tests import test_collision_surface_acceptance as acceptance


def revised(s, **kwargs):
    frame = acceptance.saved(s, **kwargs)
    frame['collision_coast_version'] = 2
    frame[collision_coast.REFERENCE_FIELD] = s.structure['reference_elevation_m'].copy()
    return frame


class CoastalDeformationTests(unittest.TestCase):
    def shelf_fixture(self):
        s = acceptance.world([(acceptance.unit([[1.,-.24,-.15],[1.,.24,-.15],[1.,0.,.24]]), 35., 1000., False)])
        collision_surface.refresh(s)
        frame = revised(s, margins=True)
        ctx = sampling.prepare(frame)
        # Spherical construction: exactly75km inward from one edge midpoint.
        a,b = acceptance.unit([[1.,-.24,-.15],[1.,.24,-.15],[1.,0.,.24]])[:2]
        mid = acceptance.unit(a+b); normal = acceptance.unit(np.cross(a,b))
        point = (np.cos(75./6371.)*mid + np.sin(75./6371.)*normal)[None]
        result = sampling.sample_frame(frame,point,prepared=ctx)
        distance = continental_margin.distance_km(ctx['collision_coast'],point,result['material_face'])
        np.testing.assert_allclose(distance,75.,atol=1e-8)
        return s,frame,point

    def test_shelf_retains_uplift_and_subsidence_with_analytic_response(self):
        _,frame,point = self.shelf_fixture()
        # At half of a150km margin, inherited shelf=-129.375m and the
        # shelf retains the full displacement. No cap may absorb either sign.
        for delta in (-4000.,-200.,0.,200.,4000.):
            with self.subTest(delta=delta):
                frame['material_height_m'][:] = 1000.+delta
                result = sampling.sample_frame(frame,point)
                self.assertAlmostEqual(result['elevation'][0],-129.375+delta,places=6)
                np.testing.assert_array_equal(result['clipping_delta_m'],0.)

    def test_undeformed_coast_and_arc_remain_unchanged(self):
        _,frame,point = self.shelf_fixture()
        for arc in (False,True):
            if arc:frame['material_arc_id'][:]=1
            old=deepcopy(frame);old['collision_coast_version']=1
            del old[collision_coast.REFERENCE_FIELD]
            np.testing.assert_allclose(sampling.sample_frame(frame,point)['elevation'],
                sampling.sample_frame(old,point)['elevation'],atol=1e-8)

    def test_explicit_stack_support_is_added_once_without_shelf_cap(self):
        _,frame,point=self.shelf_fixture()
        frame['material_collision_support_m'][:]=1500.
        frame['material_collision_load_thickness_km'][:]=1500./acceptance.AIRY
        frame['material_height_m']+=2500.
        result=sampling.sample_frame(frame,point)
        self.assertAlmostEqual(result['physical_stack_support_m'][0],1500.,places=6)
        self.assertAlmostEqual(result['elevation'][0],-129.375+2500.+1500.,places=6)

    def test_arc_only_uplift_keeps_existing_arc_law(self):
        _,frame,point = self.shelf_fixture()
        frame['material_arc_id'][:]=1;frame['material_height_m']+=2500.
        old=deepcopy(frame);old['collision_coast_version']=1;del old[collision_coast.REFERENCE_FIELD]
        np.testing.assert_allclose(sampling.sample_frame(frame,point)['elevation'],
            sampling.sample_frame(old,point)['elevation'],atol=1e-8)

    def test_emergence_continuity_mixed_arcs_triple_layers_and_rotation(self):
        def displaced(s, **kwargs):
            frame=revised(s,**kwargs)
            frame[collision_coast.REFERENCE_FIELD] -= 700.*(1.+s.parcel_plate)
            return frame
        with patch.object(emergence,'saved',displaced):
            check=emergence.CollisionLayerEmergenceTests()
            for options in ({},dict(triple=True),dict(lower_arc=True),dict(upper_arc=True)):
                check.check_emergence(**options)
            check.test_layer_emergence_is_rotation_equivariant()

    def test_constant_column_refinement_preserves_coastal_displacement(self):
        tri=acceptance.unit([[1.,-.24,-.15],[1.,.24,-.15],[1.,0.,.24]])
        points=acceptance.unit(np.array([tri.mean(axis=0), .05*tri[2]+.475*(tri[0]+tri[1])]))
        values=[]
        for refined in (False,True):
            s=acceptance.world([(tri,35.,1000.,refined)])
            collision_surface.refresh(s);frame=revised(s,margins=True)
            frame['material_height_m']+=3000.
            values.append(sampling.sample_frame(frame,points)['elevation'])
        np.testing.assert_allclose(values[0],values[1],atol=1e-7,rtol=0.)

    def test_ocean_endpoint_and_no_material_invented(self):
        _,frame,_=self.shelf_fixture();frame['material_height_m']+=4000.
        a,b=acceptance.unit([[1.,-.24,-.15],[1.,.24,-.15],[1.,0.,.24]])[:2];mid=acceptance.unit(a+b);normal=acceptance.unit(np.cross(a,b))
        jumps=[]
        for km in (.01,.001,.0001):
            angle=np.array([-km,km])/6371.
            points=np.cos(angle)[:,None]*mid+np.sin(angle)[:,None]*normal
            result=sampling.sample_frame(frame,points)
            self.assertEqual(result['material_face'][0],-1)
            self.assertGreaterEqual(result['material_face'][1],0)
            jumps.append(abs(np.diff(result['elevation'])[0]))
        self.assertLess(jumps[-1],.01)
        self.assertLess(jumps[-1],jumps[0]/10.)

    def test_reference_is_explicit_finite_aligned_and_versioned(self):
        _,frame,point=self.shelf_fixture()
        for bad in (None,[np.nan],[1000.,1000.]):
            broken=deepcopy(frame)
            if bad is None:del broken[collision_coast.REFERENCE_FIELD]
            else:broken[collision_coast.REFERENCE_FIELD]=bad
            with self.assertRaisesRegex(ValueError,'coastal reference'):
                sampling.sample_frame(broken,point)
        frame['collision_coast_version']=1
        with self.assertRaisesRegex(ValueError,'version two'):sampling.prepare(frame)

    def test_snapshot_records_reference_without_mutating_physical_state(self):
        s,_,_=self.shelf_fixture();before=acceptance.physical_inventory(s)
        self.assertNotIn(collision_coast.REFERENCE_FIELD,collision_surface.snapshot_fields(s))
        s.collision_coast_version=2
        fields=collision_surface.snapshot_fields(s)
        np.testing.assert_array_equal(fields[collision_coast.REFERENCE_FIELD],s.structure['reference_elevation_m'])
        fields[collision_coast.REFERENCE_FIELD][:]=123.
        after=acceptance.physical_inventory(s)
        for key,value in before.items():
            if isinstance(value,dict):
                for name,array in value.items():np.testing.assert_array_equal(array,after[key][name])
            else:np.testing.assert_array_equal(value,after[key])

    def test_export_refuses_one_to_two_as_uplift(self):
        import gospl_export
        _,frame,point=self.shelf_fixture();old=deepcopy(frame)
        old['collision_coast_version']=1;del old[collision_coast.REFERENCE_FIELD]
        frame['time_myr']=old['time_myr']+2.
        with self.assertRaisesRegex(ValueError,'mix collision coast versions'):
            gospl_export.interval_forcing(point,old,frame,{})

    def test_live_snapshot_persistence_and_resampling_agree(self):
        import tempfile
        from pathlib import Path
        import server
        from tectonics import Simulation
        s=Simulation(dict(width=48,height=24,mesh_level=2,coast_geometry_level=2,
                          mechanics_nodes=128,plate_count=4,seed=12,duration_myr=2))
        s.collision_coast_version=2
        frame=s.snapshot()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[1]) as directory:
            manager=server.SimulationManager(Path(directory)/'runs')
            manager.current=dict(run_id='coast-v2',frames=[dict(index=0,time_myr=s.t)],
                                 frame_count=1,config=s.config)
            manager.path().mkdir(parents=True)
            manager.save_frame(0,frame)
            import json
            saved=json.loads((manager.path()/'frame_0000.json').read_text())
            with np.load(manager.path()/'frame_0000.npz',allow_pickle=False) as archive:
                self.assertIn(collision_coast.REFERENCE_FIELD,archive.files)
                saved.update({key:archive[key] for key in archive.files})
            np.testing.assert_array_equal(saved[collision_coast.REFERENCE_FIELD],
                                          s.structure['reference_elevation_m'])
            w,h=frame['width'],frame['height']
            lon,lat=np.meshgrid((np.arange(w)+.5)*2*np.pi/w-np.pi,
                               np.pi/2-(np.arange(h)+.5)*np.pi/h)
            points=np.column_stack((np.cos(lat).ravel()*np.cos(lon).ravel(),
                                    np.cos(lat).ravel()*np.sin(lon).ravel(),np.sin(lat).ravel()))
            result=sampling.sample_frame(saved,points)
            np.testing.assert_allclose(result['elevation'],np.asarray(frame['elevation']).ravel(),
                                       atol=1e-3,rtol=0.)


if __name__=='__main__':unittest.main()


