"""Ocean and material transport consume loading from one physical epoch."""
from types import MethodType
import unittest
from unittest.mock import patch
import numpy as np
import native_engine
import native_material_evolution as evolution
import native_processes
import raster_engine
try:
    from .test_native_material_evolution import fixture as marker_fixture
except ImportError:
    from test_native_material_evolution import fixture as marker_fixture


def fixture(version=1,enabled=True):
    s=marker_fixture();s.t=0.;s.material_mechanics_version=version
    # Native collision initialization assigns one sheet to these connected
    # material faces. The prepared-loading test uses a minimal state instead.
    s.parcel_collision_sheet=np.ones(len(s.material_surface['faces']),np.int64)
    s.config['deforming_regions']=enabled
    s.active=np.array([True,True,False]);s.plate_uid=np.array([10,11,12])
    s.omega=np.vstack((s.omega,np.zeros((1,3))))
    s.ba=np.array([0,2,4]);s.bb=np.array([1,3,5])
    s.bp=np.array([0,0,0]);s.bq=np.array([1,1,2])
    s.plate=np.array([0,1,2,1,0,2])
    s._valid_loading_edges=MethodType(raster_engine.Simulation._valid_loading_edges,s)
    unit=lambda a:a/np.linalg.norm(a,axis=1)[:,None]
    s.native_boundary_geometry=dict(segments_start=unit(np.array([[1.,0.,-.03],[1.,0.,-.01],[1.,0.,.01]])),
        segments_end=unit(np.array([[1.,0.,-.01],[1.,0.,.01],[1.,0.,.03]])),
        segment_normals=np.tile([0.,1.,0.],(3,1)),contact_index=np.arange(3))
    s.bmid=np.tile([1.,0.,0.],(3,1));s.bn=np.tile([0.,1.,0.],(3,1));s.bl=np.full(3,100.)
    s.collision_contacts=[dict(state='active',top_owner=0,under_owner=1,
        center=[1.,0.,0.],normal=[0.,1.,0.],overlap_area_km2=100.,length_km=20.)]
    return s


class StopBeforeSolve(Exception):pass


class MaterialLoadingEpochTests(unittest.TestCase):
    def test_source_validity_survives_transport_without_reactivating_stale_or_inactive_fronts(self):
        s=fixture();events=[];saved={};prepare=evolution.material_loading_boundaries
        def capture_prepare(state,version):
            events.append('prepare');saved['bundle']=prepare(state,version);return saved['bundle']
        def ocean(state,dt):
            events.append('ocean');state.plate=np.array([1,0,0,1,0,2])
        def material(state,dt,**kwargs):
            events.append('material');self.assertIs(kwargs['prepared_loading'],saved['bundle'])
            np.testing.assert_array_equal(state._valid_loading_edges(),[False,True,False])
            # Source-valid, already stale, inactive, persistent local collision.
            np.testing.assert_array_equal(saved['bundle'][0]['valid'],[True,False,False,True])
        with patch.object(evolution,'material_loading_boundaries',capture_prepare), \
             patch.object(evolution,'advect',material),patch.object(native_processes,'advect_ocean',ocean), \
             patch.object(native_engine.trench_history,'advect',lambda *a:events.append('trench')), \
             patch.object(native_engine.progressive_rifting,'advect',lambda *a:events.append('rift')), \
             patch.object(native_engine.native_initial_ownership,'active',return_value=False):
            native_engine.Simulation._advect(s,2.)
        self.assertEqual(events,['prepare','ocean','material','trench','rift'])
        for key in ('bmid','bn','bl','bp','bq'):
            self.assertFalse(np.shares_memory(saved['bundle'][0][key],getattr(s,key)))
        self.assertEqual(saved['bundle'][1]['valid_segments'],1)
        self.assertEqual(saved['bundle'][1]['total_loading_fronts'],4)

    def test_material_consumer_does_not_recheck_source_fronts_against_destination_cells(self):
        s=fixture();bundle=evolution.material_loading_boundaries(s,1)
        s.plate=np.array([1,0,0,1,0,2]);seen={}
        def deform(mesh,omega,boundaries,dt,**kwargs):
            seen['boundaries']=boundaries;raise StopBeforeSolve
        with patch.object(s,'_valid_loading_edges',side_effect=AssertionError('mixed-time validity')), \
             patch.object(evolution.deforming_regions,'deform',deform):
            with self.assertRaises(StopBeforeSolve):evolution.advect(s,2.,prepared_loading=bundle)
        self.assertIs(seen['boundaries'],bundle[0])
        np.testing.assert_array_equal(seen['boundaries']['valid'],[True,False,False,True])

    def test_unmarked_and_disabled_material_keep_the_legacy_call_order(self):
        for version,enabled in ((0,True),(1,False)):
            with self.subTest(version=version,enabled=enabled):
                s=fixture(version,enabled);events=[]
                def material(state,dt,**kwargs):
                    events.append('material');self.assertEqual(kwargs,{})
                with patch.object(evolution,'material_loading_boundaries',side_effect=AssertionError('early prep')), \
                     patch.object(evolution,'advect',material), \
                     patch.object(native_processes,'advect_ocean',lambda *a:events.append('ocean')), \
                     patch.object(native_engine.trench_history,'advect',lambda *a:events.append('trench')), \
                     patch.object(native_engine.progressive_rifting,'advect',lambda *a:events.append('rift')), \
                     patch.object(native_engine.native_initial_ownership,'active',return_value=False):
                    native_engine.Simulation._advect(s,2.)
                self.assertEqual(events,['ocean','material','trench','rift'])


if __name__=='__main__':unittest.main()
