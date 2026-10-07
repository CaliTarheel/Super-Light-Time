"""Actual native epochs survive pause, persistence, orientation and goSPL export."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from checkpoint import read_checkpoint, write_checkpoint
import gospl_export
import mesh_history
import native_frame_sampling
from orientation import orient_frame, rotation_matrix
import raster_engine
import server
from tectonics import Simulation
from validate_gospl import validate_package


CONFIG = dict(width=48,height=24,mesh_level=2,mechanics_nodes=128,
              duration_myr=6,dt_myr=2,snapshot_myr=2,plate_count=4,seed=12)


class NativeHistoryExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(dir=Path(__file__).parents[1]/'tmp')
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)

    def assert_tree_equal(self, first, second, path='state'):
        if isinstance(first,np.ndarray):
            self.assertEqual(first.dtype,second.dtype,path)
            np.testing.assert_array_equal(first,second,err_msg=path)
        elif isinstance(first,np.random.Generator):
            self.assert_tree_equal(first.bit_generator.state,second.bit_generator.state,path+'.rng')
        elif isinstance(first,dict):
            self.assertEqual(set(first),set(second),path)
            for key in first:self.assert_tree_equal(first[key],second[key],path+'.'+str(key))
        elif isinstance(first,(list,tuple)):
            self.assertEqual(type(first),type(second),path)
            self.assertEqual(len(first),len(second),path)
            for index,(a,b) in enumerate(zip(first,second)):self.assert_tree_equal(a,b,path+f'[{index}]')
        else:self.assertEqual(first,second,path)

    def saved_history(self):
        s=Simulation(CONFIG)
        manager=server.SimulationManager(self.root/'runs')
        manifest=dict(run_id='native-contract',state='complete',frame_count=3,
                      config=deepcopy(s.config),frames=[],**manager.compatibility())
        manager.current=manifest
        manager.path().mkdir(parents=True)
        (manager.path()/'engine.py').write_bytes(server.ENGINE_SOURCE)
        for name,payload in server.AUXILIARY_SOURCES.items():
            (manager.path()/name).write_bytes(payload)
        frames=[]
        for index in range(3):
            frame=s.snapshot()
            manager.save_frame(index,frame)
            manifest['frames'].append(dict(index=index,time_myr=s.t))
            frames.append(frame)
            if index<2:s.step()
        (manager.path()/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        return manager,frames

    def test_checkpoint_restores_native_geometry_memory_rng_and_exact_continuation(self):
        original=Simulation(CONFIG);original.step()
        compatibility=server.SimulationManager.compatibility()
        manifest=dict(config=deepcopy(original.config),time_myr=original.t,next_output_myr=4.)
        path=self.root/'native-checkpoint.npz'
        write_checkpoint(path,original,manifest,compatibility)
        with patch.object(Simulation,'__init__',side_effect=AssertionError('Resume must not reseed geometry')):
            restored,returned=read_checkpoint(path,compatibility,Simulation)
        self.assertEqual(returned,manifest)
        self.assert_tree_equal(vars(original),vars(restored))
        original.step();restored.step()
        self.assert_tree_equal(vars(original),vars(restored))
        self.assert_tree_equal(original.snapshot(),restored.snapshot())

    def test_step_ignores_legacy_compass_arrays_and_raster_material_deposition(self):
        simulation=Simulation(CONFIG)
        for name in ('east','west','north','south','lon','lat'):
            simulation.__dict__.pop(name,None)
        with patch.object(raster_engine,'deposit_crust',side_effect=AssertionError('Raster material transport must be unreachable')):
            for _ in range(3):simulation.step()
            frame=simulation.snapshot()
        self.assertEqual(frame['time_myr'],6.)
        self.assertEqual(frame['mesh_diagnostics']['native_cells'],320)
        self.assertTrue(np.isfinite(frame['elevation']).all())
        self.assertTrue(frame['mesh_diagnostics']['display_grid_is_output_only'])

    def test_review_resolution_does_not_change_native_motion_or_material_history(self):
        low=Simulation(CONFIG)
        high=Simulation(dict(CONFIG,width=96,height=48))
        for _ in range(3):low.step();high.step()
        for field in ('omega','plate','age','support','pos','mass','kind','parcel_plate',
                      'parcel_patch','trace_xyz','trace_relief_m','trace_erosion_m'):
            np.testing.assert_array_equal(getattr(low,field),getattr(high,field),err_msg=field)
        self.assert_tree_equal(low.material_surface,high.material_surface)
        self.assert_tree_equal(low.ocean_history,high.ocean_history)
        self.assertEqual(low.snapshot()['elevation'].shape,(48*24,))
        self.assertEqual(high.snapshot()['elevation'].shape,(96*48,))

    def test_actual_saved_epochs_preserve_native_arrays_and_consistent_review_domains(self):
        manager,frames=self.saved_history()
        for index,original in enumerate(frames):
            reviewed=manager.frame(index)
            self.assertEqual(reviewed['mesh_version'],1)
            self.assertEqual(reviewed['surface_reconstruction_version'],1)
            by_domain={row['uid']:row['plate_id'] for row in reviewed['domains']}
            np.testing.assert_array_equal([by_domain[int(x)] for x in reviewed['domain']],reviewed['plate'])
            saved=gospl_export.load_frame(manager.path(),index)
            expected=mesh_history.arrays(original)
            self.assertEqual(set(mesh_history.DTYPES).intersection(saved),set(expected))
            for field,value in expected.items():
                self.assertEqual(saved[field].dtype,np.dtype(mesh_history.DTYPES[field]),field)
                np.testing.assert_array_equal(saved[field],value,err_msg=field)
            metadata=json.loads((manager.path()/f'frame_{index:04d}.json').read_text())
            self.assertTrue(set(mesh_history.DTYPES).isdisjoint(metadata))
            self.assertEqual(saved['mesh_faces'].shape,(320,3))
            self.assertEqual(len(np.unique(saved['material_face_id'])),len(saved['material_faces']))

    def test_snapshot_and_native_export_share_the_same_continuous_reconstruction(self):
        simulation=Simulation(CONFIG);simulation.step()
        frame=simulation.snapshot()
        self.assertEqual(frame['surface_reconstruction_version'],1)
        lon,lat=np.meshgrid((np.arange(frame['width'])+.5)*2*np.pi/frame['width']-np.pi,
                           np.pi/2-(np.arange(frame['height'])+.5)*np.pi/frame['height'])
        points=raster_engine._xyz(lon.ravel(),lat.ravel())
        sampled=native_frame_sampling.sample_frame(frame,points)
        land=sampled['material_face']>=0
        self.assertTrue(land.any())
        np.testing.assert_array_equal(sampled['plate'],frame['plate'])
        np.testing.assert_array_equal(sampled['crust'][land],frame['crust'][land])
        np.testing.assert_allclose(sampled['elevation'],frame['elevation'],atol=.001,rtol=0.)
        raw=frame['material_height_m'][sampled['material_face'][land]]
        self.assertGreater(np.max(np.abs(sampled['elevation'][land]-raw)),1.)

    def test_diffuse_owner_without_control_cell_keeps_its_real_saved_identity(self):
        simulation=Simulation(CONFIG)
        owner=int(np.flatnonzero(~simulation.active)[0])
        parent=int(np.flatnonzero(simulation.active)[0])
        simulation.active[owner]=True
        simulation.count=max(simulation.count,owner+1)
        simulation._new_plate_identity(owner,parent)
        simulation.support*=.9
        simulation.support[owner]=.1
        frame=simulation.snapshot()
        self.assertNotIn(owner,np.unique(frame['mesh_plate']))
        self.assertIn(owner,frame['mesh_owner_slots'])
        recorded={row['id']:row for row in frame['plates']}
        self.assertIn(owner,recorded)
        self.assertEqual(recorded[owner]['uid'],int(simulation.plate_uid[owner]))
        self.assertEqual(recorded[owner]['name'],simulation.names[owner])
        self.assertEqual(recorded[owner]['area_km2'],0.)
        mesh_history.arrays(frame)
        native_frame_sampling.prepare(frame)

    def unresolved_ocean_frame(self):
        manager,frames=self.saved_history()
        source=deepcopy(frames[1])
        cell=int(np.flatnonzero((source['crust']==0)&(source['domain']>0))[0])
        source['domain'][cell]=0
        source['mesh_diagnostics']['unresolved_ocean_domain_queries']=int(np.count_nonzero(source['domain']==0))
        return manager,source,cell

    def test_explicit_unresolved_ocean_domain_survives_review_history_and_exports(self):
        manager,source,cell=self.unresolved_ocean_frame()
        manager.save_frame(1,source)
        for reviewed in (manager.frame(1),manager.native_frame(1)):
            self.assertEqual(reviewed['domain'][cell],0)
            self.assertEqual(reviewed['crust'][cell],0)
            self.assertEqual(reviewed['plate'][cell],source['plate'][cell])
            self.assertNotIn(0,[row['uid'] for row in reviewed['domains']])
        inspected=manager.history(1,cell)
        self.assertEqual(inspected['mode'],'location')
        point=next(row for row in inspected['points'] if row['index']==1)
        self.assertIsNone(point['domain_uid'])
        self.assertIsNone(point['domain_name'])
        self.assertEqual(point['plate_id'],int(source['plate'][cell]))
        server.write_json(manager.path()/'config.json',manager.current['config'])
        initial=server.native(server.make_initial(manager.current['config']))
        server.write_json(manager.path()/'initial.json',initial)
        with manager.export_file(1) as stream,zipfile.ZipFile(stream) as archive:
            with np.load(io.BytesIO(archive.read('history/frame_0001.npz')),allow_pickle=False) as saved:
                self.assertEqual(saved['domain'][cell],0)
                self.assertEqual(saved['plate'][cell],source['plate'][cell])
            metadata=json.loads(archive.read('history/frame_0001.json'))
            self.assertNotIn(0,[row['uid'] for row in metadata['domains']])
            self.assertIsNone(archive.testzip())
        before=gospl_export.source_fingerprint(manager.path(),1)
        target=self.root/'unresolved-gospl'
        gospl_export.build_history(manager.path(),target,manager.current,0,2,
                                   subdivisions=1,dt_years=100_000)
        self.assertTrue(validate_package(target)['passed'])
        self.assertEqual(gospl_export.source_fingerprint(manager.path(),1),before)

    def test_unresolved_domain_requires_exact_count_native_flag_and_ocean_crust(self):
        manager,source,cell=self.unresolved_ocean_frame()
        manager.save_frame(1,source)
        previous=(manager.path()/'frame_0001.npz').read_bytes()
        mutations=[]
        for count in (0,2,1.5,-1):
            bad=deepcopy(source)
            bad['mesh_diagnostics']['unresolved_ocean_domain_queries']=count
            mutations.append(('count '+str(count),bad))
        bad=deepcopy(source);bad['mesh_diagnostics'].pop('unresolved_ocean_domain_queries')
        mutations.append(('missing count',bad))
        bad=deepcopy(source);bad['crust'][cell]=1
        # Keep the independent material-only review fields consistent so this
        # case specifically exercises the unresolved-domain land rejection.
        for key in ('deformation_weight','refinement_level'):
            if key in bad:bad[key][cell]=0
        mutations.append(('land zero',bad))
        bad=deepcopy(source);bad.pop('owner_reconstruction_version')
        bad.pop('mesh_owner_slots');bad.pop('mesh_vertex_support')
        mutations.append(('unflagged zero',bad))
        fabricated=deepcopy(bad)
        fabricated['domains'].append(dict(uid=0,name='Invented parent',plate_id=int(source['plate'][cell])))
        mutations.append(('unflagged fabricated zero identity',fabricated))
        bad=deepcopy(source);bad.pop('mesh_version')
        mutations.append(('non-native zero',bad))
        for label,bad in mutations:
            with self.subTest(case=label):
                with self.assertRaisesRegex(ValueError,'domain|Domain'):
                    manager.save_frame(1,bad)
                self.assertEqual((manager.path()/'frame_0001.npz').read_bytes(),previous)

    def test_unresolved_ocean_does_not_relax_positive_domain_identity_or_owner_checks(self):
        manager,source,_=self.unresolved_ocean_frame()
        cell=int(np.flatnonzero(source['domain']>0)[0])
        uid=int(source['domain'][cell])
        bad=deepcopy(source);bad['domain'][cell]=max(row['uid'] for row in bad['domains'])+100
        with self.assertRaisesRegex(ValueError,'matching domain metadata'):
            manager.save_frame(1,bad)
        bad=deepcopy(source)
        for row in bad['domains']:
            if row['uid']==uid:row['plate_id']=int(source['plate'][cell])+100
        with self.assertRaisesRegex(ValueError,'underlying plate'):
            manager.save_frame(1,bad)
        bad=deepcopy(source);bad['domains']=[row for row in bad['domains'] if row['uid']!=uid]
        with self.assertRaisesRegex(ValueError,'matching domain metadata'):
            manager.save_frame(1,bad)

    def test_native_reader_and_oriented_epoch_keep_shared_topology_and_material_ids(self):
        manager,frames=self.saved_history()
        # The normal browser review remains lean; a separate reader carries
        # the authoritative arrays needed for native inspection/download.
        native=manager.native_frame(2)
        native_arrays=mesh_history.arrays(frames[2])
        self.assertEqual(set(mesh_history.DTYPES).intersection(native),set(native_arrays))
        for field,value in native_arrays.items():
            np.testing.assert_array_equal(native[field],value,err_msg=field)
        angles=dict(yaw=42.,pitch=65.,roll=-17.);matrix=rotation_matrix(angles)
        frame=manager.native_frame(2,angles)
        expected=orient_frame(gospl_export.load_frame(manager.path(),2),angles)
        self.assertEqual(set(mesh_history.DTYPES).intersection(frame),set(native_arrays))
        for key in native_arrays:
            np.testing.assert_array_equal(frame[key],expected[key],err_msg=key)
        for key in ('mesh_vertices','material_vertices'):
            np.testing.assert_allclose(frame[key],frames[2][key]@matrix,atol=2e-15)
        for key in ('mesh_faces','material_faces','material_face_id','material_owner',
                    'material_kind','material_reference_area_km2'):
            np.testing.assert_array_equal(frame[key],frames[2][key])

    def test_oriented_gospl_package_from_actual_native_history_is_finite_closed_and_reproducible(self):
        manager,frames=self.saved_history()
        before=[gospl_export.source_fingerprint(manager.path(),i) for i in range(3)]
        angles=dict(yaw=42.,pitch=65.,roll=-17.)
        destination=self.root/'gospl'
        meta=gospl_export.build_history(manager.path(),destination,manager.current,
                                       0,2,subdivisions=3,dt_years=100_000,orientation=angles)
        result=validate_package(destination)
        self.assertEqual((result['nodes'],result['triangles'],result['intervals']),(642,1280,2))
        self.assertTrue(result['mesh_closed']);self.assertFalse(result['solver_run'])
        self.assertEqual((result['time_start_year'],result['time_end_year']),(0,4_000_000))
        self.assertEqual(meta['units']['upsub'],'m/year')
        self.assertEqual(meta['orientation'],angles)
        self.assertEqual(meta['source_frames'],before)
        sources=destination/'source/model'
        for name in ('native_engine.py','mesh_geometry.py','mesh_transport.py','material_surface.py',
                     'native_processes.py','native_topology.py','native_rift_material.py'):
            self.assertIn(name,meta['source_auxiliary_sources_sha256'])
        for name,digest in meta['source_auxiliary_sources_sha256'].items():
            self.assertEqual(hashlib.sha256((sources/name).read_bytes()).hexdigest(),digest,name)
        for index in range(2):
            with np.load(destination/f'input/forcing_{index:04d}.npz',allow_pickle=False) as data:
                for name in data.files:self.assertTrue(np.isfinite(data[name]).all(),name)
                self.assertEqual((int(data['start_year']),int(data['end_year'])),(index*2_000_000,(index+1)*2_000_000))
        with zipfile.ZipFile(destination/'gospl-history.zip') as archive:
            self.assertIsNone(archive.testzip())
        self.assertEqual(before,[gospl_export.source_fingerprint(manager.path(),i) for i in range(3)])


if __name__=='__main__':unittest.main()
