"""Optional named domains survive history saves without changing motion owners."""
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from server import SimulationManager, boundary_geojson, json_bytes, write_json


class DomainPersistenceTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1]/'tmp'
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.manager = SimulationManager(self.temporary.name)
        self.write_run()

    def frame(self, index, domains=True):
        w, h = 8, 4
        cell = 9+index
        # A persistent marker passes underneath another surface plate at t=2.
        # Its domain must then be unknown, not borrowed from the overriding map.
        surface_owner = 2 if index == 1 else 1
        domain_uid = (40, 70, 41)[index]
        longitude = np.radians(-180+(cell % w+.5)*360/w)
        latitude = np.radians(90-(cell//w+.5)*180/h)
        result = dict(width=w, height=h, time_myr=index*2,
                      elevation=np.full(w*h, 300., np.float32),
                      plate=np.zeros(w*h, np.int32), crust=np.zeros(w*h, np.uint8),
                      age=np.zeros(w*h, np.float32), boundary=np.zeros(w*h, np.uint8),
                      stats={}, events=[], history_version=1,
                      plates=[dict(id=0, uid=3, name='Ocean motion'),
                              dict(id=1, uid=11, name='Parent motion'),
                              dict(id=2, uid=12, name='Overriding motion')],
                      trace_id=np.array([5], np.int64), trace_plate=np.array([1], np.int32),
                      trace_plate_uid=np.array([11], np.int64),
                      trace_xyz=np.array([[np.cos(latitude)*np.cos(longitude),
                                           np.cos(latitude)*np.sin(longitude), np.sin(latitude)]]))
        result['plate'][cell] = surface_owner
        result['crust'][cell] = 1
        result['boundary'][cell] = 4
        if domains:
            result['domain'] = np.full(w*h, 10, np.int32)
            result['domain'][cell] = domain_uid
            result['domains'] = [dict(uid=10, name='Ocean domain', plate_id=0,
                                      parent_plate_uid=3, area_km2=31., created_myr=0,
                                      source_domain_uid=None),
                                 dict(uid=domain_uid, name=('Western fragment', 'Overrider', 'Eastern fragment')[index],
                                      plate_id=surface_owner, parent_plate_uid=12 if index==1 else 11,
                                      area_km2=1., created_myr=0 if index==0 else 2,
                                      source_domain_uid=40 if index==2 else None)]
        return result

    def write_run(self, domains=True):
        run_id='domains' if domains else 'legacy'
        path=self.manager.path(run_id)
        path.mkdir(exist_ok=True)
        config=dict(width=8,height=4,duration_myr=4,dt_myr=2,snapshot_myr=2)
        self.manager.current=dict(state='complete',run_id=run_id,config=config,frame_count=3,
                                  time_myr=4,duration_myr=4,
                                  frames=[dict(index=i,time_myr=i*2,stats={}) for i in range(3)])
        for i in range(3):self.manager.save_frame(i,self.frame(i,domains))
        write_json(path/'initial.json',dict(width=8,height=4,crust=self.frame(0)['crust']))
        write_json(path/'config.json',config)
        self.manager.persist()

    def test_named_domain_raster_roundtrip_api_and_export_preserve_parent_motion(self):
        loaded=SimulationManager(self.temporary.name)
        frame=loaded.frame(2)
        np.testing.assert_array_equal(frame['domain'],self.frame(2)['domain'])
        self.assertEqual(frame['domain'].dtype,np.int32)
        self.assertEqual(frame['domains'],self.frame(2)['domains'])
        payload=json.loads(json_bytes(frame))
        self.assertEqual(payload['domain'][11],41)
        self.assertEqual(payload['plate'][11],1)
        with loaded.export_file(2) as stream,zipfile.ZipFile(stream) as archive:
            self.assertIsNone(archive.testzip())
            with np.load(io.BytesIO(archive.read('history/frame_0002.npz'))) as arrays:
                np.testing.assert_array_equal(arrays['domain'],frame['domain'])
            metadata=json.loads(archive.read('history/frame_0002.json'))
            self.assertNotIn('domain',metadata)
            self.assertEqual(metadata['domains'],frame['domains'])
            self.assertIn('domain',json.loads(archive.read('history_schema.json'))['optional_grid_fields'])

    def test_marker_keeps_motion_parent_and_never_borrows_overlying_domain(self):
        result=self.manager.history(2,11)
        self.assertEqual(result['mode'],'material')
        self.assertEqual([p['domain_uid'] for p in result['points']],[40,None,41])
        self.assertEqual([p['domain_name'] for p in result['points']],['Western fragment',None,'Eastern fragment'])
        self.assertEqual([p['plate_name'] for p in result['points']],['Parent motion']*3)
        self.assertEqual([p['plate_uid'] for p in result['points']],[11]*3)
        site=self.manager.history(2,0)
        self.assertEqual(site['mode'],'location')
        self.assertEqual([p['domain_name'] for p in site['points']],['Ocean domain']*3)

    def test_legacy_frames_stay_readable_without_invented_domains(self):
        self.write_run(domains=False)
        loaded=SimulationManager(self.temporary.name)
        frame=loaded.frame(2)
        self.assertNotIn('domain',frame)
        self.assertNotIn('domains',frame)
        self.assertTrue(all(p['domain_uid'] is None and p['domain_name'] is None
                            for p in loaded.history(2,11)['points']))
        self.assertNotIn('domain_uid',boundary_geojson(frame)['features'][0]['properties'])

    def test_boundary_export_identifies_domain_and_adjacent_domain_without_renaming_plate(self):
        feature=boundary_geojson(self.frame(2))['features'][0]['properties']
        self.assertEqual(feature['plate_id'],1)
        self.assertEqual(feature['domain_uid'],41)
        self.assertEqual(feature['domain_name'],'Eastern fragment')
        self.assertEqual(feature['adjacent_domain_uids'],[10])

    def test_invalid_domain_raster_identity_or_parent_cannot_be_saved(self):
        broken=self.frame(2)
        broken['domain']=broken['domain'][:-1]
        with self.assertRaisesRegex(ValueError,'invalid domain'):self.manager.save_frame(2,broken)
        broken=self.frame(2)
        broken['domain']=broken['domain'].astype(float)
        broken['domain'][0]=np.nan
        with self.assertRaisesRegex(ValueError,'invalid domain'):self.manager.save_frame(2,broken)
        broken=self.frame(2)
        broken['domains'].pop()
        with self.assertRaisesRegex(ValueError,'matching domain metadata'):self.manager.save_frame(2,broken)
        broken=self.frame(2)
        broken['domains'][1]['plate_id']=2
        with self.assertRaisesRegex(ValueError,'underlying plate'):self.manager.save_frame(2,broken)
        broken=self.frame(2)
        broken['domains'].append(broken['domains'][0])
        with self.assertRaisesRegex(ValueError,'unique'):self.manager.save_frame(2,broken)


if __name__=='__main__':unittest.main()
