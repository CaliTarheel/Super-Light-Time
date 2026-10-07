"""goSPL MPI reconstruction, unsafe-reference rejection and native sampling."""
from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gospl_results as results
import gospl_hdf_reader
import mesh_geometry


def archive(path,**arrays):
    with Path(path).open('wb') as handle:np.savez(handle,**arrays)


def fake_reader(requests,work_dir,cancel):
    rows=[]
    for i,request in enumerate(requests):
        with np.load(request['path'],allow_pickle=False) as source:array=source[request['dataset'].strip('/')]
        if list(array.shape)!=request['shape']:raise ValueError('Declared dimensions disagree.')
        name=f'dataset-{i:05d}.npy';np.save(work_dir/name,array)
        rows.append(dict(filename=name,shape=list(array.shape),dtype=str(array.dtype),units=None))
    return rows,'test numeric reader'


def fixture(folder):
    folder=Path(folder);(folder/'xmf').mkdir();(folder/'h5').mkdir()
    mesh=mesh_geometry.icosphere(1);vertices=(mesh['vertices']*6371000.).astype(np.float32);faces=mesh['faces']
    selected=[np.r_[np.arange(40),np.arange(40,44)],np.r_[np.arange(40,80),np.arange(36,40)]]
    heights=(220.+vertices[:,2]/6371000.*700.).astype(np.float32)
    for rank,which in enumerate(selected):
        nodes,indices=np.unique(faces[which],return_inverse=True)
        archive(folder/'h5'/f'topology.p{rank}.h5',coords=vertices[nodes],cells=indices.reshape(-1,3).astype(np.int32)+1)
        for epoch in (0,1):archive(folder/'h5'/f'gospl.{epoch}.p{rank}.h5',elev=(heights[nodes]+epoch*30.)[:,None])
    for epoch in (0,1):
        blocks=[]
        for rank,which in enumerate(selected):
            node_count=len(np.unique(faces[which]));count=len(which)
            blocks.append(f'''<Grid Name="Block.{rank}"><Topology Type="Triangle" NumberOfElements="{count}" BaseOffset="1">
<DataItem Format="HDF" Dimensions="{count} 3">h5/topology.p{rank}.h5:/cells</DataItem></Topology>
<Geometry Type="XYZ"><DataItem Format="HDF" Dimensions="{node_count} 3">h5/topology.p{rank}.h5:/coords</DataItem></Geometry>
<Attribute Name="Z" Center="Node"><DataItem Format="HDF" Dimensions="{node_count} 1">h5/gospl.{epoch}.p{rank}.h5:/elev</DataItem></Attribute></Grid>''')
        text='<?xml version="1.0"?><!DOCTYPE Xdmf SYSTEM "Xdmf.dtd"><Xdmf Version="2.0"><Domain><Grid GridType="Collection" CollectionType="Spatial"><Time Value="'+str(epoch*2_000_000)+'"/>'+''.join(blocks)+'</Grid></Domain></Xdmf>'
        (folder/'xmf'/f'gospl{epoch}.xmf').write_text(text)
    (folder/'gospl.xdmf').write_text('''<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude"><Domain><Grid GridType="Collection" CollectionType="Temporal"><xi:include href="xmf/gospl0.xmf" xpointer="xpointer(//Xdmf/Domain/Grid)"/><xi:include href="xmf/gospl1.xmf" xpointer="xpointer(//Xdmf/Domain/Grid)"/></Grid></Domain></Xdmf>''')
    return vertices,faces,heights


class GosplResultTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.vertices,self.faces,self.height=fixture(self.root)
        self.reader=patch.object(results,'_read_hdf',side_effect=fake_reader);self.reader.start()
    def tearDown(self):self.reader.stop();self.temp.cleanup()

    def load(self,**kwargs):return results.load_result(self.root,**kwargs)

    def test_multipart_epoch_welds_ghost_nodes_faces_and_samples_vertex_elevations(self):
        inspected=results.inspect_result(self.root)
        self.assertEqual([(e['index'],e['time_years'],e['parts']) for e in inspected['epochs']],[(0,0.,2),(1,2_000_000.,2)])
        source=self.load();meta=source['metadata']
        self.assertEqual((len(source['vertices']),len(source['faces'])),(42,80))
        self.assertEqual(meta['merged_duplicate_triangles'],8)
        self.assertGreater(meta['merged_duplicate_nodes'],0)
        self.assertTrue(meta['geometry']['closed'])
        value=results.sample_height(source,self.vertices.astype(float))
        np.testing.assert_allclose(value,self.height+30.,atol=2e-5)
        self.assertEqual(meta['time_years'],2_000_000.)
        self.assertTrue(all(len(row['sha256'])==64 for row in meta['source_files']))

    def test_single_xmf_or_hdf_selects_the_complete_declared_mpi_epoch(self):
        for path in (self.root/'xmf/gospl0.xmf',self.root/'h5/gospl.0.p1.h5'):
            source=results.load_result(path)
            self.assertEqual(source['metadata']['selected_epoch'],0)
            self.assertEqual(len(source['faces']),80)
        with self.assertRaisesRegex(ValueError,'topology-only'):results.inspect_result(self.root/'h5/topology.p0.h5')

    def test_explicit_epoch_and_invalid_selection(self):
        source=self.load(epoch_index=0)
        np.testing.assert_allclose(results.sample_height(source,self.vertices),self.height,atol=2e-5)
        for index in (8,True,1.5):
            with self.assertRaises(ValueError):self.load(epoch_index=index)

    def test_empty_or_malformed_descriptor_has_a_reviewable_value_error(self):
        target=self.root/'gospl.xdmf'
        for text in ('', '<Xdmf/>', '<Xdmf><Domain/></Xdmf>',
            '<Xdmf><Domain><Grid GridType="Collection" CollectionType="Temporal"/></Domain></Xdmf>',
            '<Xdmf><Domain><Grid><Time Value="0"/></Grid></Domain></Xdmf>'):
            target.write_text(text)
            with self.subTest(text=text),self.assertRaises(ValueError):results.inspect_result(self.root)

    def test_shared_elevation_disagreement_is_refused_instead_of_averaged(self):
        target=self.root/'h5/gospl.1.p1.h5'
        with np.load(target) as f:z=f['elev'].copy()
        z+=100.;archive(target,elev=z)
        with self.assertRaisesRegex(ValueError,'inconsistent elevations'):self.load()

    def test_missing_rank_and_wrong_dimensions_are_refused(self):
        target=self.root/'h5/gospl.1.p1.h5';target.unlink()
        with self.assertRaisesRegex(ValueError,'missing referenced'):self.load()
        archive(target,elev=np.zeros((1,1)))
        with self.assertRaisesRegex(ValueError,'dimensions'):self.load()

    def test_invalid_indices_nonfinite_values_and_non_spherical_coordinates(self):
        topology=self.root/'h5/topology.p0.h5'
        with np.load(topology) as f:xyz=f['coords'].copy();cells=f['cells'].copy()
        bad=cells.copy();bad[0,0]=99999;archive(topology,coords=xyz,cells=bad)
        with self.assertRaisesRegex(ValueError,'missing node'):self.load()
        changed=xyz.copy();changed[0]*=.9;archive(topology,coords=changed,cells=cells)
        with self.assertRaisesRegex(ValueError,'sphere'):self.load()
        changed=xyz.copy();changed[0,0]=np.nan;archive(topology,coords=changed,cells=cells)
        with self.assertRaisesRegex(ValueError,'Non-finite'):self.load()

    def test_nonclosed_result_is_refused_without_nearest_fill(self):
        topology=self.root/'h5/topology.p0.h5'
        with np.load(topology) as f:xyz=f['coords'].copy();cells=f['cells'].copy()
        cells[0]=cells[1];archive(topology,coords=xyz,cells=cells)
        with self.assertRaisesRegex(ValueError,'closed mesh'):self.load()

    def test_safe_descriptor_relative_hdf_paths_and_unsafe_escape(self):
        target=self.root/'xmf/gospl1.xmf';text=target.read_text()
        target.write_text(text.replace('>h5/','>../h5/'))
        self.assertEqual(len(self.load()['faces']),80)
        target.write_text(text.replace('h5/topology.p0.h5','../../outside.h5'))
        with self.assertRaisesRegex(ValueError,'escapes'):self.load()

    def test_external_xml_entities_urls_and_function_elevation_are_not_evaluated(self):
        target=self.root/'xmf/gospl1.xmf';text=target.read_text()
        invalid=[text.replace('SYSTEM "Xdmf.dtd"','SYSTEM "https://example.invalid/evil.dtd"'),
            '<!DOCTYPE Xdmf [<!ENTITY ex SYSTEM "file:///private">]>'+text,
            text.replace('h5/topology.p0.h5','https://example.invalid/data.h5'),
            text.replace('Format="HDF" Dimensions=','Format="XML" Dimensions=',1)]
        for value in invalid:
            target.write_text(value)
            with self.assertRaises(ValueError):self.load()

    def test_duplicate_times_or_ambiguous_reference_are_rejected(self):
        target=self.root/'xmf/gospl1.xmf';original=target.read_text()
        target.write_text(original.replace('2000000','0'))
        with self.assertRaisesRegex(ValueError,'unique'):self.load()
        target.write_text(original)
        (self.root/'xmf/h5').mkdir()
        (self.root/'xmf/h5/topology.p0.h5').write_bytes((self.root/'h5/topology.p0.h5').read_bytes())
        with self.assertRaisesRegex(ValueError,'Ambiguous'):self.load()

    def test_declared_nonmetre_units_memory_budget_and_cancel(self):
        target=self.root/'xmf/gospl1.xmf';text=target.read_text()
        target.write_text(text.replace('<Geometry Type="XYZ">','<Geometry Type="XYZ" Units="km">'))
        with self.assertRaisesRegex(ValueError,'metre'):self.load()
        target.write_text(text)
        with self.assertRaisesRegex(ValueError,'memory budget'):self.load(max_memory_bytes=1)
        with self.assertRaises(results.ResultCancelled):self.load(cancel=lambda:True)

    def test_provenance_keeps_applied_orientation_and_elapsed_time_mapping(self):
        (self.root/'export_metadata.json').write_text(json.dumps(dict(format='deep-time-gospl-history-v1',
            run_id='a-world',start_myr=100.,end_myr=120.,orientation=dict(yaw=42.,pitch=65.,roll=-17.),radius_m=6371000.)))
        source=self.load();meta=source['metadata']
        self.assertEqual(meta['tectonic_time_myr'],102.)
        self.assertEqual(meta['source_export']['orientation']['yaw'],42.)
        self.assertTrue(meta['orientation_in_coordinates'])
        np.testing.assert_allclose(results.sample_height(source,self.vertices),self.height+30.,atol=2e-5)

    def test_native_barycentric_interpolation_and_polar_rotation(self):
        source=self.load();faces=np.arange(0,len(source['faces']),7)
        bary=np.broadcast_to([.17,.29,.54],(len(faces),3))
        triangle=source['vertices'][source['faces'][faces]]
        points=np.einsum('ni,nij->nj',bary,triangle);points/=np.linalg.norm(points,axis=1)[:,None]
        expected=np.sum(source['elevation_m'][source['faces'][faces]]*bary,axis=1)
        np.testing.assert_allclose(results.sample_height(source,points),expected,atol=1e-10)
        rotation=np.array([[0,0,-1],[0,1,0],[1,0,0.]])
        moved=dict(source,vertices=source['vertices']@rotation)
        np.testing.assert_allclose(results.sample_height(moved,points@rotation),expected,atol=1e-10)


@unittest.skipUnless(importlib.util.find_spec('h5py'),'Requires optional h5py; also exercised in installed goSPL WSL Python.')
class HdfReaderTests(unittest.TestCase):
    def test_actual_numeric_dataset_and_external_link_rejection(self):
        import h5py
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary);source=path/'source.h5'
            with h5py.File(source,'w') as f:f.create_dataset('coords',data=np.ones((4,3)))
            request=[dict(path=str(source),dataset='/coords',shape=[4,3],role='coords')]
            rows=gospl_hdf_reader.extract(request,path/'good')
            np.testing.assert_array_equal(np.load(path/'good'/rows[0]['filename']),np.ones((4,3)))
            with h5py.File(path/'link.h5','w') as f:f['coords']=h5py.ExternalLink(str(source),'/coords')
            request[0]['path']=str(path/'link.h5')
            with self.assertRaisesRegex(ValueError,'hard links'):gospl_hdf_reader.extract(request,path/'bad')

    def test_actual_shape_and_numeric_dtype_checks(self):
        import h5py
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary);source=path/'source.h5'
            with h5py.File(source,'w') as f:f.create_dataset('elev',data=np.zeros((3,1)))
            with self.assertRaisesRegex(ValueError,'shape'):
                gospl_hdf_reader.extract([dict(path=str(source),dataset='/elev',shape=[4,1],role='elevation')],path/'bad')
            with h5py.File(source,'a') as f:f.create_dataset('text',data=np.array([b'no',b'height',b'field']))
            with self.assertRaisesRegex(ValueError,'numeric type'):
                gospl_hdf_reader.extract([dict(path=str(source),dataset='/text',shape=[3],role='elevation')],path/'text')


if __name__=='__main__':unittest.main()
