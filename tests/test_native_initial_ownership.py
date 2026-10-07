"""Independent geometry, conservation and first-transport initialization checks."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

import checkpoint
import mesh_geometry
import native_boundary_geometry
import native_initial_ownership as initial_ownership
import native_spreading
import native_subduction
import raster_engine
from native_engine import Simulation, make_initial


def fixture(control_level=2, *, hidden=False, source_level=1, mixed=False):
    source=mesh_geometry.icosphere(source_level)
    owner=(source['xyz'][:,0]>0).astype(np.int16)
    fitted=dict(vertices=source['vertices'],faces=source['faces'],face_owner=owner,
                face_kind=np.zeros(len(owner),np.uint8),area_km2=source['area_km2'])
    if mixed:
        fitted['face_kind'] = ((np.arange(len(owner)) % 3 == 0) & (owner == 1)).astype(np.uint8)
    mesh=mesh_geometry.icosphere(control_level)
    locator=mesh_geometry.build_locator(mesh['vertices'],mesh['faces'])
    label=(mesh['xyz'][:,0]>0).astype(np.int16)
    if hidden: label[:]=0
    s=SimpleNamespace(native_mesh=mesh,native_locator=locator,plate=label,
        cell_area=mesh['area_km2'],support=np.zeros((2,len(label))),initial_geometry_diagnostics={})
    initial_ownership.initialize(s,fitted)
    edge=mesh['edge_faces'];indices=np.flatnonzero(label[edge[:,0]]!=label[edge[:,1]])
    return s,fitted,indices


class InitialOwnershipTests(unittest.TestCase):
    def test_mixed_fronts_keep_exact_length_and_separate_material_polarity(self):
        s,_,indices=fixture(1, source_level=3, mixed=True)
        owners=s.plate.copy();support=s.support.copy()
        result=initial_ownership.reconstruct(s,indices)
        self.assertGreater(result['diagnostics']['material_front_split_contacts'],0)
        self.assertAlmostEqual(result['lengths'].sum(),
            s.native_initial_owner_interfaces['length_km'].sum(),places=7)
        graph=result['graph_edge_indices']
        faces=s.native_mesh['edge_faces'][graph]
        source=s.native_initial_owner_interfaces
        self.assertEqual(result['diagnostics']['unmapped_segments'],0)
        for contact in np.unique(result['contact_index']):
            pieces=result['contact_index']==contact
            # Each force quadrature contact has one material pair even when
            # ocean/ocean and coastal source pieces share a control graph key.
            owner_p=s.plate[faces[contact,0]]
            forward=source['owner_a'][pieces]==owner_p
            ka=source['kind_a'][pieces];kb=source['kind_b'][pieces]
            categories=np.where(forward,ka,kb)+2*np.where(forward,kb,ka)
            self.assertEqual(len(np.unique(categories)),1)
        np.testing.assert_array_equal(s.plate,owners)
        np.testing.assert_array_equal(s.support,support)

    def test_projection_conserves_complete_owner_areas_across_meshes(self):
        for level in (2,3,4):
            s,fitted,indices=fixture(level)
            expected=np.bincount(fitted['face_owner'],weights=fitted['area_km2'],minlength=2)
            np.testing.assert_allclose(s.support@s.cell_area,expected,rtol=0,atol=2e-6)
            np.testing.assert_allclose(s.support.sum(axis=0),1.,rtol=0,atol=2e-12)
            geometry=initial_ownership.reconstruct(s,indices)
            self.assertEqual(geometry['diagnostics']['unmapped_segments'],0)
            self.assertEqual(geometry['diagnostics']['fallback_edges'],0)
            self.assertAlmostEqual(geometry['lengths'].sum(),s.native_initial_owner_interfaces['length_km'].sum(),places=7)

    def test_genuine_water_interface_is_preserved_not_categorically_suppressed(self):
        s,_,indices=fixture()
        contour=initial_ownership.reconstruct(s,indices)
        s.native_boundary_geometry=contour
        s.ba,s.bb=s.native_mesh['edge_faces'][indices].T
        s.bp,s.bq=s.plate[s.ba],s.plate[s.bb]
        s.omega=np.array([[0.,0.,0.],[0.,0.,.005]])
        s.active=np.ones(2,bool)
        s.material_surface=dict(vertices=np.empty((0,3)),faces=np.empty((0,3),np.int32))
        exact=native_spreading.prepare(s)
        self.assertGreater(exact['ocean']['length'][exact['ocean']['divergent']].sum(),1000.)
        self.assertAlmostEqual(exact['ocean']['length'].sum(),contour['lengths'].sum(),places=7)
        self.assertEqual(len(exact['continental']['length']),0)

    def test_unrepresented_owner_is_reported_without_fabricated_graph(self):
        # One small real source triangle lies wholly between control centres.
        # It is a valid owner area, despite having no categorical graph key.
        source=mesh_geometry.icosphere(3)
        mesh=mesh_geometry.icosphere(1)
        source_locator=mesh_geometry.build_locator(source['vertices'],source['faces'])
        sampled,_=mesh_geometry.locate_points(mesh['xyz'],source_locator)
        tiny=int(np.setdiff1d(np.arange(len(source['faces'])),sampled)[0])
        owner=np.zeros(len(source['faces']),np.int16);owner[tiny]=1
        fitted=dict(vertices=source['vertices'],faces=source['faces'],face_owner=owner,
            face_kind=np.zeros(len(owner),np.uint8),area_km2=source['area_km2'])
        s=SimpleNamespace(native_mesh=mesh,native_locator=mesh_geometry.build_locator(mesh['vertices'],mesh['faces']),
            plate=owner[sampled],cell_area=mesh['area_km2'],support=np.zeros((2,len(mesh['faces']))),
            initial_geometry_diagnostics={})
        initial_ownership.initialize(s,fitted)
        self.assertGreater(s.support[1]@s.cell_area,0.)
        self.assertTrue(np.all(s.plate==0))
        indices=np.empty(0,np.int32)
        result=initial_ownership.reconstruct(s,indices)
        self.assertEqual(len(result['lengths']),0)
        self.assertEqual(len(result['segments_start']),0)
        self.assertGreater(result['diagnostics']['unmapped_segments'],0)
        self.assertEqual(result['diagnostics']['unmapped_length_km'],result['diagnostics']['source_length_km'])

    def test_geometry_rotates_covariantly(self):
        s,_,indices=fixture()
        before=initial_ownership.reconstruct(s,indices)
        q=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        rotated=deepcopy(s)
        for name in ('edge_mid','edge_normal'):
            rotated.native_mesh[name]=rotated.native_mesh[name]@q.T
        for name in ('start','end','normal'):
            rotated.native_initial_owner_interfaces[name]=rotated.native_initial_owner_interfaces[name]@q.T
        after=initial_ownership.reconstruct(rotated,indices)
        np.testing.assert_allclose(after['lengths'],before['lengths'],rtol=0,atol=1e-8)
        np.testing.assert_allclose(after['midpoints'],before['midpoints']@q.T,atol=1e-13)
        np.testing.assert_allclose(after['normals'],before['normals']@q.T,atol=1e-13)


class InitialTransportTests(unittest.TestCase):
    def setUp(self):
        config=dict(width=48,height=24,coast_geometry_level=2,mesh_level=3,
                    plate_count=4,seed=37,mechanics_nodes=128,adaptive_refinement=0)
        self.s=Simulation(config,make_initial(config))

    def test_force_and_first_production_use_actual_initial_interface(self):
        s=self.s
        self.assertIs(Simulation._forces,raster_engine.Simulation._forces)
        expected=s.support@s.cell_area
        advance=native_spreading.advance
        observations=[]
        def observe(sim,*args,**kwargs):
            self.assertTrue(initial_ownership.active(sim))
            self.assertEqual(sim.boundary_geometry_diagnostics['representation'],'exact fixed initial owner partition')
            self.assertEqual(sim.boundary_geometry_diagnostics['fallback_edges'],0)
            observations.append(len(sim.native_boundary_geometry['segments_start']))
            return advance(sim,*args,**kwargs)
        with patch.object(native_boundary_geometry,'reconstruct',side_effect=AssertionError('raw/P1 fallback at initial epoch')):
            s._rasterize();s._boundaries()
            np.testing.assert_allclose(s.support@s.cell_area,expected,rtol=0,atol=1e-8)
            self.assertTrue(np.all(s.bcode[s.bl==0]==0))
            self.assertTrue(np.all(s.normal_speed[s.bl==0]==0))
            s._forces(.1)
            with patch.object(native_spreading,'advance',observe):s._advect(.1)
        self.assertTrue(observations)
        self.assertFalse(initial_ownership.active(s))
        s._boundaries()
        self.assertIn('support contours',s.boundary_geometry_diagnostics['representation'])

    def test_zero_epoch_checkpoint_restores_geometry_and_first_transport_exactly(self):
        s=self.s
        compatibility=dict(engine_sha256='test',auxiliary_sources_sha256={},numpy_version=np.__version__)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.npz'
            checkpoint.write_checkpoint(path,s,dict(config=s.config),compatibility)
            restored,_=checkpoint.read_checkpoint(path,compatibility,Simulation)
        self.assertTrue(initial_ownership.active(restored))
        restored._boundaries()
        for name in ('midpoints','normals','lengths','segments_start','segments_end','contact_index'):
            np.testing.assert_array_equal(restored.native_boundary_geometry[name],s.native_boundary_geometry[name])
        for state in (s,restored):state._forces(.1);state._advect(.1)
        for name in ('support','plate','age','mass','pos','omega','mantle'):
            np.testing.assert_array_equal(getattr(restored,name),getattr(s,name),err_msg=name)
        self.assertEqual(restored.rng.bit_generator.state,s.rng.bit_generator.state)

    def test_source_capture_includes_both_new_geometry_helpers(self):
        import server
        capture=server.capture_auxiliary_sources(server.ENGINE_SOURCE)
        for name in ('native_initial_ownership.py','native_subduction.py'):
            self.assertIn(name,capture)
            self.assertEqual(capture[name],(server.ROOT/name).read_bytes())

    def test_initial_and_first_evolved_snapshots_validate_geometry_versions(self):
        s=self.s
        initial=s.snapshot()
        self.assertTrue(initial['native_initial_ownership_active'])
        self.assertEqual(initial['native_subduction_version'],1)
        initial_ownership.validate_frame(initial)
        native_subduction.validate_frame(initial)
        native_spreading.validate_frame(initial)
        s.step(.2)
        evolved=s.snapshot()
        self.assertFalse(evolved['native_initial_ownership_active'])
        initial_ownership.validate_frame(evolved)
        native_subduction.validate_frame(evolved)
        native_spreading.validate_frame(evolved)

    def test_initial_saved_schema_rejects_tampered_geometry_and_stripped_gate(self):
        frame=self.s.snapshot()
        mutants=[]
        def change(fn):
            mutant=deepcopy(frame);fn(mutant);mutants.append(mutant)
        change(lambda f:f.pop('native_initial_ownership_version'))
        change(lambda f:f.update(native_initial_ownership_version=True))
        change(lambda f:f.update(native_initial_ownership_active=1))
        change(lambda f:f.update(time_myr=1.))
        change(lambda f:f['mesh_diagnostics']['initial_geometry']['initial_owner_area_km2'].__setitem__(0,float('nan')))
        change(lambda f:f['mesh_diagnostics']['initial_geometry'].update(initial_owner_area_max_error_km2=100.))
        change(lambda f:f['mesh_diagnostics']['initial_geometry'].update(initial_owner_partition_max_error=1e-4))
        change(lambda f:f['boundary_geometry_diagnostics'].update(fallback_edges=1))
        change(lambda f:f['boundary_geometry_diagnostics'].update(source_length_km=1.))
        change(lambda f:f['boundary_geometry_diagnostics'].update(contour_segments=1_000_000))
        change(lambda f:f['boundary_segments'].pop())
        change(lambda f:f['boundary_segments'][0]['geometry_xyz'][0].__setitem__(0,10.))
        for index,mutant in enumerate(mutants):
            with self.subTest(mutant=index),self.assertRaises(ValueError):initial_ownership.validate_frame(mutant)
        initial_ownership.validate_frame({})
        # Both persistent history and public saved sampling enforce the gate.
        import mesh_history,native_frame_sampling
        for validate in (mesh_history.arrays,native_frame_sampling.prepare):
            with self.assertRaises(ValueError):validate(mutants[0])


if __name__=='__main__':unittest.main()
