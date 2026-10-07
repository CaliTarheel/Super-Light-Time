"""Finite source integration, honest placement, material provenance and restart."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
import lip_events as lip
import crustal_structure as columns
import structure_engine as structure
import material_surface
import native_material_adaptivity as adapt
import checkpoint


def event(**updates):
    row=dict(id='province-a',lon_deg=0.,lat_deg=0.,radius_km=500.,start_myr=.25,
        duration_myr=.5,volume_km3=100000.,intrusive_fraction=.7,
        thermal_support_m=450.,cooling_myr=30.)
    row.update(updates);return row


def world(events=None,enabled=True):
    vertices=np.array([[1,0,0],[1,-.2,-.2],[1,.2,-.2],[1,.2,.2],[1,-.2,.2]],float)
    vertices/=np.linalg.norm(vertices,axis=1)[:,None]
    faces=np.array([[0,1,2],[0,2,3],[0,3,4],[0,4,1]],np.int32)
    area=material_surface.spherical_face_areas(vertices,faces)
    pos=vertices[faces].mean(axis=1);pos/=np.linalg.norm(pos,axis=1)[:,None]
    config=dict(duration_myr=20.,lip_events=dict(version=1,enabled=enabled,seed=33,
        generated_count=0,events=[event()] if events is None else events))
    s=SimpleNamespace(config=config,t=0.,rng=np.random.default_rng(12),mass=area.copy(),pos=pos,
        kind=np.ones(4,np.uint8),parcel_patch=np.arange(10,14,dtype=np.int64),
        parcel_arc_id=np.zeros(4,np.int64),trace_patch=np.arange(10,14,dtype=np.int64),
        parcel_plate=np.zeros(4,np.int32),trace_plate=np.zeros(4,np.int32))
    s.material_surface=dict(vertices=vertices,faces=faces,area_km2=area,radius_km=6371.)
    s.structure=structure._new(s.kind,np.zeros(4));s.trace_structure=deepcopy(s.structure)
    lip.initialize(s);return s


def advance(s,dt):
    s.t+=dt
    result=lip.evolve_columns(s,dt)
    lip.evolve_columns(s,dt,trace=True)
    return result


class LipEventsTests(unittest.TestCase):
    def test_disabled_is_exact_noop_and_rng_untouched(self):
        s=world(enabled=False);before=deepcopy(vars(s));rng=deepcopy(s.rng.bit_generator.state)
        result=advance(s,2.)
        self.assertFalse(hasattr(s,'lip_version'))
        self.assertEqual(s.rng.bit_generator.state,rng)
        for key,value in before['structure'].items():np.testing.assert_array_equal(s.structure[key],value)
        self.assertFalse(result['magmatic_m'].any());self.assertEqual(lip.snapshot(s),{})

    def test_short_pulse_integrates_identically_across_step_partition(self):
        whole=world();split=world();advance(whole,2.)
        for _ in range(8):advance(split,.25)
        self.assertAlmostEqual(whole.lip_events[0]['supplied_volume_km3'],100000.,places=7)
        self.assertAlmostEqual(whole.lip_events[0]['rejected_volume_km3'],0.,places=6)
        for name in ('parcel_lip_intrusive','parcel_lip_extrusive','parcel_lip_heat'):
            np.testing.assert_allclose(getattr(whole,name),getattr(split,name),rtol=2e-13,atol=1e-12)
        np.testing.assert_allclose(whole.structure['thickness_km'],split.structure['thickness_km'],atol=3e-14)

    def test_source_adds_physical_volume_once_and_no_reference_area(self):
        s=world();area=s.mass.copy();old=float(area@s.structure['thickness_km']);advance(s,2.)
        self.assertAlmostEqual(float(area@s.structure['thickness_km'])-old,100000.,places=6)
        self.assertAlmostEqual(float(area@s.structure['added_volume_km_per_reference_km2']),100000.,places=6)
        self.assertAlmostEqual(float(area@s.parcel_lip_intrusive[:,0]),70000.,places=6)
        self.assertAlmostEqual(float(area@s.parcel_lip_extrusive[:,0]),30000.,places=6)
        np.testing.assert_array_equal(s.mass,area);np.testing.assert_array_equal(s.parcel_arc_id,0)

    def test_unsupported_footprint_share_is_rejected_not_redistributed(self):
        s=world();s.kind[:2]=3;s.parcel_arc_id[:2]=1;before=s.structure['thickness_km'].copy();advance(s,2.)
        self.assertAlmostEqual(s.lip_events[0]['rejected_unoccupied_km3'],50000.,places=5)
        np.testing.assert_array_equal(s.structure['thickness_km'][:2],before[:2])
        self.assertAlmostEqual(float(s.mass@(s.structure['thickness_km']-before)),50000.,places=5)

    def test_capacity_rejection_and_overlapping_source_mixtures(self):
        s=world([event(),event(id='province-b')]);s.structure['thickness_km'][:]=74.999
        s.trace_structure['thickness_km'][:]=74.999;advance(s,2.)
        self.assertTrue(np.all(s.structure['thickness_km']<=75.))
        self.assertGreater(s.lip_events[0]['rejected_capacity_km3'],0.)
        self.assertEqual(s.lip_events[1]['intrusive_volume_km3'],0.)
        lip.validate_state(s)
        s=world([event(),event(id='province-b',intrusive_fraction=.2)]);advance(s,2.)
        self.assertEqual(s.parcel_lip_intrusive.shape,(4,2));lip.validate_state(s)
        self.assertAlmostEqual(s.lip_events[1]['intrusive_volume_km3'],20000.,places=6)

    def test_overlapping_material_does_not_double_source(self):
        s=world();s.material_surface['faces'][2:]=s.material_surface['faces'][:2]
        advance(s,2.)
        self.assertAlmostEqual(s.lip_events[0]['rejected_unoccupied_km3'],50000.,places=5)
        self.assertAlmostEqual(s.lip_events[0]['intrusive_volume_km3'],35000.,places=5)

    def test_cooling_removes_support_but_preserves_material_origin(self):
        s=world();advance(s,2.);heat=s.structure['lip_heat_m'].copy();thickness=s.structure['thickness_km'].copy()
        provenance=s.parcel_lip_extrusive.copy();result=advance(s,30.)
        np.testing.assert_allclose(s.structure['lip_heat_m'],heat/np.e,rtol=2e-15)
        np.testing.assert_array_equal(s.structure['thickness_km'],thickness)
        np.testing.assert_array_equal(s.parcel_lip_extrusive,provenance)
        self.assertTrue(np.all(result['thermal_m']<0))

    def test_generated_schedule_does_not_consume_simulation_random_stream(self):
        a=world(enabled=False);b=world(enabled=False)
        for s in (a,b):
            s.config['lip_events'].update(enabled=True,generated_count=2,events=[])
            before=deepcopy(s.rng.bit_generator.state);lip.initialize(s)
            self.assertEqual(s.rng.bit_generator.state,before)
        self.assertEqual(a.lip_events,b.lip_events)

    def erupted(self,steps=(1.25,)):
        s=world(events=[event(id='generated-1',start_myr=1.,duration_myr=.5)])
        row=s.lip_events[0]
        row['generated']=True
        # Its t=0 site, which no current material occupies.
        row['lon_deg'],row['lat_deg']=0.,0.
        for dt in steps: advance(s,dt)
        return s,row

    def test_a_generated_source_chooses_its_ground_when_it_erupts(self):
        # initialize freezes a site in t=0 geography while generated start times
        # are uniform over the whole run, so a late source erupts where its
        # continent used to be. The host is re-chosen when the source opens.
        s,row=self.erupted()
        self.assertTrue(row['started'])
        self.assertEqual(row['placed_myr'],s.t)
        lon,lat=np.radians([row['lon_deg'],row['lat_deg']])
        centre=np.array([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])
        self.assertTrue(np.isclose(s.pos@centre,1.).any(),'centre must sit on current material')

    def test_an_authored_source_keeps_the_coordinates_it_was_given(self):
        s=world(events=[event(id='province-a',lon_deg=4.,lat_deg=-3.,start_myr=1.)])
        advance(s,1.25)
        row=s.lip_events[0]
        self.assertTrue(row['started'])
        self.assertEqual((row['lon_deg'],row['lat_deg']),(4.,-3.))
        self.assertNotIn('placed_myr',row)

    def test_generated_placement_is_reproducible_from_the_seed_and_id(self):
        first,second=self.erupted()[1],self.erupted()[1]
        self.assertEqual((first['lon_deg'],first['lat_deg']),(second['lon_deg'],second['lat_deg']))

    def test_generated_placement_survives_a_step_that_ends_exactly_at_onset(self):
        # The native engine splits every step at lip.next_transition, so the
        # step that reaches start_myr ends exactly on it and supplies nothing
        # while marking the source started. Placement must still happen there,
        # before the first deposit, not be skipped for the rest of the event.
        whole,expected=self.erupted()
        split,row=self.erupted(steps=(1.,.25))
        self.assertEqual(row['placed_myr'],1.)
        self.assertEqual((row['lon_deg'],row['lat_deg']),(expected['lon_deg'],expected['lat_deg']))
        self.assertAlmostEqual(row['supplied_volume_km3'],expected['supplied_volume_km3'],places=6)
        np.testing.assert_allclose(split.structure['thickness_km'],whole.structure['thickness_km'],atol=1e-12)

    def test_checkpoint_resume_preserves_future_source_and_provenance(self):
        s=world([event(start_myr=1.,duration_myr=2.)]);advance(s,2.)
        manifest=dict(config=s.config);compatibility={}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.npz';checkpoint.write_checkpoint(path,s,manifest,compatibility)
            restored,_=checkpoint.read_checkpoint(path,compatibility,SimpleNamespace)
            advance(s,2.);advance(restored,2.)
        for name in ('parcel_lip_intrusive','parcel_lip_extrusive','parcel_lip_heat'):
            np.testing.assert_array_equal(getattr(s,name),getattr(restored,name))
        self.assertEqual(s.lip_events,restored.lip_events)

    def test_refinement_transfer_and_owner_change_keep_source_inventory(self):
        s=world();advance(s,2.);old=s.mass.copy()
        mapping=dict(source_indptr=np.arange(9,dtype=np.int64),source_indices=np.repeat(np.arange(4),2),
                     source_area_fractions=np.full(8,.5))
        # Exercise the actual adaptation transfer kernel used for parcel cohorts.
        for name in ('parcel_lip_intrusive','parcel_lip_extrusive'):
            value=getattr(s,name)
            # The production mapping's extensive fractions are reference shares.
            try: transferred=adapt._transfer(value,mapping,old)
            except KeyError as exc: self.fail('Fixture must use actual mapping schema: '+str(exc))
            np.testing.assert_allclose(np.repeat(old/2,2)@transferred,old@value,rtol=2e-15)
        s.parcel_plate[:]=1;lip.validate_state(s)

    def test_saved_schema_rejects_missing_provenance_and_tampering(self):
        s=world();advance(s,2.)
        frame=dict(material_faces=s.material_surface['faces'],material_reference_area_km2=s.mass,**lip.snapshot(s))
        lip.validate_frame(frame)
        bad=deepcopy(frame);bad['material_lip_intrusive_km_per_reference_km2'][0,0]+=1.
        with self.assertRaisesRegex(ValueError,'provenance'):lip.validate_frame(bad)
        bad=deepcopy(frame);bad.pop('material_lip_heat_m')
        with self.assertRaisesRegex(ValueError,'incomplete'):lip.validate_frame(bad)

    def test_public_native_step_closes_columns_markers_and_saved_arrays(self):
        from native_engine import Simulation
        import mesh_history
        config=dict(width=48,height=24,mesh_level=2,plate_count=4,seed=12,duration_myr=2,
            erosion=0.,deforming_regions=0,adaptive_refinement=0)
        initial=Simulation(config)
        point=initial.pos[np.flatnonzero(initial.kind==1)[0]]
        row=event(lon_deg=float(np.degrees(np.arctan2(point[1],point[0]))),
            lat_deg=float(np.degrees(np.arcsin(point[2]))),start_myr=0.,duration_myr=.1)
        config['lip_events']=dict(version=1,enabled=True,seed=33,generated_count=0,events=[row])
        s=Simulation(config);s.step(.1)
        lip.validate_state(s)
        self.assertGreater(s.lip_events[0]['intrusive_volume_km3'],0.)
        self.assertLess(abs(s.material_column_budget['residual_km3']),1e-4)
        np.testing.assert_allclose(columns.elevation(s.structure),structure.material_height(s.kind,s.relief),atol=3e-9)
        np.testing.assert_allclose(columns.elevation(s.trace_structure),structure.material_height(s.trace_kind,s.trace_relief_m),atol=3e-9)
        frame=s.snapshot();arrays=mesh_history.arrays(frame)
        self.assertEqual(arrays['material_lip_heat_m'].shape,(len(s.mass),1))
        self.assertEqual(frame['lip_deposited_km'].shape,(s.w*s.h,))

    def test_actual_native_refinement_preserves_lip_cohorts_and_heat(self):
        from tests.test_native_material_adaptivity import world as adaptive_world
        s=adaptive_world();s.config['lip_events']=dict(version=1,enabled=True,seed=33,
            generated_count=0,events=[event(start_myr=s.t,duration_myr=.1,radius_km=2000)])
        lip.initialize(s);advance(s,.1);count=len(s.mass)
        self.assertTrue(adapt.adapt(s,.1));self.assertGreater(len(s.mass),count)
        lip.validate_state(s)
        np.testing.assert_allclose(s.parcel_lip_heat.sum(axis=1),s.structure['lip_heat_m'],atol=1e-12)


if __name__=='__main__':unittest.main()
