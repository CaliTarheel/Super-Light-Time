"""Runtime geometry adaptation preserves material, columns and marker histories."""
from copy import deepcopy
from types import SimpleNamespace,MethodType
import unittest
import numpy as np

import material_surface
from mesh_geometry import icosphere,build_locator,locate_points,geometry
import structure_engine
import native_rift_material
from native_material_adaptivity import adapt,_prospective_belts


def unit(p):return p/np.maximum(np.linalg.norm(p,axis=-1,keepdims=True),1e-30)


def world():
    control=icosphere(1);n=len(control['faces']);ids=1000+np.arange(n)
    surface=material_surface.initialize_surface(control['vertices'],control['faces'],np.zeros(n,int),np.ones(n,int),face_id=ids)
    pos=material_surface.face_centres(surface)
    trace=unit(np.einsum('j,njk->nk',np.array([.2,.3,.5]),surface['vertices'][surface['faces']]))
    s=SimpleNamespace(material_surface=surface,pos=pos,mass=surface['reference_area_km2'].copy(),kind=np.ones(n,np.uint8),
        parcel_patch=ids.copy(),parcel_plate=np.zeros(n,np.int16),parcel_craton=np.full(n,-1,np.int32),
        parcel_birth=np.zeros(n),parcel_extra_vector=np.column_stack((np.arange(n),np.arange(n)+1,np.arange(n)+2)).astype(float),
        relief=np.full(n,100.),suture=np.full(n,.2),rift_id=np.zeros(n,np.int64),rift_birth_myr=np.full(n,-1.),
        rift_tangent=np.zeros((n,3)),rift_extension_m=np.zeros(n),inversion_uplift_m=np.zeros(n),geometric_log_area=np.zeros(n),
        material_lineage=dict(face_ids=ids.copy(),root_id=ids.copy(),parent_id=np.full(n,-1,np.int64),
            reference_corners=np.broadcast_to(np.eye(3),(n,3,3)).copy(),level=np.zeros(n,np.int32)),
        material_deformation=dict(face_weight=np.ones(n),face_rigid=np.zeros(n,bool),face_strain=np.full(n,.001)),
        config=dict(adaptive_refinement=2,deformation_width_km=400.,material_face_budget=0,mechanics_nodes=128),
        t=2.,initial_material_faces=n,next_patch_uid=2000,
        trace_patch=ids.copy(),trace_xyz=trace,trace_kind=np.ones(n,np.uint8),trace_plate=np.zeros(n,np.int16),
        trace_id=np.arange(n),trace_uplift_m=np.arange(n,dtype=float),trace_relief_m=np.full(n,100.),trace_suture=np.full(n,.2),
        rift_properties=dict(patch=ids.copy(),damage=np.zeros(n),strength=np.ones(n)),
        rift_material=dict(version=2,backend='native_material_triangles',target_nodes=128,next_base_id=n,
            patch_ids=ids.copy(),patch_bases=np.arange(n),base_edges=np.empty((0,2),np.int64),
            topology_signature='',topology_revision=0,face_contacts=np.empty((0,2),np.int32)),
        rift_bonds={(0,1,31):dict(damage=.73,strain=.18,extension_km=12.)},rift_pending={'old_mesh':True},
        native_domains=dict(material_ids=ids.copy(),material_domains=np.ones(n,np.int64),records={1:dict(plate_id=0,plate_uid=31)}),
        material_domain=np.ones(n,np.int64),plate_uid=np.array([31,42]),omega=np.array([[0.,0.,.001],[0.,0.,-.001]]),
        n=n,xyz=control['xyz'].copy(),age=np.arange(n,dtype=float),native_mesh=control,
        bmid=np.empty((0,3)),bn=np.empty((0,3)),bl=np.empty(0),bp=np.empty(0,int),bq=np.empty(0,int),bcode=np.empty(0,int),
        _coverage_signature='old',_owner_occupancy_signature='old',_material_coverage={'old':True})
    locator=build_locator(control['vertices'],control['faces'])
    s._indices=lambda points:locate_points(points,locator)[0]
    def sync(self):
        self.pos=material_surface.face_centres(self.material_surface)
        self.parcel_cell=self._indices(self.pos)
        self.parcel_east=unit(np.cross([0.,0.,1.],self.pos))
        self.parcel_extent=np.repeat(np.sqrt(self.mass)[:,None],2,axis=1)
    s._sync_material=MethodType(sync,s);s._sync_material()
    structure_engine.initialize_parcels(s)
    structure_engine.initialize_traces(s,np.arange(n))
    return s


class NativeMaterialAdaptivityTests(unittest.TestCase):
    def test_refinement_transfers_every_material_column_and_preserves_control_and_markers(self):
        s=world();old=deepcopy(s);old_n=len(s.mass)
        self.assertTrue(adapt(s,2.))
        self.assertEqual(len(s.mass),4*old_n)
        self.assertTrue(np.all(geometry(s.material_surface['vertices'],s.material_surface['faces'])['edge_faces']>=0))
        self.assertAlmostEqual(s.mass.sum()/old.mass.sum(),1.,places=14)
        self.assertAlmostEqual(np.sum(s.mass*s.structure['area_factor']*s.structure['thickness_km'])/
            np.sum(old.mass*old.structure['area_factor']*old.structure['thickness_km']),1.,places=14)
        parent=s.material_lineage['parent_id']-1000
        for name in ('parcel_birth','parcel_craton','parcel_extra_vector','suture','relief','rift_id','rift_tangent','inversion_uplift_m'):
            np.testing.assert_array_equal(getattr(s,name),getattr(old,name)[parent])
        for name in s.structure:np.testing.assert_allclose(s.structure[name],old.structure[name][parent],rtol=2e-15)
        np.testing.assert_array_equal(s.age,old.age)
        np.testing.assert_array_equal(s.xyz,old.xyz)
        np.testing.assert_array_equal(s.trace_xyz,old.trace_xyz)
        np.testing.assert_array_equal(s.trace_uplift_m,old.trace_uplift_m)
        for name in s.trace_structure:np.testing.assert_array_equal(s.trace_structure[name],old.trace_structure[name])
        self.assertTrue(np.all(np.isin(s.trace_patch,s.parcel_patch)))
        self.assertIsNone(s.rift_pending)
        self.assertIsNone(s._coverage_signature)
        self.assertEqual(s.rift_bonds,old.rift_bonds)
        np.testing.assert_array_equal(s.material_surface['reference_area_km2'],s.mass)

    def test_root_charts_and_marker_containment_survive_remesh_exactly(self):
        s=world();old=deepcopy(s);adapt(s,2.)
        at={int(uid):i for i,uid in enumerate(s.parcel_patch)}
        faces=np.array([at[int(uid)] for uid in s.trace_patch])
        triangle=s.material_surface['vertices'][s.material_surface['faces'][faces]]
        bary=np.linalg.solve(triangle.transpose(0,2,1),s.trace_xyz[...,None])[...,0]
        bary/=bary.sum(axis=1)[:,None]
        self.assertTrue(np.all(bary>=-1e-12))
        chart=np.einsum('ni,nij->nj',bary,s.material_lineage['reference_corners'][faces]);chart/=chart.sum(axis=1)[:,None]
        np.testing.assert_allclose(chart,np.broadcast_to([.2,.3,.5],chart.shape),atol=7e-16)
        np.testing.assert_array_equal(s.material_lineage['root_id'][faces],old.trace_patch)

    def test_mechanics_bases_and_fault_damage_survive_refinement(self):
        s=world();old_bonds=deepcopy(s.rift_bonds);adapt(s,2.)
        mechanical=native_rift_material.refresh(s)
        self.assertEqual(len(mechanical['xyz']),80)
        np.testing.assert_array_equal(mechanical['bases'],np.arange(80))
        self.assertEqual(s.rift_bonds,old_bonds)
        self.assertTrue(np.all(mechanical['trace_node']>=0))
        self.assertTrue(np.all(s.native_domains['material_domains']==1))

    def test_quiet_exact_siblings_coarsen_after_hysteresis_without_losing_ids_or_volume(self):
        s=world();original=deepcopy(s);adapt(s,2.)
        for t in (12.,22.,32.):
            s.t=t;s.material_deformation=dict(face_weight=np.zeros(len(s.mass)),face_rigid=np.ones(len(s.mass),bool),face_strain=np.zeros(len(s.mass)))
            self.assertFalse(adapt(s,2.))
        s.t=42.
        self.assertTrue(adapt(s,2.))
        self.assertEqual(len(s.mass),len(original.mass))
        np.testing.assert_array_equal(s.parcel_patch,original.parcel_patch)
        np.testing.assert_array_equal(s.material_surface['faces'],original.material_surface['faces'])
        np.testing.assert_array_equal(s.trace_patch,original.trace_patch)
        np.testing.assert_allclose(s.mass,original.mass,rtol=4e-16)
        self.assertEqual(len(s.material_adaptivity['registries']),0)

    def test_distinct_child_column_history_blocks_coarsening(self):
        s=world();adapt(s,2.)
        s.structure['denudation_m'][0]=3.
        s.material_deformation=dict(face_weight=np.zeros(len(s.mass)),face_rigid=np.ones(len(s.mass),bool),face_strain=np.zeros(len(s.mass)))
        s.t=42.
        self.assertFalse(adapt(s,2.))
        self.assertEqual(len(s.mass),320)
        self.assertEqual(s.structure['denudation_m'][0],3.)

    def test_registry_survives_whole_family_owner_vertex_duplication(self):
        s=world();original=deepcopy(s);adapt(s,2.)
        moving=s.material_lineage['root_id']==1000
        s.parcel_plate[moving]=1
        material_surface.reassign_owners(s.material_surface,s.parcel_plate)
        s.trace_plate[0]=1
        s.material_deformation=dict(face_weight=np.zeros(len(s.mass)),face_rigid=np.ones(len(s.mass),bool),face_strain=np.zeros(len(s.mass)))
        s.t=42.
        self.assertTrue(adapt(s,2.))
        self.assertEqual(len(s.mass),len(original.mass))
        self.assertEqual(s.parcel_plate[np.flatnonzero(s.parcel_patch==1000)[0]],1)
        np.testing.assert_allclose(s.mass,original.mass,rtol=4e-16)
        np.testing.assert_array_equal(s.trace_patch,original.trace_patch)
        self.assertLess(abs(s.adaptivity_diagnostics['coarsening']['geometric_volume_error_fraction']),1e-12)

    def test_surface_categorical_provenance_is_copied_exactly(self):
        s=world();s.material_surface['provenance']['source_name']=np.array(['root-'+str(i) for i in range(len(s.mass))])
        adapt(s,2.)
        parent=s.material_lineage['root_id']-1000
        np.testing.assert_array_equal(s.material_surface['provenance']['source_name'],['root-'+str(i) for i in parent])

    def test_unknown_marker_aborts_transaction_without_changing_surface_or_ledger(self):
        s=world();s.trace_patch[0]=999999;old=deepcopy(s)
        self.assertFalse(adapt(s,2.))
        np.testing.assert_array_equal(s.material_surface['faces'],old.material_surface['faces'])
        np.testing.assert_array_equal(s.mass,old.mass)
        np.testing.assert_array_equal(s.parcel_patch,old.parcel_patch)
        self.assertEqual(s.adaptivity_diagnostics['refinement_skipped']['unresolved_markers'],1)

    def test_cadence_budget_disabled_and_growing_arc_exclusion(self):
        s=world();s.config['adaptive_refinement']=0
        self.assertFalse(adapt(s,2.));self.assertEqual(len(s.mass),80)
        s.config['adaptive_refinement']=2;s.config['material_face_budget']=90
        self.assertTrue(adapt(s,2.));self.assertLessEqual(len(s.mass),90)
        count=len(s.mass);s.t=4.;self.assertFalse(adapt(s,2.));self.assertEqual(len(s.mass),count)
        arc=world();arc.kind[:]=3;arc.material_surface['face_kind'][:]=3
        self.assertFalse(adapt(arc,2.));self.assertEqual(len(arc.mass),80)

    def test_prospective_face_belt_intersection_refines_without_fake_deformation(self):
        s=world();s.material_deformation=dict(face_weight=np.zeros(80),face_rigid=np.ones(80,bool),face_strain=np.zeros(80))
        middle=s.pos[0];normal=unit(np.cross(middle,[0.,0.,1.]))
        s.bmid=middle[None];s.bn=normal[None];s.bl=np.array([50.]);s.bp=np.array([0]);s.bq=np.array([1]);s.bcode=np.array([5])
        selected=_prospective_belts(s,100.)
        self.assertTrue(selected[0]);self.assertLess(np.mean(selected),.2)
        self.assertTrue(adapt(s,2.))
        self.assertTrue(np.all(s.material_deformation['face_weight']==0.))
        self.assertTrue(np.all(s.material_deformation['face_rigid']))
        np.testing.assert_array_equal(s.geometric_log_area,np.zeros(len(s.mass)))
        blocked=world();blocked.bmid=s.bmid;blocked.bn=s.bn;blocked.bl=s.bl;blocked.bp=s.bp;blocked.bq=s.bq;blocked.bcode=s.bcode
        blocked._valid_loading_edges=lambda:np.array([False])
        self.assertFalse(np.any(_prospective_belts(blocked,100.)))

    def test_actual_native_step_adapts_and_snapshot_keeps_lineage_aligned(self):
        from native_engine import Simulation
        s=Simulation(dict(width=96,height=48,mesh_level=2,plate_count=4,mechanics_nodes=128,
            duration_myr=20.,adaptive_refinement=1,deforming_regions=1))
        original=len(s.mass);s.step(2.)
        self.assertGreater(len(s.mass),original)
        self.assertEqual(len(s.material_lineage['face_ids']),len(s.mass))
        self.assertTrue(np.all(np.isin(s.trace_patch,s.parcel_patch)))
        self.assertLess(abs(s.adaptivity_diagnostics['refinement']['mass_error_fraction']),1e-12)
        snapshot=s.snapshot()
        self.assertEqual(len(snapshot['material_reference_corners']),len(s.mass))
        self.assertEqual(len(snapshot['material_root_id']),len(s.mass))


if __name__=='__main__':unittest.main()
