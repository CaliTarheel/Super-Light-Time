"""Connected volcanic geometry, source volume and surviving material histories."""
from copy import deepcopy
import pickle
import unittest
import numpy as np

from test_native_processes import ocean_fixture
import native_arc_material as arcs
import native_material_adaptivity as adaptivity
import native_material_evolution as evolution
import material_surface
import material_transport
import crustal_structure as columns
import structure_engine
from mesh_geometry import connected_components


def frame(s):
    return dict(material_vertices=s.material_surface['vertices'].copy(),material_faces=s.material_surface['faces'].copy(),
        material_face_id=s.parcel_patch.copy(),**evolution.snapshot_fields(s),**arcs.snapshot_fields(s))


class NativeArcMaterialTests(unittest.TestCase):
    def test_connected_birth_has_interior_flanks_and_submerged_apron(self):
        s=ocean_fixture();result=arcs.add_arc_crust(s,np.array([17]),np.array([1000.]))
        self.assertEqual(result['new_faces'],24)
        surface=s.material_surface
        self.assertEqual(len(surface['vertices']),17)
        faces=surface['faces']
        edges=np.sort(np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1)
        unique,counts=np.unique(edges,axis=0,return_counts=True)
        self.assertEqual(np.count_nonzero(counts==1),8)
        self.assertTrue(np.all(counts<=2))
        self.assertEqual(len(surface['vertices'])-len(unique)+len(faces),1)
        self.assertTrue(np.all(s.parcel_arc_id==1))
        height=structure_engine.material_height(s.kind,s.relief)
        self.assertGreater(height.max(),900.)
        self.assertLess(height.min(),-1500.)
        self.assertGreater(np.ptp(s.structure['thickness_km']),10.)
        np.testing.assert_allclose(columns.elevation(s.structure),height,atol=1e-12)
        self.assertAlmostEqual(s.mass.sum(),1000.)
        self.assertAlmostEqual(result['added_volume_km3'],float(surface['area_km2']@s.structure['thickness_km']),places=8)

    def test_growth_keeps_core_and_adds_fixed_source_volume_not_thick_old_columns(self):
        s=ocean_fixture();arcs.add_arc_crust(s,np.array([17]),np.array([1000.]))
        s.structure['thickness_km'][:]=60.
        s.relief=columns.elevation(s.structure)-120.
        initial_vertices=s.material_surface['vertices'].copy();ids=s.parcel_patch.copy()
        old_volume=float(s.material_surface['area_km2']@s.structure['thickness_km'])
        result=arcs.add_arc_crust(s,np.array([17]),np.array([2500.]))
        self.assertEqual(result['grown_patches'],1);self.assertEqual(result['new_faces'],0)
        np.testing.assert_array_equal(ids,s.parcel_patch)
        np.testing.assert_array_equal(s.material_surface['vertices'][:9],initial_vertices[:9])
        self.assertGreater(np.max(np.abs(s.material_surface['vertices'][9:]-initial_vertices[9:])),0.)
        actual=float(s.material_surface['area_km2']@s.structure['thickness_km'])
        self.assertAlmostEqual(actual-old_volume,2500.*25.,places=6)
        self.assertLess(actual-old_volume,2500.*60.)
        self.assertAlmostEqual(s.mass.sum(),3500.,places=9)
        np.testing.assert_allclose(s.structure['area_factor'],s.material_surface['area_km2']/s.mass,rtol=0,atol=0)

    def test_growth_preserves_cumulative_point_history_and_exact_chart_transport(self):
        s=ocean_fixture();arcs.add_arc_crust(s,np.array([17]),np.array([1000.]))
        preserved=('denudation_m','rebound_m','thermal_uplift_m','thermal_subsidence_m','magmatic_uplift_m')
        for i,name in enumerate(preserved):
            s.structure[name][:]=20.+i;s.trace_structure[name][:]=10.+i
        old={name:s.structure[name].copy() for name in preserved}
        old_trace={name:s.trace_structure[name].copy() for name in preserved}
        before=material_transport.prepare(frame(s));xyz=s.trace_xyz.copy();patches=s.trace_patch.copy()
        order=np.argsort(s.parcel_patch);face=order[np.searchsorted(s.parcel_patch[order],patches)]
        triangle=s.material_surface['vertices'][s.material_surface['faces'][face]]
        bary=np.linalg.solve(triangle.transpose(0,2,1),xyz[...,None])[...,0];bary/=bary.sum(axis=1)[:,None]
        arcs.add_arc_crust(s,np.array([17]),np.array([4000.]))
        after=material_transport.prepare(frame(s));mapped=material_transport.map_hits(before,after,face,bary)
        np.testing.assert_allclose(mapped['xyz'],s.trace_xyz,rtol=0,atol=5e-16)
        np.testing.assert_array_equal(patches,s.trace_patch)
        for name in preserved:
            np.testing.assert_array_equal(s.structure[name],old[name])
            np.testing.assert_array_equal(s.trace_structure[name],old_trace[name])

    def test_empty_ocean_refines_connected_arc_then_growth_retains_ancestry(self):
        s=ocean_fixture();s.config['adaptive_refinement']=2;s.config['deformation_width_km']=80.
        arcs.add_arc_crust(s,np.array([17]),np.array([100000.]))
        old_roots=s.parcel_patch.copy();old_mass=s.mass.copy();evolution.ensure_lineage(s)
        s.material_deformation=dict(face_weight=np.ones(len(s.mass)),face_rigid=np.zeros(len(s.mass),bool),face_strain=np.zeros(len(s.mass)))
        s.t=2.
        self.assertTrue(adaptivity.adapt(s,2.))
        self.assertGreater(len(s.mass),24);self.assertLessEqual(len(s.mass),96)
        self.assertEqual(s.adaptivity_diagnostics['face_budget'],96)
        self.assertTrue(np.all(s.parcel_arc_id==1))
        np.testing.assert_array_equal(np.unique(s.material_lineage['root_id']),old_roots)
        order=np.argsort(old_roots);at=np.searchsorted(old_roots[order],s.material_lineage['root_id'])
        np.testing.assert_allclose(np.bincount(at,weights=s.mass,minlength=24),old_mass[order],rtol=1e-14)
        charts=s.material_lineage['reference_corners'].copy();ids=s.parcel_patch.copy()
        old_volume=float(s.material_surface['area_km2']@s.structure['thickness_km'])
        result=arcs.add_arc_crust(s,np.array([17]),np.array([20000.]))
        self.assertEqual(result['new_faces'],0);self.assertEqual(result['grown_patches'],1)
        np.testing.assert_array_equal(ids,s.parcel_patch)
        np.testing.assert_array_equal(charts,s.material_lineage['reference_corners'])
        self.assertAlmostEqual(float(s.material_surface['area_km2']@s.structure['thickness_km'])-old_volume,20000.*25.,places=5)
        self.assertTrue(np.all(np.isin(s.trace_patch,s.parcel_patch)))

    def test_legacy_repeated_growth_and_kind_change_preserve_old_material(self):
        s=ocean_fixture()
        # Historical cell-only worlds allowed a new arc over reclassified
        # continental material. Preserve that legacy replay contract; marked
        # finite-footprint worlds explicitly reject that source instead.
        s.native_arc_emplacement_version=0
        arcs.add_arc_crust(s,np.array([17]),np.array([1000.]))
        for _ in range(20):arcs.add_arc_crust(s,np.array([17]),np.array([1000.]))
        self.assertEqual(len(s.mass),24);self.assertEqual(s.next_arc_material_id,2)
        self.assertAlmostEqual(s.mass.sum(),21000.,places=7)
        s.kind[:]=1;s._sync_material()
        before=s.material_surface['vertices'].copy()
        arcs.add_arc_crust(s,np.array([17]),np.array([1000.]))
        self.assertEqual(len(s.mass),48)
        np.testing.assert_array_equal(s.material_surface['vertices'][:len(before)],before)
        self.assertEqual(len(np.unique(s.parcel_arc_id)),2)

    def test_checkpoint_continuation_matches_after_growth_and_refinement(self):
        s=ocean_fixture();s.config['adaptive_refinement']=1;s.config['deformation_width_km']=80.
        arcs.add_arc_crust(s,np.array([17]),np.array([100000.]))
        s.material_deformation=dict(face_weight=np.ones(len(s.mass)),face_rigid=np.zeros(len(s.mass),bool),face_strain=np.zeros(len(s.mass)))
        s.t=2.;adaptivity.adapt(s,2.)
        restored=pickle.loads(pickle.dumps(s,protocol=5))
        for world in (s,restored):arcs.add_arc_crust(world,np.array([17]),np.array([4000.]))
        for name in ('mass','parcel_patch','parcel_arc_id','parcel_arc_basal_m','trace_patch','trace_xyz'):
            np.testing.assert_array_equal(getattr(s,name),getattr(restored,name))
        for key in s.structure:np.testing.assert_array_equal(s.structure[key],restored.structure[key])
        for key in s.material_lineage:np.testing.assert_array_equal(s.material_lineage[key],restored.material_lineage[key])
        np.testing.assert_array_equal(s.material_surface['vertices'],restored.material_surface['vertices'])

    def test_snapshot_fields_are_pure_once_initialized(self):
        s=ocean_fixture();arcs.ensure_fields(s)
        before=pickle.dumps(s,protocol=5)
        result=arcs.snapshot_fields(s)
        self.assertEqual(before,pickle.dumps(s,protocol=5))
        self.assertEqual(result['arc_material_version'],1)
        self.assertEqual(result['material_arc_id'].shape,(0,))

    def test_growth_after_crossing_pole_keeps_shared_patch_and_markers(self):
        s=ocean_fixture();arcs.add_arc_crust(s,np.array([17]),np.array([10000.]))
        ids=s.parcel_patch.copy();s.omega[:]=[.045,.017,.002]
        for _ in range(30):
            evolution.advect(s,2.);s._sync_material()
        centre=arcs._unit(np.sum(s.pos*s.material_surface['area_km2'][:,None],axis=0))
        cell=int(s._indices(centre[None])[0])
        # Rotation changes the bookkeeping cell; growth still occurs at the
        # actual transported arc center, not that cell's unrelated center.
        result=arcs.add_arc_crust(s,np.array([cell]),np.array([1000.]),
                                positions=centre[None],owners=np.array([0]))
        self.assertEqual(result['new_faces'],0);self.assertEqual(result['grown_patches'],1)
        np.testing.assert_array_equal(ids,s.parcel_patch)
        self.assertAlmostEqual(s.mass.sum(),11000.,places=7)
        positions={int(uid):i for i,uid in enumerate(s.parcel_patch)}
        face=np.array([positions[int(uid)] for uid in s.trace_patch])
        triangles=s.material_surface['vertices'][s.material_surface['faces'][face]]
        weights=np.linalg.solve(triangles.transpose(0,2,1),s.trace_xyz[...,None])[...,0]
        self.assertTrue(np.all(weights>=-1e-8))

    def test_inadmissible_growth_retains_magma_without_partial_geometry_change(self):
        s=ocean_fixture();s.config.update(adaptive_refinement=1,deformation_width_km=80.)
        arcs.add_arc_crust(s,np.array([17]),np.array([100000.]))
        s.material_deformation=dict(face_weight=np.ones(len(s.mass)),face_rigid=np.zeros(len(s.mass),bool),face_strain=np.zeros(len(s.mass)))
        s.t=2.;adaptivity.adapt(s,2.)
        s.structure['thickness_km'][:]=75.;s.trace_structure['thickness_km'][:]=75.
        vertices=s.material_surface['vertices'].copy();mass=s.mass.copy()
        result=arcs.add_arc_crust(s,np.array([17]),np.array([20000.]))
        self.assertEqual(result['added_area_km2'],0.)
        self.assertEqual(result['pending_area_km2'],20000.)
        np.testing.assert_array_equal(vertices,s.material_surface['vertices'])
        np.testing.assert_array_equal(mass,s.mass)

    def test_explicit_refinement_budget_stays_fixed_for_new_arcs(self):
        s=ocean_fixture();s.config.update(adaptive_refinement=2,deformation_width_km=40.,material_face_budget=25)
        arcs.add_arc_crust(s,np.array([17]),np.array([100000.]))
        s.material_deformation=dict(face_weight=np.ones(len(s.mass)),face_rigid=np.zeros(len(s.mass),bool),face_strain=np.zeros(len(s.mass)))
        s.t=2.;adaptivity.adapt(s,2.)
        self.assertEqual(s.adaptivity_diagnostics['face_budget'],25)
        self.assertLessEqual(len(s.mass),25)


if __name__=='__main__':unittest.main()
