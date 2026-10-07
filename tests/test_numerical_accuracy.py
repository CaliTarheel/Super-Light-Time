"""Isolated dimensional admission, volume and future-frame schema regressions."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np
import numerical_accuracy as accuracy
import crustal_structure as columns
import structure_engine
import adaptive_material
import viscous_sheet
import mesh_history
import gravity_constraints_schema
import dense_crust


def column(area=95.,thickness=75.):
    state=columns.initialize_structure(np.array([1]),np.array([220.]))
    state['thickness_km'][:]=thickness
    accuracy.initialize_columns(state,np.array([area]))
    return state


def policy(state):
    area=state[accuracy.AREA]
    return accuracy.NumericalPolicy(state[accuracy.REFERENCE],state[accuracy.BUDGET],state[accuracy.SPENT],area,
        area*state['thickness_km']/75.,area*columns.restored_thickness(state)/8.,'isolated persistent material identity')


def move(state,area):
    p=policy(state).advance(np.array([area]))
    return columns.evolve_structure(state,1.,geometric_log_area=np.log(np.array([area])/state[accuracy.AREA]),
        geometric_accuracy=accuracy.endpoint_columns(state,np.array([area]),p.spent_km2))[0]


def synthetic_model():
    state=columns.initialize_structure(np.array([1]),np.array([220.]))
    return SimpleNamespace(t=105.,structure=state,trace_structure=deepcopy(state),
        material_surface={'area_km2':np.array([95.])},mass=np.array([95.]),
        parcel_patch=np.array([10],np.int64),trace_patch=np.array([10],np.int64))


def frame(model):
    optional=(set(mesh_history.OWNER_DTYPES)|set(mesh_history.TRANSPORT_DTYPES)|set(mesh_history.DEFORMATION_DTYPES)
        |set(mesh_history.ARC_DTYPES)|set(mesh_history.collision_contacts.ARRAY_FIELDS)|set(mesh_history.collision_surface.ARRAY_FIELDS)
        |set(mesh_history.COAST_DTYPES)|set(mesh_history.localized_accretion.ARRAY_FIELDS)|set(mesh_history.lip_events.ARRAY_FIELDS)
        |set(mesh_history.ENHANCED_RIFT_DTYPES)|set(mesh_history.eclogite_sink.ARRAY_FIELDS)|set(accuracy.ARRAY_FIELDS))
    result={name:np.zeros(1,dtype=dtype) for name,dtype in mesh_history.DTYPES.items() if name not in optional}
    vertices=np.eye(3);faces=np.array([[0,1,2]],np.int32)
    for prefix in ('mesh','material'):
        result[prefix+'_vertices']=vertices.copy();result[prefix+'_faces']=faces.copy()
    for name in tuple(result):
        if name.startswith('native_boundary_'):result[name]=np.zeros((0,2) if name.endswith('edges') else (0,),dtype=mesh_history.DTYPES[name])
    result.update(mesh_version=1,material_transport_version=1,gravity_constraint_version=0,
        mesh_area_km2=np.array([95.]),material_reference_area_km2=np.array([95.]),material_kind=np.array([1],np.uint8),
        material_crustal_thickness_km=model.structure['thickness_km'].copy(),
        material_root_id=np.array([10],np.int64),material_parent_id=np.array([-1],np.int64),
        material_reference_corners=np.eye(3)[None],material_erosion_total_m=np.zeros(1),
        material_actual_area_km2=model.material_surface['area_km2'].copy(),
        material_refinement_level=np.zeros(1,np.int32),material_deformation_weight=np.zeros(1),
        material_rigid=np.ones(1,np.uint8),material_geometric_log_area=np.zeros(1))
    result.update(accuracy.snapshot(model))
    return result


class NumericalAccuracyTests(unittest.TestCase):
    def test_exact_volume_no_thickness_clipping(self):
        old=column();new=move(old,93.)
        self.assertGreater(new['thickness_km'][0],75.)
        self.assertEqual(new['thickness_km'][0],95.*75./93.)
        self.assertEqual(new[accuracy.AREA][0]*new['thickness_km'][0],95.*75.)
        self.assertAlmostEqual(new['area_factor'][0]*new['thickness_km'][0],75.,places=13)
        columns.elevation(new)

    def test_cumulative_budget_cannot_reset_at_current_area(self):
        first=move(column(),94.);second=move(first,93.)
        self.assertEqual(second[accuracy.SPENT][0],2.)
        self.assertEqual(policy(second).minimum_area_km2[0],95.)
        with self.assertRaises(ValueError):move(second,92.)

    def test_return_to_nominal_does_not_refund_credit(self):
        spent=move(column(),93.);returned=move(spent,95.)
        self.assertEqual(returned[accuracy.SPENT][0],2.)
        with self.assertRaises(ValueError):move(returned,94.)

    def test_small_face_has_proportional_not_absolute_credit(self):
        old=column(9.5);self.assertAlmostEqual(old[accuracy.BUDGET][0],.2)
        self.assertTrue(policy(old).feasible(np.array([9.4]),np.array([9.5]),np.array([89.0625])))
        with self.assertRaises(ValueError):move(old,9.)

    def test_cumulative_evidence_is_bound_to_actual_credit(self):
        p=policy(column()).advance(np.array([94.]))
        evidence=p.evidence(np.array([93.]),p.minimum_area_km2,p.maximum_area_km2)
        accuracy.validate_evidence(evidence)
        self.assertEqual(evidence['cumulative_spent_km2'],[2.])
        self.assertEqual(evidence['maximum_budget_fraction'],1.)
        bad=deepcopy(evidence);bad['total_cumulative_spent_km2']=3.
        with self.assertRaises(ValueError):accuracy.validate_evidence(bad)

    def test_policy_is_immutable_and_inputs_untouched(self):
        state=column();before=deepcopy(state);p=policy(state)
        with self.assertRaises(ValueError):p.reference_area_km2.setflags(write=True)
        p.evidence(np.array([94.]),p.minimum_area_km2,p.maximum_area_km2)
        for name in state:np.testing.assert_array_equal(state[name],before[name])

    def test_contact_bridge_retains_serial_spending_not_only_final_miss(self):
        p=policy(column())
        result={'diagnostics':{'numerical_accuracy_final_spent_km2':[1.8]}}
        accepted=accuracy.accepted_contact_policy(p,result,np.array([94.]))
        self.assertEqual(accepted.spent_km2[0],1.8)
        self.assertAlmostEqual(accepted.allowance_km2[0],1.2)
        with self.assertRaises(ValueError):accuracy.accepted_contact_policy(p,{},np.array([94.]))
        with self.assertRaises(ValueError):accuracy.accepted_contact_policy(p,
            {'diagnostics':{'numerical_accuracy_final_spent_km2':[.5]}},np.array([94.]))

    def test_refinement_transfers_credit_and_spending_extensively(self):
        old=move(column(190.),189.)
        mapping=dict(source_indptr=np.array([0,1,2]),source_indices=np.array([0,0]),source_area_fractions=np.array([.5,.5]))
        new={}
        self.assertTrue(accuracy.remap_columns(old,new,mapping))
        np.testing.assert_array_equal(new[accuracy.BUDGET],[1.,1.])
        np.testing.assert_array_equal(new[accuracy.SPENT],[.5,.5])
        self.assertEqual(new[accuracy.BUDGET].sum(),old[accuracy.BUDGET].sum())

    def test_incompatible_merge_cannot_renew_large_face_credit(self):
        old={name:np.array([value[0],value[0]]) for name,value in column().items()}
        mapping=dict(source_indptr=np.array([0,2]),source_indices=np.array([0,1]),source_area_fractions=np.array([1.,1.]))
        self.assertFalse(accuracy.remap_columns(old,{},mapping))

    def test_reader_rejects_missing_forged_or_exhausted_metadata(self):
        state=move(column(),93.)
        for change in ('missing','budget','area'):
            bad=deepcopy(state)
            if change=='missing':bad.pop(accuracy.SPENT)
            elif change=='budget':bad[accuracy.BUDGET][:]=3.
            else:bad[accuracy.AREA][:]=200.
            with self.assertRaises(ValueError):columns.elevation(bad)

    def test_zero_source_does_not_clip_an_admitted_column(self):
        state=move(column(),93.);old=deepcopy(state)
        gain,added=structure_engine._surface_change(state,np.zeros(1),conserve_volume=False)
        np.testing.assert_array_equal(state['thickness_km'],old['thickness_km'])
        np.testing.assert_array_equal(gain,[0.]);np.testing.assert_array_equal(added,[0.])
        gain,added=structure_engine._surface_change(state,np.ones(1)*1000.,conserve_volume=False)
        np.testing.assert_array_equal(gain,[0.]);np.testing.assert_array_equal(added,[0.])

    def test_erosion_removes_exact_volume_without_spending_new_area(self):
        state=column(95.,35.);p=policy(state).advance(np.array([100.]))
        new,budget=columns.evolve_structure(state,1.,geometric_log_area=np.log(np.array([100.])/95.),
            geometric_accuracy=accuracy.endpoint_columns(state,[100.],p.spent_km2),denudation_m=100.)
        self.assertAlmostEqual(100.*new['thickness_km'][0]+95.*budget['removed_volume_km_per_reference_km2'][0],95.*35.,places=10)
        self.assertEqual(new[accuracy.SPENT][0],0.)

    def test_migration_adds_only_declared_arrays_and_is_one_time(self):
        model=synthetic_model();before=deepcopy(vars(model))
        receipt=accuracy.migrate(model,boundary_myr=105.,parent_checkpoint_sha256='a'*64,authorization_sha256='b'*64)
        self.assertEqual(receipt['fields'],list(accuracy.FIELDS))
        for field in ('structure','trace_structure'):
            self.assertEqual(set(getattr(model,field))-set(before[field]),set(accuracy.FIELDS))
            for name,value in before[field].items():np.testing.assert_array_equal(getattr(model,field)[name],value)
        result=accuracy.validate_simulation(model)
        self.assertEqual(result['cumulative_spent_km2'],0.)
        with self.assertRaises(ValueError):accuracy.migrate(model,boundary_myr=105.,parent_checkpoint_sha256='a'*64,authorization_sha256='b'*64)

    def test_inherited_reference_roundoff_uses_original_aggregate_gate(self):
        model=synthetic_model()
        model.structure=columns.initialize_structure(np.ones(2,np.uint8),np.ones(2)*220.)
        model.structure['thickness_km'][:]=[8.,35.]
        actual=np.array([15.666316636333217,1e6]);mass=np.array([5.17021503616207,1e6])
        model.structure['area_factor'][:]=[15.666316592613887/mass[0],1.]
        model.trace_structure=deepcopy(model.structure);model.mass=mass
        model.material_surface={'area_km2':actual};model.parcel_patch=np.array([10,11]);model.trace_patch=np.array([10,11])
        before=deepcopy(vars(model))
        accuracy.migrate(model,boundary_myr=105.,parent_checkpoint_sha256='a'*64,authorization_sha256='b'*64)
        diagnostic=accuracy.validate_simulation(model)
        self.assertGreater(diagnostic['maximum_physical_reference_volume_relative_difference'],2e-9)
        self.assertLess(abs(diagnostic['physical_reference_volume_relative_residual']),1e-12)
        for name,value in before['structure'].items():np.testing.assert_array_equal(model.structure[name],value)
        model.structure['area_factor'][1]*=1.00001
        with self.assertRaises(ValueError):accuracy.validate_simulation(model)

    def test_mechanical_transaction_preserves_inherited_reference_offset(self):
        old=column();old['area_factor'][:]=.999999997
        before=deepcopy(old);new=move(old,93.)
        self.assertEqual(new[accuracy.AREA][0]*new['thickness_km'][0],95.*75.)
        np.testing.assert_allclose(new['area_factor']*new['thickness_km'],
            old['area_factor']*old['thickness_km'],rtol=2e-15,atol=0.)
        for name,value in before.items():np.testing.assert_array_equal(old[name],value)

    def test_opposite_future_volume_errors_cannot_cancel(self):
        old=dict(thickness_km=np.array([35.,35.]),area_factor=np.ones(2))
        area=np.array([95.,95.]);new=deepcopy(old)
        new['thickness_km']+=np.array([.1,-.1])
        self.assertEqual(float(area@new['thickness_km']),float(area@old['thickness_km']))
        with self.assertRaises(ValueError):columns._validate_exact_volume_transaction(old,new,area,area)
        new=deepcopy(old);new['area_factor']+=np.array([.1,-.1])
        self.assertEqual(float(new['area_factor']@new['thickness_km']),float(old['area_factor']@old['thickness_km']))
        with self.assertRaises(ValueError):columns._validate_exact_volume_transaction(old,new,area,area)

    def test_actual_native_frame_array_writer_preserves_new_fields(self):
        model=synthetic_model();accuracy.migrate(model,boundary_myr=105.,parent_checkpoint_sha256='a'*64,authorization_sha256='b'*64)
        saved=frame(model);arrays=mesh_history.arrays(saved)
        for name in accuracy.ARRAY_FIELDS:
            self.assertEqual(arrays[name].dtype,np.dtype('float64'))
            self.assertEqual(arrays[name].tobytes(),np.asarray(saved[name],np.float64).tobytes())
        legacy={name:value for name,value in saved.items() if name not in accuracy.ARRAY_FIELDS
            and name not in ('numerical_accuracy_version','numerical_accuracy_migration','numerical_accuracy_diagnostics')}
        mesh_history.arrays(legacy)
        bad=deepcopy(saved);bad['material_area_accuracy_budget_km2'][:]=3.
        with self.assertRaises(ValueError):mesh_history.arrays(bad)

    def test_saved_restored_column_is_bound_to_retained_phase(self):
        model=synthetic_model();accuracy.migrate(model,boundary_myr=105.,parent_checkpoint_sha256='a'*64,authorization_sha256='b'*64)
        saved=frame(model);saved['retained_dense_crust_version']=1
        phase={name:np.zeros(1) for name in dense_crust.FIELDS}
        phase.update(thickness_km=model.structure['thickness_km'].copy(),area_factor=np.ones(1))
        phase[dense_crust.DENSE][:]=.1
        saved.update({name:phase[field].copy() for name,field in dense_crust.FRAME_FIELDS.items()})
        saved[accuracy.RESTORED_FRAME_FIELD]=dense_crust.restored_thickness(phase)
        accuracy.validate_frame(saved)
        forged=deepcopy(saved);forged[accuracy.RESTORED_FRAME_FIELD][:]+=1.
        with self.assertRaises(ValueError):accuracy.validate_frame(forged)
        forged=deepcopy(saved);forged[accuracy.RESTORED_FRAME_FIELD][:]=0.
        with self.assertRaises(ValueError):accuracy.validate_frame(forged)

    def test_new_marker_does_not_relax_legacy_stationarity(self):
        state=column();p=policy(state)
        row=dict(dt_myr=1.,solver=dict(converged=True,stationarity_recomputed=True,true_stationarity_norm=5e-7,
            rhs_norm=1.,tolerance=1e-6,relative_residual=5e-7,iterations=1,maximum_iterations=2,stationarity_kind='unconstrained_linear'))
        with self.assertRaises(ValueError):gravity_constraints_schema.validate_accepted_row(row)
        row['numerical_accuracy_policy']=p.evidence([95.],p.minimum_area_km2,p.maximum_area_km2,include_arrays=False)
        gravity_constraints_schema.validate_accepted_row(row)
        row['solver']['relative_residual']=2e-6
        with self.assertRaises(ValueError):gravity_constraints_schema.validate_accepted_row(row)

    def test_active_sheet_tiny_norm_failure_is_uncertified(self):
        points=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
        faces=np.array([[0,1,2]],np.int64);area=np.array([np.pi/2]);empty=np.zeros(3,bool)
        pattern=np.array([[1.,2.,0.],[0.,1.,-1.],[2.,0.,.5]])
        for magnitude in (1e-200,1e200):
            with np.errstate(all='ignore'):
                velocity,info=viscous_sheet.solve(points,faces,area,np.zeros_like(points),empty,empty,0.,radius=1.,
                    body_force=pattern*magnitude,tolerance=accuracy.OUTER_RELATIVE_TOLERANCE,iterations=8)
            self.assertFalse(info['converged']);self.assertIsNone(info['relative_residual'])
            self.assertEqual(info['failure_reason'],'arithmetic_rhs_norm_range_failure')
            np.testing.assert_array_equal(velocity,np.zeros_like(points))


if __name__=='__main__':unittest.main()
