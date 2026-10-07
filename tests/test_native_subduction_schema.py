"""New capture evidence cannot be downgraded or loaded with invented budgets."""
from copy import deepcopy
import unittest
import numpy as np
import native_subduction
import native_frame_sampling
import mesh_history
from tests.test_native_subduction_oracle import fixture


class NativeSubductionSchemaTests(unittest.TestCase):
    def frame(self):
        s,_,transported=fixture();s.t=0.
        native_subduction.removal(s,transported,2.);s.t=2.
        return dict(native_subduction.snapshot_metadata(s),mesh_version=1,time_myr=2.,mesh_owner_slots=np.array([0,1,2]))

    def test_valid_measured_and_explicit_initial_records(self):
        frame=self.frame();native_subduction.validate_frame(frame)
        s,_,_=fixture();s.t=0.
        initial=native_subduction.snapshot_metadata(s)
        native_subduction.validate_frame(dict(initial,time_myr=0.))
        self.assertEqual(initial['native_subduction_diagnostics']['state'],'initialized')
        s.t=2.
        with self.assertRaisesRegex(ValueError,'lacks its measured'):native_subduction.snapshot_metadata(s)

    def test_unversioned_legacy_allowed_but_retained_new_record_requires_version(self):
        native_subduction.validate_frame({})
        for value in (None,0,True,1.,2):
            frame=self.frame()
            if value is None:frame.pop('native_subduction_version')
            else:frame['native_subduction_version']=value
            with self.subTest(value=value),self.assertRaises(ValueError):native_subduction.validate_frame(frame)

    def test_malformed_owner_time_count_and_nonfinite_fields_fail_through_both_loaders(self):
        mutations=[lambda d:d.update(owner_slots=[0,0,2]),
                   lambda d:d.update(owner_slots=[0.,1.,2.]),
                   lambda d:d.update(removed_area_by_owner_km2=[1.]),
                   lambda d:d.update(removed_area_by_owner_km2=[1.,-1.,0.]),
                   lambda d:d.update(removed_area_km2=float('nan')),
                   lambda d:d.update(step_duration_myr=-2.),
                   lambda d:d.update(step_end_myr=3.),
                   lambda d:d.update(capture_polygons=.5),
                   lambda d:d.update(capture_polygons=0),
                   lambda d:d.update(rejected_unrelated_owner_water_area_km2=1e10),
                   lambda d:d.update(center_crust_gate_used=True),
                   lambda d:d.update(maximum_support_bound_residual=.01)]
        for mutate in mutations:
            frame=self.frame();mutate(frame['native_subduction_diagnostics'])
            for loader in (native_subduction.validate_frame,native_frame_sampling.prepare,mesh_history.arrays):
                with self.subTest(mutate=mutate,loader=loader.__name__),self.assertRaises(ValueError):loader(frame)

    def test_tampered_area_sum_or_exceeded_physical_bound_rejected(self):
        for key in ('removed_area_km2','duplicate_capture_area_removed_km2','unsupported_or_unavailable_capture_area_km2'):
            frame=self.frame();frame['native_subduction_diagnostics'][key]+=100.
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'components disagree'):native_subduction.validate_frame(frame)
        frame=self.frame();frame['native_subduction_diagnostics']['available_total_overlap_km2']=0.
        with self.assertRaisesRegex(ValueError,'exceeded'):native_subduction.validate_frame(frame)

    def test_wrong_epoch_or_missing_current_native_owner_rejected(self):
        frame=self.frame();frame['time_myr']=4.
        with self.assertRaisesRegex(ValueError,'saved epoch'):native_subduction.validate_frame(frame)
        frame=self.frame();frame['mesh_owner_slots']=np.array([0,1,4])
        with self.assertRaisesRegex(ValueError,'omit'):native_subduction.validate_frame(frame)

    def test_post_transport_slot_growth_has_explicit_zero_removal_rows(self):
        s,_,transported=fixture();s.t=0.
        native_subduction.removal(s,transported,2.);s.t=2.
        original=deepcopy(s.native_subduction_diagnostics)
        s.support=np.vstack((s.support,np.zeros((2,s.n))))
        frame=native_subduction.snapshot_metadata(s)
        d=frame['native_subduction_diagnostics']
        self.assertEqual(d['owner_slots'],[0,1,2,3,4])
        self.assertEqual(d['removed_area_by_owner_km2'][-2:],[0.,0.])
        self.assertEqual(s.native_subduction_diagnostics,original)
        native_subduction.validate_frame(dict(frame,time_myr=2.,mesh_owner_slots=np.arange(5)))


if __name__=='__main__':unittest.main()
