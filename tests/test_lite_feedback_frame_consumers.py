"""New source-policy metadata must survive native persistence consumers."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
import effective_subduction_carrier_traction as carrier
import mesh_history
import native_frame_sampling

class FeedbackFrameConsumers(unittest.TestCase):
    def test_consumers_reject_removed_bounded_force_policy(self):
        frame=dict(mesh_version=1,force_rifting_policy_version=1)
        for call in (mesh_history.arrays,native_frame_sampling.prepare):
            with self.assertRaisesRegex(ValueError,'bounded|Bounded|force|Force'):
                call(frame)

    def test_legacy_policy_absence_preserves_raster_history(self):
        self.assertEqual(mesh_history.arrays({}), {})
        self.assertFalse(carrier.validate_frame({}))

    def test_consumers_reject_orphan_or_unknown_policy_before_material_decode(self):
        for frame in ({'effective_subduction_carrier_traction_version':1},
                      {'effective_subduction_carrier_traction_version':True},
                      {'effective_subduction_carrier_traction_version':2}):
            with self.subTest(frame=frame):
                with self.assertRaisesRegex(ValueError,'Carrier-traction frame'):
                    mesh_history.arrays(dict(frame,mesh_version=1))
                with self.assertRaisesRegex(ValueError,'Carrier-traction frame'):
                    native_frame_sampling.prepare(dict(frame,mesh_version=1))

    def test_typed_future_only_epoch_roundtrip_and_corrupt_work_rejected(self):
        controls=carrier.normalize(dict(enabled=True))
        state=SimpleNamespace(t=42.123456789,config={'effective_subduction':{'enabled':True,
            'carrier_traction':controls}}, effective_subduction_version=1,backarc_basins=[])
        setattr(state,carrier.FIELD,dict(version=1,enabled=True,activation_myr=42.123456789,
            parameters=controls,retrospective_work_j=0.,law=carrier.POLICY))
        frame=dict(carrier.snapshot(state),effective_subduction_version=1,time_myr=42.123457)
        self.assertTrue(carrier.validate_frame(frame))
        changed=deepcopy(frame)
        changed['effective_subduction_carrier_traction_diagnostics']['retrospective_work_j']=1.
        for call in (mesh_history.arrays,native_frame_sampling.prepare):
            with self.assertRaisesRegex(ValueError,'zero historical work'):
                call(dict(changed,mesh_version=1))

if __name__=='__main__': unittest.main()
