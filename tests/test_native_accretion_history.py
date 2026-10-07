"""The local welding policy is versioned, captured and carried in native history."""
from copy import deepcopy
from pathlib import Path
import unittest
import numpy as np
import localized_accretion as welding
import mesh_history
import native_frame_sampling
from native_engine import Simulation
from orientation import orient_frame
from server import capture_auxiliary_sources, make_initial


class NativeAccretionHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config=dict(width=48,height=24,coast_geometry_level=2,mesh_level=2,
                    plate_count=4,seed=37,mechanics_nodes=128,adaptive_refinement=0)
        cls.simulation=Simulation(config,make_initial(config))
        cls.frame=cls.simulation.snapshot()

    def test_new_simulation_records_empty_policy_and_source_closure(self):
        self.assertEqual(self.simulation.native_accretion_version,1)
        self.assertEqual(self.frame['native_accretion_version'],1)
        self.assertEqual(self.frame['native_accretion_welding_time_myr'],0.)
        arrays=mesh_history.arrays(self.frame)
        for name in welding.ARRAY_FIELDS:
            self.assertIn(name,arrays)
            self.assertEqual(arrays[name].shape,(0,))
        root=Path(__file__).resolve().parents[1]
        sources=capture_auxiliary_sources((root/'tectonics.py').read_bytes(),root)
        self.assertEqual(sources['localized_accretion.py'],(root/'localized_accretion.py').read_bytes())
        self.assertEqual(sources['local_accretion.py'],(root/'local_accretion.py').read_bytes())

    def test_policy_arrays_remain_independent_of_material_face_count_and_orientation(self):
        frame=deepcopy(self.frame)
        # Two zero-dose directed records are legal initialization state; they
        # intentionally do not have one entry for each material face.
        values=dict(source_uid=np.array([2,3],np.int64),target_uid=np.array([3,2],np.int64),
                    root_id=np.array([11,12],np.int64),loading=np.zeros(2),active_weight=np.zeros(2))
        for name,value in values.items():frame['accretion_welding_'+name]=value
        frame['native_accretion_diagnostics']['directed_root_records']=2
        arrays=mesh_history.arrays(frame)
        rotated=orient_frame(frame,{'yaw':12.,'pitch':25.,'roll':-8.})
        native_frame_sampling.prepare(rotated)
        for name in welding.ARRAY_FIELDS:
            np.testing.assert_array_equal(arrays[name],frame[name])
            np.testing.assert_array_equal(rotated[name],frame[name])

    def test_history_and_native_sampler_reject_stripped_version_and_bad_alignment(self):
        for mode in ('version','alignment','nonfinite','diagnostic_count','diagnostic_width'):
            frame=deepcopy(self.frame)
            if mode=='version':del frame['native_accretion_version']
            elif mode=='alignment':frame['accretion_welding_root_id']=np.array([7],np.int64)
            elif mode=='nonfinite':frame['accretion_welding_loading']=np.array([np.nan])
            elif mode=='diagnostic_count':frame['native_accretion_diagnostics']['directed_root_records']=1
            else:frame['native_accretion_diagnostics']['process_zone_width_km']=0.
            for validate in (mesh_history.arrays,native_frame_sampling.prepare):
                with self.subTest(mode=mode,consumer=validate.__module__):
                    with self.assertRaises(ValueError):validate(frame)

    def test_genuine_unmarked_frame_retains_legacy_semantics(self):
        frame=deepcopy(self.frame)
        for name in (*welding.ARRAY_FIELDS,'native_accretion_version','native_accretion_welding_time_myr','native_accretion_diagnostics'):
            frame.pop(name,None)
        arrays=mesh_history.arrays(frame)
        self.assertFalse(set(welding.ARRAY_FIELDS).intersection(arrays))
        native_frame_sampling.prepare(frame)


if __name__=='__main__':unittest.main()
