"""Small activation/config/metadata checks; no solver or world constructor."""
from pathlib import Path
import hashlib,json,os,sys,time,types,unittest
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import numpy as np
import coarse_history as mode
import coarse_diagnostics
SOURCE=dict(run_id='fixture-parent',checkpoint_file='immutable-fixture.npz',checkpoint_sha256='a'*64)


def state():
    return types.SimpleNamespace(t=121.,steps=130,config=dict(dt_myr=1.,output_every=2.,deforming_regions=1,adaptive_refinement=2),
        collision_surface_version=1,material_mechanics_version=1,
        deformation_diagnostics={'model':'historical fine solver','gravitational_relaxation':{'internal_steps':[{'dt_myr':1.}],'completed_dt_myr':1.}},
        mass=np.array([2.,3.]),omega=np.array([[1.,0.,0.]]),rng={'counter':7},
        structure={'spent':np.array([.1,.2]),'thickness':np.array([30.,40.])})


class Tests(unittest.TestCase):
    def test_config_explicit_and_conflicts(self):
        x={'dt_myr':1.};mode.select_config(x,mode.MODE)
        self.assertEqual(x,{'dt_myr':1.,'deforming_regions':0,'adaptive_refinement':0})
        for bad in (True,'other',1):
            with self.assertRaises(ValueError):mode.normalize(bad)
        with self.assertRaises(ValueError):mode.select_config({'deforming_regions':1},mode.MODE)
        untouched={'deforming_regions':1};mode.select_config(untouched,None)
        self.assertEqual(untouched,{'deforming_regions':1})
    def test_activation_changes_only_declared_state(self):
        s=state();before=dict(vars(s));mass=s.mass.tobytes();spent=s.structure['spent'].tobytes()
        marker=mode.activate(s,source_receipt=SOURCE)
        allowed={'config','deformation_diagnostics','coarse_diagnostic_compaction','coarse_history_policy'}
        for key,value in before.items():
            if key not in allowed:self.assertIs(getattr(s,key),value)
        self.assertEqual(s.mass.tobytes(),mass);self.assertEqual(s.structure['spent'].tobytes(),spent)
        self.assertEqual(s.t,121.);self.assertEqual(s.config['output_every'],2.)
        self.assertFalse(marker['physical_force_laws_changed'])
        self.assertEqual(s.coarse_diagnostic_compaction['physical_time_advanced_myr'],0.)
        self.assertEqual(s.coarse_diagnostic_compaction['detailed_source']['checkpoint_sha256'],'a'*64)
    def test_no_fine_certificate_relabel(self):
        s=state();mode.activate(s,source_receipt=SOURCE)
        frame={'deformation_diagnostics':s.deformation_diagnostics}
        mode.annotate_snapshot(s,frame)
        self.assertEqual(frame['history_mode'],mode.MODE)
        self.assertNotIn('gravitational_relaxation',frame['deformation_diagnostics'])
        frame['deformation_diagnostics']=dict(model='rigid material transport',deforming_vertices=0,gravitational_relaxation={})
        with self.assertRaises(ValueError):mode.annotate_snapshot(s,frame)
    def test_marker_one_shot_and_missing_contract(self):
        s=state();mode.activate(s,source_receipt=SOURCE)
        with self.assertRaises(ValueError):mode.activate(s,source_receipt=SOURCE)
        s.coarse_history_policy=None
        with self.assertRaises(ValueError):mode.validate(s)
    def test_invalid_provenance_leaves_state(self):
        s=state();before=dict(vars(s))
        with self.assertRaises(ValueError):mode.activate(s,source_receipt={'checkpoint_sha256':'wrong'})
        self.assertEqual(set(vars(s)),set(before))
        for key,value in before.items():self.assertIs(getattr(s,key),value)
    def test_mode_gate_cannot_reenable_fine_solver(self):
        s=state();mode.activate(s,source_receipt=SOURCE)
        s.config['deforming_regions']=1
        with self.assertRaises(ValueError):mode.validate(s)
        s.config['deforming_regions']=0;s.t=120.
        with self.assertRaises(ValueError):mode.validate(s)


if __name__=='__main__':
    start=time.perf_counter();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    receipt={'kind':'coarse_history_small_activation_tests','passed':result.wasSuccessful(),'tests_run':result.testsRun,
      'failures':len(result.failures),'errors':len(result.errors),'elapsed_seconds':time.perf_counter()-start,
      'model_constructors':0,'world_steps':0,'workers':0,
      'files':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('coarse_history.py','coarse_diagnostics.py','native_engine.py','test_coarse_history.py')}}
    with (HERE/'activation-tests-v1.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 1)
