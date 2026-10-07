"""Stdlib-only lifecycle checks: fake states, no real world/solver execution."""
import ast,copy,importlib.util,math,sys,threading,types,unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import budget

def extracted(file,name,namespace):
    nodes=[n for n in ast.walk(ast.parse((HERE/file).read_text())) if isinstance(n,ast.FunctionDef) and n.name==name]
    assert len(nodes)==1
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(HERE/file),'exec'),namespace)
    return namespace[name]
class IncompleteContactStepError(RuntimeError):pass
stub=types.ModuleType('deforming_regions');stub.IncompleteContactStepError=IncompleteContactStepError
sys.modules['deforming_regions']=stub
spec=importlib.util.spec_from_file_location('candidate_adaptive_for_budget_test',HERE/'adaptive_timestepping.py')
adaptive=importlib.util.module_from_spec(spec);spec.loader.exec_module(adaptive)
class RuntimeBudgetTests(unittest.TestCase):
    def setUp(self):self.old=budget._clock;self.now=100.;budget._clock=lambda:self.now
    def tearDown(self):budget._clock=self.old;self.assertIsNone(budget.current())
    def test_native_nested_step_keeps_outer_clock(self):
        fn=extracted('native_engine.py','step',{'coarse_history':types.SimpleNamespace(validate=lambda s:None)})
        world=types.SimpleNamespace(config={'dt_myr':1.},_budgeted_step=lambda dt:(dt,budget.snapshot()))
        with budget.step_budget(1.) as state:
            expected=budget.snapshot();self.now+=20;dt,observed=fn(world,.25)
            self.assertEqual(dt,.25);self.assertEqual(observed,expected);self.assertEqual(state.deadline,1900.)
    def test_exhausted_after_accepted_half_restores_entire_world(self):
        world=types.SimpleNamespace(t=121.,values=[]);before=copy.deepcopy(vars(world));calls=[]
        def raw(dt):
            calls.append((dt,budget.current().deadline));world.values.append(dt)
            if dt==1.:raise IncompleteContactStepError('physical rejection')
            world.t+=dt;self.now=1780.
        with budget.step_budget(1.):
            with self.assertRaises(budget.BudgetExhausted):adaptive.advance(world,1.,raw)
            self.assertEqual(vars(world),before);self.assertEqual(calls,[(1.,1900.),(.5,1900.)])
    def test_adaptive_halves_do_not_refill_budget(self):
        world=types.SimpleNamespace(t=121.,values=[]);calls=[]
        def raw(dt):
            calls.append((dt,budget.current().deadline));self.now+=10
            if dt==1.:raise IncompleteContactStepError('split')
            world.t+=dt;world.values.append(dt)
        with budget.step_budget(1.):
            adaptive.advance(world,1.,raw)
            self.assertEqual(world.t,122.);self.assertEqual(world.timestep_diagnostics['accepted_substeps'],2)
            self.assertEqual(calls,[(1.,1900.),(.5,1900.),(.5,1900.)])
    def test_phase_performs_actual_work_then_rejects_overrun(self):
        fn=extracted('native_engine.py','_budget_stage',{});calls=[]
        def work():calls.append('called');self.now=1780.;return 'value'
        with budget.step_budget(1.):
            with self.assertRaises(budget.BudgetExhausted):fn('physical_phase',work)
            self.assertEqual(calls,['called']);self.assertEqual(budget.evidence()['events'][-1]['status'],'physical_budget_exhausted')
    def test_worker_inherits_clock_and_merges_evidence(self):
        fn=extracted('parallel_runtime.py','_budgeted_worker_call',{'budget':budget})
        def work(item):budget.record('worker_fixture',value=item);return (item,budget.snapshot())
        with budget.step_budget(1.):
            original=budget.snapshot()
            with ThreadPoolExecutor(max_workers=1) as executor:value,report=executor.submit(fn,(work,7,original)).result(timeout=2)
            self.assertEqual(value,(7,original));budget.merge_evidence(report);self.assertTrue(budget.evidence()['events'][0]['worker_observation'])
    def test_worker_preserves_physical_exception(self):
        fn=extracted('parallel_runtime.py','_budgeted_worker_call',{'budget':budget})
        def work(item):raise IncompleteContactStepError('physical failure')
        with budget.step_budget(1.):
            with self.assertRaises(IncompleteContactStepError) as caught:fn((work,0,budget.snapshot()))
            self.assertEqual(caught.exception.runtime_budget_evidence['deadline'],1900.)
    def server_fixture(self,fail=False):
        sim=types.SimpleNamespace(t=121.)
        def step(dt):
            self.assertIsNotNone(budget.current());self.now+=10
            if fail:raise budget.BudgetExhausted('fixture deadline')
            sim.t+=dt
        sim.step=step
        class Flag:
            def is_set(self):return False
        class Timing:
            def update(self,t):return {}
            def finish(self):return {}
            def pause(self):return {}
        m=types.SimpleNamespace(lock=threading.RLock(),stop=Flag(),pause_requested=Flag(),current={'frame_count':1,'frames':[{'time_myr':120.}], 'time_myr':120.},timing=Timing(),recovery_interval_myr=1.,_resume_checkpoint_name='checkpoint.npz',worker=None)
        m._open_numerical_budget=lambda source,dt:None
        m.path=lambda:Path('.');m.persist=lambda:None;m.frame=lambda i:{'index':i};m.snapshots=[];m.final=[];m.reports=[]
        def snapshot(world,*a,**k):
            self.assertIsNotNone(budget.current());m.snapshots.append(world.t);m.current['frame_count']+=1;m.current['time_myr']=world.t
        m._append_snapshot=snapshot
        def final(frame):self.assertIsNotNone(budget.current());m.final.append(frame)
        m.save_final=final;m._commit_checkpoint=lambda *a,**k:None
        m._record_numerical_budget=lambda start,dt,done,error:m.reports.append((start,dt,done,error,budget.evidence()))
        fn=extracted('server.py','_run_owned',{'budget':budget,'math':math,'Simulation':object,'read_checkpoint':lambda *a:(sim,{'next_output_myr':122.})})
        m.compatibility=lambda:{}
        return fn,m
    def test_server_budget_contains_step_snapshot_and_final_output(self):
        fn,m=self.server_fixture();fn(m,{'duration_myr':122.,'snapshot_myr':2.,'dt_myr':1.},None,True)
        self.assertEqual(m.snapshots,[122.]);self.assertEqual(len(m.final),1);self.assertEqual(m.current['state'],'complete')
        self.assertTrue(m.reports[0][2]);self.assertEqual(m.reports[0][4]['deadline'],1900.)
    def test_server_records_exhaustion_without_fake_public_time(self):
        fn,m=self.server_fixture(True);fn(m,{'duration_myr':122.,'snapshot_myr':2.,'dt_myr':1.},None,True)
        self.assertEqual(m.current['state'],'error');self.assertEqual(m.current['time_myr'],120.);self.assertEqual(m.snapshots,[])
        self.assertFalse(m.reports[0][2]);self.assertEqual(m.reports[0][3]['type'],'BudgetExhausted')
if __name__=='__main__':unittest.main()
