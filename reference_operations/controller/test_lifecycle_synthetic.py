"""Bounded stdlib-only contract tests. No simulation import, process launch or live state write."""
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
import types
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('synthetic_controller',HERE/'activate.py')
controller=importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)
CHILD=controller.serve.CHILD
BUDGET_PATH=HERE.parent/'coarse-production-recovery-v1/runtime/budget.py'
budget_spec=importlib.util.spec_from_file_location('synthetic_budget',BUDGET_PATH)
clock_budget=importlib.util.module_from_spec(budget_spec)
sys.modules['synthetic_budget']=clock_budget
budget_spec.loader.exec_module(clock_budget)

def reference(path):
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def write(path,row):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(row),encoding='utf-8')
    return reference(path)

source=(HERE/'budget-overlay/server.py').read_text(encoding='utf-8')
module=ast.parse(source)
manager=next(node for node in module.body if isinstance(node,ast.ClassDef) and node.name=='SimulationManager')
selected=[node for node in manager.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))
    and node.name in ('_guarded_checkpoint','_output_reference')]
namespace=dict(Path=Path,json=json,re=re,hashlib=hashlib,math=math)
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='FixtureManager',bases=[],keywords=[],body=selected,decorator_list=[])],type_ignores=[])),'<contract methods>','exec'),namespace)
FixtureManager=namespace['FixtureManager']

class Fixture:
    def __init__(self,directory):
        self.directory=Path(directory)
        self.public=self.directory/'runs'/CHILD
        self.guardian=self.directory/'guardian'
        self.public.mkdir(parents=True)
        self.guardian.mkdir()
        self.gate=dict(run_id=CHILD,control_sha256='c'*64,child=dict(pid=123,creation_time_100ns=456),
            guardian=dict(pid=789,creation_time_100ns=1011),initial_interval_token='a'*32,
            initial_budget=controller.new_budget(100.))
        write(self.guardian/'launch-gate.json',self.gate)
        self.manager=FixtureManager()
        self.manager.path=lambda run_id=None:self.public
        self.manager._guardian_directory=self.guardian
        self.manager._guardian_control_sha256='c'*64
    def interval(self,token,source,budget,previous=None):
        dest=source+1.
        base=self.public/'numerical-budget'/token
        base.mkdir(parents=True)
        cp=base/'checkpoint.npz'
        cp.write_bytes(('synthetic checkpoint '+str(dest)).encode())
        checkpoint=dict(reference(cp),time_myr=dest)
        frames=[]
        if dest%2==0:
            index=62+int((dest-122.)/2.)
            fields={}
            for extension in ('npz','json'):
                path=self.public/('frame_%04d.%s'%(index,extension))
                path.write_bytes(('synthetic frame '+extension).encode())
                fields[extension]=reference(path)
            frames.append(dict(index=index,time_myr=dest,**fields))
        started=budget['started']
        done=started+50
        completion=dict(kind='lite_budgeted_history_interval',version=2,run_id=CHILD,token=token,
            source_time_myr=source,destination_time_myr=dest,requested_dt_myr=1.,budget=budget,
            interval_and_required_output_completed=True,checkpoint=checkpoint,frames=frames,
            completed_at_perf_counter=done,evidence=[],error=None)
        completion_ref=write(base/'completion.json',completion)
        next_token=chr(ord(token[0])+1)*32
        ack=dict(kind='guardian_accepted_interval',version=1,run_id=CHILD,token=token,budget=budget,
            source_time_myr=source,destination_time_myr=dest,requested_dt_myr=1.,control_sha256='c'*64,
            child=self.gate['child'],guardian=self.gate['guardian'],previous_acceptance_token=previous,
            completion=completion_ref,checkpoint=checkpoint,frames=frames,accepted_at_perf_counter=done+10,
            next_interval_token=next_token,next_budget=controller.new_budget(done+10))
        ack_ref=write(self.guardian/('accepted-interval-'+token+'.json'),ack)
        proof=dict(kind='server_observed_guardian_acceptance',version=1,run_id=CHILD,token=token,
            budget=budget,completion=completion_ref,guardian_acceptance=ack_ref,checkpoint=checkpoint,
            observed_at_perf_counter=done+20)
        proof_ref=write(base/'accepted.json',proof)
        closed=dict(kind='guardian_closed_interval',version=1,run_id=CHILD,token=token,budget=budget,
            source_time_myr=source,destination_time_myr=dest,control_sha256='c'*64,
            guardian_acceptance=ack_ref,server_acceptance=proof_ref,checkpoint=checkpoint,
            closed_at_perf_counter=done+30)
        closure_ref=write(self.guardian/('closed-interval-'+token+'.json'),closed)
        transition=dict(kind='guardian_interval_transition',version=1,run_id=CHILD,token=token,
            control_sha256='c'*64,budget=budget,source_time_myr=source,destination_time_myr=dest,
            closed_interval=closure_ref,transitioned_at_perf_counter=done+31,
            next_interval_token=next_token,next_budget=ack['next_budget'])
        write(self.guardian/('transitioned-interval-'+token+'.json'),transition)
        return dict(ack=ack,closed=closed,proof=proof,completion=completion,checkpoint=cp,transition=transition)
    def selected(self):
        return self.manager._guarded_checkpoint({},CHILD)

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='synthetic-',dir=HERE)
        self.f=Fixture(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def first(self):
        return self.f.interval('a'*32,121.,self.f.gate['initial_budget'])
    def test_no_closed_interval_is_source_boundary(self):
        self.assertIsNone(self.f.selected())
    def test_valid_closed_checkpoint_is_selected(self):
        row=self.first()
        self.assertEqual(self.f.selected(),row['checkpoint'])
    def test_ack_without_closure_cannot_promote(self):
        self.first()
        (self.f.guardian/('closed-interval-'+'a'*32+'.json')).unlink()
        self.assertIsNone(self.f.selected())
    def test_closed_without_server_proof_rejected(self):
        row=self.first()
        row['checkpoint'].with_name('accepted.json').unlink()
        with self.assertRaises(OSError): self.f.selected()
    def test_contiguous_two_interval_chain(self):
        first=self.first()
        second=self.f.interval('b'*32,122.,first['ack']['next_budget'],'a'*32)
        self.assertEqual(self.f.selected(),second['checkpoint'])
    def test_cutoff_of_second_keeps_first(self):
        first=self.first()
        self.f.interval('b'*32,122.,first['ack']['next_budget'],'a'*32)
        write(self.f.guardian/'guardian-completion.json',dict(incomplete_interval=dict(token='b'*32)))
        self.assertEqual(self.f.selected(),first['checkpoint'])
    def test_cutoff_of_first_keeps_source_boundary(self):
        self.first()
        write(self.f.guardian/'guardian-completion.json',dict(incomplete_interval=dict(token='a'*32)))
        self.assertIsNone(self.f.selected())
    def test_token_clock_cannot_be_renewed(self):
        first=self.first()
        self.f.interval('b'*32,122.,controller.new_budget(999.),'a'*32)
        with self.assertRaisesRegex(ValueError,'clock changed'): self.f.selected()
    def test_late_closure_rejected(self):
        row=self.first()
        row['closed']['closed_at_perf_counter']=1900.
        write(self.f.guardian/('closed-interval-'+'a'*32+'.json'),row['closed'])
        row['transition']['closed_interval']=reference(self.f.guardian/('closed-interval-'+'a'*32+'.json'))
        row['transition']['transitioned_at_perf_counter']=1900.
        write(self.f.guardian/('transitioned-interval-'+'a'*32+'.json'),row['transition'])
        with self.assertRaisesRegex(ValueError,'Late guardian transition'): self.f.selected()
    def test_checkpoint_tamper_rejected(self):
        row=self.first()
        row['checkpoint'].write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'checkpoint hash'): self.f.selected()
    def test_frame_tamper_rejected(self):
        self.first()
        (self.f.public/'frame_0062.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'frame hash'): self.f.selected()
    def test_bad_completion_hash_rejected(self):
        row=self.first()
        row['checkpoint'].with_name('completion.json').write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError,'file pin'): self.f.selected()

    def test_closure_without_transition_cannot_promote(self):
        self.first()
        (self.f.guardian/('transitioned-interval-'+'a'*32+'.json')).unlink()
        self.assertIsNone(self.f.selected())
    def test_transition_clock_tamper_rejected(self):
        row=self.first()
        row['transition']['budget']=controller.new_budget(101.)
        write(self.f.guardian/('transitioned-interval-'+'a'*32+'.json'),row['transition'])
        with self.assertRaisesRegex(ValueError,'transition witness differs'): self.f.selected()
    def test_late_transition_rejected(self):
        row=self.first()
        row['transition']['transitioned_at_perf_counter']=1900.
        write(self.f.guardian/('transitioned-interval-'+'a'*32+'.json'),row['transition'])
        with self.assertRaisesRegex(ValueError,'Late guardian transition'): self.f.selected()

class InstalledBudgetTests(unittest.TestCase):
    def test_installed_deadline_covers_nested_retry_and_worker_install(self):
        snapshot=controller.new_budget(100.)
        old_clock=clock_budget._clock
        try:
            clock_budget._clock=lambda:150.
            with clock_budget.installed(snapshot) as outer:
                with clock_budget.step_budget(.125) as retry:
                    self.assertIs(outer,retry)
                    self.assertEqual(clock_budget.snapshot(),snapshot)
                    with clock_budget.installed(snapshot) as worker:
                        self.assertIs(outer,worker)
                        self.assertEqual(clock_budget.snapshot(),snapshot)
                with self.assertRaisesRegex(ValueError,'replace an active'):
                    with clock_budget.installed(controller.new_budget(200.)): pass
                clock_budget._clock=lambda:1900.
                with self.assertRaises(clock_budget.BudgetExhausted):
                    clock_budget.require_finalization_time('synthetic output')
            self.assertIsNone(clock_budget.current())
        finally:
            clock_budget._clock=old_clock
    def test_worker_evidence_cannot_replace_deadline(self):
        snapshot=controller.new_budget(100.)
        with clock_budget.installed(snapshot):
            proof=clock_budget.evidence()
            proof['deadline']+=1.
            with self.assertRaisesRegex(ValueError,'changed interval identity'):
                clock_budget.merge_evidence(proof)

class GuardTests(unittest.TestCase):
    def fake(self):
        self.calls=[]
        k=types.SimpleNamespace(TerminateJobObject=lambda job,code:self.calls.append((job,code)) or 1)
        return types.SimpleNamespace(k=k,check=lambda result,label:None)
    def test_independent_cutoff_while_controller_is_blocked(self):
        api=self.fake()
        clock=controller.new_budget(time.perf_counter())
        clock['deadline']=time.perf_counter()+.05
        guard=controller.DeadlineGuard(api,'owned-job','token',clock)
        guard.thread.start()
        try:
            time.sleep(.15) # Simulates controller file/HTTP wait; deadline thread must still run.
            self.assertTrue(guard.cutoff.is_set())
            self.assertEqual(self.calls,[('owned-job',124)])
            self.assertEqual(guard.receipt['token'],'token')
        finally: guard.close()
    def test_expired_interval_cannot_advance(self):
        api=self.fake()
        clock=controller.new_budget(time.perf_counter()-1801)
        guard=controller.DeadlineGuard(api,'owned-job','token',clock)
        with self.assertRaisesRegex(ValueError,'exceeded deadline'):
            guard.advance('token','next',controller.new_budget(time.perf_counter()))
        self.assertEqual(guard.token,'token')
    def test_wrong_token_cannot_advance(self):
        guard=controller.DeadlineGuard(self.fake(),'owned-job','token',controller.new_budget(time.perf_counter()))
        with self.assertRaisesRegex(ValueError,'interval changed'):
            guard.advance('other','next',controller.new_budget(time.perf_counter()))
    def test_terminal_disarm_requires_current_unexpired_token(self):
        guard=controller.DeadlineGuard(self.fake(),'owned-job','token',controller.new_budget(time.perf_counter()))
        guard.advance('token',None,None)
        self.assertIsNone(guard.snapshot)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    paths=[HERE/'serve.py',HERE/'activate.py',HERE/'budget-overlay/server.py',BUDGET_PATH,Path(__file__)]
    receipt=dict(kind='bounded_synthetic_lifecycle_tests',version=1,tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),
        no_simulation_import=True,no_model_process_launch=True,no_live_state_write=True,
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    (HERE/'synthetic-test-receipt-v2.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
