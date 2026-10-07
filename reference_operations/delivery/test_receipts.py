"""Synthetic receipt-only acceptance tests. No model or existing run files."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import receipt_chain as chain
import delivery

HERE=Path(__file__).resolve().parent

def digest(raw):return hashlib.sha256(raw).hexdigest()
def clock(start):return dict(policy=chain.POLICY,dt_myr=1.,started=float(start),deadline=start+1800.,physical_deadline=start+1680.,force_refinement_deadline=start+900.)
def token(n):return '%032x'%n

class Fixture:
    def __init__(self,base):
        self.run=base/'run';self.guardian=base/'guardian';self.run.mkdir();self.guardian.mkdir()
        self.binding=dict(run_directory=str(self.run),guardian_directory=str(self.guardian),control_sha256='a'*64,source_boundary_checkpoint_sha256=digest(b'source121'),target_time_myr=1000.)
        self.launch=dict(run_id=chain.RUN,control_sha256='a'*64,child=dict(pid=101,creation_time_100ns=999),guardian=dict(pid=102,creation_time_100ns=998),initial_interval_token=token(1),initial_budget=clock(100.))
        self.binding['_launch']=self.launch
        (self.run/'checkpoint.npz').write_bytes(b'source121')
    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        raw=json.dumps(value,sort_keys=True,allow_nan=False).encode();path.write_bytes(raw)
        return dict(path=str(path),sha256=digest(raw))
    def blob(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
        return dict(path=str(path),sha256=digest(value))
    def add(self,n):
        src=120+n;dest=src+1;t=token(n);b=clock(100.+(n-1)*10.);directory=self.run/'numerical-budget'/t
        cp=self.blob(directory/'checkpoint.npz',('checkpoint'+str(dest)).encode());cp['time_myr']=dest
        frames=[]
        if dest%2==0:
            index=62+int((dest-122)/2)
            frames=[dict(index=index,time_myr=dest,npz=self.blob(self.run/('frame_%04d.npz'%index),b'frame'),json=self.blob(self.run/('frame_%04d.json'%index),b'frame-json'))]
        c=dict(kind='lite_budgeted_history_interval',version=2,run_id=chain.RUN,token=t,source_time_myr=src,destination_time_myr=dest,requested_dt_myr=1.,budget=b,interval_and_required_output_completed=True,checkpoint=cp,frames=frames,completed_at_perf_counter=b['started']+5,error=None,evidence={})
        cr=self.write(directory/'completion.json',c)
        g=dict(kind='guardian_accepted_interval',version=1,run_id=chain.RUN,token=t,source_time_myr=src,destination_time_myr=dest,requested_dt_myr=1.,control_sha256='a'*64,child=self.launch['child'],guardian=self.launch['guardian'],budget=b,completion=cr,checkpoint=cp,frames=frames,accepted_at_perf_counter=b['started']+10,next_interval_token=token(n+1),next_budget=clock(b['started']+10),previous_acceptance_token=token(n-1) if n>1 else None)
        gr=self.write(self.guardian/('accepted-interval-'+t+'.json'),g)
        a=dict(kind='server_observed_guardian_acceptance',version=1,run_id=chain.RUN,token=t,budget=b,completion=cr,guardian_acceptance=gr,checkpoint=cp,observed_at_perf_counter=b['started']+11)
        ar=self.write(directory/'accepted.json',a)
        z=dict(kind='guardian_closed_interval',version=1,run_id=chain.RUN,token=t,budget=b,source_time_myr=src,destination_time_myr=dest,control_sha256='a'*64,guardian_acceptance=gr,server_acceptance=ar,checkpoint=cp,closed_at_perf_counter=b['started']+12)
        zp=self.guardian/('closed-interval-'+t+'.json');zr=self.write(zp,z)
        tp=self.guardian/('transitioned-interval-'+t+'.json')
        transition=dict(kind='guardian_interval_transition',version=1,run_id=chain.RUN,token=t,control_sha256='a'*64,budget=b,source_time_myr=src,destination_time_myr=dest,closed_interval=zr,transitioned_at_perf_counter=b['started']+13,next_interval_token=g['next_interval_token'],next_budget=g['next_budget'])
        self.write(tp,transition)
        return dict(c=c,g=g,a=a,z=z,cp=directory/'checkpoint.npz',completion=directory/'completion.json',guardian=self.guardian/('accepted-interval-'+t+'.json'),ack=directory/'accepted.json',closed=zp,t=transition,transition=tp)
    def select(self):return chain.select(self.binding,self.launch,lambda p:Path(p).read_bytes(),lambda p:Path(p).is_file())

class ReceiptContract(unittest.TestCase):
    def setUp(self):
        work=HERE/'tests-work';work.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=work);self.f=Fixture(Path(self.tmp.name))
    def tearDown(self):self.tmp.cleanup()
    def test_zero_ack_is_source_boundary(self):
        result=self.f.select();self.assertEqual(result['accepted_checkpoint_myr'],121);self.assertEqual(result['frames'],[])
    def test_partial_ack_or_missing_release_does_not_advance(self):
        x=self.f.add(1);x['closed'].unlink();self.assertEqual(self.f.select()['accepted_checkpoint_myr'],121)
        x['ack'].unlink();self.assertEqual(self.f.select()['accepted_checkpoint_myr'],121)
    def test_two_links_keep_physical_time_and_two_myr_output(self):
        self.f.add(1);self.f.add(2);r=self.f.select()
        self.assertEqual(r['accepted_checkpoint_myr'],123);self.assertEqual([x['time_myr'] for x in r['frames']],[122]);self.assertEqual(len(r['links']),2)
    def test_partial_later_interval_keeps_prior_boundary(self):
        self.f.add(1);x=self.f.add(2);x['closed'].unlink()
        self.assertEqual(self.f.select()['accepted_checkpoint_myr'],122)
    def test_older_guardian_receipt_hash_mutation_rejected(self):
        x=self.f.add(1);self.f.add(2);x['guardian'].write_bytes(x['guardian'].read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'Guardian receipt hash'):self.f.select()
    def test_server_ack_hash_mutation_rejected(self):
        x=self.f.add(1);x['ack'].write_bytes(x['ack'].read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'Server acknowledgment hash'):self.f.select()
    def test_late_final_release_rejected(self):
        x=self.f.add(1);x['z']['closed_at_perf_counter']=x['z']['budget']['deadline'];zr=self.f.write(x['closed'],x['z']);x['t']['closed_interval']=zr;self.f.write(x['transition'],x['t'])
        with self.assertRaisesRegex(ValueError,'Late or reordered'):self.f.select()
    def test_physical_time_gap_rejected(self):
        self.f.add(1);x=self.f.add(2);x['g']['source_time_myr']=123.;self.f.write(x['guardian'],x['g'])
        with self.assertRaisesRegex(ValueError,'Noncontiguous'):self.f.select()
    def test_false_physical_completion_rejected(self):
        x=self.f.add(1);x['c']['interval_and_required_output_completed']=False
        cr=self.f.write(x['completion'],x['c']);x['g']['completion']=cr;gr=self.f.write(x['guardian'],x['g']);x['a'].update(completion=cr,guardian_acceptance=gr);ar=self.f.write(x['ack'],x['a']);x['z'].update(guardian_acceptance=gr,server_acceptance=ar);zr=self.f.write(x['closed'],x['z']);x['t']['closed_interval']=zr;self.f.write(x['transition'],x['t'])
        with self.assertRaisesRegex(ValueError,'Incomplete physical'):self.f.select()
    def test_final_same_token_incomplete_overrides_all_acknowledgments(self):
        self.f.add(1);self.f.add(2)
        self.f.write(self.f.guardian/'guardian-completion.json',dict(kind='burial_successor_owned_interval_guardian',run_id=chain.RUN,control_sha256='a'*64,child=self.f.launch['child'],guardian=self.f.launch['guardian'],incomplete_interval=dict(token=token(2))))
        r=self.f.select();self.assertEqual(r['accepted_checkpoint_myr'],122);self.assertEqual(r['stop_reason'],'same_token_marked_incomplete')
    def test_wrong_guardian_identity_rejected(self):
        x=self.f.add(1);x['g']['guardian']=dict(pid=102,creation_time_100ns=997);self.f.write(x['guardian'],x['g'])
        with self.assertRaisesRegex(ValueError,'ownership'):self.f.select()
    def test_changed_latest_checkpoint_bytes_rejected_by_export(self):
        x=self.f.add(1);x['cp'].write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Latest accepted checkpoint hash'):
            delivery.export(self.f.binding)
    def test_wrong_frame_cadence_rejected(self):
        x=self.f.add(1);x['c']['frames'][0]['index']=63
        cr=self.f.write(x['completion'],x['c']);x['g'].update(completion=cr,frames=x['c']['frames']);gr=self.f.write(x['guardian'],x['g']);x['a'].update(completion=cr,guardian_acceptance=gr);ar=self.f.write(x['ack'],x['a']);x['z'].update(guardian_acceptance=gr,server_acceptance=ar);zr=self.f.write(x['closed'],x['z']);x['t']['closed_interval']=zr;self.f.write(x['transition'],x['t'])
        with self.assertRaisesRegex(ValueError,'cadence/index'):self.f.select()

    def test_missing_guard_transition_is_not_accepted(self):
        x=self.f.add(1);x['transition'].unlink()
        self.assertEqual(self.f.select()['accepted_checkpoint_myr'],121)
    def test_late_guard_transition_is_rejected(self):
        x=self.f.add(1);x['t']['transitioned_at_perf_counter']=x['t']['budget']['deadline'];self.f.write(x['transition'],x['t'])
        with self.assertRaisesRegex(ValueError,'Late or reordered'):self.f.select()
    def test_guard_transition_closure_hash_is_checked(self):
        x=self.f.add(1);x['closed'].write_bytes(x['closed'].read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'Guardian closure hash'):self.f.select()
    def test_pause_transition_does_not_consume_unused_next_clock(self):
        x=self.f.add(1);x['t'].update(next_interval_token=None,next_budget=None);self.f.write(x['transition'],x['t'])
        r=self.f.select();self.assertEqual(r['accepted_checkpoint_myr'],122);self.assertEqual(r['stop_reason'],'closed_without_next_interval')

if __name__=='__main__':unittest.main(verbosity=2)
