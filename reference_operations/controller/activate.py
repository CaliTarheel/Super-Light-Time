"""One-attempt Windows Job controller with a watchdog independent of file/HTTP reads."""
import argparse
import ctypes
from ctypes import wintypes as W
import json
import math
import os
from pathlib import Path
import secrets
import socket
import threading
import time
import traceback
import uuid
from urllib.request import Request, urlopen
import serve

HERE, ROOT, r = serve.HERE, serve.ROOT, serve.r
OLD = HERE.parent/'coarse-production-recovery-v1'

def new_budget(start, dt=1.):
    return dict(policy='budgeted_history_1800s_per_myr_v1', dt_myr=dt, started=start,
                deadline=start+1800.*dt, physical_deadline=start+1680.*dt,
                force_refinement_deadline=start+900.*dt)

class DeadlineGuard:
    """Only this thread performs deadline cutoff; it never reads disk, Watch or HTTP."""
    def __init__(self, api, job, token, snapshot):
        self.api, self.job = api, job
        self.lock = threading.Lock()
        self.token, self.snapshot = token, dict(snapshot)
        self.done = threading.Event()
        self.cutoff = threading.Event()
        self.receipt = None
        self.thread = threading.Thread(target=self.run, name='owned-interval-deadline', daemon=True)
    def run(self):
        while not self.done.wait(.02):
            with self.lock:
                state = self.snapshot
                if state is not None and time.perf_counter() >= state['deadline']:
                    self.receipt = dict(token=self.token, budget=dict(state), observed_at_perf_counter=time.perf_counter())
                    try:
                        self.api.check(self.api.k.TerminateJobObject(self.job,124),'Owned Job deadline cutoff')
                    except BaseException as error:
                        self.receipt['cutoff_error'] = str(error)
                    self.cutoff.set()
                    return
    def advance(self, old_token, new_token, snapshot):
        with self.lock:
            serve.require(not self.cutoff.is_set() and self.token == old_token, 'Guard interval changed')
            transitioned = time.perf_counter()
            serve.require(self.snapshot is not None and transitioned < self.snapshot['deadline'], 'Completion exceeded deadline')
            self.token, self.snapshot = new_token, (None if snapshot is None else dict(snapshot))
            return transitioned
    def close(self):
        self.done.set()
        self.thread.join(timeout=1.)

def verify_old_absence(api, old):
    api._bind('OpenProcess',[W.DWORD,W.BOOL,W.DWORD],W.HANDLE)
    for identity in (old['child'],old['guardian']):
        handle = api.k.OpenProcess(0x1000,False,identity['pid'])
        if handle:
            try:
                serve.require(api.identity(handle,identity['pid']) != identity, 'Exact old process is still alive')
            finally:
                api.k.CloseHandle(handle)
        else:
            serve.require(ctypes.get_last_error() == 87, 'Cannot verify old process absence')
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
        listener.bind(('127.0.0.1',8771))

def job_members(api, job):
    class ProcessList(ctypes.Structure):
        _fields_ = [('assigned',W.DWORD),('listed',W.DWORD),('pids',ctypes.c_size_t*64)]
    rows = ProcessList()
    api.check(api.k.QueryInformationJobObject(job,3,ctypes.byref(rows),ctypes.sizeof(rows),None), 'Owned Job members')
    serve.require(rows.assigned == rows.listed and rows.listed <= 64, 'Job process list truncated')
    result = []
    for pid in rows.pids[:rows.listed]:
        handle = api.k.OpenProcess(0x1000,False,int(pid))
        if not handle:
            # A worker can exit between the Job list and OpenProcess.
            serve.require(ctypes.get_last_error() == 87,'Cannot inspect owned worker')
            continue
        try:
            member = W.BOOL()
            api.check(api.k.IsProcessInJob(handle,job,ctypes.byref(member)),'Verify exact owned Job membership')
            serve.require(member.value,'Worker escaped owned Job')
            result.append(api.identity(handle,pid))
        finally:
            api.k.CloseHandle(handle)
    return sorted(result,key=lambda item:item['pid'])

def validate_completion(marker, expected, public, control_pin):
    serve.require(marker['token'] == expected['token'] and marker['budget'] == expected['budget'], 'Interval token/clock mutated')
    serve.require(marker['source_time_myr'] == expected['source'] and marker['requested_dt_myr'] == expected['budget']['dt_myr'], 'Interval source/dt changed')
    ref = marker['completion']
    path = public/'numerical-budget'/expected['token']/'completion.json'
    serve.require(Path(ref['path']).resolve() == path.resolve() and r.sha(path) == ref['sha256'], 'Completion reference differs')
    report = r.json_read(path)
    serve.require(report.get('kind') == 'lite_budgeted_history_interval' and report.get('version') == 2
        and report.get('run_id') == serve.CHILD and report.get('token') == expected['token']
        and report.get('budget') == expected['budget'] and report.get('source_time_myr') == expected['source']
        and report.get('requested_dt_myr') == expected['budget']['dt_myr']
        and report.get('destination_time_myr') == expected['source']+expected['budget']['dt_myr']
        and report.get('interval_and_required_output_completed') is True, 'Incomplete or different interval')
    finished = report['completed_at_perf_counter']
    serve.require(type(finished) in (int,float) and math.isfinite(finished)
        and expected['budget']['started'] <= finished < expected['budget']['deadline'], 'Late completion')
    checkpoint = report['checkpoint']
    serve.require(Path(checkpoint['path']).resolve() == path.with_name('checkpoint.npz').resolve()
        and checkpoint['time_myr'] == report['destination_time_myr']
        and r.sha(checkpoint['path']) == checkpoint['sha256'], 'Immutable checkpoint differs')
    destination = report['destination_time_myr']
    due = destination % 2 == 0
    frames = report['frames']
    serve.require(len(frames) == int(due), 'Wrong due frame count')
    if due:
        row = frames[0]
        expected_index = 62+int((destination-122.)/2.)
        serve.require(row['index'] == expected_index and row['time_myr'] == destination, 'Wrong cadence frame')
        for extension in ('npz','json'):
            item = row[extension]
            required = public/('frame_%04d.%s'%(expected_index,extension))
            serve.require(Path(item['path']).resolve() == required.resolve() and r.sha(required) == item['sha256'], 'Due output differs')
    serve.require(time.perf_counter() < expected['budget']['deadline'], 'Independent output verification exceeded deadline')
    return report

def validate_server_closure(marker, accepted, public):
    token = accepted['token']
    path = public/'numerical-budget'/token/'accepted.json'
    reference = marker['server_acceptance']
    serve.require(Path(reference['path']).resolve() == path.resolve()
        and r.sha(path) == reference['sha256'], 'Server closure reference differs')
    proof = r.json_read(path)
    acknowledgement = HERE/('accepted-interval-'+token+'.json')
    ack_ref = dict(path=str(acknowledgement.resolve()),sha256=r.sha(acknowledgement))
    serve.require(proof.get('kind') == 'server_observed_guardian_acceptance' and proof.get('version') == 1
        and proof.get('run_id') == serve.CHILD and proof.get('token') == token
        and proof.get('budget') == accepted['budget'] and proof.get('completion') == accepted['completion']
        and proof.get('guardian_acceptance') == ack_ref and proof.get('checkpoint') == accepted['checkpoint'],
        'Server did not close the acknowledged interval')
    finished = proof['observed_at_perf_counter']
    serve.require(type(finished) in (int,float) and math.isfinite(finished)
        and accepted['accepted_at_perf_counter'] <= finished < accepted['budget']['deadline']
        and time.perf_counter() < accepted['budget']['deadline'], 'Late server closure')
    return ack_ref,reference

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control',required=True)
    parser.add_argument('--control-sha256',required=True)
    args = parser.parse_args()
    control, prepared = serve.control(args.control,args.control_sha256)
    serve.require(not (HERE/'activation-intent.json').exists(),'Activation already attempted; preserve attempt')
    old_ref = control['old_guardian_completion']
    serve.require(Path(old_ref['path']).resolve() == OLD/'guardian-completion.json','Wrong old guardian receipt')
    old = r.pinned_json(old_ref)
    serve.require(old['reason'] == 'hard_interval_deadline' and old['cleanup_verified'] is True
        and old['active_processes_after'] == 0
        and old['child'] == dict(pid=2760,creation_time_100ns=134358094008519317)
        and old['guardian'] == dict(pid=71932,creation_time_100ns=134358093940188917), 'Old cutoff/cleanup differs')
    api = serve.load(HERE.parent/'supervisor.py','burial_owned_job').Win32()
    verify_old_absence(api,old)
    serve.require(serve.steering(),'Human hold/pause before launch')
    public = ROOT/'output/runs'/serve.CHILD
    serve.require(public == Path(prepared['intended_public_child']) and public.is_dir(),'Wrong published child')
    serve.require(r.sha(public/'checkpoint.npz') == prepared['checkpoint_sha256'],'Published boundary differs')
    serve.require(r.sha(ROOT/'output/runs'/serve.PARENT/'checkpoint.npz') == prepared['parent_checkpoint_sha256'],'Old accepted boundary changed')
    record = dict(kind='burial_successor_owned_interval_guardian',version=1,run_id=serve.CHILD,
        control_sha256=args.control_sha256,guardian=api.identity(api.k.GetCurrentProcess(),os.getpid()),
        old_guardian_completion=old_ref,created_at_utc=r.utc(),physical_completion_claimed=False)
    r.write_new(HERE/'activation-intent.json',record)
    job = child = guard = None
    assigned = False
    reason = 'guardian_error'
    error = None
    expected = None
    closed = set()
    try:
        initial_token = uuid.uuid4().hex
        initial_budget = new_budget(time.perf_counter())
        expected = dict(token=initial_token,budget=initial_budget,source=121.)
        secret = secrets.token_hex(32)
        command = [str(serve.PYTHON),'-B','-u',str(HERE/'serve.py'),'--control',str(Path(args.control).resolve()),
            '--control-sha256',args.control_sha256,'--gate',str(HERE/'launch-gate.json'),'--token',secret]
        job = api.make_job()
        guard = DeadlineGuard(api,job,initial_token,initial_budget)
        guard.thread.start()
        with (HERE/'server.stdout.log').open('xb',buffering=0) as out,(HERE/'server.stderr.log').open('xb',buffering=0) as err:
            child = api.create_suspended(command,HERE,out,err)
        serve.require(not guard.cutoff.is_set() and time.perf_counter() < initial_budget['deadline'],
            'Child creation exhausted initial interval clock')
        api.check(api.k.AssignProcessToJobObject(job,child.hProcess),'Assign newly suspended child to owned Job')
        assigned = True
        member = W.BOOL()
        api.check(api.k.IsProcessInJob(child.hProcess,job,ctypes.byref(member)),'Verify child Job membership')
        serve.require(member.value,'Child not in owned Job')
        identity = api.identity(child.hProcess,child.dwProcessId)
        record.update(child=identity,command=command,initial_budget=initial_budget,initial_interval_token=initial_token,
            child_assigned_to_job=True,job_membership_verified=True,kill_on_job_close=True,breakaway_allowed=False)
        r.write_new(HERE/'launched.json',record)
        gate = dict(token=secret,child=identity,guardian=record['guardian'],run_id=serve.CHILD,
            control_sha256=args.control_sha256,initial_budget=initial_budget,initial_interval_token=initial_token)
        r.write_new(HERE/'launch-gate.json',gate)
        serve.require(not guard.cutoff.is_set() and time.perf_counter() < initial_budget['deadline'],
            'Launch preparation exhausted initial interval clock')
        serve.require(api.k.ResumeThread(child.hThread) != 0xffffffff,'Resume owned suspended child failed')
        announced = False
        last_watch = last_members = 0.
        pause_sent = False
        accepted = {}
        closed = set()
        membership = None
        while api.k.WaitForSingleObject(child.hProcess,0) == 258 and not guard.cutoff.is_set():
            now = time.perf_counter()
            if now-last_watch >= 1.:
                last_watch = now
                running = serve.steering()
                if not running and not pause_sent:
                    # This may block; the independent deadline thread remains armed.
                    request = Request('http://127.0.0.1:8771/api/pause',data=b'{}',
                        headers={'Content-Type':'application/json'},method='POST')
                    with urlopen(request,timeout=2) as response:
                        json.load(response)
                    pause_sent = True
                    r.write_new(HERE/'human-pause-request.json',dict(time_utc=r.utc(),child=identity))
            if now-last_members >= 1.:
                last_members = now
                rows = job_members(api,job)
                if rows != membership:
                    membership = rows
                    r.write_new(HERE/('job-members-'+uuid.uuid4().hex+'.json'),dict(time_utc=r.utc(),members=rows,
                        guardian=record['guardian'],exact_job_membership_verified=True))
            if not announced and (HERE/'resume.json').is_file():
                resumed = r.json_read(HERE/'resume.json')
                activation = r.json_read(HERE/'activation.json')
                serve.require(resumed['child'] == identity and resumed['supported_resume_calls'] == 1
                    and resumed['initial_budget'] == initial_budget and resumed['initial_interval_token'] == initial_token
                    and activation['child'] == identity and activation['control_sha256'] == args.control_sha256,
                    'Startup identity/clock differs')
                r.write_new(HERE/'activation-result.json',dict(record,replacement_verified=True,
                    activation_sha256=r.sha(HERE/'activation.json'),resume_sha256=r.sha(HERE/'resume.json'),time_utc=r.utc()))
                announced = True
            marker_path = public/'numerical-budget-active.json'
            if marker_path.exists():
                marker = r.json_read(marker_path)
                serve.require(marker.get('run_id') == serve.CHILD and marker.get('pid') == identity['pid'], 'Marker owner differs')
                token = marker.get('token')
                if token in accepted:
                    serve.require(marker['budget'] == accepted[token]['budget'], 'Accepted clock mutated')
                    if marker.get('phase') == 'incomplete':
                        reason = 'incomplete_interval'
                        record['incomplete_interval'] = marker
                        break
                    if marker.get('phase') == 'accepted' and token not in closed:
                        receipt = accepted[token]
                        ack_ref,server_ref = validate_server_closure(marker,receipt,public)
                        release = dict(kind='guardian_closed_interval',version=1,run_id=serve.CHILD,
                            token=token,budget=receipt['budget'],source_time_myr=receipt['source_time_myr'],
                            destination_time_myr=receipt['destination_time_myr'],control_sha256=args.control_sha256,
                            guardian_acceptance=ack_ref,server_acceptance=server_ref,checkpoint=receipt['checkpoint'],
                            paused=bool(marker.get('paused')),complete=bool(marker.get('complete')),
                            closed_at_perf_counter=time.perf_counter())
                        release_path = HERE/('closed-interval-'+token+'.json')
                        r.write_new(release_path,release)
                        serve.require(r.json_read(release_path) == release,'Guardian closure readback differs')
                        terminal = bool(marker.get('paused') or marker.get('complete'))
                        release_ref = dict(path=str(release_path.resolve()),sha256=r.sha(release_path))
                        transitioned_at = guard.advance(token,None if terminal else receipt['next_interval_token'],
                            None if terminal else receipt['next_budget'])
                        transition = dict(kind='guardian_interval_transition',version=1,run_id=serve.CHILD,token=token,
                            control_sha256=args.control_sha256,budget=receipt['budget'],
                            source_time_myr=receipt['source_time_myr'],destination_time_myr=receipt['destination_time_myr'],
                            closed_interval=release_ref,transitioned_at_perf_counter=transitioned_at,
                            next_interval_token=None if terminal else receipt['next_interval_token'],
                            next_budget=None if terminal else receipt['next_budget'])
                        r.write_new(HERE/('transitioned-interval-'+token+'.json'),transition)
                        closed.add(token)
                        record['last_accepted_token'] = token
                        record['last_accepted_checkpoint_myr'] = receipt['destination_time_myr']
                        record['last_closure'] = release_ref
                        expected = dict(token=receipt['next_interval_token'],budget=receipt['next_budget'],
                            source=receipt['destination_time_myr'])
                        if terminal:
                            reason = 'completed_history' if marker.get('complete') else 'accepted_pause'
                            break
                else:
                    serve.require(token == expected['token'] and marker.get('budget') == expected['budget']
                        and marker.get('source_time_myr') == expected['source'], 'Unissued interval token/clock')
                    if marker.get('phase') == 'incomplete':
                        reason = 'incomplete_interval'
                        record['incomplete_interval'] = marker
                        break
                    if marker.get('phase') == 'awaiting_guardian':
                        report = validate_completion(marker,expected,public,args.control_sha256)
                        accepted_at = time.perf_counter()
                        destination = report['destination_time_myr']
                        next_token = uuid.uuid4().hex if destination < 1000. else None
                        following = new_budget(accepted_at,min(1.,1000.-destination)) if next_token else None
                        receipt = dict(kind='guardian_accepted_interval',version=1,run_id=serve.CHILD,token=token,
                            source_time_myr=expected['source'],destination_time_myr=destination,
                            requested_dt_myr=expected['budget']['dt_myr'],control_sha256=args.control_sha256,
                            child=identity,guardian=record['guardian'],budget=expected['budget'],
                            completion=marker['completion'],checkpoint=report['checkpoint'],frames=report['frames'],
                            accepted_at_perf_counter=accepted_at,next_interval_token=next_token,next_budget=following,
                            previous_acceptance_token=record.get('last_accepted_token'))
                        # Keep the old hard guard armed through acknowledgement publication.
                        r.write_new(HERE/('accepted-interval-'+token+'.json'),receipt)
                        accepted[token] = receipt
            time.sleep(.05)
        if guard.cutoff.is_set():
            reason = 'hard_interval_deadline'
            record['incomplete_interval'] = guard.receipt
        elif reason == 'guardian_error':
            reason = 'server_exited'
    except BaseException as exception:
        error = dict(type=type(exception).__name__,message=str(exception),traceback=traceback.format_exc())
    finally:
        if guard is not None and guard.cutoff.is_set():
            reason = 'hard_interval_deadline'
            record['incomplete_interval'] = guard.receipt
        elif reason not in ('accepted_pause','completed_history') and expected is not None and expected['token'] not in closed:
            record['incomplete_interval'] = dict(token=expected['token'],budget=expected['budget'],
                source_time_myr=expected['source'],reason=reason,error=error)
        if job is not None:
            if api.active(job):
                api.check(api.k.TerminateJobObject(job,124 if reason=='hard_interval_deadline' else 125),'Cleanup exact owned Job')
            until = time.monotonic()+10.
            while api.active(job) and time.monotonic() < until:
                time.sleep(.05)
            record['active_processes_after'] = api.active(job)
            record['cleanup_verified'] = record['active_processes_after'] == 0
        if guard is not None:
            guard.close()
        if child is not None:
            if not assigned and api.k.WaitForSingleObject(child.hProcess,0) == 258:
                api.check(api.k.TerminateProcess(child.hProcess,125),'Cleanup exact unassigned suspended child')
                api.k.WaitForSingleObject(child.hProcess,10000)
            record['child_exit_code'] = api.exit_code(child.hProcess)
            api.k.CloseHandle(child.hThread)
            api.k.CloseHandle(child.hProcess)
        if job is not None:
            api.k.CloseHandle(job)
        record.update(reason=reason,error=error,finished_at_utc=r.utc())
        r.write_new(HERE/'guardian-completion.json',record)
    if error:
        raise RuntimeError(error['message'])

if __name__ == '__main__':
    main()
