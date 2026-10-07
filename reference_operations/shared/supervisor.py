"""Windows-only private diagnostic supervisor. No model imports or launch at import.

Production CLI requires an explicitly reviewed ready plan, pinned helper/supervisor,
exact stopped-error run identity and current human steering. The child starts
suspended, enters a non-inheritable kill-on-close Job, then receives an atomic gate.
Every exit terminates remaining job descendants and records bounded cleanup.
"""
from __future__ import annotations
import argparse
import ctypes
from ctypes import wintypes as W
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FAILURE_ROOT = HERE
WATCH = ROOT / 'reviews/asein-integration/watch-state.json'
PYTHON = Path(r'C:\Users\LOCAL_USER\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe')
RUN = '20261006-contact114-gravity-admission'
CLEANUP_SECONDS = 5.0
POLL_SECONDS = .10
STEERING_POLL_SECONDS = 1.0


def now(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok, message):
    if not ok: raise ValueError(message)


def atomic_json(path, value, *, exclusive=False):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    with temporary.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    if exclusive and path.exists():
        temporary.unlink()
        raise FileExistsError(path)
    # Gate is in a fresh private directory with this supervisor as sole writer.
    os.replace(temporary, path)


class Win32:
    def __init__(self):
        require(os.name == 'nt', 'Windows Job Objects required')
        self.k = ctypes.WinDLL('kernel32', use_last_error=True)
        self._bind('CreateJobObjectW', [W.LPVOID, W.LPCWSTR], W.HANDLE)
        self._bind('SetInformationJobObject', [W.HANDLE, ctypes.c_int, W.LPVOID, W.DWORD], W.BOOL)
        self._bind('QueryInformationJobObject', [W.HANDLE, ctypes.c_int, W.LPVOID, W.DWORD, W.LPVOID], W.BOOL)
        self._bind('AssignProcessToJobObject', [W.HANDLE, W.HANDLE], W.BOOL)
        self._bind('IsProcessInJob', [W.HANDLE, W.HANDLE, ctypes.POINTER(W.BOOL)], W.BOOL)
        self._bind('TerminateJobObject', [W.HANDLE, W.UINT], W.BOOL)
        self._bind('GetProcessTimes', [W.HANDLE, W.LPVOID, W.LPVOID, W.LPVOID, W.LPVOID], W.BOOL)
        self._bind('GetExitCodeProcess', [W.HANDLE, ctypes.POINTER(W.DWORD)], W.BOOL)
        self._bind('WaitForSingleObject', [W.HANDLE, W.DWORD], W.DWORD)
        self._bind('TerminateProcess', [W.HANDLE, W.UINT], W.BOOL)
        self._bind('ResumeThread', [W.HANDLE], W.DWORD)
        self._bind('CloseHandle', [W.HANDLE], W.BOOL)
        self._bind('GetCurrentProcess', [], W.HANDLE)
        self._bind('SetHandleInformation', [W.HANDLE, W.DWORD, W.DWORD], W.BOOL)
        self._bind('InitializeProcThreadAttributeList', [W.LPVOID, W.DWORD, W.DWORD, ctypes.POINTER(ctypes.c_size_t)], W.BOOL)
        self._bind('UpdateProcThreadAttribute', [W.LPVOID, W.DWORD, ctypes.c_size_t, W.LPVOID, ctypes.c_size_t, W.LPVOID, W.LPVOID], W.BOOL)
        self._bind('DeleteProcThreadAttributeList', [W.LPVOID], None)
        self._bind('CreateProcessW', [W.LPCWSTR, W.LPWSTR, W.LPVOID, W.LPVOID, W.BOOL, W.DWORD, W.LPVOID, W.LPCWSTR, W.LPVOID, W.LPVOID], W.BOOL)

    def _bind(self, name, arguments, result):
        fn = getattr(self.k, name); fn.argtypes = arguments; fn.restype = result

    def check(self, result, label):
        if not result: raise ctypes.WinError(ctypes.get_last_error(), label)
        return result

    def identity(self, handle, pid):
        values = [ctypes.c_ulonglong() for _ in range(4)]
        self.check(self.k.GetProcessTimes(handle, *[ctypes.byref(v) for v in values]), 'GetProcessTimes')
        return dict(pid=int(pid), creation_time_100ns=values[0].value)

    def make_job(self):
        class BasicLimits(ctypes.Structure):
            _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong), ('PerJobUserTimeLimit', ctypes.c_longlong),
                        ('LimitFlags', W.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t), ('MaximumWorkingSetSize', ctypes.c_size_t),
                        ('ActiveProcessLimit', W.DWORD), ('Affinity', ctypes.c_size_t), ('PriorityClass', W.DWORD), ('SchedulingClass', W.DWORD)]
        class IOCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in ('ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount', 'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]
        class ExtendedLimits(ctypes.Structure):
            _fields_ = [('BasicLimitInformation', BasicLimits), ('IoInfo', IOCounters), ('ProcessMemoryLimit', ctypes.c_size_t),
                        ('JobMemoryLimit', ctypes.c_size_t), ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]
        job = self.k.CreateJobObjectW(None, None)
        self.check(job, 'CreateJobObjectW')
        try:
            self.check(self.k.SetHandleInformation(job, 1, 0), 'clear Job HANDLE_FLAG_INHERIT')
            limits = ExtendedLimits(); limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE; no breakaway flags
            self.check(self.k.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)), 'set kill-on-close Job limits')
            return job
        except BaseException:
            self.k.CloseHandle(job); raise

    def active(self, job):
        class Accounting(ctypes.Structure):
            _fields_ = [(n, ctypes.c_longlong) for n in ('TotalUserTime', 'TotalKernelTime', 'ThisPeriodTotalUserTime', 'ThisPeriodTotalKernelTime')] + [(n, W.DWORD) for n in ('TotalPageFaultCount', 'TotalProcesses', 'ActiveProcesses', 'TotalTerminatedProcesses')]
        info = Accounting()
        self.check(self.k.QueryInformationJobObject(job, 1, ctypes.byref(info), ctypes.sizeof(info), None), 'query Job accounting')
        return int(info.ActiveProcesses)

    def create_suspended(self, command, cwd, stdout, stderr):
        import msvcrt
        class StartupInfo(ctypes.Structure):
            _fields_ = [('cb', W.DWORD), ('lpReserved', W.LPWSTR), ('lpDesktop', W.LPWSTR), ('lpTitle', W.LPWSTR),
                        ('dwX', W.DWORD), ('dwY', W.DWORD), ('dwXSize', W.DWORD), ('dwYSize', W.DWORD),
                        ('dwXCountChars', W.DWORD), ('dwYCountChars', W.DWORD), ('dwFillAttribute', W.DWORD),
                        ('dwFlags', W.DWORD), ('wShowWindow', W.WORD), ('cbReserved2', W.WORD), ('lpReserved2', W.LPVOID),
                        ('hStdInput', W.HANDLE), ('hStdOutput', W.HANDLE), ('hStdError', W.HANDLE)]
        class StartupInfoEx(ctypes.Structure):
            _fields_ = [('StartupInfo', StartupInfo), ('lpAttributeList', W.LPVOID)]
        class ProcessInformation(ctypes.Structure):
            _fields_ = [('hProcess', W.HANDLE), ('hThread', W.HANDLE), ('dwProcessId', W.DWORD), ('dwThreadId', W.DWORD)]
        with open(os.devnull, 'rb') as stdin:
            raw = [msvcrt.get_osfhandle(f.fileno()) for f in (stdin, stdout, stderr)]
            handles = (W.HANDLE * 3)(*raw)
            size = ctypes.c_size_t()
            self.k.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(size))
            require(size.value > 0, 'Missing attribute-list size')
            attributes = ctypes.create_string_buffer(size.value)
            self.check(self.k.InitializeProcThreadAttributeList(attributes, 1, 0, ctypes.byref(size)), 'initialize handle allowlist')
            try:
                for handle in raw: self.check(self.k.SetHandleInformation(handle, 1, 1), 'enable allowed stdio handle inheritance')
                self.check(self.k.UpdateProcThreadAttribute(attributes, 0, 0x00020002, handles, ctypes.sizeof(handles), None, None), 'set inherited handle allowlist')
                startup = StartupInfoEx(); startup.StartupInfo.cb = ctypes.sizeof(startup)
                startup.StartupInfo.dwFlags = 0x100 | 1  # USESTDHANDLES | USESHOWWINDOW
                startup.StartupInfo.wShowWindow = 0
                startup.StartupInfo.hStdInput, startup.StartupInfo.hStdOutput, startup.StartupInfo.hStdError = raw
                startup.lpAttributeList = ctypes.cast(attributes, W.LPVOID)
                process = ProcessInformation(); line = ctypes.create_unicode_buffer(subprocess.list2cmdline([str(x) for x in command]))
                flags = 0x00000004 | 0x08000000 | 0x00080000  # SUSPENDED | NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT
                self.check(self.k.CreateProcessW(str(command[0]), line, None, None, True, flags, None, str(cwd), ctypes.byref(startup), ctypes.byref(process)), 'CreateProcessW suspended hidden child')
                return process
            finally:
                for handle in raw: self.k.SetHandleInformation(handle, 1, 0)
                self.k.DeleteProcThreadAttributeList(attributes)

    def exit_code(self, handle):
        code = W.DWORD(); self.check(self.k.GetExitCodeProcess(handle, ctypes.byref(code)), 'GetExitCodeProcess')
        return int(code.value)



def capture_control_binding(plan):
    """One fixed gravity-only call or its separate decode preflight; no generic extension."""
    mode=plan.get('mode');wall=plan.get('wall_seconds')
    require(mode in ('gravity_full','decode_only'), 'Unsupported gravity diagnostic mode')
    require(type(wall) is int, 'Wall limit must be an integer')
    control_id='gravity-admission-full-3600-v1' if mode=='gravity_full' else 'gravity-admission-full-decode-v1'
    require(wall==3600 if mode=='gravity_full' else 1<=wall<=90, 'Wrong mode-specific wall bound')
    output=(FAILURE_ROOT/control_id).resolve()
    require(plan.get('control_id')==control_id and Path(plan.get('output','')).resolve()==output, 'Fixed output/control differs')
    require(plan.get('run_id')==RUN and plan.get('time_myr')==114. and plan.get('next_output_myr')==116. and plan.get('requested_dt_myr')==1., 'Run or scheduling boundary changed')
    require(plan.get('checkpoint_sha256')=='bf1261b7c7b91669acb2da674813a1a0093884b3394834c86e0efe8a3e0661bf', 'Preserved checkpoint differs')
    require(plan.get('gravity_event_sha256')=='c6bd1b5bc6b0191fd72009b0aa52417b3fc0a63119af216f51954137803494a7', 'First gravity input differs')
    bounds=dict(workers=6,blas_threads=1,evidence_bytes=536870912,terminal_reserve_bytes=16777216,max_gravity_calls=1,max_endpoint_calls=2048,max_joint_attempts=4096,event_limit=32768,rich_payload_decoded_limit_bytes=268435456)
    for key,value in bounds.items():
        require(type(plan.get(key)) is int and plan[key]==value, 'Fixed control differs: '+key)
    require(plan.get('rich_payload_codec')=='zlib-json-v1','Rich payload codec differs')
    expected=dict(maximum_attempts=1,automatic_retry=False,automatic_extension=False)
    one_shot=plan.get('one_shot')
    require(type(one_shot) is dict and set(one_shot)==set(expected), 'Missing one-shot control')
    require(type(one_shot['maximum_attempts']) is int and one_shot['maximum_attempts']==1 and one_shot['automatic_retry'] is False and one_shot['automatic_extension'] is False, 'Retry or extension forbidden')
    return dict(control_id=control_id,mode=mode,wall_seconds=wall,output=str(output),gravity_event_sha256=plan['gravity_event_sha256'],candidate_bounded_gravity_sha256=plan['candidate_bounded_gravity_sha256'],candidate_gravitational_relaxation_sha256=plan['candidate_gravitational_relaxation_sha256'],original_arguments_typed_sha256=plan['original_arguments_typed_sha256'],intentional_option=dict(gravity_area_targeting=True),one_shot=dict(one_shot),**bounds)


def validate_supervision_control(wall_seconds, directory, gate_fields, reviewed_control):
    """No IO or process work; generic harmless callers retain the original short ceiling."""
    require(type(wall_seconds) in (int, float) and type(wall_seconds) is not bool and 0 < wall_seconds, 'Invalid private wall limit')
    if reviewed_control is None:
        require(wall_seconds <= 1800, 'Extended supervision requires the reviewed fixed plan')
        require('control_binding' not in gate_fields, 'Unvalidated gate control binding')
        return
    binding = capture_control_binding(reviewed_control)
    require(wall_seconds == binding['wall_seconds'] and gate_fields.get('control_binding') == binding, 'Supervision control/gate mismatch')
    output = Path(binding['output'])
    require(Path(directory).resolve() == output.with_name(output.name + '-supervision'), 'Supervision output differs from bound plan')
    require(reviewed_control.get('ready') is True and reviewed_control.get('supervisor_review_required') is False, 'Control plan not reviewed ready')
    require(gate_fields.get('run_id') == RUN, 'Gate run differs')
    for field in ('plan_sha256', 'harness_sha256', 'supervisor_sha256'):
        value = gate_fields.get(field)
        require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'Missing exact gate hash')
    require(gate_fields['harness_sha256'] == reviewed_control.get('harness_sha256') and gate_fields['supervisor_sha256'] == reviewed_control.get('supervisor_sha256'), 'Gate helper/supervisor hash differs')

def supervise(command, *, directory, wall_seconds, gate_fields, guard=lambda: None, fault_hook=lambda stage: None, reviewed_control=None):
    """Internal tested primitive. Caller validates plan and supplies fresh isolated directory.

    The wall limit starts before child creation; cleanup gets at most five further
    seconds. Exceptions still write a receipt and clean only exact owned handles.
    """
    validate_supervision_control(wall_seconds, directory, gate_fields, reviewed_control)
    directory = Path(directory).resolve()
    require(directory.is_relative_to(FAILURE_ROOT.resolve()), 'Supervision output outside isolated failure directory')
    require(not directory.exists(), 'Supervision directory must be fresh')
    directory.mkdir(parents=False)
    receipt_path = directory / 'supervision.json'; gate_path = directory / 'gate.json'
    receipt = dict(kind='private_capture_windows_job_supervision', version=1, created_at_utc=now(),
                   command=[str(x) for x in command], wall_seconds=wall_seconds, cleanup_seconds_limit=CLEANUP_SECONDS,
                   supervision_directory=str(directory), gate=str(gate_path), gate_released=False, child_assigned_to_job=False,
                   kill_on_job_close=True, job_handle_inheritable=False, breakaway_allowed=False, output_status='initializing', **gate_fields)
    atomic_json(receipt_path, receipt)
    api = Win32(); job = None; process = None; assigned = False; started = time.monotonic(); reason = None
    receipt['supervisor'] = api.identity(api.k.GetCurrentProcess(), os.getpid())
    logs = []
    try:
        guard()
        job = api.make_job()
        logs = [(directory / name).open('xb', buffering=0) for name in ('stdout.log', 'stderr.log')]
        process = api.create_suspended(command, directory, *logs)
        receipt['child'] = api.identity(process.hProcess, process.dwProcessId)
        receipt['child_created_at_utc'] = now()
        atomic_json(receipt_path, receipt)
        api.check(api.k.AssignProcessToJobObject(job, process.hProcess), 'AssignProcessToJobObject')
        assigned = True
        member = W.BOOL()
        api.check(api.k.IsProcessInJob(process.hProcess, job, ctypes.byref(member)), 'verify assigned Job membership')
        require(member.value, 'Child is not in owned Job')
        receipt['child_assigned_to_job'] = True; receipt['job_membership_verified'] = True
        guard(); fault_hook('before_gate')
        require(time.monotonic() - started < wall_seconds, 'Deadline reached before gate release')
        gate = dict(gate_fields, child=receipt['child'], supervisor=receipt['supervisor'], kill_on_job_close=True, child_assigned_to_job=True)
        atomic_json(gate_path, gate, exclusive=True)
        receipt['gate_released'] = True; receipt['gate_sha256'] = sha(gate_path)
        receipt['gate_released_at_utc'] = now(); atomic_json(receipt_path, receipt)
        require(api.k.ResumeThread(process.hThread) != 0xFFFFFFFF, 'ResumeThread failed')
        fault_hook('after_gate')
        last_guard = time.monotonic()
        while True:
            elapsed = time.monotonic() - started
            if elapsed >= wall_seconds:
                reason = 'hard_wall_timeout'; break
            if time.monotonic() - last_guard >= STEERING_POLL_SECONDS:
                guard(); last_guard = time.monotonic()
            fault_hook('poll')
            wait = api.k.WaitForSingleObject(process.hProcess, max(1, int(min(POLL_SECONDS, wall_seconds - elapsed) * 1000)))
            if wait == 0:
                reason = 'child_exited'; receipt['child_exit_code_before_cleanup'] = api.exit_code(process.hProcess); break
            require(wait == 258, 'Unexpected child wait result')
    except BaseException as exc:
        reason = 'supervisor_exception'; receipt['error_type'] = type(exc).__name__; receipt['error'] = str(exc)
        receipt['traceback'] = traceback.format_exc()
    finally:
        receipt['termination_reason'] = reason; receipt['work_wall_elapsed_seconds'] = time.monotonic() - started
        cleanup_started = time.monotonic(); errors = []
        if job is not None:
            try:
                receipt['active_processes_before_cleanup'] = api.active(job)
                api.check(api.k.TerminateJobObject(job, 124 if reason == 'hard_wall_timeout' else 125), 'TerminateJobObject cleanup')
                while api.active(job) and time.monotonic() - cleanup_started < CLEANUP_SECONDS:
                    time.sleep(.025)
                receipt['active_processes_after_cleanup'] = api.active(job)
            except BaseException as exc: errors.append(type(exc).__name__ + ': ' + str(exc))
            finally:
                api.k.CloseHandle(job); receipt['job_handle_closed'] = True
        if process is not None:
            # If assignment failed, this exact still-suspended child has no job;
            # terminate only its retained handle, never a PID found by enumeration.
            try:
                if not assigned and api.k.WaitForSingleObject(process.hProcess, 0) == 258:
                    api.check(api.k.TerminateProcess(process.hProcess, 125), 'terminate unassigned suspended child')
                remaining = max(0., CLEANUP_SECONDS - (time.monotonic() - cleanup_started))
                receipt['child_signaled_after_cleanup'] = api.k.WaitForSingleObject(process.hProcess, int(remaining * 1000)) == 0
                receipt['child_exit_code_after_cleanup'] = api.exit_code(process.hProcess)
            except BaseException as exc: errors.append(type(exc).__name__ + ': ' + str(exc))
            finally:
                api.k.CloseHandle(process.hThread); api.k.CloseHandle(process.hProcess)
        for log in logs: log.close()
        receipt['cleanup_elapsed_seconds'] = time.monotonic() - cleanup_started
        receipt['cleanup_errors'] = errors
        receipt['cleanup_verified'] = not errors and receipt.get('active_processes_after_cleanup', 0) == 0 and receipt.get('child_signaled_after_cleanup', process is None)
        receipt['output_status'] = 'complete' if reason == 'child_exited' and receipt.get('child_exit_code_before_cleanup') == 0 and receipt['cleanup_verified'] else 'stopped_or_failed'
        receipt['finished_at_utc'] = now(); receipt['total_wall_elapsed_seconds'] = time.monotonic() - started
        atomic_json(receipt_path, receipt)
    return receipt


def read_shared(path):
    """Read active steering without denying concurrent Windows atomic replacement."""
    require(os.name == 'nt', 'Windows shared-delete reader required')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [W.LPCWSTR, W.DWORD, W.DWORD, W.LPVOID, W.DWORD, W.DWORD, W.HANDLE]
    kernel.CreateFileW.restype = W.HANDLE
    kernel.GetFileSizeEx.argtypes = [W.HANDLE, ctypes.POINTER(ctypes.c_longlong)]; kernel.GetFileSizeEx.restype = W.BOOL
    kernel.ReadFile.argtypes = [W.HANDLE, W.LPVOID, W.DWORD, ctypes.POINTER(W.DWORD), W.LPVOID]; kernel.ReadFile.restype = W.BOOL
    kernel.CloseHandle.argtypes = [W.HANDLE]; kernel.CloseHandle.restype = W.BOOL
    handle = kernel.CreateFileW(str(path), 0x80000000, 1 | 2 | 4, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value: raise ctypes.WinError(ctypes.get_last_error())
    try:
        size = ctypes.c_longlong()
        if not kernel.GetFileSizeEx(handle, ctypes.byref(size)): raise ctypes.WinError(ctypes.get_last_error())
        require(0 <= size.value <= 64 * 1024 * 1024, 'Steering file exceeds bounded read')
        buffer = ctypes.create_string_buffer(size.value)
        received = W.DWORD()
        if not kernel.ReadFile(handle, buffer, size.value, ctypes.byref(received), None): raise ctypes.WinError(ctypes.get_last_error())
        require(received.value == size.value, 'Incomplete steering snapshot')
        return buffer.raw
    finally: kernel.CloseHandle(handle)


def steering_guard():
    watch = json.loads(read_shared(WATCH).decode('utf-8-sig'))
    require(watch.get('paused_by_user') is False and watch.get('monitoring_stopped_by_user') is False, 'Human pause/stop or unavailable steering')
    require(watch.get('current_run_id') == RUN, 'Current Watch run changed')
    require(watch.get('current_server_url') == 'http://127.0.0.1:8771', 'Current Watch server changed')
    require(watch.get('contact113_delivery', {}).get('run_id') == RUN, 'Delivery binding run changed')


def validate_endpoint_review_records(review,integrity):
    require(review.get('kind')=='independent_endpoint_geometry_and_recorded_certificate_review' and review.get('stage')=='geometry','Wrong endpoint geometry review')
    for key in ('verification_complete','stage_passed','qualified_endpoint_review_pass'):
        require(review.get(key) is True,'Endpoint geometry review not passed: '+key)
    require(integrity.get('kind')=='independent_endpoint_terminal_integrity' and integrity.get('stage')=='integrity','Wrong endpoint integrity review')
    for key in ('verification_complete','stage_passed','integrity_and_complete_evidence_pass','returned_endpoint_evidence'):
        require(integrity.get(key) is True,'Endpoint integrity review not passed: '+key)
    require(integrity.get('receipt',{}).get('plan_sha256')=='71daecf6163428927a9b0665a947d0ceb93b4bdd85ee19437760b53f3fd8cf65','Endpoint review belongs to another plan')
    job=integrity.get('supervision',{})
    require(job.get('cleanup_verified') is True and type(job.get('active_processes_after_cleanup')) is int and job['active_processes_after_cleanup']==0,'Endpoint review lacks complete Job cleanup')
    require(job.get('output_status')=='complete' and job.get('child_exit_code_after_cleanup')==0,'Endpoint prerequisite Job failed')


def validate_support_files(plan):
    support=plan.get('support_files')
    require(type(support) is dict and len(support)>=2, 'Missing observation helper pins')
    for required in ('evidence.py','observers.py','reader.py','codec-review.json','codec-manifest.json'):
        require(str(HERE/required) in support, 'Missing helper: '+required)
    for filename,digest in support.items():
        path=Path(filename).resolve()
        require(path.is_relative_to(FAILURE_ROOT.resolve()) and path.is_file(), 'Support helper outside diagnostic scope')
        require(type(digest) is str and len(digest)==64 and sha(path)==digest, 'Observation helper changed')
    fixture=FAILURE_ROOT/'gravity-entry-fixture-v1'/'fixture.py';binding=fixture.with_name('binding.json')
    require(Path(plan.get('fixture_path','')).resolve()==fixture and Path(plan.get('fixture_binding','')).resolve()==binding,'Fixture paths differ')
    require(sha(fixture)==plan.get('fixture_sha256') and sha(binding)==plan.get('fixture_binding_sha256'),'Fixture pins differ')
    candidate=FAILURE_ROOT/'gravity-admission-candidate-v1'/'bounded_gravity.py'
    require(Path(plan.get('candidate_bounded_gravity_path','')).resolve()==candidate,'Wrong candidate overlay')
    require(plan.get('candidate_bounded_gravity_sha256')=='2eb39b49f4c2e36ea24af63578560c3c7b6ffa0a00becf8a1c3278bb053b440f' and sha(candidate)==plan['candidate_bounded_gravity_sha256'],'Candidate changed')
    gravity=candidate.with_name('gravitational_relaxation.py')
    require(Path(plan.get('candidate_gravitational_relaxation_path','')).resolve()==gravity,'Wrong gravity overlay')
    require(plan.get('candidate_gravitational_relaxation_sha256')=='4d9285fe818c5e7b0da5e35d5c01e6398b3de83a3bc275018e5933c6cd54c178' and sha(gravity)==plan['candidate_gravitational_relaxation_sha256'],'Gravity candidate changed')
    require(plan.get('source_overlays')=={'bounded_gravity.py':str(candidate),'gravitational_relaxation.py':str(gravity)},'Only two reviewed source overlays allowed')
    require(plan.get('overlay_sha256')=={'bounded_gravity.py':plan['candidate_bounded_gravity_sha256'],'gravitational_relaxation.py':plan['candidate_gravitational_relaxation_sha256']},'Overlay pins differ')
    require(plan.get('original_arguments_typed_sha256')=='eb737d817dd4b5f15a929ca9bf82a85b6ecd4f332311dcd297263130a92e3827','Authentic original gravity input differs')
    require(plan.get('fixture_sha256')=='6bd29c308e5c50bf11856022fe97e2bb08fe6e3ea2dc3a56e5d3a2cc38a4f0ef' and plan.get('fixture_binding_sha256')=='a3a7c9150fd2d9639f6d8ea9b5c0520a7a3eb365de6530bbf53636f9b174163d','Original gravity fixture differs')
    validation=plan.get('endpoint_validation',{})
    require(validation.get('independently_passed') is True,'Independent endpoint result review still required')
    review=validation.get('review')
    require(type(review) is dict and Path(review.get('path','')).resolve().is_relative_to(FAILURE_ROOT.resolve()) and sha(review['path'])==review.get('sha256'),'Endpoint review receipt differs')
    review_root=FAILURE_ROOT/'gravity-admission-endpoint-v1'/'result-review'
    require(Path(review['path']).resolve().is_relative_to(review_root.resolve()),'Endpoint review outside exact audit scope')
    geometry_review=json.loads(read_shared(Path(review['path'])))
    prior=geometry_review.get('integrity_receipt')
    require(type(prior) is dict and Path(prior.get('path','')).resolve().is_relative_to(review_root.resolve()) and sha(prior['path'])==prior.get('sha256'),'Endpoint integrity prerequisite differs')
    integrity_review=json.loads(read_shared(Path(prior['path'])))
    validate_endpoint_review_records(geometry_review,integrity_review)
    require(validation.get('endpoint_plan_sha256')=='71daecf6163428927a9b0665a947d0ceb93b4bdd85ee19437760b53f3fd8cf65','Endpoint prerequisite plan differs')
    reuse=plan.get('original_fixture_decode_verification',{})
    require(reuse.get('sha256')=='4ae06f5b51a235125ecd4ed5b394b784a2390d238a6b7985de701e93750c8bcc' and sha(reuse['path'])==reuse['sha256'],'Original gravity decode proof differs')
    option=plan.get('intentional_option')
    require(type(option) is dict and set(option)=={'gravity_area_targeting'} and option['gravity_area_targeting'] is True,'Missing explicit candidate opt-in')
    require(plan.get('writer_reviewed') is True,'Compressed evidence writer not reviewed')
    require(sha(HERE/'evidence.py')=='cf9a0ef2af56c7cb7236b08a8235a8f98ac06095ecd619c8bcf5668b9819df55' and sha(HERE/'reader.py')=='d2c96303e35a8516067715aac736afea3b42e3138a3b9555b94e1e622c6592ce','Reviewed codec source differs')
    require(sha(HERE/'codec-review.json')=='a7dbdaae6f423d4d2b7d2f2c6ee5e1a97c4aa71944e723d234024fd99fe26fdd','Codec review differs')



def evidence_budget_guard(plan,supervision):
    # Parent checks independently of cooperative child writes. External files can
    # grow between bounded polls; the child reserves256KiB for final Job evidence.
    total=0
    for root in (Path(plan['output']),Path(supervision)):
        if root.exists():
            for path in root.rglob('*'):
                try:
                    if path.is_file():total+=path.stat().st_size
                except FileNotFoundError:pass  # Worker scratch may disappear between observations.
    require(total<=plan['evidence_bytes']-262144, 'Diagnostic evidence ceiling reached; Job receipt allowance retained')


def validate_real_plan(args):
    plan_path = Path(args.plan).resolve(); harness = HERE / 'capture.py'
    require(plan_path.is_relative_to(FAILURE_ROOT.resolve()), 'Plan outside isolated failure directory')
    require(sha(plan_path) == args.expected_plan_sha256, 'Reviewed plan hash differs')
    require(sha(harness) == args.expected_harness_sha256, 'Reviewed capture helper hash differs')
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    require(plan.get('kind') == 'private_contact114_gravity_admission_full' and plan.get('version') == 1, 'Wrong plan kind/version')
    require(plan.get('ready') is True and plan.get('supervisor_review_required') is False, 'Plan is not reviewed ready')
    require(plan.get('supervisor_sha256') == sha(__file__), 'Reviewed supervisor hash differs')
    require(plan.get('harness_sha256') == args.expected_harness_sha256 and plan.get('run_id') == RUN, 'Plan helper/run mismatch')
    capture_control_binding(plan)
    require(plan.get('workers') == 6, 'Worker policy changed')
    output = Path(plan['output']).resolve()
    require(output.is_relative_to(FAILURE_ROOT.resolve()) and output.parent.exists() and not output.exists(), 'Need fresh isolated capture output')
    supervision = output.with_name(output.name + '-supervision')
    require(not supervision.exists(), 'Supervision output already exists')
    require(Path(sys.executable).resolve() == PYTHON.resolve(), 'Use reviewed bundled Python -B')
    require(sys.dont_write_bytecode and sys.flags.optimize == 0, 'Ordinary Python -B without optimization required')
    validate_support_files(plan)
    steering_guard()
    with urllib.request.urlopen('http://127.0.0.1:8771/api/status', timeout=5) as response:
        status = json.load(response)
    require(status.get('run_id') == RUN and status.get('state') == 'error', 'Expected exact errored run before private diagnostic')
    require(status.get('checkpoint_time_myr') == 114 and status.get('frame_count') == 59, 'Accepted boundary changed')
    return plan_path, harness, plan, supervision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--expected-harness-sha256', required=True)
    args = parser.parse_args()
    plan_path, harness, plan, supervision = validate_real_plan(args)
    command = [str(PYTHON), '-B', str(harness), 'run', '--plan', str(plan_path), '--expected-plan-sha256', args.expected_plan_sha256,
               '--expected-harness-sha256', args.expected_harness_sha256, '--supervision-gate', str(supervision / 'gate.json')]
    def guard():
        steering_guard()
        require(sha(plan_path) == args.expected_plan_sha256 and sha(harness) == args.expected_harness_sha256 and sha(__file__) == plan['supervisor_sha256'], 'Plan/helper/supervisor changed during capture')
        validate_support_files(plan)
        evidence_budget_guard(plan,supervision)
    result = supervise(command, directory=supervision, wall_seconds=plan['wall_seconds'],
                       gate_fields=dict(plan_sha256=args.expected_plan_sha256, harness_sha256=args.expected_harness_sha256,
                                        supervisor_sha256=sha(__file__), run_id=RUN, control_binding=capture_control_binding(plan)),
                       guard=guard, reviewed_control=plan)
    receipt = supervision / 'supervision.json'
    print(json.dumps(dict(status=result['output_status'], termination_reason=result['termination_reason'], receipt=str(receipt), receipt_sha256=sha(receipt), cleanup_verified=result['cleanup_verified'])))
    return 0 if result['output_status'] == 'complete' else 1


if __name__ == '__main__': sys.exit(main())
