"""One source-pinned successor; no restart, alternate run, or clock renewal."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
REPAIR = HERE.parent
ROOT = REPAIR.parents[1]
CHILD = '20261007-coarse-burial-speed-history'
PARENT = '20261007-coarse-rigid-sheet-history'
PYTHON = Path(r'C:\Users\LOCAL_USER\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe')
SHARED = ROOT/'tmp/contact114-failure-20261006/gravity-admission-recovery-v1/recover.py'
SHARED_SHA = 'f4ebf06c61ab65af7b2bf6226baa86caf80c4d37a8af160753fc1c044b9ef819'
SUPERVISOR_SHA = 'fb68c98c9c6ee894c4f78b9d04c77b8fccb84789db8d01fbdc38df8d9bd82007'

def require(ok, message):
    if not ok:
        raise ValueError(message)

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

require(hashlib.sha256(SHARED.read_bytes()).hexdigest() == SHARED_SHA, 'Shared reader changed')
r = load(SHARED, 'burial_lifecycle_shared')

def steering():
    watch = r.json_read(ROOT/'reviews/asein-integration/watch-state.json')
    require(watch.get('current_run_id') in (PARENT, CHILD)
            and watch.get('current_server_url') == 'http://127.0.0.1:8771', 'Watch binding changed')
    row = watch.get('contact121_failure', {}).get('burial_speed_implementation_steering', {})
    require(row.get('authorized') is True and row.get('child_run_id') == CHILD,
            'Successor implementation authorization changed')
    return watch.get('paused_by_user') is False and watch.get('monitoring_stopped_by_user') is False

def control(path, pin):
    require(os.name == 'nt' and sys.dont_write_bytecode and not sys.flags.optimize
            and Path(sys.executable).resolve() == PYTHON.resolve(), 'Pinned ordinary Windows Python -B required')
    require(r.sha(path) == pin, 'Control changed')
    value = r.json_read(path)
    require(value['child_run_id'] == CHILD and value['parent_run_id'] == PARENT
            and value['port'] == 8771 and value['workers'] == 6, 'Wrong successor binding')
    for name, digest in value['helpers'].items():
        require(name in ('activate.py', 'serve.py') and r.sha(HERE/name) == digest, 'Lifecycle source changed')
    require(set(value['helpers']) == {'activate.py', 'serve.py'}, 'Incomplete lifecycle pins')
    require(value['identity_helper_sha256'] == SUPERVISOR_SHA
            and r.sha(REPAIR/'supervisor.py') == SUPERVISOR_SHA, 'Job helper changed')
    authorization = r.pinned_json(value['authorization_receipt'])
    require(authorization.get('authorized') is True, 'Missing pinned successor authorization')
    prepared = r.pinned_json(value['prepared_receipt'])
    require(prepared['child_run_id'] == CHILD and prepared['supported_paused_loader_verified'] is True,
            'Unsupported prepared successor')
    require(prepared['checkpoint_time_myr'] == 121. and prepared['frame_count'] == 62
            and prepared['next_output_myr'] == 122., 'Prepared accepted boundary differs')
    require(r.inventory(HERE/'runtime') == prepared['runtime_files_sha256'], 'Sealed successor source changed')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    return value, prepared

def main():
    parser = argparse.ArgumentParser()
    for name in ('control', 'control-sha256', 'gate', 'token'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    value, prepared = control(args.control, args.control_sha256)
    require(steering(), 'Human hold or pause')
    require(Path(args.gate).resolve() == HERE/'launch-gate.json', 'Wrong launch gate')
    gate = r.json_read(args.gate)
    require(gate['token'] == args.token and gate['control_sha256'] == args.control_sha256
            and gate['run_id'] == CHILD and gate['child']['pid'] == os.getpid(), 'Gate identity differs')
    api = load(REPAIR/'supervisor.py', 'burial_job_identity').Win32()
    require(api.identity(api.k.GetCurrentProcess(), os.getpid()) == gate['child'], 'Child PID reused')
    require(time.perf_counter() < gate['initial_budget']['physical_deadline'], 'Initial preparation budget exhausted')
    child = ROOT/'output/runs'/CHILD
    require(r.sha(child/'checkpoint.npz') == prepared['checkpoint_sha256'], 'Source boundary changed')
    sys.path.insert(0, str(HERE/'runtime'))
    import budget
    import server
    require(server.SimulationManager.compatibility() == prepared['compatibility'], 'Runtime compatibility differs')

    class Manager(server.SimulationManager):
        def list_runs(self):
            return []
        def start(self, *args, **kwargs):
            raise ValueError('This controller owns one prepared successor only')
        def load(self, run_id):
            require(run_id == CHILD, 'Alternate run forbidden')
            return super().load(run_id)
        def resume(self, run_id=None):
            require(run_id in (None, CHILD) and not getattr(self, '_resume_used', False), 'Only one resume is allowed')
            self._resume_used = True
            return super().resume(CHILD)
        def branch(self, *args, **kwargs):
            raise ValueError('Branching is outside this owned runtime')

    manager = Manager(root=child.parent, workers=6, recovery_interval_myr=1.)
    manager._initial_numerical_budget = gate['initial_budget']
    manager._initial_numerical_token = gate['initial_interval_token']
    manager._guardian_directory = HERE
    manager._guardian_control_sha256 = args.control_sha256
    httpd = None
    try:
        with budget.installed(gate['initial_budget']):
            budget.require_physical_time('supported_load')
            status = manager.load(CHILD)
            require(status['state'] == 'paused' and status['can_resume']
                    and status['checkpoint_time_myr'] == 121. and status['frame_count'] == 62
                    and status['next_output_myr'] == 122., 'Supported paused load differs')
            budget.require_physical_time('supported_load:completed')
            httpd = server.ThreadingHTTPServer(('127.0.0.1', 8771), server.Handler)
            server.Handler.manager = manager
            r.write_new(HERE/'activation.json', dict(run_id=CHILD, child=gate['child'], guardian=gate['guardian'],
                control_sha256=args.control_sha256, prepared_sha256=value['prepared_receipt']['sha256'],
                source_revision=prepared['source_revision'], time_utc=r.utc(), stage='loaded_before_resume'))
            require(steering(), 'Human hold before resume')
            budget.require_physical_time('supported_resume')
            resumed = manager.resume(CHILD)
            require(resumed['state'] in ('running', 'resuming'), 'Supported resume failed')
            r.write_new(HERE/'resume.json', dict(run_id=CHILD, child=gate['child'], supported_resume_calls=1,
                initial_interval_token=gate['initial_interval_token'], initial_budget=gate['initial_budget'], time_utc=r.utc()))
        httpd.serve_forever(poll_interval=.2)
    finally:
        manager.prepare_shutdown()
        if manager.worker and manager.worker.is_alive():
            manager.worker.join()
        manager.parallel.close()
        if httpd is not None:
            httpd.server_close()

if __name__ == '__main__':
    main()
