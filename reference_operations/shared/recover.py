"""Reviewed-control-gated exact accepted114 recovery. UNREADY until separately bound.

prepare creates a new paused source boundary; serve loads it and resumes once.
No command runs without an exact ready control file and independent full result.
"""
from __future__ import annotations
import argparse,contextlib,ctypes,hashlib,importlib.util,io,json,os,re,sys,time,uuid,zipfile
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
from urllib.request import urlopen
HERE=Path(__file__).resolve().parent;FAILURE=HERE.parent;ROOT=FAILURE.parent.parent
PYTHON=Path(r'C:\Users\LOCAL_USER\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe')
PARENT='20261005-contact113-convex';CP='bf1261b7c7b91669acb2da674813a1a0093884b3394834c86e0efe8a3e0661bf'
PATCHES={'bounded_gravity.py':'2eb39b49f4c2e36ea24af63578560c3c7b6ffa0a00becf8a1c3278bb053b440f','gravitational_relaxation.py':'4d9285fe818c5e7b0da5e35d5c01e6398b3de83a3bc275018e5933c6cd54c178'}
META='__checkpoint_json.npy'
def require(value,message):
    if not value:raise ValueError(message)
def utc():return datetime.now(timezone.utc).isoformat()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def loaded_tree(value):
    # A decoded set has no iteration-order contract. Every other tagged order stays exact.
    if isinstance(value,list):return [loaded_tree(x) for x in value]
    if isinstance(value,dict):
        result={k:loaded_tree(v) for k,v in value.items()}
        if set(value)=={'t','v'} and value['t']=='set':
            require(type(value['v']) is list,'Invalid typed set');result['v']=sorted(result['v'],key=canonical)
        return result
    return value
def digest(value):return hashlib.sha256(value).hexdigest()
def lp(path):
    value=str(Path(path).absolute());return value if value.startswith('\\\\?\\') else '\\\\?\\'+value
def safe(path,root):
    path=Path(path);root=Path(root).resolve();require(path.is_absolute() and path.resolve()==path and path.is_relative_to(root),'Path escapes declared root or uses symlink');return path
@contextlib.contextmanager
def shared_stream(path):
    import msvcrt
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.CreateFileW.argtypes=[ctypes.c_wchar_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p];kernel.CreateFileW.restype=ctypes.c_void_p
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.CreateFileW(lp(path),0x80000000,7,None,3,0x80,None)
    if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    try:fd=msvcrt.open_osfhandle(handle,os.O_RDONLY|os.O_BINARY)
    except BaseException:kernel.CloseHandle(handle);raise
    with os.fdopen(fd,'rb') as stream:yield stream
def read(path,limit=64*1024*1024):
    with shared_stream(path) as stream:raw=stream.read(limit+1)
    require(len(raw)<=limit,'Read exceeds declared cap');return raw
def sha(path):
    h=hashlib.sha256()
    with shared_stream(path) as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def json_read(path):return json.loads(read(path))
def write_new(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('xb') as stream:stream.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False).encode()+b'\n');stream.flush();os.fsync(stream.fileno())
    # Windows rename atomically publishes and refuses an existing destination.
    os.rename(temporary,path)
    return dict(path=str(path),sha256=sha(path))
def copy_shared(source,target):
    target=Path(lp(target));require(not target.exists(),'Refuse overwrite');target.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256()
    with shared_stream(source) as before,target.open('xb') as after:
        for chunk in iter(lambda:before.read(1024*1024),b''):h.update(chunk);after.write(chunk)
        after.flush();os.fsync(after.fileno())
    require(sha(source)==sha(target)==h.hexdigest(),'Copy source changed or copy differs')
def inventory(root):
    result={};root=Path(lp(root))
    for path in root.rglob('*'):
        require(not path.is_symlink() and path.resolve()==path,'Unsafe source history path')
        if path.is_file():result[path.relative_to(root).as_posix()]=sha(path)
    return result
def pinned_json(ref):
    path=Path(ref['path']);require(sha(path)==ref['sha256'],'Evidence pin differs');return json_read(path)
def control(path,expected,mode):
    require(os.name=='nt' and Path(sys.executable).resolve()==PYTHON.resolve() and sys.dont_write_bytecode,'Pinned Windows Python -B required')
    raw=read(path);require(digest(raw)==expected,'Recovery control changed');plan=json.loads(raw)
    require(plan.get('ready') is True and plan.get('root_reviewed') is True and plan.get('allow_'+mode) is True,'Recovery mode not independently ready')
    require(plan['helpers']['recover.py']==sha(__file__),'Recovery driver source changed')
    require(plan['parent_run_id']==PARENT and plan['parent_checkpoint_sha256']==CP and plan['patches_sha256']==PATCHES,'Wrong accepted source boundary')
    require(plan['workers']==6 and plan['blas_threads']==1 and plan['port']==8771,'Runtime configuration changed')
    require(plan['child_run_id']!=PARENT and re.fullmatch('[A-Za-z0-9_-]{1,80}',plan['child_run_id']),'Unsafe child identity')
    require(Path(plan['runtime'])==HERE/'runtime' and Path(plan['child_path'])==ROOT/'output/runs'/plan['child_run_id'],'Wrong child/runtime destination')
    review=pinned_json(plan['full_gravity_review'])
    require(review.get('verification_complete') is True and review.get('qualified_full_gravity_evidence_pass') is True and review.get('stage_passed') is True,'Independent full-gravity result did not pass')
    terminal=pinned_json(review['terminal_verification'])
    require(terminal['plan']['sha256']==plan['full_gravity_plan_sha256'],'Full-gravity result uses different experiment')
    for ref in plan['authority_receipts']:pinned_json(ref)
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):os.environ[name]='1'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    return plan
def steering(expected_run=PARENT):
    watch=json_read(ROOT/'reviews/asein-integration/watch-state.json')
    require(watch.get('current_run_id')==expected_run and watch.get('current_server_url')=='http://127.0.0.1:8771','Selected run/server changed')
    require(watch.get('paused_by_user') is False and watch.get('monitoring_stopped_by_user') is False,'Human stop/pause is active')
    require(watch.get('solver_fix_and_resume_human_instruction')=='Get it sorted out, fixed and resumed','Repair authority changed')
    return watch
def parent_status():
    with urlopen('http://127.0.0.1:8771/api/status',timeout=5) as response:status=json.load(response)
    require(status.get('run_id')==PARENT and status.get('state')=='error' and status.get('checkpoint_time_myr')==114. and status.get('frame_count')==59,'Parent is not the exact errored114 run')
    return status
def import_runtime(plan):
    runtime=Path(plan['runtime']);sys.path.insert(0,str(runtime))
    require(not any(name in sys.modules for name in ('native_engine','server','checkpoint')),'Scientific modules already loaded')
    import checkpoint,native_engine,server,parallel_runtime
    for module in (checkpoint,native_engine,server,parallel_runtime):require(Path(module.__file__).resolve().parent==runtime,'Wrong runtime module')
    return checkpoint,native_engine,server,parallel_runtime
def metadata(path):
    import numpy as np
    with zipfile.ZipFile(io.BytesIO(read(path))) as archive:
        names=archive.namelist();require(len(names)==len(set(names)) and names.count(META)==1 and archive.testzip() is None,'Invalid checkpoint archive')
        array=np.load(io.BytesIO(archive.read(META)),allow_pickle=False)
    require(array.dtype==np.uint8 and array.ndim==1,'Invalid checkpoint metadata type');return json.loads(array.tobytes())
def rewrite_checkpoint(source,target,record):
    import numpy as np
    buffer=io.BytesIO();np.save(buffer,np.frombuffer(json.dumps(record,separators=(',',':'),allow_nan=False).encode(),dtype=np.uint8),allow_pickle=False)
    with zipfile.ZipFile(io.BytesIO(read(source))) as before,zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as after:
        names=before.namelist();require(len(names)==len(set(names)) and all(name==META or re.fullmatch('a[0-9]{5}\\.npy',name) for name in names),'Unexpected archive members')
        for entry in before.infolist():after.writestr(entry,buffer.getvalue() if entry.filename==META else before.read(entry))
    with zipfile.ZipFile(io.BytesIO(read(source))) as before,zipfile.ZipFile(target) as after:
        require(before.namelist()==after.namelist() and after.testzip() is None,'Child archive members/CRC differ')
        require(all(before.read(name)==after.read(name) for name in before.namelist() if name!=META),'Raw typed array members changed')
        count=len(before.namelist())-1
    actual=metadata(target);require(canonical(actual)==canonical(record) and canonical(actual['state'])==canonical(metadata(source)['state']),'Checkpoint metadata/state differ')
    return count
def prepare(args,plan):
    steering();parent_status();parent=ROOT/'output/runs'/PARENT;cp=Path(plan['preserved_checkpoint']);source=Path(plan['source_root']);runtime=Path(plan['runtime']);child=Path(plan['child_path'])
    require(sha(cp)==sha(parent/'checkpoint.npz')==CP,'Accepted114 checkpoint changed')
    require(not runtime.exists() and not child.exists() and not (HERE/'prepared.json').exists(),'Recovery output already exists; no automatic retry')
    caches=plan['source_derived_cache_files']
    require(len(caches)==35 and not (set(caches)&set(plan['source_files'])) and all(re.fullmatch('__pycache__/[^/]+\\.cpython-312\\.pyc',n) for n in caches),'Unexpected derived cache inventory')
    full_source_inventory=dict(plan['source_files'],**caches)
    require(inventory(source)==full_source_inventory and len(plan['source_files'])==227,'Sealed source/assets or retained derived cache changed')
    for name,path in plan['patch_paths'].items():require(sha(path)==PATCHES[name],'Candidate source changed')
    before=metadata(cp);saved=before['manifest'];parent_files=inventory(parent)
    stage=HERE/('stage-'+uuid.uuid4().hex);stage.mkdir();runtime.mkdir()
    write_new(HERE/'preparation-intent.json',dict(control_sha256=args.expected_control_sha256,parent_files_sha256=parent_files,stage=str(stage),child=str(child),time_utc=utc()))
    for name in plan['source_files']:
        copy_shared(Path(plan['patch_paths'].get(name,str(source/name))),safe(runtime/name,runtime))
    current_inventory=inventory(runtime);require(set(current_inventory)==set(plan['source_files']),'Runtime membership changed')
    require({n for n in current_inventory if current_inventory[n]!=plan['source_files'][n]}==set(PATCHES),'Unexpected runtime source delta')
    checkpoint,native,server,parallel=import_runtime(plan);compatibility=server.SimulationManager.compatibility()
    proposal=load_proposal(plan)
    changes=proposal.source_delta(before['compatibility'],compatibility,plan['source_files'],current_inventory)
    for name in parent_files:copy_shared(safe(parent/name,parent),safe(stage/'inherited-parent'/name,stage))
    require(inventory(stage/'inherited-parent')==parent_files,'Inherited parent copy differs')
    for name in ('initial.json','config.json',*[f'frame_{i:04d}{suffix}' for i in range(59) for suffix in ('.npz','.json')]):copy_shared(parent/name,stage/name)
    (stage/'engine.py').write_bytes(server.ENGINE_SOURCE)
    for name,payload in server.AUXILIARY_SOURCES.items():safe(stage/name,stage).write_bytes(payload)
    record=proposal.boundary_record(before,plan['child_run_id'],compatibility,parent_files,created_utc=utc())
    array_count=rewrite_checkpoint(cp,stage/'checkpoint.npz',record);copy_shared(stage/'checkpoint.npz',stage/'checkpoint.source-boundary.npz');child_cp=sha(stage/'checkpoint.npz')
    write_new(stage/'manifest.json',record['manifest'])
    # Guard every construction/evolution entry while exercising the real loader.
    guarded=[(native.Simulation,n) for n in ('__init__','step','snapshot')]+[(server.SimulationManager,n) for n in ('start','resume')]+[(parallel.RuntimePool,'map')]
    originals=[(owner,name,getattr(owner,name)) for owner,name in guarded]
    def forbidden(*a,**kw):raise AssertionError('No construction, evolution, resume or worker submission during prepare')
    manager=None
    try:
        for owner,name,method in originals:setattr(owner,name,forbidden)
        restored,embedded=checkpoint.read_checkpoint(stage/'checkpoint.npz',compatibility,native.Simulation)
        arrays={};encoded=checkpoint._encode(vars(restored),arrays)
        require(canonical(loaded_tree(encoded))==canonical(loaded_tree(before['state'])) and len(arrays)==array_count,'Supported load changed full typed state/RNG')
        import numpy as np
        with np.load(io.BytesIO(read(cp)),allow_pickle=False) as archive:
            for name,array in arrays.items():
                original=archive[name];require(original.dtype==array.dtype and original.shape==array.shape and original.tobytes(order='C')==array.tobytes(order='C'),'Supported load changed array: '+name)
        require(restored.t==114. and canonical(restored.config)==canonical(saved['config']) and canonical(embedded)==canonical(record['manifest']),'Supported load scheduler/config differs')
        class PausedManager(server.SimulationManager):
            def list_runs(self):return []
        # Validate under a fresh isolated root containing the actual child name.
        validation_root=HERE/('load-validation-'+uuid.uuid4().hex);validation_root.mkdir();validation_child=validation_root/plan['child_run_id'];stage.rename(validation_child);stage=validation_child
        manager=PausedManager(root=validation_root,workers=6,recovery_interval_myr=1.)
        loaded=manager.load(plan['child_run_id']);workers=manager.workers_status()
        require(loaded['state']=='paused' and loaded['can_resume'] is True and loaded['time_myr']==114. and loaded['frame_count']==59 and loaded['next_output_myr']==116.,'Supported paused manager load differs')
        require(workers['requested_workers']==workers['max_workers']==6 and workers['pool_started'] is False,'Unexpected preparation worker work')
        del restored
    finally:
        for owner,name,method in originals:setattr(owner,name,method)
        if manager is not None:manager.parallel.close()
    require(sha(stage/'checkpoint.npz')==child_cp and inventory(parent)==parent_files and inventory(source)==full_source_inventory and inventory(runtime)==current_inventory,'Preparation mutated boundary/parent/source')
    steering();parent_status()
    receipt=dict(kind=proposal.KIND,version=1,child_run_id=plan['child_run_id'],child_path=str(child),runtime_path=str(runtime),parent_run_id=PARENT,parent_path=str(parent),parent_checkpoint_sha256=CP,
        parent_files_sha256=parent_files,source_parent_revision=saved['source_commit'],source_revision=record['manifest']['source_commit'],source_patches_sha256=PATCHES,changed_helpers=changes,compatibility=compatibility,
        runtime_files_sha256=current_inventory,child_checkpoint_sha256=child_cp,entire_typed_state_exact=True,preexisting_array_count=array_count,inherited_frame_count=59,parent_epoch_myr=114.,next_output_myr=116.,
        physical_time_advanced_myr=0.,model_constructed=False,model_stepped=False,workers_started=False,policy_migration=False,area_credit_renewed=False,control_sha256=args.expected_control_sha256,
        producer_sha256=sha(__file__),full_gravity_review=plan['full_gravity_review'],source_derived_cache_files=caches,derived_cache_files_copied=False,prepared_at_utc=utc(),supported_loader_verified=True)
    write_new(stage/'source-transition.json',receipt)
    require(not child.exists(),'Child appeared during preparation');stage.rename(child);write_new(HERE/'prepared.json',receipt)
    print(json.dumps(dict(prepared=str(HERE/'prepared.json'),sha256=sha(HERE/'prepared.json'),child_run_id=plan['child_run_id'],checkpoint_sha256=child_cp)),flush=True)
def load_proposal(plan):
    path=HERE/'recovery_proposal.py';require(sha(path)==plan['helpers']['recovery_proposal.py'],'Metadata contract changed');spec=importlib.util.spec_from_file_location('recovery_contract',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def serve(args,plan):
    require(args.launch_gate and args.launch_token and re.fullmatch('[0-9a-f]{64}',args.launch_token),'Explicit owned startup gate required')
    gate_path=safe(args.launch_gate,HERE);deadline=time.monotonic()+60
    while not gate_path.exists():
        require(time.monotonic()<deadline,'No owned gate released within60s');time.sleep(.05)
    gate_raw=read(gate_path);gate=json.loads(gate_raw);gate_sha=digest(gate_raw)
    require(gate['launch_token']==args.launch_token and gate['control_sha256']==args.expected_control_sha256 and gate['child_run_id']==plan['child_run_id'] and gate['child']['pid']==os.getpid(),'Wrong owned launch gate')
    # Exact creation identity is verified using the already reviewed stdlib API.
    sup_path=Path(plan['identity_helper']['path']);require(sha(sup_path)==plan['identity_helper']['sha256'],'Identity helper changed')
    spec=importlib.util.spec_from_file_location('recovery_identity',sup_path);sup=importlib.util.module_from_spec(spec);spec.loader.exec_module(sup);api=sup.Win32()
    require(api.identity(api.k.GetCurrentProcess(),os.getpid())==gate['child'],'Reused child PID')
    steering();receipt=pinned_json(plan['prepared_receipt']);runtime=Path(plan['runtime']);child=Path(plan['child_path'])
    require(receipt['control_sha256']==plan['preparation_control_sha256'] and receipt['child_run_id']==plan['child_run_id'] and receipt['compatibility'] and receipt['supported_loader_verified'] is True,'Prepared boundary is not verified')
    require(inventory(runtime)==receipt['runtime_files_sha256'] and sha(child/'checkpoint.npz')==receipt['child_checkpoint_sha256'],'Prepared source/boundary changed')
    checkpoint,native,server,parallel=import_runtime(plan);require(server.SimulationManager.compatibility()==receipt['compatibility'],'Runtime compatibility differs')
    class Manager(server.SimulationManager):
        bootstrap=True
        first116_attempted=False
        def list_runs(self):return [] if self.bootstrap else super().list_runs()
        def _commit_checkpoint(self,simulation,next_output,paused=False):
            committed=super()._commit_checkpoint(simulation,next_output,paused=paused)
            try:
                if committed.get('checkpoint_time_myr')==116. and not self.first116_attempted:
                    self.first116_attempted=True;target=HERE/'first116';target.mkdir();current=self.path()
                    header=checkpoint.checkpoint_header(current/'checkpoint.npz',receipt['compatibility'])
                    require(header['time_myr']==116. and header['manifest']==committed,'First116 accepted checkpoint identity differs')
                    external=json_read(current/'manifest.json')
                    require(all(external.get(k)==committed.get(k) for k in ('run_id','frame_count','frames','checkpoint_time_myr')),'First116 external history differs')
                    copy_shared(current/'checkpoint.npz',target/'checkpoint.npz');copy_shared(current/'manifest.json',target/'external-manifest.json')
                    require(json_read(target/'external-manifest.json')==external,'First116 external manifest changed')
                    for suffix in ('.npz','.json'):copy_shared(current/('frame_0059'+suffix),target/('frame_0059'+suffix))
                    write_new(target/'checkpoint-manifest.json',committed)
                    write_new(target/'capture.json',dict(kind='exact_accepted_first116_capture',child_run_id=child.name,checkpoint_time_myr=116.,frame_count=committed['frame_count'],checkpoint_sha256=sha(target/'checkpoint.npz'),checkpoint_manifest_sha256=sha(target/'checkpoint-manifest.json'),external_manifest_sha256=sha(target/'external-manifest.json'),frame_files={suffix:sha(target/('frame_0059'+suffix)) for suffix in ('.npz','.json')},source_revision=receipt['source_revision'],model_state_modified=False,captured_at_utc=utc()))
            except BaseException as exc:
                self.first116_capture_error=exc
                try:write_new(HERE/'first116-capture-error.json',dict(error=type(exc).__name__+': '+str(exc),child_run_id=child.name,native_commit_retained=True,automatic_capture_retry=False,time_utc=utc()))
                except BaseException as log_error:self.first116_capture_log_error=log_error
            return committed
    manager=Manager(root=child.parent,workers=6,recovery_interval_myr=1.);httpd=None
    manager.bootstrap=False
    try:
        loaded=manager.load(child.name);require(loaded['state']=='paused' and loaded['can_resume'] and loaded['time_myr']==114. and loaded['frame_count']==59 and loaded['next_output_myr']==116.,'Child failed supported load')
        steering()
        httpd=server.ThreadingHTTPServer(('127.0.0.1',8771),server.Handler);server.Handler.manager=manager
        write_new(HERE/'activation.json',dict(pid=os.getpid(),creation_time_100ns=gate['child']['creation_time_100ns'],executable=sys.executable,argv=sys.orig_argv,time_utc=utc(),child_run_id=child.name,
            port=8771,source_revision=receipt['source_revision'],checkpoint_sha256=receipt['child_checkpoint_sha256'],workers=manager.workers_status(),control_sha256=args.expected_control_sha256,gate_sha256=gate_sha,stage='supported_load_succeeded_before_resume'))
        steering()
        resumed=manager.resume(child.name);require(resumed['state'] in ('resuming','running'),'Supported resume did not start')
        write_new(HERE/'resume.json',dict(supported_resume_calls=1,child_run_id=child.name,pid=os.getpid(),time_utc=utc(),state=resumed['state']))
        httpd.serve_forever(poll_interval=.5)
    finally:
        manager.prepare_shutdown()
        if manager.worker and manager.worker.is_alive():manager.worker.join()
        manager.parallel.close()
        if httpd is not None:httpd.server_close()
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','serve'));p.add_argument('--control',required=True);p.add_argument('--expected-control-sha256',required=True);p.add_argument('--launch-gate');p.add_argument('--launch-token');args=p.parse_args()
    plan=control(args.control,args.expected_control_sha256,args.mode);return prepare(args,plan) if args.mode=='prepare' else serve(args,plan)
if __name__=='__main__':main()
