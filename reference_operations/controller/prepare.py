"""Prepare a source-only successor of accepted121. Never evolve or launch it."""
import argparse,copy,hashlib,importlib.util,json,os,re,shutil,sys
from pathlib import Path
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent
REPAIR=HERE.parent
ROOT=REPAIR.parents[1]
OLD=REPAIR/'coarse-production-recovery-v1'
PARENT_ID='20261007-coarse-rigid-sheet-history'
CHILD_ID='20261007-coarse-burial-speed-history'
PARENT=ROOT/'output/runs'/PARENT_ID
CP_SHA='c4557974cfc5cdff8a3556500121abe75be5e6f094d8a832ee10aae7419bad3e'
PREP_SHA='06f94caf54924fcea5949eb78ed583fceab71285e11b7bdf6529ded6205d493f'
SHARED=ROOT/'tmp/contact114-failure-20261006/gravity-admission-recovery-v1/recover.py'
SHARED_SHA='f4ebf06c61ab65af7b2bf6226baa86caf80c4d37a8af160753fc1c044b9ef819'
PYTHON=Path(r'C:\Users\LOCAL_USER\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe')

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return dict(path=str(path),sha256=sha(path))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--policy-sha256',required=True);a=ap.parse_args()
    require(os.name=='nt' and sys.dont_write_bytecode and not sys.flags.optimize and Path(sys.executable).resolve()==PYTHON.resolve(),'Pinned ordinary Windows Python-B required')
    require(sha(SHARED)==SHARED_SHA,'Shared reader differs');r=load(SHARED,'burial_stage_shared')
    require(sha(OLD/'prepared.json')==PREP_SHA,'Old boundary receipt differs');old=json.loads((OLD/'prepared.json').read_bytes())
    require(r.sha(PARENT/'checkpoint.npz')==CP_SHA,'Parent durable checkpoint changed')
    require(r.sha(PARENT/'checkpoint.source-boundary.npz')==CP_SHA,'Parent source boundary changed')
    policy_path=Path(a.policy).resolve();require(policy_path==HERE/'source-policy.json' and sha(policy_path)==a.policy_sha256,'Wrong exact source policy')
    policy=json.loads(policy_path.read_bytes())
    require(policy['kind']=='burial_speed_source_only_policy' and policy['mode']=='coarse_rigid_sheet_history_v1' and policy['seconds_per_myr']==1800,'Wrong source policy')
    authority=r.pinned_json(policy['authorization'])
    require(authority['authorized'] is True and authority['human_instruction']=='go for your plan' and authority['child_run_id']==CHILD_ID,'Wrong current human approval')
    watch=r.json_read(ROOT/'reviews/asein-integration/watch-state.json')
    require(watch['current_run_id']==PARENT_ID and watch['current_server_url']=='http://127.0.0.1:8771' and not watch['paused_by_user'] and not watch['monitoring_stopped_by_user'],'Current steering changed')
    require(len(policy['required_reviews'])>=3,'Source, staging, and lifecycle reviews required')
    for pin in policy['required_reviews']:
        review=r.pinned_json(pin)
        predicates=pin.get('required_values',{})
        require(bool(predicates),'Review needs explicit positive admission predicates')
        for key,expected_value in predicates.items():
            actual=review
            for part in key.split('.'):actual=actual[part]
            require(type(actual) is type(expected_value) and actual==expected_value,'Review admission failed: '+key)
    patches=policy['files'];allowed={'burial_depth.py','collision_interface.py','server.py'}
    require(set(patches)==allowed,'Unexpected source closure')
    expected=copy.deepcopy(old['runtime_files_sha256'])
    for name,pin in expected.items():require(sha(OLD/'runtime'/name)==pin,'Sealed old runtime changed: '+name)
    for name,ref in patches.items():
        source=Path(ref['path']).resolve();require(source.is_relative_to(REPAIR) and sha(source)==ref['sha256'],'Candidate patch differs: '+name)
        require(name in expected and expected[name]!=ref['sha256'],'Expected a real source delta: '+name)
        expected[name]=ref['sha256']
    runtime=HERE/'runtime';runs=HERE/'staged-runs';child=runs/CHILD_ID
    require(not runtime.exists() and not runs.exists() and not (ROOT/'output/runs'/CHILD_ID).exists(),'One-shot staging target exists')
    metadata=r.metadata(PARENT/'checkpoint.npz');saved=metadata['manifest'];prior=metadata['compatibility']
    require(metadata['time_myr']==121 and saved['run_id']==PARENT_ID and saved['frame_count']==len(saved['frames'])==62 and saved['frames'][-1]['time_myr']==120 and saved['next_output_myr']==122,'Parent epoch/history mismatch')
    config=saved['config'];require(config['history_mode']=='coarse_rigid_sheet_history_v1' and config['deforming_regions']==config['adaptive_refinement']==0 and config['duration_myr']==1000 and config['snapshot_myr']==2 and config['width']==512 and config['height']==256,'Changed coarse science/config')
    files=['checkpoint.npz','manifest.json','config.json','initial.json','source-transition.json',*[f'frame_{i:04d}{s}' for i in range(62) for s in ('.npz','.json')]]
    parent_pins={n:r.sha(PARENT/n) for n in files}
    ancestor=PARENT/'inherited-parent';require(ancestor.is_dir(),'Original coarse ancestry missing')
    ancestor_files=[p for p in ancestor.rglob('*') if p.is_file()]
    require(all(not p.is_symlink() for p in ancestor_files),'Unexpected ancestry symlink')
    need=sum((PARENT/n).stat().st_size for n in files)+sum(p.stat().st_size for p in ancestor_files)+512*1024**2
    require(shutil.disk_usage(HERE).free>need,'Insufficient staging space')
    write_new(HERE/'intent.json',dict(kind='burial_speed_source_only_stage',parent_run_id=PARENT_ID,child_run_id=CHILD_ID,parent_checkpoint_sha256=CP_SHA,policy=dict(path=str(policy_path),sha256=a.policy_sha256),producer_sha256=sha(__file__),model_steps=0))
    runtime.mkdir();child.mkdir(parents=True)
    for name,pin in expected.items():
        source=Path(patches[name]['path']) if name in patches else OLD/'runtime'/name
        dest=runtime/name;dest.parent.mkdir(parents=True,exist_ok=True);r.copy_shared(source,dest);require(sha(dest)==pin,'Runtime copy changed')
    # Preserve the old coarse marker's relative inherited-parent references exactly.
    ancestry_pins={}
    for source in ancestor_files:
        name=str(source.relative_to(ancestor));dest=child/'inherited-parent'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        pin=r.sha(source);r.copy_shared(source,dest);require(sha(dest)==pin,'Original ancestry copy changed');ancestry_pins[name]=pin
    for name in files:
        if name in ('checkpoint.npz','manifest.json','config.json','source-transition.json'):
            dest=child/'source-parent'/name
        else:dest=child/name
        dest.parent.mkdir(parents=True,exist_ok=True);r.copy_shared(PARENT/name,dest);require(sha(dest)==parent_pins[name],'Parent copy changed')
    sys.path.insert(0,str(runtime))
    import checkpoint,native_engine,server,parallel_runtime,coarse_history
    for module in (checkpoint,native_engine,server,parallel_runtime):require(Path(module.__file__).resolve().parent==runtime,'Wrong imported runtime')
    compatibility=server.SimulationManager.compatibility()
    for key in prior:
        if key!='auxiliary_sources_sha256':require(prior[key]==compatibility[key],'Engine/backend changed: '+key)
    old_aux=prior['auxiliary_sources_sha256'];new_aux=compatibility['auxiliary_sources_sha256']
    require(set(old_aux)==set(new_aux),'Scientific helper set changed')
    require({n for n in old_aux if old_aux[n]!=new_aux[n]}==set(patches)&set(old_aux),'Unexpected compatibility delta')
    def forbidden(*a,**k):raise AssertionError('No model construction/evolution/render/resume/worker dispatch during staging')
    for name in ('__init__','step','snapshot'):setattr(native_engine.Simulation,name,forbidden)
    parallel_runtime.RuntimePool.map=forbidden
    def fingerprint(value):
        arrays={};tree=r.loaded_tree(checkpoint._encode(value,arrays))
        info={n:dict(dtype=v.dtype.str,shape=list(v.shape),sha256=hashlib.sha256(v.tobytes(order='C')).hexdigest()) for n,v in arrays.items()}
        return hashlib.sha256(canonical(dict(tree=tree,arrays=info))).hexdigest()
    simulation,embedded=checkpoint.read_checkpoint(child/'source-parent/checkpoint.npz',prior,native_engine.Simulation)
    require(embedded==saved and simulation.t==121,'Decoded parent differs');coarse_history.validate(simulation)
    state_pins={n:fingerprint(v) for n,v in vars(simulation).items()}
    revision='sha256:'+hashlib.sha256(canonical(expected)).hexdigest()
    manifest=copy.deepcopy(saved)
    manifest.update(run_id=CHILD_ID,created=datetime.now(timezone.utc).isoformat(),state='paused',can_resume=True,source_commit=revision,source_revision_kind='source_tree_sha256',source_parent_revision=saved['source_commit'],source_patches_sha256={n:v['sha256'] for n,v in patches.items()},time_myr=121.,integration_time_myr=121.,checkpoint_time_myr=121.,next_output_myr=122.,eta_seconds=None,
        branch_origin=dict(run_id=PARENT_ID,time_myr=121.,checkpoint_file='source-parent/checkpoint.npz',checkpoint_sha256=CP_SHA,original_run_directory=str(PARENT)),
        source_transition=dict(kind='coarse_burial_speed_future_only',version=1,epoch_myr=121.,receipt='source-transition.json',inherited_frame_count=62,policy=dict(path=str(policy_path),sha256=a.policy_sha256),seconds_per_myr=1800,physical_time_advanced_myr=0.,state_changed=False,area_credit_renewed=False,mode='coarse_rigid_sheet_history_v1',geometry_remeshed=False),**copy.deepcopy(compatibility))
    for key in ('error','resume_reason','recovery','source_patch_sha256','numerical_budget','numerical_budget_record'):manifest.pop(key,None)
    checkpoint.write_checkpoint(child/'checkpoint.npz',simulation,manifest,compatibility)
    require({n:fingerprint(v) for n,v in vars(simulation).items()}==state_pins,'Checkpoint writing changed physical state')
    write_new(child/'config.json',config);write_new(child/'manifest.json',manifest)
    shutil.copyfile(child/'checkpoint.npz',child/'checkpoint.source-boundary.npz')
    (child/'engine.py').write_bytes(server.ENGINE_SOURCE)
    for name,data in server.AUXILIARY_SOURCES.items():
        dest=child/name;require(dest.resolve().is_relative_to(child),'Saved source escapes child');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    restored,restored_manifest=checkpoint.read_checkpoint(child/'checkpoint.npz',compatibility,native_engine.Simulation)
    require(restored_manifest==manifest and {n:fingerprint(v) for n,v in vars(restored).items()}==state_pins,'Supported reload changed typed state')
    coarse_history.validate(restored)
    class Manager(server.SimulationManager):
        def list_runs(self):return []
        def start(self,*a,**k):return forbidden()
        def resume(self,*a,**k):return forbidden()
    manager=Manager(root=runs,workers=6,recovery_interval_myr=1.)
    try:
        loaded=manager.load(CHILD_ID)
        require(loaded['state']=='paused' and loaded['can_resume'] and loaded['checkpoint_time_myr']==121 and loaded['frame_count']==62 and loaded['next_output_myr']==122,'Supported paused load differs')
        require(manager.workers_status()['pool_started'] is False,'Unexpected worker creation')
    finally:manager.parallel.close()
    require(r.sha(PARENT/'checkpoint.npz')==CP_SHA and all(r.sha(PARENT/n)==pin for n,pin in parent_pins.items()),'Parent changed during source staging')
    require(all(sha(runtime/n)==pin for n,pin in expected.items()),'Candidate runtime changed')
    result=dict(kind='private_coarse_burial_source_only121_stage',version=1,child_run_id=CHILD_ID,parent_run_id=PARENT_ID,staged_child=str(child),runtime=str(runtime),intended_public_child=str(ROOT/'output/runs'/CHILD_ID),source_policy=dict(path=str(policy_path),sha256=a.policy_sha256),source_revision=revision,compatibility=compatibility,runtime_files_sha256=expected,checkpoint_sha256=sha(child/'checkpoint.npz'),parent_checkpoint_sha256=CP_SHA,parent_files_sha256=parent_pins,original_ancestor_directory=str(PARENT),ancestry_policy='Immediate parent checkpoint/source receipt in source-parent; original coarse inherited-parent preserved byte-for-byte, all62saved slices unchanged; original historical frame_sources retained.',ancestry_files_sha256=ancestry_pins,inherited_frame_count=62,frame_count=62,checkpoint_time_myr=121.,next_output_myr=122.,target_myr=1000.,physical_time_advanced_myr=0.,geometry_remeshed=False,unchanged_state_fields_sha256=state_pins,changed_state_fields=[],mode_policy=copy.deepcopy(simulation.coarse_history_policy),original_config=config,new_config=config,supported_paused_loader_verified=True,model_constructor_calls=0,model_step_calls=0,production_writes=False,server_actions=0,budget_enforcement=policy['budget_enforcement'],producer_sha256=sha(__file__),created_at_utc=datetime.now(timezone.utc).isoformat())
    write_new(child/'source-transition.json',result);ref=write_new(HERE/'prepared.json',result)
    print(json.dumps(dict(prepared=ref,staged_child=str(child),checkpoint_sha256=result['checkpoint_sha256'],unchanged_state_fields=len(state_pins),production_writes=False)))

if __name__=='__main__':main()
