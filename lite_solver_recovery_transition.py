"""Explicit repaired-source and numerical-policy continuation at accepted105.

The interrupted parent is preserved verbatim. New policy metadata is declared;
all preexisting state imports exactly. No constructor, step or cache rebuilding
is permitted during publication. The manifest commits only after readback.
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import hashlib, io, json, os
from pathlib import Path
import re, shutil, sys, uuid, subprocess, tarfile
ROOT=Path(__file__).resolve().parent
REV=ROOT/'reviews/asein-integration'
PARENT_ID='20261003-131535-7383c8'
STOP=REV/'human-stop-20261003-131535-7383c8-20261003T234351329996Z'
CP_SHA='f3fd3c204ff426f4eec433fa6894fc885f715f5750746d17c2139bf3196a3773'
PRES_SHA='57ace40a4ccb487a8d70f9d13bb231ac9ed35fe5886e6d227a6b8d3eb14f3acb'
AUTH=REV/'lite-solver-fix-and-resume-authority-20261004T0222161909210Z.json'
AUTH_SHA='b5cd7992fda0fe9d5708a88464510403e3f9604060186f180d3decfb9017b63c'
KIND='lite_contact_geometry_numerical_policy_recovery'
def require(c,m):
    if not c:raise ValueError('Lite solver recovery transition: '+m)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def lp(p):return Path('\\\\?\\'+str(Path(p).resolve())) if sys.platform=='win32' else Path(p)
def scoped(p):
    p=Path(p).resolve();require(p.is_relative_to(ROOT),'Evidence escaped isolated checkout.');return p
def write_new(p,v):
    with Path(p).open('xb') as f:f.write((json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode());f.flush();os.fsync(f.fileno())
def files(p):
    base=lp(p)
    require(not any(x.is_symlink() for x in base.rglob('*')),'Historical symlinks unsupported.')
    return {x.relative_to(base).as_posix():sha(x) for x in base.rglob('*') if x.is_file()}
def publish(*,source_seal,source_seal_sha256,private_receipt,private_receipt_sha256,repair_review,repair_review_sha256,source_commit,run_id=None):
    """One fresh, paused child. Failed staging remains separately preserved."""
    import numpy as np
    import checkpoint,native_engine,numerical_accuracy,server
    from lite_initiation_transition import _fingerprint
    from lite_gravity_iteration_transition import _verify_recursive_history
    require(Path.cwd().resolve()==ROOT,'Exact isolated cwd required.')
    refs=[(scoped(source_seal),source_seal_sha256,'runtime-source-seal.json'),(scoped(private_receipt),private_receipt_sha256,'private-complete106-validation.json'),(scoped(repair_review),repair_review_sha256,'repair-review.json'),(AUTH,AUTH_SHA,'application-authority.json'),(STOP/'preservation-completion.json',PRES_SHA,'parent-preservation.json')]
    require(all(sha(p)==h for p,h,_ in refs),'External evidence pins differ.')
    seal=json.loads(Path(source_seal).read_bytes());validation=json.loads(Path(private_receipt).read_bytes());review=json.loads(Path(repair_review).read_bytes())
    require(validation['whole_interval_accepted'] is True and validation['outcome']=='private_complete106_interval_verified' and validation['source_seal_sha256']==source_seal_sha256 and all(validation['post_checks'].values()) and validation['checks_passed']==validation['checks_total']>0,'Complete private106 qualification is missing.')
    require(review['passed'] is True and review['source_seal_sha256']==source_seal_sha256 and review['private_receipt_sha256']==private_receipt_sha256,'Fresh independent repair review differs.')
    for row in review['evidence_pins']:require(sha(scoped(row['path']))==row['sha256'],'Independent review artifact changed.')
    watch_raw=(REV/'watch-state.json').read_bytes();watch=json.loads(watch_raw)
    require(watch['current_run_id']==PARENT_ID and watch['solver_resume_authorized'] and watch['solver_fix_and_resume_authority_sha256']==AUTH_SHA and not watch['paused_by_user'],'Human steering no longer authorizes publication.')
    current=server.SimulationManager.compatibility();require(current==seal['compatibility'],'Sealed current compatibility changed.')
    require(re.fullmatch('[0-9a-f]{40}',source_commit) is not None,'Full source commit required.')
    parent=STOP/'run';inventory=json.loads((STOP/'preservation-completion.json').read_bytes())['files']
    expected={r['name'].replace('\\','/'):r['sha256'] for r in inventory}
    require(len(expected)==2106 and expected['checkpoint.npz']==CP_SHA and files(parent)==expected and files(ROOT/'output/runs'/PARENT_ID)==expected,'Exact complete interrupted parent history differs.')
    payload=(parent/'checkpoint.npz').read_bytes();head=checkpoint.checkpoint_header(io.BytesIO(payload));prior=head['compatibility']
    require(head['time_myr']==105 and head['manifest']['run_id']==PARENT_ID and head['manifest']['next_output_myr']==106 and head['manifest']['frame_count']==54,'Accepted boundary/scheduler differs.')
    require(all(prior[k]==current[k] for k in prior if k!='auxiliary_sources_sha256'),'Engine/backend/NumPy changes are unauthorized.')
    require({n for n in prior['auxiliary_sources_sha256'] if prior['auxiliary_sources_sha256'][n]!=current['auxiliary_sources_sha256'].get(n)}==set(seal['changed_helpers']) and set(current['auxiliary_sources_sha256'])-set(prior['auxiliary_sources_sha256'])==set(seal['added_helpers']),'Runtime source delta differs.')
    source_bytes={n:(ROOT/n).read_bytes() for n in current['auxiliary_sources_sha256']};engine=(ROOT/'tectonics.py').read_bytes()
    require(hashlib.sha256(engine).hexdigest()==current['engine_sha256'] and all(hashlib.sha256(b).hexdigest()==current['auxiliary_sources_sha256'][n] for n,b in source_bytes.items()),'Current source bytes differ.')
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==source_commit,'Source provenance must name exact current commit.')
    archive=subprocess.check_output(['git','archive','--format=tar',source_commit,'--','tectonics.py',*source_bytes],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        committed={m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in tar if m.isfile()}
    require(committed=={'tectonics.py':current['engine_sha256'],**current['auxiliary_sources_sha256']},'Commit does not bind every actual future runtime source byte.')
    require(review['source_commit']==source_commit,'Independent review does not bind intended source commit.')
    original_init=native_engine.Simulation.__init__;original_step=native_engine.Simulation.step;original_snapshot=native_engine.Simulation.snapshot
    def forbidden(*a,**k):raise AssertionError('Boundary publication forbids constructors/evolution/sampling.')
    native_engine.Simulation.__init__=forbidden;native_engine.Simulation.step=forbidden;native_engine.Simulation.snapshot=forbidden
    try:
        original,saved=checkpoint.read_checkpoint(io.BytesIO(payload),prior,native_engine.Simulation)
        before=_fingerprint(vars(original));require(len(before['arrays'])==477,'Preexisting typed array count differs.')
        derived=deepcopy(original)
        migration=numerical_accuracy.migrate(derived,boundary_myr=105.,parent_checkpoint_sha256=CP_SHA,authorization_sha256=AUTH_SHA)
        check=deepcopy(vars(derived));newfields=set(check)-set(vars(original))
        require(newfields==set(seal['policy_object_fields']),'New policy object fields differ.')
        for name in newfields:check.pop(name)
        for state in ('structure','trace_structure'):
            require(set(check[state])-set(vars(original)[state])==set(seal['policy_column_fields']),'New policy column fields differ.')
            for name in seal['policy_column_fields']:check[state].pop(name)
        require(_fingerprint(check)==before,'A preexisting array/state/RNG/config/cache/clock changed at import.')
        migrated=_fingerprint(vars(derived))
        _verify_recursive_history(lp(parent),saved)
        run_id=run_id or datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
        require(re.fullmatch('[A-Za-z0-9_-]{1,80}',run_id) is not None and run_id!=PARENT_ID,'Invalid fresh child identity.')
        destination=ROOT/'output/runs'/run_id;stage=ROOT/'output/runs'/('.fix-'+uuid.uuid4().hex[:8])
        require(destination.resolve().is_relative_to(ROOT/'output/runs') and stage.resolve().is_relative_to(ROOT/'output/runs') and not destination.exists() and not stage.exists(),'Fresh isolated publication paths required.')
        manifest=deepcopy(saved)
        inherited={n:h for n,h in expected.items() if re.fullmatch(r'frame_\d+\.(json|npz)',n)}
        manifest.update(run_id=run_id,created=datetime.now(timezone.utc).isoformat(),state='paused',can_resume=True,source_commit=source_commit,branch_origin=dict(run_id=PARENT_ID,time_myr=105.,checkpoint_file='inherited-parent/checkpoint.npz',checkpoint_sha256=CP_SHA,source_transition_kind=KIND),source_transition=dict(kind=KIND,version=1,receipt='source-transition.json',epoch_myr=105.,parent_run_id=PARENT_ID,inherited_frame_count=54,inherited_frame_files_sha256=inherited,inherited_source_compatibility=prior,boundary_kind='human_interrupted_durable_accepted_checkpoint',scope='Repaired contact area/quality solve, exact geometry reuse and explicitly declared future numerical/material policy.'),numerical_accuracy_transition=migration,frame_sources=[dict(index=r['index'],time_myr=r['time_myr'],run_id=PARENT_ID,source_manifest='inherited-parent/manifest.json',source_manifest_sha256=expected['manifest.json'],engine_sha256=prior['engine_sha256'],auxiliary_sources_sha256=prior['auxiliary_sources_sha256'],files_sha256={suffix:expected[f"frame_{r['index']:04d}{suffix}"] for suffix in ('.npz','.json')},inherited=True,parent_frame_source=deepcopy(saved.get('frame_sources',[])[i]) if i<len(saved.get('frame_sources',[])) else None) for i,r in enumerate(saved['frames'])],**deepcopy(current))
        for name in ('error','resume_reason','recovery'):manifest.pop(name,None)
        stage.mkdir();shutil.copytree(lp(parent),lp(stage/'inherited-parent'))
        require(files(stage/'inherited-parent')==expected,'Copied parent bytes differ.')
        for p,h,n in refs:shutil.copy2(p,stage/n);require(sha(stage/n)==h,'Evidence copy differs.')
        (stage/'engine.py').write_bytes(engine)
        for n,b in source_bytes.items():(stage/n).write_bytes(b)
        for n in ('initial.json','config.json',*inherited):shutil.copy2(parent/n,stage/n)
        checkpoint.write_checkpoint(stage/'checkpoint.npz',derived,manifest,current)
        shutil.copy2(stage/'checkpoint.npz',stage/'checkpoint.source-boundary.npz')
        cp=sha(stage/'checkpoint.npz')
        for n in ('checkpoint.npz','checkpoint.source-boundary.npz'):
            restored,embedded=checkpoint.read_checkpoint(stage/n,current,native_engine.Simulation)
            require(_fingerprint(vars(restored))==migrated and embedded==manifest,'Constructorless child readback differs.')
        require(all(sha(stage/n)==h for n,h in inherited.items()),'Inherited frame bytes differ.')
        _verify_recursive_history(lp(stage),manifest)
        require(files(parent)==expected and files(ROOT/'output/runs'/PARENT_ID)==expected,'Parent changed during publication.')
        require(all(sha(ROOT/n)==h for n,h in current['auxiliary_sources_sha256'].items()),'Source changed during publication.')
        receipt=dict(kind=KIND,version=1,child_run_id=run_id,source_commit=source_commit,parent_run_id=PARENT_ID,parent_source_commit='3e190ee7a070b5607c559e28d2573effb16d2276',parent_checkpoint_sha256=CP_SHA,parent_manifest_sha256=expected['manifest.json'],parent_files_sha256=expected,parent_file_count=2106,producer_sha256=sha(__file__),authority_sha256=AUTH_SHA,parent_preservation_sha256=PRES_SHA,source_seal_sha256=source_seal_sha256,repair_review_sha256=repair_review_sha256,private_complete106_receipt_sha256=private_receipt_sha256,child_checkpoint_sha256=cp,permanent_source_boundary_sha256=cp,parent_compatibility=prior,child_compatibility=current,preexisting_state_fingerprint=before['sha256'],derived_state_fingerprint=migrated['sha256'],preexisting_array_count=477,derived_array_count=len(migrated['arrays']),preexisting_state_exact=True,entire_typed_state_unchanged=False,declared_added_object_fields=sorted(newfields),declared_added_column_fields=seal['policy_column_fields'],numerical_migration=migration,boundary_kind='human_interrupted_durable_accepted_checkpoint',physical_time_advanced_myr=0.,model_constructed=False,model_stepped=False,geometry_rebuilt_at_import=False,inherited_frame_count=54,next_output_myr=106.,historical_frames_rewritten=False,created_at_utc=datetime.now(timezone.utc).isoformat())
        write_new(stage/'source-transition.json',receipt);write_new(stage/'manifest.json',manifest)
        require(json.loads((stage/'manifest.json').read_bytes())==manifest and sha(stage/'checkpoint.source-boundary.npz')==cp,'Manifest/source-boundary readback differs.')
        require((REV/'watch-state.json').read_bytes()==watch_raw and json.loads(watch_raw)['phase']=='repairing_authorized_solver_recovery','Later human steering changed before publication.')
        require(all(sha(p)==h for p,h,_ in refs),'Immutable authority/review/validation changed before publication.')
        stage.rename(destination)
        require(files(destination/'inherited-parent')==expected and sha(destination/'checkpoint.source-boundary.npz')==cp,'Published readback differs.')
        return dict(run_id=run_id,path=str(destination),receipt_path=str(destination/'source-transition.json'),receipt_sha256=sha(destination/'source-transition.json'),checkpoint_source_boundary_sha256=cp)
    finally:
        native_engine.Simulation.__init__=original_init;native_engine.Simulation.step=original_step;native_engine.Simulation.snapshot=original_snapshot
