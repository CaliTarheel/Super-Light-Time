"""Validate the immutable receipt chain; no model imports or file mutations.

All small receipt hashes are rechecked. Historical checkpoint hashes are carried
from guardian-verified receipts; only the selected checkpoint is read by export.
"""
import hashlib
import json
import math
from pathlib import Path
import re

RUN='20261007-coarse-burial-speed-history'
POLICY='budgeted_history_1800s_per_myr_v1'

def require(ok,message):
    if not ok:raise ValueError(message)

def numeric(value):
    return type(value) in (int,float) and math.isfinite(value)

def token(value):
    require(isinstance(value,str) and re.fullmatch('[0-9a-f]{32}',value),'Invalid interval token')
    return value

def digest(value):
    require(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value),'Invalid content digest')
    return value

def identity(value):
    require(isinstance(value,dict) and set(value)=={'pid','creation_time_100ns'} and all(type(v) is int and v>0 for v in value.values()),'Invalid process identity')
    return value

def budget(value,dt=1.):
    fields={'policy','dt_myr','started','deadline','physical_deadline','force_refinement_deadline'}
    require(isinstance(value,dict) and set(value)==fields and value['policy']==POLICY,'Budget schema/policy differs')
    require(all(numeric(value[k]) for k in fields-{'policy'}) and value['started']>0 and value['dt_myr']==dt,'Budget values differ')
    start=value['started']
    require(value['deadline']==start+1800.*dt and value['physical_deadline']==start+1680.*dt and value['force_refinement_deadline']==start+900.*dt,'Budget clock or allowance changed')
    return value

def reference(value,path):
    require(isinstance(value,dict) and Path(value['path']).resolve()==Path(path).resolve(),'Artifact path differs')
    digest(value['sha256'])
    return value

def parse(raw):
    def invalid(value):raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(raw,parse_constant=invalid)

def select(binding,launch,read_bytes,exists):
    run=Path(binding['run_directory']);guardian=Path(binding['guardian_directory'])
    control=digest(binding['control_sha256']);child=identity(launch['child']);owner=identity(launch['guardian'])
    require(launch['run_id']==RUN and launch['control_sha256']==control,'Launch binding differs')
    current=token(launch['initial_interval_token']);clock=budget(launch['initial_budget'])
    source=121.;previous=None;links=[];frames=[];seen=set();receipts=[]
    checkpoint=dict(path=str(run/'checkpoint.npz'),sha256=digest(binding['source_boundary_checkpoint_sha256']),time_myr=121.)
    final_path=guardian/'guardian-completion.json';incomplete=None
    if exists(final_path):
        raw=read_bytes(final_path);final=parse(raw)
        require(final['kind']=='burial_successor_owned_interval_guardian' and final['run_id']==RUN and final['control_sha256']==control and final.get('child')==child and final.get('guardian')==owner,'Final guardian identity differs')
        incomplete=final.get('incomplete_interval',{}).get('token')
        if incomplete is not None:token(incomplete)
        receipts.append(dict(path=str(final_path),sha256=hashlib.sha256(raw).hexdigest(),role='final_guardian'))
    stop='awaiting_both_acknowledgments'
    while current is not None:
        token(current);require(current not in seen,'Receipt-chain cycle');seen.add(current)
        if current==incomplete:
            stop='same_token_marked_incomplete';break
        gp=guardian/('accepted-interval-'+current+'.json');ap=run/'numerical-budget'/current/'accepted.json';cp=ap.with_name('completion.json')
        closed_path=guardian/('closed-interval-'+current+'.json')
        transition_path=guardian/('transitioned-interval-'+current+'.json')
        if not exists(gp) or not exists(ap) or not exists(closed_path) or not exists(transition_path):break
        gr=read_bytes(gp);ar=read_bytes(ap);zr=read_bytes(closed_path);g=parse(gr);a=parse(ar);z=parse(zr);tr=read_bytes(transition_path);t=parse(tr)
        for item,kind in ((g,'guardian_accepted_interval'),(a,'server_observed_guardian_acceptance')):
            require(item.get('kind')==kind and item.get('version')==1 and item.get('run_id')==RUN and item.get('token')==current and item.get('budget')==clock,'Acceptance identity/clock differs')
        require(g['control_sha256']==control and g['child']==child and g['guardian']==owner,'Guardian ownership differs')
        require(g['previous_acceptance_token']==previous and g['source_time_myr']==source and g['destination_time_myr']==source+1 and g['requested_dt_myr']==1.,'Noncontiguous physical-time chain')
        reference(a['guardian_acceptance'],gp);require(a['guardian_acceptance']['sha256']==hashlib.sha256(gr).hexdigest(),'Guardian receipt hash differs')
        reference(g['completion'],cp);require(a['completion']==g['completion'],'Server completion reference differs')
        cr=read_bytes(cp);require(hashlib.sha256(cr).hexdigest()==g['completion']['sha256'],'Completion receipt hash differs');c=parse(cr)
        require(c.get('kind')=='lite_budgeted_history_interval' and c.get('version')==2 and c.get('run_id')==RUN and c.get('token')==current and c.get('budget')==clock,'Completion identity/clock differs')
        require(c.get('source_time_myr')==source and c.get('destination_time_myr')==source+1 and c.get('requested_dt_myr')==1. and c.get('interval_and_required_output_completed') is True and c.get('error') is None,'Incomplete physical interval')
        require(z.get('kind')=='guardian_closed_interval' and z.get('version')==1 and z.get('run_id')==RUN and z.get('token')==current and z.get('budget')==clock and z.get('control_sha256')==control and z.get('source_time_myr')==source and z.get('destination_time_myr')==source+1,'Guardian closure identity/clock differs')
        require(z.get('guardian_acceptance')==a['guardian_acceptance'],'Closure guardian reference differs')
        reference(z['server_acceptance'],ap);require(z['server_acceptance']['sha256']==hashlib.sha256(ar).hexdigest(),'Server acknowledgment hash differs')
        require(t.get('kind')=='guardian_interval_transition' and t.get('version')==1 and t.get('run_id')==RUN and t.get('token')==current and t.get('budget')==clock and t.get('control_sha256')==control and t.get('source_time_myr')==source and t.get('destination_time_myr')==source+1,'Guard transition identity/clock differs')
        reference(t['closed_interval'],closed_path);require(t['closed_interval']['sha256']==hashlib.sha256(zr).hexdigest(),'Guardian closure hash differs')
        times=[c.get('completed_at_perf_counter'),g.get('accepted_at_perf_counter'),a.get('observed_at_perf_counter'),z.get('closed_at_perf_counter'),t.get('transitioned_at_perf_counter')]
        require(all(numeric(x) for x in times) and clock['started']<=times[0]<=times[1]<=times[2]<=times[3]<=times[4]<clock['deadline'],'Late or reordered acceptance')
        dest=source+1;require(dest<=binding['target_time_myr'],'Interval exceeds history target')
        ref=reference(c['checkpoint'],cp.with_name('checkpoint.npz'))
        require(ref.get('time_myr')==dest and g['checkpoint']==ref and a['checkpoint']==ref and z.get('checkpoint')==ref,'Accepted checkpoint reference differs')
        due=c['frames'];require(isinstance(due,list) and due==g['frames'] and len(due)==int(dest%2==0),'Due-frame count differs')
        if due:
            row=due[0];index=62+int((dest-122)/2)
            require(row['index']==index and row['time_myr']==dest,'Frame cadence/index differs')
            for extension in ('npz','json'):reference(row[extension],run/('frame_%04d.%s'%(index,extension)))
        issued=g['next_interval_token'];issued_clock=g['next_budget']
        nxt=t['next_interval_token'];following=t['next_budget']
        if dest==binding['target_time_myr']:
            require(nxt is None and following is None and issued is None and issued_clock is None,'Completed target has another interval')
        else:
            token(issued);budget(issued_clock);require(issued_clock['started']==times[1],'Next clock was renewed or detached')
            require((nxt is None and following is None) or (nxt==issued and following==issued_clock),'Transition issued a different next clock')
        for p,raw,role in ((gp,gr,'guardian_acceptance'),(ap,ar,'server_observation'),(cp,cr,'completion'),(closed_path,zr,'guardian_closure'),(transition_path,tr,'guard_transition')):receipts.append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),role=role))
        links.append(dict(token=current,source_time_myr=source,destination_time_myr=dest,checkpoint=ref,guardian_acceptance_sha256=hashlib.sha256(gr).hexdigest(),server_observation_sha256=hashlib.sha256(ar).hexdigest(),guardian_closure_sha256=hashlib.sha256(zr).hexdigest(),guard_transition_sha256=hashlib.sha256(tr).hexdigest()))
        frames.extend(due);checkpoint=ref;source=dest;previous=current;current=nxt;clock=following
        if current is None:stop='target_complete' if dest==binding['target_time_myr'] else 'closed_without_next_interval'
    return dict(checkpoint=checkpoint,frames=frames,links=links,receipts=receipts,accepted_checkpoint_myr=source,stop_reason=stop,incomplete_token=incomplete)
