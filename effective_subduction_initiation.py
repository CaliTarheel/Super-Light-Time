"""Opt-in induced initiation along finite, weakening ocean-bearing faults.

This Lite closure records actual accepted shortening and along-strike slip;
it is not a resolved brittle/thermal failure calculation. Old ocean age is an
incoming control-volume cooling-age approximation. No historical damage,
elapsed time or missing slab inventory is reconstructed on activation.
"""
from copy import copy, deepcopy
import math
import numpy as np
from ridge_geometry import rotate

VERSION=1
POLICY='induced oceanic transform/oblique fault initiation from accepted local shortening and shear'
DEFAULTS=dict(version=VERSION,enabled=False,minimum_convergence_km_myr=2.,
    minimum_consecutive_myr=10.,minimum_shortening_km=100.,minimum_shear_slip_km=100.,
    minimum_ocean_age_myr=20.,minimum_ocean_fraction=.8,minimum_length_km=300.)


def normalize(raw=None):
    if raw is None:return deepcopy(DEFAULTS)
    if not isinstance(raw,dict) or set(raw)-set(DEFAULTS):raise ValueError('Unknown Lite initiation controls.')
    value=dict(DEFAULTS,**raw)
    if type(value['version']) is not int or value['version']!=VERSION or type(value['enabled']) is not bool:
        raise ValueError('Lite initiation needs original version1 and boolean enabled controls.')
    for key in set(DEFAULTS)-{'version','enabled'}:
        number=value[key]
        if (isinstance(number,(bool,np.bool_)) or not isinstance(number,(int,float,np.number))
                or not np.isfinite(number) or number<=0.):raise ValueError('Lite initiation thresholds must be finite and positive.')
        value[key]=float(number)
    if value['minimum_ocean_fraction']>1.:raise ValueError('Lite incoming ocean fraction must be in (0,1].')
    return value


def enabled(s):
    raw=getattr(s,'config',{}).get('effective_subduction',{}).get('initiation')
    state=getattr(s,'effective_subduction_initiation',None)
    if raw is None or not normalize(raw)['enabled']:
        if state is not None:raise ValueError('Lite initiation state has no matching enabled policy.')
        return False
    import effective_subduction
    if not effective_subduction.enabled(s):raise ValueError('Lite initiation requires effective subduction.')
    if (not isinstance(state,dict) or state.get('version')!=VERSION or type(state.get('version')) is not int
            or state.get('policy')!=POLICY or state.get('parameters')!=normalize(raw)
            or not isinstance(state.get('candidates'),list) or not isinstance(state.get('births'),list)
            or not isinstance(state.get('events'),list) or type(state.get('next_candidate_id')) is not int
            or state['next_candidate_id']<1):raise ValueError('Invalid Lite initiation checkpoint metadata.')
    activation,epoch=state.get('activation_myr'),state.get('epoch_myr')
    if (any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float,np.number)) or not np.isfinite(v)
            for v in (activation,epoch)) or not 0.<=activation<=epoch<=float(s.t)):
        raise ValueError('Invalid Lite initiation activation or accepted clock.')
    ids=[]
    for row in state['candidates']:
        if (not isinstance(row,dict) or type(row.get('id')) is not int or row['id']<1
                or row.get('phase') not in ('weakening','pending','born','retired')
                or any(type(row.get(k)) is not int or row[k]<=0 for k in ('incoming_plate_uid','overriding_plate_uid'))
                or row.get('incoming_plate_uid')==row.get('overriding_plate_uid')):
            raise ValueError('Invalid Lite initiation candidate identity.')
        ids.append(row['id'])
        for key in ('created_myr','last_seen_myr','consecutive_myr','shortening_km','shear_slip_km','length_km'):
            value=row.get(key)
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.number)) or not np.isfinite(value) or value<0.:
                raise ValueError('Invalid Lite initiation candidate history.')
        if not activation<=row['created_myr']<=row['last_seen_myr']<=epoch:raise ValueError('Lite initiation invents candidate history outside accepted time.')
        for name in ('segments_start','segments_end'):
            xyz=np.asarray(row.get(name),float)
            if xyz.ndim!=2 or xyz.shape[1]!=3 or not len(xyz) or not np.isfinite(xyz).all() or np.any(abs(np.linalg.norm(xyz,axis=1)-1.)>1e-10):
                raise ValueError('Lite initiation requires finite unit candidate fault geometry.')
        if len(row['segments_start'])!=len(row['segments_end']):raise ValueError('Lite initiation candidate endpoints must align.')
        if np.any(np.linalg.norm(np.cross(row['segments_start'],row['segments_end']),axis=1)<=1e-12):
            raise ValueError('Lite initiation candidate finite arcs must be nondegenerate.')
    if ids!=sorted(set(ids)) or any(ident>=state['next_candidate_id'] for ident in ids):
        raise ValueError('Lite initiation candidate IDs must remain unique and ordered.')
    return True


def upgrade(s,parameters=None):
    """Add only configuration and zero-history policy metadata at this epoch."""
    if enabled(s):return deepcopy(s.effective_subduction_initiation)
    import effective_subduction
    effective_subduction.validate_compatibility(s)
    if not effective_subduction.enabled(s):raise ValueError('Lite initiation activation requires existing effective subduction.')
    controls=normalize(dict(enabled=True) if parameters is None else parameters)
    if not controls['enabled']:raise ValueError('Explicit Lite initiation activation must be enabled.')
    config=deepcopy(s.config);config['effective_subduction']=dict(config['effective_subduction'],initiation=controls)
    state=dict(version=VERSION,policy=POLICY,parameters=deepcopy(controls),activation_myr=float(s.t),
        epoch_myr=float(s.t),candidates=[],next_candidate_id=1,births=[],events=[],
        retrospective_time_myr=0.,retrospective_shortening_km=0.,retrospective_shear_slip_km=0.,
        physical_state_changed=False,
        provenance='Only subsequent accepted finite-fault observations contribute; no inherited ocean damage or slab history.',
        ocean_age_scope='Existing incoming control-volume cooling age; wet length uses actual native material exclusion.')
    subject=copy(s);subject.config=config;subject.effective_subduction_initiation=state
    enabled(subject)
    s.config=config;s.effective_subduction_initiation=state
    return deepcopy(state)


def initialize(s):
    """Fresh opt-in worlds use the same zero-history activation contract."""
    controls=normalize(s.config['effective_subduction'].get('initiation'))
    if not controls['enabled']:return False
    # A newly authored configuration is not active metadata until this call.
    raw=s.config['effective_subduction']['initiation']
    subject=copy(s);subject.config=deepcopy(s.config);subject.config['effective_subduction'].pop('initiation')
    upgrade(subject,raw)
    s.config=subject.config;s.effective_subduction_initiation=subject.effective_subduction_initiation
    return True


def _pieces(s):
    """Current finite fault pieces; never use a parent's averaged convergence."""
    contour=getattr(s,'native_boundary_geometry',None)
    if not isinstance(contour,dict):raise ValueError('Lite initiation requires resolved native finite fault geometry.')
    a,b,n=(np.asarray(contour.get(key),float) for key in ('segments_start','segments_end','segment_normals'))
    parent=np.asarray(contour.get('contact_index'))
    if (a.ndim!=2 or a.shape[1]!=3 or b.shape!=a.shape or n.shape!=a.shape
            or parent.shape!=(len(a),) or parent.dtype.kind not in 'iu'
            or np.any(parent<0) or np.any(parent>=len(s.ba))
            or not all(np.isfinite(v).all() for v in (a,b,n))
            or any(np.any(abs(np.linalg.norm(v,axis=1)-1.)>1e-10) for v in (a,b,n))):
        raise ValueError('Lite initiation finite native pieces are malformed or unaligned.')
    import effective_subduction
    import trench_history
    ids=trench_history._match(s,include_shutdown=False)
    reserved=np.zeros(len(ids),bool)
    for trace in s.trench_systems:
        if not trace.get('effective_subduction') or trace['phase'] in ('shutdown','joined'):continue
        reserved|=supported_parents(s,trace,ids==trace['id'])
    ages=np.asarray(s.age,float)
    if not np.isfinite(ages).all() or np.any(ages<0.):raise ValueError('Lite initiation requires finite nonnegative control ocean ages.')
    middle=(a+b);middle/=np.linalg.norm(middle,axis=1)[:,None]
    length=6371.*np.arctan2(np.linalg.norm(np.cross(a,b),axis=1),np.sum(a*b,axis=1))
    n=n-middle*np.sum(n*middle,axis=1)[:,None]
    magnitude=np.linalg.norm(n,axis=1)
    if np.any(magnitude<=1e-12):raise ValueError('Lite fault normals need a resolved midpoint tangent.')
    n/=magnitude[:,None]
    import native_subduction
    if native_subduction.enabled(s):
        contexts={int(owner):native_subduction._material(s,int(owner)) for owner in np.unique(np.r_[s.bp,s.bq])}
        water=[]
        for first,last,edge,extent in zip(a,b,parent,length):
            fractions=[]
            for owner in (s.bp[edge],s.bq[edge]):
                wet=0.
                for low,high in native_subduction.exact.ocean_intervals(first,last,contexts[int(owner)]):
                    left=(1-low)*first+low*last;left/=np.linalg.norm(left)
                    right=(1-high)*first+high*last;right/=np.linalg.norm(right)
                    wet+=6371.*math.atan2(float(np.linalg.norm(np.cross(left,right))),float(left@right))
                fractions.append(float(np.clip(wet/max(extent,1e-30),0.,1.)))
            water.append(tuple(fractions))
    else:
        water=[(float(s.crust[s.ba[edge]]==0),float(s.crust[s.bb[edge]]==0)) for edge in parent]
    relative=np.cross(np.asarray(s.omega)[np.asarray(s.bq)[parent]]-np.asarray(s.omega)[np.asarray(s.bp)[parent]],middle)*6371.
    normal=np.sum(relative*n,axis=1)
    shear=np.linalg.norm(relative-normal[:,None]*n,axis=1)
    records=[]
    for piece,edge in enumerate(parent):
        edge=int(edge);p,q=int(s.bp[edge]),int(s.bq[edge])
        if (reserved[edge] or not s.active[p] or not s.active[q] or p==q or length[piece]<=1e-10
                or int(s.bcode[edge]) not in (2,3,4)):
            continue
        records.append(dict(index=piece,parent=edge,start=a[piece],end=b[piece],mid=middle[piece],
            length=float(length[piece]),normal_speed=float(normal[piece]),shear=float(shear[piece]),
            slots=(p,q),uids=(int(s.plate_uid[p]),int(s.plate_uid[q])),
            water=water[piece],
            age=(float(ages[s.ba[edge]]),float(ages[s.bb[edge]]))))
    return records


def _groups(records):
    parents=list(range(len(records)));seen={}
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    for i,row in enumerate(records):
        pair=tuple(sorted(row['uids']))
        for xyz in (row['start'],row['end']):
            key=(pair,*tuple(np.round(xyz,12)))
            if key in seen:
                first,second=find(i),find(seen[key]);parents[max(first,second)]=min(first,second)
            else:seen[key]=i
    groups={}
    for i,row in enumerate(records):groups.setdefault(find(i),[]).append(row)
    return list(groups.values())


def _measure(group,controls,incoming=None):
    length=math.fsum(row['length'] for row in group)
    if length<controls['minimum_length_km']:return None
    pair=tuple(sorted(group[0]['uids']))
    options=[]
    for uid in pair:
        # Parent-scale force/intake cannot qualify dry or hot sibling support.
        if any(row['water'][row['uids'].index(uid)]<controls['minimum_ocean_fraction']
                or row['age'][row['uids'].index(uid)]<controls['minimum_ocean_age_myr'] for row in group):continue
        water=math.fsum(row['length']*row['water'][row['uids'].index(uid)] for row in group)/length
        age=math.fsum(row['length']*row['water'][row['uids'].index(uid)]*row['age'][row['uids'].index(uid)] for row in group)
        age/=max(water*length,1e-30)
        if water>=controls['minimum_ocean_fraction'] and age>=controls['minimum_ocean_age_myr']:
            options.append((age,-uid,uid,water))
    if incoming is None:
        if not options:return None
        age,_,incoming,water=max(options)
    else:
        options=[row for row in options if row[2]==incoming]
        if not options:return None
        age,_,_,water=options[0]
    over=pair[1] if pair[0]==incoming else pair[0]
    return dict(incoming_plate_uid=incoming,overriding_plate_uid=over,length_km=length,
        mean_ocean_age_myr=age,mean_ocean_fraction=water,
        minimum_convergence_km_myr=min(-row['normal_speed'] for row in group),
        mean_convergence_km_myr=math.fsum(row['length']*max(-row['normal_speed'],0.) for row in group)/length,
        mean_shear_km_myr=math.fsum(row['length']*row['shear'] for row in group)/length)


def _distance(group,row):
    """Distance to finite carried arcs, including their real endpoints."""
    a,b=np.asarray(row['segments_start']),np.asarray(row['segments_end'])
    normals=np.cross(a,b);normals/=np.linalg.norm(normals,axis=1)[:,None]
    arc=np.arctan2(np.linalg.norm(np.cross(a,b),axis=1),np.sum(a*b,axis=1))
    worst=0.
    for record in group:
        point=record['mid'];dot=normals@point
        foot=point-dot[:,None]*normals;foot/=np.linalg.norm(foot,axis=1)[:,None]
        left=np.arccos(np.clip(np.sum(foot*a,axis=1),-1.,1.));right=np.arccos(np.clip(np.sum(foot*b,axis=1),-1.,1.))
        on=left+right<=arc+128*np.finfo(float).eps
        distance=np.minimum(np.arccos(np.clip(a@point,-1.,1.)),np.arccos(np.clip(b@point,-1.,1.)))
        distance[on]=np.arcsin(np.minimum(abs(dot[on]),1.))
        worst=max(worst,float(distance.min())*6371.)
    return worst


def _covered(points,a,b,radius):
    """Each point projects inside real finite support, with a lateral allowance."""
    normals=np.cross(a,b);normals/=np.linalg.norm(normals,axis=1)[:,None]
    arc=np.arctan2(np.linalg.norm(np.cross(a,b),axis=1),np.sum(a*b,axis=1))
    for point in points:
        dot=normals@point;foot=point-dot[:,None]*normals
        magnitude=np.linalg.norm(foot,axis=1)
        foot/=np.maximum(magnitude[:,None],1e-30)
        left=np.arctan2(np.linalg.norm(np.cross(foot,a),axis=1),np.sum(foot*a,axis=1))
        right=np.arctan2(np.linalg.norm(np.cross(foot,b),axis=1),np.sum(foot*b,axis=1))
        on=(magnitude>1e-12)&(left+right<=arc+128*np.finfo(float).eps)
        lateral=6371.*np.arcsin(np.minimum(abs(dot),1.))
        if not np.any(on & (lateral<=radius)):return False
    return True


def _same_support(group,row,radius):
    """Bidirectional finite coverage prevents halo growth and borrowed history."""
    a,b=np.asarray(row['segments_start']),np.asarray(row['segments_end'])
    c,d=np.array([r['start'] for r in group]),np.array([r['end'] for r in group])
    old_mid=(a+b);old_mid/=np.linalg.norm(old_mid,axis=1)[:,None]
    new_mid=(c+d);new_mid/=np.linalg.norm(new_mid,axis=1)[:,None]
    return (_covered(np.r_[c,d,new_mid],a,b,radius)
        and _covered(np.r_[a,b,old_mid],c,d,radius))


def _complete_parents(s,group):
    parent=np.asarray(s.native_boundary_geometry['contact_index'])
    selected={row['index'] for row in group}
    return all(set(np.flatnonzero(parent==edge))<=selected for edge in {row['parent'] for row in group})


def supported_parents(s,row,select):
    """New force sources cannot recruit untrained parents through a match halo."""
    law=row.get('effective_subduction',{})
    if not law.get('initiation_finite_support',False):return select
    if row['id']!=law.get('support_origin_trench_id'):
        raise ValueError('Initiated finite support needs an explicit owner/topology transfer before loading a descendant trace.')
    geometry=np.asarray(row['geometry_xyz'],float)
    if geometry.shape!=(law.get('support_endpoint_count'),3) or len(geometry)%2:
        raise ValueError('Initiated finite force support lost its carried endpoint representation.')
    a,b=geometry[::2],geometry[1::2]
    if np.any(np.linalg.norm(np.cross(a,b),axis=1)<=1e-12):
        raise ValueError('Initiated finite force support needs nondegenerate arcs.')
    contour=s.native_boundary_geometry;parent=np.asarray(contour['contact_index'])
    starts,ends=np.asarray(contour['segments_start']),np.asarray(contour['segments_end'])
    radius=max(120.,1.8*__import__('trench_history')._spacing(s));result=np.asarray(select,bool).copy()
    for edge in np.flatnonzero(result):
        pieces=parent==edge
        if not np.any(pieces):result[edge]=False;continue
        c,d=starts[pieces],ends[pieces];mid=c+d;mid/=np.linalg.norm(mid,axis=1)[:,None]
        if not _covered(np.r_[c,d,mid],a,b,radius):result[edge]=False
    return result


def prepare(s):
    """Pure current observations; never age a candidate or activate a load."""
    if not enabled(s):return []
    controls=s.effective_subduction_initiation['parameters']
    return [dict(_measure(group,controls),parent_indices=sorted(set(row['parent'] for row in group)))
        for group in _groups(_pieces(s)) if _measure(group,controls) is not None]


def update(s,dt):
    """Commit exactly one accepted clock interval and any pending births."""
    if not enabled(s):return
    state=s.effective_subduction_initiation;epoch=float(state['epoch_myr']);now=float(s.t)
    if isinstance(dt,(bool,np.bool_)) or not np.isscalar(dt) or not np.isfinite(dt) or dt<0.:
        raise ValueError('Lite initiation needs finite nonnegative accepted elapsed time.')
    if now==epoch:
        return
    elapsed=now-epoch
    if elapsed<0. or not math.isclose(elapsed,float(dt),rel_tol=0.,abs_tol=64*np.finfo(float).eps*max(now,float(dt),1.)):
        raise ValueError('Lite initiation clock does not match the accepted motion interval.')
    import trench_history
    proposed=deepcopy(state);controls=state['parameters'];slots=trench_history._slots(s)
    live=[row for row in proposed['candidates'] if row['phase']=='weakening']
    for row in live:
        owner=slots.get(row['overriding_plate_uid'])
        if owner is None or row['incoming_plate_uid'] not in slots:
            row.update(phase='retired',reason='owner_changed');continue
        rotation=np.asarray(s.omega[owner])*elapsed
        row['segments_start']=rotate(np.asarray(row['segments_start']),rotation).tolist()
        row['segments_end']=rotate(np.asarray(row['segments_end']),rotation).tolist()
    groups=[group for group in _groups(_pieces(s)) if _measure(group,controls) is not None]
    radius=max(120.,1.8*trench_history._spacing(s))
    matches={i:[] for i in range(len(groups))};old_matches={row['id']:[] for row in live if row['phase']=='weakening'}
    for i,group in enumerate(groups):
        pair=set(group[0]['uids'])
        for row in live:
            if (row['phase']=='weakening' and pair=={row['incoming_plate_uid'],row['overriding_plate_uid']}
                    and _measure(group,controls,row['incoming_plate_uid']) is not None and _same_support(group,row,radius)):
                matches[i].append(row);old_matches[row['id']].append(i)
    accepted=set();births=[]
    for i,group in enumerate(groups):
        choices=matches[i]
        reuse=len(choices)==1 and len(old_matches[choices[0]['id']])==1
        if reuse:
            row=choices[0];measurement=_measure(group,controls,row['incoming_plate_uid']);accepted.add(row['id'])
            row['shear_slip_km']+=measurement['mean_shear_km_myr']*elapsed
            if measurement['minimum_convergence_km_myr']>=controls['minimum_convergence_km_myr']:
                row['consecutive_myr']+=elapsed
                row['shortening_km']+=measurement['mean_convergence_km_myr']*elapsed
            else:row['consecutive_myr']=row['shortening_km']=0.
        else:
            measurement=_measure(group,controls)
            row=dict(id=proposed['next_candidate_id'],phase='weakening',created_myr=now,
                consecutive_myr=0.,shortening_km=0.,shear_slip_km=0.)
            proposed['next_candidate_id']+=1;proposed['candidates'].append(row)
        row.update(measurement,last_seen_myr=now,
            segments_start=[r['start'].tolist() for r in group],segments_end=[r['end'].tolist() for r in group])
        if (row['consecutive_myr']>=controls['minimum_consecutive_myr']
                and row['shortening_km']>=controls['minimum_shortening_km']
                and row['shear_slip_km']>=controls['minimum_shear_slip_km']
                and _complete_parents(s,group)):
            row['phase']='pending';births.append((row,group))
    for row in live:
        if row['phase']=='weakening' and row['id'] not in accepted:
            row.update(phase='retired',reason='finite_fault_unmatched_or_ambiguous_topology')
    proposed['epoch_myr']=now
    # Publish only after all measurements and clock checks. Native callers
    # own the adaptive deep-copy/restore transaction including trace births.
    s.effective_subduction_initiation=proposed
    for candidate,group in births:
        edges=np.array(sorted(set(row['parent'] for row in group)),int)
        down=np.full(len(s.ba),slots[candidate['incoming_plate_uid']],int)
        row=trench_history._birth(s,edges,down,inherited=False)
        points=np.array([r['mid'] for r in group]);weights=np.array([r['length'] for r in group])
        center=np.sum(points*weights[:,None],axis=0);center/=np.linalg.norm(center)
        endpoints=np.array([[r['start'],r['end']] for r in group]).reshape(-1,3)
        row.update(geometry_xyz=endpoints.tolist(),center=center.tolist(),length_km=float(weights.sum()),anchor_gap_km=0.)
        force=float(s.effective_subduction_settings['force_n_per_m'])
        row['effective_subduction']=dict(version=1,force_n_per_m=force,activation_pending=True,
            initiation_candidate_id=candidate['id'],eligible_myr=now,
            initiation_finite_support=True,support_origin_trench_id=row['id'],support_endpoint_count=len(endpoints),
            provenance='Accepted induced finite-fault gates; no slab inventory, retrospective intake or startup clock.')
        row.update(phase='initiating',maturity=0.)
        candidate['trench_id']=row['id']
        event=dict(time_myr=now,candidate_id=candidate['id'],trench_id=row['id'],kind='eligible_pending',
            incoming_plate_uid=candidate['incoming_plate_uid'],overriding_plate_uid=candidate['overriding_plate_uid'],
            force_n_per_m=force,consecutive_myr=candidate['consecutive_myr'],
            shortening_km=candidate['shortening_km'],shear_slip_km=candidate['shear_slip_km'],
            force_activation='before next instantaneous solve; no force or intake at eligibility snapshot')
        proposed['births'].append(deepcopy(event));proposed['events'].append(event)
        if hasattr(s,'_record'):
            s._record('effective_subduction_initiation',
                'Accepted local oceanic shortening and shear made a finite induced trench eligible; traction starts at the next solved source.',
                ('effective_subduction_initiation',VERSION,candidate['id']),
                plates=(slots[candidate['incoming_plate_uid']],slots[candidate['overriding_plate_uid']]),
                xyz=row['center'],details=dict(event,mechanism=POLICY,parameters=deepcopy(controls)))


def activate_pending(s):
    """Source boundary only: activate before the next solve under rollback."""
    if not enabled(s):return False
    import effective_subduction
    import trench_history
    slots=trench_history._slots(s);changed=False
    for row in s.trench_systems:
        law=row.get('effective_subduction',{})
        if not law.get('activation_pending',False):continue
        if row['downgoing_plate_uid'] not in slots or row['overriding_plate_uid'] not in slots:
            trench_history._shutdown(s,row,'plate_owner_lost');law['activation_pending']=False;continue
        law.update(activation_pending=False,activated_myr=float(s.t))
        row.update(phase='mature',maturity=1.)
        candidate=next(r for r in s.effective_subduction_initiation['candidates'] if r['id']==law['initiation_candidate_id'])
        candidate['phase']='born'
        s.effective_subduction_initiation['events'].append(dict(kind='source_activated',time_myr=float(s.t),
            candidate_id=candidate['id'],trench_id=row['id'],force_n_per_m=law['force_n_per_m']))
        changed=True
    if changed:effective_subduction.prepare(s)
    return changed


def snapshot(s):
    if not enabled(s):return {}
    state=s.effective_subduction_initiation
    return dict(effective_subduction_initiation_version=VERSION,effective_subduction_initiation=deepcopy(state),
        effective_subduction_initiation_diagnostics=dict(parameters=deepcopy(state['parameters']),
            activation_myr=state['activation_myr'],accepted_epoch_myr=state['epoch_myr'],
            weakening_candidates=sum(r['phase']=='weakening' for r in state['candidates']),
            pending_trenches=sum(r['phase']=='pending' for r in state['candidates']),
            activated_trenches=sum(r['phase']=='born' for r in state['candidates']),
            birth_count=len(state['births']),
            activation_scope='Pending eligible traces carry no force or intake until the next solved source interval.'))
