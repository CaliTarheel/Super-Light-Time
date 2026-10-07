"""Finite continental large-igneous-province source events.

Sources are prescribed geographic 48-sided spherical footprints. Deposits are
finite-volume native face averages, attached to their host crust. Intrusive and
extrusive provenance uses the existing single-density compensated column law;
these are not separate resolved magma chambers, lava layers or mantle plumes.
Unsupported ocean/arc area and full columns reject their share explicitly.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re
import zlib
import numpy as np
import crustal_structure as columns
import native_spreading as clipping

VERSION = 1
MAX_EVENTS = 16
COHORT_FIELDS = ('lip_intrusive', 'lip_extrusive', 'lip_heat')
ARRAY_FIELDS = {'material_lip_intrusive_km_per_reference_km2': np.float64,
                'material_lip_extrusive_km_per_reference_km2': np.float64,
                'material_lip_heat_m': np.float64,
                'material_lip_thermal_support_m': np.float64}


def _number(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError('LIP '+name+' must be a finite number.')
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError('LIP '+name+' is outside its documented bounds.')
    return value


def normalize(config=None):
    config = {} if config is None else deepcopy(config)
    if not isinstance(config, dict) or set(config)-{'version','enabled','seed','generated_count','events'}:
        raise ValueError('Invalid LIP configuration fields.')
    version = config.get('version', VERSION)
    if isinstance(version, bool) or version != VERSION:
        raise ValueError('Unsupported LIP configuration version.')
    enabled = config.get('enabled', False)
    if not isinstance(enabled, bool): raise ValueError('LIP enabled must be boolean.')
    seed = _number(config.get('seed', 37001), 'seed', 0, 2**32-1)
    count = _number(config.get('generated_count', 0), 'generated_count', 0, MAX_EVENTS)
    if seed != int(seed) or count != int(count): raise ValueError('LIP seed/count must be integers.')
    events = config.get('events', [])
    if not isinstance(events, list) or len(events)+count > MAX_EVENTS:
        raise ValueError('Too many LIP events.')
    rows, ids = [], set()
    limits = dict(lon_deg=(-180,180),lat_deg=(-90,90),radius_km=(100,2000),
        start_myr=(0,1000),duration_myr=(.05,30),volume_km3=(1,1e8),
        intrusive_fraction=(0,1),thermal_support_m=(0,2000),cooling_myr=(1,200))
    for event in events:
        if not isinstance(event,dict) or set(event) != {'id',*limits}:
            raise ValueError('Every authored LIP needs its complete source specification.')
        uid = event['id']
        if not isinstance(uid,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',uid) or uid in ids:
            raise ValueError('LIP event IDs must be distinct portable strings.')
        ids.add(uid)
        row = {'id':uid, **{key:_number(event[key],key,*bounds) for key,bounds in limits.items()}}
        row['lon_deg'] = (row['lon_deg']+180)%360-180
        rows.append(row)
    return dict(version=VERSION,enabled=enabled,seed=int(seed),generated_count=int(count),events=rows)


def enabled(s):
    return getattr(s,'lip_version',0) == VERSION


def initialize(s):
    config = normalize(s.config.get('lip_events'))
    if not config['enabled']: return
    if hasattr(s,'lip_version'): raise ValueError('LIP state may only be initialized once.')
    if not hasattr(s,'material_surface'): raise ValueError('LIPs require native continental columns.')
    rng = np.random.default_rng(np.random.SeedSequence([config['seed'], 0x4c4950]))
    events = deepcopy(config['events'])
    eligible = np.flatnonzero(np.isin(s.kind,(1,2)) & (np.asarray(getattr(s,'parcel_arc_id',np.zeros(len(s.mass)))) == 0))
    if config['generated_count'] and not len(eligible):
        raise ValueError('Generated continental LIPs need original continental material.')
    weights = np.asarray(s.material_surface['area_km2'])[eligible]
    duration = float(s.config.get('duration_myr',300))
    for index in range(config['generated_count']):
        at = int(rng.choice(eligible,p=weights/weights.sum()))
        point = s.pos[at]
        uid = 'generated-'+str(index+1)
        if any(row['id']==uid for row in events): raise ValueError('Generated LIP ID collides with authored ID.')
        # The site is provisional: a generated source re-chooses its ground when
        # it erupts, because this one belongs to t=0 geography. See _place_generated.
        events.append(dict(id=uid,generated=True,
            lon_deg=float(np.degrees(np.arctan2(point[1],point[0]))),
            lat_deg=float(np.degrees(np.arcsin(np.clip(point[2],-1,1)))),
            radius_km=float(rng.uniform(400,800)),start_myr=float(rng.uniform(0,max(0,duration-5))),
            duration_myr=float(rng.uniform(1,5)),volume_km3=float(rng.uniform(1e6,3e6)),
            intrusive_fraction=.7,thermal_support_m=450.,cooling_myr=35.))
    s.lip_version = VERSION
    s.lip_seed = config['seed']
    s.lip_events = []
    for row in events:
        row.setdefault('generated',False)
        row.update(supplied_volume_km3=0.,intrusive_volume_km3=0.,extrusive_volume_km3=0.,
            pending_volume_km3=0.,rejected_volume_km3=0.,rejected_unoccupied_km3=0.,
            rejected_capacity_km3=0.,started=False,ended=False,
            source_policy=('fixed geographic 48-edge polygon; native face-average deposits; a generated '
                           'source fixes its centre at eruption, an authored one at its given coordinates'),
            density_policy='single-density crust; intrusive/extrusive are cumulative provenance')
        s.lip_events.append(row)
    s.lip_last_time_myr = float(s.t)
    ensure_fields(s)


def ensure_fields(s):
    if not enabled(s): return
    count = len(s.lip_events)
    for prefix,n,state in (('parcel_',len(s.mass),s.structure),('trace_',len(s.trace_patch),s.trace_structure)):
        for field in COHORT_FIELDS:
            name = prefix+field
            old = getattr(s,name,np.zeros((0,count)))
            if old.ndim != 2 or old.shape[1] != count or len(old)>n:
                raise ValueError('LIP provenance lost its native material alignment.')
            if len(old)<n: old=np.concatenate((old,np.zeros((n-len(old),count))),axis=0)
            setattr(s,name,old)
        if 'lip_heat_m' not in state: state['lip_heat_m']=np.zeros(n)
        if len(state['lip_heat_m']) != n: raise ValueError('LIP thermal columns lost alignment.')


def next_transition(s):
    if not enabled(s): return None
    transitions=[value for row in s.lip_events for value in
                 (row['start_myr'],row['start_myr']+row['duration_myr']) if value>s.t+1e-10]
    return min(transitions) if transitions else None


def pulse_fraction(row,start,end):
    """Exactly integrate a finite uniform source pulse, even between snapshots."""
    a=row['start_myr']; b=a+row['duration_myr']
    return max(0.,min(end,b)-max(start,a))/(b-a)


def _eligible_continental(s):
    """Original continental material, excluding accreted arcs: the same test
    `initialize` uses to choose a generated source's host."""
    arcs=np.asarray(getattr(s,'parcel_arc_id',np.zeros(len(s.mass))))
    return np.flatnonzero(np.isin(s.kind,(1,2)) & (arcs==0))


def _place_generated(s,row):
    """Give a generated source its ground at eruption, not at t=0.

    `initialize` picks a continental parcel and freezes its longitude and
    latitude, but a generated start time is uniform over the whole run. Plates
    move, so a source that fires late erupts at a point its intended continent
    left long ago: in the 500 Myr SEP20T configuration, 7 of 12 generated events
    fire more than 100 Myr after their site was chosen, at a median of 255 Myr,
    which is thousands of kilometres of drift. The generator deliberately
    selects continental material and then, for most events, erupts in whatever
    happens to lie there later.

    Re-choosing the host when the source opens keeps the declared intent -- a
    continental flood basalt on continental crust -- for every event rather than
    only the early ones. The draw is deterministic in the run's own LIP seed and
    the event id, so it is reproducible and independent of arrival order. This
    is a worldbuilding closure: a real plume is fixed in the mantle and the
    plate drifts over it, which this does not model. Authored events keep the
    coordinates they were given.
    """
    if not row.get('generated',str(row.get('id','')).startswith('generated-')):
        return
    eligible=_eligible_continental(s)
    if not len(eligible):
        return
    weights=np.asarray(s.material_surface['area_km2'])[eligible]
    total=float(weights.sum())
    if not np.isfinite(total) or total<=0.:
        return
    rng=np.random.default_rng(np.random.SeedSequence(
        [int(getattr(s,'lip_seed',0)),0x4c495031,int(zlib.crc32(str(row['id']).encode('utf-8')))]))
    point=s.pos[int(rng.choice(eligible,p=weights/total))]
    row['lon_deg']=float(np.degrees(np.arctan2(point[1],point[0])))
    row['lat_deg']=float(np.degrees(np.arcsin(np.clip(point[2],-1,1))))
    row['placed_myr']=float(s.t)


def footprint(row,radius=6371.):
    lon,lat=np.radians([row['lon_deg'],row['lat_deg']])
    center=np.array([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])
    east=np.array([-np.sin(lon),np.cos(lon),0.]); north=np.cross(center,east)
    angle=np.arange(48)*2*np.pi/48; distance=row['radius_km']/radius
    return np.cos(distance)*center+np.sin(distance)*(np.cos(angle)[:,None]*east+np.sin(angle)[:,None]*north)


def _source_areas(s,row):
    """Partition the finite source once; overlapping sheets cannot double it."""
    context=clipping.prepare_material(s.material_surface)
    polygon=footprint(row,context['radius_km']); area=clipping._area(polygon,context)
    result=np.zeros(len(s.mass)); candidates=clipping._candidates(polygon,context)
    # Highest resolved face-average column owns overlapping exposed footprint.
    # Tie by persistent face ID; exact spatially varying sheet order is deferred.
    heights=columns.elevation(s.structure)
    order=sorted(candidates,key=lambda i:(-float(heights[i]),int(s.parcel_patch[i])))
    blockers=[]
    arc=np.asarray(getattr(s,'parcel_arc_id',np.zeros(len(s.mass))))
    for index in order:
        intersection=clipping._intersect_polygon(polygon,context['triangles'][index])
        if clipping._area(intersection,context)<=1e-8: continue
        pieces=clipping._subtract_polygons(intersection,blockers,context)
        visible=math.fsum(clipping._area(piece,context) for piece in pieces)
        blockers.append(intersection)
        if s.kind[index] in (1,2) and arc[index]==0: result[index]=visible
    if result.sum()>area+max(1e-6,area*1e-10): raise ValueError('LIP footprint was counted more than once.')
    return result,area


def _trace_faces(s):
    order=np.argsort(s.parcel_patch); found=np.searchsorted(s.parcel_patch[order],s.trace_patch)
    if np.any(found>=len(order)) or np.any(s.parcel_patch[order[found]]!=s.trace_patch):
        raise ValueError('LIP markers require their actual containing material face.')
    return order[found]


def evolve_columns(s,dt,*,trace=False):
    """Apply the source once to physical columns; markers track the same source."""
    n=len(s.trace_patch) if trace else len(s.mass)
    zero=np.zeros(n)
    if not enabled(s): return dict(magmatic_m=zero,thermal_m=zero)
    ensure_fields(s)
    state=s.trace_structure if trace else s.structure
    prefix='trace_' if trace else 'parcel_'
    before=columns.elevation(state)
    heat=getattr(s,prefix+'lip_heat')
    durations=np.array([row['cooling_myr'] for row in s.lip_events])
    heat *= np.exp(-float(dt)/durations)[None,:]
    thickness_delta=np.zeros((n,len(s.lip_events)))
    heat_delta=np.zeros_like(thickness_delta)
    if trace:
        if getattr(s,'_lip_step',{}).get('time_myr') != float(s.t):
            raise ValueError('LIP trace source requires the same committed physical source step.')
        face=_trace_faces(s)
        requested=s._lip_step['thickness_delta'][face]
        capacity=np.maximum(columns.MAX_THICKNESS_KM-state['thickness_km'],0.)
        scale=np.minimum(1.,np.divide(capacity,requested.sum(axis=1),out=np.ones(n),where=requested.sum(axis=1)>0))
        thickness_delta=requested*scale[:,None]
        heat_delta=s._lip_step['heat_delta'][face]
    else:
        end=float(s.t);start=end-float(dt)
        if abs(start-s.lip_last_time_myr)>1e-8: raise ValueError('LIP source step was skipped or repeated.')
        area=np.asarray(s.material_surface['area_km2'])
        capacity=np.maximum(columns.MAX_THICKNESS_KM-state['thickness_km'],0.)*area
        for column,row in enumerate(s.lip_events):
            fraction=pulse_fraction(row,start,end); supplied=row['volume_km3']*fraction
            if end>=row['start_myr'] and not row['started']:
                # Once, at onset and before the first deposit, so the footprint
                # below and the onset record both use the erupting centre rather
                # than the t=0 one. Keyed on onset, not on the first positive
                # supply: the engine splits steps at next_transition, so the step
                # that lands exactly on start_myr supplies nothing yet still
                # marks the source started, and a supply-keyed test never fires.
                _place_generated(s,row)
            if supplied>0:
                overlap,source_area=_source_areas(s,row)
                requested=supplied*overlap/source_area
                admitted=np.minimum(requested,capacity); capacity-=admitted
                thickness_delta[:,column]=admitted/area
                occupied=float(requested.sum()); actual=float(admitted.sum())
                unoccupied=max(0.,supplied-occupied); rejected_capacity=max(0.,occupied-actual)
                row['supplied_volume_km3']+=supplied
                row['intrusive_volume_km3']+=actual*row['intrusive_fraction']
                row['extrusive_volume_km3']+=actual*(1-row['intrusive_fraction'])
                row['rejected_unoccupied_km3']+=unoccupied
                row['rejected_capacity_km3']+=rejected_capacity
                row['rejected_volume_km3']+=unoccupied+rejected_capacity
                # Analytic decay of uniform source heating within this interval.
                a=max(start,row['start_myr']);b=min(end,row['start_myr']+row['duration_myr']);tau=row['cooling_myr']
                heat_integral=tau*(np.exp(-(end-b)/tau)-np.exp(-(end-a)/tau))/row['duration_myr']
                admitted_fraction=np.divide(admitted,requested,out=np.zeros(n),where=requested>0)
                heat_delta[:,column]=row['thermal_support_m']*heat_integral*(overlap/area)*admitted_fraction
            if end>=row['start_myr'] and not row['started']:
                row['started']=True
                if hasattr(s,'_record'): s._record('lip_onset','A finite continental large igneous province source began.',('lip_onset',row['id']),details=deepcopy(row))
            if end>=row['start_myr']+row['duration_myr'] and not row['ended']:
                row['ended']=True
                if hasattr(s,'_record'): s._record('lip_end','A large igneous province source ended; its material and cooling history persist.',('lip_end',row['id']),details=deepcopy(row))
        s.lip_last_time_myr=end
        s._lip_step=dict(time_myr=end,thickness_delta=thickness_delta.copy(),heat_delta=heat_delta.copy())
    state['thickness_km']+=thickness_delta.sum(axis=1)
    import crust_inventory
    crust_inventory.add(state, thickness_delta.sum(axis=1)*state['area_factor'])
    after_magma=columns.elevation(state)
    magmatic=after_magma-before
    for column,row in enumerate(s.lip_events):
        volume=thickness_delta[:,column]*state['area_factor']
        getattr(s,prefix+'lip_intrusive')[:,column]+=volume*row['intrusive_fraction']
        getattr(s,prefix+'lip_extrusive')[:,column]+=volume*(1-row['intrusive_fraction'])
    state['added_volume_km_per_reference_km2']+=thickness_delta.sum(axis=1)*state['area_factor']
    state['magmatic_uplift_m']+=magmatic
    heat+=heat_delta
    state['lip_heat_m']=heat.sum(axis=1)
    thermal=columns.elevation(state)-after_magma
    state['thermal_uplift_m']+=np.maximum(thermal,0.)
    state['thermal_subsidence_m']+=np.maximum(-thermal,0.)
    if not trace: validate_state(s)
    return dict(magmatic_m=magmatic,thermal_m=thermal)


def validate_state(s):
    if not enabled(s): return
    for i,row in enumerate(s.lip_events):
        supplied=row['supplied_volume_km3'];intrusive=row['intrusive_volume_km3'];extrusive=row['extrusive_volume_km3']
        accounted=intrusive+extrusive+row['pending_volume_km3']+row['rejected_volume_km3']
        if not np.isclose(supplied,accounted,rtol=1e-11,atol=1e-6): raise ValueError('LIP source volume ledger does not close.')
        for field,expected in (('lip_intrusive',intrusive),('lip_extrusive',extrusive)):
            actual=float(s.mass@getattr(s,'parcel_'+field)[:,i])
            if not np.isclose(actual,expected,rtol=2e-11,atol=1e-6): raise ValueError('LIP material provenance volume was not conserved.')


def snapshot(s):
    if not enabled(s): return {}
    ensure_fields(s);validate_state(s)
    cold=dict(s.structure,lip_heat_m=np.zeros(len(s.mass)))
    return dict(lip_version=VERSION,lip_seed=s.lip_seed,lip_events=deepcopy(s.lip_events),
        material_lip_intrusive_km_per_reference_km2=s.parcel_lip_intrusive.copy(),
        material_lip_extrusive_km_per_reference_km2=s.parcel_lip_extrusive.copy(),
        material_lip_heat_m=s.parcel_lip_heat.copy(),
        material_lip_thermal_support_m=columns.elevation(s.structure)-columns.elevation(cold),
        lip_provenance_meaning='Cumulative emplaced volumes per reference area; retained origin history, not remaining uneroded lava thickness.')


def project_grid(s,material):
    """Review grid samples the selected exposed native material, never nearest land."""
    if not enabled(s): return {}
    ensure_fields(s)
    index=np.asarray(material);land=index>=0
    deposited=np.zeros(len(index),np.float32);thermal=np.zeros(len(index),np.float32)
    total=(s.parcel_lip_intrusive+s.parcel_lip_extrusive).sum(axis=1)
    equivalent=total*s.mass/s.material_surface['area_km2']
    cold=dict(s.structure,lip_heat_m=np.zeros(len(s.mass)))
    support=columns.elevation(s.structure)-columns.elevation(cold)
    deposited[land]=equivalent[index[land]];thermal[land]=support[index[land]]
    return dict(lip_deposited_km=deposited,lip_thermal_support_m=thermal)


def validate_frame(frame):
    version=frame.get('lip_version',0)
    found=set(ARRAY_FIELDS).intersection(frame)
    if isinstance(version,bool) or version not in (0,VERSION): raise ValueError('Unsupported saved LIP version.')
    if version==0:
        if found or 'lip_events' in frame: raise ValueError('LIP arrays require their schema version.')
        return
    rows=frame.get('lip_events'); n=len(frame['material_faces'])
    if not isinstance(rows,list) or len(rows)>MAX_EVENTS or found!=set(ARRAY_FIELDS): raise ValueError('Saved LIP history is incomplete.')
    ids=[row['id'] for row in rows]
    if len(set(ids))!=len(ids): raise ValueError('Saved LIP origins must remain distinct.')
    for key in ARRAY_FIELDS:
        values=np.asarray(frame[key]); expected=(n,) if key.endswith('thermal_support_m') else (n,len(rows))
        if values.shape!=expected or not np.isfinite(values).all() or np.any(values < -1e-8): raise ValueError('Invalid native LIP arrays.')
    reference=np.asarray(frame['material_reference_area_km2'])
    for i,row in enumerate(rows):
        supplied=row['supplied_volume_km3']; accounted=sum(row[key] for key in ('intrusive_volume_km3','extrusive_volume_km3','pending_volume_km3','rejected_volume_km3'))
        if not np.isclose(supplied,accounted,rtol=1e-11,atol=1e-6): raise ValueError('Saved LIP source ledger does not close.')
        for field,term in (('intrusive','intrusive_volume_km3'),('extrusive','extrusive_volume_km3')):
            actual=float(reference@np.asarray(frame['material_lip_'+field+'_km_per_reference_km2'])[:,i])
            if not np.isclose(actual,row[term],rtol=2e-11,atol=1e-6): raise ValueError('Saved LIP material provenance does not close.')
