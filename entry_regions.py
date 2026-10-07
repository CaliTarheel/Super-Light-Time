"""Face-resolved material entry regions and moving-hinge histories.

An opt-in native experiment can register an approaching buoyant face from its
finite inherited trench front while its entry energy is still zero. The
reference-linear potential integrates the entered fraction within that face.
The registry still assigns whole material-face identities; ocean/stack
subregions and overriding-owner transfers are not resolved. Existing
inventories remain authoritative. This is plain checkpoint data, never a saved
FrozenPotential or another executable object.
"""
from copy import deepcopy,copy
import numpy as np
import continental_entry as energy
import finite_entry_arc
from ridge_geometry import rotate

FIELD='parcel_entry_region'
AUTOMATIC_VERSION=1


def automatic_enabled(s):
    """Automatic admission is an explicit experiment, never a checkpoint default."""
    version=getattr(s,'automatic_entry_version',0)
    if (isinstance(version,(bool,np.bool_)) or not isinstance(version,(int,np.integer))
            or version not in (0,AUTOMATIC_VERSION)):
        raise ValueError('Unsupported automatic continental-entry version.')
    return int(version)==AUTOMATIC_VERSION


def _edge_approach_candidates(triangles, center, radius, eligible, midpoint, normal,
                              half_length, down_omega, over_omega, dt):
    """Predict which faces touch one finite trench edge in the overrider frame.

    Inputs are geometry and rotations only; state lookup and region registration
    belong to the caller. The broad distance bound only prunes candidates. The
    signed current and future finite-arc integrals decide admission.
    """
    guard_m=1.
    guard_angle=guard_m/6371e3
    left,right,_,_=finite_entry_arc.endpoint_planes(normal,midpoint,half_length,6371.)
    distance=np.arccos(np.clip(center[eligible]@midpoint,-1.,1.))*6371.
    motion_km=6371.*(np.linalg.norm(down_omega)+np.linalg.norm(over_omega))*dt
    within=(np.ones(len(eligible),bool) if dt==0. else
            distance<=radius[eligible]+half_length+motion_km+guard_m/1000.)
    candidates=eligible[within]
    if not len(candidates):return []
    future=(triangles[candidates] if dt==0. else
        rotate(rotate(triangles[candidates].reshape(-1,3),down_omega*dt),
               -over_omega*dt).reshape(-1,3,3))
    approaching_faces=[]
    for face_index,d,after in zip(candidates,distance[within],future):
        q=6371e3*np.arcsin(np.clip(triangles[face_index]@normal,-1.,1.))
        if q.max(initial=-np.inf)+motion_km*1000.+guard_m<=0.:
            continue
        along_left=triangles[face_index]@left
        along_right=triangles[face_index]@right
        active=finite_entry_arc.integrate(q,along_left,along_right)[4]
        if active>0.:
            raise ValueError('Incoming continental material crossed an inherited trench before zero-energy admission.')
        future_q=6371e3*np.arcsin(np.clip(after@normal,-1.,1.))
        approaching=finite_entry_arc.integrate(future_q+guard_m,
            after@left+guard_angle,after@right+guard_angle)[4]
        if approaching>0.:
            approaching_faces.append((int(face_index),float(d)))
    return approaching_faces


def _approaching_faces(s,dt,forecast_omega=None):
    """Find incoming material touching a finite inherited trench segment.

    The current entry coordinate must have zero positive area. A solved Euler
    lookahead and a metre-scale geometry guard bound the face that can touch
    the actual finite arc during this source interval. The lookahead rotates
    the incoming face into the overriding plate's frame: an opening face must
    not be admitted merely because its relative speed is large.
    """
    import slab_memory
    import trench_history

    surface=s.material_surface
    vertices=np.asarray(surface['vertices'],float)
    faces=np.asarray(surface['faces'],int)
    triangles=vertices[faces]
    count=len(faces)
    owners=np.asarray(s.parcel_plate,int)
    kinds=np.asarray(s.kind,int)
    if owners.shape!=(count,) or kinds.shape!=(count,):
        raise ValueError('Automatic entry needs aligned material owners and kinds.')
    if not count:
        return []
    membership=np.asarray(getattr(s,FIELD,np.zeros(count,np.int64)))
    if membership.shape!=(count,):
        raise ValueError('Automatic entry membership lost its material alignment.')
    angular=np.asarray(s.omega if forecast_omega is None else forecast_omega,float)
    if angular.shape!=np.shape(s.omega) or not np.isfinite(angular).all():
        raise ValueError('Automatic entry requires aligned finite forecast plate rotations.')
    center=triangles.sum(axis=1)
    center/=np.maximum(np.linalg.norm(center,axis=1)[:,None],1e-30)
    radius=np.max(np.arccos(np.clip(np.einsum('fi,fji->fj',center,triangles),-1.,1.)),axis=1)*6371.
    best=np.full(count,np.inf)
    chosen={}
    slots={int(uid):p for p,uid in enumerate(s.plate_uid) if s.active[p]}
    ids=np.asarray(s.trench_id)
    coupled=trench_history._rupture_governed(s)
    for row in s.trench_systems:
        if row['phase'] in ('shutdown','joined'):
            continue
        if coupled and not trench_history._local_neck_state(row)[1]:
            # A remembered trace with no attached slab is not a physical
            # entry front, even before its lifecycle update shuts it down.
            continue
        down=slots.get(int(row['downgoing_plate_uid']))
        over=slots.get(int(row['overriding_plate_uid']))
        if down is None or over is None:
            continue
        eligible=np.flatnonzero((owners==down)&(kinds>0)&(membership==0))
        if not len(eligible):
            continue
        edges=np.flatnonzero((ids==row['id'])&(np.asarray(s.bl)>0.))
        for edge in edges:
            if {int(s.bp[edge]),int(s.bq[edge])}!={down,over}:
                continue
            midpoint=np.asarray(s.bmid[edge],float)
            midpoint/=np.linalg.norm(midpoint)
            normal=np.asarray(s.bn[edge],float)*(1. if s.bp[edge]==down else -1.)
            normal-=midpoint*np.dot(midpoint,normal)
            normal/=np.linalg.norm(normal)
            half_length=float(s.bl[edge])*.5
            # The geometry predictor has no access to region or plate state.
            for face_index,d in _edge_approach_candidates(triangles,center,radius,eligible,
                    midpoint,normal,half_length,angular[down],angular[over],dt):
                if d<best[face_index]:
                    best[face_index]=d
                    chosen[int(face_index)]=(row,normal.copy(),midpoint.copy(),half_length)
    specs=[]
    for index,(row,normal,midpoint,half_length) in sorted(chosen.items()):
        specs.append(dict(face_ids=np.asarray(s.parcel_patch)[index:index+1].copy(),
            trench_id=int(row['id']),overriding_plate_uid=int(row['overriding_plate_uid']),
            hinge_normal=normal,dip_degrees=float(slab_memory.SUBDUCTION_DIP_DEG),
            finite_midpoint=midpoint,finite_half_length_km=half_length))
    return specs


def admit_approaching(s,dt,*,forecast_omega=None):
    """Atomically register zero-work incoming faces before coupled force solve."""
    if not automatic_enabled(s):
        return dict(enabled=False,admitted_face_ids=[])
    if not np.isscalar(dt) or isinstance(dt,(bool,np.bool_)) or not np.isfinite(dt) or dt<0.:
        raise ValueError('Automatic entry lookahead must be finite and nonnegative.')
    if enabled(s):
        _validate(s)
        if s.continental_entry_regions.get('assignment')!='automatic finite trench front':
            raise ValueError('Automatic entry cannot mix with manually registered regions.')
    specs=_approaching_faces(s,float(dt),forecast_omega)
    if not specs:
        return dict(enabled=True,admitted_face_ids=[])
    if not enabled(s):
        initialize(s,specs)
        s.continental_entry_regions['assignment']='automatic finite trench front'
    else:
        staged=copy(s)
        staged.continental_entry_regions=deepcopy(s.continental_entry_regions)
        membership=np.asarray(getattr(s,FIELD)).copy()
        rows=staged.continental_entry_regions['regions']
        lookup={int(ident):i for i,ident in enumerate(s.parcel_patch)}
        next_id=max((row['id'] for row in rows),default=0)+1
        for spec in specs:
            face_id=int(spec['face_ids'][0]);index=lookup[face_id]
            if membership[index]:
                raise ValueError('Automatic entry cannot register a material face twice.')
            ident=next_id;next_id+=1;membership[index]=ident
            rows.append(dict(id=ident,source_trench_id=spec['trench_id'],
                overriding_plate_uid=spec['overriding_plate_uid'],
                hinge_normal=spec['hinge_normal'].copy(),
                finite_midpoint=spec['finite_midpoint'].copy(),
                finite_half_length_km=spec['finite_half_length_km'],
                dip_degrees=spec['dip_degrees'],created_myr=float(s.t)))
        setattr(staged,FIELD,membership)
        _validate(staged)
        frozen(staged)
        s.continental_entry_regions=staged.continental_entry_regions
        setattr(s,FIELD,membership)
    return dict(enabled=True,admitted_face_ids=[int(spec['face_ids'][0]) for spec in specs],
        rule='finite inherited front, solved lookahead and zero source-epoch entry energy')


def enabled(s):
    state=getattr(s,'continental_entry_regions',None)
    if state is None:return False
    if not isinstance(state,dict) or type(state.get('version')) is not int or state['version']!=1:
        raise ValueError('Unsupported persistent entry-region schema.')
    return True


def ensure_fields(s):
    """Align genuine material births; remeshing must explicitly carry its map."""
    if not enabled(s):return
    state=s.continental_entry_regions
    old=np.asarray(state['face_ids']);now=np.asarray(s.parcel_patch)
    values=np.asarray(getattr(s,FIELD))
    if values.shape!=old.shape or values.dtype.kind not in 'iu':
        raise ValueError('Entry region membership lost its persistent face alignment.')
    if np.array_equal(old,now):return
    if np.any((values>0)&~np.isin(old,now)):
        raise ValueError('Entry-bearing material disappeared without an explicit remesh map.')
    lookup={int(ident):int(value) for ident,value in zip(old,values)}
    membership=np.array([lookup.get(int(ident),0) for ident in now],np.int64)
    setattr(s,FIELD,membership);state['face_ids']=now.copy()


def _validate(s,*,check_epoch=True):
    if not enabled(s):raise ValueError('Persistent entry regions were not explicitly initialized.')
    if (getattr(s,'plate_balance_version',0)!=1 or getattr(s,'plate_resistance_version',0)!=1
            or getattr(s,'material_mechanics_version',0)!=1 or not s.config.get('deforming_regions',1)):
        raise ValueError('Persistent entry cannot run without both passive plate and sheet mechanics.')
    ensure_fields(s);state=s.continental_entry_regions
    ids=np.asarray(s.parcel_patch);membership=np.asarray(getattr(s,FIELD))
    if (not np.array_equal(ids,s.material_surface['face_id']) or len(np.unique(ids))!=len(ids)
            or np.any(membership<0) or not np.isfinite(state['epoch_myr'])):
        raise ValueError('Invalid persistent entry material identities or epoch.')
    if check_epoch and float(state['epoch_myr'])!=float(s.t):
        raise ValueError('Entry hinge and native material epochs disagree.')
    rows={int(row['id']):row for row in state['regions']}
    if len(rows)!=len(state['regions']) or any(ident<=0 for ident in rows) or not set(membership)-{0}<=set(rows):
        raise ValueError('Entry material refers to a missing or repeated region identity.')
    slots={int(uid):i for i,uid in enumerate(s.plate_uid) if s.active[i]}
    import trench_history
    coupled=trench_history._rupture_governed(s)
    sources={int(row['id']):row for row in s.trench_systems} if coupled else {}
    for row in rows.values():
        normal=np.asarray(row['hinge_normal'],float)
        if (normal.shape!=(3,) or not np.isfinite(normal).all() or abs(np.linalg.norm(normal)-1.)>1e-10
                or not np.isfinite(row['dip_degrees']) or not 0.<row['dip_degrees']<90.
                or row['overriding_plate_uid'] not in slots):
            raise ValueError('Entry hinge lost valid geometry or its overriding plate; an explicit transfer is required.')
        selected=membership==row['id']
        if coupled and np.any(selected) and not row.get('source_detached',False):
            source=sources.get(int(row['source_trench_id']))
            if (source is None or source['phase'] in ('shutdown','joined')
                    or not trench_history._local_neck_state(source)[1]):
                raise ValueError('Coupled entry lost its attached source slab without a breakoff handoff.')
        if np.any(np.asarray(s.parcel_plate)[selected]==slots[row['overriding_plate_uid']]):
            raise ValueError('Entry material and hinge acquired the same owner; resolve the internal buried domain explicitly.')
        if 'finite_midpoint' in row or 'finite_half_length_km' in row:
            midpoint=np.asarray(row['finite_midpoint'],float)
            half=float(row['finite_half_length_km'])
            if (midpoint.shape!=(3,) or not np.isfinite(midpoint).all()
                    or abs(np.linalg.norm(midpoint)-1.)>1e-10
                    or abs(np.dot(midpoint,normal))>1e-10
                    or not np.isfinite(half) or not 0.<half<np.pi*6371.):
                raise ValueError('Entry region lost its finite inherited trench arc.')
            energy.validate_finite_front(normal[None],midpoint[None],
                np.array([half]),s.material_surface.get('radius_km',6371.))
    return rows,slots


def initialize(s,regions):
    """Atomically bind explicit whole-face regions to existing trench histories.

    Each specification names face_ids, trench_id, overriding_plate_uid,
    hinge_normal and dip_degrees. The supplied hinge is a local reduced-model
    geometry, not a fitted slab shape or an automatic eligibility decision.
    """
    if enabled(s):raise ValueError('Entry regions are already initialized; histories cannot be reset.')
    if (getattr(s,'plate_balance_version',0)!=1 or getattr(s,'plate_resistance_version',0)!=1
            or getattr(s,'material_mechanics_version',0)!=1 or not s.config.get('deforming_regions',1)):
        raise ValueError('Entry transport requires passive instantaneous plates and native sheet mechanics.')
    staged=copy(s);rows=[]
    ids=np.asarray(s.parcel_patch);lookup={int(ident):i for i,ident in enumerate(ids)}
    membership=np.zeros(len(ids),np.int64)
    trenches={int(row['id']):row for row in s.trench_systems}
    for index,spec in enumerate(regions):
        face_ids=np.asarray(spec['face_ids'])
        if (face_ids.ndim!=1 or face_ids.dtype.kind not in 'iu' or not len(face_ids)
                or len(np.unique(face_ids))!=len(face_ids) or any(int(i) not in lookup for i in face_ids)):
            raise ValueError('Entry initialization needs unique existing material face IDs.')
        selected=np.array([lookup[int(i)] for i in face_ids],int)
        if np.any(membership[selected]):raise ValueError('A material face cannot enter two regions at once.')
        trench=trenches.get(spec['trench_id'])
        if trench is None or trench['overriding_plate_uid']!=spec['overriding_plate_uid']:
            raise ValueError('Entry region must name its existing inherited trench and overriding plate.')
        incoming=np.asarray(s.plate_uid)[np.asarray(s.parcel_plate)[selected]]
        if np.any(incoming!=trench['downgoing_plate_uid']):
            raise ValueError('Entry material must start on the named trench incoming plate.')
        membership[selected]=index+1
        new_row=dict(id=index+1,source_trench_id=int(spec['trench_id']),
            overriding_plate_uid=int(spec['overriding_plate_uid']),
            hinge_normal=np.array(spec['hinge_normal'],float,copy=True),
            dip_degrees=float(spec['dip_degrees']),created_myr=float(s.t))
        if 'finite_midpoint' in spec or 'finite_half_length_km' in spec:
            new_row['finite_midpoint']=np.array(spec['finite_midpoint'],float,copy=True)
            new_row['finite_half_length_km']=float(spec['finite_half_length_km'])
        rows.append(new_row)
    if not rows:raise ValueError('Entry initialization needs at least one explicit material region.')
    staged.continental_entry_regions=dict(version=1,epoch_myr=float(s.t),face_ids=ids.copy(),
        regions=rows,remesh_energy_change_j=0.,assignment='explicit whole material faces')
    setattr(staged,FIELD,membership)
    _validate(staged)
    frozen(staged)  # Reject inadmissible stack/phase/hinge geometry before commit.
    s.continental_entry_regions=staged.continental_entry_regions
    setattr(s,FIELD,membership)


def mark_source_detached(s,trench_id):
    """Retain admitted continental entry work after its source slab breaks off.

    The registered buoyant material keeps its finite hinge and energy history;
    only the requirement for an attached source slab is released. This lets
    the same entry potential drive post-breakoff rebound/exhumation instead of
    deleting stored work or silently reattaching a failed slab.
    """
    if not enabled(s):
        return []
    state=s.continental_entry_regions
    changed=[]
    for row in state['regions']:
        if int(row.get('source_trench_id',-1))!=int(trench_id) or row.get('source_detached',False):
            continue
        row['source_detached']=True
        row['source_detached_myr']=float(s.t)
        changed.append(int(row['id']))
    if changed:
        _validate(s)
    return changed


def specification(s,*,advance_myr=0.):
    rows,slots=_validate(s)
    if not np.isscalar(advance_myr) or isinstance(advance_myr,(bool,np.bool_)) or not np.isfinite(advance_myr) or advance_myr<0.:
        raise ValueError('Entry hinge advance must be finite and nonnegative.')
    membership=getattr(s,FIELD);selected=np.flatnonzero(membership>0)
    if not len(selected):return None
    normals={ident:rotate(np.asarray(row['hinge_normal'])[None],np.asarray(s.omega[slots[row['overriding_plate_uid']]])*advance_myr)[0]
             for ident,row in rows.items()}
    midpoints={ident:(rotate(np.asarray(row['finite_midpoint'])[None],
        np.asarray(s.omega[slots[row['overriding_plate_uid']]])*advance_myr)[0]
        if 'finite_midpoint' in row else np.zeros(3)) for ident,row in rows.items()}
    return dict(face_ids=np.asarray(s.parcel_patch)[selected].copy(),
        hinge_normals=np.array([normals[int(membership[i])] for i in selected]),
        dip_degrees=np.array([rows[int(membership[i])]['dip_degrees'] for i in selected]),
        finite_midpoints=np.array([midpoints[int(membership[i])] for i in selected]),
        finite_half_lengths_km=np.array([rows[int(membership[i])].get('finite_half_length_km',np.inf)
            for i in selected]),
        overriding_plate_uids=np.array([rows[int(membership[i])]['overriding_plate_uid'] for i in selected],np.int64))


def frozen(s,*,advance_myr=0.,coordinate_mode='along_slab_conveyor'):
    spec=specification(s,advance_myr=advance_myr)
    if spec is None:return None
    return energy.FrozenPotential.from_state(s,**{key:spec[key] for key in
        ('face_ids','hinge_normals','dip_degrees','finite_midpoints','finite_half_lengths_km')},
        coordinate_mode=coordinate_mode)


def energy_j(s):
    potential=frozen(s)
    if potential is None:return 0.
    mesh=s.material_surface
    return potential.evaluate(mesh['vertices'],mesh['faces'],potential.volumes,potential.sheets,radius=potential.radius)['energy_j']


def advance_hinges(s,dt):
    """Commit the same upper-plate rotation used in the preceding sheet stage."""
    rows,slots=_validate(s)
    if not np.isscalar(dt) or isinstance(dt,(bool,np.bool_)) or not np.isfinite(dt) or dt<=0.:
        raise ValueError('Entry hinge timestep must be finite and positive.')
    changed={ident:rotate(np.asarray(row['hinge_normal'])[None],np.asarray(s.omega[slots[row['overriding_plate_uid']]])*dt)[0]
             for ident,row in rows.items()}
    changed_midpoints={ident:rotate(np.asarray(row['finite_midpoint'])[None],
        np.asarray(s.omega[slots[row['overriding_plate_uid']]])*dt)[0]
        for ident,row in rows.items() if 'finite_midpoint' in row}
    epoch=float(s.t)+float(dt)
    if not np.isfinite(epoch) or epoch<=s.t:raise ValueError('Entry hinge timestep is not representable.')
    for ident,row in rows.items():
        row['hinge_normal']=changed[ident]
        if ident in changed_midpoints:row['finite_midpoint']=changed_midpoints[ident]
    s.continental_entry_regions['epoch_myr']=epoch


def prepare_remap(s,fields,surface,structure,new_ids):
    """Validate the proposed native category transfer before its transaction."""
    before=energy_j(s);staged=copy(s)
    for name,value in fields.items():setattr(staged,name,value)
    staged.material_surface=surface;staged.structure=structure
    staged.parcel_patch=new_ids;staged.parcel_plate=surface['face_owner']
    staged.continental_entry_regions=deepcopy(s.continental_entry_regions)
    staged.continental_entry_regions['face_ids']=new_ids.copy()
    _validate(staged)
    change=energy_j(staged)-before
    staged.continental_entry_regions['remesh_energy_change_j']+=change
    return staged.continental_entry_regions
