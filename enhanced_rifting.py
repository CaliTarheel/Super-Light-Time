"""Inherited material weakness coupled to realized native sheet deformation.

Weakness is a seeded smooth initial material property, not an applied force or
preexisting extension. Persistent link damage uses measured material-node
separation, including failure/healing through the existing rift lifecycle.
"""
from __future__ import annotations
import numpy as np

VERSION=1
ARRAY_FIELDS={'material_rift_seed_weakness':'float64',
              'material_rift_realized_tension':'float64',
              'material_rift_realized_compression':'float64'}

def normalize(value=None):
    value={} if value is None else value
    if not isinstance(value,dict) or set(value)-{'version','enabled','seed','amplitude','correlation_km','craton_margin_km'}:
        raise ValueError('Unknown enhanced rifting control.')
    out=dict(version=1,enabled=False,seed=37,amplitude=.25,correlation_km=350.,craton_margin_km=150.)
    out.update(value)
    if type(out['version']) is not int or out['version']!=1 or type(out['enabled']) is not bool:
        raise ValueError('Invalid enhanced rifting version or enabled flag.')
    if type(out['seed']) is not int or not 0<=out['seed']<2**32:
        raise ValueError('Enhanced rift seed must be an unsigned32-bit integer.')
    for key,low,high in [('amplitude',0.,.6),('correlation_km',100.,2000.),('craton_margin_km',0.,400.)]:
        v=out[key]
        if isinstance(v,bool) or not isinstance(v,(float,int)) or not np.isfinite(v) or not low<=v<=high:
            raise ValueError('Invalid enhanced rift '+key)
        out[key]=float(v)
    return out

def enabled(s):
    return bool(getattr(s,'config',{}).get('enhanced_rifting',{}).get('enabled',False))

def seeded_weakness(points,config):
    """Fixed spherical wavelengths; independent of query order and face count."""
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
        raise ValueError('Weakness sampling needs finite spherical points.')
    rng=np.random.default_rng(np.random.SeedSequence([config['seed'],0x52494654]))
    directions=rng.normal(size=(24,3));directions/=np.linalg.norm(directions,axis=1)[:,None]
    phase=rng.uniform(-np.pi,np.pi,24)
    wave=6371./config['correlation_km']
    raw=np.sin((points@directions.T)*wave+phase).sum(axis=1)/np.sqrt(12.)
    return config['amplitude']*(.5+.5*np.tanh(raw))

def initialize(s):
    if not enabled(s):return
    s.enhanced_rifting_version=1
    s.parcel_rift_seed_weakness=seeded_weakness(s.pos,s.config['enhanced_rifting'])
    s.parcel_rift_seed_weakness[s.kind!=1]=0.
    s.parcel_rift_realized_tension=np.zeros(len(s.mass))
    s.parcel_rift_realized_compression=np.zeros(len(s.mass))
    s.rift_realized_motion=None

def ensure_fields(s):
    if not enabled(s):return
    for name in ARRAY_FIELDS:
        target=name.replace('material_','parcel_',1)
        old=np.asarray(getattr(s,target,np.empty(0)),float)
        if old.ndim!=1 or len(old)>len(s.mass) or not np.isfinite(old).all() or np.any(old<0):
            raise ValueError('Enhanced rift field lost its material alignment: '+target)
        if name=='material_rift_seed_weakness' and np.any(old>.6+1e-12):
            raise ValueError('Inherited rift weakness exceeds its allowed amplitude.')
        if len(old)<len(s.mass):old=np.r_[old,np.zeros(len(s.mass)-len(old))]
        setattr(s,target,old)

# Share of peak rift damage retained as durable weakness once the damage itself
# heals away. Damage is transient by design -- it decays on a thermal timescale
# and much faster under compression -- so without this a healed rift leaves no
# trace and the next extensional episode starts from virgin strength wherever it
# happens to land.
#
# That is what this run does. After 340 Myr and five systems that reached
# peak_damage 1.0, the correlation between seeded weakness and current rift
# damage was -0.02: the weakness field, computed once at t=0 from spatial noise,
# had no idea where any rift was. Meanwhile 45 of the 60 currently-extending
# links carried damage below 0.1, because tension kept landing on fresh crust
# with nothing drawing it back to the belt that had already failed.
#
# Earth localises the other way round, and inheritance is why: the Atlantic
# opened along the Appalachian-Caledonian suture, the East African Rift follows
# Proterozoic mobile belts. Crust that has broken breaks again.
#
# .5 keeps the ratchet inside the 0.6 amplitude that ensure_fields and the frame
# validator both enforce, given the seeded field already reaches ~0.25. It is a
# maximum, not a sum: a scar records the worst damage that crust ever suffered.
RIFT_SCAR_FRACTION = .5


def accumulate_scar(s, parcel_damage):
    """Ratchet healed rift damage into durable inherited weakness.

    Applies only to ordinary continental material. Cratons are excluded, exactly
    as the seeded field excludes them at initialization: they already carry a 5x
    strength bonus precisely because they are meant to survive, and scarring them
    would undo the one structure built to be durable.
    """
    if not enabled(s) or not hasattr(s, 'parcel_rift_seed_weakness'):
        return
    ensure_fields(s)
    damage = np.asarray(parcel_damage, float)
    if damage.shape != (len(s.mass),):
        raise ValueError('Rift scar accumulation needs material-aligned damage.')
    scar = np.clip(RIFT_SCAR_FRACTION*np.nan_to_num(damage, nan=0.), 0., RIFT_SCAR_FRACTION)
    scar[s.kind != 1] = 0.
    np.maximum(s.parcel_rift_seed_weakness, scar, out=s.parcel_rift_seed_weakness)


def material_viscosity(s):
    if not enabled(s):return None
    ensure_fields(s)
    import progressive_rifting
    damage=progressive_rifting.material_fields(s)['rift_damage']
    return resistance(s,damage)


def craton_anchors(surface, width_km=150.):
    """Rigid cores inside strong, deformable craton rims of physical width.

    Distance travels along edges of each connected cratonic material body.
    Narrow bodies retain at least their inner half-depth as an anchor; bodies
    without resolved interior vertices stay fully rigid. This is a mechanical
    distinction only: no crust class, reference mass, or breakup protection is
    changed. Zero width recovers the previous fully rigid craton treatment.
    """
    import heapq
    points=np.asarray(surface['vertices'],float)
    faces=np.asarray(surface['faces']); kind=np.asarray(surface['face_kind'])
    if isinstance(width_km,bool) or not np.isfinite(width_km) or not 0<=width_km<=400.:
        raise ValueError('Craton margin width must be from zero to400 km.')
    craton=np.zeros(len(points),bool); craton[np.unique(faces[kind==2])]=True
    diagnostic=dict(model='rigid craton cores with strong mobile rims',requested_width_km=float(width_km),
                    craton_vertices=int(craton.sum()),mobile_rim_vertices=0,components=[])
    if not craton.any() or width_km==0:
        return craton,diagnostic
    cf=faces[kind==2]
    edges,count=np.unique(np.sort(np.concatenate((cf[:,[0,1]],cf[:,[1,2]],cf[:,[2,0]])),axis=1),axis=0,return_counts=True)
    boundary=np.zeros(len(points),bool); boundary[np.unique(edges[count==1])]=True
    radius=float(surface.get('radius_km',6371.))
    length=radius*np.arctan2(np.linalg.norm(np.cross(points[edges[:,0]],points[edges[:,1]]),axis=1),
                            np.sum(points[edges[:,0]]*points[edges[:,1]],axis=1))
    adjacency={int(v):[] for v in np.flatnonzero(craton)}
    for (a,b),distance in zip(edges,length):
        adjacency[int(a)].append((int(b),float(distance)));adjacency[int(b)].append((int(a),float(distance)))
    remaining=set(adjacency); rigid=craton.copy()
    while remaining:
        start=min(remaining); component=[]; pending=[start];remaining.remove(start)
        while pending:
            at=pending.pop();component.append(at)
            for other,_ in adjacency[at]:
                if other in remaining:remaining.remove(other);pending.append(other)
        component=np.asarray(sorted(component),int)
        seeds=component[boundary[component]]
        if not len(seeds):continue
        distances={int(v):float('inf') for v in component};queue=[]
        for v in seeds:distances[int(v)]=0.;heapq.heappush(queue,(0.,int(v)))
        while queue:
            distance,at=heapq.heappop(queue)
            if distance!=distances[at]:continue
            for other,length in adjacency[at]:
                proposed=distance+length
                if proposed<distances[other]:distances[other]=proposed;heapq.heappush(queue,(proposed,other))
        depth=np.array([distances[int(v)] for v in component]); deepest=float(depth.max(initial=0.))
        effective=min(float(width_km),.5*deepest)
        # A roundoff-scale tie must not make a rotated equivalent core mobile.
        mobile=depth<effective-1e-10
        rigid[component[mobile]]=False
        diagnostic['components'].append(dict(vertices=len(component),maximum_depth_km=deepest,
            effective_width_km=effective,mobile_rim_vertices=int(mobile.sum()),rigid_core_vertices=int((~mobile).sum())))
    diagnostic['mobile_rim_vertices']=int(np.count_nonzero(craton & ~rigid))
    return rigid,diagnostic

def resistance(s,damage):
    """Crust thickness is not a temperature or lithosphere-thickness proxy.

    Actual column thickness already enters GPE. Composition, documented heat
    support and material damage supply the reduced constitutive modifiers.
    """
    heat=s.structure['rift_heat_m']+s.structure.get('lip_heat_m',0.)
    strength=(1.+5.*(s.kind==2))/(1.+1.2*s.suture+heat/800.+2.5*np.asarray(damage))
    return np.clip(strength*(1.-s.parcel_rift_seed_weakness),.12,12.)


# Integrated strength of cold continental lithosphere, i.e. the depth integral of
# its yield envelope: 1e13 N/m for a 100-125 km column at geological strain rate
# (Kusznir & Park 1987; Ranalli 1995 give 5e12-2e13 N/m across the plausible
# geotherm and rheology band, so this is a factor-of-two quantity, not a measured
# one). The dimensionless field `resistance` above modulates it: 0.12 for
# thoroughly damaged, hot, sutured crust and 12 for an intact craton. Pairing the
# two turns one worldbuilding closure into a force per metre that a plate-scale
# balance and a suture weld can both quote.
F_REF_N_PER_M = 1e13


def edge_strength(s):
    """Sample the material resistance field onto the current boundary edges.

    Returns one dimensionless multiplier of F_REF_N_PER_M per boundary edge, so
    that a line force on a plate boundary can carry the same inherited weakness,
    suture and thermal history the sheet solver already uses. A boundary with no
    material under it -- a ridge or trench in open ocean -- gets 1.0, the
    unmodified reference; nothing about the rift lifecycle changes here.

    Stacked sheets at a suture are averaged, not maximised: the weld has to shear
    whatever is engaged across the whole trace, and taking the strongest or the
    weakest of a stack would make the answer depend on how many daughter sheets
    the collision happens to have split into.
    """
    count=len(getattr(s,'ba',()))
    if not count:return np.zeros(0)
    if not enabled(s) or not hasattr(s,'material_surface'):return np.ones(count)
    ensure_fields(s)
    import progressive_rifting,material_surface
    value=resistance(s,progressive_rifting.material_fields(s)['rift_damage'])
    hits=material_surface.sample_surface(s.material_surface,s.bmid)
    query,face=hits['query_index'],hits['face_index']
    total=np.bincount(query,weights=np.asarray(value,float)[face],minlength=count)
    seen=np.bincount(query,minlength=count)
    return np.where(seen>0,np.divide(total,np.maximum(seen,1)),1.)

def node_mean(s,mesh,values):
    mapping=np.asarray(mesh['parcel_node']);valid=mapping>=0;n=len(mesh['xyz'])
    weight=np.bincount(mapping[valid],weights=s.mass[valid],minlength=n)
    return np.divide(np.bincount(mapping[valid],weights=s.mass[valid]*np.asarray(values)[valid],minlength=n),
        weight,out=np.zeros(n),where=weight>0)

def capture_motion(s):
    if not enabled(s):return None
    ensure_fields(s)
    import rift_material
    mesh=rift_material.refresh(s)
    return {key:np.asarray(mesh[key]).copy() for key in ('xyz','edges','bases','owner_uids','owners')}

def finish_motion(s,before,dt):
    if before is None:return
    import rift_material
    from ridge_geometry import rotate
    mesh=rift_material.refresh(s)
    if any(not np.array_equal(before[key],mesh[key]) for key in ('edges','bases','owner_uids')):
        raise ValueError('Rift topology changed inside an atomic material transport stage.')
    a,b=mesh['edges'].T
    def length(x):return 6371.*np.arctan2(np.linalg.norm(np.cross(x[a],x[b]),axis=1),np.sum(x[a]*x[b],axis=1))
    old,new=length(before['xyz']),length(mesh['xyz'])
    strain=np.log(np.maximum(new,1e-20)/np.maximum(old,1e-20))
    strain[np.abs(strain)<1e-11]=0.
    rate=np.where(strain==0.,0.,(new-old)/dt)
    rigid=rotate(before['xyz'],s.omega[before['owners']]*dt)
    velocity=(mesh['xyz']-rigid)*6371./dt
    velocity-=mesh['xyz']*np.sum(velocity*mesh['xyz'],axis=1)[:,None]
    s.rift_realized_motion=dict(bases=mesh['bases'].copy(),owner_uids=mesh['owner_uids'].copy(),
        edges=mesh['edges'].copy(),strain=strain,edge_extension=rate,velocity=velocity,dt_myr=float(dt))
    tension=np.zeros(len(mesh['xyz']));compression=tension.copy()
    for indices in (a,b):
        np.maximum.at(tension,indices,np.maximum(strain,0.))
        np.maximum.at(compression,indices,np.maximum(-strain,0.))
    mapping=mesh['parcel_node'];valid=mapping>=0
    s.parcel_rift_realized_tension[valid]+=tension[mapping[valid]]
    s.parcel_rift_realized_compression[valid]+=compression[mapping[valid]]

def realized_response(s,mesh,response,dt):
    if not enabled(s):return None
    motion=getattr(s,'rift_realized_motion',None)
    if motion is None or not np.isclose(motion['dt_myr'],dt,rtol=0,atol=1e-12):
        raise ValueError('Enhanced rifting requires this step\'s realized material motion.')
    if any(not np.array_equal(motion[k],mesh[k]) for k in ('bases','owner_uids','edges')):
        raise ValueError('Realized rift motion no longer matches its material links.')
    response.update(velocity=motion['velocity'].copy(),edge_extension=motion['edge_extension'].copy())
    return np.maximum(motion['strain'],0.),np.maximum(-motion['strain'],0.)

def snapshot_fields(s):
    if not enabled(s):return {}
    ensure_fields(s)
    return dict(enhanced_rifting_version=1,
        **{k:getattr(s,k.replace('material_','parcel_',1)).copy() for k in ARRAY_FIELDS})

def validate_frame(frame):
    version=frame.get('enhanced_rifting_version',0)
    if isinstance(version,(bool,np.bool_)) or not isinstance(version,(int,np.integer)) or version not in (0,1):
        raise ValueError('Unsupported enhanced rift snapshot version.')
    if not version:
        if set(ARRAY_FIELDS).intersection(frame):raise ValueError('Enhanced rift arrays require their schema version.')
        return
    for name in ARRAY_FIELDS:
        value=np.asarray(frame.get(name))
        if value.shape!=(len(frame['material_faces']),) or value.dtype.kind not in 'fiu' or not np.isfinite(value).all() or np.any(value<0):
            raise ValueError('Invalid saved enhanced rift field: '+name)
        if name=='material_rift_seed_weakness' and np.any(value>.6+1e-12):
            raise ValueError('Saved inherited weakness exceeds its allowed amplitude.')
