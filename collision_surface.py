"""Explicit reduced accommodation of retained overlapping crustal columns.

Extra lower-sheet column volume supplies local Airy support to the upper sheet.
The exact overlap integral is averaged on persistent birth-root/sheet groups,
so subdivision alone cannot change the physical support field. This translates
columns; it neither creates crust nor edits thickness/reference material. The
surface sampler supplies finite contact toes and ordered vertical placement.
This is a reduced compensation law, not a pressure/temperature/work solver.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import numpy as np
import crustal_structure
import mesh_coverage

VERSION=1
ARRAY_FIELDS={'material_collision_support_m':'float64',
              'material_collision_load_thickness_km':'float64',
              'material_collision_dense_load_km':'float64'}
PARCEL_FIELDS=('parcel_collision_support_m','parcel_collision_load_thickness_km')

def parcel_fields(s):
    return PARCEL_FIELDS+(('parcel_collision_dense_load_km',) if _signed(s) else ())

def _signed(s):
    return getattr(s,'retained_dense_crust_version',0)==1

def _valid_values(s,name,values):
    return np.isfinite(values).all() and (name=='parcel_collision_support_m' and _signed(s) or np.all(values>=0))

def ensure_fields(s):
    if getattr(s,'collision_surface_version',0)!=VERSION:return
    n=len(s.mass)
    for name in parcel_fields(s):
        old=getattr(s,name,None)
        if old is None:old=np.empty(0,float)
        if not isinstance(old,np.ndarray) or old.ndim!=1 or len(old)>n:
            raise ValueError('Collision support lost material alignment: '+name)
        if len(old)<n:old=np.r_[old,np.zeros(n-len(old))]
        if not _valid_values(s,name,old):
            raise ValueError('Collision support must be finite; signed load requires retained density state.')
        setattr(s,name,old)

def validate_alignment(s):
    if getattr(s,'collision_surface_version',0)!=VERSION:return
    for name in parcel_fields(s):
        array=getattr(s,name,None)
        if not isinstance(array,np.ndarray) or array.shape!=(len(s.mass),) or not _valid_values(s,name,array):
            raise ValueError('Collision support must align before material adaptation: '+name)

def descendants(contacts):
    children={}
    for row in contacts:
        children.setdefault(int(row['top_sheet']),set()).add(int(row['under_sheet']))
    result={}
    for top in children:
        reached=set();pending=list(children[top])
        while pending:
            at=pending.pop()
            if at==top:raise ValueError('Collision support requires an acyclic layer order.')
            if at in reached:continue
            reached.add(at);pending.extend(children.get(at,()))
        result[top]=reached
    return result

def ordered_overlap_pairs(surface,sheets,graph,overlap=None):
    """Return conserved-area face pairs with their persistent vertical order.

    This is geometry and contact identity only. It assigns no support, energy,
    pressure or force, so a future entry/contact law can use the same ordered
    pair inventory without inheriting the current Airy closure.
    """
    if overlap is None:
        overlap=mesh_coverage.material_overlaps(surface['vertices'],surface['faces'],sheets,
                                              radius_km=surface.get('radius_km',6371.))
    first,second,pair_area=(np.asarray(overlap[name]) for name in ('first','second','area_km2'))
    upper=[];lower=[];weights=[]
    for a,b,weight in zip(first,second,pair_area):
        one,two=int(sheets[a]),int(sheets[b])
        if two in graph.get(one,()):upper.append(int(a));lower.append(int(b))
        elif one in graph.get(two,()):upper.append(int(b));lower.append(int(a))
        else:raise ValueError('An overlapping pair has no persistent vertical order.')
        weights.append(float(weight))
    return dict(upper_face=np.asarray(upper,int),lower_face=np.asarray(lower,int),
                area_km2=np.asarray(weights,float))


def refresh(s,overlap=None):
    """Refresh explicit support without changing any physical material column."""
    if getattr(s,'collision_surface_version',0)!=VERSION:return {}
    ensure_fields(s)
    surface=s.material_surface
    area=np.asarray(surface['area_km2'],float)
    thickness=np.asarray(s.structure['thickness_km'],float)
    sheets=np.asarray(s.parcel_collision_sheet)
    roots=np.asarray(s.material_lineage['root_id'])
    if thickness.shape!=area.shape or roots.shape!=area.shape or sheets.shape!=area.shape:
        raise ValueError('Collision support inputs must align to material faces.')
    if np.any(area<=0) or np.any(thickness<=0) or not np.isfinite(thickness).all() or not np.isfinite(area).all():
        raise ValueError('Collision support needs positive finite column geometry.')
    graph=descendants(s.collision_contacts)
    dense=np.zeros(len(area))
    if _signed(s):
        import dense_crust
        dense_crust.validate(s.structure)
        if not dense_crust.present(s.structure):raise ValueError('Dense stack support needs explicit retained phase state.')
        dense=s.structure[dense_crust.DENSE]/s.structure['area_factor']
    digest=hashlib.sha256()
    for value in (surface['vertices'],surface['faces'],area,thickness,sheets,roots):digest.update(value.tobytes())
    if _signed(s):digest.update(dense.tobytes())
    digest.update(repr(sorted((top,sorted(below)) for top,below in graph.items())).encode())
    signature=digest.hexdigest()
    if signature==getattr(s,'_collision_surface_signature',None):
        return s.collision_stack_diagnostics
    pairs=ordered_overlap_pairs(surface,sheets,graph,overlap)
    upper,lower,weights=(pairs[name] for name in ('upper_face','lower_face','area_km2'))
    _,groups=np.unique(np.column_stack((roots,sheets)),axis=0,return_inverse=True)
    group_area=np.bincount(groups,weights=area).astype(float)
    integrated=np.bincount(groups[upper],weights=weights*thickness[lower],minlength=len(group_area)).astype(float)
    group_load=np.divide(integrated,group_area,out=np.zeros_like(group_area),where=group_area>0)
    own_integrated=np.bincount(groups,weights=area*thickness,minlength=len(group_area))
    group_own=np.divide(own_integrated,group_area,out=np.zeros_like(group_area),where=group_area>0)
    load=group_load[groups]
    support=load*crustal_structure.AIRY_M_PER_KM
    dense_integrated=np.bincount(groups[upper],weights=weights*dense[lower],minlength=len(group_area)).astype(float)
    dense_load=np.divide(dense_integrated,group_area,out=np.zeros_like(group_area),where=group_area>0)[groups]
    if _signed(s):
        import column_density
        support-=dense_load*(column_density.RHO_DENSE-column_density.RHO_CRUST)/column_density.RHO_MANTLE*1000.
        s.parcel_collision_dense_load_km=dense_load
    # Each pair contributes each lower physical column exactly once to its
    # upper sheet. Triple stacks include each distinct lower layer once, not
    # that lower layer's already accumulated support a second time.
    residual=float(area@load-integrated.sum())
    s.parcel_collision_load_thickness_km=load
    s.parcel_collision_support_m=support
    s.collision_stack_diagnostics=dict(version=VERSION,
        model='root-area-averaged Airy support from retained lower columns',
        reference_groups=len(group_area),overlapping_face_pairs=len(upper),
        lower_column_load_volume_km3=float(integrated.sum()),
        support_load_volume_residual_km3=residual,
        maximum_root_averaged_lower_thickness_km=float(load.max(initial=0.)),
        maximum_root_averaged_combined_column_km=float((group_own+group_load).max(initial=0.)),
        combined_column_interpretation='Own column and lower-sheet load both averaged over each birth-root/sheet area; not a pointwise maximum or a total-stack thickness cap.',
        stack_collapse_resolved=False,
        maximum_physical_support_m=float(support.max(initial=0.)),
        compensation_m_per_km=float(crustal_structure.AIRY_M_PER_KM),
        physical_column_volume_km3=float(area@thickness),
        changes_material_volume=False,pressure_temperature_work_resolved=False,
        support_resolution='persistent birth-root and sheet; refinement alone does not add vertical mechanics')
    s._collision_surface_signature=signature
    if _signed(s):
        s.collision_stack_diagnostics.update(density_policy='ordinary crust plus retained dense basal phase',
            lower_dense_load_volume_km3=float(dense_integrated.sum()),
            minimum_physical_support_m=float(support.min(initial=0.)),
            compensation_m_per_km=None)
    return s.collision_stack_diagnostics

def erosion_support(s,trace=False,*,parcel_support_m=None):
    """Map a same-time support snapshot to parcels or their source markers.

    With no supplied snapshot, refresh from the current physical columns.
    During a combined parcel/trace evolution the trace pass must reuse the
    parcel pass's pre-erosion support, not the already eroded lower-column load.
    """
    if getattr(s,'collision_surface_version',0)!=VERSION:
        return np.zeros(len(s.trace_xyz) if trace else len(s.mass))
    if parcel_support_m is None:
        refresh(s,getattr(s,'_collision_overlap',None))
        support=s.parcel_collision_support_m
    else:
        support=np.asarray(parcel_support_m,float)
        if support.shape!=(len(s.mass),) or not _valid_values(s,'parcel_collision_support_m',support):
            raise ValueError('Erosion support snapshot must align with finite density-aware source columns.')
    if not trace:return support.copy()
    ids=np.asarray(s.parcel_patch);order=np.argsort(ids)
    query=np.asarray(s.trace_patch);at=np.searchsorted(ids[order],query)
    valid=at<len(ids)
    valid[valid]&=ids[order[at[valid]]]==query[valid]
    if not valid.all():raise ValueError('A supported erosion trace has no material face.')
    return support[order[at]].copy()

def snapshot_fields(s):
    if getattr(s,'collision_surface_version',0)!=VERSION:return {}
    validate_alignment(s)
    result=dict(collision_surface_version=VERSION,
        material_collision_support_m=s.parcel_collision_support_m.copy(),
        material_collision_load_thickness_km=s.parcel_collision_load_thickness_km.copy(),
        collision_stack_diagnostics=deepcopy(getattr(s,'collision_stack_diagnostics',{})))
    if getattr(s,'collision_coast_version',0):
        import collision_coast
        result['collision_coast_version']=s.collision_coast_version
        collision_coast.version(result)
        if s.collision_coast_version >= 2:
            result[collision_coast.REFERENCE_FIELD] = s.structure['reference_elevation_m'].copy()
            collision_coast.reference_field(result, len(s.mass))
    if _signed(s):result['material_collision_dense_load_km']=s.parcel_collision_dense_load_km.copy()
    return result

def validate_frame(frame):
    import collision_coast
    collision_coast.reference_field(frame, len(frame.get('material_faces', ())))
    version=frame.get('collision_surface_version',0)
    if version not in (0,VERSION):raise ValueError('Unsupported collision surface version.')
    if not version:
        if any(name in frame for name in ARRAY_FIELDS):raise ValueError('Collision support fields need their surface version.')
        return
    if frame.get('collision_contact_version')!=1:raise ValueError('Collision support requires persistent contact identity.')
    n=len(frame['material_faces'])
    dense=frame.get('retained_dense_crust_version',0)==1
    fields=set(ARRAY_FIELDS) if dense else set(ARRAY_FIELDS)-{'material_collision_dense_load_km'}
    if not dense and 'material_collision_dense_load_km' in frame:raise ValueError('Dense stack load requires retained phase version.')
    for name in fields:
        values=np.asarray(frame.get(name))
        signed=dense and name=='material_collision_support_m'
        if values.shape!=(n,) or values.dtype.kind not in 'fiu' or not np.isfinite(values).all() or (not signed and np.any(values<0)):
            raise ValueError('Invalid saved collision support: '+name)
    support=np.asarray(frame['material_collision_support_m'])
    load=np.asarray(frame['material_collision_load_thickness_km'])
    expected=load*crustal_structure.AIRY_M_PER_KM
    if dense:
        import column_density
        expected=expected-np.asarray(frame['material_collision_dense_load_km'])*(column_density.RHO_DENSE-column_density.RHO_CRUST)/column_density.RHO_MANTLE*1000.
    if not np.allclose(support,expected,atol=1e-7,rtol=1e-12):
        raise ValueError('Saved collision support disagrees with its explicit load.')
