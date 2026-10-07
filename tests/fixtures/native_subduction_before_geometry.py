"""Finite incoming-ocean capture on native contours, without a grid-sized belt.

This removes available fractional support where a finite convergent interface
sweeps actual downgoing ocean. It is a reduced kinematic sink, not a pressure or
slab-volume solution. Upper-plate continent is permitted; only the incoming
owner's continental union excludes ocean. Raster and unversioned native worlds
keep their historical operator.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import math
import numpy as np
import mesh_coverage
import native_spreading as exact
from ridge_geometry import rotate

VERSION=1
RADIUS_KM=6371.
AREA_FIELDS=(
    'raw_pair_capture_area_km2','local_water_area_before_union_km2',
    'unique_downgoing_water_area_km2','duplicate_capture_area_removed_km2',
    'rejected_unrelated_owner_water_area_km2','maturity_weighted_water_area_km2',
    'available_total_overlap_km2','requested_removal_area_km2','removed_area_km2',
    'unsupported_or_unavailable_capture_area_km2')


def enabled(s):
    return isinstance(getattr(s,'native_mesh',None),dict) and getattr(s,'native_subduction_version',0)==VERSION


def _unit(x):
    x=np.asarray(x,float)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-30)


def _owners(s):
    surface=s.material_surface
    result=np.asarray(getattr(s,'parcel_plate',surface.get('face_owner',np.empty(0,int))))
    if result.shape!=(len(surface['faces']),):
        raise ValueError('Incoming-water material owners must align with actual faces.')
    return result


def _material(s,p,dt=0.):
    surface=s.material_surface
    vertices=np.asarray(surface['vertices'],float)
    if dt:vertices=rotate(vertices,np.asarray(s.omega[p])*dt)
    return exact.prepare_material(dict(vertices=vertices,faces=np.asarray(surface['faces'])[_owners(s)==p],
                                       radius_km=surface.get('radius_km',RADIUS_KM)))


def edge_ocean_fraction(s,down=None):
    """Exact incoming-owner water length fraction on each unchanged parent edge.

    A scalar owner queries every parent for that candidate polarity. An owner
    array selects its local polarity per parent; omitted uses current s.down.
    No center-cell crust flag participates in this new version.
    """
    contour=s.native_boundary_geometry
    requested=np.asarray(s.down if down is None else down)
    if requested.ndim==0:requested=np.full(len(s.ba),int(requested))
    if requested.shape!=(len(s.ba),):raise ValueError('Incoming owner must align with parent contacts.')
    digest=hashlib.sha256()
    for array in (np.array([len(s.ba)]),s.bp,s.bq,s.material_surface['vertices'],s.material_surface['faces'],_owners(s),
                  contour['segments_start'],contour['segments_end'],contour['contact_index']):
        digest.update(np.asarray(array).tobytes())
    signature=digest.hexdigest()
    if signature!=getattr(s,'_native_subduction_water_signature',None):
        s._native_subduction_water_signature=signature;s._native_subduction_water_cache={}
    cache=s._native_subduction_water_cache
    result=np.zeros(len(s.ba))
    for p in np.unique(requested[requested>=0]):
        p=int(p)
        if p not in cache:
            context=_material(s,p);total=np.zeros(len(s.ba));wet=np.zeros(len(s.ba))
            for a,b,parent in zip(contour['segments_start'],contour['segments_end'],contour['contact_index']):
                if p not in (s.bp[parent],s.bq[parent]):continue
                length=RADIUS_KM*math.atan2(float(np.linalg.norm(np.cross(a,b))),float(a@b))
                total[parent]+=length
                for low,high in exact.ocean_intervals(a,b,context):
                    first=_unit((1-low)*a+low*b);last=_unit((1-high)*a+high*b)
                    wet[parent]+=RADIUS_KM*math.atan2(float(np.linalg.norm(np.cross(first,last))),float(first@last))
            cache[p]=np.clip(np.divide(wet,total,out=np.zeros_like(wet),where=total>0),0.,1.)
        take=requested==p;result[take]=cache[p][take]
    return result


def capture_polygons(s,dt):
    """Finite converging p/q shore polygons with parent/maturity provenance."""
    if not np.isfinite(dt) or dt<=0:raise ValueError('Native capture needs positive finite time.')
    contour=s.native_boundary_geometry
    maturity=np.asarray(getattr(s,'trench_maturity',np.zeros(len(s.ba))),float)
    if maturity.shape!=(len(s.ba),) or not np.isfinite(maturity).all() or np.any((maturity<0)|(maturity>1)):
        raise ValueError('Native trench maturity must be finite and lie in [0,1].')
    rows=[]
    for a,b,normal,parent in zip(contour['segments_start'],contour['segments_end'],
                                 contour['segment_normals'],contour['contact_index']):
        parent=int(parent);p=int(s.down[parent])
        if s.bcode[parent]!=2 or maturity[parent]<=0 or p not in (s.bp[parent],s.bq[parent]):continue
        q=int(s.bq[parent] if p==s.bp[parent] else s.bp[parent])
        if not (s.active[p] and s.active[q]):continue
        endpoints=np.array([a,b],float)
        normal=np.asarray(normal,float)*(1. if p==s.bp[parent] else -1.)
        normals=_unit(normal-endpoints*(endpoints@normal)[:,None])
        incoming=rotate(endpoints,np.asarray(s.omega[p])*dt)
        overriding=rotate(endpoints,np.asarray(s.omega[q])*dt)
        convergence=np.sum((incoming-overriding)*normals,axis=1)
        if np.all(convergence<=0.):continue
        if np.any(convergence<0.):
            fraction=float(-convergence[0]/(convergence[1]-convergence[0]))
            crossing=_unit(endpoints[0]*(1-fraction)+endpoints[1]*fraction)
            endpoints[0 if convergence[0]<0. else 1]=crossing
            incoming=rotate(endpoints,np.asarray(s.omega[p])*dt)
            overriding=rotate(endpoints,np.asarray(s.omega[q])*dt)
        polygon=np.array([incoming[0],incoming[1],overriding[1],overriding[0]])
        if np.linalg.det(polygon[:3])<0:polygon=polygon[::-1]
        # Geodesic Euler shores share each internal subdivision boundary;
        # a finite front has no artificial round cap beyond either endpoint.
        area=mesh_coverage._polygon_area(polygon,RADIUS_KM)
        if area<=exact.AREA_TOLERANCE_KM2:continue
        rows.append(dict(parent=parent,down=p,over=q,maturity=float(maturity[parent]),polygon=polygon,
                         raw_area_km2=float(area)))
    # The maximum local maturity owns any geometrically coincident footprint.
    # Ties do not change its union integral, irrespective of contact ordering.
    rows.sort(key=lambda row:(-row['maturity'],row['down'],row['over'],
                              tuple(np.round(row['polygon'].mean(axis=0),14))))
    return rows


def _polygon_context(polygons):
    vertices=[];faces=[]
    for polygon in polygons:
        offset=len(vertices);vertices.extend(polygon)
        faces.extend((offset,offset+i,offset+i+1) for i in range(1,len(polygon)-1))
    return exact.prepare_material(dict(vertices=np.asarray(vertices,float).reshape(-1,3),
                                       faces=np.asarray(faces,np.int32).reshape(-1,3)))


def integrate_capture(s,rows,dt):
    """Exact unique incoming-water area and maturity-weighted area per owner/cell.

    Previous pieces are subtracted per incoming owner and actual receiving
    control cell. This avoids duplicate sinks at shared fronts or junctions,
    while a blocked unrelated-owner piece cannot suppress an eligible piece.
    """
    shape=np.asarray(s.support).shape
    admitted=np.zeros(shape);weighted=np.zeros(shape)
    contexts={};accepted=defaultdict(list);prior_context={}
    raw=0.;water_before_union=0.;foreign=0.;pieces=0
    for row in rows:
        p,q=row['down'],row['over'];polygon=row['polygon'];raw+=row['raw_area_km2']
        if p not in contexts:contexts[p]=_material(s,p,dt)
        context=contexts[p]
        local=(np.asarray(s.plate)==p)|(np.asarray(s.plate)==q)
        # Incoming support is fractional. A categorical incoming-arrival field
        # must not erase valid minority ocean in a partly continental cell.
        hit=exact.integrate_polygon(s.native_mesh,polygon,context)
        for cell,amount in zip(hit['cells'],hit['area_km2']):
            cell=int(cell)
            if not local[cell]:foreign+=float(amount);continue
            clipped=mesh_coverage.clip_triangle(polygon,s.native_mesh['vertices'][s.native_mesh['faces'][cell]])
            water=exact.uncovered_polygons(clipped,context);water_before_union+=float(amount)
            key=(p,cell)
            if accepted[key]:
                if key not in prior_context:prior_context[key]=_polygon_context(accepted[key])
                water=[part for patch in water for part in exact.uncovered_polygons(patch,prior_context[key])]
            area=math.fsum(mesh_coverage._polygon_area(part,RADIUS_KM) for part in water)
            if area<=exact.AREA_TOLERANCE_KM2:continue
            admitted[p,cell]+=area;weighted[p,cell]+=area*row['maturity'];pieces+=len(water)
            accepted[key].extend(water);prior_context.pop(key,None)
    return admitted,weighted,dict(capture_polygons=len(rows),unique_water_parts=pieces,
        raw_pair_capture_area_km2=raw,local_water_area_before_union_km2=water_before_union,
        unique_downgoing_water_area_km2=float(admitted.sum()),
        duplicate_capture_area_removed_km2=water_before_union-float(admitted.sum()),
        rejected_unrelated_owner_water_area_km2=foreign)


def removal(s,transported,dt,arrivals=None):
    """Return bounded per-owner removed support; do not mutate its input."""
    values=np.asarray(transported,float)
    if values.shape!=np.asarray(s.support).shape or not np.isfinite(values).all() or np.any(values<0):
        raise ValueError('Captured incoming support must be finite, nonnegative and aligned.')
    rows=capture_polygons(s,dt)
    admitted,weighted,diagnostics=integrate_capture(s,rows,dt)
    area=np.asarray(s.cell_area,float)
    if area.shape!=(values.shape[1],) or not np.isfinite(area).all() or np.any(area<=0.):
        raise ValueError('Finite capture requires aligned positive control-cell areas.')
    excess=np.maximum(values.sum(axis=0)-1.,0.)
    effective=np.divide(weighted,admitted,out=np.zeros_like(weighted),where=admitted>0)
    requests=np.minimum(np.minimum(values,weighted/area),excess[None,:]*effective)
    total=requests.sum(axis=0)
    scale=np.minimum(1.,np.divide(excess,total,out=np.ones_like(total),where=total>0))
    removed=requests*scale[None,:]
    if np.any(removed>values+1e-12) or np.any(removed.sum(axis=0)>excess+1e-12):
        raise ValueError('Finite native subduction exceeded incoming support or shared overlap.')
    diagnostics.update(version=VERSION,state='measured_step',
        model='finite Euler convergence footprint, exact incoming continental-union exclusion, bounded fractional support sink',
        step_start_myr=float(getattr(s,'t',0.)),step_end_myr=float(getattr(s,'t',0.))+float(dt),
        step_duration_myr=float(dt),grid_radius_used=False,center_crust_gate_used=False,
        owner_slots=list(range(values.shape[0])),
        maturity_weighted_water_area_km2=float(weighted.sum()),
        available_total_overlap_km2=float(area@excess),
        requested_removal_area_km2=float(area@total),removed_area_km2=float(area@removed.sum(axis=0)),
        removed_area_by_owner_km2=(removed@area).tolist(),
        unsupported_or_unavailable_capture_area_km2=float(weighted.sum()-area@removed.sum(axis=0)),
        maximum_support_bound_residual=float(np.max(removed-values,initial=0.)),
        maximum_shared_excess_bound_residual=float(np.max(removed.sum(axis=0)-excess,initial=0.)),
        maturity_combination='maximum per unique geometric footprint; area-weighted within each receiving cell',
        owner_competition='simultaneous proportional allocation of shared available excess',
        incoming_arrival_categories_used=False,force_parent_graph_unchanged=True,
        limitation='Reduced kinematic capture of available finite-volume support; remnant transport/normalization error is separately recorded, not declared physical subduction.')
    s.native_subduction_diagnostics=diagnostics
    return removed


def snapshot_metadata(s):
    """Saved law and last measured sink; an initial epoch has explicit zeros."""
    if not enabled(s):return {}
    record=getattr(s,'native_subduction_diagnostics',None)
    if record is None:
        if float(s.t)!=0.:
            raise ValueError('A progressed finite-subduction world lacks its measured step record.')
        record=dict(version=VERSION,state='initialized',step_start_myr=0.,step_end_myr=0.,step_duration_myr=0.,
            owner_slots=list(range(len(s.support))),removed_area_by_owner_km2=[0.]*len(s.support),
            capture_polygons=0,unique_water_parts=0,grid_radius_used=False,center_crust_gate_used=False,
            incoming_arrival_categories_used=False,force_parent_graph_unchanged=True,
            maximum_support_bound_residual=0.,maximum_shared_excess_bound_residual=0.,
            model='finite Euler convergence footprint; no transport step yet',**dict.fromkeys(AREA_FIELDS,0.))
    result=dict(native_subduction_version=VERSION,native_subduction_diagnostics=deepcopy(record))
    # A topology event can grow the slot buffer after transport. New slots did
    # not participate in the recorded sink and need explicit zero rows; the
    # existing per-owner amounts and total are never redistributed.
    row=result['native_subduction_diagnostics']
    missing=[p for p in range(len(s.support)) if p not in row['owner_slots']]
    if missing:
        row['owner_slots'].extend(missing)
        row['removed_area_by_owner_km2'].extend([0.]*len(missing))
    validate_frame(dict(result,time_myr=float(s.t)))
    return result


def validate_frame(frame):
    """Reject malformed versioned sink evidence without changing legacy frames."""
    version=frame.get('native_subduction_version',0)
    if isinstance(version,(bool,np.bool_)) or not isinstance(version,(int,np.integer)) or version not in (0,VERSION):
        raise ValueError('Unsupported native subduction version.')
    if not version:
        if 'native_subduction_diagnostics' in frame:
            raise ValueError('Finite native subduction diagnostics require their subduction version.')
        return
    row=frame.get('native_subduction_diagnostics')
    if not isinstance(row,dict) or row.get('version')!=VERSION or isinstance(row.get('version'),(bool,np.bool_)) or not isinstance(row.get('version'),(int,np.integer)):
        raise ValueError('Finite native subduction needs a versioned diagnostic record.')
    def number(name, *, nonnegative=True):
        value=row.get(name)
        if (isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.number))
                or not np.isfinite(value) or (nonnegative and value < -1e-5)):
            raise ValueError('Invalid finite subduction diagnostic: '+name)
        return float(value)
    for name in AREA_FIELDS:number(name)
    for name in ('capture_polygons','unique_water_parts'):
        value=row.get(name)
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<0:
            raise ValueError('Finite capture counts must be nonnegative integers.')
    for name in ('grid_radius_used','center_crust_gate_used','incoming_arrival_categories_used'):
        if row.get(name) is not False:raise ValueError('Finite capture cannot claim a categorical or radius operator.')
    if row.get('force_parent_graph_unchanged') is not True:
        raise ValueError('Finite capture must retain its parent force graph.')
    slots=np.asarray(row.get('owner_slots',[]));amount=np.asarray(row.get('removed_area_by_owner_km2',[]))
    if (slots.ndim!=1 or not len(slots) or not np.issubdtype(slots.dtype,np.integer)
            or np.any(slots<0) or np.any(np.diff(slots)<=0) or amount.shape!=slots.shape
            or not np.issubdtype(amount.dtype,np.number) or not np.isfinite(amount).all() or np.any(amount<0.)):
        raise ValueError('Finite capture owner rows need aligned finite nonnegative areas and unique integer slots.')
    for key in ('mesh_owner_slots','mesh_plate','native_boundary_owner_a','native_boundary_owner_b'):
        if key in frame and not np.isin(frame[key],slots).all():
            raise ValueError('Finite capture owner rows omit a recorded native owner.')
    scale=max(1.,max(number(key) for key in AREA_FIELDS));tol=max(1e-5,scale*2e-11)
    equalities=(
        (float(amount.sum()),number('removed_area_km2')),
        (number('local_water_area_before_union_km2')-number('unique_downgoing_water_area_km2'),number('duplicate_capture_area_removed_km2')),
        (number('maturity_weighted_water_area_km2')-number('removed_area_km2'),number('unsupported_or_unavailable_capture_area_km2')))
    if any(abs(a-b)>tol for a,b in equalities):
        raise ValueError('Finite subduction area components disagree.')
    bounds=(('local_water_area_before_union_km2','raw_pair_capture_area_km2'),
            ('unique_downgoing_water_area_km2','local_water_area_before_union_km2'),
            ('maturity_weighted_water_area_km2','unique_downgoing_water_area_km2'),
            ('requested_removal_area_km2','maturity_weighted_water_area_km2'),
            ('removed_area_km2','requested_removal_area_km2'),('removed_area_km2','available_total_overlap_km2'))
    if any(number(a)>number(b)+tol for a,b in bounds):
        raise ValueError('Finite subduction exceeded a measured water, maturity, request or overlap bound.')
    if (number('local_water_area_before_union_km2')+number('rejected_unrelated_owner_water_area_km2')
            >number('raw_pair_capture_area_km2')+tol):
        raise ValueError('Finite capture local and rejected water exceed the actual polygon area.')
    if ((number('raw_pair_capture_area_km2')>tol and row['capture_polygons']==0)
            or (number('unique_downgoing_water_area_km2')>tol and row['unique_water_parts']==0)):
        raise ValueError('Nonzero capture area requires actual polygon parts.')
    for key in ('maximum_support_bound_residual','maximum_shared_excess_bound_residual'):
        if abs(number(key,nonnegative=False))>1e-12:
            raise ValueError('Finite capture violated its per-cell support bound.')
    start,end,dt=(number(key) for key in ('step_start_myr','step_end_myr','step_duration_myr'))
    if min(start,end,dt)<0. or abs(end-start-dt)>1e-9:
        raise ValueError('Finite capture step times are inconsistent.')
    state=row.get('state')
    if state=='initialized':
        if (end!=0. or start!=0. or dt!=0. or any(number(key)!=0. for key in AREA_FIELDS) or np.any(amount)
                or row['capture_polygons']!=0 or row['unique_water_parts']!=0):
            raise ValueError('Unstepped finite capture must have zero elapsed time and zero sinks.')
    elif state!='measured_step' or dt<=0.:
        raise ValueError('Finite capture needs a positive measured step or explicit initial state.')
    epoch=frame.get('time_myr')
    if epoch is not None and (isinstance(epoch,(bool,np.bool_)) or not isinstance(epoch,(int,float,np.number))
            or not np.isfinite(epoch) or abs(float(epoch)-end)>1e-6):
        raise ValueError('Finite capture record does not end at its saved epoch.')
