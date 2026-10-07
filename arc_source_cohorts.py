"""Persistent volcanic-source identity from finite connected trench geometry.

Coordinates never identify magma. An origin retains its own point, time and
volume; compatible observations connect through current finite trace joints
and positive overlap of advected trench-to-source corridors. This module only
proposes connected source sets. Emplacement must still test the final footprint
against every consumed contributor, material exclusion, and column capacity.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import product
from numbers import Integral,Real
import numpy as np
import mesh_coverage
import native_spreading
from ridge_geometry import rotate

VERSION=1
JOINT_TOLERANCE=2e-11
AREA_TOLERANCE_KM2=1e-8
LINEAGE=('trench_id','episode','downgoing_plate_uid','overriding_plate_uid')


def _unit(value):
    value=np.asarray(value,float)
    return value/np.maximum(np.linalg.norm(value,axis=-1,keepdims=True),1e-30)


def _integer(value,minimum=0):
    return isinstance(value,Integral) and not isinstance(value,(bool,np.bool_)) and value>=minimum


def _state(s):
    if not hasattr(s,'arc_source_cohort_state'):
        s.arc_source_cohort_state=dict(version=VERSION,next_origin_id=1,next_component_id=1,
            tracks=[],last_observed_myr=None)
    state=s.arc_source_cohort_state
    if type(state.get('version')) is not int or state['version']!=VERSION:
        raise ValueError('Unsupported volcanic source cohort state.')
    if not all(_integer(state.get(name),1) for name in ('next_origin_id','next_component_id')):
        raise ValueError('Invalid next volcanic source identity.')
    return state


def validate_state(state,time_myr=None):
    """Validate the persisted finite-observation registry without rebuilding it."""
    if not isinstance(state,dict) or type(state.get('version')) is not int or state['version']!=VERSION:
        raise ValueError('Unsupported volcanic source cohort state.')
    for field in ('next_origin_id','next_component_id'):
        if not _integer(state.get(field),1):raise ValueError('Invalid next volcanic source identity.')
    def epoch(value):
        return isinstance(value,Real) and not isinstance(value,(bool,np.bool_)) and np.isfinite(value) and value>=0
    last=state.get('last_observed_myr')
    if last is not None and not epoch(last):raise ValueError('Invalid last volcanic source observation epoch.')
    if time_myr is not None and (not epoch(time_myr) or last is not None and last>time_myr+1e-10):
        raise ValueError('Volcanic source observations cannot come from a future epoch.')
    tracks=state.get('tracks')
    if not isinstance(tracks,list) or tracks and last is None:
        raise ValueError('A volcanic cohort registry needs a dated list of finite observations.')
    components=[]
    policies={'new_connected_front','positive_advected_corridor_overlap','ambiguous_split_or_join_new_identity'}
    for track in tracks:
        if not isinstance(track,dict) or not _integer(track.get('component_id'),1):
            raise ValueError('Invalid volcanic component observation.')
        component=int(track['component_id']);components.append(component)
        if component>=state['next_component_id']:raise ValueError('Volcanic component counter would reuse an ID.')
        key=track.get('key')
        if not isinstance(key,(list,tuple)) or len(key)!=4 or not all(_integer(v,1) for v in key) or key[2]==key[3]:
            raise ValueError('A volcanic observation needs complete distinct down/up plate lineage.')
        if not _integer(track.get('overriding_plate_uid'),1) or track['overriding_plate_uid']!=key[3]:
            raise ValueError('A volcanic track cannot change its original overriding lineage.')
        observed=track.get('last_observed_myr')
        if not epoch(observed) or observed>last:raise ValueError('Volcanic track epochs cannot exceed their latest observation batch.')
        corridors=np.asarray(track.get('corridors'),float)
        if (corridors.ndim!=3 or corridors.shape[1:]!=(3,3) or not len(corridors)
                or not np.isfinite(corridors).all()
                or not np.allclose(np.linalg.norm(corridors,axis=2),1.,rtol=0.,atol=2e-10)):
            raise ValueError('Volcanic continuity needs finite unit triangular corridors.')
        origins=track.get('origin_ids')
        if not isinstance(origins,list) or len(origins)!=len(corridors) or any(
                not _integer(v,1) or v>=state['next_origin_id'] for v in origins):
            raise ValueError('Volcanic corridor origins must align and precede the next ID.')
        parents=track.get('predecessor_component_ids')
        if not isinstance(parents,list) or len(set(parents))!=len(parents) or any(
                not _integer(v,1) or v>=state['next_component_id'] for v in parents):
            raise ValueError('Invalid volcanic component predecessor identities.')
        policy=track.get('continuity')
        if policy not in policies:raise ValueError('Unknown volcanic source continuity decision.')
        if (policy=='new_connected_front' and parents or
                policy=='positive_advected_corridor_overlap' and parents!=[component] or
                policy=='ambiguous_split_or_join_new_identity' and (not parents or component in parents)):
            raise ValueError('Volcanic continuity decision disagrees with its predecessor identities.')
    if len(set(components))!=len(components):raise ValueError('Disconnected volcanic observations cannot share a current component ID.')
    return state


def validate(rows,count=None):
    if not isinstance(rows,list) or (count is not None and len(rows)!=count):
        raise ValueError('Volcanic source provenance must align with every origin.')
    ids=[]
    for row in rows:
        if not isinstance(row,dict) or type(row.get('version')) is not int or row['version']!=VERSION:
            raise ValueError('Unsupported volcanic source provenance.')
        if not _integer(row.get('origin_id'),1):raise ValueError('A volcanic origin needs a positive persistent ID.')
        ids.append(int(row['origin_id']))
        for name in (*LINEAGE,'component_id','advection_host_plate_uid'):
            if not _integer(row.get(name)):raise ValueError('Invalid volcanic lineage ID: '+name)
        for name in ('source_time_myr','original_area_km2'):
            value=row.get(name)
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not np.isfinite(value) or value<0:
                raise ValueError('Invalid volcanic origin scalar: '+name)
        if type(row.get('valid')) is not bool or not isinstance(row.get('reason'),str):
            raise ValueError('A volcanic origin requires an explicit lineage decision.')
        trace=np.asarray(row.get('trace_xyz'),float)
        if trace.size==0:trace=trace.reshape(0,2,3)
        if (trace.ndim!=3 or trace.shape[1:]!=(2,3) or not np.isfinite(trace).all()
                or not np.allclose(np.linalg.norm(trace,axis=2),1.,rtol=0.,atol=2e-10)):
            raise ValueError('Volcanic lineage traces must contain actual finite unit segments.')
        if row['valid'] and (not len(trace) or any(row[name]<=0 for name in (*LINEAGE,'component_id'))):
            raise ValueError('Qualified volcanic origins require complete local lineage.')
        links=row.get('links')
        if not isinstance(links,list) or any(not _integer(value,1) or value==row['origin_id'] for value in links):
            raise ValueError('Volcanic source adjacency needs distinct positive origin IDs.')
    if len(set(ids))!=len(ids):raise ValueError('A volcanic source origin cannot be spent twice.')
    return rows


def normalize(s,rows_or_none,count,*,owners=None,areas=None):
    state=_state(s)
    if rows_or_none is None:
        rows=[]
        for _ in range(count):
            rows.append(dict(version=VERSION,origin_id=int(state['next_origin_id']),
                trench_id=0,episode=0,downgoing_plate_uid=0,overriding_plate_uid=0,
                component_id=0,source_time_myr=float(s.t),original_area_km2=0.,
                advection_host_plate_uid=0,valid=False,reason='unknown_source_lineage',trace_xyz=[],links=[]))
            state['next_origin_id']+=1
    else:rows=deepcopy(rows_or_none)
    validate(rows,count)
    if owners is not None:bind_new_hosts(s,rows,owners)
    if areas is not None:
        areas=np.asarray(areas,float)
        if areas.shape!=(count,) or not np.isfinite(areas).all() or np.any(areas<0):
            raise ValueError('New volcanic origin amounts must be finite and nonnegative.')
        for row,area in zip(rows,areas):
            if row['original_area_km2']==0.:row['original_area_km2']=float(area)
    state['next_origin_id']=max(state['next_origin_id'],max((r['origin_id']+1 for r in rows),default=1))
    return validate(rows,count)


def concat(*parts):
    rows=[deepcopy(row) for part in parts for row in part]
    return validate(rows)


def subset(rows,indices):
    return validate([deepcopy(rows[int(i)]) for i in indices])


def compatibility_key(row):
    if not row.get('valid',False):return None
    return tuple(int(row[name]) for name in (*LINEAGE,'component_id'))


def bind_new_hosts(s,rows,owners):
    """Explicitly bind newly supplied origins, never infer an old pending host."""
    validate(rows,len(owners))
    for row,owner in zip(rows,owners):
        owner=int(owner)
        if owner<0 or owner>=len(s.plate_uid) or not s.active[owner] or s.plate_uid[owner]<=0:
            raise ValueError('A newly supplied volcanic origin requires an active advection host.')
        uid=int(s.plate_uid[owner])
        if row['advection_host_plate_uid'] not in (0,uid):
            raise ValueError('A supplied volcanic origin cannot change its declared advection host.')
        row['advection_host_plate_uid']=uid
    return rows


def host_mask(s,owners,rows):
    """Unknown or reused slots never authorize transport or magma consumption."""
    owners=np.asarray(owners)
    if owners.ndim!=1 or owners.dtype.kind not in 'iu':raise ValueError('Volcanic advection hosts require integer owner slots.')
    validate(rows,len(owners))
    result=np.zeros(len(rows),bool)
    for i,(row,owner) in enumerate(zip(rows,owners)):
        owner=int(owner);uid=int(row['advection_host_plate_uid'])
        result[i]=(uid>0 and 0<=owner<len(s.plate_uid) and s.active[owner]
            and int(s.plate_uid[owner])==uid)
    return result


def _trace_connections(traces,origin):
    """Sparse endpoint-joint graph of a partitioned finite contour.

    Cartesian bins enumerate candidates only. Actual Euclidean distance tests
    decide roundoff-equivalent joints, including poles and the longitude seam.
    A joint stores one representative for each origin, avoiding a dense clique
    of coincident repeated observations.
    """
    parent=np.arange(len(traces));bins={};links=set()
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=int(parent[i])
        return i
    shifts=list(product((-1,0,1),repeat=3))
    for index,segment in enumerate(traces):
        for point in segment:
            key=tuple(np.floor(point/JOINT_TOLERANCE).astype(np.int64))
            representatives=[]
            for offset in shifts:
                for other,vertex in bins.get(tuple(key[k]+offset[k] for k in range(3)),()):
                    if np.linalg.norm(point-vertex)<=JOINT_TOLERANCE:
                        a,b=find(index),find(other)
                        if a!=b:parent[max(a,b)]=min(a,b)
                        if origin[index]!=origin[other]:links.add(tuple(sorted((int(origin[index]),int(origin[other])))))
                        representatives.append(other)
            # One coincident representative suffices for all future joints.
            if not representatives:bins.setdefault(key,[]).append((index,point.copy()))
    return np.array([find(i) for i in range(len(traces))]),links


def _corridors(traces,points,origin):
    triangles=np.stack((traces[:,0],traces[:,1],points[origin]),axis=1)
    reverse=np.einsum('ij,ij->i',triangles[:,0],np.cross(triangles[:,1],triangles[:,2]))<0
    triangles[reverse]=triangles[reverse][:,[1,0,2]]
    return triangles


def _overlap_links(current,previous,context=None):
    """One local positive-area witness per current origin, with sparse lookup."""
    old=np.asarray(previous['corridors'],float).reshape(-1,3,3)
    if not len(old):return set()
    if context is None:
        context=native_spreading.prepare_material(dict(vertices=old.reshape(-1,3),
            faces=np.arange(old.size//3).reshape(-1,3)))
    if 'source_directions' not in context:
        midpoint=_unit(old[:,0]+old[:,1])
        context['source_directions']=_unit(old[:,2]-midpoint*np.sum(old[:,2]*midpoint,axis=1)[:,None])
    links=set();matched=set()
    for triangle,origin in zip(current['corridors'],current['origin_ids']):
        if origin in matched:continue
        midpoint=_unit(triangle[0]+triangle[1])
        direction=_unit(triangle[2]-midpoint*np.dot(triangle[2],midpoint))
        for candidate in native_spreading._candidates(triangle,context):
            # Narrow U-shaped trenches can have overlapping source corridors
            # on opposite arms. Opposite trench-to-arc directions are not one
            # local continuation, even when the down/up plate IDs agree.
            if np.dot(direction,context['source_directions'][candidate])<=0.:continue
            clipped=mesh_coverage._clip_planes(triangle,context['intersection_planes'][candidate])
            if mesh_coverage._polygon_area(clipped,6371.)>AREA_TOLERANCE_KM2:
                links.add((int(origin),int(previous['origin_ids'][int(candidate)])))
                matched.add(origin);break
    return links


def _track_bounds(track):
    vertices=np.asarray(track['corridors'],float).reshape(-1,3)
    center=_unit(vertices.sum(axis=0));cosine=float(np.min(vertices@center))
    if cosine<=0.:return np.full(3,-1.),np.full(3,1.)
    # A minor convex triangle lies in its vertices' positive cone, so this
    # enclosing spherical cap covers curved edges as well as the vertices.
    radius=np.sqrt(max(0.,2.-2.*cosine))+1e-10
    return np.maximum(-1.,center-radius),np.minimum(1.,center+radius)


def _track_candidate_map(current,previous):
    """Conservative Cartesian bins; only actual corridor overlap is evidence."""
    resolution=16;lookup={};global_rows={};bounds=[]
    def bins(low,high):
        first=np.clip(np.floor((low+1.)*resolution*.5).astype(int),0,resolution-1)
        last=np.clip(np.floor((high+1.)*resolution*.5).astype(int),0,resolution-1)
        return first,last
    for index,track in enumerate(previous):
        low,high=_track_bounds(track);bounds.append((low,high));key=tuple(track['key'])
        first,last=bins(low,high)
        if np.prod(last-first+1)>512:
            global_rows.setdefault(key,[]).append(index);continue
        table=lookup.setdefault(key,{})
        for cell in product(*(range(a,b+1) for a,b in zip(first,last))):table.setdefault(cell,[]).append(index)
    result=[]
    for track in current:
        low,high=_track_bounds(track);first,last=bins(low,high);key=tuple(track['key'])
        selected=set(global_rows.get(key,()));table=lookup.get(key,{})
        for cell in product(*(range(a,b+1) for a,b in zip(first,last))):selected.update(table.get(cell,()))
        result.append([i for i in sorted(selected) if np.all(bounds[i][0]<=high) and np.all(low<=bounds[i][1])])
    return result


def _retain_pending_tracks(s,tracks,excluded=()):
    """Keep unseen local geometry only for live, compatible pending origins."""
    pending=getattr(s,'native_arc_pending',{})
    rows=pending.get('source_provenance',[])
    if not rows:return []
    validate(rows,len(pending['owner']))
    allowed=host_mask(s,pending['owner'],rows)
    needed={compatibility_key(row) for row,area,host in zip(rows,pending['area'],allowed)
            if host and area>0 and compatibility_key(row) is not None}
    systems={int(row['id']):row for row in getattr(s,'trench_systems',[])}
    result=[]
    for track in tracks:
        component=track['component_id'];key=tuple(track['key']);system=systems.get(key[0])
        if component in excluded or (*key,component) not in needed or system is None:continue
        if (system.get('phase') in ('shutdown','joined') or
                tuple(int(system[name]) for name in ('id','episode','downgoing_plate_uid','overriding_plate_uid'))!=key):continue
        result.append(deepcopy(track))
    return result


def build(s,source_edges,*,positions=None,owners=None,areas=None):
    """Observe actual connected source fronts and assign persistent origin IDs."""
    edges=np.asarray(source_edges)
    if edges.ndim!=1 or edges.dtype.kind not in 'iu' or np.any(edges<0) or np.any(edges>=len(s.down)):
        raise ValueError('Volcanic source edges must be valid current integer contact indices.')
    count=len(edges);state=_state(s)
    rows=normalize(s,None,count)
    if not count:
        state['tracks']=_retain_pending_tracks(s,state['tracks'])
        return rows
    if positions is None:
        sign=np.where(s.down[edges]==s.bp[edges],1.,-1.)
        positions=np.cos(180./6371.)*s.bmid[edges]+np.sin(180./6371.)*s.bn[edges]*sign[:,None]
    positions=np.asarray(positions,float)
    if owners is None:
        owners=np.where(s.down[edges]==s.bp[edges],s.bq[edges],s.bp[edges])
    owners=np.asarray(owners,int)
    areas=np.zeros(count) if areas is None else np.asarray(areas,float)
    if positions.shape!=(count,3) or owners.shape!=(count,) or areas.shape!=(count,) or np.any(areas<0) or not np.isfinite(areas).all():
        raise ValueError('Volcanic source observation arrays must align.')
    systems={int(row['id']):row for row in getattr(s,'trench_systems',[])}
    geometry=getattr(s,'native_boundary_geometry',{})
    contacts=np.asarray(geometry.get('contact_index',[]),int)
    starts=np.asarray(geometry.get('segments_start',[]),float).reshape(-1,3)
    ends=np.asarray(geometry.get('segments_end',[]),float).reshape(-1,3)
    traces_at_contact={}
    for index,contact in enumerate(contacts):traces_at_contact.setdefault(int(contact),[]).append(index)
    batches={}
    for index,edge in enumerate(edges):
        row=rows[index];row['original_area_km2']=float(areas[index])
        tid=int(s.trench_id[edge]) if hasattr(s,'trench_id') and edge<len(s.trench_id) else 0
        system=systems.get(tid)
        if system is None:continue
        row.update(trench_id=tid,episode=int(system['episode']),
            downgoing_plate_uid=int(system['downgoing_plate_uid']),overriding_plate_uid=int(system['overriding_plate_uid']))
        if (row['overriding_plate_uid']!=int(s.plate_uid[owners[index]])
                or row['downgoing_plate_uid']!=int(s.plate_uid[s.down[edge]])
                or system.get('phase') in ('shutdown','joined')):
            row['reason']='incompatible_current_trench_lineage';continue
        row['advection_host_plate_uid']=int(s.plate_uid[owners[index]])
        take=np.asarray(traces_at_contact.get(int(edge),[]),int)
        if not len(take):row['reason']='missing_finite_source_trace';continue
        trace=np.stack((starts[take],ends[take]),axis=1)
        row['trace_xyz']=trace.tolist()
        batches.setdefault(tuple(row[name] for name in LINEAGE),[]).append(index)
    candidates=[]
    for key,indices in batches.items():
        traces=np.concatenate([np.asarray(rows[i]['trace_xyz']) for i in indices])
        origins=np.concatenate([np.full(len(rows[i]['trace_xyz']),i,int) for i in indices])
        labels,joints=_trace_connections(traces,origins)
        # A single source contact assigned to disconnected finite pieces has
        # ambiguous locality. Preserve it explicitly rather than bridging them.
        labels_at_origin={}
        for origin,label in zip(origins,labels):labels_at_origin.setdefault(int(origin),set()).add(int(label))
        ambiguous={i for i,values in labels_at_origin.items() if len(values)!=1}
        for i in ambiguous:rows[i]['reason']='disconnected_trace_for_one_source'
        at_label={};joints_at_label={}
        for index,(origin,label) in enumerate(zip(origins,labels)):
            if int(origin) not in ambiguous:at_label.setdefault(int(label),[]).append(index)
        for a,b in joints:
            if a not in ambiguous and b not in ambiguous:
                label=next(iter(labels_at_origin[a]));joints_at_label.setdefault(label,[]).append((a,b))
        for label,take in sorted(at_label.items()):
            take=np.asarray(take,int)
            local=sorted(set(map(int,origins[take])))
            if not local:continue
            corridors=_corridors(traces[take],positions,origins[take])
            candidates.append(dict(key=key,indices=local,corridors=corridors,
                origin_ids=np.array([rows[i]['origin_id'] for i in origins[take]],np.int64),
                joints=joints_at_label.get(label,[])))
    old=state['tracks'];matches={};old_degree={};old_contexts={}
    for current_index,(current,possible) in enumerate(zip(candidates,_track_candidate_map(candidates,old))):
        for old_index in possible:
            previous=old[old_index]
            if old_index not in old_contexts:
                triangles=np.asarray(previous['corridors'],float).reshape(-1,3,3)
                old_contexts[old_index]=native_spreading.prepare_material(dict(vertices=triangles.reshape(-1,3),
                    faces=np.arange(triangles.size//3).reshape(-1,3)))
            links=_overlap_links(current,previous,old_contexts[old_index])
            if links:
                matches[(current_index,old_index)]=links
                old_degree[old_index]=old_degree.get(old_index,0)+1
    tracks=[]
    for index,current in enumerate(candidates):
        parents=[j for (i,j) in matches if i==index]
        unique=len(parents)==1 and old_degree[parents[0]]==1
        if unique:
            component=int(old[parents[0]]['component_id']);continuity='positive_advected_corridor_overlap'
            temporal=matches[(index,parents[0])]
        else:
            component=int(state['next_component_id']);state['next_component_id']+=1
            continuity='new_connected_front' if not parents else 'ambiguous_split_or_join_new_identity'
            temporal=set()
        for i in current['indices']:
            row=rows[i];row.update(component_id=component,valid=True,reason=continuity)
            row['links'].extend(b for a,b in temporal if a==row['origin_id'])
        for a,b in current['joints']:rows[b]['links'].append(rows[a]['origin_id'])
        for i in current['indices']:rows[i]['links']=sorted(set(rows[i]['links']))
        # A zero-flux observation carries geometric continuity but no magma
        # origin that will survive in pending. Thread its local positive-area
        # witness through to the preceding origin instead of cutting the graph.
        aliases={}
        for i in current['indices']:
            if areas[i]==0.:
                ancestors=sorted(b for a,b in temporal if a==rows[i]['origin_id'])
                if ancestors:aliases[rows[i]['origin_id']]=ancestors[0]
        track_origins=[aliases.get(int(origin),int(origin)) for origin in current['origin_ids']]
        tracks.append(dict(component_id=component,key=list(current['key']),
            corridors=current['corridors'].tolist(),origin_ids=track_origins,
            overriding_plate_uid=current['key'][3],last_observed_myr=float(s.t),
            predecessor_component_ids=[old[j]['component_id'] for j in parents],continuity=continuity))
    observed_components={track['component_id'] for track in tracks}
    retired_components={old[j]['component_id'] for _,j in matches}
    tracks.extend(_retain_pending_tracks(s,old,observed_components|retired_components))
    state['tracks']=tracks;state['last_observed_myr']=float(s.t)
    return validate(rows,count)


def advect(s,dt):
    """Advance provenance geometry once with the same owner motion as magma."""
    slots={int(uid):i for i,uid in enumerate(s.plate_uid) if uid>0}
    pending=getattr(s,'native_arc_pending',{})
    rows=pending.get('source_provenance')
    if rows is not None:
        validate(rows,len(pending['owner']))
        allowed=host_mask(s,pending['owner'],rows)
        for index,(row,owner,valid_host) in enumerate(zip(rows,pending['owner'],allowed)):
            if not valid_host:
                row.update(valid=False,reason='unresolved_advection_host')
                continue
            pending['xyz'][index]=_unit(rotate(pending['xyz'][index],s.omega[int(owner)]*dt))
            trace=np.asarray(row['trace_xyz'],float).reshape(-1,2,3)
            row['trace_xyz']=_unit(rotate(trace,s.omega[int(owner)]*dt)).tolist()
    state=getattr(s,'arc_source_cohort_state',None)
    if state is not None:
        for track in state['tracks']:
            owner=slots.get(int(track['overriding_plate_uid']))
            if owner is None or not s.active[owner]:continue
            triangle=np.asarray(track['corridors'],float).reshape(-1,3,3)
            track['corridors']=_unit(rotate(triangle,s.omega[owner]*dt)).tolist()


def groups(s,positions,owners,amounts,provenance,eligible_ocean_mask):
    """Disjoint lineage/trace-connected sets; final footprint membership is external."""
    positions=np.asarray(positions,float);owners=np.asarray(owners);amounts=np.asarray(amounts,float)
    mask=np.asarray(eligible_ocean_mask)
    if (positions.shape!=(len(positions),3) or not np.isfinite(positions).all()
            or owners.shape!=(len(positions),) or owners.dtype.kind not in 'iu'
            or amounts.shape!=(len(positions),) or not np.isfinite(amounts).all() or np.any(amounts<0)
            or mask.shape!=(len(positions),) or mask.dtype.kind!='b'):
        raise ValueError('Volcanic grouping requires aligned finite source arrays and a boolean eligibility mask.')
    rows=validate(provenance,len(positions));eligible=mask&(amounts>0)
    hosts=host_mask(s,owners,rows)
    selected=np.flatnonzero(eligible);parent={int(i):int(i) for i in selected}
    at={int(rows[i]['origin_id']):int(i) for i in selected};links=[]
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i in selected:
        key=compatibility_key(rows[i])
        if key is None or not hosts[i]:continue
        for origin in rows[i]['links']:
            j=at.get(int(origin))
            if j is not None and hosts[j] and owners[i]==owners[j] and compatibility_key(rows[j])==key:
                a,b=find(int(i)),find(j)
                if a!=b:parent[max(a,b)]=min(a,b)
                links.append((int(i),j))
    grouped={}
    for i in selected:grouped.setdefault(find(int(i)),[]).append(int(i))
    grouped_links={}
    for a,b in links:grouped_links.setdefault(find(a),[]).append((a,b))
    result=[]
    for root,indices in grouped.items():
        anchor=min(indices,key=lambda i:(rows[i]['source_time_myr'],rows[i]['origin_id']))
        key=compatibility_key(rows[anchor]) if hosts[anchor] else None
        result.append(dict(indices=indices,anchor_index=anchor,qualified=key is not None,
            compatibility_key=key,links=grouped_links.get(root,[]),
            reason='connected_source_origins' if key is not None else
                rows[anchor]['reason'] if hosts[anchor] else 'unresolved_advection_host'))
    return result
