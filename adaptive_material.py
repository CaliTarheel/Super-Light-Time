"""Conforming refinement and registered sibling undo on a spherical surface.

These kernels change material resolution, not tectonic ownership or positions
of existing vertices. Refinement bisects selected shared great-circle edges.
Neighboring faces receive conforming green splits. Coarsening only reverses
recorded complete split families with unchanged categories and footprint.

Mappings expose CSR source groups. source_area_fractions transfer extensive
quantities; source_weights average intensive quantities by geometric area.
Use transfer_intensive(..., weights=reference_mass) to conserve column volume
when reference and current geometric areas differ. No physics grid is changed.
"""
from __future__ import annotations

import numpy as np

RADIUS_KM=6371.


def _unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-30)


def _area(vertices,faces,radius):
    if not len(faces):return np.empty(0)
    a,b,c=vertices[faces].transpose(1,0,2)
    det=np.einsum('ij,ij->i',a,np.cross(b,c))
    if np.any(det<=0.):raise ValueError('Material triangles must be outward and nondegenerate.')
    denominator=1.+np.sum(a*b,axis=1)+np.sum(b*c,axis=1)+np.sum(c*a,axis=1)
    if np.any(denominator<=0.):raise ValueError('Material triangles must lie inside an open hemisphere.')
    return 2.*np.arctan2(det,denominator)*float(radius)**2


def _aligned(value,n,name,dtype=None,default=None):
    if value is None:value=default
    result=np.asarray(value,dtype=dtype)
    if result.ndim==0:result=np.full(n,result,dtype=result.dtype)
    if result.shape!=(n,):raise ValueError(f'{name} must be scalar or aligned to material faces.')
    return result


def _validate(vertices,faces,owners,kinds,radius,categories):
    vertices=np.asarray(vertices,float)
    faces=np.asarray(faces)
    if (vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all()
            or not np.allclose(np.linalg.norm(vertices,axis=1),1.,atol=2e-10,rtol=0.)):
        raise ValueError('Material vertices must be finite unit spherical vectors.')
    if faces.size==0:faces=np.empty((0,3),np.int64)
    if (faces.ndim!=2 or faces.shape[1]!=3 or faces.dtype.kind not in 'iu'
            or np.any(faces<0) or np.any(faces>=len(vertices))):
        raise ValueError('Material faces must contain valid indexed triangles.')
    faces=faces.astype(np.int64,copy=False)
    n=len(faces)
    owners=_aligned(owners,n,'owners')
    kinds=_aligned(kinds,n,'kinds')
    if n==0:owners=owners.astype(np.int64);kinds=kinds.astype(np.int64)
    if owners.dtype.kind not in 'iu' or kinds.dtype.kind not in 'iu' or np.any(owners<0) or np.any(kinds<0):
        raise ValueError('Owner and kind identities must be nonnegative integers.')
    if not np.isfinite(radius) or radius<=0:raise ValueError('Radius must be positive.')
    if categories is None:categories=np.empty((n,0),np.int64)
    categories=np.asarray(categories)
    if categories.ndim==1:categories=categories[:,None]
    if categories.ndim!=2 or len(categories)!=n or categories.dtype.kind not in 'biuf':
        raise ValueError('Source categories must be numeric and aligned to faces.')
    if not np.isfinite(categories).all():raise ValueError('Source categories must be finite.')
    return vertices,faces,owners,kinds,categories,_area(vertices,faces,radius)


def _edges(faces):
    raw=np.sort(faces[:,[[0,1],[1,2],[2,0]]],axis=2).reshape(-1,2)
    edges,inverse,counts=np.unique(raw,axis=0,return_inverse=True,return_counts=True)
    if np.any(counts>2):raise ValueError('Material topology has a non-manifold edge.')
    return edges,inverse.reshape(-1,3),counts


def _quality(points,triangles):
    """Minimum normalized planar area; used only to choose a green diagonal."""
    xyz=points[np.asarray(triangles)]
    a,b,c=xyz.transpose(1,0,2)
    area2=np.linalg.norm(np.cross(b-a,c-a),axis=1)
    squared=np.sum((a-b)**2,axis=1)+np.sum((b-c)**2,axis=1)+np.sum((c-a)**2,axis=1)
    return float(np.round(np.min(2.*np.sqrt(3.)*area2/np.maximum(squared,1e-30)),12))


def _split_face(vertices,face,midpoints):
    """One shared-edge bisection generation, preserving outward winding."""
    flags=midpoints>=0
    count=int(flags.sum())
    a,b,c=face
    ab,bc,ca=midpoints
    if count==0:return [tuple(face)]
    if count==3:return [(a,ab,ca),(b,bc,ab),(c,ca,bc),(ab,bc,ca)]
    if count==1:
        j=int(np.flatnonzero(flags)[0]);x,y,z=face[[(j+k)%3 for k in range(3)]];mid=midpoints[j]
        return [(x,mid,z),(mid,y,z)]
    j=next(k for k in range(3) if flags[k] and flags[(k+1)%3])
    x,y,z=face[[(j+k)%3 for k in range(3)]]
    xy,yz=midpoints[j],midpoints[(j+1)%3]
    first=[(x,xy,z),(xy,yz,z)]
    second=[(x,xy,yz),(x,yz,z)]
    rest=first if _quality(vertices,first)>=_quality(vertices,second) else second
    return [(y,yz,xy),*rest]


def _csr(groups):
    lengths=np.array([len(group) for group in groups],np.int64)
    return np.r_[0,np.cumsum(lengths)],np.concatenate(groups).astype(np.int64) if groups else np.empty(0,np.int64)


def refine(vertices,faces,owners,kinds,*,desired_edge_km,face_level=None,
           protected=None,max_level=6,max_faces=None,categories=None,
           face_ids=None,radius_km=RADIUS_KM,min_area_km2=1e-5):
    """Bisect overlong edges in one bounded, conforming material generation.

    Protected faces never request a split; adjoining active faces may subdivide
    their boundary without moving the protected geometry. A shared edge is
    rejected if either incident face reaches max_level. The strict face budget
    includes every green neighbor; highest relative edge excess has priority.
    Call again for another generation if the desired lengths remain unresolved.

    Registry IDs default to row indices. After assigning persistent output IDs,
    call bind_registry(registry, output_ids) before later coarsening.
    """
    v,f,owner,kind,category,areas=_validate(vertices,faces,owners,kinds,radius_km,categories)
    n=len(f)
    desired=_aligned(desired_edge_km,n,'desired edge length',float)
    level=_aligned(face_level,n,'face level',np.int64,0)
    protect=_aligned(protected,n,'protected mask',bool,False)
    identities=_aligned(face_ids,n,'face IDs',np.int64,np.arange(n))
    if np.any(desired<=0.) or np.any(np.isnan(desired)):raise ValueError('Desired edge lengths must be positive.')
    if np.any(level<0) or int(max_level)!=max_level or max_level<0:raise ValueError('Refinement levels must be nonnegative integers.')
    if len(np.unique(identities))!=n:raise ValueError('Input face IDs must be unique.')
    budget=n*4 if max_faces is None else max_faces
    if int(budget)!=budget or budget<n:raise ValueError('Face budget cannot be smaller than the existing mesh.')
    edges,face_edges,incidence=_edges(f)
    a,b=v[edges].transpose(1,0,2) if len(edges) else (np.empty((0,3)),np.empty((0,3)))
    length=np.arctan2(np.linalg.norm(np.cross(a,b),axis=1),np.sum(a*b,axis=1))*radius_km
    request=(length[face_edges]>desired[:,None])&~protect[:,None]&(level[:,None]<max_level)
    priority=np.zeros(len(edges))
    np.maximum.at(priority,face_edges.ravel(),np.where(request,length[face_edges]/desired[:,None],0.).ravel())
    blocked=np.zeros(len(edges),bool)
    np.logical_or.at(blocked,face_edges.ravel(),np.broadcast_to(((level>=max_level)|(areas<4.1*min_area_km2))[:,None],(n,3)).ravel())
    candidates=np.flatnonzero((priority>1.)&~blocked)
    order=np.lexsort((candidates,-np.round(priority[candidates],12)))
    marked=np.zeros(len(edges),bool)
    available=int(budget)-n
    for edge in candidates[order]:
        cost=int(incidence[edge])
        if cost<=available:marked[edge]=True;available-=cost
    selected=np.flatnonzero(marked)
    mids=np.full(len(edges),-1,np.int64)
    mids[selected]=len(v)+np.arange(len(selected))
    new_vertices=np.r_[v,_unit(a[selected]+b[selected])]
    new_faces=[];sources=[];corner_bary=[];scales=[]
    for source,face in enumerate(f):
        local_mid=mids[face_edges[source]]
        lookup={int(index):np.eye(3)[j] for j,index in enumerate(face)}
        for j,index in enumerate(local_mid):
            if index>=0:
                weight=np.zeros(3);weight[j]=weight[(j+1)%3]=.5;lookup[int(index)]=weight
        triangles=_split_face(new_vertices,face,local_mid)
        for triangle in triangles:
            bary=np.asarray([lookup[int(index)] for index in triangle])
            new_faces.append(triangle);sources.append(source);corner_bary.append(bary)
            scales.append(np.linalg.norm(bary@v[face],axis=1))
    new_faces=np.asarray(new_faces,np.int64).reshape(-1,3)
    source=np.asarray(sources,np.int64)
    child_areas=_area(new_vertices,new_faces,radius_km)
    fractions=child_areas/areas[source] if len(source) else np.empty(0)
    # Normalize each exact spherical partition by its measured family sum to
    # make extensive transfer conservative even at floating-point roundoff.
    partition=np.bincount(source,weights=fractions,minlength=n)
    fractions/=partition[source]
    child_count=np.bincount(source,minlength=n)
    bary=np.asarray(corner_bary,float).reshape(-1,3,3)
    radial=np.asarray(scales,float).reshape(-1,3)
    # Keep only actual split families in the enduring registry. Unchanged
    # regions do not accumulate a copy of their topology at every adaptation.
    parents=np.flatnonzero(child_count>1)
    recorded_children=np.flatnonzero(child_count[source]>1)
    registry=dict(version=1,input_faces=f[parents].copy(),input_face_ids=identities[parents].copy(),
        input_source_face=parents,input_level=level[parents].copy(),input_owner=owner[parents].copy(),input_kind=kind[parents].copy(),
        input_categories=category[parents].copy(),child_source_face=np.searchsorted(parents,source[recorded_children]),
        child_face_ids=recorded_children.copy(),child_output_index=recorded_children,output_face_count=len(source),
        child_corner_barycentric=bary[recorded_children].copy(),
        child_corner_radial_scale=radial[recorded_children].copy(),radius_km=float(radius_km))
    return dict(vertices=new_vertices,faces=new_faces,face_owner=owner[source].copy(),face_kind=kind[source].copy(),
        categories=category[source].copy(),face_level=level[source]+(child_count[source]>1),
        source_face=source,area_fraction=fractions,area_km2=child_areas,
        source_indptr=np.arange(len(source)+1,dtype=np.int64),source_indices=source.copy(),
        source_area_fractions=fractions.copy(),source_weights=np.ones(len(source)),
        corner_source_face=np.repeat(source[:,None],3,axis=1),corner_barycentric=bary,corner_radial_scale=radial,
        registry=registry,diagnostics=dict(input_faces=n,output_faces=len(source),split_edges=len(selected),
            requested_edges=len(candidates),budget_deferred_edges=len(candidates)-len(selected),
            protected_subdivided_faces=int(np.count_nonzero(protect&(child_count>1))),
            maximum_partition_error=float(np.max(np.abs(partition-1.),initial=0.)),
            method='shared great-circle edge bisection with conforming green neighbors'))


def bind_registry(registry,output_face_ids):
    """Bind a split record to root-assigned persistent child identities in place."""
    identities=np.asarray(output_face_ids)
    n=int(registry['output_face_count'])
    if identities.shape!=(n,) or identities.dtype.kind not in 'iu' or len(np.unique(identities))!=n:
        raise ValueError('Each registered child requires a unique persistent face ID.')
    registry['child_face_ids']=identities[registry['child_output_index']].astype(np.int64,copy=True)
    return registry


def coarsen(vertices,faces,owners,kinds,registry,*,face_ids=None,face_level=None,
            active=None,protected=None,categories=None,footprint_tolerance_km=1e-6):
    """Undo inactive complete sibling splits without deleting needed edge nodes.

    The registry may describe an earlier generation. Missing descendants,
    changed categories, a moved midpoint, or an uncoarsened neighbor blocks that
    entire conforming family closure. Existing vertices are retained (including
    now-unused points) so external vertex IDs remain valid.
    """
    if registry.get('version')!=1:raise ValueError('Unsupported material split registry.')
    radius=float(registry['radius_km'])
    v,f,owner,kind,category,areas=_validate(vertices,faces,owners,kinds,radius,categories)
    _edges(f)
    n=len(f)
    identities=_aligned(face_ids,n,'face IDs',np.int64,np.arange(n))
    level=_aligned(face_level,n,'face level',np.int64,0)
    active=_aligned(active,n,'active mask',bool,False)
    protect=_aligned(protected,n,'protected mask',bool,False)
    if len(np.unique(identities))!=n:raise ValueError('Current face IDs must be unique.')
    if not np.isfinite(footprint_tolerance_km) or footprint_tolerance_km<0:raise ValueError('Footprint tolerance must be nonnegative.')
    recorded=np.asarray(registry['child_face_ids'],np.int64)
    order=np.argsort(identities)
    at=np.searchsorted(identities[order],recorded)
    rows=np.full(len(recorded),-1,np.int64)
    valid=at<n
    rows[valid]=order[at[valid]]
    valid&=identities[np.maximum(rows,0)]==recorded if n else False
    rows[~valid]=-1
    sources=np.asarray(registry['child_source_face'],np.int64)
    parent_count=len(registry['input_faces'])
    source_order=np.argsort(sources,kind='stable')
    source_indptr=np.r_[0,np.cumsum(np.bincount(sources,minlength=parent_count))]
    families=[];family_parents=[];family_corners=[];family_corner_sources=[];family_removed=[]
    rejected=dict(incomplete=0,active_or_protected=0,category=0,footprint=0,neighbor_closure=0)
    for parent in range(parent_count):
        registered=source_order[source_indptr[parent]:source_indptr[parent+1]]
        if len(registered)<2:continue
        group=rows[registered]
        if np.any(group<0):rejected['incomplete']+=1;continue
        if np.any(active[group]|protect[group]):rejected['active_or_protected']+=1;continue
        if (np.any(owner[group]!=owner[group[0]]) or np.any(kind[group]!=kind[group[0]])
                or category.shape[1]!=registry['input_categories'].shape[1]
                or not np.all(category[group]==registry['input_categories'][parent])):
            rejected['category']+=1;continue
        bary=np.asarray(registry['child_corner_barycentric'])[registered]
        corners=[];corner_sources=[]
        for corner in range(3):
            hits=np.argwhere(np.max(np.abs(bary-np.eye(3)[corner]),axis=2)<1e-12)
            indices=np.unique(f[group[hits[:,0]],hits[:,1]]) if len(hits) else np.empty(0,int)
            if len(indices)!=1:break
            corners.append(int(indices[0]));corner_sources.append((int(group[hits[0,0]]),int(hits[0,1])))
        if len(corners)!=3:rejected['footprint']+=1;continue
        expected=_unit(np.einsum('fij,jk->fik',bary,v[corners]))
        deviation=np.linalg.norm(expected-v[f[group]],axis=2)*radius
        if np.any(deviation>footprint_tolerance_km):rejected['footprint']+=1;continue
        # Require the complete parent boundary and interior partition, not just
        # the presence of three coincident corners from unrelated triangles.
        parent_area=_area(v,np.asarray([corners]),radius)[0]
        if abs(areas[group].sum()-parent_area)>max(parent_area*1e-10,1e-7):
            rejected['footprint']+=1;continue
        families.append(group);family_parents.append(parent);family_corners.append(corners)
        family_corner_sources.append(corner_sources)
        family_removed.append(set(f[group].ravel())-set(corners))
    assigned=np.full(n,-1,int)
    for family,group in enumerate(families):assigned[group]=family
    incident={}
    for face,vertices_in_face in enumerate(f):
        for vertex in vertices_in_face:incident.setdefault(int(vertex),[]).append(face)
    dependencies=[set() for _ in families]
    blocked=set()
    for family,removed in enumerate(family_removed):
        for vertex in removed:
            others=set(assigned[incident[vertex]])
            if -1 in others:blocked.add(family)
            for other in others-{-1,family}:
                if vertex not in family_removed[other]:blocked.add(family)
                else:dependencies[family].add(int(other));dependencies[other].add(family)
    queue=list(blocked)
    while queue:
        family=queue.pop()
        for other in dependencies[family]-blocked:blocked.add(other);queue.append(other)
    rejected['neighbor_closure']=len(blocked)
    allowed=set(range(len(families)))-blocked
    new_faces=[];groups=[];levels=[];restore=[];corner_sources=[];corner_bary=[]
    emitted=set()
    for current,face in enumerate(f):
        family=int(assigned[current])
        if family in allowed:
            if family in emitted:continue
            emitted.add(family)
            parent=family_parents[family]
            new_faces.append(family_corners[family]);groups.append(families[family])
            levels.append(registry['input_level'][parent]);restore.append(registry['input_face_ids'][parent])
            origin=family_corner_sources[family]
            corner_sources.append([pair[0] for pair in origin])
            corner_bary.append([np.eye(3)[pair[1]] for pair in origin])
        else:
            new_faces.append(face);groups.append(np.array([current],np.int64));levels.append(level[current]);restore.append(identities[current])
            corner_sources.append([current]*3);corner_bary.append(np.eye(3))
    new_faces=np.asarray(new_faces,np.int64).reshape(-1,3)
    indptr,indices=_csr(groups)
    new_areas=_area(v,new_faces,radius)
    first=indices[indptr[:-1]] if len(groups) else np.empty(0,int)
    weights=areas[indices]/np.repeat(np.array([areas[g].sum() for g in groups]),np.diff(indptr)) if len(indices) else np.empty(0)
    return dict(vertices=v.copy(),faces=new_faces,face_owner=owner[first].copy(),face_kind=kind[first].copy(),
        categories=category[first].copy(),face_level=np.asarray(levels,np.int64),restored_parent_id=np.asarray(restore,np.int64),
        source_indptr=indptr,source_indices=indices,source_area_fractions=np.ones(len(indices)),source_weights=weights,
        area_km2=new_areas,corner_source_face=np.asarray(corner_sources,np.int64).reshape(-1,3),
        corner_barycentric=np.asarray(corner_bary,float).reshape(-1,3,3),corner_radial_scale=np.ones((len(new_faces),3)),
        diagnostics=dict(input_faces=n,output_faces=len(new_faces),coarsened_families=len(allowed),rejected=rejected,
            method='exact registered sibling undo with shared midpoint closure'))


def transfer_extensive(values,mapping):
    """Conserve totals of per-face mass, reference area, or column volume."""
    values=np.asarray(values)
    indices=mapping['source_indices'];n=len(mapping['source_indptr'])-1
    rows=np.repeat(np.arange(n),np.diff(mapping['source_indptr']))
    weights=np.asarray(mapping['source_area_fractions']).reshape((-1,)+(1,)*(values.ndim-1))
    result=np.zeros((n,)+values.shape[1:],dtype=np.result_type(values.dtype,float))
    np.add.at(result,rows,values[indices]*weights)
    return result


def transfer_intensive(values,mapping,*,weights=None):
    """Average face fields; optional reference-mass weights preserve volumes."""
    values=np.asarray(values)
    indices=mapping['source_indices'];n=len(mapping['source_indptr'])-1
    rows=np.repeat(np.arange(n),np.diff(mapping['source_indptr']))
    if weights is None:weight=np.asarray(mapping['source_weights'],float)
    else:
        weight=np.asarray(weights,float)[indices]*mapping['source_area_fractions']
        if np.any(weight<0.) or not np.isfinite(weight).all():
            raise ValueError('Intensive transfer weights must be nonnegative and finite.')
        total=np.bincount(rows,weights=weight,minlength=n)
        if np.any(total<=0.) or not np.isfinite(total).all():raise ValueError('Intensive transfer weights must have positive finite group totals.')
        weight=weight/total[rows]
    result=np.zeros((n,)+values.shape[1:],dtype=np.result_type(values.dtype,float))
    np.add.at(result,rows,values[indices]*weight.reshape((-1,)+(1,)*(values.ndim-1)))
    return result
