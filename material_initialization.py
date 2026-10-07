"""Conforming initial spherical material contours from starting artwork.

Shared P1 category scores fit coastlines, owner cuts and craton boundaries through
triangle interiors. Adaptive red/green refinement spends its extra vertices near
those transitions. All joint owner/type categories partition the source sphere;
there is no post-hoc coastline smoothing or whole-control-cell land promotion.
"""
from __future__ import annotations
import numpy as np

from mesh_geometry import spherical_area


def _unit(value):
    value=np.asarray(value,float)
    return value/np.maximum(np.linalg.norm(value,axis=-1,keepdims=True),1e-30)


def _source_categories(source,owner,kind):
    """Carry the source's material craton identities through the fitted contour."""
    craton=np.full(len(kind),-1,np.int64)
    patches=np.asarray(getattr(source,'parcel_patch',[]),np.int64)
    identities=np.asarray(getattr(source,'parcel_craton',[]),np.int64)
    if patches.shape==identities.shape and patches.size:
        valid=(patches>=0)&(patches<len(kind))&(identities>=0)
        craton[patches[valid]]=identities[valid]
    missing=(kind==2)&(craton<0)
    if np.any(missing):
        # Fixtures and imported artwork may lack a source material ledger.
        # The same source-grid grouping used by the legacy initializer supplies
        # identities before the sole transfer from artwork to native geometry.
        from fracture import component_labels
        groups,_=component_labels(kind==2,source.w,source.h)
        for group in np.unique(groups[missing]):
            region=(groups==group)&(kind==2)
            known=craton[region&(craton>=0)]
            identity=int(known.min()) if len(known) else int(craton.max()+1)
            craton[region&(craton<0)]=identity
    craton[kind!=2]=-1
    table,codes=np.unique(np.column_stack((owner,kind,craton)),axis=0,return_inverse=True)
    return table,codes


def _matching_seeds(source,centers,output_codes,codes,parents,working,points):
    """Pick actual source contributors matching each final owner/kind/craton."""
    seeds=np.asarray(source._indices(centers),np.int64)
    wrong=codes[seeds]!=output_codes
    corrected=int(np.count_nonzero(wrong))
    chosen=np.arange(len(centers))
    best=np.full(len(chosen),-1.,float)
    for cells,weights in source._sample_coordinates(centers[chosen]):
        weights=np.round(weights,12)
        improves=(weights>best)|((weights==best)&(cells<seeds[chosen]))
        eligible=(codes[cells]==output_codes[chosen])&improves&(weights>0.)
        seeds[chosen[eligible]]=cells[eligible]
        best[eligible]=weights[eligible]
    fallback=chosen[best<0.]
    if len(fallback):
        # A fitted P1 region can extend beyond its source pixel. Its parent
        # vertices still have positive contributions from that exact category;
        # use one of those local contributors, never a remote nearest craton.
        nodes=working[parents[fallback]]
        samples=source._sample_coordinates(points[nodes.ravel()])
        candidate_cells=np.stack([x[0].reshape(-1,3) for x in samples],axis=-1).reshape(-1,12).astype(np.int64)
        candidate_weights=np.round(np.stack([x[1].reshape(-1,3) for x in samples],axis=-1).reshape(-1,12),12)
        valid=(codes[candidate_cells]==output_codes[fallback,None])&(candidate_weights>0.)
        if np.any(~np.any(valid,axis=1)):
            raise ValueError('A fitted material category has no matching source contributor.')
        # Highest sampled contribution is stable without needing a source XYZ
        # array, and remains local to the small parent working triangle.
        maximum=np.max(np.where(valid,candidate_weights,-1.),axis=1)
        seeds[fallback]=np.min(np.where(valid&(candidate_weights==maximum[:,None]),
            candidate_cells,np.iinfo(np.int64).max),axis=1)
    return seeds,corrected,len(fallback)


def _sample_scores(source,points,categories,codes):
    """Sparse indicators: at most four source categories per shared vertex."""
    keys=np.empty((len(points),4),np.int64)
    result=np.zeros((len(points),4),float)
    for start in range(0,len(points),8192):
        section=slice(start,start+8192)
        local=points[section]
        samples=source._sample_coordinates(local)
        local_keys=np.stack([codes[cells] for cells,_ in samples],axis=1)
        values=np.stack([weights for _,weights in samples],axis=1)
        order=np.argsort(local_keys,axis=1,kind='stable')
        local_keys=np.take_along_axis(local_keys,order,axis=1)
        values=np.take_along_axis(values,order,axis=1)
        same=local_keys[:,:,None]==local_keys[:,None,:]
        values=np.sum(same*values[:,None,:],axis=2)
        first=np.column_stack((np.ones(len(local),bool),local_keys[:,1:]!=local_keys[:,:-1]))
        values*=first
        # The same shared vertices receive the same values. Quantization only
        # removes floating-point tie noise, at twelve decimal places of an
        # indicator, before any contour or region is constructed.
        result[section]=np.round(values,12)
        keys[section]=local_keys
    return keys,result


def _winners(scores):
    return scores[0][np.arange(len(scores[0])),np.argmax(scores[1],axis=1)]


def _working_mesh(source,mesh,categories,codes,refine):
    vertices=np.asarray(mesh['vertices'],float)
    faces=np.asarray(mesh['faces'],np.int64)
    scores=_sample_scores(source,vertices,categories,codes)
    if not refine:
        return vertices.copy(),faces.copy(),scores,dict(refined_control_faces=0,green_control_faces=0,
            interior_feature_points=0,working_faces=len(faces))
    edges=np.asarray(mesh['edge_vertices'])
    midpoints=_unit(vertices[edges[:,0]]+vertices[edges[:,1]])
    midpoint_scores=_sample_scores(source,midpoints,categories,codes)
    centres=_unit(vertices[faces].sum(axis=1))
    center_scores=_sample_scores(source,centres,categories,codes)
    winners=_winners(scores)
    mid_winners=_winners(midpoint_scores)
    center_winners=_winners(center_scores)
    around=np.column_stack((winners[faces],mid_winners[mesh['face_edges']]))
    red=np.any(around!=around[:,:1],axis=1)|(center_winners!=around[:,0])
    # A small island/craton may be detected only at the face centre. Include
    # that measured point rather than red-refining into four still-empty cells.
    interior=red&~np.any(around==center_winners[:,None],axis=1)
    split_edges=np.zeros(len(edges),bool)
    split_edges[np.asarray(mesh['face_edges'])[red].ravel()]=True
    flags=split_edges[mesh['face_edges']]
    number=flags.sum(axis=1)
    midpoint_index=np.arange(len(edges),dtype=np.int64)+len(vertices)
    mids=midpoint_index[mesh['face_edges']]
    interior_indices=np.full(len(faces),-1,np.int64)
    chosen=np.flatnonzero(interior)
    interior_indices[chosen]=len(vertices)+len(edges)+np.arange(len(chosen))
    points=np.r_[vertices,midpoints,centres[chosen]]
    node_scores=tuple(np.r_[scores[k],midpoint_scores[k],center_scores[k][chosen]] for k in (0,1))
    refined=[]
    for i,(a,b,c) in enumerate(faces):
        ab,bc,ca=mids[i]
        if number[i]==0:
            refined.append((a,b,c))
        elif number[i]==1:
            # Rotate local labels so the split edge is a->b.
            j=int(np.flatnonzero(flags[i])[0])
            x,y,z=faces[i][[(j+k)%3 for k in range(3)]]
            mid=mids[i,j]
            refined.extend(((x,mid,z),(mid,y,z)))
        elif number[i]==2:
            # The two split edges are x->y and y->z. This is the conforming
            # green three-triangle subdivision of the neighboring face.
            j=next(k for k in range(3) if flags[i,k] and flags[i,(k+1)%3])
            x,y,z=faces[i][[(j+k)%3 for k in range(3)]]
            xy,yz=mids[i,j],mids[i,(j+1)%3]
            refined.extend(((y,yz,xy),(x,xy,z),(xy,yz,z)))
        elif interior[i]:
            center=interior_indices[i]
            boundary=(a,ab,b,bc,c,ca)
            refined.extend((center,boundary[k],boundary[(k+1)%6]) for k in range(6))
        else:
            refined.extend(((a,ab,ca),(b,bc,ab),(c,ca,bc),(ab,bc,ca)))
    return points,np.asarray(refined,np.int64),node_scores,dict(
        refined_control_faces=int(np.count_nonzero(red)),
        green_control_faces=int(np.count_nonzero((number>0)&~red)),
        interior_feature_points=len(chosen),working_faces=len(refined))


def _clip_barycentric(polygon,difference):
    value=polygon@difference
    inside=value>=0.
    if np.all(inside):return polygon
    if not np.any(inside):return np.empty((0,3))
    output=[]
    for i in range(len(polygon)):
        previous=(i-1)%len(polygon)
        if inside[i]!=inside[previous]:
            t=value[previous]/(value[previous]-value[i])
            output.append((1-t)*polygon[previous]+t*polygon[i])
        if inside[i]:output.append(polygon[i])
    result=np.asarray(output,float).reshape(-1,3)
    if len(result)>1:
        keep=np.linalg.norm(result-np.roll(result,1,axis=0),axis=1)>2e-12
        result=result[keep]
    return result


def _triangulate_polygon(indices,polygon):
    """Convex ears preserve all shared edge vertices without extra fan faces."""
    indices=list(indices)
    polygon=np.asarray(polygon).copy()
    triangles=[]
    while len(indices)>3:
        before=np.roll(polygon,1,axis=0)[:,:2]-polygon[:,:2]
        after=np.roll(polygon,-1,axis=0)[:,:2]-polygon[:,:2]
        ears=np.abs(before[:,0]*after[:,1]-before[:,1]*after[:,0])
        total=abs(np.sum(polygon[:,0]*np.roll(polygon[:,1],-1)-polygon[:,1]*np.roll(polygon[:,0],-1)))
        # Avoid removing the only non-collinear corner, which would leave a
        # degenerate remainder even though the current ear itself is valid.
        choose=int(np.argmax(np.minimum(ears,np.maximum(total-ears,0.))))
        triangles.append((indices[choose-1],indices[choose],indices[(choose+1)%len(indices)]))
        indices.pop(choose)
        polygon=np.delete(polygon,choose,axis=0)
    triangles.append(tuple(indices))
    return triangles


def build_initial_material(source,native_mesh,*,refine_boundaries=True,boundary_refinement_levels=1):
    """Return a closed, typed spherical partition for initial material import.

    Output: vertices, faces, face_owner, face_kind, face_craton, source_seed_cells and
    diagnostics. Ocean regions (kind zero) are included to certify the complete
    partition; material_surface.initialize_surface may then omit them. Scores
    are P1 within the conforming working triangles. They approximate the finite
    source artwork, not an invented continuous geological past.
    """
    if not isinstance(refine_boundaries,(bool,np.bool_)):
        raise ValueError('Boundary refinement must be boolean.')
    if (isinstance(boundary_refinement_levels,(bool,np.bool_))
            or int(boundary_refinement_levels)!=boundary_refinement_levels
            or not 1<=boundary_refinement_levels<=5):
        raise ValueError('Boundary refinement needs one through five conforming levels.')
    owner=np.asarray(source.plate)
    kind=np.asarray(source.initial_crust)
    if (owner.ndim!=1 or kind.shape!=owner.shape or owner.dtype.kind not in 'iu'
            or kind.dtype.kind not in 'iu' or np.any(owner<0) or np.any(kind>3)):
        raise ValueError('Initial artwork needs aligned owner and crust-category arrays.')
    category_table,codes=_source_categories(source,owner,kind)
    categories=np.arange(len(category_table))
    working_mesh=native_mesh
    passes=[]
    for refinement_pass in range(int(boundary_refinement_levels) if refine_boundaries else 1):
        points,working,scores,diagnostic=_working_mesh(source,working_mesh,categories,codes,bool(refine_boundaries))
        passes.append(dict(diagnostic))
        if refinement_pass+1<int(boundary_refinement_levels) and refine_boundaries:
            # Only mixed artwork boundaries receive another conforming level.
            # Unmixed continental interiors and the fixed ocean control mesh
            # retain their original resolution. Re-sampling the artwork here
            # adds actual contour detail, rather than bisecting an old outline.
            from mesh_geometry import geometry
            used,inverse=np.unique(working.ravel(),return_inverse=True)
            working_mesh=geometry(points[used],inverse.reshape(-1,3),native_mesh.get('radius_km',6371.))
    output_points=[point.copy() for point in points]
    point_ids={}
    output_faces=[];output_codes=[];output_parents=[]
    clipped_faces=0

    def vertex_index(face_number,vertices,barycentric):
        barycentric=np.maximum(np.asarray(barycentric,float),0.)
        barycentric/=barycentric.sum()
        nonzero=np.flatnonzero(barycentric>2e-11)
        if len(nonzero)==1:
            return int(vertices[nonzero[0]])
        if len(nonzero)==2:
            first,second=nonzero
            a,b=int(vertices[first]),int(vertices[second])
            fraction=barycentric[second]/(barycentric[first]+barycentric[second])
            if a>b:a,b,fraction=b,a,1-fraction
            key=('edge',a,b,int(round(fraction*1e11)))
        else:
            key=('face',face_number,*np.rint(barycentric[:2]*1e11).astype(np.int64))
        if key not in point_ids:
            point_ids[key]=len(output_points)
            output_points.append(_unit(barycentric@points[vertices]))
        return point_ids[key]

    for face_number,vertices in enumerate(working):
        keys,weights=scores[0][vertices],scores[1][vertices]
        candidates=np.unique(keys[weights>0.])
        values=np.sum((keys[:,:,None]==candidates[None,None,:])*weights[:,:,None],axis=1)
        dominant=np.argmax(values,axis=1)
        if np.all(dominant==dominant[0]):
            output_faces.append(tuple(vertices));output_codes.append(candidates[dominant[0]])
            output_parents.append(face_number)
            continue
        clipped_faces+=1
        regions=[]
        for category in range(len(candidates)):
            polygon=np.eye(3)
            for other in range(len(candidates)):
                if other==category:continue
                difference=values[:,category]-values[:,other]
                if np.all(difference==0.):
                    if other<category:polygon=np.empty((0,3));break
                    continue
                polygon=_clip_barycentric(polygon,difference)
                if len(polygon)<3:break
            if len(polygon)<3:continue
            # Barycentric polygons partition the planar source triangle; radial
            # projection makes their straight edges exact great-circle arcs.
            area2=abs(np.sum(polygon[:,0]*np.roll(polygon[:,1],-1)-polygon[:,1]*np.roll(polygon[:,0],-1)))
            if area2<1e-18:continue
            indices=[vertex_index(face_number,vertices,p) for p in polygon]
            keep=[i for i,index in enumerate(indices) if index!=indices[i-1]]
            indices=[indices[i] for i in keep]
            polygon=polygon[keep]
            if len(indices)<3:continue
            regions.append((category,indices,polygon))
        # A third category can touch a shared interface at a single point and
        # leave a collinear vertex in only one clipped polygon. Insert that
        # same vertex in its neighbor before triangulation: no internal T seam.
        region_points={index:p for _,indices,polygon in regions for index,p in zip(indices,polygon)}
        local_indices=np.asarray(list(region_points),np.int64)
        local_points=np.asarray(list(region_points.values()))
        for category,indices,polygon in regions:
            conformed=[];conformed_points=[]
            for j,index in enumerate(indices):
                start=polygon[j];end=polygon[(j+1)%len(polygon)]
                direction=end-start
                length2=float(direction@direction)
                t=(local_points-start)@direction/length2
                residual=np.linalg.norm(local_points-start-t[:,None]*direction,axis=1)
                along=np.flatnonzero((t>2e-11)&(t<1-2e-11)&(residual<2e-11))
                along=along[np.argsort(t[along])]
                conformed.append(index);conformed_points.append(start)
                conformed.extend(local_indices[along]);conformed_points.extend(local_points[along])
            indices=conformed;polygon=np.asarray(conformed_points)
            triangles=_triangulate_polygon(indices,polygon)
            output_faces.extend(triangles)
            output_codes.extend([candidates[category]]*len(triangles))
            output_parents.extend([face_number]*len(triangles))
    faces=np.asarray(output_faces,np.int64)
    used,inverse=np.unique(faces.ravel(),return_inverse=True)
    vertices=np.asarray(output_points)[used]
    faces=inverse.reshape(-1,3).astype(np.int32)
    output_codes=np.asarray(output_codes,np.int64)
    a,b,c=vertices[faces].transpose(1,0,2)
    determinant=np.einsum('ij,ij->i',a,np.cross(b,c))
    if np.any(determinant<=0.):
        raise ValueError('Initial contour partition created an inverted or degenerate triangle.')
    areas=spherical_area(vertices,faces,native_mesh.get('radius_km',6371.))
    edges=np.sort(np.concatenate((faces[:,(0,1)],faces[:,(1,2)],faces[:,(2,0)])),axis=1)
    keys=edges[:,0].astype(np.int64)*len(vertices)+edges[:,1]
    unique,counts=np.unique(keys,return_counts=True)
    if np.any(counts!=2):
        raise ValueError('Initial contour partition has a nonconforming shared edge.')
    centers=_unit(vertices[faces].sum(axis=1))
    seed_cells,corrected,fallback=_matching_seeds(source,centers,output_codes,codes,
        np.asarray(output_parents,np.int64),working,points)
    owners=category_table[output_codes,0].astype(np.int16)
    kinds=category_table[output_codes,1].astype(np.uint8)
    cratons=category_table[output_codes,2].astype(np.int64)
    total=float(np.sum(native_mesh['area_km2']))
    original_kind=np.asarray(source.initial_crust)
    source_area=np.asarray(source.cell_area,float)
    initial_land=float(np.sum(source_area[original_kind>0]))
    diagnostic.update(model='shared P1 owner/type/craton contours with conforming adaptive spherical refinement',
        native_control_faces=len(native_mesh['faces']),material_partition_faces=len(faces),
        buoyant_faces=int(np.count_nonzero(kinds)),vertices=len(vertices),
        clipped_working_faces=clipped_faces,contour_vertices=len(point_ids),
        corrected_seed_cells=corrected,parent_contributor_seed_cells=fallback,
        global_area_km2=float(areas.sum()),global_area_error_fraction=float(areas.sum()/total-1.),
        buoyant_area_km2=float(areas[kinds>0].sum()),source_buoyant_area_km2=initial_land,
        buoyant_area_change_fraction=float((areas[kinds>0].sum()-initial_land)/max(initial_land,1.)),
        refinement=f'adaptive {int(boundary_refinement_levels)}-level red/green' if refine_boundaries else 'unrefined P1 contours',
        boundary_refinement_levels=int(boundary_refinement_levels) if refine_boundaries else 0,
        refinement_passes=passes,
        interpolation='bilinear initial raster indicators, then P1 inside shared spherical triangles')
    return dict(vertices=vertices,faces=faces,face_owner=owners,face_kind=kinds,face_craton=cratons,
                source_seed_cells=seed_cells,area_km2=areas,diagnostics=diagnostic)
