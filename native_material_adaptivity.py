"""Runtime conservative material adaptation; the ocean control mesh stays fixed.

Refinement follows active deformation/weakness belts every ten Myr. Cratons
and rigid interiors only accept the boundary subdivisions needed for conformity.
Exact sibling coarsening requires forty quiet Myr, compatible scalar histories,
unchanged source categories, and an exactly representable projective chart.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np

import adaptive_material as kernel
import material_surface
from mesh_geometry import build_locator,locate_points


PARCEL_FIELDS=('mass','kind','relief','suture','rift_id','rift_birth_myr','rift_tangent',
               'rift_extension_m','inversion_uplift_m','geometric_log_area','material_domain')
DERIVED_PARCEL={'parcel_patch','parcel_cell','parcel_east','parcel_extent','parcel_plate'}
# Coarsening transactions allowed per adaptation pass. Refinement is a single
# batch over every eligible face, so one-per-pass could never keep up with it.
# Each transaction is a full coarsen plus apply over the material surface, so
# this is a work bound, not a target: passes with nothing eligible still cost
# nothing, and the 40 Myr registry-age and quiet-time gates are unchanged.
COARSENING_TRANSACTIONS_PER_PASS=8


def _lookup(ids,query,values,default=0):
    result=np.full((len(query),)+np.asarray(values).shape[1:],default,dtype=np.asarray(values).dtype)
    if len(ids):
        order=np.argsort(ids);at=np.searchsorted(np.asarray(ids)[order],query)
        valid=at<len(ids)
        valid[valid]&=np.asarray(ids)[order[at[valid]]]==query[valid]
        result[valid]=np.asarray(values)[order[at[valid]]]
    return result


def _parcel_fields(s,n):
    names=set(PARCEL_FIELDS)|{name for name in vars(s) if name.startswith('parcel_') and name not in DERIVED_PARCEL}
    # Extensions can declare additional persistent material fields explicitly;
    # shape alone must never mistake a same-sized ocean/marker array for crust.
    names.update(getattr(s,'additional_material_fields',()))
    return {name:getattr(s,name) for name in sorted(names) if isinstance(getattr(s,name,None),np.ndarray)
            and getattr(s,name).ndim>=1 and len(getattr(s,name))==n}


def _first(mapping):return mapping['source_indices'][mapping['source_indptr'][:-1]]


def _transfer(values,mapping,mass,*,extensive=False):
    values=np.asarray(values)
    if extensive:return kernel.transfer_extensive(values,mapping)
    if values.dtype.kind not in 'fc':return values[_first(mapping)].copy()
    return kernel.transfer_intensive(values,mapping,weights=mass).astype(values.dtype,copy=False)


def _charts(old,mapping):
    source=mapping['corner_source_face']
    return np.einsum('mci,mcij->mcj',mapping['corner_barycentric'],old[source])/mapping['corner_radial_scale'][...,None]


def _trace_patches(s,mapping,new_ids,old_ids):
    traces=np.asarray(getattr(s,'trace_patch',np.empty(0,np.int64)))
    if not len(traces):return traces.copy(),0
    old_index=_lookup(old_ids,traces,np.arange(len(old_ids),dtype=np.int64),-1)
    if np.any(old_index<0):return None,int(np.count_nonzero(old_index<0))
    query,face,_=locate_points(s.trace_xyz,build_locator(mapping['vertices'],mapping['faces']),all_hits=True)
    rows=np.repeat(np.arange(len(new_ids)),np.diff(mapping['source_indptr']))
    pair=np.sort(mapping['source_indices'].astype(np.int64)*len(new_ids)+rows)
    sought=old_index[query]*len(new_ids)+face
    at=np.searchsorted(pair,sought)
    valid=at<len(pair)
    valid[valid]&=pair[at[valid]]==sought[valid]
    query,face=query[valid],face[valid]
    order=np.lexsort((new_ids[face],query))
    _,first=np.unique(query[order],return_index=True)
    selected=order[first]
    result=np.full(len(traces),-1,np.int64)
    result[query[selected]]=new_ids[face[selected]]
    missing=int(np.count_nonzero(result<0))
    return (None if missing else result),missing


def _rift_properties(s,old_ids,mapping,new_ids):
    previous=getattr(s,'rift_properties',None)
    if not isinstance(previous,dict):return None
    result={'patch':np.sort(new_ids)}
    order=np.argsort(new_ids)
    for name,values in previous.items():
        if name=='patch':continue
        current=_lookup(previous['patch'],old_ids,values,1. if name=='strength' else 0.)
        result[name]=_transfer(current,mapping,s.mass)[order]
    return result


def _base_ids(s,ids):
    state=getattr(s,'rift_material',None)
    if not isinstance(state,dict) or state.get('backend')!='native_material_triangles':
        return np.full(len(ids),-1,np.int64)
    return _lookup(state['patch_ids'],ids,state['patch_bases'],-1)


def _categories(s):
    n=len(s.mass)
    return np.column_stack((s.material_lineage['root_id'],
        getattr(s,'parcel_craton',np.full(n,-1,np.int64)),
        s.material_surface['face_region'],_base_ids(s,s.parcel_patch)))


def _prospective_belts(s,width):
    """Conservative finite boundary/face-cap intersections for resolution only."""
    n=len(s.mass);result=np.zeros(n,bool)
    if not all(hasattr(s,key) for key in ('bmid','bn','bl','bp','bq')):return result
    points=material_surface.face_centres(s.material_surface)
    vertices=s.material_surface['vertices'][s.material_surface['faces']]
    radius=np.arccos(np.clip(np.einsum('fi,fji->fj',points,vertices),-1.,1.)).max(axis=1)
    reach=radius+float(width)/s.material_surface['radius_km']
    valid=s._valid_loading_edges() if hasattr(s,'_valid_loading_edges') else np.ones(len(s.bl),bool)
    for i,(middle,normal,length,p,q) in enumerate(zip(s.bmid,s.bn,s.bl,s.bp,s.bq)):
        if not valid[i] or length<=0. or (hasattr(s,'bcode') and s.bcode[i]==0):continue
        if hasattr(s,'omega') and np.linalg.norm(s.omega[int(p)]-s.omega[int(q)])<1e-12:continue
        selected=np.flatnonzero(((s.parcel_plate==p)|(s.parcel_plate==q))&~result)
        if not len(selected):continue
        tangent=np.cross(middle,normal);tangent/=max(np.linalg.norm(tangent),1e-30)
        half=min(float(length)/(2*s.material_surface['radius_km']),np.pi*.49)
        local=points[selected]
        along=np.arctan2(local@tangent,local@middle)
        across=np.arcsin(np.clip(np.abs(local@normal),0.,1.))
        start=np.cos(half)*middle-np.sin(half)*tangent
        end=np.cos(half)*middle+np.sin(half)*tangent
        distance=np.where(np.abs(along)<=half,across,
            np.arccos(np.clip(np.maximum(local@start,local@end),-1.,1.)))
        result[selected]=distance<=reach[selected]
    return result


def _compatible_families(s,registry,active):
    """Additional history/column gates before the kernel's geometric undo."""
    blocked=active.copy();ids=s.parcel_patch;n=len(ids)
    rows=_lookup(ids,registry['child_face_ids'],np.arange(n,dtype=np.int64),-1)
    source=registry['child_source_face'];order=np.argsort(source,kind='stable')
    starts=np.r_[0,np.cumsum(np.bincount(source,minlength=len(registry['input_faces'])))]
    fields=_parcel_fields(s,n)
    # Reference mass may differ between children; its extensive sum is kept.
    scalars=[value for name,value in fields.items() if name not in ('mass','geometric_log_area','material_domain')]
    scalars.extend(s.structure.values())
    scalars.extend(s.material_surface['columns'].values())
    scalars.extend(s.material_surface['provenance'].values())
    props=getattr(s,'rift_properties',None)
    if isinstance(props,dict):
        scalars.extend(_lookup(props['patch'],ids,value,0.) for name,value in props.items() if name!='patch')
    for parent in range(len(starts)-1):
        registered=order[starts[parent]:starts[parent+1]];group=rows[registered]
        if np.any(group<0):continue
        def same(value):
            values=value[group]
            if values.dtype.kind in 'fc':return np.allclose(values,value[group[0]],rtol=1e-10,atol=1e-8)
            return np.array_equal(values,np.broadcast_to(value[group[0]],values.shape))
        if any(not same(value) for value in scalars):
            blocked[group]=True;continue
        bary=registry['child_corner_barycentric'][registered]
        corners=[]
        for corner in range(3):
            hit=np.argwhere(np.max(np.abs(bary-np.eye(3)[corner]),axis=2)<1e-12)
            if not len(hit):break
            corners.append(s.material_surface['faces'][group[hit[0,0]],hit[0,1]])
        if len(corners)!=3:blocked[group]=True;continue
        radial=np.linalg.norm(np.einsum('fij,jk->fik',bary,s.material_surface['vertices'][corners]),axis=2)
        expected=np.einsum('fij,jk->fik',bary,registry['parent_reference_corners'][parent])/radial[...,None]
        if not np.allclose(s.material_lineage['reference_corners'][group],expected,rtol=1e-11,atol=1e-11):
            blocked[group]=True
    return blocked


def _trim_registry(registry,completed):
    keep=~np.isin(registry['input_face_ids'],completed)
    if np.all(keep):return registry
    if not np.any(keep):return None
    result=dict(registry)
    parent_index=np.cumsum(keep)-1
    children=keep[registry['child_source_face']]
    for name in ('input_faces','input_face_ids','input_source_face','input_level','input_owner','input_kind',
                 'input_categories','parent_reference_corners','parent_lineage_parent_id'):
        result[name]=registry[name][keep].copy()
    for name in ('child_face_ids','child_output_index','child_corner_barycentric','child_corner_radial_scale'):
        result[name]=registry[name][children].copy()
    result['child_source_face']=parent_index[registry['child_source_face'][children]]
    return result


def _apply(s,mapping,*,refining,registry=None):
    """Prepare a complete geometry/state transaction before changing the model."""
    old_ids=s.parcel_patch.copy();old_mass=s.mass.copy();old_lineage=s.material_lineage
    count=len(old_ids);new_count=len(mapping['faces']);first=_first(mapping)
    lengths=np.diff(mapping['source_indptr'])
    import channel_region_native
    if channel_region_native.enabled(s) and not refining and not channel_region_native.coarsening_compatible(s,mapping):
        return dict(changed=False,reason='regional phase history coarsening would mix different source states')
    if refining:
        children=np.bincount(mapping['source_face'],minlength=count)
        changed=children[mapping['source_face']]>1
        new_ids=old_ids[first].copy()
        next_id=max(int(s.next_patch_uid),int(s.material_surface['next_face_id']),int(old_ids.max(initial=-1))+1)
        new_ids[changed]=next_id+np.arange(np.count_nonzero(changed))
        next_id+=int(np.count_nonzero(changed))
        parents=old_lineage['parent_id'][first].copy();parents[changed]=old_ids[first[changed]]
    else:
        changed=lengths>1;new_ids=mapping['restored_parent_id'].copy();next_id=int(s.next_patch_uid)
        parents=old_lineage['parent_id'][first].copy()
        parents[changed]=_lookup(registry['input_face_ids'],new_ids[changed],registry['parent_lineage_parent_id'],-1)
    if len(np.unique(new_ids))!=new_count:raise ValueError('Adaptation would reuse a live material identity.')
    trace_patch,unresolved=_trace_patches(s,mapping,new_ids,old_ids)
    if unresolved:return dict(changed=False,reason='material marker containment',unresolved_markers=unresolved)
    fields={name:_transfer(value,mapping,old_mass,extensive=name=='mass') for name,value in _parcel_fields(s,count).items()}
    new_mass=fields['mass']
    structure={name:_transfer(value,mapping,old_mass) for name,value in s.structure.items()}
    import numerical_accuracy
    if numerical_accuracy.version(s):
        if not numerical_accuracy.present(s.structure):raise ValueError('Adaptation lacks its persistent numerical area ledger.')
        if not numerical_accuracy.remap_columns(s.structure,structure,mapping):
            return dict(changed=False,reason='incompatible persistent numerical area allocations')
    # T*A is the actual effective column volume per reference-area material.
    effective_area=old_mass*s.structure['area_factor']
    structure['thickness_km']=kernel.transfer_intensive(s.structure['thickness_km'],mapping,weights=effective_area)
    old_volume=float(np.sum(effective_area*s.structure['thickness_km']))
    old_physical_volume=float(np.sum(s.material_surface['area_km2']*s.structure['thickness_km']))
    lineage=dict(face_ids=new_ids.copy(),root_id=old_lineage['root_id'][first].copy(),parent_id=parents,
        reference_corners=_charts(old_lineage['reference_corners'],mapping),level=mapping['face_level'].astype(np.int32))
    old_surface=s.material_surface
    surface=dict(old_surface)
    regions=old_surface['face_region'][first].copy()
    vertex_count=len(mapping['vertices'])
    vertex_owner=np.zeros(vertex_count,dtype=old_surface['vertex_owner'].dtype)
    vertex_region=np.zeros(vertex_count,dtype=old_surface['vertex_region'].dtype)
    vertex_owner[:len(old_surface['vertices'])]=old_surface['vertex_owner']
    vertex_region[:len(old_surface['vertices'])]=old_surface['vertex_region']
    face_owner=mapping['face_owner'].astype(s.parcel_plate.dtype)
    used=np.column_stack((mapping['faces'].ravel(),np.repeat(face_owner,3),np.repeat(regions,3)))
    keys=np.unique(used,axis=0)
    if len(np.unique(keys[:,0]))!=len(keys):
        return dict(changed=False,reason='shared vertex crosses owner or material region')
    vertex_owner[keys[:,0]]=keys[:,1];vertex_region[keys[:,0]]=keys[:,2]
    surface.update(vertices=mapping['vertices'].copy(),faces=mapping['faces'].copy(),vertex_owner=vertex_owner,
        vertex_region=vertex_region,face_owner=face_owner,face_kind=mapping['face_kind'].astype(s.kind.dtype),
        face_id=new_ids.copy(),face_region=regions,area_km2=mapping['area_km2'].copy(),reference_area_km2=new_mass.copy(),
        next_face_id=max(next_id,int(old_surface['next_face_id'])),geometry_revision=old_surface['geometry_revision']+1,
        columns={name:_transfer(value,mapping,old_mass) for name,value in old_surface['columns'].items()},
        provenance={name:_transfer(value,mapping,old_mass) for name,value in old_surface['provenance'].items()})
    material_surface.refresh_geometry(surface)
    if numerical_accuracy.version(s):
        structure[numerical_accuracy.AREA]=surface['area_km2'].copy()
    deformation={name:_transfer(value,mapping,old_mass) for name,value in s.material_deformation.items()
        if name.startswith('face_') and isinstance(value,np.ndarray) and value.ndim and len(value)==count}
    deformation['face_area_km2']=surface['area_km2'].copy()
    props=_rift_properties(s,old_ids,mapping,new_ids)
    old_bases=_base_ids(s,old_ids);new_bases=old_bases[first]
    prior=getattr(s,'native_domains',None)
    if isinstance(prior,dict):
        previous=_lookup(prior.get('material_ids',np.empty(0,np.int64)),old_ids,
            prior.get('material_domains',np.empty(0,np.int64)),0)
        domain=previous[first]
    else:domain=np.zeros(new_count,np.int64)
    quiet=np.full(new_count,-np.inf)
    np.maximum.at(quiet,np.repeat(np.arange(new_count),lengths),
        s.material_adaptivity['quiet_since'][mapping['source_indices']])
    regional=(channel_region_native.prepare_remap(s,mapping,new_ids,surface,structure,trace_patch)
              if channel_region_native.enabled(s) else None)
    if regional is not None:
        structure=regional['structure'];new_mass=regional['mass']
        fields['mass']=new_mass
        surface['reference_area_km2']=new_mass.copy()
    import crust_inventory
    if crust_inventory.present(s.structure):
        crust_inventory.validate(structure)
        for name in crust_inventory.FIELDS:
            if not np.isclose(old_mass@s.structure[name],new_mass@structure[name],rtol=2e-13,atol=1e-6):
                raise ValueError('Adaptation changed the removable-crust inventory: '+name)
    import dense_crust
    if dense_crust.present(s.structure):
        dense_crust.validate(structure)
        for name in dense_crust.FIELDS:
            if not np.isclose(old_mass@s.structure[name],new_mass@structure[name],rtol=2e-13,atol=1e-6):
                raise ValueError('Adaptation changed the retained phase/heat inventory: '+name)
    new_volume=float(np.sum(new_mass*structure['area_factor']*structure['thickness_km']))
    new_physical_volume=float(np.sum(surface['area_km2']*structure['thickness_km']))
    if not np.isclose(new_mass.sum(),old_mass.sum(),rtol=2e-13,atol=1e-6) or not np.isclose(new_volume,old_volume,rtol=2e-13,atol=1e-5):
        raise ValueError('Material adaptation failed its reference-mass or column-volume budget.')
    if not np.isclose(old_physical_volume,new_physical_volume,rtol=2e-12,atol=1e-5):
        return dict(changed=False,reason='incompatible geometric column volume')
    import entry_regions
    entry_state=(entry_regions.prepare_remap(s,fields,surface,structure,new_ids)
                 if entry_regions.enabled(s) else None)
    # Commit: no borrowed source arrays are modified during preparation.
    for name,value in fields.items():setattr(s,name,value)
    s.material_surface=surface;s.material_lineage=lineage;s.material_deformation=deformation
    s.structure=structure;s.parcel_patch=new_ids;s.parcel_plate=face_owner.copy();s.kind=surface['face_kind'].copy()
    if entry_state is not None:s.continental_entry_regions=entry_state
    if regional is not None:s.channel_region_store=regional['store']
    if getattr(s,'lip_version',0)==1:
        # Cohort arrays transfer as intensive volume per reference area. Their
        # weighted global inventory must survive every refinement/coarsening.
        import lip_events
        lip_events.validate_state(s)
    s.next_patch_uid=next_id;s.trace_patch=trace_patch;s.pos=material_surface.face_centres(surface)
    if numerical_accuracy.version(s):
        order=np.argsort(new_ids);found=np.searchsorted(new_ids[order],trace_patch)
        if np.any(found>=len(new_ids)) or np.any(new_ids[order[found]]!=trace_patch):
            raise ValueError('Adaptation lost a marker numerical area identity.')
        for name in numerical_accuracy.FIELDS:s.trace_structure[name]=structure[name][order[found]].copy()
    if hasattr(s,'rift_tangent'):
        tangent=s.rift_tangent-s.pos*np.sum(s.rift_tangent*s.pos,axis=1)[:,None]
        s.rift_tangent=tangent/np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-30)
    s.material_domain=fields.get('material_domain',domain).copy()
    s.material_adaptivity.update(face_ids=new_ids.copy(),quiet_since=quiet)
    if props is not None:s.rift_properties=props
    if isinstance(getattr(s,'rift_material',None),dict):
        state=s.rift_material
        known=np.r_[state['patch_ids'],new_ids[new_bases>=0]]
        bases=np.r_[state['patch_bases'],new_bases[new_bases>=0]]
        order=np.argsort(known,kind='stable');known,bases=known[order],bases[order]
        _,last=np.unique(known[::-1],return_index=True);chosen=np.sort(len(known)-1-last)
        state['patch_ids']=known[chosen];state['patch_bases']=bases[chosen];state['topology_signature']=''
    if isinstance(prior,dict):prior.update(material_ids=new_ids.copy(),material_domains=domain.copy())
    # Pending mechanics and sampled occupancy refer to the old face indexing.
    # Persistent bonds, base lineage and independent trace ledgers survive.
    s.rift_pending=None;s._coverage_signature=None;s._owner_occupancy_signature=None
    for name in ('_material_occupancy_hits','_material_coverage','exposed_material'):
        if hasattr(s,name):delattr(s,name)
    if hasattr(s,'_sync_material'):s._sync_material()
    elif hasattr(s,'_indices'):s.parcel_cell=s._indices(s.pos)
    return dict(changed=True,refinement=refining,old_faces=count,new_faces=new_count,
        mass_error_fraction=float(new_mass.sum()/old_mass.sum()-1.) if old_mass.sum() else 0.,
        effective_volume_error_fraction=float(new_volume/old_volume-1.) if old_volume else 0.,
        geometric_volume_error_fraction=float(new_physical_volume/old_physical_volume-1.) if old_physical_volume else 0.,
        remapped_markers=len(trace_patch),new_ids=new_ids,completed_parent_ids=new_ids[changed])


def adapt(s,dt):
    """Adapt at most one refinement generation every ten Myr; return changed."""
    if not np.isfinite(dt) or dt<=0:raise ValueError('Adaptivity timestep must be positive.')
    if getattr(s,'lip_version',0)==1:
        import lip_events
        lip_events.ensure_fields(s)
    import enhanced_rifting
    if enhanced_rifting.enabled(s):
        enhanced_rifting.ensure_fields(s)
    import collision_contacts
    if any(hasattr(s,name) for name in collision_contacts.PARCEL_FIELDS):
        collision_contacts.validate_alignment(s)
    import collision_surface
    collision_surface.validate_alignment(s)
    n=len(s.mass);now=float(s.t)
    state=getattr(s,'material_adaptivity',None)
    if state is None:
        state=dict(version=1,initial_faces=int(getattr(s,'initial_material_faces',n)),last_adapt_myr=-10.,
            face_ids=s.parcel_patch.copy(),quiet_since=np.full(n,now),registries=[])
        s.material_adaptivity=state
    if not np.array_equal(state['face_ids'],s.parcel_patch):
        state['quiet_since']=_lookup(state['face_ids'],s.parcel_patch,state['quiet_since'],now)
        state['face_ids']=s.parcel_patch.copy()
    maximum=int(s.config.get('adaptive_refinement',0))
    birth_faces=int(getattr(s,'native_arc_birth_faces',0))
    budget=int(s.config.get('material_face_budget',0)) or max(n,4*(state['initial_faces']+birth_faces))
    deformation=s.material_deformation
    damage=np.zeros(n)
    props=getattr(s,'rift_properties',None)
    if isinstance(props,dict):damage=_lookup(props['patch'],s.parcel_patch,props['damage'],0.)
    weight=np.asarray(deformation['face_weight'])
    rigid=np.asarray(deformation['face_rigid'],bool)
    strain=np.asarray(deformation['face_strain'])
    active=(weight>.2)|(damage>.15)
    quiet=(weight<.08)&(damage<.05)&(np.abs(strain)<1e-5)
    state['quiet_since'][~quiet]=now
    diagnostic=dict(enabled=maximum>0,input_faces=n,output_faces=n,face_budget=budget,changed=False,
        active_faces=int(np.count_nonzero(active)),cadence_myr=10.,coarsening_quiet_myr=40.,
        existing_geometry_unchanged=True,control_mesh_changed=False,
        limitations='Protected blocks only subdivide for conformity; legacy private arc triangles stay unsplit. Connected volcanic patches can adapt. Coarsening rejects scalar/chart changes and bent boundaries.')
    s.adaptivity_diagnostics=diagnostic
    if maximum<=0 or n==0 or now-state['last_adapt_myr']<10.-1e-8:return False
    state['last_adapt_myr']=now
    width=float(s.config.get('deformation_width_km',400.))
    prospective=_prospective_belts(s,width)
    diagnostic['prospective_boundary_faces']=int(np.count_nonzero(prospective))
    any_change=False
    coarsened=0
    # Newer sibling generations undo before their ancestors.
    #
    # Refinement below runs as ONE batch over every eligible face, so bounding
    # coarsening to a single transaction per pass made the two structurally
    # asymmetric: a pass could add faces everywhere and shed at most one
    # registry's worth. That is a ratchet, and it shows -- parcels grew
    # 9,585 -> 20,959 at about 42/Myr while the face budget (62,804) never came
    # close to binding, and step cost scales as parcel_count^2.9, so the mesh
    # alone accounts for the run going from 3.3 to 40+ minutes per step.
    #
    # A bound is still wanted, because each transaction is a full coarsen plus
    # apply over the whole material surface. Take several per pass instead of one.
    for index in range(len(state['registries'])-1,-1,-1):
        if coarsened>=COARSENING_TRANSACTIONS_PER_PASS:break
        registry=state['registries'][index]
        if now-registry['created_myr']<40.:continue
        legacy_arc=(s.kind==3)&(getattr(s,'parcel_arc_id',np.zeros(len(s.mass),np.int64))<=0)
        blocked=(now-state['quiet_since']<40.)|legacy_arc|prospective
        blocked=_compatible_families(s,registry,blocked)
        mapping=kernel.coarsen(s.material_surface['vertices'],s.material_surface['faces'],s.parcel_plate,s.kind,
            registry,face_ids=s.parcel_patch,face_level=s.material_lineage['level'],active=blocked,categories=_categories(s))
        if len(mapping['faces'])==len(s.mass):continue
        report=_apply(s,mapping,refining=False,registry=registry)
        if report['changed']:
            trimmed=_trim_registry(registry,report['completed_parent_ids'])
            if trimmed is None:state['registries'].pop(index)
            else:state['registries'][index]=trimmed
            diagnostic['coarsening']={k:v for k,v in report.items() if not isinstance(v,np.ndarray)}
            any_change=True;coarsened+=1
            # _apply changed the parcel count, so every array sized to the old
            # mesh is now stale. quiet_since is remapped inside _apply, and
            # legacy_arc/blocked are rebuilt at the top of each iteration, but
            # `prospective` was computed once before this loop and would be read
            # at the wrong length on the next pass.
            prospective=_prospective_belts(s,width)
            continue
        diagnostic['coarsening_skipped']=report
    # Re-read aligned arrays after a possible undo.
    n=len(s.mass);weight=s.material_deformation['face_weight'];rigid=s.material_deformation['face_rigid']
    damage=np.zeros(n)
    if isinstance(getattr(s,'rift_properties',None),dict):damage=_lookup(s.rift_properties['patch'],s.parcel_patch,s.rift_properties['damage'],0.)
    prospective=_prospective_belts(s,width)
    requested=(weight>.2)|(damage>.15)|prospective
    connected_arc=(s.kind==3)&(getattr(s,'parcel_arc_id',np.zeros(len(s.mass),np.int64))>0)
    eligible=((s.kind==1)|connected_arc)&requested&(~rigid|(damage>.15)|prospective)
    target=max(width*.5,1.)
    mapping=kernel.refine(s.material_surface['vertices'],s.material_surface['faces'],s.parcel_plate,s.kind,
        desired_edge_km=np.where(eligible,target,np.inf),face_level=s.material_lineage['level'],
        protected=~eligible,max_level=maximum,max_faces=max(budget,n),categories=_categories(s),face_ids=s.parcel_patch)
    diagnostic['refinement_request']=mapping['diagnostics']
    if len(mapping['faces'])>n:
        registry=mapping['registry']
        parent=registry['input_source_face']
        registry.update(parent_reference_corners=s.material_lineage['reference_corners'][parent].copy(),
            parent_lineage_parent_id=s.material_lineage['parent_id'][parent].copy(),created_myr=now)
        report=_apply(s,mapping,refining=True)
        if report['changed']:
            kernel.bind_registry(registry,report['new_ids']);state['registries'].append(registry)
            diagnostic['refinement']={k:v for k,v in report.items() if not isinstance(v,np.ndarray)}
            any_change=True
        else:diagnostic['refinement_skipped']=report
    diagnostic.update(changed=any_change,coarsening_transactions=coarsened,
        coarsening_transaction_budget=COARSENING_TRANSACTIONS_PER_PASS,
        output_faces=len(s.mass),registered_batches=len(state['registries']),
        budget_exceeded_by_existing_material=len(s.mass)>budget)
    return any_change
