"""Route volcanic sources at their actual spherical positions.

Control cells supply transport fields, never a replacement source position or
a continental mask. Existing juvenile growth requires containment in a real,
same-owner connected arc. Unordered overlapping material is treated as an
ambiguous source rather than granting permission to create another sheet.
"""
from __future__ import annotations

import numpy as np
import mesh_geometry
import native_boundary_geometry


def validate_positions(positions, owners, count, owner_count):
    points=np.asarray(positions,float);owners=np.asarray(owners)
    if (points.shape!=(count,3) or not np.isfinite(points).all()
            or not np.allclose(np.linalg.norm(points,axis=1),1.,rtol=0.,atol=2e-10)
            or owners.shape!=(count,) or owners.dtype.kind not in 'iu'
            or np.any(owners<0) or np.any(owners>=owner_count)):
        raise ValueError('Arc sources need finite unit positions and aligned valid owner slots.')
    return points,owners


def classify(s, positions, owners):
    """Return exact source eligibility, containing material and juvenile arc ID.

    ``eligible``, ``material_index`` (-1 for ocean), ``arc_id`` (0 for ocean),
    ``point_owner`` and ``reason`` are aligned with the supplied positions.
    Recorded collision order is transitive even when an intermediate sheet is
    absent at a query. Continental/cratonic upper material blocks juvenile
    emplacement; it continues to use the separate intrusive column process.
    """
    points,owners=validate_positions(positions,owners,len(positions),len(s.support))
    count=len(points)
    result=dict(eligible=np.zeros(count,bool),material_index=np.full(count,-1,np.int32),
                arc_id=np.zeros(count,np.int64),point_owner=np.full(count,-1,np.int32),
                reason=np.full(count,'ocean',dtype='<U32'))
    if not count:return result
    surface=s.material_surface
    locator=mesh_geometry.build_locator(surface['vertices'],surface['faces'])
    query,face,_=mesh_geometry.locate_points(points,locator,all_hits=True)
    sheets=np.asarray(getattr(s,'parcel_collision_sheet',np.arange(len(s.kind))+1))
    if sheets.shape!=(len(s.kind),):raise ValueError('Arc classification requires aligned collision sheets.')
    # Traverse the recorded DAG, not a height envelope or one-hop relation.
    children={}
    for row in getattr(s,'collision_contacts',[]):
        children.setdefault(int(row['top_sheet']),set()).add(int(row['under_sheet']))
    descendants={}
    for sheet in children:
        pending=list(children[sheet]);seen=set()
        while pending:
            child=pending.pop()
            if child==sheet:raise ValueError('Arc source collision order cannot contain a cycle.')
            if child in seen:continue
            seen.add(child);pending.extend(children.get(child,()))
        descendants[sheet]=seen
    arc_ids=np.asarray(getattr(s,'parcel_arc_id',np.zeros(len(s.kind),np.int64)))
    for q in np.unique(query):
        hits=face[query==q];present=set(map(int,sheets[hits]));hidden=set()
        for sheet in present:hidden.update(descendants.get(sheet,()))
        hits=hits[~np.isin(sheets[hits],list(hidden))]
        if not len(hits):raise ValueError('Arc source has no upper material in its collision order.')
        continental=hits[np.isin(s.kind[hits],(1,2))]
        selected=int(continental[0] if len(continental) else hits[0])
        result['material_index'][q]=selected
        result['point_owner'][q]=int(s.parcel_plate[selected])
        if len(continental):result['reason'][q]='continental_material';continue
        if np.any(s.parcel_plate[hits]!=owners[q]):
            result['reason'][q]='foreign_material';continue
        identities=np.unique(arc_ids[hits])
        if np.any(s.kind[hits]!=3) or len(identities)!=1 or identities[0]<=0:
            result['reason'][q]='unresolved_juvenile_material';continue
        result['eligible'][q]=True;result['arc_id'][q]=int(identities[0])
        result['reason'][q]='existing_arc'
    ocean=np.flatnonzero(result['material_index']<0)
    if len(ocean):
        prepared=native_boundary_geometry.prepare_owner_sampling(s.native_mesh,s.plate,s.support,
                                                                 locator=s.native_locator)
        actual=native_boundary_geometry.sample_owners(points[ocean],prepared)
        result['point_owner'][ocean]=actual
        result['eligible'][ocean]=actual==owners[ocean]
        result['reason'][ocean]=np.where(actual==owners[ocean],'ocean','foreign_ocean_owner')
    return result


def basement(s, positions):
    """Sample the same continuous stored-ocean datum used by native surfaces."""
    import native_frame_sampling
    context=dict(mesh=s.native_mesh,locator=s.native_locator,stencil=s.native_stencil,
                 mesh_age_myr=s.age,mesh_ocean_relief_m=s.ocean_relief,mesh_crust=s.crust)
    native_frame_sampling.prepare_ocean_datum(context)
    return np.minimum(-200.,native_frame_sampling.sample_ocean_datum(context,positions))
