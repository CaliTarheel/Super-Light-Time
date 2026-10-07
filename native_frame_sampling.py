"""Sample live and saved native surfaces without using their display rasters.

Moving material triangles determine exposed crust and column height. Ocean age
retains owner-restricted reconstruction. The physical-stack surface uses a
continuous shared nodal bathymetric datum independent of discrete ownership;
legacy heights retain their owner-restricted datum. Material surfaces reconstruct height and
erosion over their actual shared vertices; unflagged epochs retain their
original face-constant sampling. Opt-in surface erosion integrates this shared
reconstruction over native material; it does not transport crust or create
extra physical resolution. Instantaneous slab-window support is
evaluated once at the query point after the exposed owner has been selected.
"""
from __future__ import annotations

import numpy as np
import mesh_geometry as geometry
import mesh_transport as transport
import ridge_interaction
import material_reconstruction
import native_boundary_geometry
import material_transport
import arc_surface
import continental_margin
import gravity_constraints_schema
import progressive_rifting
import collision_contacts
import collision_surface
import collision_coast
import native_subduction
import native_initial_ownership
import native_spreading
import arc_emplacement_geometry
import arc_birth_profile
import localized_accretion
import backarc
import eclogite_sink
import effective_subduction
import force_rifting


def is_native(frame):
    return int(frame.get('mesh_version', 0)) > 0


def _field(frame, name, count, *, integer=False, nonnegative=False):
    if name not in frame:
        raise ValueError(f'Native history lacks its saved {name} field.')
    value = np.asarray(frame[name])
    if (value.shape != (count,) or not np.isfinite(value).all()
            or (integer and not np.issubdtype(value.dtype, np.integer))
            or (nonnegative and np.any(value < 0))):
        raise ValueError(f'Invalid native history field: {name}.')
    return value


def validate_collision_surface(frame):
    """A physical support field must never silently change an old epoch."""
    collision_coast.version(frame)
    version=frame.get('collision_surface_version',0)
    if isinstance(version,bool) or not isinstance(version,(int,np.integer)) or version not in (0,1):
        raise ValueError('Unsupported collision surface version.')
    if not version:
        if 'material_collision_support_m' in frame:
            raise ValueError('Material collision support requires its surface version.')
        return None
    if (frame.get('collision_contact_version')!=1 or frame.get('surface_reconstruction_version')!=1
            or frame.get('arc_surface_version')!=2):
        raise ValueError('Physical collision surfaces require persistent contacts and continuous version-two arc surfaces.')
    return _field(frame,'material_collision_support_m',len(frame['material_owner']),
                  nonnegative=frame.get('retained_dense_crust_version',0)!=1)


def prepare_collision_surface(context, vertices, faces):
    """Shared live/saved preparation for explicit support and ordered sheets.

    Support is a separate physical datum from the unchanged rock columns.
    The same shared-vertex operator reconstructs both. The finite-margin
    fallback is part of this NEW version only; old contexts are untouched.
    """
    support=validate_collision_surface(context)
    if support is None:return context
    collision_surface.validate_frame(dict(context,material_faces=faces))
    stencil=context['material_reconstruction']
    reference=collision_coast.reference_field(context,len(faces))
    if reference is not None:
        context['coastal_reference_vertices_m']=material_reconstruction.vertex_values(stencil,reference)
    context['material_collision_support_m_vertices']=material_reconstruction.vertex_values(stencil,support)
    prepare_ocean_datum(context)
    if 'continental_margin' not in context:
        context['continental_margin']=continental_margin.prepare(vertices,faces,context['material_kind'],
                                                                  context['material_arc_id'])
    lower={int(sheet):set() for sheet in np.unique(context['material_collision_sheet'])}
    for row in context['collision_contacts']:
        top,under=int(row['top_sheet']),int(row['under_sheet'])
        lower.setdefault(top,set()).add(under);lower.setdefault(under,set())
    remaining=set(lower);ordered=[]
    while remaining:
        ready=sorted(sheet for sheet in remaining if not lower[sheet].intersection(remaining))
        if not ready:raise ValueError('Physical collision surface relationships must be acyclic.')
        ordered.extend(ready);remaining.difference_update(ready)
    context['collision_sheet_order']=ordered
    if collision_coast.version(context):
        context['collision_coast']=collision_coast.prepare(vertices,faces,context['continental_margin'])
    return context


def prepare_ocean_datum(context):
    """Continuous bathymetric datum, separate from categorical ocean provenance.

    Each shared control vertex uses ocean-face bathymetry when available. A
    vertex with no ocean neighbors uses the stored ocean age/relief beneath
    its incident control faces. These are fixed nodal choices, never a
    query-dependent mask/normalization threshold or continental column height.
    """
    age=np.asarray(context['mesh_age_myr'],float)
    relief=np.asarray(context['mesh_ocean_relief_m'],float)
    crust=np.asarray(context['mesh_crust'])
    indices,weights=context['stencil']
    bathymetry=np.where(age<20.,-2600.-365.*np.sqrt(age),-5651.+2473.*np.exp(-.0278*age))+relief
    water_weights=weights*(crust[indices]==0)
    total=water_weights.sum(axis=1)
    fallback=np.sum(weights*bathymetry[indices],axis=1)
    nodal=np.divide(np.sum(water_weights*bathymetry[indices],axis=1),total,
                    out=fallback.copy(),where=total>0.)
    if not np.isfinite(nodal).all():
        raise ValueError('Ocean datum reconstruction requires finite stored ocean age and relief.')
    context['ocean_datum_vertices_m']=nodal
    context['ocean_datum_diagnostics']=dict(version=1,
        model='shared-node ocean bathymetry with fixed nodal water-mask fallback',
        source='stored native ocean age and ocean relief; no continental column or display-raster heights',
        fallback='nodes without water neighbors use incident stored ocean fields',
        water_supported_vertices=int(np.count_nonzero(total>0.)),fallback_vertices=int(np.count_nonzero(total==0.)),
        minimum_datum_m=float(nodal.min()),maximum_datum_m=float(nodal.max()),
        categorical_owner_and_selected_age_unchanged=True,changes_physical_transport_or_columns=False)


def sample_ocean_datum(context, points):
    """Barycentric shared nodal height; owner changes cannot create a step."""
    cells,weights=geometry.locate_points(np.asarray(points,float),context['locator'])
    if np.any(cells<0):raise ValueError('The ocean datum requires a closed native control mesh.')
    weights=np.maximum(weights,0.);weights/=weights.sum(axis=1)[:,None]
    return np.einsum('ij,ij->i',context['ocean_datum_vertices_m'][context['mesh']['faces'][cells]],weights)


def prepare(frame):
    """Build bounded native locators and reusable interpolation adjacency."""
    if int(frame.get('mesh_version', 0)) != 1:
        raise ValueError('Unsupported native history mesh version.')
    effective_subduction.validate_frame(frame)
    force_rifting.validate_frame(frame)
    import surface_erosion
    surface_erosion.version(frame)
    gravity_constraints_schema.validate_frame(frame)
    progressive_rifting.validate_frame(frame)
    mechanics=frame.get('material_mechanics_version',0)
    if isinstance(mechanics,(bool,np.bool_)) or not isinstance(mechanics,(int,np.integer)) or mechanics not in (0,1):
        raise ValueError('Unsupported native material mechanics version.')
    native_subduction.validate_frame(frame)
    native_initial_ownership.validate_frame(frame)
    native_spreading.validate_frame(frame)
    arc_emplacement_geometry.validate_frame(frame)
    arc_birth_profile.validate_frame(frame)
    localized_accretion.validate_frame(frame)
    backarc.validate_frame(frame)
    eclogite_sink.validate_frame(frame)
    mesh = dict(vertices=np.asarray(frame['mesh_vertices'], float),
                faces=np.asarray(frame['mesh_faces']))
    count = len(mesh['faces'])
    mesh['area_km2'] = _field(frame, 'mesh_area_km2', count, nonnegative=True)
    if np.any(mesh['area_km2'] <= 0):
        raise ValueError('Native control faces must have positive area.')
    locator = geometry.build_locator(mesh['vertices'], mesh['faces'])
    control = {name: _field(frame, name, count, integer=name in ('mesh_plate', 'mesh_crust', 'mesh_boundary'),
                           nonnegative=name == 'mesh_age_myr')
               for name in ('mesh_plate', 'mesh_crust', 'mesh_boundary', 'mesh_age_myr', 'mesh_ocean_relief_m')}
    material_count = len(frame['material_faces'])
    collision_contacts.validate_frame(frame)
    collision_surface.validate_frame(frame)
    collision_support=validate_collision_surface(frame)
    material = {name: _field(frame, name, material_count,
                            integer=name in ('material_owner', 'material_kind', 'material_face_id'),
                            nonnegative=name == 'material_erosion_rate_m_myr')
                for name in ('material_owner', 'material_kind', 'material_face_id', 'material_height_m',
                             'material_erosion_rate_m_myr')}
    if len(np.unique(material['material_face_id'])) != material_count:
        raise ValueError('Saved material face IDs must be unique.')
    material_locator = geometry.build_locator(frame['material_vertices'], frame['material_faces'])
    uids = {int(row['id']): int(row['uid']) for row in frame['plates']}
    for owners in (control['mesh_plate'], material['material_owner']):
        if any(int(owner) not in uids for owner in np.unique(owners)):
            raise ValueError('Native surface owner lacks recorded plate metadata.')
    version = int(frame.get('surface_reconstruction_version', 0))
    if version not in (0, 1):
        raise ValueError('Unsupported material surface reconstruction version.')
    context = dict(mesh=mesh, locator=locator, stencil=transport.face_vertex_stencil(mesh),
                   material_locator=material_locator, uids=uids,
                   surface_reconstruction_version=version, **control, **material)
    context['collision_contact_version'] = frame.get('collision_contact_version', 0)
    context['collision_surface_version']=frame.get('collision_surface_version',0)
    context['collision_coast_version']=frame.get('collision_coast_version',0)
    if collision_coast.version(frame) >= 2:
        context[collision_coast.REFERENCE_FIELD]=collision_coast.reference_field(frame,material_count)
    context['retained_dense_crust_version']=frame.get('retained_dense_crust_version',0)
    if collision_support is not None:
        context['material_collision_support_m']=collision_support
        context['material_collision_load_thickness_km']=np.asarray(frame['material_collision_load_thickness_km'])
        if context['retained_dense_crust_version']==1:
            context['material_collision_dense_load_km']=np.asarray(frame['material_collision_dense_load_km'])
    if context['collision_contact_version'] == 1:
        context.update(material_collision_sheet=np.asarray(frame['material_collision_sheet']),
                       collision_contacts=frame.get('collision_contacts', []))
    transport_version = frame.get('material_transport_version', 0)
    if transport_version not in (0, 1):
        raise ValueError('Unsupported native material transport version.')
    context['material_transport_version'] = transport_version
    if transport_version == 1:
        context['material_transport'] = material_transport.prepare(frame)
    elif any(name in frame for name in material_transport.FIELDS):
        raise ValueError('Material reference fields require their transport version.')
    if version == 1:
        stencil = material_reconstruction.prepare(frame['material_vertices'], frame['material_faces'],
                                                  material['material_owner'])
        context['material_reconstruction'] = stencil
        for name in ('material_height_m', 'material_erosion_rate_m_myr'):
            context[name+'_vertices'] = material_reconstruction.vertex_values(stencil, material[name])
    arc_version, arc_surface_version = frame.get('arc_material_version', 0), frame.get('arc_surface_version', 0)
    if arc_version not in (0, 1) or arc_surface_version not in (0, 1, 2):
        raise ValueError('Unsupported native arc material or surface version.')
    if arc_version == 1:
        arc_ids = _field(frame, 'material_arc_id', material_count, integer=True, nonnegative=True)
        context['material_arc_basal_m'] = _field(frame, 'material_arc_basal_m', material_count)
        if np.any((arc_ids > 0) & ~np.isin(material['material_kind'], (1, 3))):
            raise ValueError('A volcanic arc identity must belong to juvenile or accreted material.')
        context['material_arc_id'] = arc_ids
    elif any(name in frame for name in ('material_arc_id', 'material_arc_basal_m')):
        raise ValueError('Arc material fields require their version.')
    if arc_surface_version in (1, 2):
        if arc_version != 1 or version != 1:
            raise ValueError('Arc surfaces require versioned material and continuous reconstruction.')
        context['arc_surface'] = arc_surface.prepare(context['material_reconstruction'], arc_ids)
    context['arc_surface_version'] = arc_surface_version
    margin_profile = continental_margin.validate_frame(frame)
    if margin_profile is not None:
        context['continental_margin'] = continental_margin.prepare(
            frame['material_vertices'], frame['material_faces'], material['material_kind'], arc_ids,
            profile=margin_profile)
    prepare_collision_surface(context,frame['material_vertices'],frame['material_faces'])
    owner_version = frame.get('owner_reconstruction_version', 0)
    if owner_version not in (0, 1):
        raise ValueError('Unsupported native owner reconstruction version.')
    context['owner_reconstruction_version'] = owner_version
    if owner_version == 1:
        if 'mesh_owner_slots' not in frame or 'mesh_vertex_support' not in frame:
            raise ValueError('Native owner reconstruction lacks saved slots or vertex scores.')
        slots, scores = np.asarray(frame['mesh_owner_slots']), np.asarray(frame['mesh_vertex_support'], float)
        if (slots.ndim != 1 or not np.issubdtype(slots.dtype, np.integer) or len(slots) == 0
                or np.any(slots < 0) or np.any(np.diff(slots) <= 0)
                or scores.shape != (len(slots), len(mesh['vertices']))
                or not np.isfinite(scores).all() or np.any(scores < 0)
                or not np.isin(control['mesh_plate'], slots).all()
                or any(int(slot) not in uids for slot in slots)):
            raise ValueError('Invalid saved native owner slots or vertex scores.')
        context['owner_sampling'] = native_boundary_geometry.owner_sampling_context(
            mesh, slots, scores, control['mesh_plate'], locator=locator)
    return context


def _ocean_fields(context, points, cells, owners):
    """Reconstruct age and relief without blending continental surface heights."""
    numerator_age = np.zeros(len(points))
    numerator_relief = np.zeros(len(points))
    total = np.zeros(len(points))
    coordinates = transport.sample_coordinates(context['mesh'], context['locator'], context['stencil'], points)
    for source, weights in coordinates:
        usable = (context['mesh_crust'][source] == 0) & (context['mesh_plate'][source] == owners)
        weights = weights*usable
        total += weights
        numerator_age += weights*context['mesh_age_myr'][source]
        numerator_relief += weights*context['mesh_ocean_relief_m'][source]
    # A sub-cell ocean gap can be surrounded by continental control centres.
    # In that underresolved case retain the containing native ocean fields;
    # do not borrow an adjacent plate's height or manufacture a land surface.
    age = np.divide(numerator_age, total, out=context['mesh_age_myr'][cells].astype(float).copy(), where=total > 1e-14)
    relief = np.divide(numerator_relief, total, out=context['mesh_ocean_relief_m'][cells].astype(float).copy(), where=total > 1e-14)
    return age, relief


def ocean_height(context, points, cells, owners):
    """Shared nonthermal ocean datum for pixels and open volcanic boundaries."""
    age, relief = _ocean_fields(context, points, cells, owners)
    return np.where(age < 20., -2600.-365.*np.sqrt(age),
                    -5651.+2473.*np.exp(-.0278*age))+relief


def material_height_at_hits(context, points, faces, weights, cells, ocean_owners):
    """Shared engine/export baseline on exact material triangle hits.

    The context contains the ordinary material reconstruction/nodal heights;
    an optional prepared arc_surface marks only actual open arc boundaries.
    Ocean owners are selected before exposed material overrides ownership.
    """
    stencil = context['material_reconstruction']
    height = material_reconstruction.sample_hits(stencil, context['material_height_m_vertices'], faces, weights)
    arc = context.get('arc_surface')
    if arc is None:
        return height
    affected = arc_surface.ocean_weight(arc, faces, weights) > 0
    if affected.any():
        base = ocean_height(context, np.asarray(points)[affected], np.asarray(cells)[affected],
                            np.asarray(ocean_owners)[affected])
        height[affected] = arc_surface.apply(arc, context['material_height_m_vertices'],
            np.asarray(faces)[affected], np.asarray(weights)[affected], base)
    return height


def material_thermal_at_hits(prepared_arc, points, faces, weights, material_thermal,
                             ocean_owners, uid_by_owner, episodes, time_myr):
    """Use the same open-boundary rule for instantaneous surface heat only.

    The physical column and erosion budgets retain their original owner heat;
    this returned surface overlay transitions to the adjacent ocean overlay.
    """
    result = np.asarray(material_thermal, float).copy()
    if prepared_arc is None or not len(result):
        return result
    fraction = arc_surface.ocean_weight(prepared_arc, faces, weights)
    affected = fraction > 0
    if affected.any():
        uids = np.array([uid_by_owner[int(owner)] for owner in np.asarray(ocean_owners)[affected]], np.int64)
        sea = ridge_interaction.sample_ridge_effects(episodes, np.asarray(points)[affected], uids,
                                                     float(time_myr))['thermal_support_m']
        result[affected] += fraction[affected]*(sea-result[affected])
    return result


def _select_surface_envelope(context, points, cells, ocean_owners, all_hits, *, episodes=(), time_myr=0.):
    """Version-two geometric union of contained volcanic surfaces and bedrock.

    Ordinary material keeps the existing owner/identity exposure policy. Each
    new arc surface meets that underlying material (or ocean) at its real open
    boundary. The highest contained surface is exposed, irrespective of arc
    owner or age; persistent identity breaks equal-height arc ties. Taking this
    envelope prevents a covered apron from cutting a trench through another
    volcanic patch. No triangles, crust columns or erosion ledgers are changed.

    Inputs reuse the caller's exact all-hit containment query. Returned height,
    thermal overlay, owner and material barycentric identity belong to the same
    selected surface. Raw owner heat remains separate for physical budgets.
    """
    if context.get('arc_surface_version') != 2:
        raise ValueError('The volcanic surface envelope requires arc surface version two.')
    points, cells = np.asarray(points, float), np.asarray(cells)
    ocean_owners = np.asarray(ocean_owners)
    query, face, weights = (np.asarray(value) for value in all_hits)
    size = len(points)
    material = np.full(size, -1, np.int32)
    material_weights = np.zeros((size, 3))
    owner = ocean_owners.copy()
    crust = np.zeros(size, np.uint8)
    age, relief = _ocean_fields(context, points, cells, ocean_owners)
    height = np.where(age < 20., -2600.-365.*np.sqrt(age),
                      -5651.+2473.*np.exp(-.0278*age))+relief
    def thermal_at(where, owners):
        uids = np.array([context['uids'][int(value)] for value in owners], np.int64)
        return ridge_interaction.sample_ridge_effects(episodes, where, uids, float(time_myr))['thermal_support_m']

    raw_thermal = thermal_at(points, owner)
    display_thermal = raw_thermal.copy()
    new_arc = context['material_arc_id'][face] > 0
    base_q, base_f, base_w = query[~new_arc], face[~new_arc], weights[~new_arc]
    margin = context.get('continental_margin')
    if len(base_q) and (margin is not None or context.get('collision_contact_version') == 1):
        # The geometric envelope prevents a free edge of one continental
        # sheet from cutting a new shelf through higher overlapping bedrock.
        # Only actually containing triangles participate; ocean gaps remain.
        candidates = material_reconstruction.sample_hits(context['material_reconstruction'],
            context['material_height_m_vertices'], base_f, base_w)
        if margin is not None:
            candidates, fraction = continental_margin.apply(margin, points[base_q], base_f, candidates, height[base_q])
        else:
            fraction = np.ones(len(base_q))
        candidate_raw = thermal_at(points[base_q], context['material_owner'][base_f])
        candidate_thermal = candidate_raw+(1.-fraction)*(raw_thermal[base_q]-candidate_raw)
        total = candidates+candidate_thermal
        base_total = height+display_thermal
        highest = base_total.copy()
        np.maximum.at(highest, base_q, total)
        eligible = (total > base_total[base_q]+1e-9) & (total >= highest[base_q]-1e-9)
        take = np.flatnonzero(eligible)
        if len(take):
            take = take[np.lexsort((context['material_face_id'][base_f[take]], base_q[take]))]
            take = take[np.r_[True, base_q[take[1:]] != base_q[take[:-1]]]]
            q, f = base_q[take], base_f[take]
            material[q], material_weights[q] = f, base_w[take]
            owner[q], crust[q] = context['material_owner'][f], context['material_kind'][f]
            height[q], raw_thermal[q], display_thermal[q] = candidates[take], candidate_raw[take], candidate_thermal[take]
    elif len(base_q):
        preferred = context['material_owner'][base_f] == ocean_owners[base_q]
        order = np.lexsort((context['material_face_id'][base_f], ~preferred, base_q))
        base_q, base_f, base_w = base_q[order], base_f[order], base_w[order]
        first = np.r_[True, base_q[1:] != base_q[:-1]]
        base_q, base_f, base_w = base_q[first], base_f[first], base_w[first]
        material[base_q], material_weights[base_q] = base_f, base_w
        owner[base_q], crust[base_q] = context['material_owner'][base_f], context['material_kind'][base_f]
        height[base_q] = material_reconstruction.sample_hits(context['material_reconstruction'],
            context['material_height_m_vertices'], base_f, base_w)
        raw_thermal = thermal_at(points, owner)
        display_thermal = raw_thermal.copy()
    arc_q, arc_f, arc_w = query[new_arc], face[new_arc], weights[new_arc]
    if len(arc_q):
        arc = context['arc_surface']
        candidate_height = arc_surface.apply(arc, context['material_height_m_vertices'], arc_f, arc_w, height[arc_q])
        fraction = arc_surface.ocean_weight(arc, arc_f, arc_w)
        candidate_raw_thermal = thermal_at(points[arc_q], context['material_owner'][arc_f])
        candidate_thermal = candidate_raw_thermal+fraction*(display_thermal[arc_q]-candidate_raw_thermal)
        candidate_total = candidate_height+candidate_thermal
        base_total = height+display_thermal
        highest = base_total.copy()
        np.maximum.at(highest, arc_q, candidate_total)
        # The underlying surface remains exposed where a patch adds no height.
        # A nanometre tolerance absorbs shared-edge roundoff, not real relief.
        eligible = ((candidate_total > base_total[arc_q]+1e-9)
                    & (candidate_total >= highest[arc_q]-1e-9))
        take = np.flatnonzero(eligible)
        if len(take):
            take = take[np.lexsort((context['material_face_id'][arc_f[take]], arc_q[take]))]
            take = take[np.r_[True, arc_q[take[1:]] != arc_q[take[:-1]]]]
            q, f = arc_q[take], arc_f[take]
            material[q], material_weights[q] = f, arc_w[take]
            owner[q], crust[q] = context['material_owner'][f], context['material_kind'][f]
            height[q], display_thermal[q] = candidate_height[take], candidate_thermal[take]
            raw_thermal[q] = candidate_raw_thermal[take]
    return dict(material_face=material, material_weights=material_weights, baseline_height_m=height,
                display_thermal_support_m=display_thermal, raw_thermal_support_m=raw_thermal,
                owner=owner, crust=crust, ocean_age_myr=age)


def _select_physical_stack(context, points, cells, ocean_owners, all_hits, *, episodes=(), time_myr=0.):
    """Build contained sheets bottom-up with explicit, separately measured support.

    A finite upper toe joins the resolved supporting surface. In its interior
    persistent order selects its own column even after height crossings. This
    is the reduced kinematic placement of underthrust columns, not a pressure
    or energy solve. No buried-inclusive height envelope overwrites identity.
    """
    points,cells=np.asarray(points,float),np.asarray(cells)
    ocean_owners=np.asarray(ocean_owners)
    query,face,weights=(np.asarray(value) for value in all_hits)
    size=len(points)
    material=np.full(size,-1,np.int32);material_weights=np.zeros((size,3))
    owner=ocean_owners.copy();crust=np.zeros(size,np.uint8)
    age,_=_ocean_fields(context,points,cells,ocean_owners)
    ocean_datum=sample_ocean_datum(context,points)
    selected_height=ocean_datum.copy()
    support=np.zeros(size)
    # A material hit is discrete, but its finite toe enters continuously.
    # Carry that participation through the stack so a lower free edge cannot
    # abruptly switch an unchanged upper face between two margin formulas.
    participation=np.zeros(size)
    def thermal_at(where,owners):
        uids=np.array([context['uids'][int(value)] for value in owners],np.int64)
        return ridge_interaction.sample_ridge_effects(episodes,where,uids,float(time_myr))['thermal_support_m']
    raw_thermal=thermal_at(points,owner);display_thermal=raw_thermal.copy()
    coast = 'collision_coast' in context
    if coast:
        interior_raw=np.zeros(size);interior_support=np.zeros(size);interior_thermal=np.zeros(size)
        interior_reference=np.zeros(size)
        ordinary_mass=np.zeros(size);ocean_thermal=raw_thermal.copy()
    stencil=context['material_reconstruction'];sheets=context['material_collision_sheet']
    for sheet in context['collision_sheet_order']:
        selected=np.flatnonzero(sheets[face]==sheet)
        if not len(selected):continue
        selected=selected[np.lexsort((context['material_face_id'][face[selected]],query[selected]))]
        selected=selected[np.r_[True,query[selected[1:]]!=query[selected[:-1]]]]
        q,f,w=query[selected],face[selected],weights[selected]
        raw=material_reconstruction.sample_hits(stencil,context['material_height_m_vertices'],f,w)
        physical=material_reconstruction.sample_hits(stencil,context['material_collision_support_m_vertices'],f,w)
        thermal=thermal_at(points[q],context['material_owner'][f])
        base,base_support=selected_height[q].copy(),support[q].copy()
        base_participation=participation[q].copy()
        base_thermal=display_thermal[q].copy()
        next_raw=raw.copy();next_support=physical.copy();fraction=np.ones(len(q));entry=np.ones(len(q))
        arc=context['material_arc_id'][f]>0
        if np.any(arc):
            top_weight=1.-arc_surface.ocean_weight(context['arc_surface'],f[arc],w[arc])
            # Use the actual retained interior nodal contribution, not simply
            # a scalar blend of the full nodal mean at an open volcanic edge.
            next_raw[arc]=arc_surface.apply(context['arc_surface'],context['material_height_m_vertices'],
                                            f[arc],w[arc],base[arc])
            next_support[arc]=arc_surface.apply(context['arc_surface'],context['material_collision_support_m_vertices'],
                                                f[arc],w[arc],base_support[arc])
            fraction[arc]=top_weight
            entry[arc]=top_weight
        ordinary=~arc
        if np.any(ordinary):
            margin=context['continental_margin']
            x=np.clip(continental_margin.distance_km(margin,points[q[ordinary]],f[ordinary])/
                np.maximum(margin['component_width_km'][margin['components'][f[ordinary]]],1e-12),0.,1.)
            blend=x*x*(3.-2.*x)
            weight=base_participation[ordinary]
            kw=dict(shelf_depth_m=margin['profile']['shelf_depth_m'])
            shelf_raw=continental_margin.profile_height(raw[ordinary],base[ordinary],x,**kw)
            shelf_total=continental_margin.profile_height(raw[ordinary]+physical[ordinary],
                                                           base[ordinary]+base_support[ordinary],x,**kw)
            contact_raw=base[ordinary]+blend*(raw[ordinary]-base[ordinary])
            contact_support=base_support[ordinary]+blend*(physical[ordinary]-base_support[ordinary])
            # Both endpoint laws meet the resolved underlying components at
            # x=0 and retain the upper column at x=1. Blend their components
            # by the underlying toe's continuous participation, not hit kind.
            next_raw[ordinary]=(1.-weight)*shelf_raw+weight*contact_raw
            next_support[ordinary]=(1.-weight)*(shelf_total-shelf_raw)+weight*contact_support
            fraction[ordinary]=(1.-weight)*x+weight*blend
            entry[ordinary]=blend
        if coast:
            raw_part=entry*raw;support_part=entry*physical
            if collision_coast.version(context) >= 2:
                reference_part=entry*material_reconstruction.sample_hits(
                    stencil,context['coastal_reference_vertices_m'],f,w)
                if np.any(arc):
                    reference_part[arc]=arc_surface.apply(context['arc_surface'],
                        context['coastal_reference_vertices_m'],f[arc],w[arc],np.zeros(arc.sum()))
                interior_reference[q]=(1.-entry)*interior_reference[q]+reference_part
            if np.any(arc):
                raw_part[arc]=arc_surface.apply(context['arc_surface'],context['material_height_m_vertices'],f[arc],w[arc],np.zeros(arc.sum()))
                support_part[arc]=arc_surface.apply(context['arc_surface'],context['material_collision_support_m_vertices'],f[arc],w[arc],np.zeros(arc.sum()))
            interior_raw[q]=(1.-entry)*interior_raw[q]+raw_part
            interior_support[q]=(1.-entry)*interior_support[q]+support_part
            interior_thermal[q]=(1.-entry)*interior_thermal[q]+entry*thermal
            ordinary_mass[q]=(1.-entry)*ordinary_mass[q]+entry*ordinary
        selected_height[q],support[q]=next_raw,next_support
        participation[q]=np.clip(entry+(1.-entry)*base_participation,0.,1.)
        display_thermal[q]=base_thermal+fraction*(thermal-base_thermal)
        raw_thermal[q]=thermal
        material[q],material_weights[q]=f,w
        owner[q],crust[q]=context['material_owner'][f],context['material_kind'][f]
    if coast:
        collision_coast.finish(context,points,material,participation,ordinary_mass,interior_raw,
            interior_support,interior_thermal,dict(selected_sheet_height_m=selected_height,
                physical_stack_support_m=support,display_thermal_support_m=display_thermal,
                ocean_datum_height_m=ocean_datum,ocean_thermal=ocean_thermal),
            reference=interior_reference)
    baseline=selected_height+support
    raw_elevation=baseline+display_thermal
    residual=raw_elevation-(selected_height+support+display_thermal)
    return dict(material_face=material,material_weights=material_weights,baseline_height_m=baseline,
                selected_sheet_height_m=selected_height,physical_stack_support_m=support,
                display_thermal_support_m=display_thermal,raw_thermal_support_m=raw_thermal,
                raw_elevation_m=raw_elevation,clipping_delta_m=np.zeros(size),collision_surface_offset_m=residual,
                owner=owner,crust=crust,ocean_age_myr=age,ocean_datum_height_m=ocean_datum,
                selected_ocean_age_myr=age.copy(),selected_ocean_owner=ocean_owners.copy())


def select_exposed_material(context, points, cells, ocean_owners, all_hits, *, episodes=(), time_myr=0.):
    """Retain a continuous surface while persistent contacts select provenance.

    The surface envelope remains a reconstruction; underlying column heights
    do not become a second tectonic uplift. Within actual overlapping sheets,
    the established upper sheet retains ownership even after height crossings.
    Its exposed area controls physical erosion in the native engine. Unversioned
    saved worlds use exactly their original selection path.
    """
    if context.get('collision_surface_version',0)==1:
        return _select_physical_stack(context,points,cells,ocean_owners,all_hits,episodes=episodes,time_myr=time_myr)
    exposed = _select_surface_envelope(context, points, cells, ocean_owners, all_hits,
        episodes=episodes, time_myr=time_myr)
    if context.get('collision_contact_version', 0) == 1:
        exposed['collision_surface_offset_m'] = np.zeros(len(points))
    if context.get('collision_contact_version', 0) != 1 or not context.get('collision_contacts'):
        return exposed
    query, face, weights = (np.asarray(value) for value in all_hits)
    allowed = collision_contacts.eligible_hits(query, face, context['material_collision_sheet'],
                                               context['collision_contacts'])
    if np.all(allowed): return exposed
    selected = _select_surface_envelope(context, points, cells, ocean_owners,
        (query[allowed], face[allowed], weights[allowed]), episodes=episodes, time_myr=time_myr)
    # A finite footprint's reconstructed toe can be below the other sheet's
    # surface. Keep the continuous outer envelope while recording the actual
    # upper material's identity, barycentric coordinates and raw owner heat.
    total = exposed['baseline_height_m']+exposed['display_thermal_support_m']
    selected['collision_surface_offset_m'] = total-(selected['baseline_height_m']+selected['display_thermal_support_m'])
    selected['baseline_height_m'] = total-selected['display_thermal_support_m']
    return selected


def sample_frame(frame, points, prepared=None):
    """Return native height, ownership, crust, boundary and material identity.

    Legacy overlapping material prefers the current control owner, then the
    smallest persistent face ID. Arc surface version two exposes the geometric
    envelope of actual contained volcanic patches and their underlying bedrock.
    """
    context = prepare(frame) if prepared is None else prepared
    points = np.asarray(points, float)
    cells, _ = geometry.locate_points(points, context['locator'])
    if np.any(cells < 0):
        raise ValueError('Saved native control mesh does not cover the sphere.')
    owners = context['mesh_plate'][cells].copy()
    if context['owner_reconstruction_version'] == 1:
        owners = native_boundary_geometry.sample_owners(points, context['owner_sampling'])
    ocean_owners = owners.copy()
    material = np.full(len(points), -1, np.int32)
    material_weights = np.zeros((len(points), 3))
    query, face, weights = geometry.locate_points(points, context['material_locator'], all_hits=True)
    if context['arc_surface_version'] == 2:
        exposed = select_exposed_material(context, points, cells, ocean_owners, (query, face, weights),
                                          episodes=frame.get('ridge_episodes', ()), time_myr=frame['time_myr'])
        material = exposed['material_face']
        material_weights = exposed['material_weights']
        land = material >= 0
        erosion = np.zeros(len(points))
        erosion[land] = material_reconstruction.sample_hits(context['material_reconstruction'],
            context['material_erosion_rate_m_myr_vertices'], material[land], material_weights[land])
        elevation=(exposed['raw_elevation_m'] if context.get('collision_surface_version',0)==1 else
                   np.clip(exposed['baseline_height_m']+exposed['display_thermal_support_m'],-10000.,9000.))
        result = dict(elevation=elevation,
                    plate=exposed['owner'], crust=exposed['crust'], boundary=context['mesh_boundary'][cells],
                    native_face=cells, material_face=material, material_weights=material_weights,
                    thermal_support_m=exposed['raw_thermal_support_m'],
                    display_thermal_support_m=exposed['display_thermal_support_m'], erosion_rate_m_myr=erosion)
        if 'collision_surface_offset_m' in exposed:
            result['collision_surface_offset_m'] = exposed['collision_surface_offset_m']
        for name in ('selected_sheet_height_m','physical_stack_support_m','raw_elevation_m','clipping_delta_m',
                     'ocean_datum_height_m','selected_ocean_age_myr','selected_ocean_owner'):
            if name in exposed:result[name]=exposed[name]
        return result
    if len(query):
        preferred = context['material_owner'][face] == owners[query]
        order = np.lexsort((context['material_face_id'][face], ~preferred, query))
        query, face, weights = query[order], face[order], weights[order]
        selected = np.r_[True, query[1:] != query[:-1]]
        material[query[selected]] = face[selected]
        material_weights[query[selected]] = weights[selected]
    land = material >= 0
    crust = np.zeros(len(points), np.uint8)
    erosion = np.zeros(len(points))
    owners[land] = context['material_owner'][material[land]]
    crust[land] = context['material_kind'][material[land]]
    elevation = np.empty(len(points))
    elevation[land] = context['material_height_m'][material[land]]
    erosion[land] = context['material_erosion_rate_m_myr'][material[land]]
    if context['surface_reconstruction_version'] == 1:
        stencil = context['material_reconstruction']
        elevation[land] = material_height_at_hits(context, points[land], material[land], material_weights[land],
                                                 cells[land], ocean_owners[land])
        erosion[land] = material_reconstruction.sample_hits(stencil, context['material_erosion_rate_m_myr_vertices'],
                                                           material[land], material_weights[land])
    ocean = ~land
    if ocean.any():
        age, relief = _ocean_fields(context, points[ocean], cells[ocean], owners[ocean])
        elevation[ocean] = np.where(age < 20., -2600.-365.*np.sqrt(age),
                                    -5651.+2473.*np.exp(-.0278*age))+relief
    uids = np.empty(len(points), np.int64)
    for owner in np.unique(owners):
        uids[owners == owner] = context['uids'][int(owner)]
    thermal = ridge_interaction.sample_ridge_effects(frame.get('ridge_episodes', []), points, uids,
                                                    float(frame['time_myr']))['thermal_support_m']
    display_thermal = thermal.copy()
    if context['arc_surface_version'] == 1 and land.any():
        display_thermal[land] = material_thermal_at_hits(context['arc_surface'], points[land],
            material[land], material_weights[land], thermal[land], ocean_owners[land], context['uids'],
            frame.get('ridge_episodes', []), frame['time_myr'])
    elevation = np.clip(elevation+display_thermal, -10000., 9000.)
    return dict(elevation=elevation, plate=owners, crust=crust,
                boundary=context['mesh_boundary'][cells], native_face=cells,
                material_face=material, material_weights=material_weights,
                thermal_support_m=thermal, display_thermal_support_m=display_thermal, erosion_rate_m_myr=erosion)


def prepare_material_correction(context, losses, supported):
    """Project exact face-counter losses with the same operator as height.

The authoritative per-face losses are untouched. Matching the projection
avoids manufacturing tectonic uplift from spatially varying pure erosion on
an unchanged material topology. Geometry/topology changes remain a sampling
limitation; this is not a conservative remeshing or new denudation ledger.
"""
    if context['surface_reconstruction_version'] == 0:
        return dict(losses=np.asarray(losses), supported=np.asarray(supported))
    stencil = context['material_reconstruction']
    return dict(losses=material_reconstruction.vertex_values(stencil, losses),
                supported=material_reconstruction.vertex_values(stencil, supported))


def sample_material_correction(context, sampled, correction):
    """Return spatially consistent loss and conservative marker-support flags."""
    material = sampled['material_face']
    land = material >= 0
    loss = np.zeros(len(material))
    supported = np.zeros(len(material), bool)
    if context['surface_reconstruction_version'] == 0:
        loss[land] = correction['losses'][material[land]]
        supported[land] = correction['supported'][material[land]]
    else:
        stencil = context['material_reconstruction']
        weights = sampled['material_weights'][land]
        loss[land] = material_reconstruction.sample_hits(stencil, correction['losses'], material[land], weights)
        fraction = material_reconstruction.sample_hits(stencil, correction['supported'], material[land], weights)
        supported[land] = fraction >= 1.-1e-12
    return loss, supported


def material_erosion_correction(first, second, a, b, prepared=None):
    """Signed marker-counter losses keyed to their persistent source faces.

    Each native face remains one material column. Faces lacking a surviving
    representative marker use the saved net denudation rate times the interval,
    with explicit support=0. The optional continuous surface projection is
    applied later by prepare_material_correction; it does not alter this ledger.
    """
    context = prepare(first) if prepared is None else prepared
    duration = float(second['time_myr']-first['time_myr'])
    losses = context['material_erosion_rate_m_myr'].astype(float)*duration
    supported = np.zeros(len(losses), bool)
    if 'trace_patch' not in first or 'trace_patch' not in second:
        raise ValueError('Native material history lacks persistent marker patch IDs.')
    patch_ids = context['material_face_id']
    order = np.argsort(patch_ids)
    patches = np.asarray(first['trace_patch'])[a]
    locations = np.searchsorted(patch_ids[order], patches)
    valid = locations < len(patch_ids)
    face = np.zeros(len(a), np.int64)
    face[valid] = order[locations[valid]]
    if len(patch_ids):
        valid &= patch_ids[face] == patches
    else:
        valid[:] = False
    valid &= patches == np.asarray(second['trace_patch'])[b]
    marker_losses = np.asarray(second['trace_erosion_m'])[b]-np.asarray(first['trace_erosion_m'])[a]
    if not np.isfinite(marker_losses).all():
        raise ValueError('Saved erosion counters contain non-finite values.')
    if valid.any():
        owner_uids = np.array([context['uids'][int(x)] for x in context['material_owner'][face[valid]]])
        valid_indices = np.flatnonzero(valid)
        valid[valid_indices] &= owner_uids == np.asarray(first['trace_plate_uid'])[a[valid]]
        count = np.bincount(face[valid], minlength=len(losses))
        sums = np.bincount(face[valid], weights=marker_losses[valid], minlength=len(losses))
        supported = count > 0
        losses[supported] = sums[supported]/count[supported]
    return losses, supported


def prepare_live_surface(s):
    """One physical surface context shared by live rendering and erosion."""
    import structure_engine, native_arc_material
    heights = structure_engine.material_height(s.kind, s.relief)
    context = dict(mesh=s.native_mesh, locator=s.native_locator, stencil=s.native_stencil,
                   mesh_crust=s.crust, mesh_plate=s.plate, mesh_age_myr=s.age,
                   mesh_ocean_relief_m=s.ocean_relief)
    reconstruction = material_reconstruction.prepare(s.material_surface['vertices'], s.material_surface['faces'],
        s.parcel_plate, area_km2=s.material_surface['area_km2'])
    nodal_height = material_reconstruction.vertex_values(reconstruction, heights)
    arc_fields = native_arc_material.snapshot_fields(s)
    arc_reconstruction = arc_surface.prepare(reconstruction, arc_fields['material_arc_id'])
    context.update(material_reconstruction=reconstruction, material_height_m_vertices=nodal_height,
                   arc_surface=arc_reconstruction, arc_surface_version=2, surface_reconstruction_version=1,
                   material_owner=s.parcel_plate, material_kind=s.kind, material_face_id=s.parcel_patch,
                   material_arc_id=arc_fields['material_arc_id'], material_height_m=heights, uids=s.plate_uid)
    context.update(collision_contact_version=1, material_collision_sheet=s.parcel_collision_sheet,
                   collision_contacts=s.collision_contacts)
    margin_version = getattr(s, 'continental_margin_version', 0)
    if margin_version == 1:
        context['continental_margin'] = continental_margin.prepare(
            s.material_surface['vertices'], s.material_surface['faces'], s.kind,
            arc_fields['material_arc_id'], profile=s.continental_margin_parameters)
    if getattr(s, 'collision_surface_version', 0) == 1:
        context.update(collision_surface.snapshot_fields(s))
        context['retained_dense_crust_version']=getattr(s,'retained_dense_crust_version',0)
        prepare_collision_surface(context,
            s.material_surface['vertices'], s.material_surface['faces'])
    return context
