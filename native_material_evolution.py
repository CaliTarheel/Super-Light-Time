"""Couple moving material geometry, crustal columns, and enduring histories."""
from __future__ import annotations

import numpy as np
import material_surface as surface
import deforming_regions
import collision_contacts
import world_design
import gravitational_relaxation
import enhanced_rifting
import column_density
import evolution_policy
from crustal_structure import (MIN_THICKNESS_KM, MAX_THICKNESS_KM,
                               conserved_volume_floor, restored_thickness)
from ridge_geometry import rotate


def _unit(value):
    return value/np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-30)


def gravitational_reference(s):
    """Keep imported reference relief separate from new magma's mechanical load."""
    reference = s.structure['reference_thickness_km'].copy()
    if getattr(s, 'native_arc_birth_profile_version', 0) == 1:
        # The thin retained-column floor is an explicit reduced balancing
        # reference, not an independently solved ocean/mantle background. Its
        # physical value follows exact retained-phase volume loss so density
        # conversion is not opposed by a fictitious 8 km restoring target.
        arc = np.asarray(s.parcel_arc_id) > 0
        reference[arc] = conserved_volume_floor(s.structure)[arc]
    return reference


def reference_energy_state(s):
    mesh = s.material_surface
    return gravitational_relaxation.reference_energy(mesh['vertices'], mesh['faces'],
        mesh['area_km2']*s.structure['thickness_km'], gravitational_reference(s),
        s.parcel_collision_sheet, radius=mesh.get('radius_km', 6371.), **column_density.options(s))


def ensure_lineage(s):
    """Initialize births and retain exact projective ancestry across owner changes."""
    import entry_regions
    entry_regions.ensure_fields(s)
    ids = s.parcel_patch
    old = getattr(s, 'material_lineage', None)
    if old is not None and np.array_equal(old['face_ids'], ids):
        return
    count = len(ids)
    lineage = dict(face_ids=ids.copy(), root_id=ids.copy(), parent_id=np.full(count, -1, np.int64),
                   reference_corners=np.broadcast_to(np.eye(3), (count, 3, 3)).copy(),
                   level=np.zeros(count, np.int32))
    now, before = np.empty(0, int), np.empty(0, int)
    if old is not None:
        _, now, before = np.intersect1d(ids, old['face_ids'], return_indices=True)
        for key in lineage:
            if key != 'face_ids':
                lineage[key][now] = old[key][before]
    s.material_lineage = lineage
    if not hasattr(s, 'initial_material_faces'):
        s.initial_material_faces = count
    previous = getattr(s, 'material_deformation', {})
    state = dict(face_weight=np.zeros(count), face_rigid=np.ones(count, bool),
                 face_strain=np.zeros(count))
    for key in state:
        if key in previous and old is not None and len(previous[key]) == len(old['face_ids']):
            state[key][now] = previous[key][before]
    s.material_deformation = state
    if hasattr(s, 'geometric_log_area'):
        change = np.zeros(count)
        if old is not None and len(s.geometric_log_area) == len(old['face_ids']):
            change[now] = s.geometric_log_area[before]
        s.geometric_log_area = change


def _transport_tangent(old_triangles, new_triangles, tangents, destinations):
    coefficients = np.linalg.solve(np.swapaxes(old_triangles, 1, 2), tangents[..., None])[..., 0]
    result = np.einsum('ni,nij->nj', coefficients, new_triangles)
    result -= destinations*np.sum(result*destinations, axis=1)[:, None]
    return _unit(result)


def material_loading_boundaries(s, mechanics_version):
    """Use each real finite interface once for material loading.

    Control-contact averages are force quadrature points. Combining a curved
    or disconnected set of source segments into one straight arc changes which
    material vertices touch a loading front. Preserve the segment geometry and
    use its parent only for ownership and stale-contact admission.

    Native owner interfaces are already cut in the material topology. Their
    reconstructed support contours cannot supply tensile attachment, even when
    a contour drifts through an intact triangle on one of the separated shores.
    Explicitly supplied internal loading fronts retain their own contact gate.
    """
    boundaries = {key: getattr(s, key) for key in ('bmid', 'bn', 'bl', 'bp', 'bq')}
    if hasattr(s, '_valid_loading_edges'):
        boundaries['valid'] = s._valid_loading_edges()
    geometry = getattr(s, 'native_boundary_geometry', None)
    description = dict(representation='supplied finite loading fronts', parent_contacts=len(boundaries['bl']))
    if mechanics_version == 1 and geometry is not None:
        first = np.asarray(geometry['segments_start'], float)
        last = np.asarray(geometry['segments_end'], float)
        normals = np.asarray(geometry['segment_normals'], float)
        parent = np.asarray(geometry['contact_index'])
        count = len(parent)
        if (first.shape != (count, 3) or last.shape != first.shape or normals.shape != first.shape
                or parent.shape != (count,) or parent.dtype.kind not in 'iu'
                or np.any(parent < 0) or np.any(parent >= len(boundaries['bl']))
                or not np.isfinite(first).all() or not np.isfinite(last).all() or not np.isfinite(normals).all()
                or np.any(np.abs(np.linalg.norm(first, axis=1)-1.) > 1e-8)
                or np.any(np.abs(np.linalg.norm(last, axis=1)-1.) > 1e-8)):
            raise ValueError('Material loading requires aligned finite native interface geometry.')
        middle = _unit(first+last)
        normal = _unit(normals-middle*np.sum(normals*middle, axis=1)[:, None])
        lengths = np.arctan2(np.linalg.norm(np.cross(first, last), axis=1), np.sum(first*last, axis=1))
        lengths *= float(s.material_surface.get('radius_km', 6371.))
        if np.any((lengths > 1e-10) & (np.linalg.norm(normal, axis=1) < .5)):
            raise ValueError('A positive finite material loading segment needs a resolved tangent normal.')
        valid = np.asarray(boundaries.get('valid', np.ones(len(boundaries['bl']), bool)), bool)[parent]
        owner_a = np.asarray(boundaries['bp'])[parent]
        owner_b = np.asarray(boundaries['bq'])[parent]
        opening = np.sum(np.cross(s.omega[owner_b]-s.omega[owner_a], middle)*normal, axis=1)
        opening *= float(s.material_surface.get('radius_km', 6371.))
        # Same angular-roundoff tolerance as the opening contact gate. This is
        # not the ocean-production speed threshold: slow or oblique separation
        # cannot restore a bond across an ownership cut either. Reject the whole
        # target, including shear, while retaining convergence and transforms.
        unbonded = valid & (lengths > 1e-10) & (owner_a != owner_b) & (opening > 1e-10)
        valid = valid & ~unbonded
        boundaries = dict(bmid=middle, bn=normal, bl=lengths,
            bp=owner_a, bq=owner_b, valid=valid)
        description.update(representation='finite native boundary segments', segments=count,
            valid_segments=int(valid.sum()), invalid_segments=int((~valid).sum()),
            valid_length_km=float(lengths[valid].sum()), invalid_length_km=float(lengths[~valid].sum()),
            unbonded_opening_segments=int(unbonded.sum()),
            unbonded_opening_length_km=float(lengths[unbonded].sum()),
            opening_contact_rule='Native ownership cuts have no tensile attachment; support contours cannot reattach a separated margin.')
    elif mechanics_version == 1 and hasattr(s, 'native_mesh'):
        raise ValueError('Native material mechanics requires its finite boundary geometry.')
    boundaries = collision_contacts.deformation_boundaries(s, boundaries)
    description['total_loading_fronts'] = len(boundaries['bl'])
    return boundaries, description


_MAX_SAME_SHEET_CONTACT_PASSES = 16


def _expand_same_sheet_clamp(points, faces, sheets, protected, *, stage,
                             radius_km=6371.):
    """Add every vertex of an intersecting same-sheet face pair to a clamp.

    This is an active-set update only.  It does not move a vertex or mutate the
    supplied mask.  The caller must discard the candidate endpoint and rerun
    the complete contact/gravity calculation from the original geometry.
    """
    import mesh_coverage
    faces = np.asarray(faces)
    mask = np.asarray(protected)
    if (mask.shape != (len(points),) or mask.dtype.kind != 'b'):
        raise ValueError('Same-sheet nonpenetration needs one Boolean constraint per vertex.')
    overlap = mesh_coverage.same_sheet_overlaps(
        points, faces, sheets, radius_km=radius_km)
    if not len(overlap['first']):
        return mask.copy(), None
    affected_faces = np.unique(np.r_[overlap['first'], overlap['second']])
    affected_vertices = np.unique(faces[affected_faces])
    result = mask.copy()
    added = affected_vertices[~result[affected_vertices]]
    result[affected_vertices] = True
    pairs = np.column_stack((overlap['first'], overlap['second']))
    event = dict(stage=str(stage), overlap_pairs=int(len(pairs)),
        overlap_area_km2=float(np.sum(overlap['area_km2'])),
        maximum_overlap_area_km2=float(np.max(overlap['area_km2'], initial=0.)),
        pairs=pairs.tolist(), faces=affected_faces.tolist(),
        vertices=affected_vertices.tolist(), added_vertices=added.tolist(),
        exact_geometry='spherical great-circle polygon intersection')
    return result, event


def advect(s, dt, *, prepared_loading=None):
    """Move shared vertices once; markers and thickness use that same deformation."""
    ensure_lineage(s)
    import numerical_accuracy
    accuracy_version=numerical_accuracy.version(s)
    if accuracy_version:numerical_accuracy.validate_simulation(s)
    mesh = s.material_surface
    surface.reassign_owners(mesh, s.parcel_plate)
    old_triangles = mesh['vertices'][mesh['faces']].copy()
    rift_motion = enhanced_rifting.capture_motion(s)
    s.material_pre_motion_volume_km3 = float(mesh['area_km2']@s.structure['thickness_km'])
    if getattr(s,'retained_dense_crust_version',0)==1:
        import dense_crust
        s.material_pre_motion_mass_kg=float(s.mass@dense_crust.mass_volume(s.structure))*2800.*1e9
    order = np.argsort(s.parcel_patch)
    found = np.searchsorted(s.parcel_patch[order], s.trace_patch)
    if np.any(found >= len(order)) or np.any(s.parcel_patch[order[found]] != s.trace_patch):
        raise ValueError('Every material marker must retain its containing face ancestry.')
    face = order[found]
    enabled = bool(s.config.get('deforming_regions', 1))
    if enabled:
        mechanics_version = getattr(s, 'material_mechanics_version', 0)
        if isinstance(mechanics_version, (bool, np.bool_)) or mechanics_version not in (0, 1):
            raise ValueError('Unsupported material mechanics version.')
        boundaries, loading_geometry = (material_loading_boundaries(s, mechanics_version)
            if prepared_loading is None else prepared_loading)
        minimum = (s.structure['thickness_km']/MAX_THICKNESS_KM if accuracy_version else
            np.minimum(1., s.structure['thickness_km']/MAX_THICKNESS_KM))
        maximum = (restored_thickness(s.structure)/MIN_THICKNESS_KM if accuracy_version else
            np.maximum(1., restored_thickness(s.structure)/MIN_THICKNESS_KM))
        # Point histories have their own erosion/magma budgets. A realized
        # face strain must also be admissible for its retained marker columns.
        if hasattr(s, 'trace_structure'):
            trace_min=s.trace_structure['thickness_km']/MAX_THICKNESS_KM
            trace_max=restored_thickness(s.trace_structure)/MIN_THICKNESS_KM
            np.maximum.at(minimum, face, trace_min if accuracy_version else np.minimum(1.,trace_min))
            np.minimum.at(maximum, face, trace_max if accuracy_version else np.maximum(1.,trace_max))
        accuracy_policy=numerical_accuracy.policy(s,mesh['area_km2']*minimum,mesh['area_km2']*maximum)
        design_options = dict(world_design.deformation_options(s))
        authored_protected = design_options.pop('vertex_protected', None)
        if authored_protected is None:
            authored_protected = np.zeros(len(mesh['vertices']), bool)
        else:
            authored_protected = np.asarray(authored_protected)
            if (authored_protected.shape != (len(mesh['vertices']),)
                    or authored_protected.dtype.kind != 'b'):
                raise ValueError('World-design protection must align with material vertices.')
            authored_protected = authored_protected.copy()
        viscosity_weights = enhanced_rifting.material_viscosity(s)
        craton_anchors = None
        craton_diagnostic = None
        if enhanced_rifting.enabled(s):
            craton_anchors, craton_diagnostic = enhanced_rifting.craton_anchors(mesh,
                s.config['enhanced_rifting'].get('craton_margin_km', 150.))
        transient_protected = np.zeros(len(mesh['vertices']), bool)
        contact_events = []
        protected_contact_faces = set()
        radius = mesh.get('radius_km', 6371.)
        collision_sheets = np.asarray(getattr(
            s, 'parcel_collision_sheet', np.arange(len(mesh['faces']), dtype=np.int64)))
        nonpenetration_version = getattr(s, 'same_sheet_nonpenetration_version', 0)
        if (isinstance(nonpenetration_version, (bool, np.bool_))
                or nonpenetration_version not in (0, 1)):
            raise ValueError('Unsupported same-sheet nonpenetration version.')
        enforce_same_sheet = nonpenetration_version == 1
        if (collision_sheets.shape != (len(mesh['faces']),)
                or collision_sheets.dtype.kind not in 'iu'):
            raise ValueError('Collision sheets must align with material faces.')
        accepted_admission = None
        entry_after = None
        if mechanics_version:
            conserved_volume = mesh['area_km2']*s.structure['thickness_km']
            import entry_regions
            entry_before=entry_regions.frozen(s) if entry_regions.enabled(s) else None
            entry_after=entry_regions.frozen(s,advance_myr=dt) if entry_before is not None else None
            if entry_before is not None:
                # Both potentials must use the same conserved inventory. A
                # second reconstruction from geometric area and rounded column
                # thickness can differ even when reference-volume bookkeeping
                # is unchanged (notably on newly born, small material faces).
                conserved_volume=entry_before.volumes.copy()
            reference = gravitational_reference(s)
            before_energy = gravitational_relaxation.reference_energy(mesh['vertices'], mesh['faces'],
                conserved_volume, reference, s.parcel_collision_sheet, radius=radius,
                **column_density.options(s))
            if entry_before is not None:
                before_energy+=entry_before.evaluate(mesh['vertices'],mesh['faces'],conserved_volume,
                    s.parcel_collision_sheet,radius=radius)['energy_j']/gravitational_relaxation.ENERGY_UNIT_J
            # Version 1 of the plate balance reads the rigid share of this same
            # potential as its collision torque, so the sheet must relax only
            # the residual. Version 0 passes nothing and is bitwise unchanged.
            owned = (np.asarray(mesh['vertex_owner']) if getattr(s, 'plate_balance_version', 0) == 1
                     else None)
            density_base=column_density.options(s)

        # A positive-area intersection of one collision sheet is an inadmissible
        # in-plane penetration, not another vertical layer.  Grow a transient
        # active set by complete offending faces and recompute every candidate
        # from the unchanged source geometry.  No rejected endpoint is committed.
        for contact_pass in range(1, _MAX_SAME_SHEET_CONTACT_PASSES+1):
            protected = authored_protected | transient_protected
            result = deforming_regions.deform(mesh, s.omega, boundaries, dt,
                belt_width_km=s.config.get('deformation_width_km', 400.),
                face_min_area_ratio=minimum, face_max_area_ratio=maximum,
                mechanics_version=mechanics_version,
                iterations=1024 if mechanics_version else 256,
                tolerance=(numerical_accuracy.OUTER_RELATIVE_TOLERANCE if accuracy_version else
                    1e-8 if mechanics_version else 1e-7),
                require_material_contact=True, viscosity_weights=viscosity_weights,
                craton_rigid_vertices=craton_anchors, vertex_protected=protected,
                require_complete_contact=evolution_policy.complete_contact_version(s) == 1,
                redistribute_limited_contact=getattr(s,'native_contact_constraint_version',0)==1,
                **({} if accuracy_policy is None else dict(numerical_policy=accuracy_policy)),
                **design_options)
            if craton_diagnostic is not None:
                result['diagnostics']['craton_anchors'] = craton_diagnostic
            if mechanics_version and not result['diagnostics']['solver']['converged']:
                failed=result['diagnostics']['solver']
                raise ValueError('Collision sheet solve failed its recomputed stationarity gate: '
                    f"{failed['failure_reason']}; iterations {failed['iterations']}/{failed['maximum_iterations']}; "
                    f"true relative residual {failed['relative_residual']}, required {failed['tolerance']}.")
            expanded, event = (_expand_same_sheet_clamp(result['vertices'], mesh['faces'],
                collision_sheets, protected, stage='deformation', radius_km=radius)
                if enforce_same_sheet else (protected, None))
            if event is not None:
                event['pass'] = contact_pass
                event['patch_ids'] = np.asarray(s.parcel_patch)[event['faces']].astype(int).tolist()
                event['sheet_ids'] = np.unique(collision_sheets[event['faces']]).astype(int).tolist()
                contact_events.append(event); protected_contact_faces.update(event['faces'])
                if not event['added_vertices']:
                    raise deforming_regions.IncompleteContactStepError(
                        'Same-sheet nonpenetration active set found an unresolved deformation overlap.')
                transient_protected = expanded & ~authored_protected
                continue
            admission = None
            accepted_policy=numerical_accuracy.accepted_contact_policy(accuracy_policy,result,
                surface.spherical_face_areas(result['vertices'],mesh['faces'],radius))
            if mechanics_version:
                density_options=dict(density_base)
                if density_options:
                    admission=column_density.ContactAdmission(s,conserved_volume)
                    admission.accept(admission.prepare(result['vertices']))
                    density_options=dict(density_profile=admission.profile,density_admission=admission)
                relaxed, gravity = gravitational_relaxation.relax(result['vertices'], mesh['faces'],
                    conserved_volume, reference, s.parcel_collision_sheet, dt,
                    rigid_mask=result['rigid_mask'] | transient_protected,
                    minimum_area_km2=mesh['area_km2']*minimum,
                    maximum_area_km2=mesh['area_km2']*maximum, radius=radius,
                    constraint_version=getattr(s, 'native_gravity_constraint_version', 0),
                    viscosity_weights=viscosity_weights, vertex_owner=owned,
                    **({} if accepted_policy is None else dict(numerical_policy=accepted_policy,
                        tolerance=numerical_accuracy.OUTER_RELATIVE_TOLERANCE)),
                    entry_potential=entry_after, **density_options)
                expanded, event = (_expand_same_sheet_clamp(relaxed, mesh['faces'],
                    collision_sheets, protected, stage='gravity', radius_km=radius)
                    if enforce_same_sheet else (protected, None))
                if event is not None:
                    event['pass'] = contact_pass
                    event['patch_ids'] = np.asarray(s.parcel_patch)[event['faces']].astype(int).tolist()
                    event['sheet_ids'] = np.unique(collision_sheets[event['faces']]).astype(int).tolist()
                    contact_events.append(event); protected_contact_faces.update(event['faces'])
                    if not event['added_vertices']:
                        raise deforming_regions.IncompleteContactStepError(
                            'Same-sheet nonpenetration active set found an unresolved gravity overlap.')
                    transient_protected = expanded & ~authored_protected
                    continue
                if owned is not None and gravity.get('rigid_mode_partition'):
                    # The balance's collision torques supplied the rigid share
                    # of this same potential over the accepted candidate only.
                    torques = gravity['rigid_mode_partition']['owner_torque_n_m']
                    gravity['rigid_mode_work_j'] = float(sum(
                        np.dot(np.asarray(value, float), s.omega[int(plate)])*dt
                        for plate, value in torques.items() if int(plate) < len(s.omega)))
                    if entry_after is not None:
                        spec=entry_regions.specification(s,advance_myr=dt)
                        entry_force=entry_after.evaluate(result['vertices'],mesh['faces'],conserved_volume,
                            s.parcel_collision_sheet,radius=radius)
                        slots={int(uid):i for i,uid in enumerate(s.plate_uid)}
                        hinge_work=-float(sum(np.dot(torque,s.omega[slots[int(uid)]])*dt
                            for torque,uid in zip(entry_force['hinge_potential_torque_n_m'],spec['overriding_plate_uids'])))
                        gravity['entry_hinge_rigid_work_j']=hinge_work
                        gravity['rigid_mode_work_j']+=hinge_work
                    gravity['rigid_mode_work_scope'] = 'first-substep torque times solved plate rotation over dt'
                gravity['energy_before_prescribed_motion_km4'] = before_energy
                gravity['prescribed_motion_energy_change_km4'] = gravity['energy_before_km4']-before_energy
                gravity['prescribed_motion_scope'] = 'full plate rotation and accepted contact residual; separate from gravitational work'
                if getattr(s, 'native_arc_birth_profile_version', 0) == 1:
                    arc = np.asarray(s.parcel_arc_id) > 0
                    values = gravitational_reference(s)[arc]
                    gravity['juvenile_reference_thickness_km'] = dict(
                        minimum=float(values.min()) if len(values) else MIN_THICKNESS_KM,
                        maximum=float(values.max()) if len(values) else MIN_THICKNESS_KM,
                        model='physical thickness at the 8 km mass-equivalent reserve under conserved-volume area change')
                else:
                    gravity['juvenile_reference_thickness_km'] = None
                result['vertices'] = relaxed
                final_area = surface.spherical_face_areas(relaxed, mesh['faces'], radius)
                if accepted_policy is not None:
                    accepted_policy=numerical_accuracy.NumericalPolicy(accepted_policy.reference_area_km2,
                        accepted_policy.budget_km2,np.asarray(gravity['numerical_accuracy_final_spent_km2']),
                        final_area,accepted_policy.minimum_area_km2,accepted_policy.maximum_area_km2,
                        accepted_policy.reference_identity)
                result['areal_strain'] = np.log(final_area/mesh['area_km2'])
                result['diagnostics']['contact_stage_area_ratio_bounds'] = [
                    result['diagnostics']['minimum_area_ratio'], result['diagnostics']['maximum_area_ratio']]
                result['diagnostics']['minimum_area_ratio'] = float((final_area/mesh['area_km2']).min(initial=1.))
                result['diagnostics']['maximum_area_ratio'] = float((final_area/mesh['area_km2']).max(initial=1.))
                result['diagnostics']['gravitational_relaxation'] = gravity
                result['diagnostics']['finite_strain'] = deforming_regions.finite_strain_summary(
                    mesh['vertices'], relaxed, mesh['faces'], mesh['face_owner'],
                    result['source_boundary'], result['region_weight'], boundaries, radius=radius)
                for row in result['diagnostics']['finite_strain']['owner_rows']:
                    if hasattr(s, 'plate_uid'):
                        row['owner_uid'] = int(s.plate_uid[row['owner']])
                    if hasattr(s, 'names'):
                        row['owner_name'] = str(s.names[row['owner']])
            accepted_admission = admission
            break
        else:
            raise deforming_regions.IncompleteContactStepError(
                f'Same-sheet nonpenetration active set exceeded {_MAX_SAME_SHEET_CONTACT_PASSES} monotonic passes.')

        # An explicit pre-commit postcondition covers both the contact response
        # and gravity, even if either implementation changes independently.
        _, final_event = (_expand_same_sheet_clamp(result['vertices'], mesh['faces'],
            collision_sheets, authored_protected | transient_protected,
            stage='final', radius_km=radius) if enforce_same_sheet else (None, None))
        if final_event is not None:
            raise deforming_regions.IncompleteContactStepError(
                'Same-sheet nonpenetration postcondition failed before geometry commit.')
        result['diagnostics']['same_sheet_nonpenetration'] = dict(version=int(enforce_same_sheet),
            model=('transient full-face active-set clamp with complete contact/gravity recomputation'
                   if enforce_same_sheet else 'historical contact law; no same-sheet postcondition'),
            passes=contact_pass, reruns=contact_pass-1, events=contact_events,
            protected_vertices=np.flatnonzero(transient_protected).tolist(),
            protected_faces=sorted(protected_contact_faces),
            protected_patch_ids=np.asarray(s.parcel_patch)[np.asarray(sorted(protected_contact_faces), int)].astype(int).tolist(),
            final_overlap_pairs=0 if enforce_same_sheet else None,
            final_overlap_area_km2=0. if enforce_same_sheet else None,
            topology_changed=False, material_removed=False,
            force_coefficients_changed=False,
            limitation=('All vertices of each offending face use rigid plate motion for this interval; sub-face tangential slip and internal strain inside the clamped faces are unresolved.'
                if enforce_same_sheet else 'Historical checkpoint predates the guard; existing same-sheet overlap is not resolved or bounded by this continuation.'))
        if accepted_admission is not None:
            accepted_admission.publish(s)
        if mechanics_version and entry_after is not None:
            s.continental_entry_regions['last_mechanical_energy_j']=gravity['entry_energy_after_j']
        mesh['vertices'] = result['vertices']
        surface.refresh_geometry(mesh)
        strain = result['areal_strain'].copy()
        if not accuracy_version:strain[np.abs(strain) < 1e-12] = 0.
        s.material_deformation = dict(face_weight=result['region_weight'][mesh['faces']].max(axis=1),
            face_rigid=result['rigid_mask'][mesh['faces']].all(axis=1), face_strain=strain.copy())
        if accepted_policy is not None:
            s.material_deformation['numerical_accuracy_endpoint']={
                numerical_accuracy.AREA:mesh['area_km2'].copy(),
                numerical_accuracy.SPENT:accepted_policy.spent_km2.copy()}
            s.numerical_accuracy_diagnostics=accepted_policy.evidence(mesh['area_km2'],
                accuracy_policy.minimum_area_km2,accuracy_policy.maximum_area_km2,include_arrays=False)
            s.numerical_accuracy_diagnostics['accepted_transaction_spent_km2']=float((accepted_policy.spent_km2-accuracy_policy.spent_km2).sum())
        result['diagnostics']['limited_material_face_ids'] = s.parcel_patch[result['limited_face_mask']].tolist()
        s.deformation_diagnostics = result['diagnostics']
        s.deformation_diagnostics['loading_geometry'] = loading_geometry
    else:
        surface.advect_surface(mesh, s.omega, dt)
        strain = (np.log(mesh['area_km2']/s.structure[numerical_accuracy.AREA])
            if accuracy_version else np.zeros(len(s.mass)))
        s.material_deformation = dict(face_weight=np.zeros(len(s.mass)),
            face_rigid=np.ones(len(s.mass), bool), face_strain=strain.copy())
        if accuracy_version:
            s.material_deformation['numerical_accuracy_endpoint']={numerical_accuracy.AREA:mesh['area_km2'].copy(),
                numerical_accuracy.SPENT:s.structure[numerical_accuracy.SPENT].copy()}
        s.deformation_diagnostics = dict(model='rigid material transport', deforming_vertices=0)
    # Even rigid native transport uses its actual zero strain. Independent
    # column thinning would otherwise invent deformation inside fixed faces.
    s.geometric_log_area = strain
    predicted = mesh['area_km2']@(s.structure['thickness_km']*np.exp(-strain))
    s.material_geometry_volume_residual_km3 = float(predicted-s.material_pre_motion_volume_km3)
    new_triangles = mesh['vertices'][mesh['faces']]
    s.pos = surface.face_centres(mesh)
    s.rift_tangent = _transport_tangent(old_triangles, new_triangles, s.rift_tangent, s.pos)
    weights = np.linalg.solve(np.swapaxes(old_triangles[face], 1, 2), s.trace_xyz[..., None])[..., 0]
    weights /= weights.sum(axis=1)[:, None]
    if np.any(weights < -1e-7):
        raise ValueError('A material marker left its recorded triangle before transport.')
    s.trace_xyz = _unit(np.einsum('ni,nij->nj', weights, new_triangles[face]))
    s.trace_rift_tangent = _transport_tangent(old_triangles[face], new_triangles[face],
                                             s.trace_rift_tangent, s.trace_xyz)
    s.trace_geometric_log_area = strain[face].copy()
    s.deformation_diagnostics['material_markers_transported'] = len(face)
    enhanced_rifting.finish_motion(s, rift_motion, dt)


def snapshot_fields(s):
    ensure_lineage(s)
    lineage = s.material_lineage
    return dict(**__import__('numerical_accuracy').snapshot(s),material_transport_version=1,
        material_mechanics_version=getattr(s, 'material_mechanics_version', 0),
        gravity_constraint_version=getattr(s, 'native_gravity_constraint_version', 0),
        contact_constraint_version=getattr(s, 'native_contact_constraint_version', 0),
        material_root_id=lineage['root_id'].copy(),
        material_parent_id=lineage['parent_id'].copy(),
        material_reference_corners=lineage['reference_corners'].copy(),
        material_erosion_total_m=(s.structure['denudation_m']-s.structure['rebound_m']).copy(),
        material_refinement_level=lineage['level'].copy(),
        material_actual_area_km2=s.material_surface['area_km2'].copy(),
        material_deformation_weight=s.material_deformation['face_weight'].copy(),
        material_rigid=s.material_deformation['face_rigid'].astype(np.uint8),
        material_geometric_log_area=s.material_deformation['face_strain'].copy())
