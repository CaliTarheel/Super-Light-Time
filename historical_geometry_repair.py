"""Explicit detached reconstruction of an inherited boundary penetration.

This is a new initial-condition adjustment, never a physical timestep. One
identified boundary vertex is projected onto the exterior of an identified
boundary edge. Topology, material identities and inventories are not edited.
Column thickness and attached markers follow the resulting local area change.
The unchanged exact same-sheet oracle must report no positive intersections.
No saved checkpoint, running simulation, or force coefficient is modified here.
"""
from copy import deepcopy
import hashlib

import numpy as np

import crustal_structure
import material_surface
import mesh_coverage
from gravitational_relaxation import ENERGY_UNIT_J
from native_material_evolution import _transport_tangent, reference_energy_state


ROUND_OFF_PADDING_RADIANS = 64 * np.finfo(np.float64).eps
MAX_METRIC_RELATIVE_CORRECTION = 1e-8


def _digest(value):
    a = np.ascontiguousarray(value)
    return dict(dtype=str(a.dtype), shape=list(a.shape),
                sha256=hashlib.sha256(a.tobytes()).hexdigest())


def _unit(values):
    return values / np.linalg.norm(values, axis=-1, keepdims=True)


def _overlap(s, vertices):
    mesh = s.material_surface
    return mesh_coverage.same_sheet_overlaps(vertices, mesh['faces'],
        s.parcel_collision_sheet, radius_km=mesh['radius_km'])


def boundary_vertex_candidate(s, *, vertex_id, separating_edge, excluded_face,
                              max_displacement_km):
    """Propose a local exterior projection, without changing the supplied state.

    The edge must be an actual boundary of the excluded face's indexed sheet.
    The moved vertex must be on that same sheet and outside the excluded face's
    vertex list. 64 float64 eps radians is a declared representational outward
    padding (about 91 nanometres on Earth), not a geometric overlap tolerance.
    """
    mesh = s.material_surface
    points, faces = np.asarray(mesh['vertices']), np.asarray(mesh['faces'])
    sheets = np.asarray(s.parcel_collision_sheet)
    vertex = int(vertex_id)
    edge = np.asarray(separating_edge, dtype=np.int64)
    excluded = int(excluded_face)
    if (edge.shape != (2,) or edge[0] == edge[1]
            or not 0 <= vertex < len(points) or not 0 <= excluded < len(faces)
            or not np.all(np.isin(edge, faces[excluded])) or vertex in faces[excluded]
            or not np.isfinite(max_displacement_km) or max_displacement_km <= 0):
        raise ValueError('Repair requires distinct valid vertex, boundary edge, face and displacement cap.')
    affected = np.flatnonzero(np.any(faces == vertex, axis=1))
    if not len(affected) or np.any(sheets[affected] != sheets[excluded]):
        raise ValueError('Repair support must belong to the identified collision sheet.')
    edge_faces = np.flatnonzero(np.all(np.any(faces[:, :, None] == edge[None, None, :], axis=1), axis=1))
    edge_faces = edge_faces[sheets[edge_faces] == sheets[excluded]]
    if len(edge_faces) != 1 or edge_faces[0] != excluded:
        raise ValueError('The separating edge must be a genuine one-face sheet boundary.')
    third = faces[excluded][~np.isin(faces[excluded], edge)][0]
    normal = _unit(np.cross(points[edge[0]], points[edge[1]]))
    if np.dot(normal, points[third]) > 0:
        normal = -normal
    signed = float(normal @ points[vertex])
    if signed >= 0:
        raise ValueError('Identified boundary vertex does not penetrate the excluded half-space.')
    result = points.copy()
    result[vertex] = _unit(points[vertex] + (-signed + ROUND_OFF_PADDING_RADIANS) * normal)
    displacement = float(2 * mesh['radius_km'] * np.arcsin(
        np.linalg.norm(result[vertex] - points[vertex]) / 2))
    if displacement > max_displacement_km:
        raise ValueError('Required historical reconstruction exceeds the explicit displacement cap.')
    return result, dict(vertex_id=vertex, separating_edge=edge.tolist(), excluded_face=excluded,
        affected_face_indices=affected.tolist(), displacement_km=displacement,
        roundoff_padding_radians=ROUND_OFF_PADDING_RADIANS,
        roundoff_padding_metres=ROUND_OFF_PADDING_RADIANS * mesh['radius_km'] * 1000,
        max_displacement_km=float(max_displacement_km))


def _trace_faces(s):
    ids = np.asarray(s.parcel_patch)
    order = np.argsort(ids)
    found = np.searchsorted(ids[order], s.trace_patch)
    if np.any(found >= len(order)) or np.any(ids[order[found]] != s.trace_patch):
        raise ValueError('Every trace must retain an identified containing material face.')
    return order[found]


def _strain_columns(old, rows, ratio):
    result = deepcopy(old)
    volume = old['thickness_km'][rows] * old['area_factor'][rows]
    result['area_factor'][rows] = old['area_factor'][rows] * ratio
    result['thickness_km'][rows] = volume / result['area_factor'][rows]
    # This validates real column limits; it does not clip a bound violation.
    crustal_structure.elevation(result)
    return result


def normalize_area_metrics(s):
    """Return a detached, conservative migration of legacy area measurements.

    The represented vertices never move. Every face receives the current
    local-difference solid-angle measurement; a relative discrepancy above
    1e-8 rejects the transaction as more than the diagnosed legacy roundoff.
    This is a source-schema metric correction, not strain or elapsed geology.
    Physical and reference-column inventories remain conserved. No burial or
    overlap acceptance threshold is changed by this initialization operation.
    """
    if (getattr(s, 'retained_dense_crust_version', 0)
            or hasattr(s, 'channel_region_store')
            or getattr(s, 'continental_entry_regions', None) is not None):
        raise ValueError('Normalize historical metrics before thermal/regional initialization.')
    if hasattr(s, 'historical_area_metric_normalization'):
        raise ValueError('Historical area metrics are already normalized; do not repeat migration.')
    mesh = s.material_surface
    if mesh.get('columns') or not np.array_equal(s.parcel_patch, mesh['face_id']):
        raise ValueError('Historical metric normalization requires aligned face columns and identities.')
    old_area = np.asarray(mesh['area_km2'], float)
    if not np.isfinite(old_area).all() or np.any(old_area <= 0):
        raise ValueError('Historical area metrics must be positive and finite.')
    geometry = dict(mesh)
    new_area = material_surface.refresh_geometry(geometry)
    relative = np.abs(new_area-old_area)/old_area
    if np.any(relative > MAX_METRIC_RELATIVE_CORRECTION):
        raise ValueError('Historical area metric correction exceeds the explicit legacy-roundoff bound.')
    rows = np.flatnonzero(new_area != old_area)
    ratios = new_area/old_area
    face = _trace_faces(s)
    traces = np.flatnonzero(np.isin(face, rows))
    updated = _strain_columns(s.structure, rows, ratios[rows])
    updated_trace = _strain_columns(s.trace_structure, traces, ratios[face[traces]])
    state = deepcopy(s)
    state.material_surface['area_km2'] = new_area
    state.material_surface['geometry_revision'] += 1
    state.structure = updated
    state.trace_structure = updated_trace
    relief = crustal_structure.elevation(updated)-crustal_structure.elevation(s.structure)
    trace_relief = crustal_structure.elevation(updated_trace)-crustal_structure.elevation(s.trace_structure)
    state.relief[rows] += relief[rows]
    state.trace_relief_m[traces] += trace_relief[traces]
    state.trace_adjustment_m[traces] += trace_relief[traces]
    before_volume = old_area*s.structure['thickness_km']
    after_volume = new_area*state.structure['thickness_km']
    if not np.allclose(before_volume, after_volume, rtol=2e-12, atol=1e-9):
        raise ValueError('Metric normalization changed per-face physical crustal volume.')
    reference_error = 0.
    for old, new in ((s.structure, state.structure), (s.trace_structure, state.trace_structure)):
        before = old['thickness_km']*old['area_factor']
        after = new['thickness_km']*new['area_factor']
        if not np.allclose(before, after, rtol=2e-12, atol=1e-12):
            raise ValueError('Metric normalization changed a face or marker reference inventory.')
        reference_error = max(reference_error, float(np.max(np.abs(after-before)/before, initial=0.)))
    before_energy = reference_energy_state(s)*ENERGY_UNIT_J
    after_energy = reference_energy_state(state)*ENERGY_UNIT_J
    worst = int(np.argmax(relative)) if len(relative) else None
    report = dict(kind='explicit_historical_area_metric_normalization', version=1,
        epoch_myr=float(s.t), physical_evolution=False, steps_advanced=0,
        time_advanced_myr=0., vertices_moved=0, topology_changed=False,
        face_or_trace_identity_changed=False, reference_area_changed=False,
        force_coefficients_changed=False, physical_strain_counters_advanced=False,
        metric_kernel='material_surface.spherical_face_areas; translated solid-angle determinant',
        relative_correction_rejection_bound=MAX_METRIC_RELATIVE_CORRECTION,
        maximum_relative_correction=float(np.max(relative, initial=0.)),
        maximum_absolute_correction_km2=float(np.max(np.abs(new_area-old_area), initial=0.)),
        changed_face_count=len(rows), adjusted_trace_count=len(traces),
        worst_face=None if worst is None else dict(index=worst, face_id=int(s.parcel_patch[worst]),
            area_before_km2=float(old_area[worst]), area_after_km2=float(new_area[worst])),
        before_area=_digest(old_area), after_area=_digest(new_area),
        geometry_vertices=_digest(mesh['vertices']), geometry_faces=_digest(mesh['faces']),
        physical_volume_max_error_km3=float(np.max(np.abs(after_volume-before_volume), initial=0.)),
        physical_volume_max_relative_error=float(np.max(np.abs(after_volume-before_volume)/before_volume, initial=0.)),
        physical_volume_total_difference_km3=float(after_volume.sum()-before_volume.sum()),
        reference_inventory_max_relative_error=reference_error,
        maximum_thickness_adjustment_km=float(np.max(np.abs(updated['thickness_km']-s.structure['thickness_km']), initial=0.)),
        maximum_relief_adjustment_m=float(np.max(np.abs(relief), initial=0.)),
        measurement_energy_before_j=float(before_energy), measurement_energy_after_j=float(after_energy),
        measurement_energy_change_j=float(after_energy-before_energy),
        interpretation='Numerical measurement consistency at initialization, not physical deformation or work by plate forces.',
        preserved_history='Geometry, identities, mass references, source inventories, ages, geological strain/extension counters and RNG are unchanged.',
        derived_cache_refresh_required=True)
    state.historical_area_metric_normalization = deepcopy(report)
    return state, report


def prepare(s, *, vertex_id, separating_edge, excluded_face, max_displacement_km):
    """Return ``(detached_state, report)`` after a conservative local repair.

    Apply before thermal/entry initialization. The return retains all history,
    reference masses, face/trace identities and unchanged columns bit-for-bit.
    Derived exposure/control-grid caches must be refreshed by the caller's
    ordinary initialization preflight, after all source-policy migrations.
    """
    if getattr(s, 'retained_dense_crust_version', 0) or hasattr(s, 'channel_region_store'):
        raise ValueError('Repair this historical phase-free geometry before thermal/regional initialization.')
    if getattr(s, 'continental_entry_regions', None) is not None:
        raise ValueError('An initialized entry energy needs a separate reconstruction work ledger.')
    if hasattr(s, 'historical_geometry_repair'):
        raise ValueError('Historical geometry repair is already recorded; do not repeat it.')
    mesh = s.material_surface
    if mesh.get('columns'):
        raise ValueError('A surface with independent embedded columns needs an explicit aligned repair policy.')
    if not np.array_equal(s.parcel_patch, mesh['face_id']):
        raise ValueError('Material and face identities must align before historical repair.')
    before = _overlap(s, mesh['vertices'])
    if not len(before['first']):
        raise ValueError('No inherited same-sheet penetration requires repair.')
    candidate, geometric = boundary_vertex_candidate(s, vertex_id=vertex_id,
        separating_edge=separating_edge, excluded_face=excluded_face,
        max_displacement_km=max_displacement_km)
    rows = np.asarray(geometric['affected_face_indices'], dtype=int)
    if not np.all(np.isin(before['first'], rows) | np.isin(before['second'], rows)):
        raise ValueError('The selected local support does not cover every inherited penetration.')
    # Validate complete geometry without changing stored areas elsewhere. Older
    # checkpoints used an algebraically equivalent area determinant; recomputing
    # unrelated stored areas would manufacture unrelated column roundoff strain.
    geometry_check = dict(mesh, vertices=candidate.copy())
    recomputed = material_surface.refresh_geometry(geometry_check)
    after = _overlap(s, candidate)
    if len(after['first']):
        raise ValueError('Reconstructed geometry still has positive same-sheet intersections.')
    ratio = recomputed[rows] / np.asarray(mesh['area_km2'])[rows]
    updated = _strain_columns(s.structure, rows, ratio)
    face = _trace_faces(s)
    traces = np.flatnonzero(np.isin(face, rows))
    ratios_by_face = np.ones(len(mesh['faces']))
    ratios_by_face[rows] = ratio
    updated_trace = _strain_columns(s.trace_structure, traces, ratios_by_face[face[traces]])
    old_triangles = np.asarray(mesh['vertices'])[mesh['faces']]
    new_triangles = candidate[mesh['faces']]
    weights = np.linalg.solve(np.swapaxes(old_triangles[face[traces]], 1, 2),
                              s.trace_xyz[traces, :, None])[..., 0]
    weights /= weights.sum(axis=1)[:, None]
    if not np.isfinite(weights).all() or np.any(weights < -1e-7):
        raise ValueError('Historical marker lies outside its recorded face before reconstruction.')
    state = deepcopy(s)
    state.material_surface['vertices'] = candidate
    state.material_surface['area_km2'][rows] = recomputed[rows]
    state.material_surface['geometry_revision'] += 1
    state.structure = updated
    state.trace_structure = updated_trace
    state.pos[rows] = material_surface.face_centres(state.material_surface)[rows]
    state.rift_tangent[rows] = _transport_tangent(old_triangles[rows], new_triangles[rows],
                                               s.rift_tangent[rows], state.pos[rows])
    if hasattr(s, 'parcel_east'):
        state.parcel_east[rows] = _transport_tangent(old_triangles[rows], new_triangles[rows],
                                                   s.parcel_east[rows], state.pos[rows])
    state.trace_xyz[traces] = _unit(np.einsum('ni,nij->nj', weights, new_triangles[face[traces]]))
    state.trace_rift_tangent[traces] = _transport_tangent(
        old_triangles[face[traces]], new_triangles[face[traces]],
        s.trace_rift_tangent[traces], state.trace_xyz[traces])
    relief = crustal_structure.elevation(updated) - crustal_structure.elevation(s.structure)
    trace_relief = (crustal_structure.elevation(updated_trace)
                    - crustal_structure.elevation(s.trace_structure))
    state.relief[rows] += relief[rows]
    state.trace_relief_m[traces] += trace_relief[traces]
    state.trace_adjustment_m[traces] += trace_relief[traces]
    previous_volume = np.asarray(mesh['area_km2']) * s.structure['thickness_km']
    new_volume = state.material_surface['area_km2'] * state.structure['thickness_km']
    if not np.allclose(previous_volume, new_volume, rtol=2e-12, atol=1e-9):
        raise ValueError('Historical geometry repair failed per-face physical volume conservation.')
    for old, new in ((s.structure, state.structure), (s.trace_structure, state.trace_structure)):
        if not np.allclose(old['thickness_km'] * old['area_factor'],
                           new['thickness_km'] * new['area_factor'], rtol=2e-12, atol=1e-12):
            raise ValueError('Historical geometry repair changed a material or marker inventory.')
    before_energy = reference_energy_state(s) * ENERGY_UNIT_J
    after_energy = reference_energy_state(state) * ENERGY_UNIT_J
    check_weights = np.linalg.solve(np.swapaxes(new_triangles[face[traces]], 1, 2),
                                    state.trace_xyz[traces, :, None])[..., 0]
    check_weights /= check_weights.sum(axis=1)[:, None]
    if not np.allclose(weights, check_weights, rtol=0., atol=1e-11):
        raise ValueError('Historical repair lost trace material-coordinate attachment.')
    report = dict(kind='explicit_historical_boundary_geometry_reconstruction', version=1,
        epoch_myr=float(s.t), steps_advanced=0, time_advanced_myr=0.,
        physical_evolution=False, force_coefficients_changed=False,
        geometric_projection=geometric,
        input_geometry={key: _digest(value) for key, value in (
            ('vertices', mesh['vertices']), ('faces', mesh['faces']),
            ('collision_sheets', s.parcel_collision_sheet))},
        output_vertices=_digest(candidate),
        inherited_overlap_pairs=len(before['first']),
        inherited_pairwise_overlap_sum_km2=float(np.sum(before['area_km2'])),
        final_overlap_pairs=0, final_overlap_area_km2=0.,
        topology_changed=False, material_removed=False, identity_or_owner_changed=False,
        reference_area_changed=False, affected_trace_count=len(traces),
        trace_barycentric_max_error=float(np.max(np.abs(check_weights - weights), initial=0.)),
        physical_volume_max_error_km3=float(np.max(np.abs(new_volume-previous_volume))),
        physical_volume_total_difference_km3=float(new_volume.sum()-previous_volume.sum()),
        reconstruction_energy_before_j=float(before_energy), reconstruction_energy_after_j=float(after_energy),
        reconstruction_energy_change_j=float(after_energy-before_energy),
        energy_interpretation='Explicit initial-condition reconstruction work, not supplied by plate forces or a physical timestep.',
        face_changes=[dict(index=int(i), face_id=int(s.parcel_patch[i]),
            area_before_km2=float(mesh['area_km2'][i]), area_after_km2=float(recomputed[i]),
            area_ratio=float(ratios_by_face[i]), thickness_before_km=float(s.structure['thickness_km'][i]),
            thickness_after_km=float(state.structure['thickness_km'][i]),
            relief_adjustment_m=float(relief[i])) for i in rows],
        derived_cache_refresh_required=True,
        preserved_history='No age, geological source counter, RNG state, slab history, material ID, owner, sheet or topology is reset; trace relief changes are explicit adjustments.',
        limitation='One identified local boundary reconstruction; no claim to recover the unique past geometry. Preserved history predates the nonpenetration guard.')
    state.historical_geometry_repair = deepcopy(report)
    return state, report
