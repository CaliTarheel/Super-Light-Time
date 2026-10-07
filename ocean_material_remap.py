"""Experimental conservative endpoint remap; no native evolution wiring.

Each donor is a uniformly occupied minor spherical triangle with a supplied
source and accepted destination triangle. Their normalized homogeneous
barycentric correspondence is the finite map. Exact great-circle clipping
measures destination area; its projective pullback measures source material.
Those measures differ under extension and are never silently normalized.

This is a bounded reference kernel, not a trajectory/force acceptance solver.
It rejects endpoint inversion and positive same-owner endpoint intersections.
It cannot certify the path between endpoints, resolve inter-owner contact,
reconstruct subcell phase geometry, or create ridge, slab, damage, or energy
sources. Curved physical images require caller-supplied refined donor faces;
the geodesic triangle map is an explicit spatial approximation.

The endpoint audit is quadratic and intentionally resource-bounded. Production
native meshes at levels 4/5 exceed its default budget; this reference kernel
needs a separately validated sparse candidate implementation before such use.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

import mesh_coverage
import mesh_geometry

VERSION = 1
MODEL = 'geodesic donor triangles with projective source pullback'
MAX_PAIR_EVALUATIONS = 1_000_000  # Resource guard, not a physical limit.
CLOSURE_RTOL = 2e-10  # Numerical clipping/inverse-map acceptance, never rescaling.


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(name+' must be an integer at least '+str(minimum)+'.')
    return int(value)


def _field(value, count, name, *, integer=False):
    raw = np.asarray(value)
    kinds = 'iu' if integer else 'iuf'
    if raw.shape != (count,) or raw.dtype.kind not in kinds:
        raise ValueError(name+' must be exactly donor-aligned real '+('integers.' if integer else 'numbers.'))
    result = raw.astype(np.int64 if integer else np.float64)
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError(name+' must be finite and nonnegative.')
    return result


def _triangles(value, name):
    array = np.asarray(value, float)
    if array.ndim != 3 or array.shape[1:] != (3, 3) or not len(array) or not np.isfinite(array).all():
        raise ValueError(name+' must contain nonempty finite Nx3x3 unit triangles.')
    if not np.allclose(np.linalg.norm(array, axis=2), 1., rtol=0., atol=2e-13):
        raise ValueError(name+' requires unit-sphere coordinates, not radius-scaled positions.')
    # Preserve shared endpoint bits: normalizing independently can perturb a
    # represented shared edge. The accepted numerical input is already unit.
    determinant = np.einsum('ni,ni->n', array[:, 0], np.cross(array[:, 1]-array[:, 0], array[:, 2]-array[:, 0]))
    if np.any(determinant <= 0):
        raise ValueError(name+' has a degenerate or inverted endpoint triangle.')
    center = array.sum(axis=1)
    center /= np.linalg.norm(center, axis=1)[:, None]
    if np.any(np.einsum('nvi,ni->nv', array, center) <= 1e-10):
        raise ValueError(name+' triangles must fit strictly inside an open hemisphere.')
    condition = np.linalg.cond(array)
    if not np.isfinite(condition).all() or np.any(condition > 1e10):
        raise ValueError(name+' has an unresolved projective inverse; refine or reject the motion.')
    return array.copy()


def _areas(triangles, radius_m):
    return np.array([mesh_coverage._polygon_area(triangle, radius_m) for triangle in triangles])


def _caps(triangles):
    center = triangles.sum(axis=1)
    center /= np.linalg.norm(center, axis=1)[:, None]
    angle = np.max(np.arccos(np.clip(np.einsum('nvi,ni->nv', triangles, center), -1., 1.)), axis=1)
    return center, angle


def _possible(first, second, first_caps, second_caps):
    a, aa = first_caps; b, ba = second_caps
    return np.einsum('ni,ni->n', a[first], b[second]) >= np.cos(np.minimum(np.pi, aa[first]+ba[second]))-2e-14


def _reject_overlap(triangles, owners, radius_m, name):
    first, second = np.triu_indices(len(triangles), 1)
    keep = owners[first] == owners[second]
    first, second = first[keep], second[keep]
    caps = _caps(triangles)
    keep = _possible(first, second, caps, caps)
    first, second = first[keep], second[keep]
    for start in range(0, len(first), 1024):
        a, b = first[start:start+1024], second[start:start+1024]
        area = mesh_coverage._intersection_areas(triangles[a], triangles[b], radius_m)
        if np.any(area > 0.):
            hit = int(np.flatnonzero(area > 0.)[0])
            raise ValueError(f'{name} has positive same-owner overlap: donors {a[hit]} and {b[hit]}.')


def _fingerprint(record):
    payload = {key: value for key, value in record.items() if key != 'fingerprint'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def build_plan(receiving_mesh, source_triangles_xyz, mapped_triangles_xyz, owner_uids,
               donor_ids, occupied_fraction, *, max_pair_evaluations=MAX_PAIR_EVALUATIONS):
    """Build a plain-JSON remap record on a closed native receiving sphere.

    Fractions describe spatially uniform subcell owner occupancy, not resolved
    subcell polygons. Different owners may coexist and are NEVER normalized or
    removed. Same-owner donor interiors must be disjoint at both endpoints.
    Source extensive inventories are uniformly distributed over each donor's
    occupied source area. Empty occupancy may carry no inventory in ``apply``.
    """
    source = _triangles(source_triangles_xyz, 'Source geometry')
    mapped = _triangles(mapped_triangles_xyz, 'Mapped geometry')
    if source.shape != mapped.shape:
        raise ValueError('Source and mapped donors must retain identical alignment.')
    count = len(source)
    owners = _field(owner_uids, count, 'owner_uids', integer=True)
    ids = _field(donor_ids, count, 'donor_ids', integer=True)
    fraction = _field(occupied_fraction, count, 'occupied_fraction')
    if np.any(owners == 0) or np.any(fraction > 1):
        raise ValueError('Owner UIDs must be positive and occupied fractions at most one.')
    if len(np.unique(np.column_stack((owners, ids)), axis=0)) != count:
        raise ValueError('Each owner UID / donor ID pair must be unique.')
    radius_km = receiving_mesh.get('radius_km')
    if (not isinstance(radius_km, (int, float, np.integer, np.floating))
            or isinstance(radius_km, (bool, np.bool_)) or not math.isfinite(radius_km) or radius_km <= 0):
        raise ValueError('Receiving native mesh must declare a positive finite radius_km.')
    radius_m = float(radius_km)*1000.
    raw_faces = np.asarray(receiving_mesh['faces'])
    if raw_faces.ndim != 2 or raw_faces.shape[1:] != (3,):
        raise ValueError('Receiving faces must be indexed triangles.')
    ncontrol = len(raw_faces)
    limit = _integer(max_pair_evaluations, 'max_pair_evaluations', 1)
    pairs = ncontrol*(ncontrol-1)//2 + count*(count-1) + count*ncontrol
    if pairs > limit:
        raise ValueError(f'Remap needs {pairs} possible pairs, above resource limit {limit}; no partial result.')
    # Rebuild rather than trusting stale saved areas or edge tables. Check the
    # resource bound before allocating connectivity or any quadratic buffers.
    receiver = mesh_geometry.geometry(receiving_mesh['vertices'], raw_faces, radius_km)
    controls = _triangles(receiver['vertices'][receiver['faces']], 'Receiving geometry')
    source_area, mapped_area = _areas(source, radius_m), _areas(mapped, radius_m)
    control_area = _areas(controls, radius_m)
    if not np.isfinite(np.r_[source_area, mapped_area, control_area]).all():
        raise ValueError('Spherical area overflows the declared radius.')
    if not math.isclose(math.fsum(control_area), 4.*math.pi*radius_m**2, rel_tol=CLOSURE_RTOL):
        raise ValueError('Receiving mesh must cover exactly one complete sphere.')
    _reject_overlap(controls, np.zeros(ncontrol, int), radius_m, 'Receiving geometry')
    _reject_overlap(source, owners, radius_m, 'Source geometry')
    _reject_overlap(mapped, owners, radius_m, 'Mapped geometry')
    control_caps, mapped_caps = _caps(controls), _caps(mapped)
    donor_index, receiving_index, overlap_area, source_measure = [], [], [], []
    inverse = np.linalg.inv(mapped)
    for donor in range(count):
        cells = np.arange(ncontrol)
        keep = _possible(np.full(ncontrol, donor), cells, mapped_caps, control_caps)
        for cell in cells[keep]:
            polygon = mesh_coverage.clip_triangle(mapped[donor], controls[cell])
            area = mesh_coverage._polygon_area(polygon, radius_m)
            if area <= 0.: continue
            # Great circles remain great circles under a projective map, so
            # this polygon is the actual source preimage of the intersection.
            back = polygon@inverse[donor]@source[donor]
            length = np.linalg.norm(back, axis=1)
            if np.any(length <= 0) or not np.isfinite(back).all():
                raise ValueError('A remap polygon has no finite projective source preimage.')
            back /= length[:, None]
            material_area = mesh_coverage._polygon_area(back, radius_m)
            donor_index.append(donor); receiving_index.append(int(cell))
            overlap_area.append(area); source_measure.append(material_area)
    donor_index = np.asarray(donor_index, int)
    overlap_area, source_measure = np.asarray(overlap_area), np.asarray(source_measure)
    source_fraction = source_measure/source_area[donor_index]
    fraction_sum = np.bincount(donor_index, weights=source_fraction, minlength=count)
    mapped_sum = np.bincount(donor_index, weights=overlap_area, minlength=count)
    if (not np.isfinite(source_fraction).all() or np.any(source_fraction < 0)
            or np.any(np.abs(fraction_sum-1.) > CLOSURE_RTOL)
            or np.any(np.abs(mapped_sum-mapped_area) > CLOSURE_RTOL*mapped_area)):
        raise ValueError('Receiving intersections do not close source material and mapped area; no renormalization permitted.')
    record = dict(version=VERSION, model=MODEL, radius_m=radius_m,
        owner_uids=owners.tolist(), donor_ids=ids.tolist(), occupied_fraction=fraction.tolist(),
        source_triangles_xyz=source.tolist(), mapped_triangles_xyz=mapped.tolist(),
        source_area_m2=source_area.tolist(), mapped_area_m2=mapped_area.tolist(),
        receiving_area_m2=control_area.tolist(),
        receiving_geometry_sha256=hashlib.sha256(controls.tobytes()).hexdigest(),
        donor_index=donor_index.tolist(), receiving_index=receiving_index,
        destination_overlap_area_m2=overlap_area.tolist(), donor_material_fraction=source_fraction.tolist(),
        source_fraction_residual=(fraction_sum-1.).tolist(),
        mapped_area_residual_m2=(mapped_sum-mapped_area).tolist(),
        diagnostics=dict(experimental=True, native_transport_coupled=False,
            trajectory_between_endpoints_validated=False, endpoint_same_owner_overlap_pairs=0,
            fractional_subcell_geometry_resolved=False, cross_owner_contact_resolved=False,
            area_or_material_renormalized=False, possible_pairs=pairs,
            closure_relative_tolerance=CLOSURE_RTOL,
            interpolation='normalized homogeneous barycentric source-to-mapped triangle correspondence'))
    record['fingerprint'] = _fingerprint(record)
    return record


def validate_plan(plan):
    """Reject unsupported or changed records, including serialized mutations."""
    if type(plan.get('version')) is not int or plan['version'] != VERSION or plan.get('model') != MODEL:
        raise ValueError('Unsupported ocean material remap record.')
    if plan.get('fingerprint') != _fingerprint(plan):
        raise ValueError('Ocean remap record changed after geometric validation.')
    return plan


def apply(plan, volume_m3, *, extensive=None, units=None):
    """Conserve supplied donor volume and inventories; return sparse owner cells.

    ``volume_m3`` and all extensive arrays are TOTAL occupied donor inventories,
    already including fractional occupancy. No source is inferred from area or
    thickness. Each optional field must have an explicit unit string and be a
    finite nonnegative inventory (e.g. fracture capacity/spent energy in J).
    """
    validate_plan(plan)
    owners = np.asarray(plan['owner_uids'], np.int64); count = len(owners)
    occupancy = np.asarray(plan['occupied_fraction'])
    volume = _field(volume_m3, count, 'volume_m3')
    extra, unit_map = {} if extensive is None else extensive, {} if units is None else units
    if not isinstance(extra, dict) or not isinstance(unit_map, dict) or set(extra) != set(unit_map):
        raise ValueError('Every extensive inventory requires an explicit matching unit.')
    if any(not isinstance(name, str) or not name or name == 'volume_m3'
           or not isinstance(unit_map[name], str) or not unit_map[name] for name in extra):
        raise ValueError('Extensive inventory names/units must be nonempty and volume_m3 is reserved.')
    payload = {'volume_m3': volume, **{name: _field(value, count, name) for name, value in extra.items()}}
    if any(np.any(values[occupancy == 0.] != 0.) for values in payload.values()):
        raise ValueError('An unoccupied donor cannot carry material or energy inventory.')
    donor = np.asarray(plan['donor_index'], int); cell = np.asarray(plan['receiving_index'], int)
    pairs, inverse = np.unique(np.column_stack((owners[donor], cell)), axis=0, return_inverse=True)
    material_fraction = np.asarray(plan['donor_material_fraction'])
    occupied_area = np.bincount(inverse, weights=np.asarray(plan['destination_overlap_area_m2'])*occupancy[donor],
                                minlength=len(pairs))
    output = {name: np.bincount(inverse, weights=values[donor]*material_fraction, minlength=len(pairs))
              for name, values in payload.items()}
    source_owner = np.unique(owners)
    residuals = {}
    for name, values in payload.items():
        old = np.array([math.fsum(values[owners == owner]) for owner in source_owner])
        new = np.array([math.fsum(output[name][pairs[:, 0] == owner]) for owner in source_owner])
        residual = new-old
        if not np.isfinite(new).all() or np.any(np.abs(residual) > CLOSURE_RTOL*old):
            raise ValueError('A remapped inventory failed per-owner conservation; no partial result.')
        residuals[name] = residual.tolist()
    cell_area = np.asarray(plan['receiving_area_m2'])
    total_area = np.bincount(pairs[:, 1], weights=occupied_area, minlength=len(cell_area))
    thickness = np.divide(output['volume_m3'], occupied_area, out=np.zeros(len(pairs)), where=occupied_area > 0)
    result = dict(version=VERSION, plan_fingerprint=plan['fingerprint'], owner_uids=pairs[:, 0].tolist(),
        receiving_index=pairs[:, 1].tolist(), occupied_area_m2=occupied_area.tolist(),
        volume_m3=output.pop('volume_m3').tolist(), thickness_m=thickness.tolist(),
        extensive={name: values.tolist() for name, values in output.items()}, units=dict(unit_map),
        source_inventory=dict(owner_uids=plan['owner_uids'].copy(), donor_ids=plan['donor_ids'].copy(),
            volume_m3=volume.tolist(), extensive={name: payload[name].tolist() for name in extra}),
        cell_total_occupied_fraction=(total_area/cell_area).tolist(),
        diagnostics=dict(owner_uids=source_owner.tolist(), conservation_residuals=residuals,
            unresolved_cell_excess_area_m2=float(np.maximum(total_area-cell_area, 0.).sum()),
            pointwise_cross_owner_capacity_checked=False, occupancy_normalized=False,
            sources_created=False, fracture_capacity_reinitialized=False,
            source_occupied_area_m2=float(np.asarray(plan['source_area_m2'])@occupancy),
            mapped_occupied_area_m2=float(np.asarray(plan['mapped_area_m2'])@occupancy)))
    result['fingerprint'] = _fingerprint(result)
    return result
