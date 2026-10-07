"""Versioned continental shelves on the actual moving material footprint.

The footprint is crust, not an emergent-land mask. A finite submerged slope,
shelf and coastal transition occupy its inside edge. Exact spherical distance
to finite free edges drives the profile: no pixel blur, outward extrapolation,
or bridging of holes. Columns, volume, lineage and erosion ledgers are untouched.
This is an explicit surface reconstruction, not a new mechanical column solve.
"""
from __future__ import annotations

import numpy as np
import mesh_geometry
import continental_contacts

RADIUS_KM = 6371.
DEFAULTS = dict(width_km=150., shelf_depth_m=180.)


def parameters(value=None):
    value = dict(DEFAULTS if value is None else value)
    if set(value) != set(DEFAULTS):
        raise ValueError('Continental margins require width_km and shelf_depth_m.')
    for key, limits in (('width_km', (10., 500.)), ('shelf_depth_m', (20., 500.))):
        number = value[key]
        if isinstance(number, (bool, np.bool_)) or not np.isscalar(number):
            raise ValueError('Continental margin parameters must be finite numbers.')
        number = float(number)
        if not np.isfinite(number) or not limits[0] <= number <= limits[1]:
            raise ValueError(f'Continental margin {key} must lie within {limits}.')
        value[key] = number
    return value


def upgrade_frame(frame, *, width_km=150., shelf_depth_m=180.):
    """Explicit, pure opt-in for a derived review of a saved native epoch.

    The returned mapping shares unchanged native arrays with its source. Callers
    must identify the derived surface when saving it; this cannot recover coast
    detail that was never present in the original material geometry.
    """
    if int(frame.get('mesh_version', 0)) != 1 or frame.get('surface_reconstruction_version') != 1:
        raise ValueError('Continental margin reconstruction needs a continuous native material epoch.')
    if frame.get('arc_surface_version') != 2:
        raise ValueError('Continental margin reconstruction requires the version-two surface selector.')
    previous = validate_frame(frame)
    requested = parameters(dict(width_km=width_km, shelf_depth_m=shelf_depth_m))
    if previous == requested:
        return dict(frame)
    result = dict(frame)
    result.update(continental_margin_version=1,
                  continental_margin_parameters=requested,
                  continental_margin_revision=dict(surface_only=True,
                      source_version=int(frame.get('continental_margin_version', 0)),
                      physical_arrays_unchanged=True, recovered_initial_geometry=False))
    return result


def validate_frame(frame):
    version = frame.get('continental_margin_version', 0)
    if version not in (0, 1):
        raise ValueError('Unsupported continental margin version.')
    if version == 0:
        if 'continental_margin_parameters' in frame:
            raise ValueError('Continental margin parameters require their surface version.')
        return None
    if frame.get('surface_reconstruction_version') != 1 or frame.get('arc_surface_version') != 2:
        raise ValueError('Continental margins require continuous native surfaces and the version-two selector.')
    if 'continental_margin_parameters' not in frame:
        raise ValueError('Continental margin version one requires saved profile parameters.')
    return parameters(frame['continental_margin_parameters'])


def _unit(x):
    return x/np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-30)


def _distance_matrix(points, a, b, normal):
    """Exact distance to each finite minor great-circle segment, in radians."""
    dot = points@normal.T
    foot = _unit(points[:, None, :]-dot[:, :, None]*normal[None, :, :])
    along = ((np.einsum('pej,ej->pe', np.cross(a[None], foot), normal) >= -2e-13)
             & (np.einsum('pej,ej->pe', np.cross(foot, b[None]), normal) >= -2e-13))
    endpoint = np.arccos(np.clip(np.maximum(points@a.T, points@b.T), -1., 1.))
    return np.where(along, np.arcsin(np.clip(np.abs(dot), 0., 1.)), endpoint)


def prepare(vertices, faces, kinds, arc_ids, *, profile=None):
    """Index finite free coast segments and connected material components.

    Identical coordinates on duplicated owner fault sides are paired before
    finding free edges. Rigid rotations remain seam/pole independent. Unjoined
    islands use only their own outline; nearby sheets cannot lend a coastline.
    """
    profile = parameters(profile)
    vertices, faces = np.asarray(vertices, float), np.asarray(faces)
    kinds, arc_ids = np.asarray(kinds), np.asarray(arc_ids)
    if (faces.ndim != 2 or faces.shape[1:] != (3,) or kinds.shape != (len(faces),)
            or arc_ids.shape != kinds.shape or not np.isfinite(vertices).all()):
        raise ValueError('Continental margin geometry and categories must align.')
    eligible = np.isin(kinds, (1, 2)) & (arc_ids == 0)
    # Interval cancellation also handles a long fault side opposite several
    # independently refined edges. Actual uncovered intervals remain coasts.
    outline = continental_contacts.free_outline(vertices, faces)
    components, number = mesh_geometry.connected_components(np.zeros(len(faces), np.int8),
        outline['joins'])
    selected = eligible[outline['face']]
    a, b = outline['a'][selected], outline['b'][selected]
    normal = _unit(np.cross(a, b))
    mid = _unit(a+b)
    half = .5*np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.sum(a*b, axis=1))
    edge_component = components[outline['face'][selected]]
    resolution = int(np.clip(np.ceil(2.*RADIUS_KM/profile['width_km']), 8, 128))
    bins, global_edges = {}, []
    radius = 2.*np.sin(np.minimum(np.pi, half+profile['width_km']/RADIUS_KM)/2.)
    lo = np.clip(np.floor((mid-radius[:, None]+1)*resolution/2).astype(int), 0, resolution-1)
    hi = np.clip(np.floor((mid+radius[:, None]+1)*resolution/2).astype(int), 0, resolution-1)
    for edge, (lower, upper) in enumerate(zip(lo, hi)):
        if np.prod(upper-lower+1) > 2048:
            global_edges.append(edge)
            continue
        for x in range(lower[0], upper[0]+1):
            for y in range(lower[1], upper[1]+1):
                for z in range(lower[2], upper[2]+1):
                    bins.setdefault(int(x+resolution*(y+resolution*z)), []).append(edge)
    result = dict(profile=profile, a=a, b=b, normal=normal, components=components,
        edge_component=edge_component, eligible=eligible, bins={key:np.asarray(value, np.int32) for key,value in bins.items()},
        global_edges=np.asarray(global_edges, np.int32), resolution=resolution,
        component_width_km=np.full(number, profile['width_km']), open_edge_count=len(a),
        contact_diagnostics=outline['diagnostics'])
    # Hydraulic radius (area / free-coast perimeter) caps small-fragment
    # margins and is invariant under conforming subdivision of unchanged
    # triangles. Face-centre estimates of an inradius would jump at remeshing.
    # This changes the surface belt, never the footprint or craton geometry.
    if len(faces):
        area = np.bincount(components, weights=mesh_geometry.spherical_area(vertices, faces), minlength=number)
        perimeter = np.bincount(edge_component, weights=2.*half*RADIUS_KM, minlength=number)
        scale = np.divide(area, perimeter, out=np.full(number, np.inf), where=perimeter > 0.)
        result['component_width_km'] = np.minimum(profile['width_km'], .55*scale)
    return result


def distance_km(prepared, points, face_indices):
    """Bounded sparse search; points must already have actual containing faces."""
    points, faces = np.asarray(points, float), np.asarray(face_indices)
    if points.shape != (len(faces), 3) or np.any(faces < 0) or np.any(faces >= len(prepared['components'])):
        raise ValueError('Margin distances require actual material hits.')
    result = np.full(len(points), np.inf)
    resolution = prepared['resolution']
    bins = np.clip(np.floor((points+1)*resolution/2).astype(np.int64), 0, resolution-1)
    keys = bins[:, 0]+resolution*(bins[:, 1]+resolution*bins[:, 2])
    order = np.argsort(keys, kind='stable')
    if not len(order):
        return result
    starts = np.r_[0, np.flatnonzero(keys[order][1:] != keys[order][:-1])+1]
    ends = np.r_[starts[1:], len(order)]
    components = prepared['components'][faces]
    for start, end in zip(starts, ends):
        indices = order[start:end]
        candidate = prepared['bins'].get(int(keys[indices[0]]), np.empty(0, np.int32))
        candidate = np.r_[candidate, prepared['global_edges']]
        if not len(candidate):
            continue
        for offset in range(0, len(indices), 512):
            rows = indices[offset:offset+512]
            for edge_start in range(0, len(candidate), 256):
                edges = candidate[edge_start:edge_start+256]
                value = _distance_matrix(points[rows], prepared['a'][edges], prepared['b'][edges], prepared['normal'][edges])*RADIUS_KM
                value[components[rows, None] != prepared['edge_component'][edges][None]] = np.inf
                result[rows] = np.minimum(result[rows], value.min(axis=1))
    result[~prepared['eligible'][faces]] = np.inf
    return result


def profile_height(interior_m, ocean_m, fraction, *, shelf_depth_m=180.):
    """Slope 0–0.35, shelf 0.35–0.75, coastal transition 0.75–1.

    Smoothstep joins have zero normal derivative. The top of a submerged
    continental basin never becomes an invented emergent shelf. At distance
    zero the actual neighboring ocean function is matched exactly.
    """
    height, ocean, x = np.broadcast_arrays(np.asarray(interior_m, float), np.asarray(ocean_m, float),
                                           np.asarray(fraction, float))
    x = np.clip(x, 0., 1.)
    smooth = lambda value: value*value*(3.-2.*value)
    shelf = np.minimum(height, -float(shelf_depth_m))
    beach = np.minimum(height, -20.)
    slope = ocean+(shelf-ocean)*smooth(np.clip(x/.35, 0., 1.))
    shelf_surface = shelf+(beach-shelf)*smooth(np.clip((x-.35)/.4, 0., 1.))
    coast = beach+(height-beach)*smooth(np.clip((x-.75)/.25, 0., 1.))
    return np.where(x >= .75, coast, np.where(x >= .35, shelf_surface, slope))


def apply(prepared, points, face_indices, interior_m, ocean_m):
    distance = distance_km(prepared, points, face_indices)
    width = prepared['component_width_km'][prepared['components'][np.asarray(face_indices)]]
    fraction = np.clip(distance/np.maximum(width, 1e-12), 0., 1.)
    return profile_height(interior_m, ocean_m, fraction,
                          shelf_depth_m=prepared['profile']['shelf_depth_m']), fraction
