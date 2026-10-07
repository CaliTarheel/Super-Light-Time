"""Opt-in reconstruction of collision coasts on the exposed material union.

This is a continuous surface interpolation, not a change to mechanical columns,
compensation, volume or erosion. Ordinary sheets share one exposed coast; their
interior columns are blended in persistent vertical order through finite toes.
"""
from __future__ import annotations
import numpy as np
import continental_contacts
import continental_margin as margin
import native_spreading

VERSION = 2
REFERENCE_FIELD = 'material_coastal_reference_elevation_m'


def version(frame):
    value = frame.get('collision_coast_version', 0)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value not in (0, 1, VERSION):
        raise ValueError('Unsupported collision coast reconstruction version.')
    if value and frame.get('collision_surface_version') != 1:
        raise ValueError('Collision coasts require explicit physical stack support.')
    return value


def reference_field(frame, count):
    """Version two needs the column's recorded neutral elevation, never a guess."""
    revision = version(frame)
    if revision < 2:
        if REFERENCE_FIELD in frame:
            raise ValueError('Coastal reference elevations require collision coast version two.')
        return None
    value = np.asarray(frame.get(REFERENCE_FIELD), dtype=float)
    if value.shape != (count,) or not np.isfinite(value).all():
        raise ValueError('Collision coast version two requires finite aligned coastal reference elevations.')
    return value


def prepare(vertices, faces, original):
    """Remove only covered portions of free edges; keep exposed finite pieces.

    Use the existing minor-arc linear parameter and containment tolerance.
    Coincident same-facing coasts remain exterior; opposite material or strict
    interior containment hides a coast. No snapping or whole-edge midpoint mask.
    """
    outline = continental_contacts.free_outline(vertices, faces)
    context = native_spreading.prepare_material(dict(vertices=vertices, faces=faces))
    components = original['components']
    parent = np.arange(len(original['component_width_km']))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    starts, ends, owners = [], [], []
    covered_count = 0
    for a, b, face in zip(outline['a'], outline['b'], outline['face']):
        component = components[face]
        inward = margin._unit(np.cross(a, b))
        covered = []
        for other in native_spreading._candidates(np.array([a, b]), context):
            if components[other] == component: continue
            planes = context['planes'][other]
            before, after = planes@a, planes@b
            coplanar = (np.abs(before) <= 2e-14) & (np.abs(after) <= 2e-14)
            if np.any(coplanar & (planes@inward > 0.)): continue
            before[np.abs(before) <= 2e-14] = 0.
            after[np.abs(after) <= 2e-14] = 0.
            low, high = 0., 1.
            for first, last in zip(before, after):
                if first < 0. and last < 0.: high = -1.; break
                if first < 0. <= last: low = max(low, float(-first/(last-first)))
                elif last < 0. <= first: high = min(high, float(-first/(last-first)))
            if high-low > 1e-12:
                covered.append((low, high))
                parent[root(int(components[other]))] = root(int(component))
        cursor = 0.; pieces = []
        for low, high in sorted(covered):
            if low > cursor+1e-12: pieces.append((cursor, low))
            cursor = max(cursor, high)
        if cursor < 1.-1e-12: pieces.append((cursor, 1.))
        covered_count += bool(covered)
        for low, high in pieces:
            starts.append(margin._unit((1.-low)*a+low*b))
            ends.append(margin._unit((1.-high)*a+high*b))
            owners.append(component)
    roots = np.array([root(i) for i in range(len(parent))], int)
    widths = np.zeros(len(parent))
    np.maximum.at(widths, roots, original['component_width_km'])
    a, b = np.asarray(starts).reshape(-1, 3), np.asarray(ends).reshape(-1, 3)
    normal = margin._unit(np.cross(a, b))
    mid = margin._unit(a+b)
    half = .5*np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.sum(a*b, axis=1))
    resolution = original['resolution']; bins = {}; global_edges = []
    radius = 2.*np.sin(np.minimum(np.pi, half+original['profile']['width_km']/margin.RADIUS_KM)/2.)
    lo = np.clip(np.floor((mid-radius[:, None]+1)*resolution/2).astype(int), 0, resolution-1)
    hi = np.clip(np.floor((mid+radius[:, None]+1)*resolution/2).astype(int), 0, resolution-1)
    for edge, (lower, upper) in enumerate(zip(lo, hi)):
        if np.prod(upper-lower+1) > 2048: global_edges.append(edge); continue
        for x in range(lower[0], upper[0]+1):
            for y in range(lower[1], upper[1]+1):
                for z in range(lower[2], upper[2]+1):
                    bins.setdefault(int(x+resolution*(y+resolution*z)), []).append(edge)
    return dict(profile=original['profile'], a=a, b=b, normal=normal,
        components=roots[components], edge_component=roots[np.asarray(owners, int)],
        eligible=np.ones(len(faces), bool), component_width_km=widths,
        resolution=resolution, bins={k:np.asarray(v, np.int32) for k,v in bins.items()},
        global_edges=np.asarray(global_edges, np.int32), open_edge_count=len(a),
        covered_source_edges=covered_count)


def finish(context, points, material, mass, ordinary_mass, raw, support, thermal, legacy,
           reference=None):
    """Normalize ordered interior weights, then apply the union's coast once.

    Arc-only surfaces keep their existing volcanic profile. A continuous
    ordinary-sheet share blends that profile into the common collision surface
    in mixed arc/continent toes. This interpolation is not a pressure solve.
    """
    active = (material >= 0) & (mass > 0.) & (ordinary_mass > 0.)
    if not np.any(active): return
    q = np.flatnonzero(active); f = material[q]
    union = context['collision_coast']
    x = np.clip(margin.distance_km(union, points[q], f)/
                np.maximum(union['component_width_km'][union['components'][f]], 1e-12), 0., 1.)
    ocean = legacy['ocean_datum_height_m'][q]
    interior = raw[q]/mass[q]
    baseline = interior if version(context) < 2 else reference[q]/mass[q]
    height = margin.profile_height(baseline, ocean, x,
                                  shelf_depth_m=union['profile']['shelf_depth_m'])
    # Version two preserves the full signed displacement on the inherited
    # shelf and inland. Only the existing ocean-slope interval needs a taper
    # to the ocean endpoint; extending it across the shelf erases relief.
    displacement_fraction = np.clip(x/.35, 0., 1.) if version(context) >= 2 else x
    blend = displacement_fraction*displacement_fraction*(3.-2.*displacement_fraction)
    if version(context) >= 2:
        # Apply the inherited shelf shape to the neutral column only. Signed
        # changes in column elevation must not be recapped at shelf depth.
        # Use the same ocean-slope taper for column and stack displacement.
        height += blend*(interior-baseline)
    share = np.clip(ordinary_mass[q]/mass[q], 0., 1.)
    for key, value in [('selected_sheet_height_m', height),
                       ('physical_stack_support_m', blend*support[q]/mass[q]),
                       ('display_thermal_support_m', legacy['ocean_thermal'][q]+x*(thermal[q]/mass[q]-legacy['ocean_thermal'][q]))]:
        old = legacy[key][q]
        legacy[key][q] = old+share*(value-old)
