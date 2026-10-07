"""Opt-in finite-volume erosion driven by the shared native exposed surface.

Quadrature is over material triangles, never over the display grid. The
surface/ownership snapshot precedes erosion of every parcel and marker.
"""
from copy import deepcopy
from numbers import Integral
import numpy as np
import mesh_geometry as geometry
import native_frame_sampling as sampling
import native_boundary_geometry
import material_surface

VERSION = 1
FIELD = 'surface_erosion_version'
LEVEL = 3  # Sixty-four spherical child-centroid samples per material face.


def version(s):
    get = s.get if isinstance(s, dict) else lambda k, d: getattr(s, k, d)
    value = get(FIELD, 0)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value not in (0, VERSION):
        raise ValueError('Unsupported surface erosion version.')
    if value and (get('collision_surface_version', 0) != 1 or get('collision_coast_version', 0) != 2):
        raise ValueError('Surface erosion requires collision surface 1 and coastal deformation 2.')
    return int(value)


def snapshot(s):
    result = {FIELD: version(s)}
    for field in ('surface_erosion_migration', 'surface_erosion_diagnostics'):
        if hasattr(s, field): result[field] = deepcopy(getattr(s, field))
    return result


def upgrade(s):
    """Explicit policy selection only; never initialize or repair old columns."""
    version(s)
    version({FIELD: VERSION, 'collision_surface_version': getattr(s, 'collision_surface_version', 0),
             'collision_coast_version': getattr(s, 'collision_coast_version', 0)})
    if not np.isfinite(s.t) or s.t < 0: raise ValueError('Invalid surface erosion migration epoch.')
    if version(s): return deepcopy(getattr(s, 'surface_erosion_migration', {}))
    receipt = dict(version=VERSION, time_myr=float(s.t), from_version=0, to_version=VERSION,
                   physical_state_changed=False, initial_conditions_rerun=False)
    s.surface_erosion_version = VERSION
    s.surface_erosion_migration = receipt
    return deepcopy(receipt)


def quadrature(vertices, faces, level=LEVEL):
    """Rotation-equivariant positive weights from exact spherical child areas."""
    if isinstance(level, bool) or not isinstance(level, Integral) or not 0 <= level <= 5:
        raise ValueError('Invalid surface quadrature subdivision level.')
    triangles = np.asarray(vertices, float)[np.asarray(faces)]
    count = len(triangles)
    triangles = triangles[:, None]
    unit = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)
    for _ in range(level):
        a, b, c = triangles[..., 0, :], triangles[..., 1, :], triangles[..., 2, :]
        ab, bc, ca = unit(a+b), unit(b+c), unit(c+a)
        triangles = np.stack((np.stack((a, ab, ca), -2), np.stack((ab, b, bc), -2),
                              np.stack((ca, bc, c), -2), np.stack((ab, bc, ca), -2)), -3)
        triangles = triangles.reshape(count, 4**(_+1), 3, 3)
    a, b, c = triangles[..., 0, :], triangles[..., 1, :], triangles[..., 2, :]
    dot = lambda x,y: np.einsum('...i,...i->...', x,y)
    # Equivalent determinant using edge differences avoids cancellation of
    # large Cartesian products for small/skinny material triangles.
    area = 2.*np.arctan2(np.abs(dot(a,np.cross(b-a,c-a))), 1.+dot(a,b)+dot(b,c)+dot(c,a))
    if not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError('Surface erosion requires positive finite material triangles.')
    weights = area/area.sum(axis=1, keepdims=True)
    return unit(triangles.sum(axis=-2)), weights


def integrate(vertices, faces, sample, *, level=LEVEL, batch_faces=256):
    """Area-mean positive height and dry fraction of this exposed source face.

    The selector supplies exactly the same sea-level datum and provenance as
    the viewer/export. Buried or submerged samples contribute zero. Applying
    another exposure fraction downstream would count coverage twice.
    """
    n = len(faces)
    height, dry = np.zeros(n), np.zeros(n)
    for start in range(0, n, batch_faces):
        stop = min(n, start+batch_faces)
        points, weights = quadrature(vertices, faces[start:stop], level)
        result = sample(points.reshape(-1,3))
        h = np.asarray(result['raw_elevation_m'], float).reshape(weights.shape)
        selected = np.asarray(result['material_face']).reshape(weights.shape)
        if not np.isfinite(h).all(): raise ValueError('Nonfinite exposed erosion surface.')
        active = (selected == np.arange(start, stop)[:, None]) & (h > 0.)
        height[start:stop] = np.sum(weights*np.where(active, h, 0.), axis=1)
        dry[start:stop] = np.sum(weights*active, axis=1)
    return height, dry


def sampler(s):
    """Prepare a same-time native query operator; no raster or full snapshot."""
    if not version(s): raise ValueError('Surface erosion is not selected.')
    context = sampling.prepare_live_surface(s)
    owner_field = native_boundary_geometry.prepare_owner_sampling(
        s.native_mesh, s.plate, s.support, locator=s.native_locator)
    material_locator = material_surface.build_surface_locator(s.material_surface)
    def sample(points):
        cells, _ = geometry.locate_points(points, s.native_locator)
        owners = native_boundary_geometry.sample_owners(points, owner_field)
        hits = material_surface.sample_surface(s.material_surface, points, locator=material_locator)
        return sampling.select_exposed_material(context, points, cells, owners,
            (hits['query_index'], hits['face_index'], hits['weights']),
            episodes=s.ridge_episodes, time_myr=s.t)
    return sample


def prepare(s):
    """Freeze the live pre-erosion integral once for parcels and markers."""
    sample = sampler(s)
    height, dry = integrate(s.material_surface['vertices'], s.material_surface['faces'], sample)
    area = np.asarray(s.material_surface['area_km2'], float)
    s.surface_erosion_diagnostics = dict(version=VERSION, sample_epoch_myr=float(s.t),
        sampling_skipped=False, quadrature_samples_per_face=4**LEVEL,
        dry_area_km2=float(area@dry), positive_height_integral_m_km2=float(area@height),
        faces_with_dry_samples=int(np.count_nonzero(dry)),
        partial_dry_sample_faces=int(np.count_nonzero((dry>0)&(dry<1-1e-12))),
        faces_without_dry_samples=int(np.count_nonzero(dry==0)),
        maximum_mean_positive_height_m=float(height.max(initial=0.)),
        surface_time='after deformation and magma, before parcel erosion and phase evolution',
        markers='same-time source-face finite-volume forcing', quadrature_is_approximate=True)
    return height


def trace_values(s, heights):
    values = np.asarray(heights, float)
    ids = np.asarray(s.parcel_patch)
    if values.shape != ids.shape or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Invalid source-face erosion height snapshot.')
    order = np.argsort(ids)
    at = np.searchsorted(ids[order], s.trace_patch)
    valid = at < len(ids)
    valid[valid] &= ids[order[at[valid]]] == s.trace_patch[valid]
    if not valid.all(): raise ValueError('A surface erosion trace has no source face.')
    return values[order[at]].copy()
