"""Display-only sampling of saved native mesh fields onto a finer output grid.

The engine integrates on 81,920 spherical faces but each frame also stores a 192x96
equirectangular raster for display, and `mesh_diagnostics.display_grid_is_output_only`
says plainly that the grid is an output. Measured on frame 214 (496 Myr), that grid shows
16,212 distinct faces -- 19.8% of the mesh. Four fifths of the computed cells never reach
the picture. At 768x384 every one of the 81,920 faces appears.

This module changes nothing about the physics: it reads fields the run already wrote and
resamples them. It is imported by server.py only, never by the engine, so it stays outside
`capture_auxiliary_sources` and no checkpoint is affected by editing it.

The mesh geometry is FIXED for a run (vertices, faces and areas are byte-identical between
272 Myr and 496 Myr), so the pixel -> face map depends only on the grid size and the view
orientation. It is built once (~3 s) and cached, after which each frame is one gather.
"""
from __future__ import annotations

import hashlib
import numpy as np

# Every face is represented at 768x384; beyond that only the picture gets bigger.
DEFAULT_WIDTH = 768
MAX_WIDTH = 2048
# Fields stored per mesh face, and the raster name each corresponds to.
MESH_FIELDS = {'elevation': 'mesh_elevation_m', 'plate': 'mesh_plate', 'crust': 'mesh_crust',
               'age': 'mesh_age_myr', 'boundary': 'mesh_boundary'}
_CACHE: dict[tuple, np.ndarray] = {}
_CACHE_LIMIT = 8


def available(archive) -> bool:
    """True when this frame carries the native mesh, not just the display raster."""
    return all(name in archive.files for name in ('mesh_vertices', 'mesh_faces')) \
        and any(name in archive.files for name in MESH_FIELDS.values())


def _directions(width, height):
    """Unit vectors for the centre of every output pixel, north-to-south, row-major."""
    lon = np.radians((np.arange(width) + .5) / width * 360. - 180.)
    lat = np.radians(90. - (np.arange(height) + .5) / height * 180.)
    cos_lat = np.cos(lat)[:, None]
    return np.stack([(cos_lat * np.cos(lon)[None, :]).ravel(),
                     (cos_lat * np.sin(lon)[None, :]).ravel(),
                     np.repeat(np.sin(lat), width)], axis=1)


def index_map(vertices, faces, width, height, rotation=None):
    """Face index for each output pixel: the face whose centroid is nearest its direction.

    Faces are near-equilateral, so nearest-centroid is their spherical Voronoi cell to
    within a fraction of a face. Candidates are restricted by latitude band, which keeps
    the search from being 81,920 x 294,912.
    """
    key = (hashlib.sha256(np.ascontiguousarray(vertices).tobytes()).hexdigest()[:16], len(faces),
           int(width), int(height), None if rotation is None else np.asarray(rotation, float).tobytes())
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    centroid = vertices[faces].mean(axis=1)
    centroid /= np.linalg.norm(centroid, axis=1)[:, None]
    points = _directions(width, height)
    if rotation is not None:
        # Rotate the sample directions, never the data, so the cache key covers the view.
        # TRANSPOSED to match orientation.reproject_raster, which inverse-warps destination
        # cells into the source. Verified against the engine's own rotated raster at yaw 100:
        # this convention agrees on 97.8% of land/sea (corr .973); the untransposed matrix
        # gives 76.3% (corr .110), i.e. a scrambled map.
        points = points @ np.asarray(rotation, float).T
    bands = max(8, min(180, height // 2))
    face_band = np.clip(((1. - centroid[:, 2]) * .5 * bands).astype(np.int64), 0, bands - 1)
    order = np.argsort(face_band, kind='stable')
    sorted_band = face_band[order]
    start = np.searchsorted(sorted_band, np.arange(bands), 'left')
    stop = np.searchsorted(sorted_band, np.arange(bands), 'right')
    point_band = np.clip(((1. - points[:, 2]) * .5 * bands).astype(np.int64), 0, bands - 1)
    result = np.empty(len(points), np.int32)
    for band in range(bands):
        rows = np.flatnonzero(point_band == band)
        if not len(rows):
            continue
        near = [order[start[b]:stop[b]] for b in (band - 1, band, band + 1) if 0 <= b < bands]
        candidates = np.concatenate(near)
        # Chunked so the dot-product block stays modest regardless of grid size.
        for piece in np.array_split(rows, max(1, len(rows) * len(candidates) // 4_000_000 + 1)):
            result[piece] = candidates[np.argmax(points[piece] @ centroid[candidates].T, axis=1)]
    if len(_CACHE) >= _CACHE_LIMIT:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = result
    return result


def sample(archive, names, width=DEFAULT_WIDTH, height=None, rotation=None):
    """Resample the requested display fields onto a width x height grid.

    Returns {name: 1-D array} plus the grid size. Names without a mesh-resolution field are
    skipped: the caller keeps the stored raster for those layers rather than inventing detail.
    """
    width = int(width)
    if not 2 <= width <= MAX_WIDTH or width % 2:
        raise ValueError(f'Display width must be even and between 2 and {MAX_WIDTH}.')
    height = int(height or width // 2)
    if height * 2 != width:
        raise ValueError('Display grid must be 2:1.')
    if not available(archive):
        raise ValueError('This frame has no saved native mesh to resample.')
    lookup = index_map(archive['mesh_vertices'], archive['mesh_faces'], width, height, rotation)
    out = {'width': width, 'height': height, 'fields': {}}
    for name in names:
        source = MESH_FIELDS.get(name)
        if source is None or source not in archive.files:
            continue
        values = archive[source]
        out['fields'][name] = values[lookup]
    return out
