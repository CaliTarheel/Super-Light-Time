"""Continuous finite boundary belts sampled at spherical material columns.

The geographic bins below only accelerate candidate lookup. Physical distances
and amplitudes do not depend on a display raster or its pixel size. The compact
cosine-squared belt has unit peak and a cross-belt integral equal to its 400 km
cutoff, matching the reduced column model's existing strain length. Overlapping
segments combine by maximum, so subdivision never multiplies the imposed load.
"""
from __future__ import annotations

import numpy as np

from ridge_geometry import candidate_cells
import trench_history
import normal_partition

RADIUS_KM = 6371.
BELT_WIDTH_KM = 400.


def boundary_belts(s):
    """Existing boundary rates, with explicit affected material owners.

    Columns are shortening km/Myr, extension km/Myr, volcanism m/Myr. Collision
    and continental-rift belts load their two participants. An arc belt is
    displaced 180 km from the trench and loads only the overriding participant.
    No new plate force or velocity is introduced here.
    """
    convergence = np.maximum(-s.normal_speed, 0.)
    subduction = normal_partition.subduction(s)
    collision = normal_partition.collision(s)
    retreat = getattr(s, 'trench_retreat_speed', np.empty(0))
    if len(retreat) == len(convergence):
        convergence = convergence + np.where(subduction, retreat, 0.)
    exact_native=getattr(s,'native_spreading_version',0)==1
    selected = np.flatnonzero(collision | ((s.bcode == 5)&(not exact_native)) | subduction)
    mid, normal = s.bmid[selected].copy(), s.bn[selected].copy()
    length = s.bl[selected].copy()
    pa, pb = s.bp[selected].copy(), s.bq[selected].copy()
    code = np.where(collision[selected], 4, np.where(subduction[selected], 2, s.bcode[selected]))
    values = np.zeros((len(selected), 3))
    values[:, 0] = np.where(code == 4, convergence[selected], 0.)
    values[:, 1] = np.where(code == 5, np.maximum(s.normal_speed[selected], 0.), 0.)
    arc = code == 2
    if np.any(arc):
        indices = selected[arc]
        down_a = s.down[indices] == s.bp[indices]
        direction = s.bn[indices]*np.where(down_a, 1., -1.)[:, None]
        angle = 180./RADIUS_KM
        mid[arc] = np.cos(angle)*s.bmid[indices]+np.sin(angle)*direction
        normal[arc] = -np.sin(angle)*s.bmid[indices]+np.cos(angle)*direction
        above = np.where(down_a, s.bq[indices], s.bp[indices])
        pa[arc], pb[arc] = above, above
        values[arc, 2] = convergence[indices]*1.25*trench_history.weights(s)[indices]
    rift_id = np.full(len(selected), -1, np.int64)
    if hasattr(s, 'rift_pair_ids'):
        for i in np.flatnonzero((code == 5) & (values[:, 1] > 0)):
            key = tuple(sorted((int(s.plate_uid[pa[i]]), int(s.plate_uid[pb[i]]))))
            rift_id[i] = s.rift_pair_ids.get(key, -1)
    if exact_native:
        from native_spreading import prepare
        pieces=prepare(s)['continental']
        take=pieces['active'] & pieces['divergent']
        extra=np.zeros((int(take.sum()),3));extra[:,1]=pieces['normal_speed'][take]
        ra,rb=pieces['owner_a'][take],pieces['owner_b'][take]
        identities=np.full(len(ra),-1,np.int64)
        if hasattr(s,'rift_pair_ids'):
            for i,(a,b) in enumerate(zip(ra,rb)):
                identities[i]=s.rift_pair_ids.get(tuple(sorted((int(s.plate_uid[a]),int(s.plate_uid[b])))),-1)
        mid=np.concatenate((mid,pieces['mid'][take]));normal=np.concatenate((normal,pieces['normal'][take]))
        length=np.r_[length,pieces['length'][take]];pa=np.r_[pa,ra];pb=np.r_[pb,rb]
        values=np.concatenate((values,extra));rift_id=np.r_[rift_id,identities]
    return dict(mid=mid, normal=normal, length_km=length,
                owner_a=pa, owner_b=pb, values=values, rift_id=rift_id)


def sample_belts(xyz, owners, belts, *, width_km=BELT_WIDTH_KM, chunk_size=8192, return_sources=False):
    """Return N×K nonnegative smooth forcing without crossing plate owners.

    Segments are finite great-circle intervals represented by midpoint, tangent
    normal and length. Endcaps taper continuously with true spherical distance.
    The local maximum of their continuous profiles is itself continuous, even
    when the nearest segment changes. Working storage is linear in material
    points plus nearby segment/bin pairs; no point-by-segment matrix is formed.
    """
    points, owner = np.asarray(xyz, float), np.asarray(owners)
    mid, normal = np.asarray(belts['mid'], float), np.asarray(belts['normal'], float)
    length, values = np.asarray(belts['length_km'], float), np.asarray(belts['values'], float)
    pa, pb = np.asarray(belts['owner_a']), np.asarray(belts['owner_b'])
    if (points.ndim != 2 or points.shape[1] != 3 or owner.shape != (len(points),)
            or mid.ndim != 2 or mid.shape[1] != 3 or normal.shape != mid.shape
            or values.ndim != 2 or len(values) != len(mid)
            or any(v.shape != (len(mid),) for v in (length, pa, pb))
            or not np.isfinite(points).all() or not np.isfinite(mid).all()
            or not np.isfinite(normal).all() or not np.isfinite(length).all()
            or not np.isfinite(values).all() or np.any(values < 0)
            or np.any(length < 0) or np.any(length >= np.pi*RADIUS_KM)
            or not np.isfinite(width_km) or not 0 < width_km < np.pi*RADIUS_KM
            or chunk_size < 1):
        raise ValueError('Finite spherical boundary geometry, owners and nonnegative forcing must align.')
    result = np.zeros((len(points), values.shape[1]))
    source_index = np.full(result.shape, -1, np.int32) if return_sources else None
    if not len(points) or not len(mid):
        return (result, source_index) if return_sources else result
    pn, mn = np.linalg.norm(points, axis=1), np.linalg.norm(mid, axis=1)
    if np.any(pn < 1e-12) or np.any(mn < 1e-12):
        raise ValueError('Spherical points and boundary midpoints must be nonzero.')
    points, mid = points/pn[:, None], mid/mn[:, None]
    normal = normal-mid*np.sum(normal*mid, axis=1)[:, None]
    nn = np.linalg.norm(normal, axis=1)
    if np.any(nn < 1e-12):
        raise ValueError('Boundary normals must have a nonzero tangent component.')
    normal /= nn[:, None]
    tangent = np.cross(normal, mid)
    # A fixed geographic index is only a search aid; exact finite spherical
    # distances below reject the excess candidates, including at either pole.
    w, h = 128, 64
    lon = np.arctan2(points[:, 1], points[:, 0])
    lat = np.arcsin(np.clip(points[:, 2], -1, 1))
    cells = np.clip(((np.pi/2-lat)*h/np.pi).astype(int), 0, h-1)*w + (((lon+np.pi)*w/(2*np.pi)).astype(int) % w)
    order = np.argsort(cells, kind='stable')
    counts = np.bincount(cells, minlength=w*h)
    starts = np.r_[0, np.cumsum(counts)]
    padding = np.sqrt(2.)*np.pi/h*RADIUS_KM
    segments, bins = candidate_cells(mid, width_km+length/2+padding, w, h)
    segment_starts = np.r_[0, np.cumsum(np.bincount(segments, minlength=len(mid)))]
    for k in range(len(mid)):
        if not np.any(values[k] > 0):
            continue
        selected_bins = bins[segment_starts[k]:segment_starts[k+1]]
        parts = [order[starts[b]:starts[b+1]] for b in selected_bins if counts[b]]
        if not parts:
            continue
        relevant = np.concatenate(parts)
        relevant = relevant[(owner[relevant] == pa[k]) | (owner[relevant] == pb[k])]
        half = length[k]/(2*RADIUS_KM)
        end_a = np.cos(half)*mid[k]+np.sin(half)*tangent[k]
        end_b = np.cos(half)*mid[k]-np.sin(half)*tangent[k]
        for begin in range(0, len(relevant), int(chunk_size)):
            selected = relevant[begin:begin+int(chunk_size)]
            x = points[selected]
            cross = x@normal[k]
            along = np.arctan2(x@tangent[k], x@mid[k])
            # atan2 cross/dot is stable at the end points, unlike arccos of a
            # near-unit dot product; both paths are invariant under rotation.
            da = np.arctan2(np.linalg.norm(np.cross(x, end_a), axis=1), x@end_a)
            db = np.arctan2(np.linalg.norm(np.cross(x, end_b), axis=1), x@end_b)
            distance = np.where(np.abs(along) <= half,
                                np.abs(np.arcsin(np.clip(cross, -1, 1))), np.minimum(da, db))*RADIUS_KM
            weight = np.cos(np.pi*.5*np.minimum(distance/width_km, 1.))**2
            weight[distance >= width_km] = 0.
            candidate = weight[:, None]*values[k]
            if return_sources:
                source_index[selected] = np.where(candidate > result[selected], k, source_index[selected])
            result[selected] = np.maximum(result[selected], candidate)
    return (result, source_index) if return_sources else result
