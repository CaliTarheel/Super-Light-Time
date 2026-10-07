"""Local spherical geometry for mechanically active raster plate interfaces.

Winner-label staircases alternate grid-axis normals even when a physical contact
is straight. Here the normal comes from the local fractional p/q support field.
This is a subcell reconstruction of that field, not an imposed arc or a new
boundary. No support, plate labels or buoyant material are changed.
"""
import numpy as np


def _unit(vectors):
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-30)


def _lookup(points, width, height):
    """Bilinear pixel-centre weights, reflecting across each same pole."""
    x = (np.arctan2(points[..., 1], points[..., 0])+np.pi)*width/(2*np.pi)-.5
    y = (np.pi/2-np.arcsin(np.clip(points[..., 2], -1, 1)))*height/np.pi-.5
    iy = np.floor(y).astype(np.int64)
    fy = y-iy
    result = []
    for row, wy in ((iy, 1-fy), (iy+1, fy)):
        crossed = (row < 0) | (row >= height)
        reflected = np.where(row < 0, -row-1, np.where(row >= height, 2*height-row-1, row)).clip(0, height-1)
        xx = x+crossed*(width/2)
        ix = np.floor(xx).astype(np.int64)
        fx = xx-ix
        result.extend(((reflected*width+ix % width, (1-fx)*wy),
                       (reflected*width+(ix+1) % width, fx*wy)))
    nearest = np.clip(np.floor(y+.5).astype(np.int64), 0, height-1)*width + np.floor(x+.5).astype(np.int64) % width
    return result, nearest


def refine_boundary_geometry(plate, support, xyz, width, height, edge_a, edge_b,
                             edge_mid, edge_normal, edge_length, *, chunk_size=2048):
    """Reconstruct interface normals and projected lengths in bounded batches.

    Inputs are the existing actual contact edges (different owners at a/b),
    their unit-sphere midpoints/normals, and face lengths in any consistent unit.
    ``support`` is shape (plate capacity, width*height); ``xyz`` is the existing
    (width*height, 3) unit-sphere cell-centre grid. Longitude wraps and stencil
    spacing is physical angle, so the stencil does not collapse near the poles.

    A weighted tangent-plane fit uses the normalized support difference q-p on
    two rings, radius 1.1 and 2.2 latitude cells. Samples dominated by another
    plate are excluded. An inner-ring third plate marks an ambiguous junction;
    those edges, ill-conditioned fits and orientation conflicts keep their raw
    geometry. This deliberately preserves junctions and unresolved small pieces.

    On accepted edges, length = raw length * dot(fitted normal, raw normal).
    Thus a staircase approximating a diagonal front contributes its projected
    length, not the longer sum of orthogonal steps. This is not an exact area
    conservation scheme and does not evolve the boundary or select its type.
    The caller must use the returned geometry consistently in all mechanics.

    Returns new arrays: midpoints, normals, lengths, refined and junction masks,
    plus compact diagnostics. Inputs remain unchanged. No SciPy or full-world
    temporary per-pair fields are used.
    """
    width, height = int(width), int(height)
    if width < 2 or height < 2 or chunk_size < 1:
        raise ValueError('Boundary reconstruction requires width/height at least two and positive chunk size.')
    plate = np.asarray(plate).reshape(-1)
    support, xyz = np.asarray(support), np.asarray(xyz)
    a, b = np.asarray(edge_a, dtype=np.int64), np.asarray(edge_b, dtype=np.int64)
    mid = np.asarray(edge_mid, dtype=np.float64)
    raw_normal = np.asarray(edge_normal, dtype=np.float64)
    raw_length = np.asarray(edge_length, dtype=np.float64)
    count = len(a)
    if (plate.shape != (width*height,) or xyz.shape != (width*height, 3) or
            support.ndim != 2 or support.shape[1] != width*height or b.shape != (count,) or
            mid.shape != (count, 3) or raw_normal.shape != (count, 3) or raw_length.shape != (count,)):
        raise ValueError('Boundary reconstruction input shapes do not match.')
    if np.any(a < 0) or np.any(b < 0) or np.any(a >= len(plate)) or np.any(b >= len(plate)):
        raise ValueError('Boundary cells lie outside the globe.')
    if np.any(plate[a] == plate[b]):
        raise ValueError('Boundary reconstruction accepts only actual two-owner contact edges.')
    result_mid, result_normal, result_length = mid.copy(), raw_normal.copy(), raw_length.copy()
    refined, junction = np.zeros(count, bool), np.zeros(count, bool)
    # Equal angular directions give an isotropic stencil before raster sampling.
    angle = np.arange(8)*np.pi/4
    radii = np.repeat([1.1, 2.2], 8)
    u = radii*np.tile(np.cos(angle), 2)
    v = radii*np.tile(np.sin(angle), 2)
    base_weight = np.exp(-.5*radii*radii)
    theta = radii*np.pi/height
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    weak_count, conflict_count = 0, 0
    for start in range(0, count, int(chunk_size)):
        stop = min(start+int(chunk_size), count)
        aa, bb = a[start:stop], b[start:stop]
        p, q = plate[aa], plate[bb]
        phi_a = support[q, aa]-support[p, aa]
        phi_b = support[q, bb]-support[p, bb]
        difference = phi_b-phi_a
        fraction = np.divide(-phi_a, difference, out=np.full(len(aa), .5), where=np.abs(difference) > 1e-10).clip(0, 1)
        center = _unit(xyz[aa]*(1-fraction[:, None])+xyz[bb]*fraction[:, None])
        # Pole-safe tangent frame; the resulting fit is a Cartesian vector.
        reference = np.zeros_like(center)
        reference[:, 2] = 1
        polar = np.abs(center[:, 2]) > .9
        reference[polar] = [1, 0, 0]
        east = _unit(np.cross(reference, center))
        north = np.cross(center, east)
        tangent = (u[None, :, None]*east[:, None, :]+v[None, :, None]*north[:, None, :])/radii[None, :, None]
        probes = center[:, None, :]*cos_theta[None, :, None]+tangent*sin_theta[None, :, None]
        lookup, nearest = _lookup(probes, width, height)
        value_p, value_q = np.zeros(nearest.shape), np.zeros(nearest.shape)
        for cells, weight in lookup:
            value_p += support[p[:, None], cells]*weight
            value_q += support[q[:, None], cells]*weight
        total = value_p+value_q
        contrast = np.divide(value_q-value_p, total, out=np.zeros_like(total), where=total > 1e-8)
        owner = plate[nearest]
        pair = (owner == p[:, None]) | (owner == q[:, None])
        is_junction = np.any(~pair[:, :8], axis=1)
        junction[start:stop] = is_junction
        weights = base_weight[None, :]*pair*(total > 1e-8)
        weight_sum = np.maximum(weights.sum(axis=1), 1e-20)
        mean_u = (weights*u).sum(axis=1)/weight_sum
        mean_v = (weights*v).sum(axis=1)/weight_sum
        mean_c = (weights*contrast).sum(axis=1)/weight_sum
        du, dv = u[None, :]-mean_u[:, None], v[None, :]-mean_v[:, None]
        dc = contrast-mean_c[:, None]
        uu, uv, vv = (weights*du*du).sum(axis=1), (weights*du*dv).sum(axis=1), (weights*dv*dv).sum(axis=1)
        uc, vc = (weights*du*dc).sum(axis=1), (weights*dv*dc).sum(axis=1)
        determinant = uu*vv-uv*uv
        gx = np.divide(vv*uc-uv*vc, determinant, out=np.zeros(len(aa)), where=determinant > 1e-12)
        gy = np.divide(uu*vc-uv*uc, determinant, out=np.zeros(len(aa)), where=determinant > 1e-12)
        gradient_size = np.hypot(gx, gy)
        normal = _unit(gx[:, None]*east+gy[:, None]*north)
        face_normal = _unit(raw_normal[start:stop]-center*np.sum(raw_normal[start:stop]*center, axis=1)[:, None])
        alignment = np.sum(normal*face_normal, axis=1)
        sided = np.any((contrast < -.03) & pair, axis=1) & np.any((contrast > .03) & pair, axis=1)
        well_conditioned = (determinant > .01*(uu+vv)**2) & (gradient_size > .03) & sided
        consistent = alignment > .015
        accepted = ~is_junction & well_conditioned & consistent & np.isfinite(normal).all(axis=1)
        weak_count += int(np.count_nonzero(~is_junction & ~well_conditioned))
        conflict_count += int(np.count_nonzero(~is_junction & well_conditioned & ~consistent))
        selected = np.flatnonzero(accepted)+start
        refined[selected] = True
        result_mid[selected] = center[accepted]
        result_normal[selected] = normal[accepted]
        result_length[selected] = raw_length[selected]*np.clip(alignment[accepted], 0, 1)
    return dict(midpoints=result_mid, normals=result_normal, lengths=result_length,
                refined=refined, junction=junction,
                diagnostics=dict(edges=count, refined_edges=int(refined.sum()),
                                 junction_edges=int(junction.sum()), weak_fit_edges=weak_count,
                                 orientation_conflict_edges=conflict_count,
                                 raw_length=float(raw_length.sum()), corrected_length=float(result_length.sum())))
