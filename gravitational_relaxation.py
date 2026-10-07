"""Virtual-work-consistent gravity for conserved overlapping material columns.

The reduced potential divided by K = rho_c*g*(1-rho_c/rho_m) is
  E/K = 1/2 sum A_i (h_i-h0_i)^2 + sum A_ij h_i h_j.
Areas and thicknesses use km here. The immutable h0 is an advected balanced
reference column: its area term represents prescribed background traction,
not a solved ocean or mantle column. Each distinct sheet overlap enters once.
Physical column volumes are held fixed when differentiating geometry.

The overlap uses physical great-circle halfspaces and its area derivative is
the swept work of active original boundary arcs. At genuinely coincident
edges, equal sharing selects the midpoint of the two one-sided
directional works. Numerical predicate filters do not enlarge the intersection.
No material, column, height, contact ordering or simulation object is mutated.

This potential has two consumers. rigid_mode_partition splits its gradient into
the per-owner rigid rotations, which the instantaneous plate force balance
applies as torques, and a residual, which relax spreads as internal sheet flow.
When relax is given vertex_owner it uses that residual, so one potential is
spent once. overlap_area_rigid_modes differentiates the same clipped areas
without the thickness weights, for the contact's overlap-area coordinate.
"""
from __future__ import annotations

import numpy as np

import mesh_coverage
from material_surface import spherical_face_areas
# Shared with the constrained endpoint solver: geometry it accepts as feasible
# must not be rejected here, or the backtracking search below shrinks the step
# to nothing and the interval fails instead of advancing.
from bounded_gravity import FEASIBILITY_TOLERANCE

SECONDS_PER_MYR = 365.25 * 86400. * 1e6

# The one reduced density coefficient K = rho_c g (1 - rho_c/rho_m) that turns
# the km^4 functional above into joules. normalized_body_force computes the same
# product from its own keywords; these are its defaults, named once so that the
# rigid-mode partition below and the plate balance quote the same number.
CRUST_DENSITY_KG_M3 = 2800.
MANTLE_DENSITY_KG_M3 = 3300.
GRAVITY_M_S2 = 9.81
RIGID_PARTITION_VERSION = 1
ENERGY_UNIT_J = CRUST_DENSITY_KG_M3*GRAVITY_M_S2*(1-CRUST_DENSITY_KG_M3/MANTLE_DENSITY_KG_M3)*1e12


def _paired_area_derivative(subjects, clippers, radius):
    """Boundary shape derivative of the physical spherical intersection.

    For an active original edge with inward normal n, area variation is
    R**2 integral(dn.x ds). Corners have zero boundary measure, so the formula
    does not differentiate clipped endpoint positions or vanishing fan areas.
    Genuine coincident arcs share their work equally; this is the
    midpoint of the two one-sided directional works for every joint motion.
    """
    from spherical_predicates import edge_distances

    count = len(subjects)
    if not count:
        return np.empty(0), np.empty((0, 18)), 0
    triangles = (np.asarray(subjects, float), np.asarray(clippers, float))
    gradient = np.zeros((count, 6, 3))
    near_edges = 0
    for side in (0, 1):
        own, other = triangles[side], triangles[1-side]
        following = np.roll(own, -1, axis=1)
        other_following = np.roll(other, -1, axis=1)
        cross = np.cross(own, following-own)
        length = np.linalg.norm(cross, axis=2)
        normal = cross/length[..., None]
        other_cross = np.cross(other, other_following-other)
        distance = edge_distances(own[:, :, None, :], other[:, None, :, :],
                                  other_following[:, None, :, :])
        near_edges += int(np.count_nonzero(np.abs(distance)
            <= 1e-12*np.linalg.norm(other_cross, axis=2)[:, None, :]))
        before, after = distance, np.roll(distance, -1, axis=1)
        denominator = before-after
        fraction = np.divide(before, denominator, out=np.zeros_like(before),
                             where=denominator != 0.)
        low = np.max(np.where(before < 0., fraction, 0.), axis=2)
        high = np.min(np.where(after < 0., fraction, 1.), axis=2)
        coincident = (before == 0.) & (after == 0.)
        blocked = np.any((before < 0.) & (after < 0.), axis=2)
        active = (high > low) & ~blocked
        # Opposite-inward coincident arcs describe area birth. They need the
        # same midpoint selection as existing-area arcs for adjacent triangle
        # contributions to cancel under continuous material motion.
        share = np.where(np.any(coincident, axis=2), .5, 1.)
        # Inactive intervals may have out-of-range fractions. They contribute
        # no work; keep their scratch coordinates finite without changing any
        # active edge or admitting an out-of-segment physical crossing.
        low = np.where(active, low, 0.)
        high = np.where(active, high, 0.)
        start = mesh_coverage._unit(own+low[..., None]*(following-own))
        end = mesh_coverage._unit(own+high[..., None]*(following-own))
        # tan(theta/2)*(start+end) is the exact vector integral along the arc;
        # chord lengths avoid cross-product cancellation on very short arcs.
        moment = (start+end)*(np.linalg.norm(end-start, axis=2)
            / np.linalg.norm(end+start, axis=2))[..., None]
        moment -= normal*np.sum(normal*moment, axis=2)[..., None]
        weight = radius**2*share[..., None]*moment/length[..., None]
        weight *= active[..., None]
        local = np.cross(following, weight) + np.roll(np.cross(weight, own), 1, axis=1)
        gradient[:, side*3:side*3+3] = local
    area = mesh_coverage._intersection_areas(triangles[0], triangles[1], radius)
    return area, gradient.reshape(count, 18), near_edges


def face_area_gradients(points, faces, radius=6371.):
    """Exact signed spherical-area derivative with respect to unit vertices."""
    a, b, c = points[faces].transpose(1, 0, 2)
    numerator = np.sum(a*np.cross(b-a, c-a), axis=1)
    denominator = 1+np.sum(a*b+b*c+c*a, axis=1)
    scale = 2*radius**2 / np.maximum(numerator**2+denominator**2, 1e-300)
    result = []
    for first, second in ((b, c), (c, a), (a, b)):
        result.append(scale[:, None]*(denominator[:, None]*np.cross(first, second)
                                      - numerator[:, None]*(first+second)))
    return np.stack(result, axis=1)


def _physical_overlap(points, faces, sheets, radius, overlap):
    """Complete force geometry, independent of the contact bookkeeping floor.

    A thin positive overlap can have finite boundary work. A zero-area shared
    edge can also be the boundary of a newly appearing area. Both belong to
    the force stencil even when they do not yet establish a contact record.
    Validate supplied cached areas before extending an ordinary contact ledger.
    """
    if overlap is not None:
        nfaces = len(faces)
        first, second, expected = (np.asarray(overlap[k]) for k in ('first', 'second', 'area_km2'))
        if (first.ndim != 1 or second.shape != first.shape or expected.shape != first.shape
                or first.dtype.kind not in 'iu' or second.dtype.kind not in 'iu'
                or np.any(first < 0) or np.any(second >= nfaces) or np.any(first >= second)
                or not np.isfinite(expected).all() or np.any(expected < 0.)):
            raise ValueError('Cached overlap requires aligned integer pairs and finite nonnegative areas.')
        original = first.astype(np.int64)*nfaces+second
        if len(np.unique(original)) != len(original):
            raise ValueError('Cached overlap contains duplicate material pairs.')
    physical = mesh_coverage.material_overlaps(points, faces, sheets,
        radius_km=radius, include_touching=True)
    if overlap is not None:
        keys = physical['first'].astype(np.int64)*nfaces+physical['second']
        where = np.searchsorted(keys, original)
        if np.any(where >= len(keys)) or np.any(keys[where] != original):
            raise ValueError('Cached overlap pairs disagree with physical force geometry.')
        error = np.max(np.abs(physical['area_km2'][where]-expected), initial=0.)
        if error > max(1e-8, 1e-10*float(expected.max(initial=0.))):
            raise ValueError('Cached overlap areas disagree with physical force geometry.')
        if overlap.get('complete_force_stencil', False) and len(original) != len(keys):
            raise ValueError('Cached force stencil is incomplete for physical geometry.')
    return physical


def reference_energy(points, faces, volumes_km3, reference_thickness_km, sheets, *, radius=6371., overlap=None,
                     density_profile=None):
    """Reduced E/K in km4 at conserved face volumes; no state changes."""
    area = spherical_face_areas(points, faces, radius)
    height = np.asarray(volumes_km3)/area
    overlap = _physical_overlap(points, faces, sheets, radius, overlap)
    first, second, weight = (np.asarray(overlap[k]) for k in ('first', 'second', 'area_km2'))
    import column_density
    own, pair = column_density.coefficients(density_profile, sheets, first, second, weight)
    return float(.5*np.sum(own*area*(height-reference_thickness_km)**2)
                 + np.sum(pair*weight*height[first]*height[second]))


def energy_gradient(points, faces, thickness_km, reference_thickness_km, sheets, *,
                    radius=6371., overlap=None, chunk_pairs=128, density_profile=None):
    """Return reference energy and its exact fixed-topology geometric gradient."""
    points, faces = np.asarray(points, float), np.asarray(faces)
    height, reference = np.asarray(thickness_km, float), np.asarray(reference_thickness_km, float)
    if (points.ndim != 2 or points.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3
            or faces.dtype.kind not in 'iu' or height.shape != (len(faces),) or reference.shape != height.shape
            or np.any(faces < 0) or np.any(faces >= len(points))
            or np.asarray(sheets).shape != height.shape or not np.isfinite(points).all()
            or not np.isfinite(height).all() or not np.isfinite(reference).all()
            or np.any(height <= 0) or np.any(reference <= 0) or not np.isfinite(radius) or radius <= 0
            or np.any(np.abs(np.linalg.norm(points, axis=1)-1) > 1e-8)
            or isinstance(chunk_pairs, bool) or int(chunk_pairs) != chunk_pairs or chunk_pairs < 1):
        raise ValueError('Gravity requires positive, aligned material columns and unit spherical geometry.')
    area = spherical_face_areas(points, faces, radius)
    triangle = points[faces]
    oriented = np.einsum('ij,ij->i', triangle[:,0], np.cross(triangle[:,1],triangle[:,2]))
    if np.any(area <= 0) or np.any(oriented <= 0):
        raise ValueError('Gravity cannot differentiate degenerate material faces.')
    overlap = _physical_overlap(points, faces, sheets, radius, overlap)
    first, second, weight = (np.asarray(overlap[k]) for k in ('first', 'second', 'area_km2'))
    import column_density
    own, pair = column_density.coefficients(density_profile, sheets, first, second, weight)
    derivative_area = .5*own*(reference**2-height**2)
    pair_load = pair*weight*height[first]*height[second]
    np.add.at(derivative_area, first, -pair_load/area[first])
    np.add.at(derivative_area, second, -pair_load/area[second])
    face_gradient = derivative_area[:, None, None]*face_area_gradients(points, faces, radius)
    gradient = np.zeros_like(points)
    np.add.at(gradient, faces.ravel(), face_gradient.reshape(-1, 3))
    near_edges, max_area_error = 0, 0.
    triangles = points[faces]
    for start in range(0, len(first), int(chunk_pairs)):
        sl = slice(start, start+int(chunk_pairs)); aa, bb = first[sl], second[sl]
        measured, forward, near = _paired_area_derivative(triangles[aa], triangles[bb], radius)
        reverse_area, reverse, other_near = _paired_area_derivative(triangles[bb], triangles[aa], radius)
        reverse = reverse.reshape(-1, 6, 3)[:, [3, 4, 5, 0, 1, 2]].reshape(-1, 18)
        derivative = .5*(forward+reverse)*height[aa, None]*height[bb, None]*pair[sl, None]
        np.add.at(gradient, np.concatenate((faces[aa], faces[bb]), axis=1).ravel(), derivative.reshape(-1, 3))
        near_edges += near+other_near
        error = float(max(np.max(np.abs(measured-weight[sl]), initial=0.),
                          np.max(np.abs(reverse_area-weight[sl]), initial=0.)))
        max_area_error = max(max_area_error, error)
        if error > max(1e-8, 1e-10*float(weight[sl].max(initial=0.))):
            raise ValueError('Differentiated overlap disagrees with the independent geometric area.')
    gradient -= points*np.sum(gradient*points, axis=1)[:, None]
    energy = float(.5*np.sum(own*area*(height-reference)**2)+pair_load.sum())
    return energy, gradient, dict(model='balanced-reference self GPE plus distinct-sheet overlap GPE',
        energy_over_density_coefficient_km4=energy,
        overlap_pairs=int(np.count_nonzero(weight > 0.)), force_stencil_pairs=len(first),
        maximum_overlap_area_check_error_km2=max_area_error,
        clipping_near_edge_observations=near_edges,
        derivative_scope='physical boundary shape derivative; equal work sharing on exact coincident arcs',
        reference_scope='advected stress-free column with prescribed balancing background traction',
        density_scope=('uniform legacy crust' if density_profile is None
                       else 'retained basal dense phase; ordered locally compensated stack'),
        changes_material_volume=False)


def overlap_area_rigid_modes(points, faces, first, second, vertex_owner, *,
                             radius=6371., chunk_pairs=128):
    """Rate of each distinct-sheet overlap area under rigid rotation of its owners.

    The suture weld's generalized coordinate is the OVERLAP AREA of a contact,
    not an edge-normal speed, because that is the coordinate whose conjugate is
    a line force in N/m and because tangential sliding along a suture -- 95% of
    the relative motion at contact 4 -- changes it by nothing, as it should.
    This returns, for every ordered pair of the cached ledger, the two vectors
    a with dA/dt = a_first . w_first + a_second . w_second (km^2 per unit time
    for w in radians per the same unit).

    It reuses the SAME differentiated spherical clipping and the SAME
    exchange-symmetric average as energy_gradient, so the weld and the GPE
    driver differentiate one geometry, not two approximations of it.

    Pairs whose two faces have the same owner are skipped exactly, not
    approximately: a pair area is invariant under a rigid rotation applied to
    both of its triangles, so a_first + a_second is zero there identically.
    """
    points, faces = np.asarray(points, float), np.asarray(faces)
    first, second = np.asarray(first), np.asarray(second)
    owner = np.asarray(vertex_owner)
    if (points.ndim != 2 or points.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3
            or first.shape != second.shape or first.ndim != 1
            or (len(first) and (first.min() < 0 or first.max() >= len(faces)
                                or second.min() < 0 or second.max() >= len(faces)))
            or owner.shape != (len(points),) or owner.dtype.kind not in 'iu'
            or not np.isfinite(points).all() or not np.isfinite(radius) or radius <= 0
            or isinstance(chunk_pairs, bool) or int(chunk_pairs) != chunk_pairs or chunk_pairs < 1):
        raise ValueError('Overlap area modes require an aligned pair ledger and owned unit geometry.')
    owner_first = owner[faces[first, 0]] if len(first) else np.empty(0, owner.dtype)
    owner_second = owner[faces[second, 0]] if len(second) else np.empty(0, owner.dtype)
    mode_first, mode_second = np.zeros((len(first), 3)), np.zeros((len(second), 3))
    active = np.flatnonzero(owner_first != owner_second)
    triangles = points[faces]
    for start in range(0, len(active), int(chunk_pairs)):
        take = active[start:start+int(chunk_pairs)]
        aa, bb = first[take], second[take]
        _, forward, _ = _paired_area_derivative(triangles[aa], triangles[bb], radius)
        _, reverse, _ = _paired_area_derivative(triangles[bb], triangles[aa], radius)
        reverse = reverse.reshape(-1, 6, 3)[:, [3, 4, 5, 0, 1, 2]].reshape(-1, 18)
        derivative = .5*(forward+reverse).reshape(-1, 6, 3)
        mode_first[take] = np.cross(triangles[aa], derivative[:, :3]).sum(axis=1)
        mode_second[take] = np.cross(triangles[bb], derivative[:, 3:]).sum(axis=1)
    return dict(owner_first=owner_first, owner_second=owner_second,
                mode_first=mode_first, mode_second=mode_second,
                differentiated_pairs=int(len(active)),
                scope='physical boundary shape derivative; equal work sharing on exact coincident arcs')


def normalized_body_force(points, faces, area_km2, gradient, *, radius=6371.,
                          viscosity_pa_s=1e23, lithosphere_thickness_km=100., length_km=400.,
                          crust_density_kg_m3=2800., mantle_density_kg_m3=3300., gravity_m_s2=9.81):
    """Convert negative virtual-work gradient to the viscous solver's km/Myr RHS."""
    controls = (radius, viscosity_pa_s, lithosphere_thickness_km, length_km,
                crust_density_kg_m3, mantle_density_kg_m3, gravity_m_s2)
    if not np.isfinite(controls).all() or min(controls) <= 0 or crust_density_kg_m3 >= mantle_density_kg_m3:
        raise ValueError('Gravity and viscosity controls must be finite, positive and buoyant.')
    lumped = np.bincount(np.asarray(faces).ravel(), weights=np.repeat(np.asarray(area_km2)/3, 3), minlength=len(points))
    density_coefficient = crust_density_kg_m3*gravity_m_s2*(1-crust_density_kg_m3/mantle_density_kg_m3)
    basal_drag = 2*viscosity_pa_s*(lithosphere_thickness_km*1000)/(length_km*1000)**2
    scale = density_coefficient*SECONDS_PER_MYR/(radius*basal_drag)
    force = np.divide(-scale*np.asarray(gradient), lumped[:, None], out=np.zeros_like(gradient), where=lumped[:, None] > 0)
    return force, dict(density_coefficient_n_m3=density_coefficient, basal_drag_pa_s_m=basal_drag,
        viscosity_pa_s=viscosity_pa_s, lithosphere_thickness_km=lithosphere_thickness_km,
        maximum_normalized_body_force_km_myr=float(np.max(np.linalg.norm(force, axis=1), initial=0.)))


def _lumped_vertex_area(faces, area_km2, count):
    """Vertex quadrature weight of the piecewise-linear sheet: a third per face."""
    return np.bincount(np.asarray(faces).ravel(),
                       weights=np.repeat(np.asarray(area_km2, float)/3., 3), minlength=count)


def rigid_mode_partition(points, faces, area_km2, gradient, vertex_owner, *,
                         crust_density_kg_m3=CRUST_DENSITY_KG_M3,
                         mantle_density_kg_m3=MANTLE_DENSITY_KG_M3, gravity_m_s2=GRAVITY_M_S2):
    """Split this functional's gradient into per-plate rigid rotations and a residual.

    The potential above is one potential. Its gradient drives two different
    things in the engine -- whole-plate rotation, solved by the force balance,
    and internal spreading of the sheet, solved by relax -- and if both read the
    unsplit gradient the same collision GPE is spent twice: once as a torque that
    closes or opens the suture and again as a body force that flattens it. This
    function performs that split exactly, in the same lumped-area metric the
    viscous sheet uses, so that nothing is created or destroyed by the split.

    For each owner p, the three rigid modes at its vertices are m_k(v) = e_k x v.
    The least-squares rigid part in the lumped-area inner product solves
    J_p w_p = t_p, with t_p = sum_v v x g_v and J_p = sum_v L_v (I - v v^T).
    The residual g' = g - L_v (w_p x v) then has zero rigid component per owner,
    which is the statement that relax can no longer rotate a plate.

    Two exact properties follow from the functional and are asserted by the
    caller's tests rather than assumed here. The self term 1/2 A (h-h0)^2 is
    invariant under a rigid rotation of one owner's own vertices (faces never
    straddle owners, so its areas do not change), so t_p carries ONLY the
    distinct-sheet overlap cross term -- there is no separate continental GPE
    torque to add and no h0 = h trick is needed to isolate it. And the whole
    functional is invariant under a global rotation, so sum_p t_p = 0 to
    roundoff: the collision torque is equal and opposite on the pair, which is
    Newton's third law for a contact force and the only reason a summed
    (never averaged) driver can be trusted.

    Torques are returned in N m: the km^4 gradient times K = rho_c g (1-rho_c/rho_m)
    times (1e3 m/km)^4. The residual is returned in the gradient's own km^4 units,
    ready for normalized_body_force.
    """
    points = np.asarray(points, float)
    faces = np.asarray(faces)
    gradient = np.asarray(gradient, float)
    area = np.asarray(area_km2, float)
    owner = np.asarray(vertex_owner)
    controls = (crust_density_kg_m3, mantle_density_kg_m3, gravity_m_s2)
    if (points.ndim != 2 or points.shape[1] != 3 or gradient.shape != points.shape
            or faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu'
            or np.any(faces < 0) or np.any(faces >= len(points)) or area.shape != (len(faces),)
            or owner.shape != (len(points),) or owner.dtype.kind not in 'iu'
            or not np.isfinite(points).all() or not np.isfinite(gradient).all()
            or not np.isfinite(area).all() or np.any(area <= 0)
            or not np.isfinite(controls).all() or min(controls) <= 0
            or crust_density_kg_m3 >= mantle_density_kg_m3):
        raise ValueError('The rigid-mode partition requires an aligned tangent gradient and owned vertices.')
    lumped = _lumped_vertex_area(faces, area, len(points))
    coefficient = crust_density_kg_m3*gravity_m_s2*(1-crust_density_kg_m3/mantle_density_kg_m3)
    moment = np.cross(points, gradient)
    owners = np.unique(owner)
    torque = np.zeros((len(owners), 3))
    rotation = np.zeros((len(owners), 3))
    residual = gradient.copy()
    conditioning = []
    for index, plate in enumerate(owners):
        select = owner == plate
        coordinates, weights = points[select], lumped[select]
        raw = moment[select].sum(axis=0)
        torque[index] = -coefficient*1e12*raw
        inertia = (np.eye(3)*weights.sum()
                   - np.einsum('v,vi,vj->ij', weights, coordinates, coordinates))
        # A patch whose vertices are nearly coplanar through the origin has one
        # weak rigid mode (rotation about its own normal moves it least). Report
        # the conditioning instead of silently amplifying roundoff into it.
        values = np.linalg.eigvalsh(inertia)
        conditioning.append(float(values.max()/max(values.min(), 1e-300)))
        rotation[index] = np.linalg.solve(inertia, raw)
        residual[select] -= weights[:, None]*np.cross(rotation[index][None, :], coordinates)
    check = np.cross(points, residual)
    remaining = np.zeros((len(owners), 3))
    for index, plate in enumerate(owners):
        remaining[index] = check[owner == plate].sum(axis=0)
    scale = max(float(np.abs(moment).sum()), 1e-300)
    magnitude = np.linalg.norm(torque, axis=1)
    return dict(owners=owners.copy(), torque_n_m=torque, rigid_rotation=rotation, residual=residual,
        diagnostics=dict(version=RIGID_PARTITION_VERSION,
            model='per-owner least-squares rigid rotation of the shared gravitational functional',
            density_coefficient_n_m3=coefficient,
            owner_torque_n_m={int(plate): torque[index].tolist() for index, plate in enumerate(owners)},
            maximum_owner_torque_n_m=float(magnitude.max(initial=0.)),
            action_reaction_residual=float(np.linalg.norm(torque.sum(axis=0))
                                           / max(float(magnitude.max(initial=0.)), 1e-300)),
            residual_rigid_fraction=float(np.abs(remaining).sum()/scale),
            rigid_gradient_fraction=float(np.linalg.norm(gradient-residual)
                                          / max(float(np.linalg.norm(gradient)), 1e-300)),
            owner_inertia_condition=conditioning,
            scope='rigid share of the same potential relax minimises; self term contributes none by construction'))


def relax(points, faces, volumes_km3, reference_thickness_km, sheets, dt, *,
          rigid_mask, minimum_area_km2, maximum_area_km2, radius=6371.,
          length_km=100., viscosity_pa_s=1e23, lithosphere_thickness_km=100.,
          max_speed_km_myr=100., iterations=2048, tolerance=1e-8, iteration_extension=2048,
          max_substeps=64, max_backtracks=32, constraint_version=0, viscosity_weights=None,
          vertex_owner=None, density_profile=None, density_admission=None, entry_potential=None,
          numerical_policy=None):
    """Advance gravity alone with exact volume and an energy-decreasing search.

    Prescribed plate/loading work is handled by the caller before this split
    stage. Every accepted geometry here uses unchanged face volumes. Adaptive
    internal steps cover the whole requested interval; failure raises before
    returning any geometry, rather than silently dropping relaxation time.
    This is an explicit, reduced constant-viscosity evolution, not a static
    equilibrium solve or a replacement for a surface-process solver.

    With vertex_owner supplied the sheet is driven by the RESIDUAL of
    rigid_mode_partition instead of the raw gradient: the per-owner rigid share
    of this same potential has been handed to the instantaneous force balance as
    a torque, and spending it here as well would count the collision GPE twice.
    The energy that is still measured and still required to decrease is the whole
    potential, because the rotation the balance applies is real motion of the
    same material; only the part of the drive that relax is allowed to claim
    changes. The acceptance work always differentiates the complete potential
    along the proposed motion, including when no entry term is supplied.

    An explicit FrozenPotential adds continental-entry energy and its gradient
    before the same rigid/residual split. Hinges remain fixed at this mechanical
    stage's epoch. Both the line search and reported work use the combined
    potential; no native domain transport or phase/source update is implied.

    The iteration budget only bounds failures: the sheet solve stops at its
    stationarity gate, so a solve that converges within any budget is
    unchanged. The unconstrained predictor may use a bounded 2048-iteration
    extension after its original 2048 boundary check. This leaves constrained
    inverse budgets unchanged. The saved 95 Myr predictor needed 2054 total
    iterations at the same 1e-8 gate; successful earlier responses stay exact.
    """
    if type(constraint_version) is not int or constraint_version not in (0,1):
        raise ValueError('Unsupported gravity constraint policy.')
    if isinstance(iteration_extension,(bool,np.bool_)) or not isinstance(iteration_extension,(int,np.integer)) or iteration_extension<0:
        raise ValueError('Gravity iteration extension must be a nonnegative integer.')
    if entry_potential is not None:
        from continental_entry import FrozenPotential, EntryGeometryError
        if not isinstance(entry_potential,FrozenPotential) or vertex_owner is None:
            raise ValueError('Entry sheet work requires a frozen potential and the plate/residual partition.')
    from viscous_sheet import solve
    from deforming_regions import _quality
    from ridge_geometry import rotate
    points = np.asarray(points, float).copy()
    faces = np.asarray(faces)
    volumes, reference = np.asarray(volumes_km3, float), np.asarray(reference_thickness_km, float)
    rigid = np.asarray(rigid_mask)
    minimum, maximum = np.asarray(minimum_area_km2, float), np.asarray(maximum_area_km2, float)
    if (not np.isfinite(dt) or dt < 0 or not np.isfinite(max_speed_km_myr) or max_speed_km_myr <= 0
            or rigid.shape != (len(points),) or rigid.dtype.kind != 'b'
            or volumes.shape != (len(faces),) or not np.isfinite(volumes).all() or np.any(volumes <= 0)
            or minimum.shape != volumes.shape or maximum.shape != volumes.shape
            or not np.isfinite(minimum).all() or not np.isfinite(maximum).all()
            or np.any(minimum <= 0) or np.any(maximum < minimum)
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 1
                   for v in (max_substeps, max_backtracks))):
        raise ValueError('Gravity stepping requires a finite interval, conserved columns and positive area bounds.')
    initial = points.copy()
    initial_area = spherical_face_areas(points, faces, radius)
    if numerical_policy is not None:
        from numerical_accuracy import NumericalPolicy
        if not isinstance(numerical_policy,NumericalPolicy):raise ValueError('Gravity requires its immutable material accuracy policy.')
        numerical_policy.validate(len(faces))
        if tolerance!=numerical_policy.outer_tolerance:raise ValueError('Gravity tolerance differs from its declared numerical policy.')
        if not np.allclose(initial_area,numerical_policy.current_area_km2,rtol=2e-12,atol=0.):
            raise ValueError('Gravity area policy refers to a different accepted geometry.')
        if not np.array_equal(minimum,numerical_policy.minimum_area_km2) or not np.array_equal(maximum,numerical_policy.maximum_area_km2):
            raise ValueError('Gravity changed the nominal physical column bounds.')
    signed, quality = _quality(points, faces)
    initial_feasible=(np.all(initial_area>=minimum*(1-1e-10)) and np.all(initial_area<=maximum*(1+1e-10))
        if numerical_policy is None else numerical_policy.feasible(initial_area,minimum,maximum))
    if np.any(signed <= 1e-15) or not initial_feasible:
        raise ValueError('Gravity starts outside its positive material geometry/column bounds.')
    quality_floor = np.minimum(.15, quality*.5)
    elapsed, backtracks, evaluations = 0., 0, 0
    first_energy = last_energy = None
    first_column = last_column = first_entry = last_entry = None
    rows = []
    zeros = np.zeros_like(points)
    first_partition, rigid_fraction = None, 0.
    for at in range(max_substeps):
        if density_admission is not None:density_profile=density_admission.profile
        if entry_potential is not None:entry_potential.validate_density(density_profile)
        area = spherical_face_areas(points, faces, radius)
        energy, gradient, geometry = energy_gradient(points, faces, volumes/area, reference, sheets,
            radius=radius,density_profile=density_profile,
            overlap=None if density_admission is None else density_admission.overlap)
        column_energy=energy
        entry=None
        if entry_potential is not None:
            entry=entry_potential.evaluate(points,faces,volumes,sheets,radius=radius)
            energy+=entry['energy_j']/ENERGY_UNIT_J
            gradient=gradient+entry['vertex_gradient_j']/ENERGY_UNIT_J
            geometry=dict(geometry,entry_energy_j=entry['energy_j'],
                energy_scope='column plus continental entry; hinge fixed at this stage epoch')
        full_gradient=gradient
        evaluations += 1
        if first_energy is None:
            first_energy = energy
            first_column=column_energy
            first_entry=0. if entry is None else entry['energy_j']
        last_energy = energy
        last_column=column_energy
        last_entry=0. if entry is None else entry['energy_j']
        if elapsed >= dt*(1-1e-12):
            break
        partition = None
        if vertex_owner is not None:
            partition = rigid_mode_partition(points, faces, area, gradient, vertex_owner)
            if entry is not None:
                reaction=-entry['hinge_potential_torque_n_m'].sum(axis=0)
                torques=partition['torque_n_m']
                scale=max(float(np.linalg.norm(torques,axis=1).max(initial=0.)),float(np.linalg.norm(reaction)),1e-300)
                partition['diagnostics'].update(external_hinge_reaction_n_m=reaction.tolist(),
                    action_reaction_residual=float(np.linalg.norm(torques.sum(axis=0)+reaction)/scale),
                    scope='rigid material share of column plus entry energy; hinge reaction belongs to overriding plate')
            geometry = dict(geometry, rigid_mode_partition=partition['diagnostics'])
            if first_partition is None:
                first_partition = partition['diagnostics']
            # The rigid share is not discarded: the force balance is applying it
            # as plate rotation in this same step. Only relax stops claiming it.
            rigid_fraction = max(rigid_fraction, partition['diagnostics']['rigid_gradient_fraction'])
            gradient = partition['residual']
        force, units = normalized_body_force(points, faces, area, gradient, radius=radius,
            length_km=length_km, viscosity_pa_s=viscosity_pa_s,
            lithosphere_thickness_km=lithosphere_thickness_km)
        velocity, solver = solve(points, faces, area, zeros, rigid, np.zeros(len(points), bool),
            length_km, radius=radius, body_force=force, iterations=iterations, tolerance=tolerance,
            viscosity_weights=viscosity_weights, extension_iterations=iteration_extension)
        if not solver['converged']:
            raise ValueError('Gravity sheet solve failed its recomputed stationarity gate: '+str(solver['failure_reason']))
        if constraint_version:
            solver['stationarity_kind'] = 'unconstrained_linear'
        scale = min(1., max_speed_km_myr/max(float(np.linalg.norm(velocity, axis=1).max(initial=0.)), 1e-300))
        velocity *= scale
        # Audit the actual derivative of the complete potential.
        # Constraints can couple residual forcing back into rigid motion; the
        # projected drive alone is then not the energy change of that motion.
        work_gradient=full_gradient
        power = float(np.sum(work_gradient*velocity)/radius)
        # This bound is floating point energy resolution, tied to column
        # energy scale. It is not a tunable geological flattening threshold.
        energy_scale = float(np.sum(area*(volumes/area)**2))
        if entry is not None:energy_scale+=entry['absolute_energy_j']/ENERGY_UNIT_J
        roundoff = 128*np.finfo(float).eps*max(1., energy_scale)
        remaining = dt-elapsed
        if abs(power)*remaining <= roundoff:
            elapsed = dt
            rows.append(dict(dt_myr=remaining, stationary_within_energy_roundoff=True,
                energy_over_density_coefficient_km4=energy, virtual_power_km4_myr=power,
                energy_roundoff_km4=roundoff, energy_roundoff_scale_km4=energy_scale,
                solver=solver, gradient=geometry))
            if numerical_policy is not None:rows[-1]['numerical_accuracy_policy']=numerical_policy.evidence(area,minimum,maximum,include_arrays=False)
            break
        if power >= 0:
            raise ValueError('Gravity velocity fails the independent negative virtual-work check.')
        step = remaining
        axis = np.cross(points, velocity)/radius
        accepted = False
        bound_diagnostic = None
        # Shared across this substep's attempts only. Its contents depend on the
        # geometry (points/faces/area/rigid/viscosity), which is fixed for the
        # whole attempt loop, and not on `step`, which is all that changes between
        # attempts. It is deliberately NOT carried past the loop: once a trial is
        # accepted, `points` moves and every cached solve becomes stale.
        endpoint_basis_cache = {}
        # Why the latest trial was refused, for the error if none is accepted.
        rejection = None
        for attempt in range(max_backtracks):
            trial_velocity, trial_solver, trial_power = velocity, solver, power
            trial_scale = scale
            trial_bound = None
            trial = rotate(points, axis*step)
            trial /= np.linalg.norm(trial, axis=1)[:, None]
            trial[rigid] = points[rigid]
            new_area = spherical_face_areas(trial, faces, radius)
            nominal_feasible=(np.all(new_area>=minimum*(1-FEASIBILITY_TOLERANCE)) and np.all(new_area<=maximum*(1+FEASIBILITY_TOLERANCE))
                if numerical_policy is None else numerical_policy.feasible(new_area,minimum,maximum))
            signed,quality=_quality(trial,faces)
            if constraint_version and (not nominal_feasible or (numerical_policy is not None and np.any(quality<quality_floor))):
                from bounded_gravity import (solve_endpoint, ConstraintSolveError, SpeedBoundError,
                                             InverseSolveError, ActiveSetBudgetError)
                try:
                    trial, trial_velocity, trial_solver, trial_bound = solve_endpoint(
                        points, faces, area, force, velocity, solver, rigid, minimum, maximum, step,
                        radius=radius, length_km=length_km, iterations=iterations,
                        tolerance=tolerance, max_speed_km_myr=max_speed_km_myr,
                        viscosity_weights=viscosity_weights, basis_cache=endpoint_basis_cache,
                        **({} if numerical_policy is None else dict(numerical_policy=numerical_policy,
                            quality_minimum=quality_floor,gravity_area_targeting=True)))
                except (SpeedBoundError, InverseSolveError, ActiveSetBudgetError):
                    # Not retryable: the speed bound is an assertion about the
                    # candidate, not a property of the interval. Shortening the step
                    # would quietly bring it into range and hide the cap, which
                    # test_constrained_speed_cap_is_never_hidden_by_postsolve_scaling
                    # exists to forbid. Let it reach the caller.
                    #
                    # An unconverged inverse is a property of the current geometry.
                    # Every shorter trial repeats the identical solve: at 122 Myr
                    # this loop spent 32 trials on it and then blamed the timestep.
                    #
                    # Draft classification for identical-input gravity trials:
                    # an entry-area violation count is independent of this trial
                    # interval. This catch does not handle the separate contact
                    # accommodation/adaptive-timestep path of the observed crash.
                    # Motion-driven capacity exhaustion still retries below.
                    raise
                except ConstraintSolveError as error:
                    rejection = f'constrained endpoint: {error}'
                    # A constrained endpoint that does not exist over this interval may
                    # exist over a shorter one: the bounds constrain endpoint AREA, and
                    # halving the interval halves the motion they must accommodate.
                    # Measured on this run's own captured failures, the identical solve
                    # goes from 0 of 4 to 4 of 4 succeeding at half the step.
                    #
                    # This branch previously let the error escape the attempt loop
                    # entirely, so the step-halving below never ran for a constraint
                    # failure and the caller abandoned the whole interval -- which is
                    # why 99% of requested gravity time was skipped for 250 Myr while
                    # isostatic disequilibrium accumulated. Treat it as a failed
                    # attempt, exactly like a rejected geometry or an energy increase.
                    # The existing floor on `step` still ends the search if shortening
                    # stops helping, so a genuinely infeasible interval still raises.
                    step *= .5
                    backtracks += 1
                    continue
                trial_power = float(np.sum(work_gradient*trial_velocity)/radius)
                trial_scale = 1.
                if trial_power >= 0 and not trial_bound.get('stationary_current_geometry', False):
                    raise ValueError('Constrained gravity velocity fails the negative virtual-work gate.')
                new_area = spherical_face_areas(trial, faces, radius)
            signed, quality = _quality(trial, faces)
            area_checks=(np.all(new_area>=minimum*(1-FEASIBILITY_TOLERANCE)),np.all(new_area<=maximum*(1+FEASIBILITY_TOLERANCE)))
            if numerical_policy is not None:area_checks=(numerical_policy.feasible(new_area,minimum,maximum),)*2
            failed = [name for name, ok in (('orientation', np.all(signed > 1e-15)),
                ('shape quality', np.all(quality >= quality_floor)),
                ('minimum column area', area_checks[0]),
                ('maximum column area', area_checks[1])) if not ok]
            if not failed:
                candidate=(None if density_admission is None else density_admission.prepare(trial))
                new_energy = reference_energy(trial, faces, volumes, reference, sheets, radius=radius,
                    density_profile=density_profile if candidate is None else candidate[1],
                    overlap=None if candidate is None else candidate[2])
                new_column=new_energy
                new_entry=0.
                if entry_potential is not None:
                    entry_potential.validate_density(density_profile if candidate is None else candidate[1])
                    try:
                        trial_entry=entry_potential.evaluate(trial,faces,volumes,sheets,radius=radius)
                    except EntryGeometryError as error:
                        rejection=f'entry geometry: {error}'
                        step*=.5
                        backtracks+=1
                        continue
                    new_entry=trial_entry['energy_j']
                    new_energy+=new_entry/ENERGY_UNIT_J
                evaluations += 1
                if new_energy <= energy+1e-4*step*trial_power+roundoff:
                    velocity, solver, power = trial_velocity, trial_solver, trial_power
                    scale = trial_scale
                    bound_diagnostic = trial_bound
                    accepted = True
                    break
                rejection = (f'energy change {new_energy-energy:.6g} km4 exceeds its allowance '
                             f'{1e-4*step*trial_power+roundoff:.6g} km4')
            else:
                rejection = 'endpoint fails its ' + ', '.join(failed) + ' bound'
            step *= .5
            backtracks += 1
        if not accepted or step < max(dt, 1.)*1e-12:
            if accepted:
                rejection = f'the accepted {step:.3g} Myr trial is below the resolvable step'
            raise ValueError('Gravity could not advance within its exact energy/geometry/column bounds '
                f'after {attempt+1} trials from {remaining:.6g} Myr; last rejection: {rejection}.')
        points = trial
        if density_admission is not None:density_admission.accept(candidate)
        elapsed += step
        last_energy = new_energy
        last_column=new_column
        last_entry=new_entry
        if bound_diagnostic is not None and bound_diagnostic.get('stationary_current_geometry', False):
            rows.append(dict(dt_myr=step, stationary_within_energy_roundoff=True,
                stationary_within_constraint_kkt=True, energy_over_density_coefficient_km4=energy,
                virtual_power_km4_myr=power, energy_roundoff_km4=roundoff,
                energy_roundoff_scale_km4=energy_scale, solver=solver, gradient=geometry,
                bound_constraints=bound_diagnostic))
        else:
            rows.append(dict(dt_myr=step, energy_before_km4=energy, energy_after_km4=new_energy,
                virtual_power_km4_myr=power, velocity_safety_scale=scale,
                energy_roundoff_km4=roundoff, energy_roundoff_scale_km4=energy_scale,
                maximum_speed_km_myr=float(np.linalg.norm(velocity, axis=1).max(initial=0.)),
                backtracks=attempt, solver=solver, gradient=geometry))
            if bound_diagnostic is not None:
                rows[-1]['bound_constraints'] = bound_diagnostic
        if numerical_policy is not None:
            rows[-1]['numerical_accuracy_policy']=numerical_policy.evidence(new_area,minimum,maximum,include_arrays=False)
            numerical_policy=numerical_policy.advance(new_area)
    if elapsed < dt*(1-1e-12):
        raise ValueError('Gravity exhausted its internal step budget before the full physical interval was covered.')
    final_area = spherical_face_areas(points, faces, radius)
    return points, dict(version=1, model='conserved-column balanced-reference gravitational relaxation',
        requested_dt_myr=float(dt), completed_dt_myr=float(elapsed), accepted_time_fraction=1.,
        energy_before_km4=first_energy, energy_after_km4=last_energy,
        gravitational_energy_change_j=(last_energy-first_energy)*2800.*9.81*(1-2800./3300.)*1e12,
        substeps=len(rows), backtracks=backtracks, energy_geometry_evaluations=evaluations,
        # Surfaced so the headroom against the active-set budget is visible in
        # saved history. Nothing recorded it before, so the load that stopped
        # run 20260917-0012-nonpenetration at 160 Myr was invisible until it did.
        maximum_active_bounds=max((row.get('bound_constraints', {}).get('maximum_active_bounds', 0)
                                   for row in rows), default=0),
        active_set_budget=max((row.get('bound_constraints', {}).get('active_set_budget', 0)
                               for row in rows), default=0),
        maximum_displacement_km=float(np.linalg.norm(points-initial, axis=1).max(initial=0.)*radius),
        maximum_column_volume_residual_km3=float(np.abs(final_area*(volumes/final_area)-volumes).max(initial=0.)),
        viscosity_pa_s=viscosity_pa_s, lithosphere_thickness_km=lithosphere_thickness_km,
        length_km=length_km, internal_steps=rows,
        material_viscosity_weight_range=(None if viscosity_weights is None else
            [float(np.min(viscosity_weights)),float(np.max(viscosity_weights))]),
        limitations='reduced material-weighted sheet viscosity and uniform basal drag; balanced reference traction; no resolved mantle/ocean column or lower-crustal channel',
        **({} if entry_potential is None else dict(
            entry_energy_before_j=first_entry,entry_energy_after_j=last_entry,
            column_energy_before_km4=first_column,column_energy_after_km4=last_column,
            energy_scope='column plus entry potential; frozen hinge, topology, composition and volume',
            entry_scope='explicit residual sheet stage; native entry transport and polarity remain incomplete')),
        **({'gravity_constraint_version': 1} if constraint_version else {}),
        **({} if numerical_policy is None else dict(numerical_accuracy_version=1,
            numerical_accuracy_final_spent_km2=numerical_policy.spent_km2.tolist(),
            numerical_accuracy_policy=numerical_policy.evidence(final_area,minimum,maximum,include_arrays=False))),
        **({} if vertex_owner is None else dict(
            rigid_mode_partition_version=RIGID_PARTITION_VERSION,
            rigid_mode_partition=first_partition,
            maximum_rigid_gradient_fraction=float(rigid_fraction),
            rigid_mode_scope='per-owner rigid share of this potential is applied by the force balance, not here')))
