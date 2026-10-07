"""A bounded kinematic closure for retreating subduction hinges.

This extends overriding oceanic lithosphere into the downgoing side. It does
not add buoyant crust or solve slab dynamics. Membership is shifted on the
sphere, including residual donor membership behind a moving interface, rather
than flipping one row of categorical cells at a time.
"""
import numpy as np
import trench_history
import normal_partition

RADIUS_KM = 6371.


def retreat_speed(convergence, ocean_age, craton_fraction, *, slab_buoyancy=None):
    """Heuristic km/Myr relative to the overriding plate, capped at 8."""
    buoyancy=(trench_history.ocean_buoyancy(np.maximum(ocean_age, 0.))
              if slab_buoyancy is None else np.asarray(slab_buoyancy,float))
    if not np.isfinite(buoyancy).all() or np.any(buoyancy<0):
        raise ValueError('Retreat requires a finite nonnegative slab buoyancy proxy.')
    return np.minimum(8., .15*np.maximum(convergence, 0.)
               * np.minimum(buoyancy, 1.2)
                      / (1.+2.5*np.clip(craton_fraction, 0., 1.)))


def extend_forearcs(simulation, dt, *, chunk_size=16384, exclude_pairs=()):
    """Return realized transfers by (down, over), updating ocean support only.

    The original fields are frozen until all proposals have been assembled.
    A physical 75-km gradient stencil reaches beyond a narrow polar longitude
    cell; the inverse spherical lookup can cross any number of those cells.
    Third-plate junctions and buoyant footprints are excluded. Per-donor
    normalization makes competing proposals independent of processing order.
    The returned area is a membership transfer, not unique consumed rock.
    """
    s = simulation
    sub = np.flatnonzero(normal_partition.subduction(s))
    if not len(sub):
        return {}
    down = s.down[sub]
    over = np.where(down == s.bp[sub], s.bq[sub], s.bp[sub])
    pairs = np.unique(np.column_stack((down, over)), axis=0)
    area = np.bincount(s.plate, weights=s.cell_area, minlength=s.capacity)
    cratons = np.bincount(s.parcel_plate[s.kind == 2], weights=s.mass[s.kind == 2], minlength=s.capacity)
    fractions = cratons/np.maximum(area, 1.)
    # A zero-density test protects even juvenile material below map visibility.
    unoccupied = (s.crust == 0) & (s.density < 1e-10)
    cells_list, donor_list, receiver_list, amount_list = [], [], [], []
    angle = 75./RADIUS_KM
    ca, sa = np.cos(angle), np.sin(angle)
    for p, q in pairs:
        if (int(p), int(q)) in exclude_pairs:
            continue
        # Only this local, currently convergent trench can drive retreat. A
        # mature margin must not activate every remote contact of the pair.
        pair_support = s.support[p]+s.support[q]
        candidates = np.flatnonzero(unoccupied & ((s.plate == p) | (s.plate == q))
                                   & (s.support[p] > 1e-7)
                                   & (pair_support > .98))
        lifecycle = None
        if hasattr(s, 'trench_systems'):
            lifecycle = np.zeros(s.n)
            lifecycle[candidates] = trench_history.retreat_weight(s, int(p), int(q), candidates)
        if lifecycle is not None:
            candidates = candidates[lifecycle[candidates] > 0]
        for start in range(0, len(candidates), chunk_size):
            cells = candidates[start:start+chunk_size]
            xyz = s.xyz[cells]
            lon = np.arctan2(xyz[:, 1], xyz[:, 0])
            east = np.column_stack((-np.sin(lon), np.cos(lon), np.zeros(len(cells))))
            north = np.cross(xyz, east)
            contrasts, pair_coverage = [], []
            for tangent in (east, north):
                samples = []
                for sign in (1., -1.):
                    lookup = s._sample_coordinates(ca*xyz+sign*sa*tangent)
                    sample_q = s._sample(s.support[q], lookup)
                    sample_p = s._sample(s.support[p], lookup)
                    samples.append(sample_q)
                    pair_coverage.append(sample_q+sample_p)
                contrasts.append(samples[0]-samples[1])
            norm = np.hypot(*contrasts)
            valid = (norm > 1e-7) & (np.min(pair_coverage, axis=0) > .98)
            if not np.any(valid):
                continue
            cells, xyz = cells[valid], xyz[valid]
            normal = (east[valid]*contrasts[0][valid, None]+north[valid]*contrasts[1][valid, None])/norm[valid, None]
            velocity = np.cross(s.omega[p]-s.omega[q], xyz)*RADIUS_KM
            convergence = np.sum(velocity*normal, axis=1)
            shear = np.linalg.norm(velocity-convergence[:, None]*normal, axis=1)
            converging = (convergence > 0. if normal_partition.enabled(s)
                          else convergence > np.maximum(2., .35*shear))
            retained=trench_history.slab_memory.enabled(s)
            speed = retreat_speed(convergence, s.age[cells], fractions[q],
                slab_buoyancy=lifecycle[cells] if retained else None)*converging
            if lifecycle is not None and not retained:
                speed *= lifecycle[cells]
            distance = speed*dt/RADIUS_KM
            origin = np.cos(distance)[:, None]*xyz+np.sin(distance)[:, None]*normal
            lookup = s._sample_coordinates(origin)
            shifted = s._sample(s.support[q], lookup)
            # The native sampler reconstructs a continuous field from adjacent
            # cell values. Its value at an undisplaced cell centre is not the
            # raw cell value. Subtract the SAME reconstruction at zero travel,
            # or a dt->0 call transfers finite membership across the trench.
            stationary = s._sample(s.support[q], s._sample_coordinates(xyz))
            gain = np.minimum(s.support[p, cells], np.maximum(shifted-stationary, 0.))
            gain[s._sample(pair_support, lookup) < .98] = 0.
            gain[~converging] = 0.
            # Resolve changes relative to stored membership precision, not a
            # fixed cutoff that erases real flux at short source timesteps.
            resolution = np.maximum(np.spacing(s.support[p, cells]),
                                    np.spacing(s.support[q, cells]))
            keep = gain > resolution
            if np.any(keep):
                cells_list.append(cells[keep])
                donor_list.append(np.full(np.count_nonzero(keep), p, np.int32))
                receiver_list.append(np.full(np.count_nonzero(keep), q, np.int32))
                amount_list.append(gain[keep])
    if not cells_list:
        return {}
    cells, donors, receivers, amounts = map(np.concatenate, (cells_list, donor_list, receiver_list, amount_list))
    keys, inverse = np.unique(donors.astype(np.int64)*s.n+cells, return_inverse=True)
    requested = np.bincount(inverse, weights=amounts)
    available = s.support[keys//s.n, keys % s.n].astype(float)
    amounts *= np.minimum(1., available/np.maximum(requested, 1e-30))[inverse]
    np.add.at(s.support, (donors, cells), -amounts)
    np.add.at(s.support, (receivers, cells), amounts)
    # Roundoff in float32 support cannot represent negative memberships.
    s.support[:, np.unique(cells)] = np.maximum(s.support[:, np.unique(cells)], 0.)
    touched = np.unique(cells)
    s.support[:, touched] /= s.support[:, touched].sum(axis=0)
    s.plate[touched] = np.argmax(s.support[:, touched], axis=0)
    result = {}
    for p, q in pairs:
        select = (donors == p) & (receivers == q)
        result[int(p), int(q)] = dict(cells=cells[select], gain=amounts[select],
                                      area_km2=float(np.sum(amounts[select]*s.cell_area[cells[select]])))
    return result
