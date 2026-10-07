"""Optional spherical starting worlds for the tectonic worldbuilding model."""
import numpy as np

from fracture import component_labels


def _coordinates(width, height):
    lon = (np.arange(width)+.5)*2*np.pi/width-np.pi
    lat = np.pi/2-(np.arange(height)+.5)*np.pi/height
    xyz = np.empty((height, width, 3), float)
    xyz[:, :, 0] = np.cos(lat)[:, None]*np.cos(lon)[None, :]
    xyz[:, :, 1] = np.cos(lat)[:, None]*np.sin(lon)[None, :]
    xyz[:, :, 2] = np.sin(lat)[:, None]
    edges = np.pi/2-np.arange(height+1)*np.pi/height
    # Fractions of the entire spherical area, not counts of rectangular pixels.
    area = np.repeat((np.sin(edges[:-1])-np.sin(edges[1:]))/(2*width), width)
    return xyz.reshape(-1, 3), area


def _weighted_count(order, area, target):
    """Nearest whole-cell prefix to a target fraction of spherical area."""
    cumulative = np.cumsum(np.asarray(area)[order])
    crossing = int(np.searchsorted(cumulative, target))
    if crossing >= len(order):
        raise ValueError('Starting-world target exceeds its available spherical region.')
    count = crossing+1
    if crossing > 0 and abs(cumulative[crossing-1]-target) < abs(cumulative[crossing]-target):
        count = crossing
    return max(1, count)


def _random_rotation(rng):
    """Seeded proper rotation without QR sign conventions."""
    u1, u2, u3 = rng.random(3)
    x = np.sqrt(1-u1)*np.sin(2*np.pi*u2)
    y = np.sqrt(1-u1)*np.cos(2*np.pi*u2)
    z = np.sqrt(u1)*np.sin(2*np.pi*u3)
    w = np.sqrt(u1)*np.cos(2*np.pi*u3)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def make_highland65(config, *, continental_fraction=.65):
    """Six separated continents covering 65% of the spherical area.

    The six source domains are a randomly rotated octahedral partition. Each
    continent grows as a lobed radial body inside one domain, while a finite
    collar around every domain edge is reserved for ocean. This guarantees a
    connected global ocean reservoir instead of beginning the high-land world
    as one nearly closed supercontinent.

    The 65% target is continental/cratonic material coverage, not reconstructed
    emergent area after shelves, basin subsidence, erosion and sea-level effects.
    Three separated cratonic shields are placed inside each continent.
    An explicit ``continental_fraction`` regenerates this same seeded design
    at lower coverage; it leaves passive-margin and physics metadata unchanged.
    """
    w, h = config['width'], config['height']
    if (isinstance(continental_fraction, (bool, np.bool_)) or
            not np.isfinite(continental_fraction) or not .3 <= continental_fraction <= .65):
        raise ValueError('Highland continental coverage must be between 30% and 65%.')
    target = float(continental_fraction)
    seed = int(config['seed'])
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x48494748]))
    xyz, area = _coordinates(w, h)
    count = len(xyz)

    # Six equal spherical neighborhoods leave enough room to assign about
    # 10-11.5% of the planet to each continent while retaining broad water
    # corridors between them. The 0.14 dot-product collar is deliberately
    # physical rather than cell-sized, so the topology survives coarse grids.
    source = np.array([[1., 0., 0.], [-1., 0., 0.],
                       [0., 1., 0.], [0., -1., 0.],
                       [0., 0., 1.], [0., 0., -1.]])
    centers = source @ _random_rotation(rng).T
    region = np.empty(count, np.int8)
    separation = np.empty(count)
    for start in range(0, count, 65_536):
        section = slice(start, min(start+65_536, count))
        dots = xyz[section] @ centers.T
        best = np.argmax(dots, axis=1)
        first = dots[np.arange(len(dots)), best]
        dots[np.arange(len(dots)), best] = -np.inf
        region[section] = best
        separation[section] = first-np.max(dots, axis=1)

    shares = rng.permutation(np.array([.115, .1125, .110, .1075, .105, .100]))
    shares = shares*(target/.65)
    phases = rng.uniform(-np.pi, np.pi, (6, 3))
    composite = np.full(count, np.inf)
    region_scores = [None]*6
    for index, center in enumerate(centers):
        reference = np.array([0., 0., 1.]) if abs(center[2]) < .85 else np.array([1., 0., 0.])
        east = np.cross(reference, center)
        east /= np.linalg.norm(east)
        north = np.cross(center, east)
        cells = np.flatnonzero((region == index) & (separation >= .14))
        points = xyz[cells]
        radial = np.arccos(np.clip(points @ center, -1, 1))
        azimuth = np.arctan2(points @ north, points @ east)
        score = radial-(.10*np.sin(3*azimuth+phases[index, 0])
                        +.06*np.sin(5*azimuth+phases[index, 1])
                        +.025*np.sin(8*azimuth+phases[index, 2]))
        order = np.argsort(score, kind='stable')
        local_count = _weighted_count(order, area[cells], shares[index])
        threshold = float(score[order[local_count-1]])
        composite[cells] = score-threshold
        region_scores[index] = (cells, score)

    candidates = np.flatnonzero(np.isfinite(composite))
    order = candidates[np.argsort(composite[candidates], kind='stable')]
    selected = _weighted_count(order, area, target)
    crust = np.zeros(count, np.uint8)
    crust[order[:selected]] = 1

    continent_labels, continent_count = component_labels(crust > 0, w, h)
    ocean_labels, ocean_count = component_labels(crust == 0, w, h)
    if continent_count != 6 or ocean_count != 1:
        raise ValueError('Highland65 topology did not resolve six continents and one connected ocean.')

    # Craton placement has its own deterministic stream so tuning coastline
    # harmonics later does not silently move all shields. Three far-separated
    # centers are selected from the inner 55% of each continent by area.
    craton_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x43524154]))
    for index, center in enumerate(centers):
        cells, score = region_scores[index]
        land = cells[crust[cells] > 0]
        values = composite[land]
        ranked = land[np.argsort(values, kind='stable')]
        cumulative = np.cumsum(area[ranked])
        interior_count = int(np.searchsorted(cumulative, .55*area[land].sum()))+1
        candidates = ranked[:max(3, interior_count)]
        first = int(np.argmax(xyz[candidates] @ center))
        chosen = [first]
        nearest = 1-xyz[candidates] @ xyz[candidates[first]]
        for _ in range(2):
            next_candidate = int(np.argmax(nearest))
            chosen.append(next_candidate)
            nearest = np.minimum(nearest, 1-xyz[candidates] @ xyz[candidates[next_candidate]])
        shield_centers = xyz[candidates[chosen]]
        spacing = np.arccos(np.clip(shield_centers @ shield_centers.T, -1, 1))
        np.fill_diagonal(spacing, np.pi)
        radii = np.minimum(craton_rng.uniform(.10, .14, 3), .25*spacing.min(axis=1))
        for point, radius in zip(shield_centers, radii):
            for start in range(0, count, 65_536):
                section = slice(start, min(start+65_536, count))
                inside = ((xyz[section] @ point >= np.cos(radius))
                          & (crust[section] > 0) & (region[section] == index))
                crust[section][inside] = 2

    _, craton_count = component_labels(crust == 2, w, h)
    if craton_count != 18:
        raise ValueError('Highland65 craton topology did not resolve eighteen separate shields.')

    continent_area = np.bincount(continent_labels[continent_labels >= 0],
                                 weights=area[continent_labels >= 0], minlength=continent_count)
    ocean_area = np.bincount(ocean_labels[ocean_labels >= 0],
                             weights=area[ocean_labels >= 0], minlength=ocean_count)
    continental_fraction = float(area[crust > 0].sum())
    return dict(width=w, height=h, crust=crust, preset=f'highland{target*100:g}',
                land_fraction=continental_fraction,
                continental_fraction=continental_fraction,
                craton_fraction=float(area[crust == 2].sum()),
                continent_count=int(continent_count), craton_count=int(craton_count),
                ocean_component_count=int(ocean_count),
                largest_continent_fraction=float(continent_area.max()),
                largest_ocean_fraction=float(ocean_area.max()),
                target_land_fraction=target, target_continental_fraction=target,
                initial_plate_topology=dict(version=1, mode='passive_margin_aprons',
                    seed=seed, passive_coast_fraction=.65, apron_width_km=650.,
                    max_attached_ocean_fraction=.45),
                initial_subduction=dict(enabled=True, initial_slab_depth_km=100.,
                    target_margin_fraction=.45, selection_seed=seed),
                continental_lifecycle=dict(version=1, enabled=True,
                    neck_viscosity_pa_s=1e23, neck_thickness_km=80.,
                    neck_length_km=100., failure_opening_km=80.,
                    support_radius_km=1500., max_source_step_myr=.25,
                    replacement_delay_myr=20., replacement_exclusion_km=600.),
                rift_traction=dict(version=1, enabled=True,
                    reference_strength_n_per_m=1e13, mobility_km_myr=12.,
                    reach_km=900., max_equivalent_speed_km_myr=40.,
                    boundary_motion_fraction=.25))


def make_land65(config):
    """A connected lobed supercontinent covering 65% of the spherical area.

    Its coastline is a radial angular contour around a seeded spherical center.
    This keeps the continent connected and the score continuous across the map
    seam and both poles. Eighteen separate angular caps represent strong cratons.
    Generation is independent of tectonic forces and uses no full-grid pairwise
    distances or resolution-sized stack of craton masks.
    """
    w, h = config['width'], config['height']
    rng = np.random.default_rng(config['seed'])
    lon, lat = rng.uniform(-.3, .3), rng.uniform(-.08, .08)
    center = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
    east = np.array([-np.sin(lon), np.cos(lon), 0.])
    north = np.cross(center, east)
    phases = rng.uniform(-np.pi, np.pi, 3)

    def score(points):
        radial = np.arccos(np.clip(points @ center, -1, 1))
        azimuth = np.arctan2(points @ north, points @ east)
        lobes = (.15*np.sin(3*azimuth+phases[0])+
                 .10*np.sin(5*azimuth+phases[1])+.045*np.sin(9*azimuth+phases[2]))
        return radial-lobes

    xyz, area = _coordinates(w, h)
    field = np.empty(w*h)
    for start in range(0, w*h, 65_536):
        field[start:start+65_536] = score(xyz[start:start+65_536])
    order = np.argsort(field, kind='stable')
    cumulative = np.cumsum(area[order])
    crossing = int(np.searchsorted(cumulative, .65))
    count = crossing+1
    if crossing > 0 and abs(cumulative[crossing-1]-.65) < abs(cumulative[crossing]-.65):
        count = crossing
    crust = np.zeros(w*h, np.uint8)
    crust[order[:count]] = 1
    threshold = float(field[order[count-1]])
    del order, cumulative

    # A nearly uniform sphere sample avoids clustering candidates at the poles.
    # Snap to real cell centers so even coarse grids retain a connected core.
    samples = 2048
    z = 1-2*(np.arange(samples)+.5)/samples
    phi = np.arange(samples)*np.pi*(3-np.sqrt(5))
    candidate_xyz = np.column_stack((np.sqrt(1-z*z)*np.cos(phi),
                                     np.sqrt(1-z*z)*np.sin(phi), z))
    x = np.floor((np.arctan2(candidate_xyz[:, 1], candidate_xyz[:, 0])+np.pi)*w/(2*np.pi)).astype(int) % w
    y = np.clip(np.floor((np.pi/2-np.arcsin(candidate_xyz[:, 2]))*h/np.pi).astype(int), 0, h-1)
    candidate_cells = np.unique(y*w+x)
    # Keep shields well inside the supercontinent, away from its narrow margins.
    candidate_cells = candidate_cells[field[candidate_cells] < threshold-.38]
    candidates = xyz[candidate_cells]
    first = int(np.argmax(candidates @ center))
    chosen = [first]
    nearest = 1-candidates @ candidates[first]
    for _ in range(17):
        next_candidate = int(np.argmax(nearest))
        chosen.append(next_candidate)
        nearest = np.minimum(nearest, 1-candidates @ candidates[next_candidate])
    centers = candidates[chosen]
    separation = np.arccos(np.clip(centers @ centers.T, -1, 1))
    np.fill_diagonal(separation, np.pi)
    radii = np.minimum(rng.uniform(.14, .185, len(centers)), .32*separation.min(axis=1))
    for point, radius in zip(centers, radii):
        for start in range(0, w*h, 65_536):
            section = slice(start, start+65_536)
            inside = (xyz[section] @ point >= np.cos(radius)) & (crust[section] > 0)
            crust[section][inside] = 2
    _, craton_count = component_labels(crust == 2, w, h)
    return dict(width=w, height=h, crust=crust, preset='land65',
                land_fraction=float(area[crust > 0].sum()),
                craton_fraction=float(area[crust == 2].sum()),
                craton_count=int(craton_count), target_land_fraction=.65)
