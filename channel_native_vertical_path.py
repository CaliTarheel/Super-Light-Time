"""Current single-upper native rock geometry above an incoming column.

The upper surface follows its native regional crustal column and its base is
that surface minus the measured physical thickness. The lower top depth is an
external contact state. This snapshot does not establish an unloaded upper
basal reference or solve uplift, bending, temperature, or multiple upper slabs.
"""

import numpy as np


def current_single_upper_path(zone_rock, lower_top_depth_m):
    """Tile sea datum to lower top with current upper rock and any mantle gap."""
    if not isinstance(zone_rock, dict):
        raise ValueError('Native vertical path needs a measured contact zone.')
    layers = zone_rock.get('rock_layers_top_to_bottom')
    lower = float(lower_top_depth_m)
    if (not isinstance(layers, list) or len(layers) > 1
            or not np.isfinite(lower) or lower < 0.):
        raise ValueError('Native vertical path needs one upper sheet and finite lower depth.')
    segments = []
    base = 0.
    if layers:
        layer = layers[0]
        if (not isinstance(layer, dict)
                or not {'sheet_id', 'current_surface_elevation_m',
                        'physical_thickness_km', 'rock_mass_kg_m2'} <= set(layer)):
            raise ValueError('Native vertical path needs the measured current upper column.')
        elevation = float(layer['current_surface_elevation_m'])
        thickness = 1000. * float(layer['physical_thickness_km'])
        if (not np.isfinite(elevation) or not np.isfinite(thickness)
                or thickness <= 0.):
            raise ValueError('Native vertical path needs a finite current upper column.')
        top = -elevation
        base = top + thickness
        if top > 0.:
            segments.append(dict(kind='water', top_depth_m=0., base_depth_m=top))
        if base < 0.:
            raise ValueError('Native upper base lies above sea datum; vertical closure is unresolved.')
        segments.append(dict(kind='rock', sheet_id=int(layer['sheet_id']),
                             top_depth_m=top, base_depth_m=base))
    if base > lower + 1e-6:
        raise ValueError('Native lower top penetrates the current upper rock; contact solve is required.')
    if lower > base:
        segments.append(dict(kind='mantle', top_depth_m=base,
                             base_depth_m=lower))
    return dict(lower_top_depth_m=lower, segments=segments,
                zone_id=zone_rock.get('zone_id'),
                scope='read-only current single-upper geometry; no unloaded reference or contact solve')
