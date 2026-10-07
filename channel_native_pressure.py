"""Read-only ordered lithostatic load from measured native upper rock.

An external contact/vertical solver still must supply every segment's actual
top and base. This module refuses caller-supplied rock mass and checks that
each rock interval has the measured regional column's physical thickness.
It does not invent basal geometry, mantle gaps, temperature, or work.
"""

import numpy as np

from entry_channel_pressure import ordered_top_pressure


def pressure_from_measured_rock(zone_rock, lower_top_depth_m, segments):
    """Apply measured ordered upper masses to an explicit vertical path."""
    if (not isinstance(zone_rock, dict)
            or not isinstance(zone_rock.get('rock_layers_top_to_bottom'), list)
            or not isinstance(segments, (list, tuple))):
        raise ValueError('Native contact pressure needs measured rock and vertical segments.')
    layers = zone_rock['rock_layers_top_to_bottom']
    rock_segments = [segment for segment in segments
                     if isinstance(segment, dict) and segment.get('kind') == 'rock']
    if len(rock_segments) != len(layers):
        raise ValueError('Native contact pressure needs one segment per upper rock layer.')
    measured = []
    rock_index = 0
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError('Native contact pressure needs explicit vertical segments.')
        if segment.get('kind') != 'rock':
            measured.append(dict(segment))
            continue
        layer = layers[rock_index]
        rock_index += 1
        if (set(segment) != {'kind', 'top_depth_m', 'base_depth_m', 'sheet_id'}
                or int(segment['sheet_id']) != int(layer['sheet_id'])):
            raise ValueError('Native rock segment disagrees with persistent sheet order.')
        span = float(segment['base_depth_m']) - float(segment['top_depth_m'])
        physical = 1000. * float(layer['physical_thickness_km'])
        mass = float(layer['rock_mass_kg_m2'])
        if (not np.isfinite(span) or not np.isfinite(physical)
                or not np.isfinite(mass) or physical <= 0. or mass <= 0.
                or not np.isclose(span, physical, rtol=2e-10, atol=1e-6)):
            raise ValueError('Native rock interval must match measured physical thickness.')
        measured.append(dict(segment, mass_kg_m2=mass))
    order = tuple(int(layer['sheet_id']) for layer in layers)
    result = ordered_top_pressure(lower_top_depth_m, measured, order)
    return dict(result, zone_id=zone_rock.get('zone_id'),
                validated_segments=tuple(measured),
                scope='read-only measured rock plus explicit vertical path; no native source commit')
