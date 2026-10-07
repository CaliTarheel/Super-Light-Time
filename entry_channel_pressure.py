"""Read-only layered overburden at the top of an underthrust crust column.

The existing entry source adds mantle load along its whole descent; the stack
source adds overlying rock. Inside a channel those terms cannot simply be
summed. This helper separates rock, surface water and the actual mantle gap.
It does not evolve retained phases or heat and is not a native source update.
"""

import numpy as np

import column_density
import crustal_structure
import entry_contact_channel


def layered_top_pressure(lower_top_depth_m, current_upper_base_depth_m,
                         upper_column_mass_kg_m2, *,
                         water_depth_m=0.):
    """Return lithostatic top load for one ordered upper/lower column pair.

    `current_upper_base_depth_m` already includes any upper uplift. The
    mantle-filled gap is only the distance from that base to the incoming
    crust's top. Upper rock mass is physical mass per horizontal area and
    excludes water; water cover is supplied separately. Contact reaction is
    a normal stress, not automatically an added hydrostatic pressure. A full
    thermal and deviatoric stress history still needs a separate source law.
    """
    lower, base, rock, water = np.broadcast_arrays(
        np.asarray(lower_top_depth_m, float),
        np.asarray(current_upper_base_depth_m, float),
        np.asarray(upper_column_mass_kg_m2, float),
        np.asarray(water_depth_m, float))
    if (not all(np.isfinite(value).all()
                for value in (lower, base, rock, water))
            or np.any(lower < 0.) or np.any(base < 0.)
            or np.any(rock < 0.) or np.any(water < 0.)
            or np.any(lower < base - 1e-8)):
        raise ValueError('Layered channel pressure needs nonpenetrating finite depths and nonnegative loads.')
    gap = np.maximum(lower - base, 0.)
    mantle_mass = column_density.RHO_MANTLE * gap
    water_mass = crustal_structure.RHO_WATER * water
    total_mass = rock + water_mass + mantle_mass
    pressure = entry_contact_channel.GRAVITY_M_S2 * total_mass
    return dict(
        pressure_pa=pressure, overburden_mass_kg_m2=total_mass,
        upper_rock_mass_kg_m2=rock, water_mass_kg_m2=water_mass,
        mantle_gap_mass_kg_m2=mantle_mass, mantle_gap_m=gap,
        scope='read-only top overburden; no phase/thermal source transaction',
    )


def ordered_top_pressure(lower_top_depth_m, segments,
                         covering_sheets_top_to_bottom):
    """Hydrostatic load from an explicit, gap-free vertical material column.

    Every segment supplies top/base depths. Rock segments supply their sheet
    identity and measured mass per horizontal area. Water/mantle segments use
    the existing model densities. Segments tile from sea datum, or from an
    exposed first rock top above datum, through the incoming crust top. They
    contain each covering rock sheet exactly once in persistent geometric
    order. No unspecified gap is filled implicitly.
    """
    lower = np.asarray(lower_top_depth_m, float)
    if lower.ndim != 0 or not np.isfinite(lower) or lower < 0.:
        raise ValueError('Ordered channel pressure needs a finite lower top depth.')
    if not isinstance(segments, (list, tuple)) or not isinstance(
            covering_sheets_top_to_bottom, (list, tuple)):
        raise ValueError('Ordered channel pressure needs segments and sheet order.')
    expected = tuple(covering_sheets_top_to_bottom)
    if (any(not isinstance(sheet, (int, np.integer)) for sheet in expected)
            or len(set(expected)) != len(expected)):
        raise ValueError('Ordered channel pressure needs unique persistent sheet IDs.')
    if not segments and lower == 0. and not expected:
        return dict(pressure_pa=0., overburden_mass_kg_m2=0.,
                    upper_rock_mass_kg_m2=0., water_mass_kg_m2=0.,
                    mantle_gap_mass_kg_m2=0., rock_sheet_ids=(),
                    scope='read-only explicit vertical column; no native source commit')
    rock_ids = []
    mass = dict(rock=0., water=0., mantle=0.)
    previous = 0.
    if segments and isinstance(segments[0], dict) and segments[0].get('kind') == 'rock':
        first_top = float(segments[0]['top_depth_m'])
        if first_top < 0.:
            # Exposed rock above the sea datum contributes its full mass.
            # Only an initial rock segment may begin above datum; water and
            # mantle cannot be assigned to that subaerial interval.
            previous = first_top
    for segment in segments:
        if not isinstance(segment, dict) or segment.get('kind') not in mass:
            raise ValueError('Ordered channel pressure needs rock, water or mantle segments.')
        kind = segment['kind']
        required = ({'kind', 'top_depth_m', 'base_depth_m',
                     'sheet_id', 'mass_kg_m2'} if kind == 'rock'
                    else {'kind', 'top_depth_m', 'base_depth_m'})
        if set(segment) != required:
            raise ValueError('Ordered channel segment has missing or extra material fields.')
        top, base = float(segment['top_depth_m']), float(segment['base_depth_m'])
        if (not np.isfinite(top) or not np.isfinite(base)
                or (top < 0. and kind != 'rock') or base <= top
                or abs(top - previous) > 1e-8 or base > lower + 1e-8):
            raise ValueError('Ordered channel segments must tile a nonoverlapping vertical column.')
        if kind == 'rock':
            sheet = segment['sheet_id']
            rock_mass = float(segment['mass_kg_m2'])
            if (not isinstance(sheet, (int, np.integer))
                    or not np.isfinite(rock_mass) or rock_mass <= 0.):
                raise ValueError('Ordered rock needs a sheet ID and positive physical mass.')
            rock_ids.append(int(sheet))
            mass['rock'] += rock_mass
        else:
            density = (crustal_structure.RHO_WATER if kind == 'water'
                       else column_density.RHO_MANTLE)
            mass[kind] += density * (base - top)
        previous = base
    if abs(previous - lower) > 1e-8:
        raise ValueError('Ordered channel segments must tile a nonoverlapping vertical column.')
    if tuple(rock_ids) != expected:
        raise ValueError('Ordered rock sheets do not match the geometric vertical order.')
    total = sum(mass.values())
    return dict(
        pressure_pa=float(entry_contact_channel.GRAVITY_M_S2 * total),
        overburden_mass_kg_m2=float(total),
        upper_rock_mass_kg_m2=float(mass['rock']),
        water_mass_kg_m2=float(mass['water']),
        mantle_gap_mass_kg_m2=float(mass['mantle']),
        rock_sheet_ids=tuple(rock_ids),
        scope='read-only explicit vertical column; no native source commit',
    )
