"""Read-only energy ledger for replacing old entry work with contact work.

The current native entry potential uses an affine nodal source coordinate.
Projected physical contact uses a shorter horizontal footprint. Before a
native source transaction can subtract the former and add a nonlocal contact
law, their common preferred-path buoyancy work must agree within an explicit
geometry/quadrature error budget.
"""

import math

import numpy as np

from channel_contact_union_loads import lump_projected_contact_loads
from channel_native_lower_inventory import bind_uniform_lower_inventory
from channel_native_projected_ray import current_native_projected_loads
from entry_projected_loads import uniform_contact_entry_work


def audit_uniform_entry_handoff(s, prepared, offsets_m, *,
                                work_tolerance=1e-4,
                                force_tolerance=1e-4):
    """Compare old affine entry work with projected full-contact ray work.

    Only one uniform native lower phase can use the exact old-entry reference
    integral here. Mixed regional phases require a separate source integral
    before the old potential can be removed locally.
    """
    if not isinstance(prepared, dict):
        raise ValueError('Entry handoff needs a prepared native projected pair.')
    tolerance = float(work_tolerance)
    force_limit = float(force_tolerance)
    if (not math.isfinite(tolerance) or not 0. < tolerance < 1.
            or not math.isfinite(force_limit) or not 0. < force_limit < 1.):
        raise ValueError('Entry handoff needs bounded work and force tolerances.')
    inventory = bind_uniform_lower_inventory(s, prepared['lower_face_id'])
    current = current_native_projected_loads(s, prepared)
    geometry = prepared['projected_pair']
    if (geometry.get('coordinate_discretization') != 'affine_nodal'
            or not math.isclose(geometry['face_reference_area_km2'],
                                inventory['physical_area_km2'], rel_tol=2e-10)):
        raise ValueError('Entry handoff needs the current affine source face.')
    old = uniform_contact_entry_work(
        inventory['lower_triangle'], prepared['hinge_normal'],
        prepared['dip_degrees'], inventory['physical_volume_km3'],
        inventory['crust_mass_kg'], geometry,
        radius_km=inventory['radius_km'])
    union = prepared['line_union']
    loads = lump_projected_contact_loads(
        union, offsets_m,
        {str(cell['cell_index']): cell for cell in current['cell_loads']})
    offsets = np.asarray(offsets_m, float)
    preferred = offsets * math.tan(math.radians(prepared['dip_degrees']))
    represented = float(union['line_weight_m']
                        @ (loads['nodal_force_n_per_m'] @ preferred))
    baseline = float(old['contact_entry_work_j'])
    scale = max(abs(baseline), abs(represented), 1.)
    mismatch = (represented - baseline) / scale
    old_force = float(old['contact_buoyancy_force_n'])
    projected_force = float(loads['source_cell_force_n'])
    force_difference = (projected_force - old_force) / max(
        abs(old_force), abs(projected_force), 1.)
    if abs(mismatch) > tolerance:
        raise ValueError('Entry handoff old and projected work exceed their error budget.')
    if abs(force_difference) > force_limit:
        raise ValueError('Entry handoff old and projected forces exceed their error budget.')
    return dict(old_entry_work_to_remove_j=baseline,
                represented_preferred_path_work_j=represented,
                work_difference_j=represented - baseline,
                relative_work_difference=mismatch,
                old_entry_contact_force_n=old_force,
                projected_contact_force_n=projected_force,
                relative_force_difference=force_difference,
                scope='read-only uniform entry/contact handoff audit; no source or plate commit')
