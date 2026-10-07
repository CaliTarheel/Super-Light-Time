"""Read-only native material binding for one projected contact hinge ray.

The lower regional inventory and overriding basal reference come from current
material faces. Projected contact geometry, conservative nodal load and a
nonlocal two-plate solve share one signed hinge. Stiffness remains supplied by
the caller; this module makes no native force, source or owner transaction.
"""

import math

import numpy as np

import channel_entry_zones
import entry_regions
from channel_contact_line_union import spherical_contact_line_union
from channel_contact_union_loads import lump_projected_contact_loads
from channel_contact_union_work import integrate_two_plate_contact_union
from channel_native_regional_contact_loads import bind_regional_projected_loads
from channel_native_upper_strip import bind_trench_normal_upper_strip
from entry_projected_contact import projected_pair_areas


def _current_ordered_zones(s, lower_face_id, upper_face_id, spec):
    records = {int(record['face_id']): record
               for record in s.channel_region_store['records']}
    if int(lower_face_id) not in records or int(upper_face_id) not in records:
        raise ValueError('Native projected pair needs both tracked material histories.')
    current = channel_entry_zones.partition_native_entry_zones(
        s.material_surface, s.parcel_collision_sheet, s.parcel_plate,
        s.plate_uid, s.collision_contacts, spec, list(records),
        epoch_myr=s.t, upper_records_by_face_id=records)
    zones = [zone for zone in current['entry_faces'][int(lower_face_id)]['zones']
             if zone['kind'] == 'entered-contact'
             and zone['adjacent_upper_face_id'] == int(upper_face_id)
             and len(zone['covering_sheets_top_to_bottom']) == 1]
    if not zones:
        raise ValueError('Native projected pair needs current ordered upper contact.')
    return zones


def prepare_native_projected_pair(s, lower_face_id, upper_face_id, *,
                                  refinement=1, area_tolerance=1e-4):
    """Bind current ordered native identities to projected physical contact."""
    if (not isinstance(lower_face_id, (int, np.integer))
            or not isinstance(upper_face_id, (int, np.integer))
            or isinstance(lower_face_id, (bool, np.bool_))
            or isinstance(upper_face_id, (bool, np.bool_))
            or int(lower_face_id) == int(upper_face_id)):
        raise ValueError('Native projected pair needs distinct material face IDs.')
    spec = entry_regions.specification(s)
    if spec is None:
        raise ValueError('Native projected pair needs persistent entry material.')
    selected = np.flatnonzero(np.asarray(spec['face_ids']) == int(lower_face_id))
    ids = np.asarray(s.material_surface['face_id'])
    lower_index = np.flatnonzero(ids == int(lower_face_id))
    upper_index = np.flatnonzero(ids == int(upper_face_id))
    if len(selected) != 1 or len(lower_index) != 1 or len(upper_index) != 1:
        raise ValueError('Native projected pair needs identified entry and upper faces.')
    zones = _current_ordered_zones(s, lower_face_id, upper_face_id, spec)
    row = int(selected[0])
    normal = np.asarray(spec['hinge_normals'][row], float)
    dip = float(spec['dip_degrees'][row])
    midpoint = np.asarray(spec['finite_midpoints'][row], float)
    half_length = float(spec['finite_half_lengths_km'][row])
    finite = {} if np.isinf(half_length) else dict(
        finite_midpoint=midpoint, finite_half_length_km=half_length)
    radius = float(s.material_surface['radius_km'])
    vertices = np.asarray(s.material_surface['vertices'])
    faces = np.asarray(s.material_surface['faces'])
    lower = vertices[faces[int(lower_index[0])]]
    upper = vertices[faces[int(upper_index[0])]]
    projected = projected_pair_areas(
        lower, upper, normal, dip, radius_km=radius,
        refinement=refinement, include_cells=True,
        coordinate_discretization='affine_nodal', **finite)
    if not projected['contact_cells']:
        raise ValueError('Native projected pair has no physical contact cells.')
    loads = bind_regional_projected_loads(s, lower_face_id, projected)
    union = spherical_contact_line_union([
        dict(zone_id=str(index), polygon=cell['physical_polygon'])
        for index, cell in enumerate(projected['contact_cells'])],
        normal, radius_km=radius, area_tolerance=area_tolerance)
    return dict(lower_face_id=int(lower_face_id),
                upper_face_id=int(upper_face_id),
                epoch_myr=float(s.t),
                geometry_revision=int(s.material_surface['geometry_revision']),
                hinge_normal=normal.copy(), dip_degrees=dip,
                projected_pair=projected, regional_loads=loads,
                line_union=union,
                native_unprojected_contact_area_km2=sum(
                    float(zone['area_km2']) for zone in zones),
                scope='read-only native identities and projected pair; no solver or source commit')


def validate_native_projected_pair(s, prepared):
    """Reject a prepared pair after its material, hinge, or contact changes."""
    if not isinstance(prepared, dict):
        raise ValueError('Native projected pair needs a prepared material pair.')
    if (prepared['epoch_myr'] != float(s.t)
            or prepared['geometry_revision']
            != int(s.material_surface['geometry_revision'])):
        raise ValueError('Native projected pair material geometry is stale.')
    spec = entry_regions.specification(s)
    if spec is None:
        raise ValueError('Native projected pair lost its persistent entry material.')
    selected = np.flatnonzero(np.asarray(spec['face_ids'])
                              == prepared['lower_face_id'])
    if (len(selected) != 1
            or not np.allclose(spec['hinge_normals'][selected[0]],
                               prepared['hinge_normal'], rtol=0., atol=1e-12)
            or float(spec['dip_degrees'][selected[0]]) != prepared['dip_degrees']):
        raise ValueError('Native projected pair hinge changed after preparation.')
    _current_ordered_zones(s, prepared['lower_face_id'],
                           prepared['upper_face_id'], spec)


def current_native_projected_loads(s, prepared):
    """Rebind phase inventories and reject stale prepared cell loads."""
    validate_native_projected_pair(s, prepared)
    current = bind_regional_projected_loads(
        s, prepared['lower_face_id'], prepared['projected_pair'])
    saved = prepared['regional_loads']['cell_loads']
    cells = current['cell_loads']
    numbers = ('physical_area_km2', 'mean_buoyancy_pa',
               'buoyancy_force_n', 'spherical_material_area_km2',
               'affine_material_area_km2')
    phase_numbers = ('material_area_km2', 'physical_volume_km3',
                     'ordinary_volume_km3', 'dense_volume_km3',
                     'crust_mass_kg', 'buoyancy_force_n')
    def same_numbers(a, b, fields):
        return all(math.isclose(float(a[name]), float(b[name]),
                                rel_tol=2e-12, abs_tol=0.)
                   for name in fields)
    if (len(cells) != len(saved)
            or any(left['cell_index'] != right['cell_index']
                   or not same_numbers(left, right, numbers)
                   or len(left['regional_contributions'])
                   != len(right['regional_contributions'])
                   or any(a['region_id'] != b['region_id']
                          or not same_numbers(a, b, phase_numbers)
                          or not np.array_equal(a['material_polygon'],
                                                b['material_polygon'])
                          for a, b in zip(left['regional_contributions'],
                                          right['regional_contributions']))
                   for left, right in zip(cells, saved))):
        raise ValueError('Native projected pair prepared material loads are stale.')
    return current


def solve_native_projected_ray(s, prepared, ray_index, offsets_m,
                               lower_rigidity_n_m, upper_rigidity_n_m,
                               upper_foundation_n_m3):
    """Solve one physically occupied ray with actual upper basal material."""
    if not isinstance(prepared, dict) or not isinstance(ray_index, (int, np.integer)):
        raise ValueError('Native projected ray needs a prepared pair and ray index.')
    union = prepared['line_union']
    count = len(union['line_weight_m'])
    index = int(ray_index)
    offsets = np.asarray(offsets_m, float)
    if (not 0 <= index < count or offsets.ndim != 1 or len(offsets) < 5
            or not np.isfinite(offsets).all() or abs(offsets[0]) > 1e-6
            or not np.isfinite(prepared['dip_degrees'])
            or offsets[-1] * math.tan(math.radians(prepared['dip_degrees']))
            > 660000. * (1. + 1e-12)):
        raise ValueError('Native projected ray needs a resolved upper-mantle hinge strip.')
    current_loads = current_native_projected_loads(s, prepared)
    loads = lump_projected_contact_loads(
        union, offsets,
        {str(cell['cell_index']): cell
         for cell in current_loads['cell_loads']})
    interval = union['contact_intervals'][index][0]
    upper_ids = np.full(len(offsets), prepared['upper_face_id'], np.int64)
    upper = bind_trench_normal_upper_strip(
        s, upper_ids, prepared['hinge_normal'],
        interval['contact_xyz'], offsets)
    one = dict(line_weight_m=union['line_weight_m'][index:index + 1],
               contact_intervals=[union['contact_intervals'][index]])
    preferred = (offsets * math.tan(math.radians(prepared['dip_degrees'])))[None, :]
    solution = integrate_two_plate_contact_union(
        one, offsets, preferred,
        upper['unloaded_upper_base_depth_m'][None, :],
        loads['buoyancy_pa'][index:index + 1],
        lower_rigidity_n_m, upper_rigidity_n_m,
        upper_foundation_n_m3, free_upper_hinge=True)
    return dict(ray_index=index, upper_material=upper,
                preferred_lower_depth_m=preferred[0],
                source_load_n=float(union['line_weight_m'][index]
                                    * loads['nodal_force_n_per_m'][index].sum()),
                solution=solution,
                scope='read-only one native-bound projected ray; no plate torque or source commit')
