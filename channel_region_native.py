"""Opt-in native storage contract for regional retained-phase source histories.

The native source law does not yet advance these columns. Ordinary timesteps
must reject an installed store. Direct material adaptation may remap its
regions atomically for testing.
"""

from copy import deepcopy

import numpy as np

import dense_crust
import phase_region_projection
import channel_region_history  # Keep the reviewed source law in native run archives.
import channel_pair_zones  # Keep the ordered contact-zone kernel in native archives.
import channel_entry_zones  # Keep native face/contact binding in run archives.
import channel_region_native_source  # Keep source staging in run archives.
import channel_region_native_remap  # Keep material remap in native archives.
import channel_region_native_motion  # Keep material motion in native archives.
import channel_region_native_zones  # Keep contact-zone remap in native archives.
import channel_native_source_preflight  # Keep ordered native source composition in archives.
import channel_native_vertical_path  # Keep current upper-column geometry in archives.
import channel_upper_basal_reference  # Keep material-attached basal datum in archives.
import channel_native_upper_strip  # Keep read-only material strip binding in archives.
import channel_native_contact_strip  # Keep current contact-zone strip binding in archives.
import channel_contact_quadrature  # Keep spherical contact integration sites in archives.
import channel_contact_line_quadrature  # Keep along-trench strip measure in archives.
import channel_contact_line_union  # Keep one strip per hinge position in archives.
import channel_two_plate_flexure  # Keep nonlocal benchmark solver in archives.
import channel_contact_union_work  # Keep shared-strip work oracle in archives.
import channel_contact_union_loads  # Keep conserved projected cell loads in archives.
import channel_native_lower_regions  # Keep regional lower inventories in archives.
import channel_native_lower_inventory  # Keep lower phase inventory binding in archives.
import channel_native_regional_contact_loads  # Keep phase-aware projected loads in archives.
import channel_native_projected_ray  # Keep bounded native-bound contact rays in archives.
import channel_native_entry_handoff  # Keep uniform entry/contact energy audit in archives.
import channel_regional_entry_work  # Keep phase-resolved entry-work audit in archives.
import channel_horizontal_slab_oracle  # Keep audited horizontal-intake alternative in archives.
from channel_marker_binding import bind_marker_regions
from channel_region_geometry import spherical_partition_intersections


VERSION = 1


def enabled(s):
    return hasattr(s, 'channel_region_store')


def _same_column(left, right):
    return (left.keys() == right.keys() and all(
        np.asarray(left[key]).dtype == np.asarray(right[key]).dtype
        and np.array_equal(left[key], right[key]) for key in left))


def coarsening_compatible(s, mapping):
    """Retain every child history only when the native face means can merge."""
    if not enabled(s):
        return True
    records = {int(record['face_id']): record
               for record in s.channel_region_store['records']}
    ids = np.asarray(s.parcel_patch)
    starts = mapping['source_indptr']
    for output in np.flatnonzero(np.diff(starts) > 1):
        sources = mapping['source_indices'][starts[output]:starts[output+1]]
        group = [records.get(int(ids[source])) for source in sources]
        if not any(record is not None for record in group):
            continue
        if any(record is None or not record['history']['regions'] for record in group):
            return False
        def same_native_face(value):
            sample = np.asarray(value)[sources]
            if sample.dtype.kind in 'fc':
                return np.allclose(sample, sample[0], rtol=1e-10, atol=1e-8)
            return np.array_equal(sample, np.broadcast_to(sample[0], sample.shape))
        if any(not same_native_face(value) for value in s.structure.values()):
            return False
    return True


def _project(record, expected_fields):
    regions = record['history']['regions']
    columns = [region['column'] for region in regions]
    if any(set(column) != expected_fields for column in columns):
        raise ValueError('Regional columns must carry every native structure field.')
    expanded = {name: np.concatenate([column[name] for column in columns])
                for name in expected_fields}
    weight = np.asarray([region['fraction'] for region in regions], float)
    return phase_region_projection.coalesce_regions(
        expanded, np.zeros(len(regions), int), weight, 1, measure='physical')


def _reference_mass(record, radius_km):
    """Recover reference area from every retained physical subregion."""
    regions = record['history']['regions']
    geometry = spherical_partition_intersections(
        record['face_triangle'], record['polygons'], record['polygons'],
        radius_km=radius_km)
    fractions = np.asarray([region['fraction'] for region in regions], float)
    if (len(regions) != len(record['polygons']) or not len(regions)
            or not np.isfinite(fractions).all()
            or not np.allclose(fractions, geometry['old_fractions'],
                               rtol=0., atol=2e-10)):
        raise ValueError('Regional reference mass needs a complete material partition.')
    factors = np.asarray([region['column']['area_factor'][0] for region in regions], float)
    if not np.isfinite(factors).all() or np.any(factors <= 0.):
        raise ValueError('Regional reference mass needs positive local area factors.')
    return float(geometry['face_area_km2'] * np.sum(fractions / factors))


def _markers(s, face_ids, *, surface=None, trace_patch=None, records=None,
             trace_xyz=None):
    patches = np.asarray(s.trace_patch if trace_patch is None else trace_patch)
    selected = np.isin(patches, face_ids)
    return bind_marker_regions(
        s.material_surface if surface is None else surface,
        s.channel_region_store['records'] if records is None else records,
        np.asarray(s.trace_id)[selected], patches[selected],
        np.asarray(s.trace_xyz if trace_xyz is None else trace_xyz)[selected])


def _trace_rows(s, bound):
    ids = np.asarray(s.trace_id)
    if (len(np.unique(ids)) != len(ids)
            or set(s.trace_structure) != set(s.structure)
            or any(np.asarray(value).shape != (len(ids),)
                   for value in s.trace_structure.values())):
        raise ValueError('Native trace columns must align with unique trace IDs and face fields.')
    by_id = {int(trace_id): index for index, trace_id in enumerate(ids)}
    return [(by_id[row['trace_id']], row['column']) for row in bound]


def _check_trace_alignment(s, bound):
    for index, column in _trace_rows(s, bound):
        for name in s.structure:
            if not np.allclose(s.trace_structure[name][index], column[name][0],
                               rtol=2e-10, atol=1e-8):
                raise ValueError('Regional source marker history disagrees with native trace state: ' + name)


def install(s, records):
    """Attach checked plain checkpoint data without enabling a source step."""
    if enabled(s):
        raise ValueError('Regional channel state is already installed.')
    if (not dense_crust.present(s.structure)
            or not dense_crust.present(s.trace_structure)
            or not isinstance(records, (list, tuple))
            or not records):
        raise ValueError('Native regional storage requires retained phases and face records.')
    ids = np.asarray(s.material_surface['face_id'])
    by_id = {int(face_id): index for index, face_id in enumerate(ids)}
    if len(by_id) != len(ids):
        raise ValueError('Native material face IDs are not unique.')
    faces = [record['face_id'] for record in records]
    if len(set(faces)) != len(faces) or any(int(face) not in by_id for face in faces):
        raise ValueError('Regional records need distinct existing material faces.')
    fields = set(s.structure)
    radius = float(s.material_surface.get('radius_km', 6371.))
    for record in records:
        index = by_id[int(record['face_id'])]
        if (record['history'].get('face_id') != int(record['face_id'])
                or record['history'].get('face_index') != index
                or not np.array_equal(record['face_triangle'],
                                      s.material_surface['vertices'][s.material_surface['faces'][index]])):
            raise ValueError('Regional store face identity or geometry is stale.')
        reference = _reference_mass(record, radius)
        if (not np.isclose(s.mass[index], reference, rtol=2e-10, atol=1e-6)
                or not np.isclose(s.material_surface['reference_area_km2'][index],
                                  reference, rtol=2e-10, atol=1e-6)):
            raise ValueError('Regional reference mass disagrees with its native face.')
        mean = _project(record, fields)
        for name in fields:
            if not np.allclose(mean[name][0], s.structure[name][index],
                               rtol=2e-10, atol=1e-8):
                raise ValueError('Regional source projection disagrees with native face state: ' + name)
    staged = deepcopy(list(records))
    _check_trace_alignment(s, _markers(s, faces, records=staged))
    s.channel_region_store = dict(version=VERSION, epoch_myr=float(s.t),
                                  geometry_revision=int(s.material_surface['geometry_revision']),
                                  records=staged)


prepare_remap = channel_region_native_remap.prepare_remap


prepare_motion = channel_region_native_motion.prepare_motion


prepare_source = channel_region_native_source.prepare_source


prepare_zone_remap = channel_region_native_zones.prepare_zone_remap
