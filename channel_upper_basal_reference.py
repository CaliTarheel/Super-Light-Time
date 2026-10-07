"""Pre-contact material reference for overriding-crust basal geometry.

This read-only stage records each upper material region's present basal depth
before any geometric overlap. The depth is an incremental unloaded datum for
a future contact solve, not a new isostatic equilibrium or a force law.
"""

from copy import deepcopy

import numpy as np

import crustal_structure
import mesh_coverage


REFERENCE = 'upper_basal_reference'


def inherit_reference(region):
    """Copy the optional material datum without placing it in rock columns."""
    if REFERENCE not in region:
        return {}
    datum = region[REFERENCE]
    if (not isinstance(datum, dict)
            or set(datum) != {'depth_m', 'epoch_myr'}
            or not np.isfinite([datum['depth_m'], datum['epoch_myr']]).all()
            or datum['depth_m'] < 0.):
        raise ValueError('Upper basal reference metadata is invalid.')
    return {REFERENCE: dict(depth_m=float(datum['depth_m']),
                            epoch_myr=float(datum['epoch_myr']))}


def prepare_unloaded_upper_bases(s, face_ids):
    """Copy a fresh native store with material-attached pre-contact bases.

    A selected face must be tracked and have no present overlap with another
    collision sheet. The snapshot is made once; a later contact response must
    compare its current base with this saved value rather than recomputing it.
    """
    import channel_region_native

    if not channel_region_native.enabled(s):
        raise ValueError('Upper basal reference needs an installed native regional store.')
    store = s.channel_region_store
    surface = s.material_surface
    if (store.get('version') != channel_region_native.VERSION
            or store.get('epoch_myr') != float(s.t)
            or store.get('geometry_revision') != int(surface['geometry_revision'])):
        raise ValueError('Upper basal reference needs current material geometry.')
    if (not isinstance(face_ids, (list, tuple, set)) or not face_ids
            or any(not isinstance(face_id, (int, np.integer)) for face_id in face_ids)):
        raise ValueError('Upper basal reference needs distinct tracked face IDs.')
    selected = {int(face_id) for face_id in face_ids}
    if len(selected) != len(face_ids):
        raise ValueError('Upper basal reference needs distinct tracked face IDs.')
    ids = np.asarray(surface['face_id'])
    by_id = {int(face_id): index for index, face_id in enumerate(ids)}
    tracked = {int(record['face_id']) for record in store['records']}
    if not selected <= tracked or not selected <= set(by_id):
        raise ValueError('Upper basal reference needs tracked material faces.')
    sheets = np.asarray(getattr(s, 'parcel_collision_sheet', []))
    if sheets.shape != (len(ids),):
        raise ValueError('Upper basal reference needs aligned collision-sheet identity.')
    overlaps = mesh_coverage.material_overlaps(
        surface['vertices'], surface['faces'], sheets,
        radius_km=float(surface.get('radius_km', 6371.)))
    selected_indices = {by_id[face_id] for face_id in selected}
    if any(int(a) in selected_indices or int(b) in selected_indices
           for a, b in zip(overlaps['first'], overlaps['second'])):
        raise ValueError('Upper basal reference must precede material overlap.')
    staged = deepcopy(store)
    references = {}
    for record in staged['records']:
        face_id = int(record['face_id'])
        if face_id not in selected:
            continue
        local = {}
        for region in record['history']['regions']:
            column = region['column']
            if REFERENCE in region:
                raise ValueError('Upper basal reference cannot be silently rebased.')
            elevation = np.asarray(crustal_structure.elevation(column), float)
            thickness = np.asarray(column['thickness_km'], float)
            if elevation.shape != (1,) or thickness.shape != (1,):
                raise ValueError('Upper basal reference needs one physical column per region.')
            base = float(1000. * thickness[0] - elevation[0])
            if not np.isfinite(base) or base < 0.:
                raise ValueError('Upper basal reference needs a finite submerged rock base.')
            region[REFERENCE] = dict(depth_m=base, epoch_myr=float(s.t))
            local[region['region_id']] = base
        references[face_id] = local
    return dict(store=staged, reference_depth_m_by_face_region=references,
                scope='read-only pre-contact material datum; no native contact solve or commit')


def install_unloaded_upper_bases(s, face_ids):
    """Atomically attach checked pre-contact datums to the native store.

    This is a one-time initialization before overlap, not an accepted native
    timestep. All geometric and material checks finish before the sole write.
    """
    staged = prepare_unloaded_upper_bases(s, face_ids)
    s.channel_region_store = staged['store']
    return dict(reference_depth_m_by_face_region=
                    staged['reference_depth_m_by_face_region'],
                scope='pre-contact native store initialization; no collision mechanics')
