"""Read-only regional pressure and thermal source preflight.

Native geometry, markers and reference-area checks remain in
channel_region_native; this module owns only source staging.
"""

import numpy as np

import channel_region_history
import crust_inventory
import dense_crust


def prepare_source(s, loads_by_face, elapsed_myr, *, constitutive_parameters=None):
    """Stage local phase/thermal sources on every tracked material region.

    The caller supplies one explicit pressure and independent thermal path per
    persistent region. The result includes projected native faces and marker
    columns but is not a timestep commit: moving geometry, reciprocal upper
    work, mantle return and owner transfer still need one atomic native step.
    """
    from channel_region_native import (VERSION, enabled, _markers,
        _check_trace_alignment, _reference_mass, _project, _trace_rows)
    if not enabled(s):
        raise ValueError('Regional source staging needs an installed native store.')
    store = s.channel_region_store
    surface = s.material_surface
    elapsed = float(elapsed_myr)
    if (store.get('version') != VERSION or store.get('epoch_myr') != float(s.t)
            or store.get('geometry_revision') != int(surface['geometry_revision'])
            or not np.isfinite(elapsed) or elapsed <= 0.
            or not isinstance(loads_by_face, dict)):
        raise ValueError('Regional source staging needs a fresh epoch and positive interval.')
    records = store['records']
    face_ids = {int(record['face_id']) for record in records}
    if set(loads_by_face) != face_ids:
        raise ValueError('Regional source staging needs loads for every tracked face.')
    old_bound = _markers(s, sorted(face_ids))
    _check_trace_alignment(s, old_bound)
    by_id = {int(face_id): index for index, face_id in enumerate(surface['face_id'])}
    staged = {name: value.copy() for name, value in s.structure.items()}
    next_records = []
    returned_volume = 0.
    contraction_volume = 0.
    maximum_pressure = 0.
    radius = float(surface.get('radius_km', 6371.))
    controls = (getattr(s, 'retained_phase_parameters', None)
                if constitutive_parameters is None else constitutive_parameters)
    for record in records:
        face_id = int(record['face_id'])
        index = by_id[face_id]
        if (record['history'].get('face_index') != index
                or not np.array_equal(record['face_triangle'],
                                      surface['vertices'][surface['faces'][index]])
                or not np.isclose(_reference_mass(record, radius), s.mass[index],
                                  rtol=2e-10, atol=1e-6)):
            raise ValueError('Regional source geometry or reference mass is stale.')
        report = channel_region_history.advance_region_history(
            record['history'], loads_by_face[face_id], elapsed,
            constitutive_parameters=controls,
            face_area_km2=float(surface['area_km2'][index]))
        next_record = dict(face_id=face_id,
                           face_triangle=record['face_triangle'].copy(),
                           history=report['history'],
                           polygons=[polygon.copy() for polygon in record['polygons']])
        mean = _project(next_record, set(staged))
        for name in staged:
            staged[name][index] = mean[name][0]
        returned_volume += report['returned_volume_km3']
        contraction_volume += surface['area_km2'][index] * report['contraction_km']
        maximum_pressure = max(maximum_pressure,
                               report['maximum_ordinary_pressure_pa'])
        next_records.append(next_record)
    crust_inventory.validate(staged)
    dense_crust.validate(staged)
    indices = np.asarray([by_id[int(record['face_id'])] for record in records], int)
    mass = np.asarray(s.mass)[indices]
    old_mass_equivalent = float(mass @ dense_crust.mass_volume(s.structure)[indices])
    new_mass_equivalent = float(mass @ dense_crust.mass_volume(staged)[indices])
    returned_equivalent = float(mass @ (
        staged[crust_inventory.RETURNED][indices]
        - s.structure[crust_inventory.RETURNED][indices]))
    returned_physical = float(mass @ (
        staged[dense_crust.RETURNED][indices]
        - s.structure[dense_crust.RETURNED][indices]))
    old_volume = float(surface['area_km2'][indices]
                       @ s.structure['thickness_km'][indices])
    new_volume = float(surface['area_km2'][indices]
                       @ staged['thickness_km'][indices])
    if (not np.isclose(old_mass_equivalent,
                       new_mass_equivalent + returned_equivalent,
                       rtol=2e-12, atol=1e-6)
            or not np.isclose(returned_volume, returned_physical,
                              rtol=2e-12, atol=1e-5)
            or not np.isclose(old_volume,
                              new_volume + returned_volume + contraction_volume,
                              rtol=2e-12, atol=1e-5)):
        raise ValueError('Regional source stage did not close mass or column volume.')
    bound = _markers(s, sorted(face_ids), records=next_records)
    if {row['trace_id'] for row in bound} != {row['trace_id'] for row in old_bound}:
        raise ValueError('Regional source stage lost a material trace marker.')
    traces = {name: value.copy() for name, value in s.trace_structure.items()}
    for index, column in _trace_rows(s, bound):
        for name in traces:
            traces[name][index] = column[name][0]
    dense_crust.validate(traces)
    return dict(structure=staged, trace_structure=traces,
                store=dict(version=VERSION, epoch_myr=float(s.t) + elapsed,
                           geometry_revision=int(surface['geometry_revision']),
                           records=next_records),
                returned_volume_km3=float(returned_volume),
                contraction_volume_km3=float(contraction_volume),
                maximum_ordinary_pressure_pa=float(maximum_pressure),
                remapped_markers=len(bound),
                scope='read-only local sources; no native motion, upper work or mantle-return commit')


