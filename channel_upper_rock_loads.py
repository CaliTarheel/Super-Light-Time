"""Read-only physical rock mass on each ordered upper contact region.

This measures the mass available to a future layered lithostatic source. It
does not infer the upper basal height, water cover, mantle gap, or temperature.
"""

import numpy as np

import column_density
import crustal_structure
import dense_crust


def physical_rock_column(column):
    """Return physical ordinary/dense thickness and mass per horizontal area."""
    thickness = np.asarray(column['thickness_km'], float)
    factor = np.asarray(column['area_factor'], float)
    if (thickness.shape != (1,) or factor.shape != (1,)
            or not np.isfinite(thickness).all() or not np.isfinite(factor).all()
            or thickness[0] <= 0. or factor[0] <= 0.):
        raise ValueError('Upper rock load needs one positive physical column.')
    if not dense_crust.present(column):
        raise ValueError('Upper rock load needs the explicit retained-phase inventory.')
    surface_elevation = float(crustal_structure.elevation(column)[0])
    dense_crust.validate(column)
    dense = float(column[dense_crust.DENSE][0] / factor[0])
    ordinary = float(thickness[0]) - dense
    if ordinary < -1e-12 or dense < 0.:
        raise ValueError('Upper rock load has invalid retained phase geometry.')
    ordinary = max(ordinary, 0.)
    mass = 1000. * (column_density.RHO_CRUST * ordinary
                    + column_density.RHO_DENSE * dense)
    conserved_mass = (1000. * column_density.RHO_CRUST
                      * dense_crust.mass_volume(column)[0] / factor[0])
    if not np.isclose(mass, conserved_mass, rtol=2e-12, atol=1e-6):
        raise ValueError('Upper rock load disagrees with conserved phase mass.')
    return dict(ordinary_thickness_km=ordinary, dense_thickness_km=dense,
                physical_thickness_km=float(thickness[0]),
                rock_mass_kg_m2=float(mass),
                current_surface_elevation_m=surface_elevation)


def measure_native_upper_rock(s, entry_geometry):
    """Bind each clipped source zone to its actual upper rock columns."""
    surface = s.material_surface
    store = s.channel_region_store
    if (store.get('version') != 1 or store.get('epoch_myr') != float(s.t)
            or store.get('geometry_revision') != int(surface['geometry_revision'])):
        raise ValueError('Native upper rock needs current regional source state.')
    ids = np.asarray(surface['face_id'])
    by_id = {int(face_id): index for index, face_id in enumerate(ids)}
    if len(by_id) != len(ids):
        raise ValueError('Native upper rock needs unique material face IDs.')
    records = {int(record['face_id']): record
               for record in store['records']}
    if len(records) != len(store['records']):
        raise ValueError('Native upper rock has duplicate regional face IDs.')
    result = {}
    for lower_id, report in entry_geometry['entry_faces'].items():
        source_zones = []
        for zone in report['zones']:
            faces = zone['covering_upper_face_ids_top_to_bottom']
            sheets = zone['covering_sheets_top_to_bottom']
            bindings = zone['upper_region_bindings_top_to_bottom']
            if (len(faces) != len(sheets) or len(faces) != len(bindings)
                    or tuple(map(int, faces)) != tuple(face_id for face_id, _ in bindings)):
                raise ValueError('Native upper rock lost ordered regional bindings.')
            layers = []
            for face_id, sheet_id, (_, region_id) in zip(faces, sheets, bindings):
                face_id = int(face_id)
                if face_id not in by_id:
                    raise ValueError('Native upper rock face is missing from the surface.')
                if int(s.parcel_collision_sheet[by_id[face_id]]) != int(sheet_id):
                    raise ValueError('Native upper rock sheet identity is stale.')
                record = records.get(face_id)
                if record is None:
                    if region_id is not None:
                        raise ValueError('Native upper rock region has no persistent record.')
                    index = by_id[face_id]
                    column = {name: np.asarray(value)[index:index+1]
                              for name, value in s.structure.items()}
                else:
                    selected = [region for region in record['history']['regions']
                                if region['region_id'] == region_id]
                    if len(selected) != 1:
                        raise ValueError('Native upper rock region identity is ambiguous.')
                    column = selected[0]['column']
                layers.append(dict(face_id=face_id, sheet_id=int(sheet_id),
                                   region_id=region_id,
                                   **physical_rock_column(column)))
            source_zones.append(dict(zone_id=zone['zone_id'], kind=zone['kind'],
                                     area_km2=float(zone['area_km2']),
                                     rock_layers_top_to_bottom=layers))
        result[int(lower_id)] = source_zones
    return dict(by_lower_face=result,
                scope='read-only measured upper rock mass; no vertical path or source commit')
