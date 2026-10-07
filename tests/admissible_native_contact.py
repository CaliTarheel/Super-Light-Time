"""Small native-format paired faces within the represented slab window.

The older active-contact source fixture places its face near the hinge pole.
This one uses the finite near-hinge pair and installs genuine regional store
records for read-only contact/force integration checks.
"""

from types import SimpleNamespace

import numpy as np

import channel_region_native
import mesh_coverage
from channel_region_history import initialize_region_history
from tests.test_channel_entry_zones import paired
from tests.test_channel_region_native import uniform_world


def admissible_native_contact_world():
    surface, sheets, owners, plate_uids, contacts, spec = paired()
    radius = float(surface['radius_km'])
    faces = surface['faces']
    vertices = surface['vertices']
    areas = np.array([mesh_coverage._polygon_area(vertices[face], radius)
                      for face in faces])
    source = uniform_world().channel_region_store['records'][0][
        'history']['regions'][0]['column']
    structure = {name: np.repeat(np.asarray(values), len(faces))
                 for name, values in source.items()}
    native_surface = dict(surface, geometry_revision=0,
                          area_km2=areas.copy(),
                          reference_area_km2=areas.copy(),
                          face_owner=owners.copy())
    s = SimpleNamespace(
        t=2., material_surface=native_surface,
        mass=areas.copy(), structure=structure,
        trace_id=np.empty(0, np.int64),
        trace_patch=np.empty(0, np.int64),
        trace_xyz=np.empty((0, 3)),
        trace_structure={name: np.empty(0, float) for name in structure},
        parcel_patch=surface['face_id'].copy(),
        parcel_collision_sheet=sheets.copy(),
        parcel_plate=owners.copy(),
        plate_uid=plate_uids.copy(),
        active=np.ones(len(plate_uids), bool),
        omega=np.zeros((len(plate_uids), 3)),
        collision_contacts=[dict(row) for row in contacts],
        trench_systems=[dict(id=1, downgoing_plate_uid=11,
                             overriding_plate_uid=22)],
        plate_balance_version=1, plate_resistance_version=1,
        material_mechanics_version=1,
        config=dict(deforming_regions=1),
        parcel_entry_region=np.array([1, 0], np.int64),
        continental_entry_regions=dict(
            version=1, epoch_myr=2., face_ids=surface['face_id'].copy(),
            regions=[dict(id=1, source_trench_id=1,
                          overriding_plate_uid=22,
                          hinge_normal=spec['hinge_normals'][0].copy(),
                          dip_degrees=50., created_myr=2.)]))
    records = []
    for index, face in enumerate(faces):
        face_id = int(surface['face_id'][index])
        triangle = vertices[face]
        history = initialize_region_history(
            s.structure, index, [dict(region_id='face', fraction=1.)],
            face_id=face_id)
        if index == 1:
            history['regions'][0]['upper_basal_reference'] = dict(
                depth_m=35000., epoch_myr=1.)
        records.append(dict(face_id=face_id, face_triangle=triangle.copy(),
                            history=history, polygons=[triangle.copy()]))
    channel_region_native.install(s, records)
    return s, int(surface['face_id'][0]), int(surface['face_id'][1])
