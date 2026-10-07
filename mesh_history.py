"""Native spherical geometry retained alongside projected review rasters."""
import numpy as np
import material_transport
import continental_margin
import gravity_constraints_schema
import progressive_rifting
import collision_contacts
import collision_surface
import collision_coast
import native_subduction
import native_initial_ownership
import native_spreading
import localized_accretion
import backarc
import arc_emplacement_geometry
import arc_birth_profile
import lip_events
import eclogite_sink
import effective_subduction
import force_rifting
import numerical_accuracy

DTYPES = {
    'mesh_vertices': np.float64, 'mesh_faces': np.int32,
    'mesh_plate': np.int32, 'mesh_crust': np.uint8, 'mesh_age_myr': np.float64,
    'mesh_elevation_m': np.float64, 'mesh_area_km2': np.float64,
    'mesh_ocean_damage': np.float64, 'mesh_ocean_strain': np.float64,
    'mesh_primordial_fraction': np.float64,
    'mesh_ocean_relief_m': np.float64, 'mesh_boundary': np.uint8,
    'mesh_ocean_weakness': np.float64, 'mesh_ocean_tensile_exposure_myr': np.float64,
    'material_vertices': np.float64, 'material_faces': np.int32,
    'material_face_id': np.int64, 'material_owner': np.int32,
    'material_kind': np.uint8, 'material_height_m': np.float64,
    'material_reference_area_km2': np.float64,
    'material_domain': np.int32,
    'material_crustal_thickness_km': np.float64, 'material_crustal_root_km': np.float64,
    'material_rift_thermal_support_m': np.float64, 'material_rift_cooling_age_myr': np.float64,
    'material_foreland_deflection_m': np.float64, 'material_erosion_rate_m_myr': np.float64,
    'material_rift_damage': np.float64, 'material_rift_strength_relative': np.float64,
    'material_rift_id': np.int64, 'material_rift_birth_myr': np.float64,
    'material_rift_extension_m': np.float64, 'material_inversion_uplift_m': np.float64,
    'native_boundary_edges': np.int32, 'native_boundary_code': np.uint8,
    'native_boundary_owner_a': np.int32, 'native_boundary_owner_b': np.int32,
    'native_boundary_down': np.int32, 'native_boundary_normal_speed_km_myr': np.float64,
}
OWNER_DTYPES = {'mesh_owner_slots': np.int32, 'mesh_vertex_support': np.float64}
DTYPES.update(OWNER_DTYPES)
TRANSPORT_DTYPES = {'material_root_id': np.int64, 'material_reference_corners': np.float64,
                    'material_erosion_total_m': np.float64, 'material_parent_id': np.int64}
DTYPES.update(TRANSPORT_DTYPES)
DEFORMATION_DTYPES = {'material_refinement_level': np.int32, 'material_actual_area_km2': np.float64,
                      'material_deformation_weight': np.float64, 'material_rigid': np.uint8,
                      'material_geometric_log_area': np.float64}
DTYPES.update(DEFORMATION_DTYPES)
ARC_DTYPES = {'material_arc_id': np.int64, 'material_arc_basal_m': np.float64}
DTYPES.update(ARC_DTYPES)
DTYPES.update(collision_contacts.ARRAY_FIELDS)
DTYPES.update(collision_surface.ARRAY_FIELDS)
COAST_DTYPES = {collision_coast.REFERENCE_FIELD: np.float64}
DTYPES.update(COAST_DTYPES)
DTYPES.update(localized_accretion.ARRAY_FIELDS)
DTYPES.update(lip_events.ARRAY_FIELDS)
DTYPES.update(eclogite_sink.ARRAY_FIELDS)
ENHANCED_RIFT_DTYPES = {'material_rift_seed_weakness': np.float64,
    'material_rift_realized_tension': np.float64, 'material_rift_realized_compression': np.float64}
DTYPES.update(ENHANCED_RIFT_DTYPES)
DTYPES.update(numerical_accuracy.ARRAY_FIELDS)


def arrays(snapshot):
    effective_subduction.validate_frame(snapshot)
    force_rifting.validate_frame(snapshot)
    import surface_erosion
    surface_erosion.version(snapshot)
    gravity_constraints_schema.validate_frame(snapshot)
    progressive_rifting.validate_frame(snapshot)
    if 'mesh_version' not in snapshot:
        return {}
    mechanics=snapshot.get('material_mechanics_version',0)
    if isinstance(mechanics,(bool,np.bool_)) or not isinstance(mechanics,(int,np.integer)) or mechanics not in (0,1):
        raise ValueError('Unsupported native material mechanics version.')
    continental_margin.validate_frame(snapshot)
    collision_contacts.validate_frame(snapshot)
    collision_surface.validate_frame(snapshot)
    native_subduction.validate_frame(snapshot)
    native_initial_ownership.validate_frame(snapshot)
    native_spreading.validate_frame(snapshot)
    arc_emplacement_geometry.validate_frame(snapshot)
    arc_birth_profile.validate_frame(snapshot)
    localized_accretion.validate_frame(snapshot)
    backarc.validate_frame(snapshot)
    lip_events.validate_frame(snapshot)
    eclogite_sink.validate_frame(snapshot)
    if snapshot.get('enhanced_rifting_version', 0) or set(ENHANCED_RIFT_DTYPES).intersection(snapshot):
        import enhanced_rifting
        enhanced_rifting.validate_frame(snapshot)
    optional = (set(OWNER_DTYPES) | set(TRANSPORT_DTYPES) | set(DEFORMATION_DTYPES)
                | set(ARC_DTYPES) | set(collision_contacts.ARRAY_FIELDS) | set(collision_surface.ARRAY_FIELDS) | set(COAST_DTYPES)
                | set(localized_accretion.ARRAY_FIELDS) | set(lip_events.ARRAY_FIELDS) | set(ENHANCED_RIFT_DTYPES)
                | set(eclogite_sink.ARRAY_FIELDS))
    optional |= set(numerical_accuracy.ARRAY_FIELDS)
    if snapshot['mesh_version'] != 1 or not (set(DTYPES)-optional).issubset(snapshot):
        raise ValueError('Native mesh history is incomplete or has an unsupported version.')
    owner_version = snapshot.get('owner_reconstruction_version', 0)
    if owner_version not in (0, 1):
        raise ValueError('Unsupported native owner reconstruction version.')
    owner_fields = set(OWNER_DTYPES).intersection(snapshot)
    if ((owner_version == 1 and owner_fields != set(OWNER_DTYPES))
            or (owner_version == 0 and owner_fields)):
        raise ValueError('Native owner reconstruction fields require their version and complete scores.')
    transport_version = snapshot.get('material_transport_version', 0)
    if transport_version not in (0, 1):
        raise ValueError('Unsupported native material transport version.')
    if transport_version == 1:
        material_transport.validate_fields(snapshot)
    elif (set(TRANSPORT_DTYPES) | set(DEFORMATION_DTYPES)).intersection(snapshot):
        raise ValueError('Material reference fields require their transport version.')
    arc_version = snapshot.get('arc_material_version', 0)
    surface_version = snapshot.get('arc_surface_version', 0)
    arc_fields = set(ARC_DTYPES).intersection(snapshot)
    if arc_version not in (0, 1) or surface_version not in (0, 1, 2):
        raise ValueError('Unsupported arc material or surface version.')
    if ((arc_version == 1 and arc_fields != set(ARC_DTYPES))
            or (arc_version == 0 and arc_fields)
            or (surface_version > 0 and arc_version != 1)):
        raise ValueError('Arc reconstruction needs versioned material identity and basal fields.')
    result = {}
    for name, dtype in DTYPES.items():
        if name not in snapshot:
            continue
        value = np.asarray(snapshot[name])
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise ValueError(f'Native history {name} must contain finite numeric values.')
        if np.issubdtype(dtype, np.integer) and (np.any(value != np.floor(value))
                or np.any(value < (-1 if name in ('native_boundary_down', 'material_rift_id', 'material_parent_id') else 0)) or np.any(value > np.iinfo(dtype).max)):
            raise ValueError(f'Native history {name} has invalid integer values.')
        result[name] = value.astype(dtype)
    for prefix in ('mesh', 'material'):
        vertices, faces = result[prefix+'_vertices'], result[prefix+'_faces']
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2
                or faces.shape[1] != 3 or np.any(faces >= len(vertices))
                or not np.allclose(np.linalg.norm(vertices, axis=1), 1., atol=1e-10)):
            raise ValueError(f'Invalid {prefix} spherical geometry.')
        for name, value in result.items():
            if (name.startswith(prefix+'_') and name not in (prefix+'_vertices', prefix+'_faces')
                    and name not in OWNER_DTYPES and name != 'material_reference_corners'
                    and name not in lip_events.ARRAY_FIELDS):
                if value.shape != (len(faces),):
                    raise ValueError(f'Native field {name} must align with its material faces.')
    if len(np.unique(result['material_face_id'])) != len(result['material_face_id']):
        raise ValueError('Material face IDs must be unique within a saved epoch.')
    edges = result['native_boundary_edges']
    if edges.ndim != 2 or edges.shape[1] != 2 or np.any(edges >= len(result['mesh_vertices'])):
        raise ValueError('Native boundary edges must refer to control-mesh vertices.')
    for name, value in result.items():
        if name.startswith('native_boundary_') and name != 'native_boundary_edges' and value.shape != (len(edges),):
            raise ValueError('Native boundary fields must align with shared edges.')
    for name in ('mesh_area_km2', 'material_reference_area_km2'):
        if np.any(result[name] <= 0):
            raise ValueError('Native reference areas must remain positive.')
    if 'material_actual_area_km2' in result and np.any(result['material_actual_area_km2'] <= 0):
        raise ValueError('Native actual material areas must remain positive.')
    for name in ('material_deformation_weight', 'material_rigid'):
        if name in result and np.any((result[name] < 0) | (result[name] > 1)):
            raise ValueError(f'Native material field {name} must lie between zero and one.')
    for name in ('mesh_ocean_damage', 'mesh_primordial_fraction'):
        if np.any((result[name] < -1e-12) | (result[name] > 1+1e-12)):
            raise ValueError(f'Native fraction {name} must lie between zero and one.')
    if owner_version == 1:
        slots, scores = result['mesh_owner_slots'], result['mesh_vertex_support']
        if (slots.ndim != 1 or len(slots) == 0 or np.any(np.diff(slots) <= 0)
                or scores.shape != (len(slots), len(result['mesh_vertices']))
                or np.any(scores < 0) or not np.isin(result['mesh_plate'], slots).all()):
            raise ValueError('Native owner scores must align with sorted unique slots and mesh vertices.')
        if 'plates' in snapshot and not np.isin(slots, [row['id'] for row in snapshot['plates']]).all():
            raise ValueError('Native owner slots require recorded plate metadata.')
    return result
