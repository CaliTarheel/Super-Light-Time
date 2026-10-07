"""Read-only native geometry and gravity port for the joint SI formulation.

No native solver selection, constitutive calibration, contact admission or
evolution occurs here. Euler slots include active ocean-only plates. Material
gravity is exported once as nodal forces, not also as a rigid collision driver.
"""
from copy import copy
from dataclasses import dataclass
import hashlib
import json

import numpy as np

import column_density
import deforming_regions
import enhanced_rifting
import entry_regions
import gravitational_relaxation as gravity
import native_material_evolution
import material_surface
import plate_balance
import world_design


def _feed(h, value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError('Native mapping cannot bind executable/object arrays.')
        h.update(b'array'); h.update(value.dtype.str.encode())
        h.update(str(value.shape).encode()); h.update(value.tobytes(order='C'))
    elif isinstance(value, dict):
        h.update(b'dict'+str(len(value)).encode()+b':')
        for key in sorted(value):
            _feed(h, key); _feed(h, value[key])
    elif isinstance(value, (list, tuple)):
        h.update(type(value).__name__.encode()); h.update(str(len(value)).encode())
        for child in value:
            _feed(h, child)
    else:
        if isinstance(value, np.generic):
            value = value.item()
        raw = json.dumps(value, allow_nan=False, sort_keys=True).encode()
        h.update(str(len(raw)).encode()+b':'+raw)


def _digest(value):
    h = hashlib.sha256(); _feed(h, value); return h.hexdigest()


def source_signature(s):
    """Bind fields used for geometry, anchors and this material GPE potential."""
    mesh = s.material_surface
    fields = {name: getattr(s, name, None) for name in (
        't', 'active', 'plate_uid', 'parcel_plate', 'parcel_patch',
        'parcel_collision_sheet', 'parcel_arc_id', 'structure', 'config',
        'world_design', 'collision_contacts', 'retained_dense_crust_version',
        'native_arc_birth_profile_version', 'continental_entry_regions',
        'parcel_entry_region', 'automatic_entry_version', 'trench_shutdown_version',
        'material_mechanics_version')}
    fields['material_surface'] = {name: mesh[name] for name in (
        'vertices', 'faces', 'vertex_owner', 'face_owner', 'face_id',
        'face_kind', 'area_km2')}
    fields['radius_km'] = mesh.get('radius_km', 6371.)
    return _digest(fields)


def _readonly(value):
    result = np.array(value, copy=True); result.flags.writeable = False
    return result


def _integers(value, shape, name):
    result = np.asarray(value)
    if (result.shape != shape or result.dtype.kind not in 'iu'
            or np.any(result < 0) or np.any(result > np.iinfo(np.int64).max)):
        raise ValueError(name+' must be aligned integer data.')
    return result


def _real(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'iuf' or not np.isfinite(raw).all():
        raise ValueError(name+' must contain finite real numbers.')
    return raw.astype(float, copy=False)


@dataclass(frozen=True)
class NativeJointState:
    points: np.ndarray
    faces: np.ndarray
    face_ids: np.ndarray
    sheets: np.ndarray
    vertex_plate: np.ndarray
    active_native_slots: np.ndarray
    plate_uids: np.ndarray
    fixed_residual_dofs: np.ndarray
    craton_anchor_vertices: np.ndarray
    authored_protected_vertices: np.ndarray
    corner_guard_vertices: np.ndarray
    tangent_basis: np.ndarray
    radius_m: float
    native_plate_capacity: int
    signature: str
    binding: str

    def _data(self):
        return {name: value for name, value in vars(self).items()
                if name not in ('signature', 'binding')}

    def validate_source(self, state):
        if _digest(self._data()) != self.binding:
            raise ValueError('Native joint mapping was modified after preparation.')
        if source_signature(state) != self.signature:
            raise ValueError('Native geometry, ownership, columns or selected policy changed.')

    @property
    def plate_count(self):
        return len(self.active_native_slots)

    @property
    def unknown_count(self):
        return 3*self.plate_count+2*len(self.points)

    def assembly_geometry(self):
        """Arguments for the full-Euler/fixed-residual sparse assembly API."""
        return dict(points=self.points, faces=self.faces, vertex_plate=self.vertex_plate,
            radius_m=self.radius_m, plate_count=self.plate_count,
            fixed_residual_dofs=self.fixed_residual_dofs, preserve_represented_points=True)

    def pack_velocity(self, native_omega_rad_s, residual_velocity_m_s):
        """Convert explicitly SI native-slot omega and tangent Cartesian residuals."""
        omega = _real(native_omega_rad_s, 'Plate angular velocity')
        u = _real(residual_velocity_m_s, 'Residual velocity')
        if (omega.shape != (self.native_plate_capacity, 3) or u.shape != self.points.shape
                or not np.isfinite(omega).all() or not np.isfinite(u).all()):
            raise ValueError('SI velocities must align with native plates and vertices.')
        if np.any(abs(np.sum(u*self.points, axis=1)) > 2e-12*np.linalg.norm(u, axis=1)):
            raise ValueError('Residual velocity must be tangent; radial motion is not discarded.')
        z = np.einsum('nik,ni->nk', self.tangent_basis, u)
        if np.any(z[self.fixed_residual_dofs] != 0.):
            raise ValueError('Prescribed zero residual coordinates must be exactly zero.')
        return np.r_[self.radius_m*omega[self.active_native_slots].ravel(), z.ravel()]

    def unpack_velocity(self, generalized_velocity_m_s):
        y = _real(generalized_velocity_m_s, 'Generalized velocity')
        if y.shape != (self.unknown_count,) or not np.isfinite(y).all():
            raise ValueError('Joint velocity must be finite and match this mapping.')
        cut = 3*self.plate_count; z = y[cut:].reshape(-1, 2)
        if np.any(z[self.fixed_residual_dofs] != 0.):
            raise ValueError('Joint velocity violates a prescribed residual coordinate.')
        omega = np.zeros((self.native_plate_capacity, 3))
        omega[self.active_native_slots] = y[:cut].reshape(-1, 3)/self.radius_m
        u = np.einsum('nik,nk->ni', self.tangent_basis, z)
        total = np.cross(y[:cut].reshape(-1, 3)[self.vertex_plate], self.points)+u
        return dict(native_omega_rad_s=omega, residual_velocity_m_s=u,
            total_velocity_m_s=total, inactive_native_slots_zeroed=True)


def prepare(s):
    """Copy and validate an existing native state; never initialize or refresh it."""
    before = source_signature(s); mesh = s.material_surface
    version = getattr(s, 'material_mechanics_version', 0)
    if (isinstance(version, (bool, np.bool_)) or not isinstance(version, (int, np.integer)) or version != 1
            or not s.config.get('deforming_regions', 1)):
        raise ValueError('This mapping requires selected native material mechanics version one.')
    for name in ('native_arc_birth_profile_version', 'retained_dense_crust_version'):
        value = getattr(s, name, 0)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value not in (0, 1):
            raise ValueError('Unsupported selected native policy: '+name)
    p = _real(mesh['vertices'], 'Native vertices'); f = np.asarray(mesh['faces'])
    active = np.asarray(s.active)
    if active.ndim != 1 or active.dtype.kind != 'b' or not active.any():
        raise ValueError('Native active mask must be nonempty Boolean data.')
    uid = _integers(s.plate_uid, active.shape, 'Plate UIDs')
    slots = np.flatnonzero(active)
    if np.any(uid[slots] < 0) or len(np.unique(uid[slots])) != len(slots):
        raise ValueError('Every active plate requires a unique persistent UID.')
    if (p.ndim != 2 or p.shape[1:] != (3,) or not len(p) or not np.isfinite(p).all()
            or np.any(abs(np.linalg.norm(p, axis=1)-1.) > 2e-12)):
        raise ValueError('Native vertices must be finite unit spherical points.')
    if (f.ndim != 2 or f.shape[1:] != (3,) or not len(f) or f.dtype.kind not in 'iu'
            or np.any(f < 0) or np.any(f >= len(p))):
        raise ValueError('Native material faces must be valid integer triples.')
    vo = _integers(mesh['vertex_owner'], (len(p),), 'Vertex owners')
    fo = _integers(mesh['face_owner'], (len(f),), 'Face owners')
    ids = _integers(mesh['face_id'], (len(f),), 'Persistent face IDs')
    sheets = _integers(s.parcel_collision_sheet, (len(f),), 'Collision sheets')
    if (np.any(vo < 0) or np.any(vo >= len(active)) or not np.all(active[vo])
            or np.any(vo[f] != fo[:, None]) or np.any(sheets < 0)
            or len(np.unique(ids)) != len(ids)):
        raise ValueError('Material faces must retain coherent active ownership and unique identities.')
    if (not np.array_equal(_integers(s.parcel_patch, ids.shape, 'Parcel IDs'), ids)
            or not np.array_equal(_integers(s.parcel_plate, fo.shape, 'Parcel owners'), fo)):
        raise ValueError('Native parcel identities and ownership must align with material faces.')
    if len(np.unique(f)) != len(p):
        raise ValueError('Unused native vertices need an explicit retirement transaction.')
    kind = _integers(mesh['face_kind'], (len(f),), 'Face kinds')
    if np.any((kind < 1) | (kind > 3)):
        raise ValueError('Unsupported native face kind.')
    area = _real(mesh['area_km2'], 'Saved face areas')
    if area.shape != (len(f),) or not np.isfinite(area).all() or np.any(area <= 0.):
        raise ValueError('Native saved face areas must be finite and positive.')
    raw_radius = _real(mesh.get('radius_km', 6371.), 'Native radius')
    if raw_radius.shape != ():
        raise ValueError('Native radius must be a scalar.')
    radius = float(raw_radius)*1000.
    if not np.isfinite(radius) or radius <= 0.:
        raise ValueError('Native radius must be finite and positive.')
    triangle = p[f]
    orientation = np.einsum('ni,ni->n', triangle[:, 0],
        np.cross(triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0]))
    if np.any(orientation <= 0.):
        raise ValueError('Native material faces must have positive orientation.')
    measured = material_surface.spherical_face_areas(p, f, radius/1000.)
    if np.any(abs(measured-area) > 64*np.finfo(float).eps*np.maximum(measured, area)):
        raise ValueError('Saved face areas disagree with native material geometry.')
    if enhanced_rifting.enabled(s):
        anchors, _ = enhanced_rifting.craton_anchors(mesh,
            s.config['enhanced_rifting'].get('craton_margin_km', 150.))
    else:
        anchors = np.zeros(len(p), bool); anchors[np.unique(f[kind == 2])] = True
    # This native helper writes diagnostics only; put those on a shallow proxy.
    options = world_design.deformation_options(copy(s))
    protected = np.asarray(options.get('vertex_protected', np.zeros(len(p), bool)))
    if protected.shape != (len(p),) or protected.dtype.kind != 'b':
        raise ValueError('Authored protection must be an aligned Boolean mask.')
    response = np.asarray(options.get('vertex_response', np.ones(len(p))), float)
    if response.shape != (len(p),) or np.any(response[~protected] != 1.):
        raise ValueError('Partial authored kinematic response has no joint constitutive mapping yet.')
    # Native mechanics fixes shared corners between different edge-connected
    # bodies. Their coincident point is not an admitted finite-area attachment.
    components, _, _, _ = deforming_regions._material_components(p, f, 0.,
        np.zeros(len(p), bool), kind == 2, radius/1000.)
    corner_guard = components < 0
    dense_slot = np.full(len(active), -1, dtype=np.int64); dense_slot[slots] = np.arange(len(slots))
    axes = np.eye(3)[np.argmin(abs(p), axis=1)]
    first = np.cross(p, axes); first /= np.linalg.norm(first, axis=1)[:, None]
    basis = np.stack((first, np.cross(p, first)), axis=2)
    fields = dict(points=p, faces=f, face_ids=ids, sheets=sheets,
        vertex_plate=dense_slot[vo], active_native_slots=slots, plate_uids=uid[slots],
        fixed_residual_dofs=np.repeat((anchors | protected | corner_guard)[:, None], 2, axis=1),
        craton_anchor_vertices=anchors, authored_protected_vertices=protected,
        corner_guard_vertices=corner_guard, tangent_basis=basis)
    fields = {name: _readonly(value) for name, value in fields.items()}
    fields.update(radius_m=radius, native_plate_capacity=len(active))
    if source_signature(s) != before:
        raise ValueError('Native source changed while preparing its joint mapping.')
    return NativeJointState(**fields, signature=before, binding=_digest(fields))


def gravitational_load(s, mapping):
    """Physical force -dE/dx[N] at every vertex, including its rigid share once.

    This uses the selected native self/ordered-stack GPE potential. It cannot
    stand in for unimplemented entry/stack coupling or an evolving work proof.
    """
    mapping.validate_source(s)
    if entry_regions.enabled(s):
        raise ValueError('Active entry requires a complete coupled entry/stack potential.')
    energy, gradient, diagnostics = gravity.energy_gradient(mapping.points, mapping.faces,
        s.structure['thickness_km'], native_material_evolution.gravitational_reference(s),
        mapping.sheets, radius=mapping.radius_m/1000.,
        overlap=getattr(s, '_collision_overlap', None), **column_density.options(s))
    force = -gravity.ENERGY_UNIT_J/mapping.radius_m*gradient
    # The nodal map already places this rigid share into Euler equations.
    torque = np.zeros((mapping.plate_count, 3))
    np.add.at(torque, mapping.vertex_plate, mapping.radius_m*np.cross(mapping.points, force))
    mapping.validate_source(s)
    return dict(external_nodal_forces_n=_readonly(force), energy_j=energy*gravity.ENERGY_UNIT_J,
        rigid_share_torque_n_m=_readonly(torque), diagnostics=diagnostics,
        add_separate_collision_torque=False, source_signature=mapping.signature)


def rigid_driver_torques(mapping, drivers_w, native_slots):
    """Port a frozen native Balance.drivers ledger, excluding its duplicate GPE.

    Input values are W conjugate to native x=R*omega/v0, not N m. Only existing
    slab and oceanic ridge terms are accepted. The caller must establish that
    its ledger belongs to this same frozen state; no cached ledger is inferred.
    """
    if mapping.radius_m != plate_balance.RADIUS_M:
        raise ValueError('Native Balance driver units require its fixed Earth radius.')
    slots = _integers(native_slots, (mapping.plate_count,), 'Driver native slots')
    if not np.array_equal(slots, mapping.active_native_slots):
        raise ValueError('Rigid drivers must retain this exact active native-slot order.')
    if set(drivers_w) != {'slab', 'ridge', 'collision'}:
        raise ValueError('Expected the complete native slab/ridge/collision driver ledger.')
    converted = {}
    for name, value in drivers_w.items():
        a = _real(value, 'Native power-scaled driver')
        if a.shape != (3*mapping.plate_count,) or not np.isfinite(a).all():
            raise ValueError('Native power-scaled driver must be finite and aligned.')
        if name != 'collision':
            converted[name] = _readonly(a.reshape(-1, 3)*(mapping.radius_m/plate_balance.CM_YR_M_S))
    return dict(external_torques_n_m=_readonly(converted['slab']+converted['ridge']),
        components_n_m=converted, excluded_duplicate_driver='collision',
        scope='Existing rigid-only native drivers; not a full native resistance port.')
