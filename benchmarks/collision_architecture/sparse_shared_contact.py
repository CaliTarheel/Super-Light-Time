"""Matrix-free local-block assembly of the reviewed joint SI functional.

Research assembly/matvec/work only: there is deliberately no velocity solver,
contact admission or native integration. NumPy is the only new-module runtime
dependency. No full unknown-by-unknown array or material nullspace is formed;
optional anchor frames use only a three-column QR and a 3x3 SVD core.
"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib

import numpy as np

import mesh_geometry
import viscous_sheet
from . import shared_contact as dense


@dataclass(frozen=True)
class BlockMap:
    """Sum small indexed linear maps; repeated output rows sum contributions."""
    blocks: np.ndarray
    rows: np.ndarray
    columns: np.ndarray
    shape: tuple

    def __post_init__(self):
        b, r, c = np.asarray(self.blocks), np.asarray(self.rows), np.asarray(self.columns)
        if (b.ndim != 3 or r.shape != b.shape[:2] or c.shape != (len(b), b.shape[2])
                or r.dtype.kind not in 'iu' or c.dtype.kind not in 'iu'
                or not np.isfinite(b).all() or np.any(r < 0) or np.any(c < 0)
                or np.any(r >= self.shape[0]) or np.any(c >= self.shape[1])):
            raise ValueError('Finite local blocks and indices must align.')

    def __matmul__(self, value):
        x = _vector(value, self.shape[1])
        local = np.einsum('bij,bj->bi', self.blocks, x[self.columns])
        result = np.bincount(self.rows.ravel(), weights=local.ravel(), minlength=self.shape[0])
        if not np.isfinite(result).all():
            raise RuntimeError('Local-block action is outside finite arithmetic range.')
        return result

    def rmatvec(self, value):
        x = _vector(value, self.shape[0])
        local = np.einsum('bij,bi->bj', self.blocks, x[self.rows])
        result = np.bincount(self.columns.ravel(), weights=local.ravel(), minlength=self.shape[1])
        if not np.isfinite(result).all():
            raise RuntimeError('Local-block transpose is outside finite arithmetic range.')
        return result

    @property
    def storage_bytes(self):
        return self.blocks.nbytes+self.rows.nbytes+self.columns.nbytes

    def to_dense(self):
        """Small audit conversion only; cannot accidentally allocate a large map."""
        if self.shape[1] > 256 or self.shape[0]*self.shape[1] > 2_000_000:
            raise ValueError('Dense conversion is limited to small parity audits.')
        result = np.zeros(self.shape)
        for b, rows, columns in zip(self.blocks, self.rows, self.columns):
            np.add.at(result, (rows[:, None], columns[None]), b)
        return result


@dataclass(frozen=True)
class GramOperator:
    velocity: BlockMap
    weights: np.ndarray
    strain: BlockMap
    other_plate_matrix: np.ndarray

    @property
    def shape(self):
        return (self.velocity.shape[1],)*2

    def __matmul__(self, value):
        y = _vector(value, self.shape[1])
        result = self.velocity.rmatvec(self.weights*(self.velocity@y))
        result += self.strain.rmatvec(self.strain@y)
        cut = len(self.other_plate_matrix)
        result[:cut] += self.other_plate_matrix@y[:cut]
        if not np.isfinite(result).all():
            raise RuntimeError('Joint Gram action is outside finite arithmetic range.')
        return result

    def to_dense(self):
        if self.shape[0] > 256:
            raise ValueError('Dense conversion is limited to small parity audits.')
        return np.column_stack([self@column for column in np.eye(self.shape[0])])


def _vector(value, count):
    x = np.asarray(value, float)
    if x.shape != (count,) or not np.isfinite(x).all():
        raise ValueError('Finite vector must align with the operator.')
    return x


def _rotation_maps(points):
    """Map equatorial plate speed a to a cross r, batched over vertices."""
    p = np.asarray(points); result = np.zeros((len(p), 3, 3))
    result[:, 0, 1] = p[:, 2]; result[:, 0, 2] = -p[:, 1]
    result[:, 1, 0] = -p[:, 2]; result[:, 1, 2] = p[:, 0]
    result[:, 2, 0] = p[:, 1]; result[:, 2, 1] = -p[:, 0]
    return result


def _exchange_basis(rows):
    """Remaining rigid exchange directions, using only three-column geometry.

    Exact binary-rational rank distinguishes true dependence from a weak but
    physical anchor direction. QR/SVD is limited to three columns/a 3x3 core;
    an independent direction below its numerical resolution fails closed.
    """
    rows = np.asarray(rows, float).reshape(-1, 3)
    if not len(rows):
        return np.eye(3), np.zeros(3), 0., 0
    independent = []
    for row in rows:
        a = tuple(Fraction(float(x)) for x in row)
        if not any(a):
            continue
        if not independent:
            independent.append(a)
        elif len(independent) == 1:
            b = independent[0]
            if any(a[i]*b[j] != a[j]*b[i] for i, j in ((0, 1), (0, 2), (1, 2))):
                independent.append(a)
        else:
            b, c = independent
            determinant = sum(a[i]*(b[(i+1)%3]*c[(i+2)%3]-b[(i+2)%3]*c[(i+1)%3]) for i in range(3))
            if determinant:
                independent.append(a)
                break
    rank = len(independent)
    core = np.zeros((3, 3))
    reduced = np.linalg.qr(rows, mode='r')
    core[:len(reduced)] = reduced
    _, singular, vh = np.linalg.svd(core)
    tolerance = 256*np.finfo(float).eps*max(len(rows), 3)*singular[0]
    if rank and singular[rank-1] <= tolerance:
        raise ValueError('Anchor exchange rank is numerically unresolved; no direction was silently removed.')
    return vh[rank:].T.copy(), singular, float(tolerance), rank


def assemble(points, faces, vertex_plate, *, radius_m, basal_drag_pa_s_per_m,
             viscosity_pa_s, sheet_thickness_m, basal_reference_velocity_m_s,
             other_plate_rotational_drag_n_m_s, external_torques_n_m,
             external_nodal_forces_n, contacts, face_chunk_size=2048,
             plate_count=None, fixed_residual_dofs=None, preserve_represented_points=False):
    """Assemble the dense reference's physical functional using local factors.

    The remaining-domain plate matrix is still dense in the number of PLATES,
    not material unknowns. Material memory grows linearly with vertices/faces
    and supplied contacts. No solve or condition-number guarantee is supplied.
    """
    radius = dense._scalar(radius_m, 'radius_m')
    if isinstance(face_chunk_size, bool) or not isinstance(face_chunk_size, (int, np.integer)) or face_chunk_size < 1:
        raise ValueError('Face chunk size must be a positive integer.')
    p = np.asarray(points, float); f = np.asarray(faces); owner = np.asarray(vertex_plate)
    if (p.ndim != 2 or p.shape[1:] != (3,) or not len(p) or not np.isfinite(p).all()
            or not np.allclose(np.linalg.norm(p, axis=1), 1., rtol=0., atol=2e-12)):
        raise ValueError('Points must be finite unit spherical vectors.')
    if not isinstance(preserve_represented_points, (bool, np.bool_)):
        raise ValueError('preserve_represented_points must be Boolean.')
    # Native geometric ledgers bind to the actual stored binary coordinates.
    # Renormalizing an already unit-to-roundoff vertex can change an exact
    # halfspace sign or tiny positive piece. Opt-in preserves those bytes;
    # it does not relax the existing unit-vector input validation.
    p = p.copy() if preserve_represented_points else p/np.linalg.norm(p, axis=1)[:, None]
    if (f.ndim != 2 or f.shape[1:] != (3,) or not len(f) or f.dtype.kind not in 'iu'
            or np.any(f < 0) or np.any(f >= len(p))):
        raise ValueError('Faces must be nonempty integer vertex triples.')
    f = f.astype(np.int64, copy=True)
    if owner.shape != (len(p),) or owner.dtype.kind not in 'iu' or np.any(owner < 0):
        raise ValueError('One nonnegative integer plate slot is required per vertex.')
    owner = owner.astype(np.int64, copy=True)
    if plate_count is None:
        count = int(owner.max())+1
        if not np.array_equal(np.unique(owner), np.arange(count)):
            raise ValueError('Inferred plate slots must be contiguous and represented.')
    else:
        if (isinstance(plate_count, (bool, np.bool_)) or not isinstance(plate_count, (int, np.integer))
                or plate_count <= int(owner.max())):
            raise ValueError('Explicit plate_count must include every nonnegative material plate slot.')
        count = int(plate_count)
    fixed = np.zeros((len(p), 2), dtype=bool) if fixed_residual_dofs is None else np.asarray(fixed_residual_dofs)
    if fixed.shape != (len(p), 2) or fixed.dtype.kind != 'b':
        raise ValueError('fixed_residual_dofs must be an explicit Boolean (vertices, 2) mask.')
    fixed = fixed.copy()
    if np.any(owner[f] != owner[f[:, :1]]):
        raise ValueError('A material face cannot straddle plate ownership.')
    nplate = 3*count; size = nplate+2*len(p)
    triangle = p[f]
    signed = np.einsum('ij,ij->i', triangle[:, 0], np.cross(triangle[:, 1], triangle[:, 2]))
    if np.any(signed <= 0.):
        raise ValueError('Material faces must have positive orientation.')
    area_km2 = mesh_geometry.spherical_area(p, f, radius_km=radius/1000.)
    area_m2 = area_km2*1e6
    nodal_area = np.bincount(f.ravel(), weights=np.repeat(area_m2/3., 3), minlength=len(p))
    if np.any(nodal_area <= 0.):
        raise ValueError('Unused material vertices are not supported.')
    beta = dense._field(basal_drag_pa_s_per_m, len(p), 'basal_drag_pa_s_per_m')
    eta = dense._field(viscosity_pa_s, len(f), 'viscosity_pa_s', positive=False)
    thick = dense._field(sheet_thickness_m, len(f), 'sheet_thickness_m')
    mantle = dense._vectors(basal_reference_velocity_m_s, len(p), 'basal_reference_velocity_m_s')
    force = dense._vectors(external_nodal_forces_n, len(p), 'external_nodal_forces_n')
    torque = dense._vectors(external_torques_n_m, count, 'external_torques_n_m')
    if np.any(np.abs(np.einsum('ij,ij->i', mantle, p)) > 1e-12*np.linalg.norm(mantle, axis=1)):
        raise ValueError('Basal reference velocities must be tangent.')
    drag = np.asarray(other_plate_rotational_drag_n_m_s, float)
    if drag.shape != (nplate, nplate) or not np.isfinite(drag).all():
        raise ValueError('Remaining-domain rotational drag must be finite and aligned.')
    if not np.allclose(drag, drag.T, rtol=64*np.finfo(float).eps, atol=0.):
        raise ValueError('Remaining-domain rotational drag must be symmetric.')
    drag = .5*(drag+drag.T)
    eigenvalues, eigenvectors = np.linalg.eigh(drag)
    if eigenvalues.min() < 0.:
        raise ValueError('Remaining-domain rotational drag must be positive semidefinite.')
    other_factor = np.sqrt(eigenvalues)[:, None]*eigenvectors.T/radius

    axes = np.eye(3)[np.argmin(np.abs(p), axis=1)]
    first = np.cross(p, axes); first /= np.linalg.norm(first, axis=1)[:, None]
    basis = np.stack((first, np.cross(p, first)), axis=2)
    rotation = _rotation_maps(p)
    vertex_columns = np.column_stack((3*owner[:, None]+np.arange(3),
        nplate+2*np.arange(len(p))[:, None]+np.arange(2)))
    velocity = BlockMap(np.concatenate((rotation, basis), axis=2),
        np.arange(3*len(p)).reshape(-1, 3), vertex_columns, (3*len(p), size))

    # Each owned face has exactly nine local unknowns: one Euler speed and
    # two residual coordinates at each of its three vertices. Use the actual
    # viscous_sheet geometry on disjoint copies in bounded-size chunks; that
    # module's full global strain field is never assembled column by column.
    face_columns = np.column_stack((3*owner[f[:, 0], None]+np.arange(3),
        (nplate+2*f[:, :, None]+np.arange(2)).reshape(-1, 6)))
    strain_blocks = np.empty((len(f), 10, 9))
    for begin in range(0, len(f), face_chunk_size):
        end = min(begin+face_chunk_size, len(f)); selected = f[begin:end]
        local_points = p[selected]; n = len(selected)
        local_faces = np.arange(3*n).reshape(n, 3)
        context = viscous_sheet.prepare(local_points.reshape(-1, 3), local_faces,
            area_km2[begin:end], 0., radius=radius/1000.)
        local_map = np.zeros((n, 3, 3, 9)); local_map[:, :, :, :3] = rotation[selected]
        for corner in range(3):
            local_map[:, corner, :, 3+2*corner:5+2*corner] = basis[selected[:, corner]]
        factor = np.sqrt(2*eta[begin:end]*thick[begin:end]*area_m2[begin:end])
        for column in range(9):
            field = viscous_sheet.strain_rate(context, local_map[:, :, :, column].reshape(-1, 3))
            values = np.column_stack((field['D'].reshape(n, 9), field['divergence']))/1000.
            strain_blocks[begin:end, :, column] = values*factor[:, None]
    strain = BlockMap(strain_blocks, np.arange(10*len(f)).reshape(-1, 10),
        face_columns, (10*len(f), size))
    drag_weight = beta*nodal_area
    operator = GramOperator(velocity, np.repeat(drag_weight, 3), strain, drag/radius**2)
    mantle_load = velocity.rmatvec((drag_weight[:, None]*mantle).ravel())
    nodal_load = velocity.rmatvec(force.ravel())
    torque_load = np.r_[torque.ravel()/radius, np.zeros(2*len(p))]
    load = mantle_load+nodal_load+torque_load

    plate_area = np.bincount(owner, weights=nodal_area, minlength=count)
    weights = nodal_area/plate_area[owner]
    raw_gauge = -np.einsum('nij,njk->nik', rotation, basis)*weights[:, None, None]
    gauge_blocks = np.zeros_like(raw_gauge)
    exchange_basis = np.zeros((count, 3, 3)); exchange_dimension = np.zeros(count, dtype=np.int64)
    anchor_rank = np.zeros(count, dtype=np.int64); anchor_singular = np.zeros((count, 3))
    anchor_tolerance = np.zeros(count)
    exchange_map = -np.einsum('nia,nij->naj', basis, rotation)
    for plate in range(count):
        selected = owner == plate
        if not np.any(selected):
            continue  # An ocean-only Euler slot has no residual exchange gauge.
        basis_plate, singular, threshold, rank = _exchange_basis(exchange_map[selected][fixed[selected]])
        dimension = basis_plate.shape[1]
        exchange_basis[plate, :, :dimension] = basis_plate
        exchange_dimension[plate] = dimension; anchor_rank[plate] = rank
        anchor_singular[plate] = singular; anchor_tolerance[plate] = threshold
        rows = np.einsum('ji,njk->nik', basis_plate, raw_gauge[selected])
        gauge_blocks[selected, :dimension] = rows
        # Only the remaining <=3 gauge moment is factored; fixed coordinates
        # are eliminated, so test its actually free columns.
        free_rows = rows*(~fixed[selected])[:, None, :]
        moment = np.einsum('nik,njk->ij', free_rows, free_rows)
        if dimension and np.linalg.matrix_rank(moment) != dimension:
            raise ValueError('Residual rigid-moment gauge is unresolved on a plate.')
    gauge = BlockMap(gauge_blocks, 3*owner[:, None]+np.arange(3),
        vertex_columns[:, 3:], (nplate, size))

    blocks = []; columns = []; locations = []; normals = []; lengths = []; pairs = []
    for sample in contacts:
        if not isinstance(sample, dense.ContactSample):
            raise ValueError('Contacts must be explicit ContactSample records.')
        ia, ib = sample.first_face, sample.second_face
        if (isinstance(ia, (bool, np.bool_)) or isinstance(ib, (bool, np.bool_))
                or not isinstance(ia, (int, np.integer)) or not isinstance(ib, (int, np.integer))
                or not 0 <= ia < len(f) or not 0 <= ib < len(f)
                or owner[f[ia, 0]] == owner[f[ib, 0]]):
            raise ValueError('Contact faces must belong to distinct represented plates.')
        c = dense._vectors(np.asarray(sample.point)[None], 1, 'Contact point')[0]
        normal = dense._vectors(np.asarray(sample.normal)[None], 1, 'Contact normal')[0]
        if abs(np.linalg.norm(c)-1.) > 2e-12 or np.linalg.norm(normal) == 0.:
            raise ValueError('Contact point must be unit and normal nonzero.')
        c /= np.linalg.norm(c)
        if abs(normal@c) > 2e-12*np.linalg.norm(normal):
            raise ValueError('Contact normal must be tangent.')
        normal -= c*(normal@c); normal /= np.linalg.norm(normal)
        length = dense._scalar(sample.length_m, 'Contact length quadrature weight')
        for sign, face_id in ((1., ia), (-1., ib)):
            face = f[face_id]; alpha = dense._trace(p, face, c)
            row = np.zeros(9)
            row[:3] = np.einsum('n,nij,i->j', sign*alpha, rotation[face], normal)
            row[3:] = (sign*alpha[:, None]*np.einsum('i,nij->nj', normal, basis[face])).ravel()
            blocks.append(row[None]); columns.append(face_columns[face_id])
        locations.append(c); normals.append(normal); lengths.append(length)
        pairs.append((int(owner[f[ia, 0]]), int(owner[f[ib, 0]])))
    length = np.asarray(lengths); contact_count = len(length)
    contact = BlockMap(np.asarray(blocks).reshape(-1, 1, 9),
        np.repeat(np.arange(contact_count), 2)[:, None], np.asarray(columns, dtype=np.int64).reshape(-1, 9),
        (contact_count, size))
    if not all(np.isfinite(a).all() for a in (drag_weight, strain_blocks, load, other_factor)):
        raise ValueError('Requested SI assembly is nonfinite.')
    return dict(points=p, faces=f, vertex_plate=owner, plate_count=count, nplate=nplate,
        radius_m=radius, velocity_map=velocity, tangent_basis=basis,
        preserve_represented_points=bool(preserve_represented_points),
        fixed_residual_dofs=fixed, gauge_exchange_basis=exchange_basis,
        fixed_residual_dofs_sha256=hashlib.sha256(fixed.tobytes()).hexdigest(),
        gauge_exchange_dimension=exchange_dimension, anchor_exchange_rank=anchor_rank,
        anchor_exchange_singular_values=anchor_singular, anchor_exchange_rank_tolerance=anchor_tolerance,
        material_plate_present=plate_area > 0.,
        area_m2=area_m2, nodal_area_m2=nodal_area, basal_drag_weight_n_s_m=drag_weight,
        strain_design=strain, other_plate_drag_factor=other_factor,
        hessian_n_s_m=operator, load_n=load, nodal_load_n=nodal_load,
        torque_load_n=torque_load, mantle_load_n=mantle_load,
        mantle_velocity_m_s=mantle, gauge_matrix=gauge, contact_matrix=contact,
        contact_weights=np.sqrt(length), contact_length_m=length,
        contact_point=np.asarray(locations).reshape(-1, 3), contact_normal=np.asarray(normals).reshape(-1, 3),
        contact_plate_pairs=np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
        assembly_scope='Matrix-free frozen SI assembly; no solver, reactions, admission or native integration.')


def evaluate(system, generalized_velocity_m_s):
    """Rayleigh potential[W] and generalized gradient[N], with no equilibrium claim."""
    y = _vector(generalized_velocity_m_s, len(system['load_n']))
    constant = .5*np.sum(system['basal_drag_weight_n_s_m'][:, None]*system['mantle_velocity_m_s']**2)
    action = system['hessian_n_s_m']@y
    value = float(.5*y@action-system['load_n']@y+constant)
    gradient = action-system['load_n']
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise RuntimeError('Joint functional is outside finite arithmetic range.')
    return value, gradient


def work(system, generalized_velocity_m_s):
    """Evaluate physical power components at a supplied velocity, not a solve."""
    y = _vector(generalized_velocity_m_s, len(system['load_n']))
    velocity = (system['velocity_map']@y).reshape(-1, 3)
    relative = velocity-system['mantle_velocity_m_s']; weight = system['basal_drag_weight_n_s_m']
    basal = float(np.sum(weight[:, None]*relative**2))
    viscous = float(np.linalg.norm(system['strain_design']@y)**2)
    other = float(np.linalg.norm(system['other_plate_drag_factor']@y[:system['nplate']])**2)
    external = float((system['nodal_load_n']+system['torque_load_n'])@y)
    mantle = float(np.sum(-weight[:, None]*relative*system['mantle_velocity_m_s']))
    value, gradient = evaluate(system, y)
    result = dict(basal_dissipation_w=basal, viscous_dissipation_w=viscous,
        other_drag_dissipation_w=other, external_power_w=external, basal_reference_input_power_w=mantle,
        unsolved_power_defect_w=basal+viscous+other-external-mantle,
        generalized_gradient_work_w=float(y@gradient), objective_w=value)
    if not all(np.isfinite(x) for x in result.values()):
        raise RuntimeError('Joint power is outside finite arithmetic range.')
    return result


def constraint_action(system, generalized_velocity_m_s):
    """Original gauge plus length-weighted normal rows; no rank truncation."""
    y = _vector(generalized_velocity_m_s, len(system['load_n']))
    result = np.r_[system['gauge_matrix']@y,
        system['contact_weights']*(system['contact_matrix']@y)]
    if not np.isfinite(result).all():
        raise RuntimeError('Constraint action is outside finite arithmetic range.')
    return result


def constraint_transpose(system, multipliers_n):
    """Exact matching transpose for supplied multipliers, not solved reactions."""
    cut = system['nplate']
    value = _vector(multipliers_n, cut+len(system['contact_weights']))
    result = system['gauge_matrix'].rmatvec(value[:cut])
    result += system['contact_matrix'].rmatvec(system['contact_weights']*value[cut:])
    if not np.isfinite(result).all():
        raise RuntimeError('Constraint transpose is outside finite arithmetic range.')
    return result


def storage_report(system):
    """Deduplicated retained NumPy buffers; excludes Python/object/BLAS overhead."""
    seen = set(); bytes_ = 0; arrays = 0
    def visit(value):
        nonlocal bytes_, arrays
        if isinstance(value, np.ndarray):
            base = value
            while isinstance(base.base, np.ndarray): base = base.base
            if id(base) not in seen:
                seen.add(id(base)); bytes_ += base.nbytes; arrays += 1
        elif isinstance(value, dict):
            for child in value.values(): visit(child)
        elif isinstance(value, (BlockMap, GramOperator)):
            visit(vars(value))
    visit(system)
    count = len(system['load_n'])
    return dict(unknowns=count, vertices=len(system['points']), faces=len(system['faces']),
        retained_numpy_bytes=bytes_, retained_numpy_buffers=arrays,
        hypothetical_dense_hessian_bytes=8*count*count,
        scope='Retained array bytes only; no solve or peak-resident-memory claim.')
