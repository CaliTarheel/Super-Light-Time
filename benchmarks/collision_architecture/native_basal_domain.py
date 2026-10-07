"""Offline native-to-finite basal allocation, with complete control coverage.

No native state, support, saved policy, or solver is changed. The explicitly
selected resolved-material-bottom policy is a new geometric owner allocation.
Every positive region needs a unique saved bottom; tiny unordered overlaps do
not inherit column_density's averaged-order exception. See the contract.
"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math

import numpy as np

import mesh_coverage
import mesh_geometry
import plate_balance
from convex_partition import Edge
from . import basal_partition as partition
from . import finite_basal_operator as finite
from .framed_basal_map import FramedBlockMap


def edge_distances(points, a, b):
    """Broadcast original-edge signs using the existing exact Edge predicate.

    The translated determinant and conservative unshifted-product filter are
    the same arithmetic as Edge.distances. Only clearly resolved entries use
    this vectorized path; ambiguous entries delegate to that established
    predicate, including its unrepresentable-nonzero rejection. Exact stored
    endpoints/zero vectors are incidences. There is no geometric inside band.
    This adapter consumes signs only, never these magnitudes as a new metric.
    """
    arrays = [np.asarray(value, float) for value in (points, a, b)]
    if any(value.ndim < 1 or value.shape[-1] != 3 for value in arrays):
        raise ValueError('Basal edge predicates require three-dimensional coordinates.')
    p, a, b = np.broadcast_arrays(*arrays)
    if any(not np.isfinite(value).all() for value in (p,a,b)):
        raise ValueError('Basal edge predicates require finite coordinates.')
    result = np.einsum('...i,...i->...', a, np.cross(b-a,p-a))
    scale = np.sum(abs(a)*(abs(b[..., [1,2,0]]*p[..., [2,0,1]])
                         +abs(b[..., [2,0,1]]*p[..., [1,2,0]])),axis=-1)
    incident = np.all(p==a,axis=-1)|np.all(p==b,axis=-1)|np.all(p==0.,axis=-1)
    uncertain = ~incident&(abs(result)<=64*np.finfo(float).eps*scale)
    result = np.where(incident,0.,result)
    if np.any(uncertain):
        flat=result.reshape(-1); aa=a.reshape(-1,3); bb=b.reshape(-1,3); pp=p.reshape(-1,3)
        edges={}
        for index in np.flatnonzero(uncertain):
            key=tuple(aa[index])+tuple(bb[index])
            if key not in edges: edges[key]=Edge(aa[index],bb[index])
            flat[index]=edges[key].distances(pp[index:index+1])[0]
    return result


class NativeBasalBlocker(ValueError):
    """A positive native region cannot be assigned without changing its law."""

    def __init__(self, message, witness):
        super().__init__(message)
        self.witness = witness


def _array(value, kind, name, ndim=None):
    result = np.asarray(value)
    if (result.dtype.kind not in kind or (result.dtype.kind == 'f' and result.dtype.itemsize > 8)
            or (ndim is not None and result.ndim != ndim) or not np.isfinite(result).all()):
        raise ValueError(f'{name} has an invalid type, shape or nonfinite value.')
    result = result.copy()
    result.setflags(write=False)
    return result


def _snapshot(s):
    material, control = s.material_surface, s.native_mesh
    result = {key: _array(value, kind, key, ndim) for key, value, kind, ndim in (
        ('points', material['vertices'], 'fiu', 2), ('faces', material['faces'], 'iu', 2),
        ('face_ids', material['face_id'], 'iu', 1), ('face_owner', material['face_owner'], 'iu', 1),
        ('vertex_owner', material['vertex_owner'], 'iu', 1),
        ('sheets', s.parcel_collision_sheet, 'iu', 1), ('parcel_owner', s.parcel_plate, 'iu', 1),
        ('parcel_ids', s.parcel_patch, 'iu', 1),
        ('control_points', control['vertices'], 'fiu', 2), ('control_faces', control['faces'], 'iu', 2),
        ('saved_area_m2', s.cell_area, 'fiu', 1),
        ('support', s.support, 'fiu', 2), ('crust', s.crust, 'iu', 1),
        ('active', s.active, 'b', 1), ('plate_uid', s.plate_uid, 'iu', 1))}
    # Preserve native fractional support and the exact _viscous coefficient.
    beta = plate_balance.ASTHENOSPHERE_DRAG_PA_S_M*np.asarray(plate_balance.KEEL)[
        np.clip(result['crust'], 0, len(plate_balance.KEEL)-1)]
    result['beta'] = _array(beta, 'fiu', 'native basal coefficient', 1)
    result['saved_area_m2'] = _array(result['saved_area_m2']*1e6, 'fiu', 'native SI area', 1)
    for key, value in [('radius_m', material['radius_km']),
                       ('control_radius_m', control['radius_km'])]:
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int,float,np.integer,np.floating))
                or not np.isscalar(value) or not np.isfinite(value) or value <= 0):
            raise ValueError('Native sphere radii must be finite and positive.')
        result[key] = float(value)*1000.
        if not np.isfinite(result[key]): raise ValueError('Native SI radius is not finite.')
    pairs = []
    for row in s.collision_contacts:
        pairs.append(tuple(partition._identifier(row[name], name) for name in ('top_sheet', 'under_sheet')))
    result['above'] = tuple(sorted(set(pairs)))
    return result


def _digest(data):
    h = hashlib.sha256()
    for key in sorted(data):
        h.update(key.encode())
        value = data[key]
        if isinstance(value, np.ndarray):
            h.update(str((value.shape, value.dtype.str)).encode()); h.update(value.tobytes())
        else:
            h.update(repr(value).encode())
    return h.hexdigest()


def _triangles(points, faces, name):
    if (points.shape[1:] != (3,) or faces.shape[1:] != (3,) or not len(faces)
            or np.any(faces >= len(points)) or np.any(faces < 0)
            or np.any(abs(np.linalg.norm(points, axis=1)-1.) > 64*np.finfo(float).eps)):
        raise ValueError(f'{name} requires indexed unit spherical triangles.')
    triangle = np.asarray(points[faces], float)
    if np.any(edge_distances(triangle[:, 2], triangle[:, 0], triangle[:, 1]) <= 0.):
        raise ValueError(f'{name} has nonpositive represented-coordinate winding.')
    centre = triangle.sum(axis=1); centre /= np.linalg.norm(centre, axis=1)[:, None]
    if np.any(np.einsum('fvi,fi->fv', triangle, centre) <= 1e-10):
        raise ValueError(f'{name} lies outside the resolved open-hemisphere locator domain.')
    return triangle, centre


def _control_certificate(points, faces, triangle, centre):
    # A closed positive spherical 2-chain has constant integer multiplicity
    # off its edges. Opposite indexed edges cancel exactly; an exact off-edge
    # one-cover witness establishes multiplicity one everywhere (a.e.).
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    signs = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    if np.any(counts != 2) or np.any(np.bincount(inverse, weights=signs) != 0):
        raise ValueError('Native controls must form an exactly closed, oppositely oriented edge chain.')
    probes = list(centre[:8])+[np.array([1., .3141592653589793, .2718281828459045])]
    for probe in probes:
        distances = np.stack([edge_distances(probe, triangle[:, k], triangle[:, (k+1)%3])
                              for k in range(3)], axis=1)
        if np.any(distances == 0.):
            continue
        multiplicity = int(np.count_nonzero(np.all(distances > 0., axis=1)))
        if multiplicity != 1:
            raise ValueError('Closed native control geometry does not cover the sphere exactly once.')
        return dict(method='closed positive spherical chain and exact off-edge multiplicity',
                    control_count=len(faces), edge_count=len(counts),
                    witness=probe.tolist(), multiplicity=multiplicity,
                    global_control_coverage=True, boundary_measure_zero=True)
    raise ValueError('Could not establish an off-edge native coverage witness.')


def _positive_candidates(control, triangles, candidates):
    """Reject only exact separating halfspaces, then exact zero chart area."""
    candidates = np.asarray(candidates, dtype=np.int64)
    if not len(candidates):
        return candidates
    other = triangles[candidates]
    separated = np.zeros(len(candidates), bool)
    for k in range(3):
        separated |= np.all(edge_distances(other, control[k], control[(k+1)%3]) < 0., axis=1)
        separated |= np.all(edge_distances(control, other[:, k, None], other[:, (k+1)%3, None]) < 0., axis=1)
    _, basis = partition._triangle(control, 'Control cell')
    reference = [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))]
    positive = []
    for index in candidates[~separated]:
        _, rational = partition._triangle(triangles[index], f'Material face index {index}')
        planes = [tuple(partition._dot(partition._cross(a, b), p) for p in basis)
                  for a, b in zip(rational, rational[1:]+rational[:1])]
        inside, _ = partition._partition(reference, planes)
        if inside and partition._area(inside) > 0:
            positive.append(int(index))
    return np.asarray(positive, dtype=np.int64)


def _fresh_locator(points, faces, *, bin_resolution=None):
    """Native bins for regular faces, mandatory global list for small faces.

    The native locator's inverse-map admission uses |det|>=1e-14, although
    candidate enumeration needs no inverse. Preserve every strictly positive
    smaller face in an always-considered list rather than imposing that floor.
    """
    unit = points/np.linalg.norm(points, axis=1)[:,None]
    tri = unit[faces]
    determinant = np.einsum('ij,ij->i', tri[:,0], np.cross(tri[:,1],tri[:,2]))
    regular = np.flatnonzero(abs(determinant) >= 1e-14)
    exceptional = np.flatnonzero(abs(determinant) < 1e-14)
    locator = mesh_geometry.build_locator(points, faces[regular], bin_resolution=bin_resolution)
    locator['candidates'] = regular[locator['candidates']]
    locator['global_faces'] = np.unique(np.r_[regular[locator['global_faces']], exceptional])
    locator['face_count'] = len(faces)
    # Candidate-only index. An inverse on a strict subset must not be mistaken
    # for an original-face point locator by another consumer.
    locator.pop('inverse'); locator.pop('faces'); locator.pop('face_vertices')
    locator['candidate_only'] = True
    return locator


def _order_failure_witness(control, records, graph):
    """Recover an exact positive atomic cover only on an ordering failure."""
    _, basis = partition._triangle(control, 'Control cell')
    pieces = [([(Fraction(0),Fraction(0)),(Fraction(1),Fraction(0)),(Fraction(0),Fraction(1))],())]
    sheets = {record['face_id']:record['sheet_id'] for record in records}
    for record in records:
        _, rational = partition._triangle(record['triangle'], 'Material face')
        planes = [tuple(partition._dot(partition._cross(a,b), p) for p in basis)
                  for a,b in zip(rational,rational[1:]+rational[:1])]
        updated = []
        for poly, cover in pieces:
            inside, outside = partition._partition(poly, planes)
            updated.extend((p,cover) for p in outside)
            if inside: updated.append((inside,cover+(record['face_id'],)))
        pieces = updated
    for poly, cover in pieces:
        for i, first in enumerate(cover):
            for second in cover[i+1:]:
                a,b = sheets[first],sheets[second]
                if a == b or (b in graph.get(a,())) == (a in graph.get(b,())):
                    return dict(unordered_face_ids=[first,second],unordered_sheets=[a,b],
                        exact_chart_area=str(partition._area(poly)),
                        exact_chart_polygon=[[str(x),str(y)] for x,y in poly])
    return {}


class _BoundFiniteBasal(finite.FiniteBasal):
    def __init__(self, system, design, reference, diagnostics, domain):
        super().__init__(system, design, reference, diagnostics)
        self._domain = domain

    def validate_system(self, system):
        self._domain.validate_source()
        self._domain.validate_system(system)
        super().validate_system(system)


@dataclass(frozen=True)
class NativeBasalBuild:
    component: finite.FiniteBasal
    certificate: dict


class NativeBasalDomain:
    """Detached geometry/support snapshot, bound to its unchanged source state."""

    def __init__(self, s, policy):
        if policy not in ('preserve-native', 'resolved-material-bottom'):
            raise ValueError('An explicit reviewed basal allocation policy is required.')
        self._source, self.allocation_policy = s, policy
        self._data = d = _snapshot(s)
        self.source_signature = _digest(d)
        self.points, self.faces, self.face_ids = d['points'], d['faces'], d['face_ids']
        self.active_native_slots = np.flatnonzero(d['active'])
        self.active_native_slots.setflags(write=False)
        self.owner_to_plate_slot = {int(owner): slot for slot, owner in enumerate(self.active_native_slots)}
        self.face_id_to_index = {int(face): index for index, face in enumerate(self.face_ids)}
        self.radius_m = d['radius_m']
        nf, nv, nc, np_slots = len(self.faces), len(self.points), len(d['control_faces']), len(d['active'])
        if (not len(self.active_native_slots) or len(self.face_id_to_index) != nf
                or any(np.any(d[key] < 0) for key in ('face_ids','face_owner','vertex_owner','sheets','parcel_ids','plate_uid'))
                or d['radius_m'] != d['control_radius_m'] or d['plate_uid'].shape != (np_slots,)
                or len(np.unique(d['plate_uid'][self.active_native_slots])) != len(self.active_native_slots)
                or any(d[key].shape != (nf,) for key in ('face_owner','sheets','parcel_owner','parcel_ids'))
                or d['vertex_owner'].shape != (nv,) or d['support'].shape != (np_slots, nc)
                or any(d[key].shape != (nc,) for key in ('saved_area_m2','crust','beta'))
                or np.any(d['saved_area_m2'] <= 0.) or np.any(d['beta'] <= 0.)
                or np.any(d['support'] < 0.) or np.any(d['support'][~d['active']] != 0.)
                or np.any(abs(d['support'].sum(axis=0)-1.) > 64*np.finfo(float).eps)
                or not np.array_equal(d['parcel_owner'], d['face_owner'])
                or not np.array_equal(d['parcel_ids'], self.face_ids)):
            raise ValueError('Native geometry, active ownership, IDs, support or saved measures are inconsistent.')
        if any(int(owner) not in self.owner_to_plate_slot for owner in np.unique(d['vertex_owner'])):
            raise ValueError('Material vertices refer to an inactive native plate.')
        self._triangles, _ = _triangles(self.points, self.faces, 'Material')
        self._controls, self._centres = _triangles(d['control_points'], d['control_faces'], 'Control')
        if np.any(d['vertex_owner'][self.faces] != d['face_owner'][:, None]):
            raise ValueError('A material face has inconsistent vertex ownership.')
        self.vertex_plate = np.array([self.owner_to_plate_slot[int(p)] for p in d['vertex_owner']], dtype=np.int64)
        self.vertex_plate.setflags(write=False)
        self.control_certificate = _control_certificate(d['control_points'], d['control_faces'], self._controls, self._centres)
        self._graph = partition._order(d['above'])
        # Fresh index; cached positive coverage and its area floor are unused.
        self._locator = _fresh_locator(self.points, self.faces)
        for value in self._locator.values():
            if isinstance(value, np.ndarray): value.setflags(write=False)
        self._chords = np.max(np.linalg.norm(self._controls-self._centres[:, None], axis=2), axis=1)
        self._prepared_signature = self._derived_signature()

    def _derived_signature(self):
        # Derived maps are part of physical identity, including ocean-only
        # slots. Accidentally changing one must not silently move basal drag.
        data = dict(points=self.points, faces=self.faces, face_ids=self.face_ids,
                    active_native_slots=self.active_native_slots, vertex_plate=self.vertex_plate,
                    radius_m=self.radius_m, policy=self.allocation_policy,
                    material_triangles=self._triangles, control_triangles=self._controls,
                    control_centres=self._centres, control_chords=self._chords,
                    owner_map=tuple(sorted(self.owner_to_plate_slot.items())),
                    face_map=tuple(sorted(self.face_id_to_index.items())),
                    graph=tuple((k,tuple(sorted(v))) for k,v in sorted(self._graph.items())),
                    control_certificate=repr(self.control_certificate))
        data.update({'locator_'+key:value for key,value in self._locator.items()})
        return _digest(data)

    def validate_source(self, source=None):
        if _digest(_snapshot(self._source if source is None else source)) != self.source_signature:
            raise ValueError('Native basal source geometry, order, support or coefficient changed.')
        if _digest(self._data) != self.source_signature:
            raise ValueError('Prepared native basal data changed.')
        if self._derived_signature() != self._prepared_signature:
            raise ValueError('Prepared native basal candidate geometry or identity mapping changed.')

    def validate_system(self, system):
        if (system['plate_count'] != len(self.active_native_slots) or system['nplate'] != 3*len(self.active_native_slots)
                or system['radius_m'] != self.radius_m
                or any(not np.array_equal(system[key], value) for key, value in
                       [('points', self.points), ('faces', self.faces), ('vertex_plate', self.vertex_plate)])):
            raise ValueError('Native basal system has different geometry or active-slot ordering.')

    def candidates(self, control_id):
        self.validate_source()
        return self._candidates(control_id)

    def _candidates(self, control_id):
        index = partition._identifier(control_id, 'control ID')
        if index >= len(self._controls): raise ValueError('Control ID is outside the native mesh.')
        broad = mesh_coverage._candidates(None, self._centres[index], self._chords[index], self._locator)
        return _positive_candidates(self._controls[index], self._triangles, broad), len(broad)

    def _cell(self, index):
        selected, count = self._candidates(index)
        d = self._data
        records = [dict(face_id=int(self.face_ids[i]), sheet_id=int(d['sheets'][i]),
                        owner=int(d['face_owner'][i]), triangle=self._triangles[i]) for i in selected]
        witness = dict(control_id=int(index), broadphase_candidate_count=count,
                       positive_candidate_face_ids=self.face_ids[selected].tolist(),
                       positive_candidate_sheets=d['sheets'][selected].tolist(),
                       positive_candidate_native_owners=d['face_owner'][selected].tolist())
        try:
            allocation = partition.partition_cell(self._controls[index], records, d['above'],
                radius_m=self.radius_m, saved_cell_area_m2=float(d['saved_area_m2'][index]),
                support={int(owner):float(d['support'][owner,index]) for owner in self.active_native_slots},
                basal_drag_pa_s_per_m=float(d['beta'][index]), allocation_policy=self.allocation_policy)
        except ValueError as error:
            if 'ordered bottom' in str(error) or 'one material sheet' in str(error):
                witness.update(_order_failure_witness(self._controls[index], records, self._graph))
            raise NativeBasalBlocker(f'Native basal control {index}: {error}', witness) from error
        # A unique bottom alone could hide an unordered pair of upper layers.
        # Require an actual total order on every exact positive atomic region.
        sheets = {record['face_id']:record['sheet_id'] for record in records}
        for piece in allocation['pieces']:
            cover = piece['cover_faces']
            for i, first in enumerate(cover):
                for second in cover[i+1:]:
                    a, b = sheets[first], sheets[second]
                    if (b in self._graph.get(a, ())) == (a in self._graph.get(b, ())):
                        detail = dict(witness, unordered_face_ids=[first,second], unordered_sheets=[a,b],
                                      exact_chart_area=str(piece['chart_area']), polygon=piece['polygon'].tolist())
                        raise NativeBasalBlocker('Positive native basal region has no unique saved layer order.', detail)
        allocation['control_id'] = int(index)
        allocation['candidate_witness'] = witness
        return allocation

    def probe_cells(self, control_ids):
        """Explicit partial diagnostic only; never returned as a global build."""
        self.validate_source()
        ids = [partition._identifier(i, 'control ID') for i in control_ids]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('Probe control IDs must be nonempty and unique.')
        allocations = [self._cell(i) for i in ids]
        self.validate_source()
        return dict(scope='partial control-cell diagnostic; not a global operator',
                    global_allocation_complete=False, control_ids=ids, allocations=allocations)

    def build(self, system, *, mantle_omega_rad_s, quadrature_relative_tolerance, max_order):
        """Visit EVERY unique native control and build one finite basal operator.

        No subset argument is accepted. A single unresolved positive region
        aborts without publishing an operator. Batches bound partition scratch;
        retained positive local factors and their diagnostics scale with pieces.
        """
        self.validate_source(); self.validate_system(system)
        blocks, columns, references, diagnostics = [], [], [], []
        frames_hi, frames_lo = [], []
        seen_faces = set(); cell_areas = []; largest_closure = 0.; piece_count = 0
        deltas = {owner:np.zeros((3,3)) for owner in self.owner_to_plate_slot}
        batch = []
        def append_batch():
            try:
                component = finite.build(system, batch, self.face_id_to_index,
                    owner_to_plate_slot=self.owner_to_plate_slot, mantle_omega_rad_s=mantle_omega_rad_s,
                    quadrature_relative_tolerance=quadrature_relative_tolerance, max_order=max_order)
            except (ValueError, RuntimeError) as error:
                raise NativeBasalBlocker(f'Native finite basal integration: {error}',
                    dict(control_ids=[a['control_id'] for a in batch], phase='finite trace quadrature')) from error
            blocks.append(component.basal_design.blocks); columns.append(component.basal_design.columns)
            frames_hi.append(component.basal_design.frame_hi); frames_lo.append(component.basal_design.frame_lo)
            references.append(component.reference_factor_sqrt_w)
            diagnostics.extend(dict(row, control_id=batch[row['cell']]['control_id']) for row in component.diagnostics)
        for index in range(len(self._controls)):
            allocation = self._cell(index)
            for piece in allocation['pieces']:
                seen_faces.update(piece['cover_faces'])
            cell_areas.append(math.fsum(piece['area_m2'] for piece in allocation['pieces']))
            largest_closure = max(largest_closure, allocation['scaled_metric_closure_error'],
                                  allocation['scaled_common_rotation_closure_error'])
            for owner, delta in allocation['per_owner_legacy_metric_change_m2'].items(): deltas[owner] += delta
            piece_count += len(allocation['pieces']); batch.append(allocation)
            if len(batch) == 16: append_batch(); batch = []
        if batch: append_batch()
        if seen_faces != set(self.face_id_to_index):
            raise ValueError('Global native basal partition omitted positive material faces.')
        total_area = math.fsum(self._data['saved_area_m2'])
        area_sum = math.fsum(cell_areas)
        if abs(area_sum-total_area) > 2e-10*total_area:
            raise ValueError('Global native basal pieces do not close the complete saved measure.')
        design_blocks = np.concatenate(blocks); design_columns = np.concatenate(columns)
        count = len(design_blocks)
        design = FramedBlockMap(design_blocks, np.arange(9*count).reshape(count,9), design_columns,
                          (9*count, len(system['load_n'])), np.concatenate(frames_hi), np.concatenate(frames_lo))
        component = _BoundFiniteBasal(system, design, np.concatenate(references), diagnostics, self)
        self.validate_source()
        certificate = dict(self.control_certificate, global_allocation_complete=True,
            source_signature=self.source_signature, allocation_policy=self.allocation_policy,
            active_native_slots=self.active_native_slots.tolist(), material_faces_covered=len(seen_faces),
            positive_piece_count=piece_count, native_area_m2=total_area, area_closure_error_m2=area_sum-total_area,
            maximum_cell_metric_closure=largest_closure,
            per_owner_legacy_metric_change_m2={str(k):v.tolist() for k,v in deltas.items()},
            candidate_policy='fresh native cap locator including global_faces; exact positive halfspaces; no area floor')
        return NativeBasalBuild(component, certificate)


def prepare(s, *, allocation_policy):
    return NativeBasalDomain(s, allocation_policy)
