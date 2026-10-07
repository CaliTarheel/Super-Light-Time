"""Sparse spherical loading response for progressive continental rifting.

This is a kinematic, relative-strength membrane closure, not a calculation of
stress in pascals. Boundary motion proxies drive a graph of material nodes;
strong links share motion and weak links allow it to localize. A weighted
least-squares Euler rotation is removed from the driving on each connected
component, so ordinary rigid plate motion cannot manufacture rift loading.

The response minimizes a positive quadratic consisting of soft boundary
constraints, parallel-transported neighbour differences, and a weak positive
regularizer. It is solved in the tangent, rigid-rotation-free subspace using
matrix-free preconditioned conjugate gradients. Storage and work per iteration
are O(nodes + edges); the only dense matrices are 3 by 3 Euler fits. Nothing
here moves material, creates a plate, or supplies an opening kick.
"""
from __future__ import annotations

import numpy as np
import material_acceleration

RADIUS_KM = 6371.


def areal_strain_rate(xyz, edges, edge_extension, *, radius_km=RADIUS_KM):
    """Infer signed areal strain from finite spherical edge separation.

    ``edge_extension`` is the signed along-edge relative speed in km/Myr,
    including compression. At each node, a weighted local least-squares fit
    reconstructs the symmetric tangent strain tensor from its axial samples:
    extension / edge_length = direction.T @ strain_tensor @ direction.
    Its trace is the area-change rate (1/Myr), not the largest positive axial
    strain. Thus shear, and extension balanced by perpendicular shortening,
    cannot thin an otherwise area-preserving column.

    Finite Euler rotation has exactly zero edge separation, avoiding spurious
    curvature terms from directly differencing ambient velocity vectors.
    No smoothing or filling of elevations takes place. ``supported`` reports
    whether the available directions determine the tensor trace; a dangling
    one-dimensional link cannot establish the missing transverse strain and
    returns zero. The small local matrices use the Frobenius-orthonormal tensor
    coordinates (E11, sqrt(2)*E12, E22), so the fit and support test are invariant
    under a change of tangent basis or rotation of the globe.
    """
    xyz = np.asarray(xyz, dtype=float)
    raw_edges = np.asarray(edges)
    extension = np.asarray(edge_extension, dtype=float)
    if (xyz.ndim != 2 or xyz.shape[1:] != (3,) or not np.isfinite(xyz).all()
            or np.any(np.abs(np.linalg.norm(xyz, axis=1)-1.) > 1e-6)):
        raise ValueError('Areal strain positions must be finite unit vectors with shape (N,3).')
    if (raw_edges.ndim != 2 or raw_edges.shape[1:] != (2,)
            or raw_edges.dtype.kind not in 'iu' or np.any(raw_edges < 0)
            or np.any(raw_edges >= len(xyz))):
        raise ValueError('Areal strain edges must be valid integer endpoint pairs.')
    if extension.shape != (len(raw_edges),) or not np.isfinite(extension).all():
        raise ValueError('Supply one finite signed extension speed per edge.')
    if not np.isscalar(radius_km) or not np.isfinite(radius_km) or radius_km <= 0:
        raise ValueError('Sphere radius must be positive and finite.')
    count = len(xyz)
    if not len(raw_edges):
        return dict(rate=np.zeros(count), supported=np.zeros(count, bool),
                    tensor_rank=np.zeros(count, np.int8))
    xyz = xyz/np.linalg.norm(xyz, axis=1, keepdims=True)
    a, b = raw_edges.T
    cosine = np.clip(np.sum(xyz[a]*xyz[b], axis=1), -1., 1.)
    sine = np.linalg.norm(np.cross(xyz[a], xyz[b]), axis=1)
    if np.any(sine < 1e-10):
        raise ValueError('Adjacent areal strain nodes must be distinct and not antipodal.')
    length = float(radius_km)*np.arctan2(sine, cosine)
    directions = np.concatenate(((xyz[b]-cosine[:, None]*xyz[a])/sine[:, None],
                                 (xyz[a]-cosine[:, None]*xyz[b])/sine[:, None]))
    nodes = np.r_[a, b]
    reference = np.where((np.abs(xyz[:, 2]) < .9)[:, None], [0., 0., 1.], [1., 0., 0.])
    first = np.cross(reference, xyz)
    first /= np.linalg.norm(first, axis=1, keepdims=True)
    second = np.cross(xyz, first)
    u = np.sum(directions*first[nodes], axis=1)
    v = np.sum(directions*second[nodes], axis=1)
    design = np.column_stack((u*u, np.sqrt(2.)*u*v, v*v))
    # Nearer neighbours give the local derivative more weight. The scale of
    # all weights at one node cancels in its least-squares reconstruction.
    weight = np.tile(1./length**2, 2)
    axial_rate = np.tile(extension/length, 2)
    normal = np.empty((count, 3, 3))
    rhs = np.empty((count, 3))
    for i in range(3):
        rhs[:, i] = np.bincount(nodes, weights=weight*design[:, i]*axial_rate, minlength=count)
        for j in range(i, 3):
            normal[:, i, j] = normal[:, j, i] = np.bincount(
                nodes, weights=weight*design[:, i]*design[:, j], minlength=count)
    eigenvalues, vectors = np.linalg.eigh(normal)
    retained = eigenvalues > np.maximum(eigenvalues[:, -1:], 1e-300)*1e-8
    inverse = np.divide(1., eigenvalues, out=np.zeros_like(eigenvalues), where=retained)
    tensor = np.einsum('nij,nj->ni', (vectors*inverse[:, None, :]) @ vectors.transpose(0, 2, 1), rhs)
    # A rank-two orthogonal cross can determine trace without determining
    # shear. Test that particular observable, rather than requiring full rank.
    projector = (vectors*retained[:, None, :]) @ vectors.transpose(0, 2, 1)
    observable = projector[:, :, 0]+projector[:, :, 2]
    supported = np.linalg.norm(observable-np.array([1., 0., 1.]), axis=1) < 1e-7
    rate = np.where(supported, tensor[:, 0]+tensor[:, 2], 0.)
    return dict(rate=rate, supported=supported,
                tensor_rank=np.count_nonzero(retained, axis=1).astype(np.int8))


def _components(n, edges):
    """Union-find keeps disconnected material mechanically independent."""
    parent = np.arange(n, dtype=np.int64)
    size = np.ones(n, dtype=np.int64)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        a, b = find(a), find(b)
        if a == b:
            continue
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]
    roots = np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)
    return np.unique(roots, return_inverse=True)[1]


class _EulerProjection:
    """Fit independent Euler fields using a batched small normal matrix."""

    def __init__(self, xyz, labels, weights):
        self.xyz, self.labels, self.weights = xyz, labels, weights
        self.count = int(labels.max()) + 1 if len(labels) else 0
        normal = np.zeros((self.count, 3, 3), dtype=np.float64)
        total = np.bincount(labels, weights=weights, minlength=self.count)
        for i in range(3):
            for j in range(i, 3):
                value = -np.bincount(labels, weights=weights*xyz[:, i]*xyz[:, j],
                                     minlength=self.count)
                if i == j:
                    value += total
                normal[:, i, j] = normal[:, j, i] = value
        eigenvalues, vectors = np.linalg.eigh(normal)
        cutoff = np.maximum(eigenvalues[:, -1:], 1e-300)*1e-12
        inverse = np.divide(1., eigenvalues, out=np.zeros_like(eigenvalues),
                            where=eigenvalues > cutoff)
        self.inverse = (vectors*inverse[:, None, :]) @ vectors.transpose(0, 2, 1)

    def fit(self, velocity):
        moment = np.cross(self.xyz, velocity)*self.weights[:, None]
        rhs = np.column_stack([
            np.bincount(self.labels, weights=moment[:, j], minlength=self.count)
            for j in range(3)])
        return np.einsum('nij,nj->ni', self.inverse, rhs)

    def remove(self, velocity):
        return velocity - np.cross(self.fit(velocity)[self.labels], self.xyz)


def solve_loading(xyz, edges, strength, loads, *, iterations=80,
                  load_weights=None, coupling_km=800., regularization=.002,
                  tolerance=1e-7, radius_km=RADIUS_KM):
    """Return the bounded-iteration deformation response of a material graph.

    ``xyz`` is (N,3) unit-sphere position, ``edges`` is (E,2) undirected unique
    adjacency, ``strength`` is positive relative node stiffness, and ``loads``
    is (N,3) velocity-derived driving in km/Myr. The small radial part of a load
    is projected away. Reversing an edge does not reverse its extension sign:
    positive means its two endpoints separate, negative means they shorten.

    Optional nonnegative ``load_weights`` sets the soft driving constraints;
    use zero for unloaded interior nodes and approximately one on a loaded
    boundary. With no weights specified, every supplied load is a constraint.
    ``coupling_km`` controls spatial sharing, with inverse-square edge length
    and symmetric degree normalization. It is an effective mechanical length,
    not lithospheric elastic thickness. The positive regularizer weakly anchors
    unconstrained deformation modes and makes the linear solve well posed.

    Velocity is tangent and excludes component-wide rigid rotation. Returned
    edge extension is the signed great-circle along-edge relative speed; shear
    is the absolute transverse parallel-transport difference. Both are km/Myr,
    not strain or stress. Convergence is reported even when the iteration cap
    is reached; a finite approximate result is returned without hidden retries.
    Input arrays are not modified.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    loads = np.asarray(loads, dtype=np.float64)
    strength = np.asarray(strength, dtype=np.float64)
    raw_edges = np.asarray(edges)
    if xyz.ndim != 2 or xyz.shape[1:] != (3,) or not np.isfinite(xyz).all():
        raise ValueError('xyz must have shape (N,3) with finite unit vectors.')
    n = len(xyz)
    norms = np.linalg.norm(xyz, axis=1)
    if np.any(np.abs(norms-1.) > 1e-6):
        raise ValueError('xyz positions must be unit vectors.')
    # Tolerate routine roundoff while retaining exact tangent projections.
    xyz = xyz / norms[:, None]
    if loads.shape != (n, 3) or not np.isfinite(loads).all():
        raise ValueError('loads must have shape (N,3) and be finite.')
    if strength.shape != (n,) or not np.isfinite(strength).all() or np.any(strength <= 0):
        raise ValueError('strength must contain one finite positive value per node.')
    if raw_edges.ndim != 2 or raw_edges.shape[1:] != (2,):
        raise ValueError('edges must have shape (E,2).')
    if raw_edges.dtype.kind not in 'iu' and len(raw_edges):
        raise ValueError('Edge indices must be integers.')
    edges = raw_edges.astype(np.int64, copy=False)
    if np.any(edges < 0) or np.any(edges >= n):
        raise ValueError('Edge indices must refer to existing nodes.')
    if np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError('Self edges are not material adjacency.')
    if (isinstance(iterations, (bool, np.bool_)) or int(iterations) != iterations
            or not 1 <= iterations <= 10000):
        raise ValueError('iterations must be an integer from 1 to 10000.')
    iterations = int(iterations)
    for name, value in (('coupling_km', coupling_km), ('regularization', regularization),
                        ('tolerance', tolerance), ('radius_km', radius_km)):
        if np.ndim(value) != 0 or not np.isfinite(value) or float(value) <= 0:
            raise ValueError(f'{name} must be one finite positive scalar.')
    weights = np.ones(n) if load_weights is None else np.asarray(load_weights, dtype=np.float64)
    if weights.shape != (n,) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError('load_weights must contain one finite nonnegative value per node.')

    def tangent(v):
        return v - xyz*np.sum(xyz*v, axis=1, keepdims=True)

    # Only tiny local dense matrices are used, including for isolated nodes.
    labels = _components(n, edges)
    rigid = _EulerProjection(xyz, labels, np.ones(n))
    loaded_rigid = _EulerProjection(xyz, labels, weights)
    loads = tangent(loads)
    removed_euler = loaded_rigid.fit(loads)
    driving = tangent(loads - np.cross(removed_euler[labels], xyz))

    def project(v):
        return rigid.remove(tangent(v))

    a, b = edges.T
    cosine = np.clip(np.sum(xyz[a]*xyz[b], axis=1), -1., 1.)
    cross = np.cross(xyz[a], xyz[b])
    sine = np.linalg.norm(cross, axis=1)
    if np.any(sine < 1e-10):
        raise ValueError('Adjacent nodes must be distinct and not antipodal.')
    direction_a = (xyz[b]-cosine[:, None]*xyz[a])/sine[:, None]
    direction_b = (cosine[:, None]*xyz[b]-xyz[a])/sine[:, None]
    transverse = cross/sine[:, None]
    length = float(radius_km)*np.arctan2(sine, cosine)
    degree = np.bincount(edges.ravel(), minlength=n)
    low, high = np.minimum(strength[a], strength[b]), np.maximum(strength[a], strength[b])
    harmonic = low*(2./(1.+low/high))
    stiffness = harmonic*(float(coupling_km)/length)**2*2./(degree[a]+degree[b])
    if not np.isfinite(stiffness).all():
        raise ValueError('Strength or adjacency scale exceeds floating-point range.')
    diagonal = (float(regularization)+weights
                + np.bincount(a, weights=stiffness, minlength=n)
                + np.bincount(b, weights=stiffness, minlength=n))

    def differences(v):
        # The two geodesic tangent bases are exact parallel transports; their
        # shared transverse vector is perpendicular to the connecting plane.
        along = np.sum(v[b]*direction_b, axis=1)-np.sum(v[a]*direction_a, axis=1)
        across = np.sum((v[b]-v[a])*transverse, axis=1)
        return along, across

    native_operator = material_acceleration.prepare_operator(
        a, b, direction_a, direction_b, transverse, stiffness, float(regularization)+weights)

    def operator(v):
        if native_operator is not None:
            return project(native_operator(v))
        along, across = differences(v)
        result = (float(regularization)+weights)[:, None]*v
        force_a = -stiffness[:, None]*(along[:, None]*direction_a+across[:, None]*transverse)
        force_b = stiffness[:, None]*(along[:, None]*direction_b+across[:, None]*transverse)
        # bincount is substantially cheaper than repeated scatter-at for this
        # fixed-width vector accumulation and creates no node-by-node matrix.
        for axis in range(3):
            result[:, axis] += np.bincount(a, weights=force_a[:, axis], minlength=n)
            result[:, axis] += np.bincount(b, weights=force_b[:, axis], minlength=n)
        return project(result)

    rhs = project(weights[:, None]*driving)
    rhs_norm = float(np.linalg.norm(rhs))
    load_norm = float(np.linalg.norm(weights[:, None]*loads))
    velocity = np.zeros((n, 3), dtype=np.float64)
    residual = rhs.copy()
    used = 0
    # Euler-fit roundoff must not become visible stress on a rigidly moving
    # plate. This is relative to the actual supplied driving, not a speed floor.
    rigid_only = rhs_norm <= max(1e-300, load_norm*2e-11)
    if not rigid_only:
        preconditioned = project(residual/diagonal[:, None])
        search = preconditioned.copy()
        rz = float(np.sum(residual*preconditioned))
        for used in range(1, iterations+1):
            applied = operator(search)
            denominator = float(np.sum(search*applied))
            if denominator <= 0 or not np.isfinite(denominator):
                used -= 1
                break
            alpha = rz/denominator
            velocity += alpha*search
            residual -= alpha*applied
            if np.linalg.norm(residual) <= float(tolerance)*rhs_norm:
                break
            preconditioned = project(residual/diagonal[:, None])
            new_rz = float(np.sum(residual*preconditioned))
            if new_rz <= 0 or not np.isfinite(new_rz):
                break
            search = preconditioned + (new_rz/rz)*search
            rz = new_rz
        velocity = project(velocity)
    residual_norm = float(np.linalg.norm(rhs-operator(velocity)))
    relative_residual = residual_norm/rhs_norm if rhs_norm > 0 else 0.
    extension, shear = differences(velocity)
    return {
        'velocity': velocity,
        'edge_extension': extension,
        'edge_shear': np.abs(shear),
        'edge_length_km': length,
        'iterations': used,
        'converged': bool(rigid_only or relative_residual <= float(tolerance)*1.01),
        'relative_residual': 0. if rigid_only else relative_residual,
        'residual_norm': residual_norm,
        'rhs_norm': rhs_norm,
        'removed_omega_radians_myr': removed_euler/float(radius_km),
        'component_count': rigid.count,
        'node_count': n,
        'edge_count': len(edges),
    }
