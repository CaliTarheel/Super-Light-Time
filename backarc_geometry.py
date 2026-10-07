"""Conservative geometry for a finite arc sliver attached to a real trench.

The inland break follows the supplied spherical trench segments. A small,
seeded width variation supplies curvature without independent cell noise.
Geometry only: no plate state, material, random generator or force is changed.
"""
from dataclasses import dataclass

import numpy as np
import normal_partition

from fracture import component_labels
from material_geometry import patch_centres, splits_protected_groups

RADIUS_KM = 6371.
SECTOR_RADIUS_KM = 2500.
POINT_CHUNK = 4096
EDGE_CHUNK = 128


def _components(s, mask):
    if isinstance(getattr(s, 'native_mesh', None), dict):
        from mesh_geometry import connected_components
        return connected_components(np.asarray(mask, dtype=np.uint8),
                                    s.native_mesh['edge_faces'], mask=np.asarray(mask, bool))
    return component_labels(mask, s.w, s.h)


def _unit(value):
    value = np.asarray(value, float)
    return value/np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-30)


@dataclass
class TrenchStrip:
    """Ephemeral continuous classifier; signed distances are in radians."""
    center: np.ndarray
    direction: np.ndarray
    midpoints: np.ndarray
    normals: np.ndarray
    lengths: np.ndarray
    width_km: float
    phases: np.ndarray

    def width_at(self, points):
        along = np.arctan2(points@np.cross(self.direction, self.center), points@self.center)*RADIUS_KM
        return np.clip(self.width_km*(1.+.11*np.sin(along/450.+self.phases[0])
                                      +.05*np.sin(along/1100.+self.phases[1])), 350., 900.)

    def _strip_distances(self, points):
        inland, front = np.empty(len(points)), np.empty(len(points))
        for offset in range(0, len(points), POINT_CHUNK):
            cells = slice(offset, offset+POINT_CHUNK)
            x = points[cells]
            distance = np.full(len(x), np.inf)
            side = np.zeros(len(x))
            for begin in range(0, len(self.midpoints), EDGE_CHUNK):
                section = slice(begin, begin+EDGE_CHUNK)
                mid, normal = self.midpoints[section], self.normals[section]
                tangent = np.cross(normal, mid)
                half = self.lengths[section]/(2*RADIUS_KM)
                cross = x@normal.T
                along = np.arctan2(x@tangent.T, x@mid.T)
                a = np.cos(half)[:, None]*mid+np.sin(half)[:, None]*tangent
                b = np.cos(half)[:, None]*mid-np.sin(half)[:, None]*tangent
                endpoint = np.arccos(np.clip(np.maximum(x@a.T, x@b.T), -1., 1.))
                angular = np.where(np.abs(along) <= half, np.abs(np.arcsin(np.clip(cross, -1., 1.))), endpoint)
                closest = np.argmin(angular, axis=1)
                candidate = angular[np.arange(len(x)), closest]*RADIUS_KM
                better = candidate < distance
                distance[better] = candidate[better]
                side[better] = cross[np.arange(len(x)), closest][better]
            inland[cells] = self.width_at(x)-distance
            front[cells] = RADIUS_KM*np.arcsin(np.clip(side, -1., 1.))
        return inland, front

    def signed_distance(self, points):
        points = np.asarray(points, float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('Sliver classification needs N by 3 unit-sphere points.')
        center_distance = RADIUS_KM*np.arccos(np.clip(points@self.center, -1., 1.))
        result = (SECTOR_RADIUS_KM-center_distance)/RADIUS_KM
        eligible = np.flatnonzero(center_distance < SECTOR_RADIUS_KM)
        inland, front = self._strip_distances(points[eligible])
        # The supplied normal points into the overriding plate. Restrict
        # the strip to that side, including its shared reconstructed hinge.
        result[eligible] = np.minimum(result[eligible], np.minimum(inland, front)/RADIUS_KM)
        return result

    def rupture_intersects(self, points, radii_km):
        """Footprints touching a new inland/end break, excluding the old trench.

        Test only the portions of each bounding curve actually belonging to
        the sliver; an unrelated craton elsewhere on the locality cap is safe.
        """
        cap = SECTOR_RADIUS_KM-RADIUS_KM*np.arccos(np.clip(points@self.center, -1., 1.))
        near = np.flatnonzero(cap >= -radii_km)
        inland, front = self._strip_distances(points[near])
        radius = radii_km[near]
        return bool(np.any((front >= -radius) &
                    ((np.abs(inland) <= radius) |
                     ((np.abs(cap[near]) <= radius) & (inland >= -radius)))))


def choose_arc_sliver(s, edges, *, seed, width_km=550.):
    """Return a coherent, trench-attached split proposal or ``None``.

    The child and the remaining affected parent component must each stay
    connected. A new child cannot contact third plates. Complete material
    patches move together; original craton groups and rupture-band footprints
    are protected even when the surface map hides their crust.
    """
    if not np.isfinite(width_km) or not 350. <= width_km <= 900.:
        raise ValueError('Back-arc width must be between 350 and 900 km.')
    edges = np.unique(np.asarray(edges, int))
    if not len(edges) or np.any(edges < 0) or np.any(edges >= len(s.ba)):
        return None
    if np.any(~normal_partition.subduction(s)[edges]):
        return None
    down = s.down[edges]
    if np.any((down != s.bp[edges]) & (down != s.bq[edges])):
        return None
    over = np.where(down == s.bp[edges], s.bq[edges], s.bp[edges])
    if len(np.unique(down)) != 1 or len(np.unique(over)) != 1:
        return None
    p, q = int(down[0]), int(over[0])
    if (p == q or not s.active[p] or not s.active[q]
            or np.any(s.plate[s.ba[edges]] != s.bp[edges])
            or np.any(s.plate[s.bb[edges]] != s.bq[edges])):
        return None
    length = np.asarray(s.bl[edges], float)
    if not np.isfinite(length).all() or np.any(length <= 0):
        return None
    midpoint = _unit(s.bmid[edges])
    summed = np.sum(midpoint*length[:, None], axis=0)
    if np.linalg.norm(summed) < .5*length.sum():
        return None
    center = _unit(summed)
    near = midpoint@center > np.cos(SECTOR_RADIUS_KM/RADIUS_KM)
    edges, length, midpoint = edges[near], length[near], midpoint[near]
    if not len(edges):
        return None
    normal = s.bn[edges]*np.where(s.down[edges] == s.bp[edges], 1., -1.)[:, None]
    normal = _unit(normal-midpoint*np.sum(normal*midpoint, axis=1)[:, None])
    direction = np.sum(normal*length[:, None], axis=0)
    direction -= center*np.dot(direction, center)
    if np.linalg.norm(direction) < .5*length.sum():
        return None
    direction = _unit(direction)
    if np.any(normal@direction < .25):
        return None
    phases = np.random.default_rng(int(seed)).uniform(-np.pi, np.pi, 2)
    classifier = TrenchStrip(center, direction, midpoint, normal, length, float(width_km), phases)
    parent = s.plate == q
    region = np.zeros(s.n, bool)
    local = np.flatnonzero(parent & (s.xyz@center > np.cos(SECTOR_RADIUS_KM/RADIUS_KM)))
    region[local] = classifier.signed_distance(s.xyz[local]) > 0
    if not np.any(region) or not np.any(parent & ~region):
        return None
    _, child_count = _components(s, region)
    if child_count != 1:
        return None
    parent_labels, _ = _components(s, parent)
    remaining_labels, _ = _components(s, parent & ~region)
    touched = np.unique(parent_labels[region])
    if any(len(np.unique(remaining_labels[(parent_labels == label) & ~region])) != 1
           or not np.any((parent_labels == label) & ~region) for label in touched):
        return None
    # All new perimeter must meet the old parent or this downgoing plate.
    a, b = s.edge_a, s.edge_b
    perimeter = region[a] != region[b]
    other = np.where(region[a[perimeter]], s.plate[b[perimeter]], s.plate[a[perimeter]])
    if not np.all(np.isin(other, (p, q))) or p not in other or q not in other:
        return None
    frontage = region[s.ba[edges]] | region[s.bb[edges]]
    actual_length = float(length[frontage].sum())
    if actual_length < 300.:
        return None
    owned = np.flatnonzero(s.parcel_plate == q)
    parcel_side = np.zeros(len(s.pos), bool)
    patch_ids, centers, inverse = patch_centres(s.pos[owned], s.mass[owned], s.parcel_patch[owned])
    patch_score = classifier.signed_distance(centers)
    patch_side = patch_score > 0
    parcel_side[owned] = patch_side[inverse]
    # A back-arc split must carry an existing volcanic/continental arc front.
    # Loading an entirely bare ocean sector does not manufacture an arc plate.
    front_material = np.flatnonzero(parcel_side & np.isin(s.kind, (1, 3)) & (s.mass > 0))
    if not len(front_material):
        return None
    _, front_distance = classifier._strip_distances(s.pos[front_material])
    footprint = np.hypot(s.parcel_extent[front_material, 0], s.parcel_extent[front_material, 1])*RADIUS_KM
    if not np.any((front_distance >= -footprint) & (front_distance <= 350.+footprint)):
        return None
    if splits_protected_groups(s.parcel_craton[owned], parcel_side[owned]):
        return None
    # The front itself is an existing trench, not the new rupture. Protect
    # cratonic footprints near the inland break or finite sector end caps.
    craton = owned[s.kind[owned] == 2]
    if len(craton):
        footprint = np.hypot(s.parcel_extent[craton, 0], s.parcel_extent[craton, 1])*RADIUS_KM
        if classifier.rupture_intersects(s.pos[craton], np.maximum(80., footprint)):
            return None
    moved_centers = centers[patch_side]
    if len(moved_centers):
        visible_owner = s.plate[s._indices(moved_centers)]
        if np.any(~np.isin(visible_owner, (p, q))):
            return None
    trace_side = np.zeros(len(s.trace_xyz), bool)
    traced = np.flatnonzero(s.trace_plate == q)
    if len(traced):
        at = np.searchsorted(patch_ids, s.trace_patch[traced])
        if not len(patch_ids) or np.any(at >= len(patch_ids)) or np.any(patch_ids[at] != s.trace_patch[traced]):
            return None
        trace_side[traced] = patch_side[at]
    # Sample the inland margin for review; full geometry remains the classifier.
    distance = classifier.width_at(midpoint)
    path = np.cos(distance/RADIUS_KM)[:, None]*midpoint+np.sin(distance/RADIUS_KM)[:, None]*normal
    order = np.argsort(np.arctan2(path@np.cross(direction, center), path@center), kind='stable')
    if len(order) > 128:
        order = order[np.linspace(0, len(order)-1, 128, dtype=int)]
    return dict(region=region, parcel_side=parcel_side, trace_side=trace_side,
                center=center, direction=direction, length_km=actual_length,
                area_fraction=float(s.cell_area[region].sum()/s.cell_area[parent].sum()),
                path=path[order].tolist(), classifier=classifier,
                width_km=float(width_km), downgoing_plate=p, overriding_plate=q)
