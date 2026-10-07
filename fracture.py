"""Seeded curved fracture paths on a sphere, with strong cratonic obstacles.

These are geometric worldbuilding heuristics, not a fracture mechanics solver.
Cuts use the same continuous classifier for grids and moving material.
"""
from dataclasses import dataclass, field
import math

import numpy as np


def _unit(value):
    value = np.asarray(value, dtype=float)
    length = np.linalg.norm(value)
    return value / length if length > 1e-10 else np.array([1., 0., 0.])


def component_labels(mask, width, height):
    """Four-neighbour spherical raster components using row runs and union-find.

    Longitude wraps, and polar continuation joins antipodal cells of the same
    cap. The north and south caps are never connected to each other.
    Returns flat int32 labels (-1 outside mask) and the number of components.
    """
    grid = np.asarray(mask, dtype=bool).reshape(height, width)
    labels = np.full((height, width), -1, dtype=np.int32)
    parent = []

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        a, b = find(int(a)), find(int(b))
        if a != b:
            parent[max(a, b)] = min(a, b)

    previous = []
    for y, row in enumerate(grid):
        changes = np.flatnonzero(np.diff(np.r_[False, row, False]))
        runs = []
        for start, end in changes.reshape(-1, 2):
            label = len(parent)
            parent.append(label)
            labels[y, start:end] = label
            runs.append((int(start), int(end), label))
        if len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == width:
            union(runs[0][2], runs[-1][2])
        i = j = 0
        while i < len(previous) and j < len(runs):
            a, b = previous[i], runs[j]
            if max(a[0], b[0]) < min(a[1], b[1]):
                union(a[2], b[2])
            if a[1] <= b[1]:
                i += 1
            else:
                j += 1
        previous = runs
    if not parent:
        return labels.ravel(), 0
    for row in (labels[0], labels[-1]):
        shifted = np.roll(row, width // 2)
        valid = (row >= 0) & (shifted >= 0)
        for a, b in np.unique(np.column_stack((row[valid], shifted[valid])), axis=0):
            union(a, b)
    roots = np.array([find(i) for i in range(len(parent))], dtype=np.int32)
    _, compact = np.unique(roots, return_inverse=True)
    valid = labels >= 0
    labels[valid] = compact[labels[valid]]
    return labels.ravel(), int(compact.max()) + 1


def barriers_from_grid(xyz, crust, width, height, mask=None, area=None):
    """Conservative bounding caps for connected strong cratonic blocks."""
    craton = np.asarray(crust).reshape(-1) == 2
    if mask is not None:
        craton &= np.asarray(mask).reshape(-1)
    labels, count = component_labels(craton, width, height)
    barriers = []
    for label in range(count):
        selected = labels == label
        points = xyz[selected]
        weights = np.ones(len(points)) if area is None else np.asarray(area)[selected]
        center = _unit(np.sum(points * weights[:, None], axis=0))
        # Include the cells' footprint, not just their centres.
        radius = float(np.max(np.arccos(np.clip(points @ center, -1, 1))))
        radius += .55 * math.hypot(math.pi / height, 2 * math.pi / width)
        barriers.append((center, min(math.pi, radius)))
    return barriers


@dataclass
class Fracture:
    center: np.ndarray
    normal: np.ndarray
    tangent: np.ndarray
    knots: np.ndarray
    offsets: np.ndarray
    slopes: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        # Periodic shape-preserving cubic tangents. Unlike an unconstrained
        # spline, these do not overshoot the craton clearance at the knots.
        spacing = (np.roll(self.knots, -1)-self.knots) % (2*math.pi)
        secant = (np.roll(self.offsets, -1)-self.offsets)/spacing
        previous = np.roll(secant, 1)
        before = np.roll(spacing, 1)
        w1, w2 = 2*spacing+before, spacing+2*before
        same_sign = previous*secant > 0
        self.slopes = np.zeros_like(self.offsets)
        self.slopes[same_sign] = ((w1+w2)[same_sign] /
            (w1[same_sign]/previous[same_sign]+w2[same_sign]/secant[same_sign]))

    def offset_at(self, along):
        """Evaluate a continuous, continuously turning periodic fracture curve."""
        along = (np.asarray(along)-self.knots[0]) % (2*math.pi)+self.knots[0]
        index = np.searchsorted(self.knots, along, side='right')-1
        following = (index+1) % len(self.knots)
        spacing = (self.knots[following]-self.knots[index]) % (2*math.pi)
        t = (along-self.knots[index])/spacing
        t2, t3 = t*t, t*t*t
        return ((2*t3-3*t2+1)*self.offsets[index] +
                (t3-2*t2+t)*spacing*self.slopes[index] +
                (-2*t3+3*t2)*self.offsets[following] +
                (t3-t2)*spacing*self.slopes[following])

    def signed_distance(self, xyz):
        """Signed cross-path angle, suitable for a consistent binary cut."""
        points = np.asarray(xyz)
        if points.ndim == 2 and len(points) > 65_536:
            result = np.empty(len(points))
            for start in range(0,len(points),65_536):
                part = points[start:start+65_536]
                along = np.arctan2(part @ self.tangent, part @ self.center)
                across = np.arcsin(np.clip(part @ self.normal,-1,1))
                result[start:start+65_536] = across-self.offset_at(along)
            return result
        along = np.arctan2(points @ self.tangent, points @ self.center)
        across = np.arcsin(np.clip(points @ self.normal, -1, 1))
        return across-self.offset_at(along)


def make_fracture(center, normal, seed, barriers=(), jaggedness=.10):
    """Make a curved crack with several scales of irregularity around cratons.

    Broad bends and smaller meanders imitate inherited mobile belts without
    introducing independent per-cell noise. Rounded periodic deflections clear
    cratonic caps without the old straight-sided rectangular detours.
    Candidate callers must still reject unbalanced or disconnected children.
    """
    center = _unit(center)
    normal = np.asarray(normal, dtype=float)
    normal = _unit(normal - center * np.dot(center, normal))
    if abs(np.dot(center, normal)) > .99:
        normal = _unit(np.cross(center, [0., 0., 1.]) if abs(center[2]) < .9 else np.cross(center, [0., 1., 0.]))
    tangent = _unit(np.cross(normal, center))
    rng = np.random.default_rng(int(seed))
    knots = np.linspace(-math.pi, math.pi, 256, endpoint=False)
    offsets = np.zeros_like(knots)
    for frequency, scale in ((1,.25), (2,.75), (3,.40), (5,.28),
                             (8,.19), (13,.12), (21,.065), (34,.032)):
        offsets += jaggedness*scale*rng.uniform(.75,1.25)*np.sin(
            frequency*knots+rng.uniform(-math.pi,math.pi))
    obstacles = []
    for point, radius in barriers:
        if radius >= 1.35:
            # A cap this broad cannot be routed around in this chart; callers
            # reject cuts through the actual component rather than fracture it.
            continue
        u = float(np.arctan2(np.dot(point, tangent), np.dot(point, center)))
        v = float(np.arcsin(np.clip(np.dot(point, normal), -1, 1)))
        path = float(np.interp(u, knots, offsets, period=2*math.pi))
        if abs(v - path) > radius + jaggedness + .05:
            continue
        # Resolve the exact spherical-cap envelope in this along/across chart.
        # Inflation covers interpolation between the compact curve's knots.
        inflated = min(1.45, radius+.025+math.pi/len(knots))
        delta = (knots-u+math.pi) % (2*math.pi)-math.pi
        a, b = np.cos(v)*np.cos(delta), math.sin(v)
        amplitude = np.sqrt(a*a+b*b)
        intersects = amplitude >= math.cos(inflated)
        angle = np.arccos(np.clip(math.cos(inflated)/np.maximum(amplitude,1e-12),-1,1))
        middle = np.arctan2(b,a)
        lower, upper = middle-angle, middle+angle
        intersects &= (upper >= -math.pi/2) & (lower <= math.pi/2)
        if not np.any(intersects):
            continue
        width = min(1.4, inflated/max(.3,math.cos(v))+.10)
        # A von Mises bump is smooth AND periodic at the spherical seam.
        # Its amplitude is chosen to clear the entire cap; no constant shelf
        # is pasted across the obstacle and the original fine bends survive.
        bump = np.exp((np.cos(delta)-1)/(width*width))
        side = 1 if v >= path else -1
        obstacles.append((intersects, lower, upper, bump, side))
    for _ in range(3):
        for inside, lower, upper, bump, side in obstacles:
            if side > 0:
                amount = max(0.,float(np.max((offsets[inside]-lower[inside])/bump[inside])))
                offsets -= amount*bump
            else:
                amount = max(0.,float(np.max((upper[inside]-offsets[inside])/bump[inside])))
                offsets += amount*bump
    # Only extreme, conflicting barriers approach the chart poles. Smooth
    # saturation avoids introducing flat clipped shelves; callers still reject
    # any candidate that cuts a protected group or fails connectivity/balance.
    magnitude = np.abs(offsets)
    large = magnitude > 1.2
    offsets[large] = np.sign(offsets[large])*(1.2+.3*np.tanh((magnitude[large]-1.2)/.3))
    return Fracture(center, normal, tangent, knots, offsets)
