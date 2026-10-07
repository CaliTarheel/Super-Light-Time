"""Scalar spherical rectangle oracle independent of production clipping."""
import math
import numpy as np

RADIUS = 6371.


def rectangle(x0, x1, y0, y1):
    """Gnomonic rectangle: every side is an actual great-circle interval."""
    points = np.array([[1., x0/RADIUS, y0/RADIUS],
                       [1., x1/RADIUS, y0/RADIUS],
                       [1., x1/RADIUS, y1/RADIUS],
                       [1., x0/RADIUS, y1/RADIUS]])
    return points / np.linalg.norm(points, axis=1)[:, None]


def polygon_area(points):
    """Independent scalar determinant/solid-angle sum, in square kilometres."""
    points = np.asarray(points)
    total = 0.
    for index in range(1, len(points)-1):
        a, b, c = points[0], points[index], points[index+1]
        determinant = (a[0]*(b[1]*c[2]-b[2]*c[1])
                       - a[1]*(b[0]*c[2]-b[2]*c[0])
                       + a[2]*(b[0]*c[1]-b[1]*c[0]))
        denominator = 1. + sum(a[k]*b[k]+b[k]*c[k]+c[k]*a[k] for k in range(3))
        total += 2.*math.atan2(determinant, denominator)
    return abs(total)*RADIUS**2


def uncovered_rectangle_area(target, covered):
    """Partition at all rectangular limits; count each uncovered tile once.

    This uses interval membership on a gnomonic chart, not spherical polygon
    clipping or sums of overlapping material areas. It therefore independently
    distinguishes a union from a pair-sum, including duplicate sheets.
    """
    x0, x1, y0, y1 = target
    xs = sorted({x0, x1, *(min(x1, max(x0, value)) for r in covered for value in r[:2])})
    ys = sorted({y0, y1, *(min(y1, max(y0, value)) for r in covered for value in r[2:])})
    areas = []
    for left, right in zip(xs, xs[1:]):
        for lower, upper in zip(ys, ys[1:]):
            x, y = (left+right)/2., (lower+upper)/2.
            if not any(a < x < b and c < y < d for a, b, c, d in covered):
                areas.append(polygon_area(rectangle(left, right, lower, upper)))
    return math.fsum(areas)


def material_rectangles(rectangles, *, refine=False, rotation=None):
    vertices, faces = [], []
    for limits in rectangles:
        points = rectangle(*limits)
        start = len(vertices)
        if refine:
            center = points.sum(axis=0)
            points = np.vstack((points, center/np.linalg.norm(center)))
            faces.extend([[start+i, start+(i+1) % 4, start+4] for i in range(4)])
        else:
            faces.extend([[start, start+1, start+2], [start, start+2, start+3]])
        vertices.extend(points)
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    if rotation is not None:
        vertices = vertices @ rotation
    return vertices, np.asarray(faces, dtype=np.int32).reshape(-1, 3)
