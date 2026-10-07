"""Integer homogeneous exact burial clipping matches the Fraction formulation bitwise."""
from fractions import Fraction
import unittest

import numpy as np

import burial_depth
from burial_depth import _positive_binary64_winding
from tests.test_burial_partition_precision import TRIANGLES as SHARED_VERTEX_TRIANGLES, RUN_52_TRIANGLES
from tests.test_burial_roundoff_fan import TRIANGLES as FAN_TRIANGLES


# The former implementation, verbatim apart from its name: an exact rational
# oracle for every sign, crossing value and returned binary64 vertex.
def _fraction_partition(triangles,face,selected,upper):
    """Reclip original represented triangles when repeated cuts lose winding.

    This never reverses a polygon, changes material, or relaxes a conservation
    or interface-work check. Exact homogeneous coordinates retain incidence
    during repeated cuts; 80-digit normalization is needed only on return.
    Both paths use the physical original-edge halfspaces without a tolerance.
    """
    from decimal import Decimal,localcontext

    def dot(a,b):return sum(x*y for x,y in zip(a,b))
    def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
    def unit(a):
        a=[Decimal(v.numerator)/Decimal(v.denominator) for v in a]
        length=dot(a,a).sqrt()
        return [x/length for x in a]
    with localcontext() as context:
        context.prec=80
        def determinant(a,b,c):
            return dot(a,cross(b,c))
        source={int(index):[[Fraction.from_float(float(v)) for v in point] for point in triangles[index]]
                for index in np.r_[face,upper[selected]]}
        original_points={tuple(point):np.asarray(point,float)
                         for triangle in source.values() for point in triangle}
        for triangle in source.values():
            if determinant(*triangle)<=0:
                raise ValueError('Burial partition requires positively wound source triangles.')
            if 1+dot(triangle[0],triangle[1])+dot(triangle[1],triangle[2])+dot(triangle[2],triangle[0])<=0:
                raise ValueError('Burial partition requires minor convex source triangles.')

        def positive(polygon):
            if len(polygon)<3:return False
            signs=[determinant(polygon[0],b,c) for b,c in zip(polygon[1:-1],polygon[2:])]
            if any(value<0 for value in signs):
                raise ValueError('Exact burial clipping did not preserve convex winding.')
            return any(value>0 for value in signs)

        def clip(polygon,plane):
            if len(polygon)<3:return []
            distance=[dot(point,plane) for point in polygon]
            inside=[value>=0 for value in distance]
            output=[]
            for index,point in enumerate(polygon):
                previous=(index-1)%len(polygon)
                if inside[index]!=inside[previous]:
                    fraction=distance[previous]/(distance[previous]-distance[index])
                    crossing=(polygon[previous] if fraction==0 else point if fraction==1 else
                              [a*(1-fraction)+b*fraction for a,b in zip(polygon[previous],point)])
                    output.append(crossing)
                if inside[index]:output.append(point)
            output=[point for index,point in enumerate(output) if point!=output[index-1]]
            # Exact redundant edge vertices have no area. Remove them before
            # binary64 conversion can round a collinear point to the wrong
            # side and invalidate an otherwise large convex region. There is
            # no distance threshold: every nonzero thin region survives.
            while len(output)>=3:
                redundant=next((index for index in range(len(output))
                                if determinant(output[index-1],output[index],output[(index+1)%len(output)])==0),None)
                if redundant is None:break
                del output[redundant]
            return output

        regions=[(source[int(face)],())]
        for pair in selected:
            triangle=source[int(upper[pair])]
            planes=[cross(a,b) for a,b in zip(triangle,triangle[1:]+triangle[:1])]
            next_regions=[]
            for polygon,cover in regions:
                inside=polygon
                for plane in planes:
                    outside=clip(inside,[-v for v in plane])
                    if positive(outside):next_regions.append((outside,cover))
                    inside=clip(inside,plane)
                    if len(inside)<3:break
                if positive(inside):next_regions.append((inside,cover+(int(pair),)))
            regions=next_regions
        result = []
        for polygon, cover in regions:
            # Original binary64 unit vectors define the source geometry and
            # cached area. Do not renormalize them a second time on return.
            represented = np.asarray([original_points[tuple(point)] if tuple(point) in original_points else unit(point)
                                      for point in polygon], float)
            if _positive_binary64_winding(represented):
                result.append((represented, cover))
            else:
                # A tiny positive fan can become inverted at binary64 return
                # while the rest of this exact polygon represents a large
                # valid region. Keep its individually representable fan
                # triangles with the same stack, rather than discard the
                # entire region. Only exact-positive fans are eligible and
                # each still passes the unchanged represented-winding guard.
                # The unchanged whole-footprint gate bounds any lost material
                # below the coordinate representation; no area is repaired.
                for index in range(1,len(polygon)-1):
                    if determinant(polygon[0],polygon[index],polygon[index+1])<=0:
                        continue
                    triangle=represented[[0,index,index+1]]
                    if _positive_binary64_winding(triangle):
                        result.append((triangle,cover))
        return result



def _outcome(function, *args):
    try:
        return function(*args), None
    except ValueError as error:
        return None, str(error)


def _random_stack(rng, covers):
    """Triangles drawn from one small vertex pool: shared vertices and edges,
    collinear crossings and thin slivers occur naturally."""
    centre = rng.normal(size=3); centre /= np.linalg.norm(centre)
    tangent = np.linalg.svd(centre[None, :])[2][1:]
    pool = centre+rng.normal(scale=.01, size=(9, 2))@tangent
    pool /= np.linalg.norm(pool, axis=1)[:, None]
    triangles = []
    while len(triangles) < covers+1:
        a, b, c = pool[rng.choice(len(pool), 3, replace=False)]
        if np.dot(a, np.cross(b, c)) < 0:
            b, c = c, b
        if abs(np.dot(a, np.cross(b, c))) > 1e-9:
            triangles.append([a, b, c])
    return np.array(triangles)


class HomogeneousExactPartitionTests(unittest.TestCase):
    def assertSamePartition(self, triangles, face, selected, upper):
        expected, expected_error = _outcome(_fraction_partition, triangles, face, selected, upper)
        actual, actual_error = _outcome(burial_depth._precise_partition, triangles, face, selected, upper)
        self.assertEqual(actual_error, expected_error)
        if expected is None:
            return
        self.assertEqual(len(actual), len(expected))
        for (polygon, cover), (reference, reference_cover) in zip(actual, expected):
            self.assertEqual(cover, reference_cover)
            self.assertEqual(polygon.dtype, reference.dtype)
            self.assertEqual(polygon.shape, reference.shape)
            self.assertEqual(polygon.tobytes(), reference.tobytes())

    def test_recorded_precision_fixtures(self):
        self.assertSamePartition(SHARED_VERTEX_TRIANGLES, 0, np.arange(9), np.arange(1, 10))
        self.assertSamePartition(RUN_52_TRIANGLES, 0, np.arange(2), np.array([1, 2]))
        self.assertSamePartition(FAN_TRIANGLES, 0, np.array([0]), np.array([1]))
        stacked = np.concatenate((FAN_TRIANGLES, FAN_TRIANGLES[[0]]))
        self.assertSamePartition(stacked, 0, np.array([0, 1]), np.array([2, 1]))

    def test_randomized_shared_vertex_stacks(self):
        rng = np.random.default_rng(20261005)
        for _ in range(60):
            covers = int(rng.integers(1, 6))
            triangles = _random_stack(rng, covers)
            self.assertSamePartition(triangles, 0, np.arange(covers), np.arange(1, covers+1))

    def test_signed_zero_source_vertex_returns_as_the_rational_formulation(self):
        def unit(*v):
            v = np.array(v, float); return v/np.linalg.norm(v)
        lower = np.array([[1., 0., -0.], unit(1., .02, -0.), unit(1., .01, .02)])
        upper = np.array([unit(1., .004, -.01), unit(1., .03, .01), unit(1., -.01, .015)])
        triangles = np.array([lower, upper])
        for t in triangles:
            self.assertGreater(np.dot(t[0], np.cross(t[1], t[2])), 0.)
        self.assertSamePartition(triangles, 0, np.array([0]), np.array([1]))
        regions = burial_depth._precise_partition(triangles, 0, np.array([0]), np.array([1]))
        corner = [polygon for polygon, _ in regions if np.any(np.all(polygon[:, :2] == [1., 0.], axis=1))]
        self.assertTrue(corner)
        self.assertFalse(any(np.signbit(polygon[:, 2]).any() and np.any(polygon[:, 2] == 0.) for polygon in corner))

    def test_rejections_keep_their_messages(self):
        reversed_source = SHARED_VERTEX_TRIANGLES[:, ::-1]
        self.assertSamePartition(reversed_source, 0, np.arange(9), np.arange(1, 10))
        reversed_cover = SHARED_VERTEX_TRIANGLES.copy()
        reversed_cover[3] = reversed_cover[3][::-1]
        self.assertSamePartition(reversed_cover, 0, np.arange(9), np.arange(1, 10))


if __name__ == '__main__':
    unittest.main()
