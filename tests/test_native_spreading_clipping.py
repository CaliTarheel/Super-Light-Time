"""Finite-edge containment, independent area, and positive saved-budget checks."""
from itertools import product
import json
import math
from pathlib import Path
import unittest

import numpy as np
import native_spreading
from tests._native_spreading_oracle import polygon_area
from tests.test_native_spreading_pairing import scene
from tests.test_native_spreading_overlap_oracle import paired_fixture


def unit(points):
    points = np.asarray(points, float)
    return points / np.linalg.norm(points, axis=-1, keepdims=True)


def thin_triangle(sign=1., reverse=False, near=1.9e-14, far=2.1e-14):
    polygon = np.array([[math.sqrt(1.-near**2-.02**2), -near, -.02],
                        [math.sqrt(1.-far**2-.02**2), -far, .02],
                        [math.sqrt(1.-.01**2), .01, 0.]])
    polygon[:, 1] *= sign
    return (polygon[::-1].copy() if reverse else polygon), [np.array([0., sign, 0.]), unit([-.02, 0., -1.])]


def chart_clip(subject, normals):
    """Independent Euclidean clipping of great circles in a gnomonic chart."""
    current = (subject[:, 1:] / subject[:, :1]).tolist()
    for normal in normals:
        if not current:
            break
        result = []
        for a, b in zip(current[-1:] + current[:-1], current):
            da = normal[0] + normal[1]*a[0] + normal[2]*a[1]
            db = normal[0] + normal[1]*b[0] + normal[2]*b[1]
            if (da >= 0.) != (db >= 0.):
                fraction = da/(da-db)
                result.append([a[k]+fraction*(b[k]-a[k]) for k in (0, 1)])
            if db >= 0.:
                result.append(b)
        current = result
    return unit(np.column_stack((np.ones(len(current)), current))) if current else np.empty((0, 3))


def containment(subject, points):
    if not len(points):
        return 0.
    source, target = subject[:, 1:]/subject[:, :1], points[:, 1:]/points[:, :1]
    cross = lambda a, b: a[0]*b[1]-a[1]*b[0]
    orientation = math.copysign(1., sum(cross(a,b) for a,b in zip(source, np.roll(source,-1,axis=0))))
    return min(orientation*cross(b-a,p-a)/np.linalg.norm(b-a)
               for a,b in zip(source,np.roll(source,-1,axis=0)) for p in target)


def area(points):
    return native_spreading._area(points, dict(radius_km=6371.))


class NativeSpreadingClippingTests(unittest.TestCase):
    def test_actual_10_myr_subtraction_cannot_create_paired_ocean_area(self):
        path = Path(__file__).parent/'fixtures/native_spreading_010_subtraction.json'
        fixture = json.loads(path.read_text())
        source = np.asarray(fixture['source_polygon'])
        blockers = [np.asarray(p) for p in fixture['blockers']]
        expected = polygon_area(source)
        self.assertAlmostEqual(expected,fixture['independent_signed_input_area_km2'],delta=1e-8)
        parts = native_spreading._subtract_polygons(source,blockers,dict(radius_km=fixture['radius_km']))
        total = math.fsum(area(p) for p in parts)
        self.assertLessEqual(total,expected+1e-6)
        self.assertAlmostEqual(total,math.fsum(polygon_area(p) for p in parts),delta=1e-8)
        # Use a local tangent chart to check the actual geographic fixture.
        center = unit(source.sum(axis=0))
        east = unit(np.cross([0.,0.,1.],center))
        basis = np.stack((center,east,np.cross(center,east)))
        for part in parts:
            self.assertGreaterEqual(containment(source@basis.T,part@basis.T),-5e-12)

    def test_tolerance_crossings_stay_inside_source_for_both_orientations(self):
        for sign, reverse in product((-1., 1.), (False, True)):
            source, normals = thin_triangle(sign, reverse)
            clipped = native_spreading._halfspace(source, normals[0])
            with self.subTest(sign=sign, reverse=reverse):
                self.assertGreaterEqual(containment(source, clipped), -5e-16)
                np.testing.assert_allclose(np.linalg.norm(clipped,axis=1), 1., atol=3e-16, rtol=0.)

    def test_sequential_cuts_do_not_create_unsigned_fan_area(self):
        for sign, reverse in product((-1.,1.), (False,True)):
            source, normals = thin_triangle(sign, reverse)
            clipped = source.copy()
            for normal in normals:
                clipped = native_spreading._halfspace(clipped, normal)
            with self.subTest(sign=sign, reverse=reverse):
                self.assertAlmostEqual(area(clipped), polygon_area(chart_clip(source,normals)), delta=1e-9)
                self.assertAlmostEqual(area(clipped), polygon_area(clipped), delta=1e-12)
                self.assertGreaterEqual(containment(source,clipped), -5e-16)

    def test_near_tolerance_grid_and_repeated_cuts_do_not_expand_geometry(self):
        for near,far,sign in product((1e-16,1.1e-14,1.9e-14,1.999e-14), (2.001e-14,2.1e-14,4e-14), (-1.,1.)):
            source,normals = thin_triangle(sign,near=near,far=far)
            clipped = source.copy()
            previous = area(source)
            for i in range(6):
                clipped = native_spreading._halfspace(clipped,normals[i%2])
                with self.subTest(near=near,far=far,sign=sign,cut=i):
                    self.assertGreaterEqual(containment(source,clipped), -5e-16)
                    self.assertLessEqual(area(clipped), previous+1e-8)
                    self.assertAlmostEqual(area(clipped), polygon_area(clipped), delta=1e-8)
                    self.assertLessEqual(len(clipped),4)
                previous = area(clipped)

    def test_real_crossings_match_independent_area_and_complement(self):
        source = unit([[1.,-.01,-.02],[1.,.01,.02],[1.,.02,-.02]])
        normal = np.array([0.,1.,0.])
        halves = [native_spreading._halfspace(source,sign*normal) for sign in (-1.,1.)]
        self.assertAlmostEqual(sum(polygon_area(p) for p in halves),polygon_area(source),delta=1e-8)
        for sign,clipped in zip((-1.,1.),halves):
            self.assertAlmostEqual(area(clipped),polygon_area(chart_clip(source,[sign*normal])),delta=1e-8)
            self.assertGreaterEqual(containment(source,clipped),-5e-16)

    def test_positive_production_and_contested_overlap_pass_saved_schema(self):
        for state,_ in (scene(),paired_fixture(.001)):
            birth = native_spreading.advance(state,state.support.copy(),2.)
            self.assertGreater(float(birth@state.cell_area),0.)
            state.bcode = np.ones(len(state.bp),np.int16)
            state.down = state.bp.copy()
            geometry=state.native_boundary_geometry
            a,b=geometry['segments_start'],geometry['segments_end']
            lengths=6371.*np.arctan2(np.linalg.norm(np.cross(a,b),axis=1),np.sum(a*b,axis=1))
            state.bl=np.bincount(geometry['contact_index'],weights=lengths,minlength=len(state.bp))
            rows = native_spreading.boundary_segments(state)
            lengths = {1:0.,5:0.}
            for row in rows:
                a,b = np.asarray(row['geometry_xyz'])
                if row['code'] in lengths:
                    lengths[row['code']] += 6371.*math.atan2(float(np.linalg.norm(np.cross(a,b))),float(a@b))
            frame = dict(native_spreading_version=1,spreading_diagnostics=state.spreading_diagnostics,
                boundary_segments=rows,native_boundary_owner_a=state.bp,native_boundary_owner_b=state.bq,
                stats=dict(ridge_length_km=lengths[1],rift_length_km=lengths[5]))
            native_spreading.validate_frame(frame)
            invalid = dict(frame,spreading_diagnostics=dict(state.spreading_diagnostics))
            invalid['spreading_diagnostics']['generated_area_km2'] += 1.
            with self.assertRaisesRegex(ValueError,'budget does not close'):
                native_spreading.validate_frame(invalid)


if __name__ == '__main__':
    unittest.main()
