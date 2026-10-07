"""Independent planar-arrangement oracle for competing spherical ridge strips."""
from itertools import combinations
import math
import unittest
import numpy as np
import native_spreading
from tests.test_native_spreading_pairing import scene
from tests._native_spreading_oracle import polygon_area


def turn(points, angle):
    c,s=math.cos(angle),math.sin(angle)
    matrix=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    return np.asarray(points)@matrix.T


def chart(polygon):
    assert np.all(polygon[:,0]>0.)
    points=polygon[:,1:]/polygon[:,:1]
    signed=sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(points,np.roll(points,-1,axis=0)))
    return points if signed>0. else points[::-1]


def intersection_area(polygons):
    """Euclidean line intersections on a gnomonic chart, then solid angle.

    Uses no production sphere clipping, candidate lookup, union or rotation.
    Every chart line corresponds to the polygon's exact great-circle edge.
    """
    subject=chart(polygons[0]).tolist()
    for polygon in polygons[1:]:
        clip=chart(polygon)
        for a,b in zip(clip,np.roll(clip,-1,axis=0)):
            if not subject:return 0.
            edge=b-a
            def side(point):return edge[0]*(point[1]-a[1])-edge[1]*(point[0]-a[0])
            result=[]
            for previous,current in zip(subject[-1:]+subject[:-1],subject):
                before,after=side(previous),side(current)
                inside_before,inside_after=before>=-1e-15,after>=-1e-15
                if inside_before!=inside_after:
                    fraction=before/(before-after)
                    result.append([previous[k]+fraction*(current[k]-previous[k]) for k in (0,1)])
                if inside_after:result.append(current)
            subject=result
    if len(subject)<3:return 0.
    xyz=np.column_stack((np.ones(len(subject)),subject))
    xyz/=np.linalg.norm(xyz,axis=1)[:,None]
    return polygon_area(xyz)


def union_less_union(first,blockers):
    """Exact finite inclusion/exclusion of convex intersections (small fixture)."""
    terms=[]
    for count in range(1,len(first)+1):
        for chosen in combinations(first,count):
            for number in range(len(blockers)+1):
                for blocked in combinations(blockers,number):
                    terms.append((-1.)**(count+1+number)*intersection_area(chosen+blocked))
    return math.fsum(terms)


def strips(axis, offset):
    current=turn(axis,offset)
    shore=turn(current,-.002)
    return np.array([current[0],shore[0],shore[1],current[1]])


def paired_fixture(offset,split=1):
    state,axis=scene(split=split,extent=120.,level=4)
    contour=state.native_boundary_geometry
    a,b=contour['segments_start'].copy(),contour['segments_end'].copy()
    contour['segments_start']=np.r_[a,turn(a,offset)]
    contour['segments_end']=np.r_[b,turn(b,offset)]
    normal=contour['segment_normals'].copy()
    contour['segment_normals']=np.r_[normal,turn(normal,offset)]
    contour['contact_index']=np.zeros(split*2,np.int32)
    return state,axis


class NativeSpreadingOverlapOracleTests(unittest.TestCase):
    def test_duplicate_physical_front_is_union_counted_once(self):
        state,axis=paired_fixture(0.)
        birth=native_spreading.advance(state,state.support.copy(),2.)
        expected=2.*polygon_area(strips(axis,0.))
        self.assertAlmostEqual(float(birth@state.cell_area),expected,delta=1e-5)
        self.assertAlmostEqual(state.spreading_diagnostics['duplicate_paired_area_km2'],expected,delta=1e-5)

    def test_parallel_competing_fronts_exclude_only_local_overlap_and_partners(self):
        values=[]
        for split in (1,2,4):
            state,axis=paired_fixture(.001,split)
            first=[strips(axis,offset) for offset in (0.,.001)]
            blockers=[turn(polygon,angle) for polygon in first for angle in (-.002,.002)]
            expected=2.*union_less_union(first,blockers)
            birth=native_spreading.advance(state,state.support.copy(),2.)
            measured=float(birth@state.cell_area)
            self.assertAlmostEqual(measured,expected,delta=1e-5)
            self.assertGreater(measured,0.)
            self.assertGreater(state.spreading_diagnostics['ambiguous_overlap_excluded_area_km2'],0.)
            self.assertLessEqual(float(birth.max()),1.)
            values.append(measured)
        self.assertLess(max(values)-min(values),1e-5)


if __name__=='__main__':unittest.main()
