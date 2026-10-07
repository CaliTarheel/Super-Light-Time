"""Independent source-graph/order oracle, without a preserved implementation."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
import arc_cohort_footprint as grouping


def fixture():
    # No sort ties: the area-weighted centre orders these actual spherical
    # source points 3,4,2,5,1,0,6. Origin3 is the bridge in the source graph.
    angle=np.array([-.030,-.020,-.010,0.,.006,.017,.035])
    points=np.column_stack((np.ones(7),angle,np.zeros(7)))
    points/=np.linalg.norm(points,axis=1)[:,None]
    areas=np.ones(7)
    group=dict(indices=[6,0,5,1,4,2,3],anchor_index=3,links=[[i,i+1] for i in range(6)])
    return points,areas,group


class ArcCohortGroupingTests(unittest.TestCase):
    def test_bridge_removal_rebuilds_induced_components_with_exact_callback_order(self):
        points,areas,group=fixture();before=deepcopy(group);original_points=points.copy()
        point_ids={tuple(value):i for i,value in enumerate(points)};events=[]
        def factory(anchor,amount):
            events.append(('factory',anchor,amount));return {'anchor':anchor}
        def contained(patch_value,queries):
            ids=[point_ids[tuple(value)] for value in queries]
            events.append(('contained',patch_value['anchor'],ids))
            return np.array([i==patch_value['anchor'] for i in ids])
        def capacity(patch_value,anchor,amount):
            events.append(('capacity',anchor,amount));return True
        with patch.object(grouping,'contained',contained):
            result=grouping.local_groups(points,areas,group,factory,capacity)
        order=[3,4,2,5,1,0,6]
        # Hand-derived components after each preceding accepted singleton.
        components=[[0,1,2,3,4,5,6],[4,5,6],[0,1,2],[5,6],[0,1],[0],[6]]
        expected=[]
        for anchor,component in zip(order,components):
            expected.extend([('factory',anchor,float(len(component))),('contained',anchor,component)])
            if len(component)>1:
                expected.extend([('factory',anchor,1.),('contained',anchor,[anchor])])
            expected.append(('capacity',anchor,1.))
        self.assertEqual(events,expected)
        self.assertEqual(result,[dict(indices=[i],anchor_index=i,links=group['links']) for i in order])
        self.assertEqual(group,before);np.testing.assert_array_equal(points,original_points)
        np.testing.assert_array_equal(areas,np.ones(7))

    def test_one_failed_anchor_cannot_reject_another_anchor_at_the_same_total_volume(self):
        points,areas,group=fixture();events=[]
        def factory(anchor,amount):
            events.append(('factory',anchor,amount));return {'anchor':anchor}
        def capacity(patch_value,anchor,amount):
            events.append(('capacity',anchor,amount));return anchor==2
        with patch.object(grouping,'contained',lambda p,q:np.ones(len(q),bool)):
            result=grouping.local_groups(points,areas,group,factory,capacity)
        self.assertEqual(events,[('factory',3,7.),('capacity',3,7.),
                                 ('factory',4,7.),('capacity',4,7.),
                                 ('factory',2,7.),('capacity',2,7.)])
        self.assertEqual(result,[dict(indices=[2,0,1,3,4,5,6],anchor_index=2,links=group['links'])])


if __name__=='__main__':unittest.main()
