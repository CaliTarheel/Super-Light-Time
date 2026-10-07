"""Colocated exposed markers and columns share one physical-time support."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import collision_contacts
import collision_surface
import crustal_structure as columns
import mesh_coverage
import structure_engine
from tests.test_collision_surface_acceptance import world, triangle, unit, AIRY


def fixture(erosion=1.):
    lower=triangle();upper=lower.copy();upper[:,1:]*=.45;upper=unit(upper)
    s=world([(upper,35.,1000.,False),(lower,40.,4000.,False)])
    s.n=2;s.parcel_cell=np.arange(2);s.config=dict(erosion=erosion);s.ridge_episodes=[]
    s.bcode=np.empty(0,int);s.normal_speed=np.empty(0);s.geometric_log_area=np.zeros(2)
    s.inversion_uplift_m=np.zeros(2)
    s.trace_xyz=s.pos[[0]].copy();s.trace_patch=s.parcel_patch[[0]].copy()
    s.trace_kind=s.kind[[0]].copy();s.trace_plate=s.parcel_plate[[0]].copy()
    s.trace_relief_m=s.relief[[0]].copy();s.trace_suture=np.zeros(1);s.trace_geometric_log_area=np.zeros(1)
    for name in ('inversion_uplift_m','uplift_m','extension_m','erosion_m','ridge_uplift_m'):
        setattr(s,'trace_'+name,np.zeros(1))
    structure_engine.initialize_traces(s,np.array([0]))
    # Inactive event bookkeeping is inert; columns and geometric exposure are real.
    s._indices=lambda points:np.zeros(len(points),int)
    s._prepare_rift_memory=lambda:None
    s._remember_rift_extension=lambda prefix,loss:None
    s._rift_inversion_gain=lambda prefix,stretch,dt:np.zeros(len(s.trace_xyz) if prefix else len(s.mass))
    s._record_rift_inversion=lambda gain,dt:None
    overlap=mesh_coverage.material_overlaps(s.material_surface['vertices'],s.material_surface['faces'],s.parcel_collision_sheet)
    s._collision_overlap=overlap;s.parcel_exposed_fraction=collision_contacts._exposure(s,overlap)
    collision_surface.refresh(s,overlap)
    return s


class CollisionErosionTimingTests(unittest.TestCase):
    def test_exposed_source_and_marker_share_analytic_pre_erosion_load(self):
        s=fixture();mass=s.mass.copy();before=deepcopy(s.structure);calls=[]
        self.assertEqual(s.parcel_exposed_fraction[0],1.)
        self.assertEqual(collision_contacts.erosion_fraction(s,trace=True)[0],1.)
        self.assertGreater(s.parcel_exposed_fraction[1],0.)
        self.assertLess(s.parcel_exposed_fraction[1],1.)
        original=collision_surface.erosion_support
        def observe(state,trace=False,**kwargs):
            support=original(state,trace=trace,**kwargs)
            calls.append((trace,support.copy(),float(state.structure['thickness_km'][1])))
            return support
        with patch.object(collision_surface,'erosion_support',side_effect=observe):
            budget=structure_engine.deform(s,np.zeros(2),np.zeros(2),np.zeros(2),2.)
        expected=(1000.+40.*AIRY)*(-np.expm1(-2./180.))
        self.assertAlmostEqual(s.structure['denudation_m'][0],expected,places=10)
        self.assertEqual(s.structure['denudation_m'][0],s.trace_structure['denudation_m'][0])
        self.assertEqual(budget['parcel']['net_erosion_m'][0],budget['trace']['net_erosion_m'][0])
        self.assertLess(calls[1][2],calls[0][2])  # The lower load did change between passes.
        np.testing.assert_array_equal(calls[0][1][[0]],calls[1][1])
        self.assertFalse(any(key.startswith('_') for row in budget.values() for key in row))
        np.testing.assert_array_equal(s.mass,mass)
        area=s.material_surface['area_km2']
        removed=(before['thickness_km']-s.structure['thickness_km'])*1000.
        np.testing.assert_allclose(removed,s.structure['denudation_m'],atol=5e-12)
        self.assertAlmostEqual(float(area@(before['thickness_km']-s.structure['thickness_km'])),
                               float(area@s.structure['denudation_m']/1000.),places=8)

    def test_completed_support_is_fresh_and_next_step_keeps_marker_parity(self):
        s=fixture();zero=np.zeros(2)
        for _ in range(2):
            before=s.parcel_collision_support_m.copy()
            structure_engine.deform(s,zero,zero,zero,2.)
            self.assertLess(s.parcel_collision_support_m[0],before[0])
            self.assertAlmostEqual(s.parcel_collision_support_m[0],s.structure['thickness_km'][1]*AIRY,places=8)
            self.assertEqual(s.structure['denudation_m'][0],s.trace_structure['denudation_m'][0])
            self.assertEqual(columns.elevation(s.structure)[0],columns.elevation(s.trace_structure)[0])
            s.t+=2.

    def test_shared_support_uses_current_forced_columns_before_denudation(self):
        s=fixture();before_relief=s.relief.copy();zero=np.zeros(2)
        structure_engine.deform(s,zero,zero,np.array([4.,8.]),2.)
        self.assertGreater(s.structure['added_volume_km_per_reference_km2'][1],0.)
        magma=np.array([4.,8.])*2.*(1.-before_relief/7000.)
        expected=(1000.+magma[0]+40.*AIRY+magma[1])*(-np.expm1(-2./180.))
        self.assertAlmostEqual(s.structure['denudation_m'][0],expected,places=10)
        self.assertEqual(s.structure['denudation_m'][0],s.trace_structure['denudation_m'][0])
        self.assertEqual(columns.elevation(s.structure)[0],columns.elevation(s.trace_structure)[0])

    def test_supplied_support_maps_persistent_patch_ids_and_rejects_invalid_snapshots(self):
        s=fixture();s.parcel_patch=np.array([42,8]);s.trace_patch=np.array([8,42,8])
        s.trace_xyz=s.pos[[1,0,1]].copy();support=np.array([50.,12.])
        with patch.object(collision_surface,'refresh',side_effect=AssertionError('must reuse snapshot')):
            result=collision_surface.erosion_support(s,trace=True,parcel_support_m=support)
        np.testing.assert_array_equal(result,[12.,50.,12.]);result[:]=0.
        np.testing.assert_array_equal(support,[50.,12.])
        for invalid in ([1.],[1.,np.nan],[1.,-1.],[1.,np.inf]):
            with self.assertRaises(ValueError):collision_surface.erosion_support(s,trace=True,parcel_support_m=invalid)
        s.trace_patch[0]=999
        with self.assertRaisesRegex(ValueError,'no material face'):
            collision_surface.erosion_support(s,trace=True,parcel_support_m=support)

    def test_zero_erosion_preserves_columns_and_shared_support(self):
        s=fixture(erosion=0.);before=s.structure['thickness_km'].copy();support=s.parcel_collision_support_m.copy()
        structure_engine.deform(s,np.zeros(2),np.zeros(2),np.zeros(2),2.)
        np.testing.assert_array_equal(s.structure['thickness_km'],before)
        np.testing.assert_array_equal(s.parcel_collision_support_m,support)
        self.assertEqual(s.structure['denudation_m'][0],0.)
        self.assertEqual(s.trace_structure['denudation_m'][0],0.)


if __name__=='__main__':unittest.main()
