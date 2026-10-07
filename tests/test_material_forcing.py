"""Boundary deformation follows physical spherical belts, not pixel steps."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import material_forcing as forcing
import crustal_structure as columns
import structure_engine as structure
from ridge_geometry import rotate


def xyz(lon, lat):
    lon, lat = np.broadcast_arrays(lon, lat)
    return np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))


def belt(length=1200., values=(12., 4., 9.), owner_a=2, owner_b=3):
    return dict(mid=np.array([[1., 0., 0.]]), normal=np.array([[0., 0., 1.]]),
                length_km=np.array([length]), values=np.array([values]),
                owner_a=np.array([owner_a]), owner_b=np.array([owner_b]))


class MaterialForcingTests(unittest.TestCase):
    def test_physical_profile_peak_cutoff_and_normalization(self):
        distances=np.linspace(-400.,400.,2001)
        points=xyz(np.zeros(len(distances)),distances/forcing.RADIUS_KM)
        actual=forcing.sample_belts(points,np.full(len(points),2),belt())[:,0]
        expected=12.*np.cos(np.pi*distances/800.)**2
        np.testing.assert_allclose(actual,expected,atol=2e-13)
        self.assertAlmostEqual(float(np.trapezoid(actual/12.,distances)),400.,places=10)
        np.testing.assert_array_equal(actual[[0,-1]],0.)

    def test_no_cross_owner_loading_and_unilateral_arc(self):
        points=np.tile([1.,0.,0.],(4,1)); owners=np.array([2,3,4,0])
        actual=forcing.sample_belts(points,owners,belt(owner_a=2,owner_b=2))
        np.testing.assert_array_equal(actual[0],[12.,4.,9.])
        np.testing.assert_array_equal(actual[1:],0.)

    def test_segment_subdivision_and_duplicate_sources_preserve_amplitude(self):
        rng=np.random.default_rng(99)
        points=xyz(rng.uniform(-.14,.14,700),rng.uniform(-.07,.07,700))
        owners=np.full(len(points),2)
        one=belt(length=1600.)
        baseline=forcing.sample_belts(points,owners,one)
        half=deepcopy(one)
        angle=400./forcing.RADIUS_KM
        half['mid']=xyz(np.array([-angle,angle]),np.zeros(2))
        half['normal']=np.repeat(one['normal'],2,axis=0)
        half['length_km']=np.array([800.,800.])
        for name in ('values','owner_a','owner_b'):
            half[name]=np.repeat(one[name],2,axis=0)
        np.testing.assert_allclose(forcing.sample_belts(points,owners,half),baseline,atol=4e-13)
        for name in half: half[name]=np.repeat(half[name],2,axis=0)
        np.testing.assert_allclose(forcing.sample_belts(points,owners,half),baseline,atol=4e-13)

    def test_rotation_poles_seam_and_candidate_index_are_equivariant(self):
        rng=np.random.default_rng(144)
        points=xyz(rng.uniform(-.15,.15,900),rng.uniform(-.07,.07,900))
        owners=rng.integers(2,5,len(points))
        source=belt()
        baseline=forcing.sample_belts(points,owners,source)
        for rotation in ([0.,0.,np.pi],[0.,np.pi/2,0.],[0.,-np.pi/2,0.],[.5,-.8,1.4]):
            transformed={**source,'mid':rotate(source['mid'],rotation),'normal':rotate(source['normal'],rotation)}
            actual=forcing.sample_belts(rotate(points,rotation),owners,transformed)
            np.testing.assert_allclose(actual,baseline,atol=7e-12)
            self.assertTrue(np.isfinite(actual).all())

    def test_vanishing_point_motion_has_vanishing_force_change(self):
        source=belt()
        # Cross a notional latitude-grid edge well inside the physical belt.
        # Nearest-cell lookup is discontinuous here; the belt has no such edge.
        differences=[]
        for epsilon in (1.,.01,.0001):
            points=xyz(np.zeros(2),(137.+np.array([-epsilon,epsilon]))/forcing.RADIUS_KM)
            values=forcing.sample_belts(points,np.array([2,2]),source)
            differences.append(float(np.max(np.abs(values[1]-values[0]))))
        self.assertLess(differences[1],differences[0]*.01001)
        self.assertLess(differences[2],differences[1]*.01001)

    def test_arc_source_uses_existing_rate_and_shifted_overriding_owner(self):
        s=SimpleNamespace(bcode=np.array([4,5,2]),normal_speed=np.array([-18.,21.,-30.]),
            trench_retreat_speed=np.array([0.,0.,4.]),bmid=np.tile([1.,0.,0.],(3,1)),
            bn=np.tile([0.,0.,1.],(3,1)),bl=np.array([100.,200.,300.]),
            bp=np.array([0,0,1]),bq=np.array([1,1,2]),down=np.array([0,0,1]))
        with patch.object(forcing.trench_history,'weights',return_value=np.array([0.,0.,.6])):
            sources=forcing.boundary_belts(s)
        np.testing.assert_array_equal(sources['values'][:2],[[18.,0.,0.],[0.,21.,0.]])
        self.assertAlmostEqual(sources['values'][2,2],34.*1.25*.6)
        self.assertEqual((sources['owner_a'][2],sources['owner_b'][2]),(2,2))
        self.assertAlmostEqual(float(np.arctan2(sources['mid'][2,2],sources['mid'][2,0])*forcing.RADIUS_KM),180.,places=10)

    def test_invalid_geometry_or_negative_rates_are_rejected(self):
        bad=belt(); bad['values'][0,0]=-1.
        with self.assertRaises(ValueError): forcing.sample_belts(np.array([[1.,0.,0.]]),np.array([2]),bad)
        bad=belt(); bad['normal']=bad['mid'].copy()
        with self.assertRaises(ValueError): forcing.sample_belts(np.array([[1.,0.,0.]]),np.array([2]),bad)

    def test_material_pipeline_ignores_pixel_values_and_closes_columns_and_traces(self):
        from tests.test_structure_engine import world, ledger
        s=world(erosion=0.)
        # A real engine boundary supplies one controlled rift. The rest are
        # transforms so there are no foreland or unrelated magmatic sources.
        s.bcode[:]=3; s.bcode[0]=5; s.normal_speed[:]=0.; s.normal_speed[0]=20.
        before=s.trace_relief_m.copy(); volume=s.structure['thickness_km']*s.structure['area_factor']
        twin=deepcopy(s)
        zero=np.zeros(s.n)
        first=structure.deform(s,zero,zero,zero,2.,spherical_boundaries=True)
        second=structure.deform(twin,np.full(s.n,99.),np.full(s.n,71.),np.full(s.n,23.),2.,spherical_boundaries=True)
        self.assertTrue(np.any(first['parcel']['extension_strain']>0))
        for name in structure.FIELDS:
            np.testing.assert_array_equal(s.structure[name],twin.structure[name],err_msg=name)
        np.testing.assert_allclose(s.structure['thickness_km']*s.structure['area_factor'],volume,atol=1e-12)
        np.testing.assert_allclose(columns.elevation(s.structure),structure.material_height(s.kind,s.relief),atol=2e-10)
        np.testing.assert_allclose(s.trace_relief_m-before,ledger(s),atol=2e-10)
        np.testing.assert_array_equal(first['trace']['extension_strain'],second['trace']['extension_strain'])
        extended=first['trace']['extension_strain']>0
        self.assertTrue(np.any(extended))
        self.assertTrue(np.all(s.trace_rift_id[extended]>0))
        np.testing.assert_allclose(s.trace_rift_extension_m,s.trace_extension_m,atol=2e-10)
        np.testing.assert_allclose(np.linalg.norm(s.trace_rift_tangent[extended],axis=1),1.,atol=1e-12)
        np.testing.assert_allclose(np.sum(s.trace_rift_tangent[extended]*s.trace_xyz[extended],axis=1),0.,atol=1e-12)


if __name__=='__main__': unittest.main()
