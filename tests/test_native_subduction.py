"""Actual finite capture, history admission and native transport integration."""
from copy import deepcopy
import unittest
import numpy as np
import native_subduction
import native_processes
import trench_history
from tests.test_native_subduction_oracle import fixture, strip_area
from tests._native_spreading_oracle import material_rectangles


def history_scene():
    s,axis,_=fixture()
    s.t=20.;s.age=np.full(s.n,80.)
    s.bmid=np.array([axis.sum(axis=0)/np.linalg.norm(axis.sum(axis=0))])
    s.bn=np.array([[0.,1.,0.]])
    s.bl=np.array([6371.*np.arccos(axis[0]@axis[1])])
    s.trench_systems=[dict(id=7,phase='mature',downgoing_plate_uid=10,overriding_plate_uid=11,
        plate_uids=[10,11],geometry_xyz=s.bmid.tolist(),last_seen_myr=20.,maturity=.75)]
    return s


def transport_scene(level=3):
    s,_,_=fixture(level=level)
    s.bmid=np.array([[1.,0.,0.]]);s.bn=np.array([[0.,1.,0.]])
    s.t=0.;s.age=80.+20.*s.xyz[:,2];s.ocean_relief=40.*s.xyz[:,0]
    s.primordial_fraction=np.full(s.n,.8)
    s.ocean_history=dict(damage=.3+.2*s.xyz[:,0])
    s.process_totals['ocean_consumed_km2']=0.
    s.centres=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    s.backarc_basins=[]
    return s


class NativeSubductionTests(unittest.TestCase):
    def test_duplicate_overlapping_fronts_use_unique_water_and_maximum_maturity(self):
        baseline,axis,transported=fixture()
        expected=native_subduction.removal(baseline,transported,2.)
        for reverse in (False,True):
            s,_,transported=fixture()
            for key in ('ba','bb','bp','bq','down','bcode','normal_speed'):
                setattr(s,key,np.repeat(getattr(s,key),2))
            s.trench_maturity=np.array([1.,.25] if reverse else [.25,1.])
            contour=s.native_boundary_geometry
            for key in ('segments_start','segments_end','segment_normals'):
                contour[key]=np.repeat(contour[key],2,axis=0)
            contour['contact_index']=np.array([0,1])
            result=native_subduction.removal(s,transported,2.)
            np.testing.assert_allclose(result,expected,atol=2e-14,rtol=1e-11)
            self.assertAlmostEqual(s.native_subduction_diagnostics['duplicate_capture_area_removed_km2'],strip_area(axis,2.),delta=1e-5)

    def test_true_unrelated_receiver_owner_cannot_be_captured(self):
        s,_,transported=fixture(level=4)
        baseline=native_subduction.removal(s,transported,2.)
        candidate=int(np.argmax(baseline[0]*s.cell_area))
        self.assertGreater(baseline[0,candidate],0.)
        s.plate[candidate]=2
        result=native_subduction.removal(s,transported,2.)
        self.assertEqual(result[:,candidate].sum(),0.)
        np.testing.assert_allclose(np.delete(result, candidate,axis=1),np.delete(baseline,candidate,axis=1),atol=2e-14)
        self.assertGreater(s.native_subduction_diagnostics['rejected_unrelated_owner_water_area_km2'],0.)

    def test_multiple_incoming_owners_share_actual_excess_simultaneously(self):
        results=[]
        for reverse in (False,True):
            s,_,transported=fixture()
            s.plate[:]=2
            for key in ('ba','bb','bcode','normal_speed'):
                setattr(s,key,np.repeat(getattr(s,key),2))
            s.bp=np.array([1,0] if reverse else [0,1]);s.bq=np.array([2,2]);s.down=s.bp.copy()
            s.trench_maturity=np.ones(2)
            s.omega=np.array([[0.,0.,.001],[0.,0.,.001],[0.,0.,-.001]])
            contour=s.native_boundary_geometry
            for key in ('segments_start','segments_end','segment_normals'):
                contour[key]=np.repeat(contour[key],2,axis=0)
            contour['contact_index']=np.array([0,1])
            transported[0]=.4;transported[1]=.4;transported[2]=.200001
            removed=native_subduction.removal(s,transported,2.)
            self.assertGreater(removed.sum(),0.)
            np.testing.assert_allclose(removed[0],removed[1],atol=1e-16)
            np.testing.assert_array_equal(removed[2],0.)
            self.assertLessEqual(float(removed.sum(axis=0).max()),1e-6+2e-16)
            self.assertAlmostEqual(float(removed[0].max()),.5e-6,delta=1e-15)
            results.append(removed)
        np.testing.assert_allclose(*results,atol=1e-16)

    def test_history_preserves_actual_partial_incoming_water_despite_center_label(self):
        s=history_scene();s.crust[:]=1
        vertices,faces=material_rectangles([(-200.,200.,0.,900.)])
        s.material_surface=dict(vertices=vertices,faces=faces,face_owner=np.zeros(len(faces),int))
        self.assertAlmostEqual(native_subduction.edge_ocean_fraction(s,0)[0],.5,places=10)
        trench_history.prepare(s)
        self.assertEqual(s.trench_id[0],7)
        self.assertEqual(s.trench_maturity[0],.75)
        # A real continent covering the whole incoming front shuts admission;
        # assigning that same physical continent to the upper owner permits it.
        vertices,faces=material_rectangles([(-200.,200.,-900.,900.)])
        s.material_surface=dict(vertices=vertices,faces=faces,face_owner=np.zeros(len(faces),int))
        trench_history.prepare(s)
        self.assertEqual(s.trench_maturity[0],0.)
        s.material_surface['face_owner'][:]=1
        trench_history.prepare(s)
        self.assertEqual(s.trench_maturity[0],.75)

    def test_real_transport_removal_matches_independently_observed_stage_deltas(self):
        for level in (3,4):
            s=transport_scene(level);column_before=deepcopy(s.material_surface)
            native_processes.advect_ocean(s,2.)
            d=s.native_subduction_diagnostics
            ledger=s.ocean_diagnostics['support_moment_ledger'];stages=ledger['stages']
            self.assertGreater(d['removed_area_km2'],0.)
            self.assertAlmostEqual(s.process_totals['ocean_consumed_km2'],d['removed_area_km2'],delta=1e-7)
            before=stages['after_transport_and_conditional_reconstruction'];after=stages['after_measured_subduction']
            delta=np.asarray(before['owner_support_area_km2'])-after['owner_support_area_km2']
            np.testing.assert_allclose(delta,d['removed_area_by_owner_km2'],atol=2e-7)
            self.assertLess(abs(ledger['subduction_area_residual_km2']),2e-7)
            self.assertLess(np.max(np.abs(ledger['subduction_moment_residuals'])),1e-5)
            self.assertAlmostEqual(d['available_total_overlap_km2'],s.ocean_diagnostics['raw_overlap_km2'],delta=1e-7)
            np.testing.assert_allclose(s.support.sum(axis=0),1.,atol=2e-16)
            for key,value in column_before.items():np.testing.assert_array_equal(s.material_surface[key],value)

    def test_actual_transport_is_independent_of_categorical_crust_and_arrivals(self):
        first=transport_scene();second=deepcopy(first);second.crust[:]=1
        native_processes.advect_ocean(first,2.);native_processes.advect_ocean(second,2.)
        np.testing.assert_array_equal(first.support,second.support)
        self.assertEqual(first.native_subduction_diagnostics,second.native_subduction_diagnostics)
        s,_,transported=fixture()
        a=native_subduction.removal(s,transported,2.,arrivals=np.zeros_like(transported))
        b=native_subduction.removal(s,transported,2.,arrivals=np.full_like(transported,100.))
        np.testing.assert_array_equal(a,b)

    def test_unversioned_native_uses_historical_operator(self):
        s=transport_scene();del s.native_subduction_version
        s.bmid=np.array([[1.,0.,0.]])
        native_processes.advect_ocean(s,2.)
        self.assertFalse(hasattr(s,'native_subduction_diagnostics'))
        self.assertNotIn('overlap_basis',s.ocean_diagnostics)


if __name__=='__main__':unittest.main()
