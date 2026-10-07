"""Measured supply limits close without changing the accepted geometric sink."""
from copy import deepcopy
import math
import unittest
import numpy as np

import native_subduction as subduction
from tests.test_native_subduction_oracle import fixture,strip_area


def scene(*,maturity=1.,memory=False,geometry=0):
    s,axis,values=fixture(level=2,maturity=maturity)
    s.t=0.;s.trench_id=np.array([7]);s.native_subduction_supply_version=1
    s.native_subduction_geometry_version=geometry
    if memory:
        s.slab_memory_version=2;s.age=np.full(s.n,80.)
        s.trench_systems=[dict(id=7)]
    return s,axis,values


class SubductionSupplyLedgerTests(unittest.TestCase):
    def measure(self,s,values):
        baseline=deepcopy(s);del baseline.native_subduction_supply_version
        before=values.copy()
        expected=subduction.removal(baseline,values,2.)
        actual=subduction.removal(s,values,2.)
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(values,before)
        # Every previously emitted diagnostic, including actual feed and neck
        # attribution, must remain byte-for-byte numerically unchanged.
        original=deepcopy(s.native_subduction_diagnostics);ledger=original.pop('supply_ledger')
        self.assertEqual(original,baseline.native_subduction_diagnostics)
        s.t=2.
        frame=dict(subduction.snapshot_metadata(s),time_myr=2.)
        subduction.validate_frame(frame)
        for name in subduction.SUPPLY_AREAS:
            self.assertAlmostEqual(ledger['totals'][name],math.fsum(row[name] for row in ledger['fronts']),delta=1e-7)
            self.assertAlmostEqual(ledger['totals'][name],math.fsum(row[name] for row in ledger['trenches']),delta=1e-7)
        totals=ledger['totals']
        self.assertAlmostEqual(totals['maturity_weighted_water_area_km2'],
            totals['realized_removal_area_km2']+sum(totals[key] for key in
            ('donor_reduction_area_km2','local_overlap_reduction_area_km2','competition_reduction_area_km2')),delta=1e-7)
        return actual,ledger,frame

    def test_unlimited_source_matches_independent_spherical_water_and_old_feed(self):
        for geometry in (0,1):
            with self.subTest(geometry=geometry):
                s,axis,values=scene(maturity=.5,memory=True,geometry=geometry)
                _,ledger,_=self.measure(s,values)
                row=ledger['fronts'][0];expected=strip_area(axis,2.)
                self.assertAlmostEqual(row['raw_geometric_source_area_km2'],expected,delta=1e-5)
                self.assertAlmostEqual(row['admitted_unique_water_area_km2'],expected,delta=1e-5)
                self.assertAlmostEqual(row['maturity_weighted_water_area_km2'],expected*.5,delta=1e-5)
                self.assertAlmostEqual(row['realized_removal_area_km2'],expected*.5,delta=1e-5)
                self.assertEqual(row['source_trench_id'],7)
                self.assertEqual(row['source_downgoing_plate_uid'],10)
                self.assertEqual(row['source_overriding_plate_uid'],11)
                self.assertAlmostEqual(sum(row[key] for key in subduction.SUPPLY_AREAS[6:9]),0.,delta=1e-7)
                geometry=row['source_geometry']
                np.testing.assert_array_equal(geometry['segment_start_xyz'],s.native_boundary_geometry['segments_start'])
                self.assertGreater(geometry['length_km'],0.)
                self.assertEqual(row['source_nominal_maturity'],.5)
                self.assertEqual(row['source_parent_normal_speed_km_myr'],s.normal_speed[0])

    def test_donor_support_and_local_overlap_are_distinct_caps(self):
        # Positive shared overlap is ample, but almost no incoming donor exists.
        s,_,values=scene();values[0]=1e-7;values[1]=2.
        removed,ledger,_=self.measure(s,values);totals=ledger['totals']
        active=removed[0]>0
        self.assertGreater(totals['donor_reduction_area_km2'],0.)
        self.assertAlmostEqual(totals['local_overlap_reduction_area_km2'],0.,delta=1e-7)
        self.assertEqual(totals['competition_reduction_area_km2'],0.)
        self.assertAlmostEqual(totals['realized_removal_area_km2'],float(s.cell_area[active].sum())*1e-7,delta=1e-7)
        # Both owners have ample donor area, but only a tiny shared excess exists.
        s,_,values=scene(maturity=.5);values[0]=.4;values[1]=.600001
        removed,ledger,_=self.measure(s,values);totals=ledger['totals']
        self.assertAlmostEqual(totals['donor_reduction_area_km2'],0.,delta=1e-7)
        self.assertGreater(totals['local_overlap_reduction_area_km2'],0.)
        self.assertEqual(totals['competition_reduction_area_km2'],0.)
        np.testing.assert_allclose(removed[0,removed[0]>0],.5*(.4+.600001-1.),atol=1e-17)

    def test_shared_competition_reduces_each_request_after_local_caps(self):
        s,_,values=scene();s.plate[:]=2
        for key in ('ba','bb','bcode','normal_speed'):
            setattr(s,key,np.repeat(getattr(s,key),2))
        s.bp=np.array([0,1]);s.bq=np.array([2,2]);s.down=s.bp.copy()
        s.trench_id=np.array([7,8]);s.trench_maturity=np.ones(2)
        s.omega=np.array([[0.,0.,.001],[0.,0.,.001],[0.,0.,-.001]])
        contour=s.native_boundary_geometry
        for key in ('segments_start','segments_end','segment_normals'):
            contour[key]=np.repeat(contour[key],2,axis=0)
        contour['contact_index']=np.array([0,1])
        values[0]=.4;values[1]=.4;values[2]=.200001
        removed,ledger,_=self.measure(s,values)
        np.testing.assert_array_equal(removed[0],removed[1])
        self.assertEqual(len(ledger['trenches']),2)
        for row in ledger['fronts']:
            self.assertGreater(row['local_overlap_reduction_area_km2'],0.)
            self.assertAlmostEqual(row['realized_removal_area_km2']*2.,row['requested_after_local_overlap_cap_km2'],delta=1e-7)
            self.assertAlmostEqual(row['competition_reduction_area_km2'],row['realized_removal_area_km2'],delta=1e-7)

    def test_no_feed_distinguishes_absent_geometry_from_unavailable_overlap(self):
        for reason in ('opening','immature','no_overlap'):
            with self.subTest(reason=reason):
                s,_,values=scene(memory=True)
                if reason=='opening':s.omega*=-1.;s.normal_speed*=-1.
                elif reason=='immature':s.trench_maturity[:]=0.
                else:values[:]=0.;values[0]=.4;values[1]=.6
                removed,ledger,_=self.measure(s,values)
                np.testing.assert_array_equal(removed,0.)
                self.assertEqual(len(ledger['fronts']),1)
                row=ledger['fronts'][0]
                self.assertEqual(row['source_trench_id'],7)
                if reason=='no_overlap':
                    self.assertGreater(row['maturity_weighted_water_area_km2'],0.)
                    self.assertGreater(row['local_overlap_reduction_area_km2'],0.)
                else:self.assertEqual(row['raw_geometric_source_area_km2'],0.)

    def test_duplicate_fronts_preserve_chosen_source_and_geometry_witnesses(self):
        for geometry in (0,1):
            with self.subTest(geometry=geometry):
                s,_,values=scene(memory=True,geometry=geometry)
                for key in ('ba','bb','bp','bq','down','bcode','normal_speed'):
                    setattr(s,key,np.repeat(getattr(s,key),2))
                s.trench_id=np.array([7,8]);s.trench_systems=[dict(id=7),dict(id=8)]
                s.trench_maturity=np.array([.25,1.])
                for key in ('segments_start','segments_end','segment_normals'):
                    s.native_boundary_geometry[key]=np.repeat(s.native_boundary_geometry[key],2,axis=0)
                s.native_boundary_geometry['contact_index']=np.array([0,1])
                _,ledger,_=self.measure(s,values)
                low,high=ledger['fronts']
                self.assertEqual(low['realized_removal_area_km2'],0.)
                self.assertGreater(high['realized_removal_area_km2'],0.)
                self.assertEqual(high['source_trench_id'],8)
                if geometry:
                    self.assertEqual(high['geometry_witnesses'],[
                        dict(source_parent_index=0,source_trench_id=7),
                        dict(source_parent_index=1,source_trench_id=8)])
                else:
                    self.assertGreater(low['duplicate_water_removed_km2'],0.)
                    self.assertEqual(low['admitted_unique_water_area_km2'],0.)

    def test_unrelated_receiver_rejection_is_measured_separately(self):
        s,_,values=scene()
        old=deepcopy(s);del old.native_subduction_supply_version
        before=subduction.removal(old,values,2.)
        cell=int(np.argmax(before[0]*s.cell_area));s.plate[cell]=2
        removed,ledger,_=self.measure(s,values)
        self.assertEqual(removed[:,cell].sum(),0.)
        self.assertGreater(ledger['fronts'][0]['rejected_unrelated_owner_water_area_km2'],0.)
        self.assertAlmostEqual(ledger['totals']['duplicate_water_removed_km2'],0.,delta=1e-7)

    def test_endpoint_splits_and_owner_relabeling_cannot_reassign_source_feed(self):
        s,_,values=scene(memory=True);_,ledger,_=self.measure(s,values)
        before=deepcopy(ledger)
        # These are endpoint identities, after source-step feeding. The saved
        # record must remain attached to source parent0/trench7/plateUID10.
        s.trench_id[:]=19;s.plate_uid[:]=[110,111,112]
        s.trench_systems=[dict(id=19,parent_id=7)]
        s.native_boundary_geometry['segments_start']*=-1.
        s.trench_maturity[:]=.1;s.normal_speed[:]=0.
        frame=dict(subduction.snapshot_metadata(s),time_myr=2.)
        self.assertEqual(frame['native_subduction_diagnostics']['supply_ledger'],before)
        subduction.validate_frame(frame)

    def test_initial_and_historical_records_do_not_invent_measured_history(self):
        s,_,values=scene();initial=subduction.snapshot_metadata(s)
        self.assertEqual(initial['native_subduction_supply_version'],1)
        self.assertEqual(initial['native_subduction_diagnostics']['supply_ledger']['fronts'],[])
        self.assertEqual(sum(initial['native_subduction_diagnostics']['supply_ledger']['totals'].values()),0.)
        del s.native_subduction_supply_version
        subduction.removal(s,values,2.);s.t=2.
        legacy=subduction.snapshot_metadata(s);subduction.validate_frame(legacy)
        self.assertNotIn('native_subduction_supply_version',legacy)
        s.native_subduction_supply_version=1
        with self.assertRaisesRegex(ValueError,'measured supply ledger'):subduction.snapshot_metadata(s)

    def test_schema_rejects_orphan_markers_caps_and_source_genealogy(self):
        s,_,values=scene(memory=True);_,_,valid=self.measure(s,values)
        mutations=[
            lambda f:f.pop('native_subduction_supply_version'),
            lambda f:f.update(native_subduction_supply_version=True),
            lambda f:f.update(native_subduction_supply_version=1.),
            lambda f:f['native_subduction_diagnostics'].pop('supply_ledger'),
            lambda f:f['native_subduction_diagnostics']['supply_ledger'].update(cap_order=['wrong']),
            lambda f:f['native_subduction_diagnostics']['supply_ledger']['totals'].update(competition_reduction_area_km2=100.),
            lambda f:f['native_subduction_diagnostics']['supply_ledger']['fronts'][0].update(source_trench_id=99),
            lambda f:f['native_subduction_diagnostics']['supply_ledger']['trenches'][0].update(source_parent_indices=[99]),
            lambda f:f['native_subduction_diagnostics']['supply_ledger']['fronts'][0].update(realized_removal_area_km2=float('nan')),
            lambda f:f['native_subduction_diagnostics']['supply_ledger']['fronts'][0].update(source_parent_index=True)]
        mutations.extend([
            lambda f:f['native_subduction_diagnostics']['supply_ledger']['fronts'][0]['source_geometry'].update(length_km=1.),
            lambda f:f['native_subduction_diagnostics']['supply_ledger']['fronts'][0].update(source_nominal_maturity=float('nan')),
            lambda f:f['native_subduction_diagnostics']['supply_ledger']['fronts'][0].update(local_water_before_union_km2=0.)])
        for mutate in mutations:
            frame=deepcopy(valid);mutate(frame)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):subduction.validate_frame(frame)


if __name__=='__main__':unittest.main()
