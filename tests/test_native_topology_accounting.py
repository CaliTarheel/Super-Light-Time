"""Reachable native topology scheduling and signed ocean-accounting checks."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

import backarc
import native_processes
import native_topology
from tests.test_native_topology import world,ocean_partition


def spreading_world():
    s=world(level=2)
    s.plate=(s.xyz[:,1]>=0).astype(np.int16)
    s.active[:]=False;s.active[:2]=True
    s.support[:]=0.;s.support[s.plate,np.arange(s.n)]=1.
    s.omega[:]=0.;s.omega[:2,2]=[-.003,.003]
    s.age[:]=80.+20.*s.xyz[:,0]
    s.primordial_fraction[:]=.8+.1*s.xyz[:,2]
    # This controlled evolved partition replaces the original single-ocean
    # source. Its boundary graph must use the new support, not initial artwork.
    s.native_initial_ownership_consumed=True
    s._boundaries()
    return s


class NativeTopologyAccountingTests(unittest.TestCase):
    def test_eligible_empty_remnant_does_not_wait_for_an_arbitrary_step_number(self):
        source=world(level=3)
        source.active[:]=False;source.active[:2]=True
        source.plate[:]=0;source.plate[0]=1
        source.support[:]=0.;source.support[0]=1.;source.support[0,0]=0.;source.support[1,0]=1.
        for steps in (1,9,19,20,21):
            s=deepcopy(source);s.steps=steps;s.t=40.
            before=s.support.sum(axis=0).copy();uid=s.next_plate_uid
            native_topology._retire_remnants(s)
            self.assertFalse(s.active[1],f'step count {steps}')
            self.assertEqual(s.next_plate_uid,uid)
            np.testing.assert_array_equal(s.plate,0)
            np.testing.assert_array_equal(s.support.sum(axis=0),before)

    def test_real_backarc_commit_keeps_the_elapsed_ocean_damage_healing_update(self):
        s,proposal=ocean_partition()
        s.omega[:]=0.;s.ocean_history['damage'][:]=.6
        s._boundaries()
        region=np.zeros(s.n,bool);region[proposal['child_faces']]=True
        s.backarc_basins=[dict(id=1,parent_plate_uid=int(s.plate_uid[0]),
            downgoing_plate_uid=int(s.plate_uid[1]),arc_plate_uid=None,
            center=[1.,0.,0.],loading_rate_km_myr=4.,loading_km=70.,opening_km=0.)]
        choice=dict(region=region,parcel_side=np.zeros(len(s.mass),bool),
            trace_side=np.zeros(len(s.trace_patch),bool),center=np.array([1.,0.,0.]),
            direction=np.array([0.,1.,0.]),area_fraction=.5)
        edges=np.flatnonzero(s._valid_loading_edges())[:1]
        self.assertEqual(len(edges),1)
        before=s.ocean_history['damage'].copy()
        old_owner=s.plate.copy();seen={}
        original=native_topology.ocean_rifting.update
        def observe(mesh,material,*args,**kwargs):
            seen['owner']=material['owner'].copy()
            return original(mesh,material,*args,**kwargs)
        with (patch.object(backarc,'choose_arc_sliver',return_value=choice),
              patch.object(native_topology.ocean_rifting,'update',side_effect=observe),
              patch.object(native_topology.progressive_rifting,'commit') as generic):
            self.assertTrue(s._topology(2.,(1,edges,0,1)))
        generic.assert_not_called()
        np.testing.assert_array_equal(seen['owner'],old_owner)
        self.assertIsNotNone(s.backarc_basins[0]['arc_plate_uid'])
        self.assertGreater(s.process_totals['backarc_ruptures'],0.)
        self.assertTrue(np.all(s.ocean_history['damage']<before))
        np.testing.assert_allclose(s.ocean_history['damage'],before*np.exp(-2./1200.),rtol=1e-12)

    def test_native_step_always_passes_pending_backarc_through_history_stage(self):
        # Exercise the production integration dispatcher without a long run.
        s=world(level=2)
        pending=object()
        with (patch.object(backarc,'update',return_value=pending),
              patch.object(s,'_topology',return_value=False) as topology):
            s.step(2.)
        topology.assert_called_once_with(2.,pending)

    def test_real_foreign_junction_rejections_are_counted_without_repainting(self):
        s=spreading_world()
        at=int(np.argmax(s.xyz@np.array([1.,0.,.35])))
        s.plate[at]=2;s.active[2]=True
        s.support[:,at]=0.;s.support[2,at]=1.
        s._boundaries()
        transported=s.support.copy()
        native_processes._spreading_advance(s,transported,2.)
        d=s.spreading_diagnostics
        self.assertGreater(d['rejected_foreign_claims'],0)
        self.assertGreater(d['rejected_foreign_owner_claims'],0)
        self.assertEqual(d['candidate_claims'],d['admitted_claims']+d['rejected_foreign_claims']+d['rejected_buoyant_claims'])
        self.assertEqual(np.argmax(transported[:,at]),2)
        self.assertAlmostEqual(d['side_p_area_km2'],d['side_q_area_km2'],delta=1e-6)

    def test_incoming_third_owner_is_measured_separately_from_existing_owners(self):
        s=spreading_world()
        arrivals=np.zeros_like(s.support)
        arrivals[2,s.xyz[:,0]>.5]=.01
        native_processes._spreading_advance(s,s.support.copy(),2.,arrivals=arrivals)
        d=s.spreading_diagnostics
        self.assertGreater(d['rejected_incoming_foreign_claims'],0)
        self.assertEqual(d['rejected_foreign_owner_claims'],0)
        self.assertEqual(d['rejected_foreign_neighbor_claims'],0)

    def test_signed_ocean_ledger_separates_removal_normalization_and_newborn_reset(self):
        s=spreading_world()
        # Prescribe maturity on the actual convergent finite fronts. Patching
        # the historical radial weight no longer exercises native capture.
        self.assertGreater(np.count_nonzero(s.bcode==2),0)
        s.trench_maturity[s.bcode==2]=.5
        native_processes.advect_ocean(s,2.)
        self.assertGreater(s.native_subduction_diagnostics['capture_polygons'],0)
        ledger=s.ocean_diagnostics['support_moment_ledger']
        stages=ledger['stages']
        self.assertTrue(ledger['normalization_is_not_subduction'])
        self.assertGreater(ledger['measured_subduction_area_km2'],0.)
        self.assertGreater(stages['after_transport_and_conditional_reconstruction']['excess_area_km2'],0.)
        self.assertGreater(stages['after_transport_and_conditional_reconstruction']['deficit_area_km2'],0.)
        self.assertAlmostEqual(ledger['measured_subduction_area_km2'],s.process_totals['ocean_consumed_km2'],places=6)
        self.assertLess(np.max(np.abs(ledger['transport_support_area_residual_by_owner_km2'])),1e-4)
        self.assertLess(np.max(np.abs(ledger['transport_operator_moment_residuals_by_owner'])),.01)
        self.assertLess(abs(ledger['subduction_area_residual_km2']),1e-4)
        self.assertLess(np.max(np.abs(ledger['subduction_moment_residuals'])),.01)
        self.assertLess(abs(ledger['normalization_area_policy_residual_km2']),1e-4)
        self.assertLess(np.max(np.abs(ledger['newborn_reset_policy_residuals'])),.01)
        self.assertGreater(np.max(np.abs(ledger['normalization_moment_deltas'])),1.)
        self.assertGreater(np.max(np.abs(ledger['newborn_reset_moment_deltas'])),1.)
        final=stages['after_newborn_reset']
        self.assertAlmostEqual(final['support_area_km2']/s.cell_area.sum(),1.,places=12)
        age_index=ledger['moment_fields'].index('age')
        self.assertAlmostEqual(final['moment_totals'][age_index]/float(s.cell_area@s.age),1.,places=12)
        from validation.audit_collision_history import audit_ocean_ledger
        meta=dict(ocean_rifting_diagnostics=s.ocean_diagnostics)
        checked=audit_ocean_ledger(meta,dict(mesh_area_km2=s.cell_area))
        self.assertTrue(checked['passed'],checked['issues'])
        # Changing an operator observation cannot be concealed by retaining its
        # old reported zero residual: the audit recalculates the difference.
        corrupted=deepcopy(meta)
        corrupted['ocean_rifting_diagnostics']['support_moment_ledger']['transport_operator_output_moments_by_owner'][0][0]+=10000.
        checked=audit_ocean_ledger(corrupted,dict(mesh_area_km2=s.cell_area))
        self.assertFalse(checked['passed'])
        self.assertTrue(any('transport changes an owner property moment' in item for item in checked['issues']))


if __name__=='__main__':unittest.main()
