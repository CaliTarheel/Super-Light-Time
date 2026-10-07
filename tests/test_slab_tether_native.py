"""Accepted rupture intervals use the actual native transport/source path."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import checkpoint
import plate_balance as pb
import native_engine
import normal_partition
import slab_memory as slab
import slab_tether_history as history
import slab_tether_local as local
import slab_tether_native as stepping
import trench_history
from tests.test_native_processes import ocean_fixture
from tests.test_slab_tether import neck
from slab_tether import Neck


def world():
    s=ocean_fixture(level=2,two=True)
    s.plate_balance_version=1;s.plate_resistance_version=1
    normal_partition.upgrade(s)
    s._boundaries()
    s.trench_systems=[];s.next_trench_id=1
    edges=np.flatnonzero(s.bl>0.)[:4]
    row=trench_history._birth(s,edges,np.zeros(len(s.ba),int))
    row.update(phase='mature',maturity=1.,active_myr=10.,shortening_km=100.)
    area=row['length_km']*300.
    mass=area*1e6*float(slab.excess_mass_per_area_kg_m2(80.))
    row.update(slab_fed_area_km2=area,slab_retained_area_km2=area,
               slab_retained_buoyancy_area_km2=area,slab_initial_excess_mass_kg=mass,
               slab_retained_excess_mass_kg=mass)
    slab.refresh_line_load(row)
    history.initialize(row,neck(damage=.95,failure_opening_m=10.))
    local.localize(row,[row['center']],[1.],support_radius_km=15000.)
    trench_history.prepare(s)
    return s


def advance(s,dt,**kwargs):
    return stepping.advance(s,dt,new_neck=neck(failure_opening_m=10.),support_radius_km=15000.,
                            max_source_step_myr=kwargs.pop('max_source_step_myr',.002),
                            absolute_tolerance=1e-6,relative_tolerance=1e-5,**kwargs)


class NativeSlabRuptureTests(unittest.TestCase):
    def test_coupled_history_cannot_initiate_from_a_new_convergent_color(self):
        s=world();s.trench_shutdown_version=1
        s.trench_systems=[];s.trench_id[:]=0
        s.bcode[:]=2;s.normal_speed[:]=-20.;s.down[:]=0
        with patch('native_subduction.edge_ocean_fraction',return_value=np.ones(len(s.ba))):
            s.t+=1.;trench_history.update(s,1.)
        self.assertEqual(s.trench_systems,[])
        np.testing.assert_array_equal(s.trench_id,0)

    def test_coupled_history_cannot_rebirth_after_detached_trace_memory_expires(self):
        s=world();s.trench_shutdown_version=1
        row=s.trench_systems[0]
        history.rupture(row,0,replace(Neck(**row[history.FIELD][0]['neck']),damage=1.))
        s.t+=1.;trench_history.update(s,1.)
        self.assertEqual(row['phase'],'shutdown')
        s.t+=trench_history.REACTIVATION_MEMORY_MYR+1.
        s.bcode[:]=2;s.normal_speed[:]=-20.;s.down[:]=0
        with patch('native_subduction.edge_ocean_fraction',return_value=np.ones(len(s.ba))):
            trench_history.update(s,1.)
        self.assertEqual(len(s.trench_systems),1)
        self.assertEqual(row['phase'],'shutdown')

    def test_coupled_history_does_not_reverse_an_attached_slabs_polarity(self):
        s=world();s.trench_shutdown_version=1;s.subduction_polarity_version=0
        row=s.trench_systems[0]
        s.bcode[:]=2;s.normal_speed[:]=-20.;s.down[:]=1
        def ocean_fraction(_world,owner):
            return np.broadcast_to(np.where(np.asarray(owner)==0,0.,1.),(len(s.ba),))
        with patch('native_subduction.edge_ocean_fraction',side_effect=ocean_fraction):
            s.t+=1.;trench_history.update(s,1.)
        self.assertEqual(len(s.trench_systems),1)
        self.assertEqual(row['phase'],'quiet')
        self.assertEqual(row['downgoing_plate_uid'],s.plate_uid[0])

    def test_attached_local_slab_survives_collision_timer_until_actual_rupture(self):
        s=world();s.trench_shutdown_version=1
        row=s.trench_systems[0]
        original_mass=row['slab_retained_excess_mass_kg']
        s.bcode[:]=4;s.normal_speed[:]=-20.
        s.crust[s.ba]=1;s.crust[s.bb]=1
        # Isolate the lifecycle from finite ocean capture: the same boundary
        # is closing, but its incoming side is entirely buoyant here.
        with patch('native_subduction.edge_ocean_fraction',return_value=np.zeros(len(s.ba))):
            for _ in range(4):
                s.t+=2.;trench_history.update(s,2.)
        self.assertEqual(row['phase'],'quiet')
        self.assertEqual(row['blocking_reason'],'buoyant_collision')
        self.assertEqual(row['slab_retained_excess_mass_kg'],original_mass)
        self.assertTrue(pb.Balance(s,1.,slab_tethers=True).tether_assembly['channels'])
        channel=row[history.FIELD][0]
        history.rupture(row,0,replace(Neck(**channel['neck']),damage=1.))
        self.assertEqual(row['slab_retained_excess_mass_kg'],0.)
        trench_history.prepare(s)
        np.testing.assert_array_equal(s.trench_maturity,0.)
        with patch('native_subduction.edge_ocean_fraction',return_value=np.zeros(len(s.ba))):
            s.t+=1.;trench_history.update(s,1.)
        self.assertEqual(row['phase'],'shutdown')
        self.assertEqual(row['episodes'][-1]['shutdown_reason'],'slab_necks_detached')
        s.bcode[:]=2;s.normal_speed[:]=-20.;s.crust[s.ba]=0;s.crust[s.bb]=0
        s.t+=1.;trench_history.update(s,1.)
        self.assertEqual(row['phase'],'shutdown')
        self.assertEqual(len(s.trench_systems),1)

    def test_partial_coincident_failure_gates_once_and_feeds_only_intact_history(self):
        import native_subduction
        from tests.test_native_subduction_oracle import fixture as capture_fixture
        from tests.test_slab_tether_history import row as inventory_row
        for geometry in (0,1):
            s,axis,support=capture_fixture()
            s.native_subduction_geometry_version=geometry
            s.t=0.;s.age=np.full(s.n,80.);s.slab_memory_version=2
            s.plate_uid=np.arange(len(s.support))+11;s.trench_id=np.ones(len(s.ba),int)
            row=inventory_row();row['overriding_plate_uid']=12
            point=axis.mean(axis=0);point/=np.linalg.norm(point);s.bmid=np.array([point])
            local.localize(row,[point,point],[.25,.75],support_radius_km=1000.)
            s.trench_systems=[row]
            whole=native_subduction.removal(s,support,2.)
            history.rupture(row,0,replace(Neck(**row[history.FIELD][0]['neck']),damage=1.))
            maturity=s.trench_maturity.copy()
            part=native_subduction.removal(s,support,2.)
            again=native_subduction.removal(s,support,2.)
            np.testing.assert_allclose(part,whole*.75,rtol=2e-12,atol=1e-15)
            np.testing.assert_array_equal(part,again)
            np.testing.assert_array_equal(s.trench_maturity,maturity)
            feed=s.native_subduction_diagnostics['removed_area_by_trench'][0]
            self.assertEqual([c['channel_index'] for c in feed['neck_feeds']],[1])
            self.assertAlmostEqual(feed['neck_feeds'][0]['area_km2']/feed['area_km2'],1.,places=13)
            s.t=2.;slab.advance(s,2.)
            updated=s.trench_systems[0];slab.validate_row(updated,require_mass=True)
            self.assertEqual(updated[history.FIELD][0]['fed_excess_mass_kg'],0.)
            self.assertGreater(updated[history.FIELD][1]['fed_excess_mass_kg'],0.)

    def test_short_source_intervals_advance_newborn_crust_through_rupture(self):
        # This real native path previously exhausted gravity backtracking on
        # metre-scale arc faces. Keep its short source intervals as a regression.
        s=world();result=advance(s,.00005,max_source_step_myr=.000005)
        self.assertEqual(s.t,.00005)
        self.assertEqual(len(result['ruptures']),1)
        self.assertGreater(len(result['source_intervals']),10)
        self.assertGreater(len(s.material_surface['faces']),0)
        for row in s.trench_systems:slab.validate_row(row,require_mass=True)

    def test_native_step_splits_rupture_and_preserves_actual_source_ledgers(self):
        s=world();initial=s.t;initial_steps=s.steps
        mass=sum(r[slab.RETAINED_MASS_FIELD] for r in s.trench_systems)
        result=advance(s,.002)
        self.assertAlmostEqual(s.t-initial,.002,places=14)
        self.assertEqual(s.steps-initial_steps,len(result['source_intervals']))
        self.assertGreater(len(result['source_intervals']),1)
        self.assertEqual(len(result['ruptures']),1)
        event=result['ruptures'][0]
        self.assertLess(event['geological_time_myr'],s.t)
        self.assertGreater(event['detached_excess_mass_kg'],0.)
        for a,b in zip(result['source_intervals'],result['source_intervals'][1:]):
            self.assertEqual(a['end_myr'],b['start_myr'])
        self.assertEqual(result['source_intervals'][-1]['end_myr'],s.t)
        for row in s.trench_systems:slab.validate_row(row,require_mass=True)
        total=sum(r['slab_retained_excess_mass_kg']+r['slab_retired_excess_mass_kg'] for r in s.trench_systems)
        fed=sum(r['slab_fed_excess_mass_kg'] for r in s.trench_systems)
        self.assertAlmostEqual(total/(mass+fed),1.,places=12)
        self.assertTrue(np.isfinite(s.support).all())
        self.assertAlmostEqual(s.native_subduction_diagnostics['step_end_myr'],s.t,places=12)
        self.assertEqual(s.plate_balance_diagnostics['plate_resistance_version'],1)
        self.assertEqual(s.plate_balance_diagnostics['negative_resisting_work_elements'],0)

    def test_failed_source_transaction_leaves_original_world_and_rng_unchanged(self):
        s=world();before=deepcopy(s.__dict__)
        original=native_engine.Simulation._advance_step
        def fail_after_source(staged,dt,**kwargs):
            original(staged,dt,**kwargs)
            staged.rng.normal(size=10)
            raise ValueError('injected failure after actual source advancement')
        with patch.object(native_engine.Simulation,'_advance_step',fail_after_source),self.assertRaisesRegex(ValueError,'injected failure'):
            advance(s,.002)
        self.assertEqual(s.t,before['t']);self.assertEqual(s.steps,before['steps'])
        self.assertEqual(s.rng.bit_generator.state,before['rng'].bit_generator.state)
        self.assertEqual(s.trench_systems,before['trench_systems'])
        np.testing.assert_array_equal(s.support,before['support'])
        np.testing.assert_array_equal(s.omega,before['omega'])
        np.testing.assert_array_equal(s.age,before['age'])

    def test_native_checkpoint_restart_repeats_accepted_sources_and_force_solution(self):
        s=world();advance(s,.000001)
        self.assertEqual(s.trench_shutdown_version,1)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.npz'
            checkpoint.write_checkpoint(path,s,dict(config=s.config),{})
            resumed,_=checkpoint.read_checkpoint(path,None,native_engine.Simulation)
        self.assertEqual(resumed.plate_resistance_version,1)
        self.assertEqual(resumed.trench_shutdown_version,1)
        with self.assertRaisesRegex(ValueError,'Rupture-governed trenches'):
            resumed.step(.000001)
        left=advance(s,.001999);right=advance(resumed,.001999)
        self.assertEqual(left,right)
        self.assertEqual(s.trench_systems,resumed.trench_systems)
        np.testing.assert_array_equal(s.support,resumed.support)
        np.testing.assert_array_equal(s.age,resumed.age)
        np.testing.assert_array_equal(s.omega,resumed.omega)
        self.assertEqual(s.rng.bit_generator.state,resumed.rng.bit_generator.state)


if __name__=='__main__':unittest.main()
