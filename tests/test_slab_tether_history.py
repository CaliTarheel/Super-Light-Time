"""Neck histories traverse the actual conservative slab lifecycle."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
import checkpoint
import slab_memory as slab
import slab_tether as tether
import slab_tether_history as history
from tests.test_slab_tether import neck
from tests.test_slab_mass_inventory import convert,legacy_row


def row(depth=100.,length=1000.,damage=0.,ident=1):
    value=convert(legacy_row(depth,length,ident)).trench_systems[0]
    history.initialize(value,neck(damage=damage))
    return value


def response(record):
    channels=record[history.FIELD]
    return tether.parallel([tether.Neck(**c['neck']) for c in channels],
        [c['retained_excess_mass_kg']*9.81/c['reference_width_m'] for c in channels],
        [1e21]*len(channels),[c['reference_width_m'] for c in channels])


class SlabTetherHistoryTests(unittest.TestCase):
    def test_actual_split_join_preserves_mass_rheology_and_transmitted_force(self):
        parent=row(damage=.7)
        original=deepcopy(parent);expected=response(parent)
        child=dict(length_km=300.)
        slab.partition(parent,child,.3)
        parent['length_km']=700.;slab.refresh_line_load(parent)
        for key in expected:
            self.assertAlmostEqual((response(parent)[key]+response(child)[key])/expected[key],1.,places=13)
        slab.join(child,parent)
        parent['length_km']=1000.;slab.refresh_line_load(parent)
        slab.validate_row(parent,require_mass=True)
        self.assertEqual(len(parent[history.FIELD]),2)
        self.assertTrue(all(c['neck']==original[history.FIELD][0]['neck'] for c in parent[history.FIELD]))
        for key in expected:self.assertAlmostEqual(response(parent)[key]/expected[key],1.,places=13)

    def test_unequal_damaged_trenches_keep_distinct_channels_on_join(self):
        first,second=row(damage=.1),row(depth=500.,damage=.9,ident=2)
        expected={key:response(first)[key]+response(second)[key] for key in response(first)}
        slab.join(first,second)
        second['length_km']=2000.;slab.refresh_line_load(second)
        self.assertEqual([c['neck']['damage'] for c in second[history.FIELD]],[.9,.1])
        for key in expected:self.assertAlmostEqual(response(second)[key]/expected[key],1.,places=13)
        self.assertEqual(first[history.FIELD],[])
        for key in slab.FIELDS+slab.MASS_FIELDS:self.assertEqual(first[key],0.)

    def test_actual_overriding_owner_transfer_partitions_necks_once(self):
        import trench_history
        from tests.test_trench_history import fixture,step
        s=fixture(((22,40,63),))
        trench_history.initialize(s)
        for _ in range(5):step(s)
        s.slab_memory_version=2
        original=row(length=s.trench_systems[0]['length_km'],damage=.6)
        for key in slab.FIELDS+slab.MASS_FIELDS+(slab.LINE_LOAD_FIELD,'slab_attachment',history.FIELD,history.VERSION_FIELD):
            s.trench_systems[0][key]=deepcopy(original[key])
        expected=response(s.trench_systems[0])
        s.active=np.array([True,True,True]);s.plate_uid=np.array([11,22,33]);s.omega=np.zeros((3,3))
        moved=np.arange(len(s.bb))>=len(s.bb)//2
        s.plate[s.bb[moved]]=2
        trench_history.transfer_overriding(s,1,2)
        self.assertEqual(len(s.trench_systems),2)
        for record in s.trench_systems:slab.validate_row(record,require_mass=True)
        for key in expected:self.assertAlmostEqual(sum(response(r)[key] for r in s.trench_systems)/expected[key],1.,places=13)

    def test_real_slab_feed_and_retirement_close_per_channel_and_resume(self):
        first,second=row(length=500.,damage=.2),row(depth=500.,length=1000.,damage=.8,ident=2)
        slab.join(first,second)
        second['length_km']=1500.;slab.refresh_line_load(second)
        s=SimpleNamespace(slab_memory_version=2,trench_systems=[second],t=2.,config={},rng=np.random.default_rng(12))
        old=deepcopy(second[history.FIELD])
        s.native_subduction_diagnostics=dict(step_end_myr=2.,step_duration_myr=2.,removed_area_km2=3000.,
            removed_area_by_trench=[dict(trench_id=2,downgoing_plate_uid=11,area_km2=3000.,buoyancy_area_km2=3000.)])
        slab.advance(s,2.)
        actual=s.trench_systems[0]
        expected_decay=np.exp(-2./50.)
        expected_feed=3000.*50./2.*(1.-expected_decay)
        widths=np.array([c['reference_width_m'] for c in old])
        for before,after,width in zip(old,actual[history.FIELD],widths):
            self.assertAlmostEqual(after['retained_area_km2'],before['retained_area_km2']*expected_decay+
                                   expected_feed*width/widths.sum(),places=8)
            self.assertEqual(after['neck'],before['neck'])
        history.validate(actual)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.npz'
            checkpoint.write_checkpoint(path,s,dict(config={}),{})
            resumed,_=checkpoint.read_checkpoint(path,None,SimpleNamespace)
        for world in (s,resumed):
            world.t=4.;world.native_subduction_diagnostics['step_end_myr']=4.
            slab.advance(world,2.)
        self.assertEqual(s.trench_systems,resumed.trench_systems)

    def test_rupture_retires_existing_material_and_cannot_reattach_from_feed(self):
        value=row(damage=.7)
        before=deepcopy(value)
        failed,event=tether.advance(tether.Neck(**value[history.FIELD][0]['neck']),1e13,1e21,0.,1000.)
        self.assertTrue(event['ruptured'])
        report=history.rupture(value,0,failed)
        self.assertEqual(report['detached_area_km2'],before['slab_retained_area_km2'])
        self.assertEqual(report['detached_excess_mass_kg'],before[slab.RETAINED_MASS_FIELD])
        self.assertEqual(value[slab.RETAINED_MASS_FIELD],0.)
        self.assertEqual(value[slab.LINE_LOAD_FIELD],0.)
        self.assertEqual(value['slab_retired_excess_mass_kg'],before[slab.RETAINED_MASS_FIELD])
        slab.validate_row(value,require_mass=True)
        with self.assertRaises(ValueError):history.advance(value,1.,1.,1.,1.)
        self.assertEqual(history.rupture(value,0,failed)['detached_excess_mass_kg'],0.)

    def test_mixed_histories_and_inconsistent_channels_fail_before_losing_inventory(self):
        source=row();target=convert(legacy_row(ident=2)).trench_systems[0]
        original=(deepcopy(source),deepcopy(target))
        with self.assertRaises(ValueError):slab.join(source,target)
        self.assertEqual((source,target),original)
        broken=deepcopy(source);broken[history.FIELD][0]['retained_excess_mass_kg']*=1.1
        with self.assertRaises(ValueError):slab.validate_row(broken,require_mass=True)
        with self.assertRaises(ValueError):history.rupture(source,0,neck())
        self.assertEqual(source,original[0])
        broken=deepcopy(source);del broken[history.VERSION_FIELD]
        with self.assertRaises(ValueError):slab.validate_row(broken,require_mass=True)

    def test_many_successive_ruptures_leave_exactly_zero_retained_inventory(self):
        value=row(length=1.137,depth=157.73)
        for i in range(1,80):
            slab.join(row(length=1.137+i*.3723,depth=157.73+i*.827),value)
        initial_mass=value[slab.RETAINED_MASS_FIELD]
        initial_area=value['slab_retained_area_km2']
        for i,channel in enumerate(value[history.FIELD]):
            history.rupture(value,i,replace(tether.Neck(**channel['neck']),damage=1.))
            slab.validate_row(value,require_mass=True)
        for key in history.INVENTORIES:self.assertEqual(value['slab_'+key],0.)
        self.assertAlmostEqual(value['slab_retired_excess_mass_kg']/initial_mass,1.,places=13)
        self.assertAlmostEqual(value['slab_retired_area_km2']/initial_area,1.,places=13)

    def test_ruptured_history_survives_later_identity_changes_without_recounting_material(self):
        first,second=row(damage=.8),row(depth=300.,damage=.2,ident=2)
        failed,_=tether.advance(tether.Neck(**first[history.FIELD][0]['neck']),1e13,1e21,0.,1000.)
        retired=history.rupture(first,0,failed)
        slab.join(first,second)
        second['length_km']=2000.;slab.refresh_line_load(second)
        child=dict(length_km=400.)
        slab.partition(second,child,.2)
        for record in (second,child):
            slab.validate_row(record,require_mass=True)
            self.assertEqual(sum(c['neck']['damage']==1. for c in record[history.FIELD]),1)
        detached=sum(c['detached_excess_mass_kg'] for r in (second,child) for c in r[history.FIELD])
        self.assertAlmostEqual(detached/retired['detached_excess_mass_kg'],1.,places=13)


if __name__=='__main__':unittest.main()
