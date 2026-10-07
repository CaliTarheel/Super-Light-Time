"""Inherited slab area is a local provenance label, never new runtime feed."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np

import checkpoint
import slab_memory as slab
import slab_tether_history as history
import slab_tether_local as local
from slab_tether import Neck
from tests.test_slab_tether_history import row as inventory_row


def inherited():
    row=inventory_row()
    row[slab.INITIAL_AREA_FIELD]=row['slab_fed_area_km2']
    row['slab_fed_area_km2']=0.
    slab.validate_row(row,require_mass=True)
    return row


class InitialSlabNeckInventoryTests(unittest.TestCase):
    def test_localization_preserves_inherited_area_without_labeling_it_as_feed(self):
        row=inherited();before=deepcopy(row)
        local.localize(row,[[1.,0.,0.],[0.,1.,0.]],[.8,.2],support_radius_km=1000.)
        for key in slab.FIELDS+slab.MASS_FIELDS+(slab.INITIAL_AREA_FIELD,):
            self.assertEqual(row[key],before[key])
        np.testing.assert_allclose([c['initial_area_km2'] for c in row[history.FIELD]],
                                   np.array([.8,.2])*before[slab.INITIAL_AREA_FIELD])
        self.assertEqual([c['fed_area_km2'] for c in row[history.FIELD]],[0.,0.])
        slab.validate_row(row,require_mass=True)

    def test_unequal_spatial_split_and_rejoin_transfer_each_initial_label_once(self):
        parent=inherited()
        local.localize(parent,[[1.,0.,0.],[0.,1.,0.]],[.8,.2],support_radius_km=1000.)
        before=deepcopy(parent);child=dict(length_km=500.)
        # Half the reported trace moves, but the selected local patch owns
        # only one fifth of the inherited area. Global fraction is not a ledger.
        slab.partition(parent,child,.5,channel_fractions=[0.,1.])
        self.assertEqual(child[slab.INITIAL_AREA_FIELD],before[slab.INITIAL_AREA_FIELD]*.2)
        self.assertEqual(parent[slab.INITIAL_AREA_FIELD],before[slab.INITIAL_AREA_FIELD]*.8)
        for row in (parent,child):slab.validate_row(row,require_mass=True)
        slab.join(child,parent)
        for key in slab.FIELDS+slab.MASS_FIELDS+(slab.INITIAL_AREA_FIELD,):
            self.assertAlmostEqual(parent[key]/max(before[key],1.),before[key]/max(before[key],1.),places=13)
            self.assertEqual(child[key],0.)
        self.assertEqual(child[history.FIELD],[])
        for row in (parent,child):slab.validate_row(row,require_mass=True)

    def test_local_feed_retirement_rupture_and_checkpoint_keep_original_baseline(self):
        row=inherited()
        local.localize(row,[[1.,0.,0.],[0.,1.,0.]],[.8,.2],support_radius_km=1000.)
        baseline=row[slab.INITIAL_AREA_FIELD]
        world=SimpleNamespace(slab_memory_version=2,trench_systems=[row],t=2.,
                              config={},rng=np.random.default_rng(12))
        gain=200.;mass=gain*1e6*float(slab.excess_mass_per_area_kg_m2(80.))
        world.native_subduction_diagnostics=dict(step_end_myr=2.,step_duration_myr=2.,removed_area_km2=gain,
            removed_area_by_trench=[dict(trench_id=1,downgoing_plate_uid=11,area_km2=gain,buoyancy_area_km2=gain,
                neck_feeds=[dict(channel_index=1,area_km2=gain,buoyancy_area_km2=gain,excess_mass_kg=mass)])])
        slab.advance(world,2.);row=world.trench_systems[0]
        self.assertEqual(row[slab.INITIAL_AREA_FIELD],baseline)
        self.assertEqual(row['slab_fed_area_km2'],gain)
        self.assertEqual([c['fed_area_km2'] for c in row[history.FIELD]],[0.,gain])
        self.assertAlmostEqual(row['slab_retained_area_km2']+row['slab_retired_area_km2'],baseline+gain,places=8)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'neck.npz'
            checkpoint.write_checkpoint(path,world,dict(config={}),{})
            resumed,_=checkpoint.read_checkpoint(path,None,SimpleNamespace)
        for state in (world,resumed):
            state.t=4.;state.native_subduction_diagnostics=dict(step_end_myr=4.,step_duration_myr=2.,
                removed_area_km2=0.,removed_area_by_trench=[])
            slab.advance(state,2.)
            value=state.trench_systems[0]
            for index,channel in enumerate(value[history.FIELD]):
                history.rupture(value,index,replace(Neck(**channel['neck']),damage=1.))
            self.assertEqual(value['slab_retained_area_km2'],0.)
            self.assertEqual(value[slab.INITIAL_AREA_FIELD],baseline)
            self.assertAlmostEqual(value['slab_retired_area_km2'],baseline+gain,places=8)
            slab.validate_row(value,require_mass=True)
        self.assertEqual(world.trench_systems,resumed.trench_systems)

    def test_old_local_channel_schema_defaults_only_missing_initial_area_to_zero(self):
        row=inventory_row()
        local.localize(row,[[1.,0.,0.],[0.,1.,0.]],[.8,.2],support_radius_km=1000.)
        for channel in row[history.FIELD]:del channel['initial_area_km2']
        del row[slab.INITIAL_AREA_FIELD]
        before=deepcopy(row);slab.validate_row(row,require_mass=True)
        self.assertEqual(row,before)
        child=dict(length_km=500.)
        slab.partition(row,child,.5,channel_fractions=[0.,1.])
        self.assertEqual(row[slab.INITIAL_AREA_FIELD],0.)
        self.assertEqual(child[slab.INITIAL_AREA_FIELD],0.)
        slab.join(child,row);slab.validate_row(row,require_mass=True)
        corrupt=deepcopy(row);corrupt[slab.INITIAL_AREA_FIELD]=1.
        with self.assertRaises(ValueError):slab.validate_row(corrupt,require_mass=True)


if __name__=='__main__':unittest.main()
