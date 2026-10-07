"""Local extensive inventories, force locations and non-cancelling rupture."""
from copy import deepcopy
from dataclasses import replace
import unittest
import numpy as np
import slab_memory as slab
import slab_tether_history as history
import slab_tether_local as local
import slab_tether_forces as forces
from slab_tether import Neck
import plate_balance as pb
from tests.test_slab_tether_forces import fixture


def world():
    s = fixture(edges=(400.,600.))
    s.bmid[1] = [0.,1.,0.]; s.bn[1] = [0.,0.,1.]
    local.localize(s.trench_systems[0],s.bmid,[.4,.6],support_radius_km=1000.)
    return s


class LocalSlabNeckTests(unittest.TestCase):
    def test_whole_patch_transfer_is_bounded_despite_reduction_roundoff(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        s=world(); row=s.trench_systems[0]; before=deepcopy(row)
        # Actual allocated trace of patch 0 in Highland50 trench 41 at 2.5 Myr.
        # The selected subset and the zero-padded full trace take different
        # reduction paths; the old ratio is 1 + one ulp. The second patch has
        # no resolved trace and must retain its inventory with the parent.
        lengths=np.zeros((2,17))
        lengths[0]=[107249.80678757568,82521.46766847894,0.,0.,0.,0.,108907.84442855073,
                    91536.62318781052,0.,0.,0.,0.,0.,0.,78734.40516075054,70925.24119230914,0.]
        selected=np.array([True,True,True,True,False,False,True,True,True,False,False,False,
                           True,True,True,True,True])
        state=SimpleNamespace(bmid=np.tile(s.bmid[0],(17,1)),bl=np.ones(17))
        self.assertGreater(lengths[:,selected].sum(axis=1)[0]/lengths.sum(axis=1)[0],1.)
        with patch.object(local,'allocation',return_value=lengths):
            shares=local.transfer_fractions(row,state,np.arange(17),selected)
        np.testing.assert_array_equal(shares,[1.,0.])
        child=dict(length_km=600.)
        slab.partition(row,child,.6,channel_fractions=shares)
        for key in local.EXTENSIVE:
            self.assertEqual(row[history.FIELD][0].get(key,0.),0.)
            self.assertEqual(child[history.FIELD][0].get(key,0.),before[history.FIELD][0].get(key,0.))
            self.assertEqual(row[history.FIELD][1].get(key,0.),before[history.FIELD][1].get(key,0.))
            self.assertEqual(child[history.FIELD][1].get(key,0.),0.)
        for record in (row,child):slab.validate_row(record,require_mass=True)

    def test_invalid_local_patch_fractions_still_reject_without_mutation(self):
        for shares in (None,[.5],[np.nan,.5],[-1e-8,.5],[1.+1e-8,.5]):
            parent=world().trench_systems[0];child=dict(length_km=600.)
            before=deepcopy((parent,child))
            with self.assertRaisesRegex(ValueError,'resolved fraction for every patch'):
                slab.partition(parent,child,.6,channel_fractions=shares)
            self.assertEqual((parent,child),before)

    def test_baseline_preserves_all_inventories_and_rejects_invalid_geometry_atomically(self):
        s = fixture(); row = s.trench_systems[0]; before = deepcopy(row)
        local.localize(row,[[1.,0.,0.],[0.,1.,0.]],[.4,.6],support_radius_km=500.)
        for key in slab.FIELDS+slab.MASS_FIELDS:self.assertEqual(row[key],before[key])
        for key in local.LEDGER:
            self.assertAlmostEqual(sum(c[key] for c in row[history.FIELD])/max(row['slab_'+key],1.),
                                   row['slab_'+key]/max(row['slab_'+key],1.),places=13)
        for anchors,fractions,radius in (([[2.,0.,0.]],[1.],500.),([[1.,0.,0.]],[.5],500.),
                                          ([[1.,0.,0.]],[1.],-1.)):
            value=deepcopy(before)
            with self.assertRaises(ValueError):local.localize(value,anchors,fractions,support_radius_km=radius)
            self.assertEqual(value,before)

    def test_distinct_patches_keep_force_location_on_join_and_no_force_outside_support(self):
        s=world(); row=s.trench_systems[0]
        row[history.FIELD][1]['neck']['damage']=.9
        before=pb.Balance(s,1.,slab_tethers=True)
        child={k:deepcopy(row[k]) for k in ('phase','maturity','downgoing_plate_uid','overriding_plate_uid')}
        child.update(id=2,length_km=600.)
        slab.partition(row,child,.6,channel_fractions=[0.,1.])
        row['length_km']=400.;slab.refresh_line_load(row)
        s.trench_systems.append(child);s.trench_id[1]=2
        split=pb.Balance(s,1.,slab_tethers=True)
        np.testing.assert_allclose(split.stiffness,before.stiffness,rtol=2e-14)
        np.testing.assert_allclose(split.torque,before.torque,rtol=2e-14)
        slab.join(child,row);child['phase']='joined';s.trench_id[:]=1
        row['length_km']=1000.;slab.refresh_line_load(row)
        joined=pb.Balance(s,1.,slab_tethers=True)
        np.testing.assert_allclose(joined.torque,before.torque,rtol=2e-14)
        np.testing.assert_allclose(joined.stiffness,before.stiffness,rtol=2e-14)
        np.testing.assert_array_equal(local.allocation(row,[[0.,0.,1.]],[1000.]),0.)

    def test_ruptured_patch_stays_a_gap_and_does_not_spread_neighbor_force(self):
        s=world();row=s.trench_systems[0]
        history.rupture(row,0,replace(Neck(**row[history.FIELD][0]['neck']),damage=1.))
        model=pb.Balance(s,1.,slab_tethers=True)
        np.testing.assert_array_equal(model.slab_owner,[-1,0])
        self.assertEqual(model.tether_assembly['loads'][0],0.)
        self.assertEqual(model.tether_assembly['channels'][0]['channel_index'],1)
        slab.validate_row(row,require_mass=True)

    def test_local_opening_and_closing_do_not_cancel_damage(self):
        s=world();row=s.trench_systems[0]
        model=pb.Balance(s,1.,slab_tethers=True)
        x=np.zeros(6)
        first,second=model.tether_assembly['channels']
        # First inlet is stationary and its slab opens the neck. Second inlet
        # feeds faster than free sinking and is compressed: it must not heal
        # or cancel the first patch's tensile damage.
        x[0]=2.*second['weight_n']/second['mantle_drag_n_s_m']/pb.CM_YR_M_S
        staged,report=local.advance_damage(row,model.tether_assembly,x,.01)
        self.assertGreater(staged[history.FIELD][0]['neck']['damage'],row[history.FIELD][0]['neck']['damage'])
        self.assertEqual(staged[history.FIELD][1]['neck']['damage'],row[history.FIELD][1]['neck']['damage'])
        self.assertEqual(report['remaining_dt_myr'],0.)
        self.assertEqual(row,s.trench_systems[0])
        power=forces.audit(model.tether_assembly,x)
        self.assertLess(abs(power['power_residual_w'])/max(abs(power['weight_power_w']),1.),1e-12)

    def test_first_rupture_returns_remainder_and_retires_only_local_material(self):
        s=world();row=s.trench_systems[0]
        row[history.FIELD][0]['neck']['damage']=.95
        model=pb.Balance(s,1.,slab_tethers=True)
        expected=row[history.FIELD][0]['retained_excess_mass_kg']
        before=deepcopy(row)
        staged,report=local.advance_damage(row,model.tether_assembly,np.zeros(6),1000.)
        self.assertEqual(report['ruptured_channel_indices'],[0])
        self.assertGreater(report['remaining_dt_myr'],0.)
        self.assertEqual(staged['slab_retired_excess_mass_kg'],expected)
        self.assertGreater(staged[history.FIELD][1]['retained_excess_mass_kg'],0.)
        self.assertEqual(row,before)
        with self.assertRaisesRegex(ValueError,'same channel state'):
            local.advance_damage(staged,model.tether_assembly,np.zeros(6),.1)

    def test_actual_owner_transfer_moves_local_mass_not_global_length_fraction(self):
        import trench_history
        from tests.test_slab_tether_history import row as inventory_row
        from tests.test_trench_history import fixture as trench_fixture,step
        s=trench_fixture(((22,40,63),))
        trench_history.initialize(s)
        for _ in range(5):step(s)
        s.slab_memory_version=2
        saved=inventory_row(length=s.trench_systems[0]['length_km'])
        row=s.trench_systems[0]
        for key in slab.FIELDS+slab.MASS_FIELDS+(slab.LINE_LOAD_FIELD,'slab_attachment',history.FIELD,history.VERSION_FIELD):
            row[key]=deepcopy(saved[key])
        anchors=s.bmid[[0,-1]]
        local.localize(row,anchors,[.8,.2],support_radius_km=6000.)
        expected=row[history.FIELD][1]['retained_excess_mass_kg']
        s.active=np.array([True,True,True]);s.plate_uid=np.array([11,22,33]);s.omega=np.zeros((3,3))
        moved=np.arange(len(s.bb))>=len(s.bb)//2
        s.plate[s.bb[moved]]=2
        trench_history.transfer_overriding(s,1,2)
        daughter=s.trench_systems[1]
        self.assertAlmostEqual(daughter[slab.RETAINED_MASS_FIELD]/expected,1.,places=12)
        self.assertEqual(daughter['overriding_plate_uid'],33)
        for record in s.trench_systems:slab.validate_row(record,require_mass=True)

    def test_actual_persistent_separation_and_rejoin_preserve_patch_inventory(self):
        import trench_history
        from tests.test_slab_tether_history import row as inventory_row
        from tests.test_trench_history import fixture as trench_fixture,step
        s=trench_fixture(((20,42,63),))
        trench_history.initialize(s)
        for _ in range(5):step(s)
        s.slab_memory_version=2
        row=s.trench_systems[0];saved=inventory_row(length=row['length_km'])
        for key in slab.FIELDS+slab.MASS_FIELDS+(slab.LINE_LOAD_FIELD,'slab_attachment',history.FIELD,history.VERSION_FIELD):
            row[key]=deepcopy(saved[key])
        local.localize(row,s.bmid[[0,-1]],[.8,.2],support_radius_km=6000.)
        before=deepcopy(row)
        s.bcode[8:14],s.normal_speed[8:14]=1,20.
        for _ in range(8):step(s)
        self.assertEqual(len(s.trench_systems),2)
        retained=sorted(r[slab.RETAINED_MASS_FIELD]/before[slab.RETAINED_MASS_FIELD] for r in s.trench_systems)
        np.testing.assert_allclose(retained,[.2,.8],rtol=1e-12)
        s.bcode[:],s.normal_speed[:]=2,-20.
        step(s)
        self.assertEqual(sum(r['phase']=='joined' for r in s.trench_systems),1)
        for key in slab.FIELDS+slab.MASS_FIELDS:
            self.assertAlmostEqual(sum(r[key] for r in s.trench_systems)/max(before[key],1.),before[key]/max(before[key],1.),places=12)
        for record in s.trench_systems:slab.validate_row(record,require_mass=True)

    def test_retirement_is_local_and_unresolved_feed_is_rejected_without_mutation(self):
        from types import SimpleNamespace
        s=world();row=s.trench_systems[0]
        state=SimpleNamespace(slab_memory_version=2,trench_systems=[row],t=2.,
            native_subduction_diagnostics=dict(step_end_myr=2.,step_duration_myr=2.,removed_area_km2=0.,removed_area_by_trench=[]))
        before=deepcopy(row)
        slab.advance(state,2.)
        for old,new in zip(before[history.FIELD],state.trench_systems[0][history.FIELD]):
            self.assertAlmostEqual(new['retained_excess_mass_kg']/old['retained_excess_mass_kg'],np.exp(-2./50.),places=13)
        state.t=4.;state.native_subduction_diagnostics=dict(step_end_myr=4.,step_duration_myr=2.,removed_area_km2=10.,
            removed_area_by_trench=[dict(trench_id=1,downgoing_plate_uid=11,area_km2=10.,buoyancy_area_km2=10.)])
        old=deepcopy(state.trench_systems)
        with self.assertRaisesRegex(ValueError,'resolved capture provenance'):slab.advance(state,2.)
        self.assertEqual(state.trench_systems,old)

    def test_actual_capture_feeds_only_its_patch_with_actual_cooling_mass_and_saved_provenance(self):
        import native_subduction
        from tests.test_native_subduction_oracle import fixture as capture_fixture
        from tests.test_slab_tether_history import row as inventory_row
        s,axis,support=capture_fixture()
        s.t=0.;s.age=np.full(s.n,400.);s.slab_memory_version=2
        s.plate_uid=np.arange(len(s.support))+11;s.trench_id=np.ones(len(s.ba),int)
        row=inventory_row();row['overriding_plate_uid']=12
        point=axis.mean(axis=0);point=point/np.linalg.norm(point)
        s.bmid=np.array([point])
        local.localize(row,[point,-point],[.2,.8],support_radius_km=1000.)
        s.trench_systems=[row]
        before=deepcopy(row)
        removed=native_subduction.removal(s,support,2.)
        area=float(removed.sum(axis=0)@s.cell_area)
        feed=s.native_subduction_diagnostics['removed_area_by_trench'][0]
        self.assertEqual([e['channel_index'] for e in feed['neck_feeds']],[0])
        self.assertAlmostEqual(feed['neck_feeds'][0]['area_km2']/area,1.,places=12)
        mass=area*1e6*(3300.-1030.)*(3051.-2473.*np.exp(-.0278*400.))
        self.assertAlmostEqual(feed['neck_feeds'][0]['excess_mass_kg']/mass,1.,places=12)
        s.t=2.;slab.advance(s,2.)
        after=s.trench_systems[0]
        self.assertAlmostEqual(after[history.FIELD][0]['fed_excess_mass_kg']/mass,1.,places=12)
        self.assertEqual(after[history.FIELD][1]['fed_excess_mass_kg'],0.)
        self.assertAlmostEqual(after[history.FIELD][1]['retained_excess_mass_kg']/before[history.FIELD][1]['retained_excess_mass_kg'],np.exp(-2./50.),places=12)
        native_subduction.snapshot_metadata(s)
        corrupt=deepcopy(feed);corrupt['neck_feeds'][0]['area_km2']*=1.1
        with self.assertRaises(ValueError):local.validate_feed(corrupt)

    def test_advection_and_checkpoint_keep_material_anchors_and_damage_response(self):
        import trench_history
        from ridge_geometry import rotate
        from pathlib import Path
        from types import SimpleNamespace
        import tempfile
        import checkpoint
        s=world();s.t=0.;s.rng=np.random.default_rng(27)
        row=s.trench_systems[0]
        row.update(geometry_xyz=s.bmid.tolist(),center=s.bmid[0].tolist())
        old_points=s.bmid.copy();s.omega[1]=[0.,0.,.02]
        trench_history.advect(s,2.)
        s.bmid=rotate(s.bmid,s.omega[1]*2.);s.bn=rotate(s.bn,s.omega[1]*2.)
        np.testing.assert_allclose(np.asarray([c['anchor_xyz'] for c in row[history.FIELD]]),s.bmid,atol=1e-14)
        self.assertGreater(np.linalg.norm(old_points-s.bmid),0.)
        expected=local.allocation(row,s.bmid,s.bl*1000.)
        trench_history.advect(s,2.)
        np.testing.assert_array_equal(local.allocation(row,s.bmid,s.bl*1000.),expected)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.npz'
            checkpoint.write_checkpoint(path,s,dict(config={}),{})
            resumed,_=checkpoint.read_checkpoint(path,None,SimpleNamespace)
        results=[]
        for source in (s,resumed):
            model=pb.Balance(source,.01,slab_tethers=True)
            results.append(local.advance_damage(source.trench_systems[0],model.tether_assembly,np.zeros(6),.01))
        self.assertEqual(results[0],results[1])

    def test_positive_opening_is_integrated_before_averaging_inside_one_patch(self):
        s=fixture();s.bmid[1]=[0.,1.,0.];s.bn[1]=[0.,0.,1.]
        anchor=np.array([1.,1.,0.])/np.sqrt(2.)
        local.localize(s.trench_systems[0],[anchor],[1.],support_radius_km=11000.)
        row=s.trench_systems[0];model=pb.Balance(s,1.,slab_tethers=True)
        record=model.tether_assembly['channels'][0]
        weight,drag=record['weight_n'],record['mantle_drag_n_s_m']
        x=np.zeros(6);x[0]=weight/drag*(1.+.4/.6)/pb.CM_YR_M_S
        staged,_=local.advance_damage(row,model.tether_assembly,x,.01)
        neck=Neck(**row[history.FIELD][0]['neck']);width=row[history.FIELD][0]['reference_width_m']
        # Independent RK4 integration: signed driving averages to zero, but
        # the tensile .4 share is nonzero and must damage this finite-volume cell.
        def rate(d):
            return .4*weight/(drag+4.*neck.viscosity_pa_s*neck.thickness_m/neck.length_m*width*(1.-d)**2)/neck.failure_opening_m*(365.25*86400.*1e6)
        d=neck.damage;step=.01/1000.
        for _ in range(1000):
            a=rate(d);b=rate(d+step*a/2);c=rate(d+step*b/2);e=rate(d+step*c)
            d+=step*(a+2*b+2*c+e)/6
        self.assertGreater(d,neck.damage)
        self.assertAlmostEqual(staged[history.FIELD][0]['neck']['damage'],d,places=12)


if __name__=='__main__':unittest.main()
