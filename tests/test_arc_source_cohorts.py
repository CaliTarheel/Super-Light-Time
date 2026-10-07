"""Physical source continuity and volume identities without a history run."""
from copy import deepcopy
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import arc_source_cohorts as cohorts
from ridge_geometry import rotate


def unit(v):
    v=np.asarray(v,float)
    return v/np.linalg.norm(v,axis=-1,keepdims=True)


def fixture(intervals=((-0.03,0.03),),rotation=None):
    n=len(intervals)
    starts=unit([[1.,a,0.] for a,b in intervals]);ends=unit([[1.,b,0.] for a,b in intervals])
    points=unit([[1.,(a+b)*.5,.03] for a,b in intervals])
    if rotation is not None:
        starts=rotate(starts,rotation);ends=rotate(ends,rotation);points=rotate(points,rotation)
    s=SimpleNamespace(t=0.,plate_uid=np.array([11,22]),active=np.array([True,True]),
        omega=np.tile([.00031,-.00027,.00043],(2,1)),
        down=np.ones(n,int),bp=np.ones(n,int),bq=np.zeros(n,int),
        trench_id=np.ones(n,int),
        trench_systems=[dict(id=1,episode=1,downgoing_plate_uid=22,overriding_plate_uid=11,phase='mature')],
        native_boundary_geometry=dict(contact_index=np.arange(n),segments_start=starts,segments_end=ends))
    return s,points


def observe(s,points,amounts=None):
    if amounts is None:amounts=np.ones(len(points))
    return cohorts.build(s,np.arange(len(points)),positions=points,owners=np.zeros(len(points),int),areas=amounts)


class ArcSourceCohortTests(unittest.TestCase):
    def test_same_physical_moving_vent_accumulates_100_myr_for_both_dt_partitions(self):
        totals=[]
        for dt in (1.,2.):
            s,point=fixture();source=deepcopy(s.native_boundary_geometry)
            pending_points=np.empty((0,3));pending_rows=[];amounts=np.empty(0)
            for time in np.arange(dt,100.+dt*.5,dt):
                s.native_arc_pending=dict(xyz=pending_points,owner=np.zeros(len(amounts),int),area=amounts,
                                          source_provenance=pending_rows)
                cohorts.advect(s,dt);s.t=float(time)
                for name in ('segments_start','segments_end'):
                    s.native_boundary_geometry[name]=unit(rotate(source[name],s.omega[0]*time))
                current=unit(rotate(point,s.omega[0]*time))
                rows=observe(s,current,[500.*dt])
                pending_rows=cohorts.concat(pending_rows,rows)
                pending_points=np.vstack((s.native_arc_pending['xyz'],current))
                amounts=np.r_[amounts,500.*dt]
                groups=cohorts.groups(s,pending_points,np.zeros(len(amounts),int),amounts,pending_rows,np.ones(len(amounts),bool))
                self.assertEqual(len(groups),1)
                self.assertTrue(groups[0]['qualified'])
                self.assertEqual(len(groups[0]['indices']),len(amounts))
                self.assertEqual(sum(r['original_area_km2'] for r in pending_rows),500.*time)
                self.assertEqual(len({r['component_id'] for r in pending_rows}),1)
                self.assertLessEqual(len(groups[0]['links']),len(amounts)-1)
            totals.append(amounts.sum()*25.)
            self.assertGreater(amounts.sum(),6757.)
        np.testing.assert_array_equal(totals,[1250000.,1250000.])

    def test_front_subdivision_preserves_source_connectivity_and_volume(self):
        for n in (1,2,4):
            split=np.linspace(-.03,.03,n+1);s,p=fixture(list(zip(split[:-1],split[1:])))
            preserved=p.copy();areas=np.full(n,10000./n)
            rows=observe(s,p,areas)
            groups=cohorts.groups(s,p,np.zeros(n,int),areas,rows,np.ones(n,bool))
            self.assertEqual(len(groups),1);self.assertEqual(len(groups[0]['indices']),n)
            self.assertEqual(sum(row['original_area_km2'] for row in rows)*25.,250000.)
            np.testing.assert_array_equal(p,preserved)
            self.assertIn(groups[0]['anchor_index'],groups[0]['indices'])

    def test_same_front_refinement_between_observations_keeps_geological_identity(self):
        s,p=fixture();old=observe(s,p,[1000.]);old_state=deepcopy(s.arc_source_cohort_state)
        refined,q=fixture(((-.03,0.),(0.,.03)))
        refined.arc_source_cohort_state=old_state;refined.t=2.
        new=observe(refined,q,[500.,500.]);all_rows=cohorts.concat(old,new)
        points=np.vstack((p,q));amount=np.array([1000.,500.,500.])
        result=cohorts.groups(refined,points,np.zeros(3,int),amount,all_rows,np.ones(3,bool))
        self.assertEqual(len(result),1)
        self.assertEqual({row['component_id'] for row in all_rows},{old[0]['component_id']})

    def test_zero_flux_observation_does_not_strand_previously_pending_volume(self):
        s,p=fixture();first=observe(s,p,[1000.])
        s.t=1.;observe(s,p,[0.])
        s.t=2.;last=observe(s,p,[500.])
        rows=cohorts.concat(first,last)
        result=cohorts.groups(s,np.vstack((p,p)),np.zeros(2,int),np.array([1000.,500.]),rows,np.ones(2,bool))
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['indices'],[0,1])

    def test_missing_observation_retains_only_compatible_unspent_source_history(self):
        s,p=fixture();first=observe(s,p,[1000.])
        s.native_arc_pending=dict(xyz=p.copy(),owner=np.array([0]),area=np.array([1000.]),source_provenance=first)
        s.t=2.;cohorts.build(s,np.empty(0,int))
        self.assertEqual(len(s.arc_source_cohort_state['tracks']),1)
        self.assertEqual(s.arc_source_cohort_state['tracks'][0]['last_observed_myr'],0.)
        cohorts.validate_state(s.arc_source_cohort_state,time_myr=2.)
        s.t=4.;last=observe(s,p,[500.])
        self.assertEqual(first[0]['component_id'],last[0]['component_id'])
        result=cohorts.groups(s,np.vstack((p,p)),np.zeros(2,int),np.array([1000.,500.]),
            cohorts.concat(first,last),np.ones(2,bool))
        self.assertEqual(len(result),1)
        s.native_arc_pending['area'][:]=0.;cohorts.build(s,np.empty(0,int))
        self.assertEqual(s.arc_source_cohort_state['tracks'],[])

    def test_rotations_through_pole_and_dateline_keep_local_grouping(self):
        intervals=((-0.03,0.),(0.,.03),(.08,.10))
        expected=None
        for rotation in ([0.,0.,0.],[0.,np.pi*.5,0.],[0.,0.,np.pi],[.2,-.7,.4]):
            s,p=fixture(intervals,rotation);rows=observe(s,p)
            groups=cohorts.groups(s,p,np.zeros(3,int),np.ones(3),rows,np.ones(3,bool))
            actual=[g['indices'] for g in groups]
            if expected is None:expected=actual
            self.assertEqual(actual,expected)
        self.assertEqual(expected,[[0,1],[2]])

    def test_disconnected_same_trench_branches_do_not_pool(self):
        s,p=fixture(((-.05,-.03),(.03,.05)));rows=observe(s,p,[5000.,5000.])
        groups=cohorts.groups(s,p,np.zeros(2,int),np.full(2,5000.),rows,np.ones(2,bool))
        self.assertEqual(len(groups),2)
        self.assertNotEqual(rows[0]['component_id'],rows[1]['component_id'])

    def test_temporal_corridors_cannot_shortcut_opposite_arms_of_a_u_shaped_front(self):
        path=np.array([[-.02,.025],[-.02,0.],[-.02,-.06],[.02,-.06],[.02,0.],[.02,.025]])
        s,_=fixture([(-.03,.03)]*5)
        vertices=unit(np.column_stack((np.ones(len(path)),path)))
        s.native_boundary_geometry.update(segments_start=vertices[:-1],segments_end=vertices[1:])
        middle=(path[:-1]+path[1:])*.5
        middle+=np.array([[.03,0.],[.03,0.],[0.,.03],[-.03,0.],[-.03,0.]])
        points=unit(np.column_stack((np.ones(len(middle)),middle)))
        first=observe(s,points,np.full(5,10000.));s.t=2.
        second=observe(s,points,np.full(5,10000.))
        groups=cohorts.groups(s,np.vstack((points,points)),np.zeros(10,int),np.full(10,10000.),
            cohorts.concat(first,second),np.ones(10,bool))
        self.assertEqual(len(groups),1)
        # A local footprint at the two nearby arms excludes the distant bend.
        # Its induced contributor graph must retain two separate source sets.
        inside={0,4,5,9};reachable={0}
        for _ in range(4):
            for a,b in groups[0]['links']:
                if a in inside and b in inside and (a in reachable or b in reachable):reachable.update((a,b))
        self.assertEqual(reachable,{0,5})

    def test_split_is_explicit_new_identity_and_cannot_consume_old_branch_as_same_cohort(self):
        s,p=fixture();before=observe(s,p,[4000.])
        new,q=fixture(((-.03,-.005),(.005,.03)));new.t=2.
        new.arc_source_cohort_state=deepcopy(s.arc_source_cohort_state)
        after=observe(new,q,[2000.,2000.])
        self.assertEqual(len({r['component_id'] for r in before+after}),3)
        self.assertTrue(all(r['reason']=='ambiguous_split_or_join_new_identity' for r in after))

    def test_reused_or_inactive_slot_never_advects_or_qualifies_old_magma(self):
        for inactive in (False,True):
            s,p=fixture();rows=observe(s,p,[1000.]);old=deepcopy(rows)
            s.native_arc_pending=dict(xyz=p.copy(),owner=np.array([0]),area=np.array([1000.]),source_provenance=rows)
            if inactive:s.active[0]=False
            else:s.plate_uid[0]=999
            cohorts.advect(s,2.)
            np.testing.assert_array_equal(s.native_arc_pending['xyz'],p)
            np.testing.assert_array_equal(rows[0]['trace_xyz'],old[0]['trace_xyz'])
            self.assertEqual(rows[0]['overriding_plate_uid'],11)
            self.assertEqual(rows[0]['advection_host_plate_uid'],11)
            self.assertFalse(cohorts.host_mask(s,[0],rows)[0])
            self.assertEqual(rows[0]['reason'],'unresolved_advection_host')
            groups=cohorts.groups(s,p,np.array([0]),np.array([1000.]),rows,np.array([True]))
            self.assertFalse(groups[0]['qualified'])

    def test_episode_change_and_unknown_origins_cannot_inherit_cohort(self):
        s,p=fixture();first=observe(s,p,[1000.]);s.t=2.;s.trench_systems[0]['episode']=2
        second=observe(s,p,[1000.])
        self.assertNotEqual(cohorts.compatibility_key(first[0]),cohorts.compatibility_key(second[0]))
        unknown=cohorts.normalize(s,None,2,owners=[0,0],areas=[2.,3.])
        self.assertEqual([r['original_area_km2'] for r in unknown],[2.,3.])
        self.assertTrue(cohorts.host_mask(s,[0,0],unknown).all())
        self.assertIsNone(cohorts.compatibility_key(unknown[0]))
        self.assertNotEqual(unknown[0]['origin_id'],unknown[1]['origin_id'])

    def test_checkpoint_roundtrip_preserves_next_observation_and_per_origin_sources(self):
        s,p=fixture();rows=observe(s,p,[1000.])
        s.native_arc_pending=dict(xyz=p.copy(),owner=np.array([0]),area=np.array([1000.]),source_provenance=rows)
        restored=pickle.loads(pickle.dumps(s,protocol=5))
        for item in (s,restored):
            cohorts.advect(item,2.);item.t=2.
            for field in ('segments_start','segments_end'):
                item.native_boundary_geometry[field]=unit(rotate(item.native_boundary_geometry[field],item.omega[0]*2.))
        point=unit(rotate(p,s.omega[0]*2.))
        self.assertEqual(observe(s,point,[500.]),observe(restored,point,[500.]))
        self.assertEqual(s.arc_source_cohort_state,restored.arc_source_cohort_state)
        self.assertEqual(s.native_arc_pending['source_provenance'],restored.native_arc_pending['source_provenance'])

    def test_bad_or_duplicate_origin_metadata_is_rejected(self):
        s,p=fixture();rows=observe(s,p)
        for field,value in [('origin_id',True),('component_id',-1),('advection_host_plate_uid',1.2),('source_time_myr',float('nan'))]:
            bad=deepcopy(rows);bad[0][field]=value
            with self.assertRaises(ValueError):cohorts.validate(bad,1)
        with self.assertRaises(ValueError):cohorts.concat(rows,rows)
        with self.assertRaises(ValueError):cohorts.validate(rows,2)

    def test_saved_registry_rejects_bad_nested_geometry_ids_and_future_epochs(self):
        s,p=fixture();observe(s,p)
        state=deepcopy(s.arc_source_cohort_state)
        self.assertIs(cohorts.validate_state(state,time_myr=0.),state)
        variants=[]
        bad=deepcopy(state);bad['tracks'][0]['corridors'][0][0][0]=float('nan');variants.append(bad)
        bad=deepcopy(state);bad['tracks'][0]['origin_ids'][0]=True;variants.append(bad)
        bad=deepcopy(state);bad['next_origin_id']=1;variants.append(bad)
        bad=deepcopy(state);bad['tracks'][0]['key'][3]=bad['tracks'][0]['key'][2];variants.append(bad)
        bad=deepcopy(state);bad['tracks'][0]['last_observed_myr']=2.;variants.append(bad)
        bad=deepcopy(state);bad['tracks'].append(deepcopy(bad['tracks'][0]));variants.append(bad)
        for bad in variants:
            with self.assertRaises(ValueError):cohorts.validate_state(bad,time_myr=0.)
        with self.assertRaises(ValueError):cohorts.validate_state(state,time_myr=-1.)

    def test_native_dispatch_preserves_explicit_source_provenance(self):
        import native_processes,native_arc_material
        s=SimpleNamespace(native_arc_material_version=1)
        rows=[dict(proof='untouched')]
        with patch.object(native_arc_material,'add_arc_crust',return_value={'ok':True}) as target:
            result=native_processes.add_arc_crust(s,np.array([0]),np.array([3.]),
                positions=np.array([[1.,0.,0.]]),owners=np.array([0]),source_provenance=rows)
        self.assertEqual(result,{'ok':True})
        self.assertIs(target.call_args.kwargs['source_provenance'],rows)


if __name__=='__main__':unittest.main()
