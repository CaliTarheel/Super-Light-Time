"""Ordinary reviewed-physics continental entry and breakoff policy."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

import continental_lifecycle as lifecycle
import native_engine
import slab_memory
import trench_history
from tests.test_slab_tether_native import world as rupture_world


class ContinentalLifecycleTests(unittest.TestCase):
    def test_config_is_explicit_and_reviewed_only(self):
        self.assertFalse(lifecycle.normalize()['enabled'])
        for bad in ({'version':2},{'enabled':1},{'failure_opening_km':0.},
                    {'max_source_step_myr':2.},{'replacement_delay_myr':-1.},
                    {'unknown':1}):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                lifecycle.normalize(bad)
        with self.assertRaisesRegex(ValueError,'reviewed_v1'):
            native_engine.validate_config(dict(continental_lifecycle={'enabled':True}))

    def test_initialization_localizes_loaded_inherited_slabs_without_changing_inventory(self):
        s=rupture_world()
        # The explicit fixture already has a local history; reset it to the
        # conservative inherited row shape to exercise production admission.
        row=s.trench_systems[0]
        import slab_tether_history as histories
        import slab_tether_local as local
        row.pop(histories.FIELD,None);row.pop(histories.VERSION_FIELD,None)
        before={key:row[key] for key in (*slab_memory.FIELDS,*slab_memory.MASS_FIELDS)}
        s.config['continental_lifecycle']=lifecycle.normalize({'enabled':True})
        s.physics_profile_version=1;s.t=0.;s.steps=0
        report=lifecycle.initialize(s)
        self.assertTrue(report['enabled'])
        self.assertEqual(s.automatic_entry_version,1)
        self.assertEqual(s.trench_shutdown_version,1)
        self.assertEqual(s.replacement_subduction_version,1)
        self.assertTrue(local.enabled(row))
        for key,value in before.items():self.assertEqual(row[key],value)

    def test_ordinary_step_dispatches_to_atomic_coupled_path(self):
        s=rupture_world();s.physics_profile_version=1;s.t=0.;s.steps=0
        s.config['continental_lifecycle']=lifecycle.normalize(
            {'enabled':True,'max_source_step_myr':.01})
        lifecycle.initialize(s)
        with patch('slab_tether_native.advance',return_value={'advanced_dt_myr':.02}) as advance:
            result=s.step(.02)
        self.assertEqual(result['advanced_dt_myr'],.02)
        advance.assert_called_once()
        self.assertEqual(advance.call_args.args[:2],(s,.02))

    def test_breakoff_handoff_retains_entry_region_potential_without_source_attachment(self):
        import entry_regions
        s=rupture_world();s.physics_profile_version=1;s.t=0.;s.steps=0
        s.config['continental_lifecycle']=lifecycle.normalize({'enabled':True})
        lifecycle.initialize(s)
        # Minimal registered state exercises the handoff schema without
        # inventing a new entry geometry in this policy test.
        row=s.trench_systems[0]
        s.continental_entry_regions=dict(version=1,epoch_myr=0.,
            face_ids=s.parcel_patch.copy(),remesh_energy_change_j=0.,
            assignment='automatic finite trench front',
            regions=[dict(id=1,source_trench_id=row['id'],
                overriding_plate_uid=row['overriding_plate_uid'],
                hinge_normal=np.asarray(row['center'],float)/np.linalg.norm(row['center']),
                dip_degrees=50.,created_myr=0.)])
        s.parcel_entry_region=np.zeros(len(s.parcel_patch),np.int64)
        # A row with no selected material can still record source detachment;
        # real selected regions additionally keep their frozen entry energy.
        changed=entry_regions.mark_source_detached(s,row['id'])
        self.assertEqual(changed,[1])
        self.assertTrue(s.continental_entry_regions['regions'][0]['source_detached'])

    def test_recent_breakoff_excludes_same_trace_but_not_distant_independent_contact(self):
        s=rupture_world();s.physics_profile_version=1
        settings=lifecycle.normalize({'enabled':True,'replacement_delay_myr':20.,
                                      'replacement_exclusion_km':600.})
        s.continental_lifecycle_version=1;s.continental_lifecycle_settings=settings
        s.replacement_subduction_version=1
        row=s.trench_systems[0]
        row['phase']='shutdown';row['episodes'][-1].update(end_myr=0.,shutdown_reason='slab_necks_detached')
        s.t=5.
        points=np.asarray(s.bmid)
        near=int(np.argmin(trench_history._nearest(points,np.asarray(row['geometry_xyz']))))
        far=int(np.argmax(trench_history._nearest(points,np.asarray(row['geometry_xyz']))))
        allowed=lifecycle.replacement_allowed(s,np.array([near,far]))
        self.assertFalse(allowed[0])
        if trench_history._nearest(points[[far]],np.asarray(row['geometry_xyz']))[0]>600.:
            self.assertTrue(allowed[1])


if __name__=='__main__':
    unittest.main()
