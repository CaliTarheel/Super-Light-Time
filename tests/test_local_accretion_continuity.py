"""A migrating local front retains history without lending it to remote contacts."""
from copy import deepcopy
import math
import unittest

import numpy as np

import local_accretion as local
from tests.test_local_accretion import world, contact


def ribbon():
    w = 32
    return world(source=[y*w+x for y in range(2,15) for x in (6,7)],
                 target=[y*w+x for y in range(2,15) for x in range(8,13)])


def observe(s, rows):
    contact(s, [(y*s.w+7,y*s.w+8) for y in rows])
    s.t += 2.
    return local.plan_accretions(s,2.)


class LocalContactContinuityTests(unittest.TestCase):
    def test_advancing_front_retains_debt_after_both_individual_anchors_leave(self):
        s = ribbon()
        observe(s, range(3,7))
        first = deepcopy(s.local_accretion_contacts[0])
        observe(s, range(5,9))
        row = s.local_accretion_contacts[0]
        self.assertEqual(len(s.local_accretion_contacts),1)
        self.assertNotIn(first['p_patch'],row['p_patches'])
        self.assertNotIn(first['q_patch'],row['q_patches'])
        self.assertTrue(set(first['p_patches']) & set(row['p_patches']))
        self.assertTrue(set(first['q_patches']) & set(row['q_patches']))
        self.assertGreater(row['loading'],first['loading'])
        identity = row['id']
        for rows in (range(7,11),range(9,13),range(8,12),range(7,11),range(6,10)):
            plans = observe(s,rows)
            self.assertEqual(len(s.local_accretion_contacts),1)
            self.assertEqual(s.local_accretion_contacts[0]['id'],identity)
        self.assertTrue(plans,'sustained migrating contact must eventually reach the unchanged threshold')
        self.assertEqual(plans[0]['contact_id'],identity)
        self.assertEqual(local.THRESHOLD,18.)
        self.assertEqual(local.MIN_PLATE_AGE_MYR,70.)

    def test_remote_front_on_same_connected_material_cannot_inherit_history(self):
        s = ribbon()
        for _ in range(4):observe(s,range(3,6))
        old = deepcopy(s.local_accretion_contacts[0])
        observe(s,range(11,14))
        active = [r for r in s.local_accretion_contacts if r['last_seen_myr']==s.t]
        self.assertEqual(len(active),1)
        self.assertNotEqual(active[0]['id'],old['id'])
        self.assertFalse(set(old['p_patches']) & set(active[0]['p_patches']))
        self.assertFalse(set(old['q_patches']) & set(active[0]['q_patches']))
        self.assertLess(active[0]['loading'],old['loading'])

    def test_contact_continuity_requires_overlap_on_both_sides(self):
        for side in ('p','q'):
            with self.subTest(side=side):
                s = ribbon()
                observe(s,range(3,7))
                row = s.local_accretion_contacts[0]
                old_id = row['id']
                # Same moving plates and same opposite-side patches are
                # insufficient if this side belongs to a different margin.
                remote = 1000+13*s.w+(7 if side=='p' else 8)
                row[side+'_patch'] = remote
                row[side+'_patches'] = [remote]
                observe(s,range(3,7))
                active = [r for r in s.local_accretion_contacts if r['last_seen_myr']==s.t]
                self.assertEqual(len(active),1)
                self.assertNotEqual(active[0]['id'],old_id)

    def test_merged_front_uses_strongest_existing_debt_without_summing(self):
        s = ribbon()
        observe(s,range(3,7))
        weaker = s.local_accretion_contacts[0]
        weaker['loading'] = 3.
        stronger = deepcopy(weaker)
        stronger['id'],stronger['loading'] = 2,12.
        s.local_accretion_contacts.append(stronger)
        s.next_accretion_contact_id = 3
        observe(s,range(3,7))
        active = [r for r in s.local_accretion_contacts if r['last_seen_myr']==s.t]
        self.assertEqual([r['id'] for r in active],[2])
        decay = math.exp(-2./local.MEMORY_MYR)
        self.assertAlmostEqual(weaker['loading'],3.*decay)
        self.assertLess(stronger['loading'],(12.+3.)*decay+4.)
        self.assertLessEqual(stronger['loading'],12.*decay+4.+1e-12)

    def test_one_old_contact_does_not_duplicate_debt_into_two_separated_fronts(self):
        s = ribbon()
        observe(s,range(3,14))
        s.local_accretion_contacts[0]['loading'] = 12.
        observe(s,[3,4,5,11,12,13])
        active = [r for r in s.local_accretion_contacts if r['last_seen_myr']==s.t]
        self.assertEqual(len(active),2)
        self.assertEqual(len({r['id'] for r in active}),2)
        self.assertEqual(sum(r['loading']>4. for r in active),1)

    def test_coalescence_maps_all_local_sets_and_ambiguity_invalidates_nonanchor_members(self):
        s = ribbon()
        observe(s,range(3,7))
        row = s.local_accretion_contacts[0]
        old = next(value for value in row['p_patches'] if value != row['p_patch'])
        original_loading = row['loading']
        local.remap_contact_patches(s,np.array([old,old]),np.array([99001,99001]))
        self.assertIn(99001,row['p_patches'])
        self.assertNotIn(old,row['p_patches'])
        self.assertEqual(row['loading'],original_loading)
        local.remap_contact_patches(s,np.array([99001,99001]),np.array([99002,99003]))
        self.assertEqual(row['loading'],0.)
        self.assertTrue(row['completed'])
        self.assertIn(99001,row['p_patches'],'ambiguous descendants are never selected arbitrarily')

    def test_legacy_single_anchor_record_upgrades_from_observed_local_contact(self):
        s = ribbon()
        observe(s,range(3,7))
        row = s.local_accretion_contacts[0]
        identity = row['id']
        del row['p_patches'],row['q_patches']
        observe(s,range(3,7))
        self.assertEqual(len(s.local_accretion_contacts),1)
        self.assertEqual(row['id'],identity)
        self.assertGreater(len(row['p_patches']),1)
        self.assertGreater(len(row['q_patches']),1)


if __name__ == '__main__':
    unittest.main()
