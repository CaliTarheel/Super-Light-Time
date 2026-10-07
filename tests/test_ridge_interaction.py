"""Behavioral ridge/trench geometry and finite pulse accounting checks."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest

import numpy as np

from ridge_interaction import (detect_ridge_trench_contacts, update_ridge_interactions,
                               sample_ridge_effects, reassign_ridge_episodes)


def xyz(lon, lat):
    lon, lat = np.radians([lon, lat])
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def junction():
    """Two spreading incoming plates meet a third overriding plate."""
    s = SimpleNamespace(
        xyz=np.array([xyz(-.5, 0), xyz(-.5, -1), xyz(.5, 0), xyz(-1, 1)]),
        plate=np.array([0, 1, 2, 0]), plate_uid=np.array([11, 22, 33]),
        active=np.ones(3, bool), crust=np.array([0, 0, 1, 0]), age=np.array([1., 2., 0., 2.]),
        bcode=np.array([1, 2]), ba=np.array([0, 0]), bb=np.array([1, 2]),
        bp=np.array([0, 0]), bq=np.array([1, 2]), down=np.array([-1, 0]),
        bmid=np.array([xyz(-.5, -.5), xyz(0, 0)]),
        bn=np.array([[0, 0, -1.], [0, 1., 0]]), bl=np.array([100., 100.]),
        omega=np.array([[0, -.001, .001], [0, .001, .001], [0, 0, 0.]]), t=0.)
    s._indices = lambda points: np.argmax(np.asarray(points) @ s.xyz.T, axis=1)
    return s


def episode():
    s = junction()
    return update_ridge_interactions(s, 0)[0]


class RidgeContactTests(unittest.TestCase):
    def test_actual_three_plate_junction_has_approaching_oceanic_axis(self):
        s = junction()
        relative = np.cross(s.omega[s.bq] - s.omega[s.bp], s.bmid) * 6371.
        normal_speed = np.sum(relative * s.bn, axis=1)
        shear = np.linalg.norm(relative - normal_speed[:, None] * s.bn, axis=1)
        self.assertGreater(normal_speed[0], max(2., .35 * shear[0]))
        self.assertLess(normal_speed[1], -max(2., .35 * shear[1]))
        contacts = detect_ridge_trench_contacts(s)
        self.assertEqual(len(contacts), 1)
        contact = contacts[0]
        self.assertEqual(contact['incoming_plate_uids'], [11, 22])
        self.assertEqual(contact['overriding_plate_uid'], 33)
        self.assertGreater(contact['evidence']['axis_approach_km_myr'], 6.)
        self.assertLess(contact['evidence']['ridge_trench_distance_km'], 100.)

    def test_young_crust_without_active_spreading_cannot_trigger(self):
        s = junction()
        s.bcode[0] = 3
        self.assertEqual(detect_ridge_trench_contacts(s), [])

    def test_continental_rift_age_zero_does_not_count_as_oceanic_ridge(self):
        s = junction()
        s.crust[1], s.age[1] = 1, 0.
        self.assertEqual(detect_ridge_trench_contacts(s), [])

    def test_nearby_parallel_edge_without_shared_down_cell_is_not_a_junction(self):
        s = junction()
        s.ba[0] = 3  # Position remains close, but there is intervening lithosphere.
        self.assertEqual(detect_ridge_trench_contacts(s), [])

    def test_backarc_or_receding_ridge_cannot_heat_this_margin(self):
        s = junction()
        s.omega[:2, 2] *= -1  # Still spreading, but its axis recedes from the trench.
        self.assertEqual(detect_ridge_trench_contacts(s), [])
        s = junction()
        s.bmid[0] = xyz(1, 0)
        self.assertEqual(detect_ridge_trench_contacts(s), [])

    def test_two_plate_boundary_and_old_crust_are_rejected(self):
        s = junction()
        s.plate_uid[1] = s.plate_uid[2]
        self.assertEqual(detect_ridge_trench_contacts(s), [])
        s = junction()
        s.age[1] = 30.
        self.assertEqual(detect_ridge_trench_contacts(s), [])

    def test_shared_coarse_cell_does_not_override_physical_distance_cap(self):
        s = junction()
        s.bmid[0] = xyz(-15, 0)
        self.assertEqual(detect_ridge_trench_contacts(s), [])

    def test_junction_geometry_is_invariant_at_poles_and_longitude_seam(self):
        expected = detect_ridge_trench_contacts(junction())[0]['evidence']
        for degrees in (90., 179.8, -179.8):
            angle = np.radians(degrees)
            rotation = np.array([[np.cos(angle), 0, np.sin(angle)],
                                 [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
            if abs(degrees) > 100:
                rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                                     [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
            s = junction()
            for name in ('xyz', 'bmid', 'bn', 'omega'):
                setattr(s, name, getattr(s, name) @ rotation.T)
            contact = detect_ridge_trench_contacts(s)[0]
            self.assertAlmostEqual(contact['evidence']['ridge_trench_distance_km'], expected['ridge_trench_distance_km'], places=6)
            self.assertAlmostEqual(contact['evidence']['axis_approach_km_myr'], expected['axis_approach_km_myr'], places=10)


class RidgeEpisodeTests(unittest.TestCase):
    def test_peak_then_fade_is_thermal_not_permanent_uplift(self):
        e = episode()
        points, owners = np.array([e['center']]), np.array([33])
        at_start = sample_ridge_effects([e], points, owners, 0)
        at_peak = sample_ridge_effects([e], points, owners, 8)
        after = sample_ridge_effects([e], points, owners, 40)
        self.assertEqual(float(at_start['thermal_support_m'][0]), 0.)
        self.assertAlmostEqual(float(at_peak['thermal_support_m'][0]), 400., places=3)
        self.assertAlmostEqual(float(at_peak['heat'][0]), 1., places=6)
        self.assertEqual(float(after['thermal_support_m'][0]), 0.)
        self.assertEqual(float(after['volcanic_addition_m'][0]), 0.)

    def test_integrated_construction_budget_is_timestep_independent(self):
        e = episode()
        points, owners = np.array([e['center']]), np.array([33])
        for steps in ([1.] * 40, [2.] * 20, [5., 5., 22., 8.]):
            now, total = 0., 0.
            for dt in steps:
                now += dt
                total += float(sample_ridge_effects([e], points, owners, now, dt)['volcanic_addition_m'][0])
            self.assertAlmostEqual(total, 200., places=4)

    def test_heat_and_construction_are_clipped_to_overrider_and_finite_footprint(self):
        e = episode()
        points = np.array([e['center'], e['center'], xyz(90, 0)])
        result = sample_ridge_effects([e], points, np.array([33, 11, 33]), 8, 8)
        self.assertGreater(result['volcanic_addition_m'][0], 0)
        for field in result.values():
            np.testing.assert_array_equal(field[1:], [0., 0.])
        overlapping = sample_ridge_effects([e, e, e], points[:1], np.array([33]), 8)
        self.assertEqual(float(overlapping['thermal_support_m'][0]), 600.)

    def test_continuous_contact_never_restarts_pulse_even_after_80_myr(self):
        s = junction()
        update_ridge_interactions(s, 0)
        for time in range(2, 202, 2):
            s.t = time
            self.assertEqual(update_ridge_interactions(s, 2), [])
        self.assertEqual(len(s.ridge_episodes), 1)
        self.assertEqual(s.ridge_episodes[0]['started_myr'], 0.)
        self.assertEqual(s.ridge_episodes[0]['last_seen_myr'], 200.)
        sample = sample_ridge_effects(s.ridge_episodes, np.array([s.ridge_episodes[0]['center']]), np.array([33]), 200, 2)
        self.assertEqual(float(sample['volcanic_addition_m'][0]), 0.)
        self.assertEqual(float(sample['thermal_support_m'][0]), 0.)

    def test_disappeared_contact_can_recur_after_latch_memory(self):
        s = junction()
        update_ridge_interactions(s, 0)
        s.bcode[0], s.t = 3, 82.
        update_ridge_interactions(s, 82)
        self.assertEqual(s.ridge_episodes, [])
        s.bcode[0], s.t = 1, 84.
        created = update_ridge_interactions(s, 2)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0]['id'], 2)

    def test_episode_center_follows_overrider_and_stable_uid_reassignment(self):
        s = junction()
        update_ridge_interactions(s, 0)
        s.bcode[0] = 3  # No new encounter during the advection check.
        original = np.array(s.ridge_episodes[0]['center'])
        s.omega[2] = [0, 0, .02]
        s.t = 2
        update_ridge_interactions(s, 2)
        moved = np.array(s.ridge_episodes[0]['center'])
        self.assertAlmostEqual(float(np.arccos(np.clip(original @ moved, -1, 1))), .04, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(moved)), 1., places=12)
        reassign_ridge_episodes(s, 33, 99, [False])
        self.assertEqual(s.ridge_episodes[0]['overriding_plate_uid'], 33)
        reassign_ridge_episodes(s, 33, 99)
        self.assertEqual(s.ridge_episodes[0]['overriding_plate_uid'], 99)
        self.assertGreater(sample_ridge_effects(s.ridge_episodes, moved[None, :], np.array([99]), 8)['heat'][0], .99)

    def test_plain_episode_state_survives_json_roundtrip_exactly(self):
        e = episode()
        restored = json.loads(json.dumps(e))
        points = np.array([e['center']])
        a = sample_ridge_effects([e], points, np.array([33]), 10, 2)
        b = sample_ridge_effects([restored], points, np.array([33]), 10, 2)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        self.assertEqual(e, restored)


if __name__ == '__main__':
    unittest.main()
