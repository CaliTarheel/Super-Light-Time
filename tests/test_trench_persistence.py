"""Opt-in attached-slab persistence: stationary loaded trenches keep pulling."""
from copy import deepcopy
import unittest

import numpy as np

import native_engine
import slab_memory
import trench_history

# Coastal inherited subduction on four primordial ocean plates, one attached
# to a continent. Its trenches are held at prescribed normal speeds below.
CONFIG = dict(width=48, height=24, mesh_level=4, coast_geometry_level=2, plate_count=4,
              mechanics_nodes=128, seed=37, physics_profile='reviewed_v1',
              primordial_subduction={'enabled': True},
              primordial_ocean={'enabled': True, 'continental_attachments':
                                [dict(region_index=1, continental_plate_uid=3)]})


def lifecycle(s, speed, myr):
    """Advance only the trench lifecycle with every trench held at one normal speed."""
    s = deepcopy(s)
    traced = s.trench_id > 0
    for _ in range(myr):
        s.t += 1.
        s.normal_speed = np.where(traced, speed, s.normal_speed)
        trench_history.update(s, 1.)
    live = [r for r in s.trench_systems if r['phase'] not in ('shutdown', 'joined')]
    return s, live


class PersistenceConfigurationTests(unittest.TestCase):
    def test_policy_values_and_profile_requirement(self):
        self.assertEqual(trench_history.normalize_persistence(), 'kinematic')
        for bad in ('attached', 1, None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                trench_history.normalize_persistence(bad)
        with self.assertRaisesRegex(ValueError, 'reviewed_v1'):
            native_engine.validate_config(dict(trench_persistence='attached_slab_v1'))

    def test_saved_state_without_field_keeps_kinematic_law(self):
        class State:
            pass
        self.assertEqual(trench_history.persistence_floor(State()), 0.)
        state = State()
        state.trench_persistence_version = 2
        with self.assertRaises(ValueError):
            trench_history.persistence_floor(state)


class StationaryTrenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kinematic = native_engine.Simulation(CONFIG)
        cls.persistent = native_engine.Simulation(dict(CONFIG, trench_persistence='attached_slab_v1'))

    def test_versions_are_saved_at_fresh_initialization(self):
        self.assertEqual(self.kinematic.trench_persistence_version, 0)
        self.assertEqual(self.persistent.trench_persistence_version, 1)
        self.assertEqual(self.persistent.physics_profile_diagnostics['trench_persistence_version'], 1)

    def test_noise_level_opening_shuts_kinematic_trenches_but_not_persistent_ones(self):
        # +1e-3 km/Myr is the balance noise on a stationary, balanced plate.
        _, kinematic = lifecycle(self.kinematic, 1e-3, 11)
        state, persistent = lifecycle(self.persistent, 1e-3, 11)
        self.assertLess(len(kinematic), len(self.kinematic.trench_systems))
        self.assertEqual(len(persistent), sum(r['phase'] != 'joined' for r in state.trench_systems))
        self.assertTrue(all(r['phase'] != 'quiet' for r in persistent))
        owners, load, _ = slab_memory.line_load(state)
        self.assertGreater(float((load*state.bl)[owners >= 0].sum()), 0.)

    def test_genuine_opening_still_shuts_persistent_trenches(self):
        original = {r['id'] for r in self.persistent.trench_systems}
        _, live = lifecycle(self.persistent, 5., 11)
        # Every held-open original system shuts down; trenches born later on
        # genuinely converging contacts are not part of this check.
        self.assertEqual([r['id'] for r in live if r['id'] in original], [])


if __name__ == '__main__':
    unittest.main()
