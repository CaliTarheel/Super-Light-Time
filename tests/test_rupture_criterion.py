"""Rupture criterion 1: realized breakup extension in km, independent of links.

Version 0 (every saved world) keeps the mode-specific strain gates covered by
test_rupture_calibration; these tests check the new law and its isolation.
"""
from copy import deepcopy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import progressive_rifting as rifts
from tests.test_rupture_calibration import link_world, prescribed_update, observed_cut_candidates

KEY = (10, 11, 7)


def criterion_world(enhanced=False, length=400., version=1):
    state, mesh = link_world(enhanced, length)
    state.rupture_criterion_version = version
    return state, mesh


def pending(mesh, *, extension_km=None, strain=0., damage=1., opening=1., **extra):
    record = dict(mesh=mesh, reliable=True, damage=np.array([damage]), strain=np.array([strain]),
                  edge_extension=np.array([opening]), velocity=np.zeros((2, 3)), **extra)
    if extension_km is not None:
        record['extension_km'] = np.array([extension_km])
    return record


def enhanced_motion(state, mesh, opening=1.):
    state.rift_realized_motion = dict(
        **{k: mesh[k].copy() for k in ('bases', 'owner_uids', 'edges')},
        strain=np.array([.001]), edge_extension=np.array([opening]),
        velocity=np.zeros((2, 3)), dt_myr=1.)


class RuptureCriterionTests(unittest.TestCase):
    def test_breakup_extension_gate_is_the_same_at_every_link_length(self):
        for enhanced, needed in ((True, 100.), (False, 400.)):
            for length in (250., 400., 700., 1100.):
                with self.subTest(enhanced=enhanced, length=length):
                    state, mesh = criterion_world(enhanced, length)
                    state.rift_pending = pending(mesh, extension_km=needed, rupture_criterion_version=1)
                    np.testing.assert_array_equal(observed_cut_candidates(state, mesh), [True])
                    state.rift_pending = pending(mesh, extension_km=np.nextafter(needed, 0.),
                                                 rupture_criterion_version=1)
                    self.assertIsNone(observed_cut_candidates(state, mesh))

    def test_version_zero_strain_gate_recorded_the_old_mesh_dependence(self):
        # 150 km of realized monotonic extension on links of different length.
        realized = 150.
        for enhanced in (True, False):
            old, new = [], []
            for length in (250., 1100.):
                strain = (math.log((length+realized)/length) if enhanced else realized/length)
                proxy = realized if enhanced else realized/rifts.PROXY_REALIZED_FRACTION
                state, mesh = criterion_world(enhanced, length, version=0)
                state.rift_pending = pending(mesh, strain=strain, extension_km=proxy)
                old.append(observed_cut_candidates(state, mesh) is not None)
                state, mesh = criterion_world(enhanced, length)
                state.rift_pending = pending(mesh, strain=strain, extension_km=proxy)
                new.append(observed_cut_candidates(state, mesh) is not None)
            self.assertEqual(old, [True, False], enhanced)
            self.assertEqual(new, [True, True], enhanced)

    def test_update_accumulates_to_the_gate_in_both_modes(self):
        for enhanced, before, speed in ((True, 99., 1.), (False, 398., 2.)):
            with self.subTest(enhanced=enhanced):
                state, mesh = criterion_world(enhanced)
                state.rift_bonds[KEY] = dict(damage=1., strain=0., extension_km=before)
                if enhanced:
                    enhanced_motion(state, mesh)
                prescribed_update(state, mesh, 1., speed=speed)
                row = state.rift_bonds[KEY]
                self.assertEqual(set(row), {'damage', 'strain', 'extension_km', 'last_seen_myr'})
                self.assertEqual(state.rift_pending['rupture_criterion_version'], 1)
                np.testing.assert_array_equal(state.rift_pending['extension_km'], [row['extension_km']])
                mechanics = state.rift_mechanics
                self.assertNotIn('rupture_strain_threshold', mechanics)
                self.assertEqual(mechanics['breakup_extension_km'], 100.)
                self.assertEqual(mechanics['realized_extension_fraction'], 1. if enhanced else .25)
                self.assertEqual(mechanics['max_bond_realized_extension_km'], 100.)
                np.testing.assert_array_equal(observed_cut_candidates(state, mesh), [True])

    def test_version_one_ignores_accumulated_strain(self):
        state, mesh = criterion_world(True)
        state.rift_pending = pending(mesh, strain=0., extension_km=100.)
        np.testing.assert_array_equal(observed_cut_candidates(state, mesh), [True])
        state.rift_pending = pending(mesh, strain=2., extension_km=99.)
        self.assertIsNone(observed_cut_candidates(state, mesh))

    def test_current_opening_and_damage_still_gate_version_one(self):
        for enhanced in (False, True):
            for speed, damage in ((-2., 1.), (0., 1.), (.02, 1.), (2., .949)):
                with self.subTest(enhanced=enhanced, speed=speed, damage=damage):
                    state, mesh = criterion_world(enhanced)
                    state.rift_pending = pending(mesh, strain=2., extension_km=5000.,
                                                 damage=damage, opening=speed)
                    self.assertIsNone(observed_cut_candidates(state, mesh))

    def test_missing_version_is_zero_and_invalid_versions_are_refused(self):
        self.assertEqual(rifts.rupture_criterion_version(SimpleNamespace()), 0)
        self.assertEqual(rifts.rupture_criterion_version(SimpleNamespace(rupture_criterion_version=np.int64(1))), 1)
        state, mesh = link_world()
        self.assertEqual(rifts._rupture_calibration(state)['rupture_strain_threshold'], .15)
        for invalid in (True, np.bool_(True), 2, -1, 1., '1', None):
            with self.subTest(invalid=invalid):
                state.rupture_criterion_version = invalid
                with self.assertRaises(ValueError):
                    rifts.rupture_criterion_version(state)
                with self.assertRaises(ValueError):
                    rifts._rupture_calibration(state)
                state.rift_pending = pending(mesh, extension_km=100.)
                with self.assertRaises(ValueError):
                    rifts.commit(state)

    def test_recorded_per_run_strain_threshold_stays_a_criterion_zero_setting(self):
        # PR #188: a criterion-0 run keeps the realized threshold it recorded;
        # criterion 1 has no strain threshold for that value to select.
        old, mesh = criterion_world(True, version=0)
        old.realized_rupture_strain_threshold = .30
        self.assertEqual(rifts._rupture_calibration(old)['rupture_strain_threshold'], .30)
        old.rift_pending = pending(mesh, strain=.30, extension_km=0.)
        np.testing.assert_array_equal(observed_cut_candidates(old, mesh), [True])
        new, mesh = criterion_world(True)
        new.realized_rupture_strain_threshold = .30
        labels = rifts._rupture_calibration(new)
        self.assertNotIn('rupture_strain_threshold', labels)
        self.assertEqual(labels['breakup_extension_km'], 100.)
        new.rift_pending = pending(mesh, strain=.30, extension_km=np.nextafter(100., 0.))
        self.assertIsNone(observed_cut_candidates(new, mesh))
        new.rift_pending = pending(mesh, strain=0., extension_km=100.)
        np.testing.assert_array_equal(observed_cut_candidates(new, mesh), [True])

    def test_pending_work_is_not_reinterpreted_across_criteria(self):
        for state_version, record_version in ((1, 0), (0, 1)):
            with self.subTest(state=state_version, record=record_version):
                state, mesh = criterion_world(version=state_version)
                state.rift_pending = pending(mesh, strain=2., extension_km=5000.,
                                             rupture_criterion_version=record_version)
                with patch.object(rifts.rift_material, 'refresh', side_effect=AssertionError('stale criterion')):
                    self.assertFalse(rifts.commit(state))

    def test_version_one_pending_without_extension_waits(self):
        state, mesh = criterion_world()
        state.rift_pending = pending(mesh, strain=2., rupture_criterion_version=1)
        with patch.object(rifts.rift_material, 'refresh', side_effect=AssertionError('no extension')):
            self.assertFalse(rifts.commit(state))

    def test_unannotated_pending_uses_the_state_criterion(self):
        state, mesh = criterion_world()
        state.rift_pending = pending(mesh, strain=2., extension_km=np.nextafter(400., 0.))
        self.assertIsNone(observed_cut_candidates(state, mesh))
        state.rift_pending = pending(mesh, strain=0., extension_km=400.)
        np.testing.assert_array_equal(observed_cut_candidates(state, mesh), [True])
        state, mesh = criterion_world(version=0)
        state.rift_pending = pending(mesh, strain=.15, extension_km=0.)
        np.testing.assert_array_equal(observed_cut_candidates(state, mesh), [True])
        state.rift_pending = pending(mesh, strain=np.nextafter(.15, 0.), extension_km=5000.)
        self.assertIsNone(observed_cut_candidates(state, mesh))

    def test_fixed_rate_extension_decision_is_additive_across_step_partitions(self):
        whole, mesh = criterion_world()
        divided = deepcopy(whole)
        prescribed_update(whole, mesh, 4., speed=100.)
        for _ in range(8):
            prescribed_update(divided, mesh, .5, speed=100.)
        decisions = []
        for state in (whole, divided):
            self.assertEqual(state.rift_bonds[KEY]['extension_km'], 400.)
            self.assertEqual(state.rift_mechanics['max_bond_realized_extension_km'], 100.)
            # Damage integration is deliberately not timestep invariant.
            state.rift_pending['damage'][:] = 1.
            decisions.append(observed_cut_candidates(state, mesh))
        for decision in decisions:
            np.testing.assert_array_equal(decision, [True])

    def test_enhanced_realized_extension_is_additive_across_step_partitions(self):
        # Realized lengthening at 25 km/Myr: 4 Myr at once or eight 0.5 Myr
        # steps both add 100 km, telescoping the logarithmic strain as well.
        whole, mesh = criterion_world(True)
        divided = deepcopy(whole)
        rate, length = 25., 400.
        for state, dt, count in ((whole, 4., 1), (divided, .5, 8)):
            for step in range(count):
                before = length + rate*dt*step
                state.rift_realized_motion = dict(
                    **{k: mesh[k].copy() for k in ('bases', 'owner_uids', 'edges')},
                    strain=np.array([math.log((before + rate*dt)/before)]),
                    edge_extension=np.array([rate]), velocity=np.zeros((2, 3)), dt_myr=dt)
                prescribed_update(state, mesh, dt)
        for state in (whole, divided):
            self.assertEqual(state.rift_bonds[KEY]['extension_km'], 100.)
            self.assertAlmostEqual(state.rift_bonds[KEY]['strain'], math.log(500./400.), places=12)
            self.assertEqual(state.rift_mechanics['max_bond_realized_extension_km'], 100.)
            state.rift_pending['damage'][:] = 1.
            np.testing.assert_array_equal(observed_cut_candidates(state, mesh), [True])

    def test_snapshot_labels_and_frame_validation(self):
        old, _ = link_world(True)
        old.rift_systems, old.rift_mechanics = [], dict(rifts._rupture_calibration(old))
        frame = rifts.snapshot(old)
        self.assertNotIn('rupture_criterion_version', frame)
        self.assertEqual(frame['rift_mechanics']['rupture_strain_threshold'], .35)
        self.assertEqual(rifts.validate_frame(frame), 0)
        self.assertEqual(rifts.validate_frame({}), 0)
        for enhanced in (True, False):
            new, _ = criterion_world(enhanced)
            new.rift_systems, new.rift_mechanics = [], dict(rifts._rupture_calibration(new))
            frame = rifts.snapshot(new)
            self.assertEqual(frame['rupture_criterion_version'], 1)
            self.assertEqual(rifts.validate_frame(frame), 1)
        good = rifts.snapshot(new)
        bad = []
        for version in (True, 2, 1., '1'):
            bad.append(dict(good, rupture_criterion_version=version))
        for value in (0., -100., float('nan'), float('inf'), True, '100', None):
            bad.append(dict(good, rift_mechanics=dict(good['rift_mechanics'], breakup_extension_km=value)))
        for value in (.5, 0., None, True):
            bad.append(dict(good, rift_mechanics=dict(good['rift_mechanics'], realized_extension_fraction=value)))
        bad.append(dict(good, rift_mechanics=dict(good['rift_mechanics'], rupture_strain_threshold=.35)))
        # Legacy labels (the last world built above): the fraction and the
        # extension measure must belong to the recorded mechanics mode.
        self.assertEqual(good['rift_mechanics']['realized_extension_fraction'], .25)
        bad.append(dict(good, rift_mechanics=dict(good['rift_mechanics'], realized_extension_fraction=1.)))
        bad.append(dict(good, rift_mechanics=dict(good['rift_mechanics'],
                                                  damage_strain_measure=rifts.REALIZED_STRAIN_MEASURE)))
        bad.append(dict(good, rift_mechanics=dict(good['rift_mechanics'],
                                                  rupture_extension_measure=rifts.REALIZED_EXTENSION_MEASURE)))
        for name in ('damage_strain_measure', 'rupture_criterion', 'rupture_extension_measure'):
            bad.append(dict(good, rift_mechanics={key: value for key, value in good['rift_mechanics'].items()
                                                  if key != name}))
        bad.append(dict(good, rift_mechanics=dict(good['rift_mechanics'], rupture_criterion_version=0)))
        bad.append({key: value for key, value in good.items() if key != 'rift_mechanics'})
        bad.append(dict(good, rift_mechanics=[]))
        bad.append({key: value for key, value in good.items() if key != 'rupture_criterion_version'})
        for index, frame in enumerate(bad):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    rifts.validate_frame(frame)


if __name__ == '__main__':
    unittest.main()
