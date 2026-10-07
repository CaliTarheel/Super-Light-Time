"""Force-limit rifting on an ordinary world whose loads overload an ocean plate."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

import checkpoint
import force_rifting
import native_engine

# Four primordial ocean plates with coastal inherited slabs through the whole
# upper mantle (660 km). Primordial ocean 02 is pulled by ~15,000 km of trench
# and its worst cut, through its young (15-44 Myr) ocean, has a loading ratio
# of about 3 at t = 0; every continental block stays below 1.
WORLD = dict(width=48, height=24, mesh_level=4, coast_geometry_level=2, plate_count=4,
             mechanics_nodes=128, seed=37, physics_profile='reviewed_v1',
             primordial_subduction={'enabled': True, 'initial_slab_depth_km': 660.},
             primordial_ocean={'enabled': True})
FAST = dict(enabled=True, rift_width_km=10., breakup_stretch=2., ocean_breakup_opening_km=1.)


class ConfigurationTests(unittest.TestCase):
    def test_defaults_ranges_and_profile(self):
        self.assertFalse(force_rifting.normalize(None)['enabled'])
        settings = force_rifting.normalize(dict(enabled=True))
        self.assertEqual((settings['rift_width_km'], settings['breakup_stretch'],
                          settings['ocean_breakup_opening_km']), (150., 3., 30.))
        for bad in (dict(enabled=1), dict(breakup_stretch=1.), dict(rift_width_km=0.), dict(heal_myr=True),
                    dict(commit_opening_km=100.)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                force_rifting.normalize(bad)
        with self.assertRaisesRegex(ValueError, 'reviewed_v1'):
            native_engine.validate_config(dict(force_limit_rifting=dict(enabled=True)))

    def test_mechanisms_match_by_their_smaller_sides(self):
        cells = list(range(100))
        # The same cut with its sides labelled the other way round.
        self.assertEqual(force_rifting._overlap(force_rifting._smaller_side(range(70, 100), cells),
                                                force_rifting._smaller_side(range(70), cells)), 1.)
        # A different, smaller corner piece: the large remainders overlap by
        # more than the threshold, but the torn-off sides barely do.
        different = force_rifting._overlap(force_rifting._smaller_side(range(97), cells),
                                           force_rifting._smaller_side(range(70), cells))
        self.assertLess(different, force_rifting.SAME_MECHANISM_OVERLAP)


class OceanTearTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = native_engine.Simulation(dict(WORLD, force_limit_rifting=FAST))
        cls.stepped = deepcopy(cls.world)
        cls.stepped.step(.5)

    def test_overloaded_ocean_plate_tears_without_moving_continent(self):
        before, after = self.world, self.stepped
        rifts = [e for e in after.events if e['type'] == 'rift']
        self.assertEqual(len(rifts), 1)
        details = rifts[0]['details']
        self.assertEqual(details['setting'], 'oceanic')
        self.assertEqual(details['loading']['cause'], 'force_limit')
        self.assertGreaterEqual(details['loading']['loading_ratio'], 1.)
        self.assertEqual(after.force_rifting_diagnostics['commits'], 1)
        self.assertEqual(int(after.active.sum()), int(before.active.sum())+1)
        # An ocean-only tear moves no continental material between plates: the
        # new plate carries none, and each plate keeps its continental mass.
        new_uid = details['new_plate_uid']
        new_slot = int(np.flatnonzero(after.active & (after.plate_uid == new_uid))[0])
        self.assertFalse(np.any((after.parcel_plate == new_slot) & (after.kind > 0)))

        def continental(s):
            return {int(s.plate_uid[p]): float(s.mass[(s.parcel_plate == p) & (s.kind > 0)].sum())
                    for p in np.flatnonzero(s.active)}
        start, end = continental(before), continental(after)
        for uid, mass in start.items():
            self.assertAlmostEqual(end.get(uid, 0.), mass, delta=1e-6*max(mass, 1.))
        self.assertAlmostEqual(float(after.mass.sum()), float(before.mass.sum()), delta=1e-6*float(before.mass.sum()))

    def test_opening_boundaries_carry_no_unresolved_resistance(self):
        # A rift or ridge that is pulling apart has no slab drag, bending or
        # megathrust resistance. Placing the balance's unresolved resistance
        # there pinned new rifts back and tore strips along them.
        import plate_balance
        import plate_limit_analysis
        s = deepcopy(self.world)
        balance = plate_balance.Balance(s, .5)
        plate = int(np.bincount(np.asarray(s.bp)[np.asarray(s.bl) > 0.]).argmax())
        mine = np.flatnonzero(((np.asarray(s.bp) == plate) | (np.asarray(s.bq) == plate)) & (np.asarray(s.bl) > 0.))
        opened = mine[:len(mine)//2]
        s.normal_speed = np.asarray(s.normal_speed).copy()
        s.normal_speed[opened] = 5.
        s.normal_speed[mine[len(mine)//2:]] = 0.
        cells, _, loads = plate_limit_analysis.plate_loads(s, plate, balance)
        weight = loads['resistance_weight']
        band = np.cos(s.config['deformation_width_km']/6371.)
        points = np.asarray(s.xyz)[cells]
        near_open = (points@np.asarray(balance.radial)[opened].T >= band).any(axis=1)
        near_closed = (points@np.asarray(balance.radial)[mine[len(mine)//2:]].T >= band).any(axis=1)
        only_open = near_open & ~near_closed
        self.assertTrue(only_open.any())
        self.assertTrue(np.all(weight[only_open] == 0.))
        self.assertGreater(weight[near_closed].sum(), .99)

    def test_breakup_distance_follows_cut_composition_and_rift_weakens(self):
        s = native_engine.Simulation(dict(WORLD, force_limit_rifting={'enabled': True}))
        force_rifting.update(s, .5)
        overloaded = [row for row in s.force_rifting_state.values() if row['ratio'] >= 1.]
        self.assertTrue(overloaded)
        row = max(overloaded, key=lambda r: r['ratio'])
        uid = next(u for u, r in s.force_rifting_state.items() if r is row)
        continental = row['cut_continental_fraction']
        self.assertAlmostEqual(row['breakup_opening_km'], continental*300.+(1.-continental)*30.)
        first = row['ratio']
        s.t += .5
        force_rifting.update(s, .5)
        row = s.force_rifting_state[uid]
        # The same cut, now stretched and weaker, is closer to failure.
        self.assertGreater(row['stretch'], 1.)
        self.assertGreater(row['ratio'], first)

    def test_check_state_survives_checkpoint(self):
        s = deepcopy(self.stepped)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'force.npz'
            checkpoint.write_checkpoint(path, s, {'config': s.config}, {})
            restored, _ = checkpoint.read_checkpoint(path, {}, native_engine.Simulation)
        self.assertEqual(force_rifting.snapshot(restored)['force_rifting_diagnostics'],
                         force_rifting.snapshot(s)['force_rifting_diagnostics'])

    def test_inherited_rift_weakens_only_its_cells(self):
        import plate_limit_analysis
        s = deepcopy(self.world)
        self.assertTrue(np.all(plate_limit_analysis.inherited_weakness(s) <= 1.))
        s.rift_id = np.zeros_like(s.rift_id)
        s.suture = np.where(s.kind == 2, .02, np.where(s.kind == 1, .2, s.suture))
        intact = plate_limit_analysis.inherited_weakness(s)
        self.assertTrue(np.allclose(intact, 1.))
        cells = np.unique(s.parcel_cell[s.kind == 1])[:5]
        mine = np.isin(s.parcel_cell, cells) & (s.kind == 1)
        s.rift_id[mine] = 1
        weak = plate_limit_analysis.inherited_weakness(s)
        self.assertTrue(np.allclose(np.delete(weak, cells), 1.))
        # Wholly continental cells on the belt carry the native weak-belt
        # resistance, (1 + 1.2*0.2)/(1 + 1.2*0.8).
        pure = [c for c in cells if np.all(s.kind[s.parcel_cell == c] == 1)
                and s.crust[c] > 0]
        self.assertTrue(pure)
        for c in pure:
            self.assertAlmostEqual(weak[c], 1.24/1.96)
        with self.assertRaises(ValueError):
            force_rifting.normalize(dict(inherited_weakness=1))

    def test_disabled_policy_changes_nothing(self):
        s = native_engine.Simulation(WORLD)
        self.assertFalse(force_rifting.enabled(s))
        force_rifting.update(s, 1.)
        self.assertFalse(force_rifting.commit(s))
        self.assertEqual(force_rifting.snapshot(s), {})


if __name__ == '__main__':
    unittest.main()
