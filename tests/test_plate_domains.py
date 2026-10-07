import unittest

import numpy as np

from plate_domains import PersistentDomainTracker
from raster_engine import Simulation


class PersistentDomainTests(unittest.TestCase):
    def setup_map(self):
        w, h = 24, 12
        grid = np.zeros((h, w), np.int16)
        grid[3:8, 3:10] = 1
        return w, h, grid, np.ones(w*h), np.array([11, 22]), ['Ocean', 'Continent']

    def test_advected_overlap_preserves_names_and_does_not_change_input(self):
        w, h, grid, area, uids, names = self.setup_map()
        tracker = PersistentDomainTracker(w, h)
        first = tracker.update(grid, area, uids, names, 0)
        moved = np.roll(grid, 10, axis=1)
        previous = np.roll(np.arange(w*h).reshape(h, w), 10, axis=1).ravel()
        untouched = moved.copy()
        second = tracker.update(moved, area, uids, names, 2, previous)
        np.testing.assert_array_equal(second.domain.reshape(h, w),
                                      np.roll(first.domain.reshape(h, w), 10, axis=1))
        np.testing.assert_array_equal(moved, untouched)
        self.assertEqual(second.created, [])
        self.assertEqual(second.retired_uids, [])

    def test_split_reunion_and_second_split_never_reuse_retired_identity(self):
        w, h, grid, area, uids, names = self.setup_map()
        tracker = PersistentDomainTracker(w, h)
        first = tracker.update(grid, area, uids, names, 0)
        parent = int(first.domain[4*w+4])
        divided = grid.copy()
        divided[3:8, 7] = 0
        split = tracker.update(divided, area, uids, names, 2)
        child = int(split.domain[4*w+8])
        self.assertEqual(split.domain[4*w+4], parent)
        self.assertNotEqual(child, parent)
        record = next(row for row in split.created if row['uid'] == child)
        self.assertEqual(record['source_domain_uid'], parent)
        self.assertEqual(record['parent_plate_uid'], 22)
        self.assertEqual(record['name'], f'Fragment {child:03d}')
        reunited = tracker.update(grid, area, uids, names, 4)
        self.assertEqual(reunited.domain[4*w+8], parent)
        self.assertIn(child, reunited.retired_uids)
        again = tracker.update(divided, area, uids, names, 6)
        self.assertNotEqual(again.domain[4*w+8], child)
        self.assertGreater(again.domain[4*w+8], child)

    def test_area_weighted_overlap_beats_cell_count(self):
        w, h, grid, area, uids, names = self.setup_map()
        area.reshape(h, w)[3:8, 8:10] = 10
        tracker = PersistentDomainTracker(w, h)
        first = tracker.update(grid, area, uids, names, 0)
        parent = first.domain[4*w+4]
        grid[3:8, 7] = 0
        second = tracker.update(grid, area, uids, names, 2)
        self.assertEqual(second.domain[4*w+8], parent)
        self.assertNotEqual(second.domain[4*w+4], parent)

    def test_kinematic_slot_reuse_cannot_inherit_an_old_domain_uid(self):
        w, h, grid, area, uids, names = self.setup_map()
        tracker = PersistentDomainTracker(w, h)
        first = tracker.update(grid, area, uids, names, 0)
        previous = int(first.domain[4*w+4])
        uids[1] = 99
        names[1] = 'New rift plate'
        second = tracker.update(grid, area, uids, names, 2)
        self.assertNotEqual(second.domain[4*w+4], previous)
        record = next(row for row in second.domains if row['parent_plate_uid'] == 99)
        self.assertEqual(record['name'], 'New rift plate')
        self.assertIsNone(record['source_domain_uid'])

    def test_disappeared_domain_gets_fresh_identity_if_it_reappears(self):
        w, h, grid, area, uids, names = self.setup_map()
        tracker = PersistentDomainTracker(w, h)
        first = tracker.update(grid, area, uids, names, 0)
        previous = int(first.domain[4*w+4])
        vanished = tracker.update(np.zeros_like(grid), area, uids, names, 2)
        self.assertIn(previous, vanished.retired_uids)
        reappeared = tracker.update(grid, area, uids, names, 4)
        self.assertNotEqual(reappeared.domain[4*w+4], previous)

    def test_wrap_and_pole_connected_regions_receive_one_identity(self):
        w, h, grid, area, uids, names = self.setup_map()
        grid[:] = 0
        grid[:4, :2] = 1
        grid[:4, -2:] = 1
        grid[:4, w//2-1:w//2+1] = 1
        tracker = PersistentDomainTracker(w, h)
        state = tracker.update(grid, area, uids, names, 0)
        self.assertEqual(len(state.domains), 2)
        self.assertEqual(len(np.unique(state.domain[grid.ravel() == 1])), 1)

    def test_initial_multiple_regions_get_distinct_names_and_one_primary(self):
        w, h, grid, area, uids, names = self.setup_map()
        grid[8:10, 17:20] = 1
        tracker = PersistentDomainTracker(w, h)
        state = tracker.update(grid, area, uids, names, 0)
        records = [row for row in state.domains if row['parent_plate_uid'] == 22]
        self.assertEqual(len(records), 2)
        self.assertEqual(len(set(row['name'] for row in records)), 2)
        self.assertEqual(sum(row['primary'] for row in records), 1)
        main = next(row for row in records if row['primary'])
        self.assertEqual(main['name'], 'Continent')
        self.assertEqual(state.domain[4*w+4], main['uid'])

    def test_matching_is_deterministic_and_unknown_source_cells_are_new(self):
        w, h, grid, area, uids, names = self.setup_map()
        a, b = PersistentDomainTracker(w, h), PersistentDomainTracker(w, h)
        for t in range(4):
            altered = grid.copy()
            if t % 2:
                altered[:, 7] = 0
            left = a.update(altered, area, uids, names, t*2)
            right = b.update(altered, area, uids, names, t*2)
            np.testing.assert_array_equal(left.domain, right.domain)
            self.assertEqual(left.domains, right.domains)
        old = set(left.domain.tolist())
        new = a.update(grid, area, uids, names, 10, np.full(w*h, -1))
        self.assertTrue(old.isdisjoint(set(new.domain.tolist())))


class InitialDomainIntegrationTests(unittest.TestCase):
    def test_ring_continent_keeps_one_initial_ocean_motion_and_distinct_basins(self):
        w, h = 64, 32
        yy, xx = np.mgrid[:h, :w]
        radius = ((xx-32)/13)**2+((yy-16)/10)**2
        crust = ((radius < 1) & (radius > .3)).astype(np.uint8)
        sim = Simulation(dict(width=w, height=h, plate_count=4, duration_myr=2),
                         initial=dict(width=w, height=h, crust=crust.ravel()))
        np.testing.assert_array_equal(sim.initial_crust, crust.ravel())
        np.testing.assert_array_equal(sim.crust, crust.ravel())
        ocean_slots = np.unique(sim.plate[crust.ravel() == 0])
        self.assertEqual(len(ocean_slots), 1)
        self.assertEqual(sim.plate_uid[ocean_slots[0]], sim.initial_ocean_plate_uid)
        ocean_domains = [row for row in sim.domains if row['parent_plate_uid'] == sim.initial_ocean_plate_uid]
        self.assertGreaterEqual(len(ocean_domains), 2)
        self.assertEqual(len({row['uid'] for row in ocean_domains}), len(ocean_domains))
        self.assertEqual(len({row['name'] for row in ocean_domains}), len(ocean_domains))
        self.assertTrue(all(row['plate_id'] == ocean_slots[0] for row in ocean_domains))
        self.assertEqual(sim.plate[16*w+32], sim.plate[0])
        self.assertNotEqual(sim.domain[16*w+32], sim.domain[0])

    def test_initial_independent_blocks_and_later_visible_fragments_have_unique_names(self):
        w, h = 64, 32
        crust = np.zeros((h, w), np.uint8)
        for start in (5, 18, 31, 44, 57):
            crust[10:16, start:start+5] = 1
        sim = Simulation(dict(width=w, height=h, plate_count=4, duration_myr=2),
                         initial=dict(width=w, height=h, crust=crust.ravel()))
        blocks = [p for p in np.flatnonzero(sim.active) if sim.names[p].startswith('Block ')]
        self.assertGreaterEqual(len(blocks), 2)
        before = {row['uid']: row['name'] for row in sim.domains}
        self.assertEqual(len(set(before.values())), len(before))
        # An occluding one-cell strip produces visible fragments while leaving
        # their underlying material and actual Euler-motion owner unchanged.
        p = blocks[0]
        mask = sim.plate.reshape(h, w) == p
        _, columns = np.where(mask)
        middle = int((columns.min()+columns.max())//2)
        grid = sim.plate.reshape(h, w)
        ocean = int(sim.plate[0])
        grid[mask & (np.arange(w)[None, :] == middle)] = ocean
        sim._update_surface_domains()
        names = [row['name'] for row in sim.domains]
        self.assertEqual(len(set(names)), len(names))
        self.assertTrue(any(name.startswith('Fragment ') for name in names))
        self.assertTrue(any(name.startswith('Block ') for name in names))
        self.assertTrue(np.any(sim.parcel_plate == p))


if __name__ == '__main__':
    unittest.main()
