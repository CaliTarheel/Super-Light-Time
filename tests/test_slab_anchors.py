"""Located slab allocation: slab pulls where it went down, inventory conserved."""
from copy import deepcopy
import math
import unittest

import numpy as np

import native_engine
import slab_anchors
import slab_memory


def unit(lat, lon):
    lat, lon = math.radians(lat), math.radians(lon)
    return np.array([math.cos(lat)*math.cos(lon), math.cos(lat)*math.sin(lon), math.sin(lat)])


def row_with(anchors):
    row = dict(slab_retained_area_km2=sum(a[1] for a in anchors),
               slab_retained_excess_mass_kg=sum(a[2] for a in anchors))
    slab_anchors._store(row, np.array([a[0] for a in anchors]), np.array([a[1] for a in anchors]),
                        np.array([a[2] for a in anchors]))
    return row


class AnchorUnitTests(unittest.TestCase):
    def test_policy_values(self):
        self.assertEqual(slab_anchors.normalize(), 'uniform')
        with self.assertRaises(ValueError):
            slab_anchors.normalize('everywhere')
        with self.assertRaisesRegex(ValueError, 'reviewed_v1'):
            native_engine.validate_config(dict(slab_allocation='fed_v1'))

    def test_edges_far_from_any_anchor_do_not_pull(self):
        row = row_with([(unit(0, 0), 100., 1e18)])
        mids = np.array([unit(0, 1), unit(0, 2), unit(0, 30)])
        mass, area, unattached = slab_anchors.edge_loads(row, np.arange(3), mids, np.array([100., 300., 100.]))
        self.assertAlmostEqual(mass[:2].sum(), 1e18)
        self.assertAlmostEqual(mass[1]/mass[0], 3.)          # shared by length
        self.assertEqual(mass[2], 0.)
        self.assertEqual(unattached, 0.)
        stranded = row_with([(unit(40, 90), 10., 5e17)])
        _, _, unattached = slab_anchors.edge_loads(stranded, np.arange(3), mids, np.ones(3))
        self.assertEqual(unattached, 5e17)

    def test_feed_lands_where_consumed_and_totals_match(self):
        row = row_with([(unit(0, 0), 100., 1e18)])
        decay = .9
        # slab_memory advances the totals first; mirror that here.
        row['slab_retained_area_km2'] = 100.*decay+50.
        row['slab_retained_excess_mass_kg'] = 1e18*decay+4e17
        slab_anchors.advance(row, decay, 1., [(unit(0, 20), 50., 1.)], 50., 4e17)
        xyz, area, mass = slab_anchors._arrays(row)
        self.assertEqual(len(xyz), 2)
        self.assertAlmostEqual(area.sum(), row['slab_retained_area_km2'])
        self.assertAlmostEqual(mass.sum()/row['slab_retained_excess_mass_kg'], 1.)
        new = int(np.argmax(xyz@unit(0, 20)))
        self.assertAlmostEqual(mass[new]/4e17, 1.)
        slab_anchors.validate(row)

    def test_partition_by_place_and_join(self):
        parent = row_with([(unit(0, 0), 60., 6e17), (unit(0, 40), 40., 4e17)])
        child = {}
        area_share, mass_share = slab_anchors.partition(parent, child, [unit(0, 41)], [unit(0, 1)])
        self.assertAlmostEqual(area_share, .4)
        self.assertAlmostEqual(mass_share, .4)
        self.assertAlmostEqual(slab_anchors._arrays(child)[2].sum(), 4e17)
        slab_anchors.join(child, parent)
        self.assertAlmostEqual(slab_anchors._arrays(parent)[2].sum(), 1e18)
        self.assertEqual(child[slab_anchors.FIELD], [])

    def test_merging_bounds_state_and_conserves(self):
        points = [unit(0, x*.3) for x in range(400)]
        row = row_with([(p, 1., 1e15) for p in points])
        slab_anchors._merge(row)
        xyz, area, mass = slab_anchors._arrays(row)
        self.assertLessEqual(len(xyz), slab_anchors.MAX_ANCHORS)
        self.assertAlmostEqual(area.sum(), 400.)
        self.assertAlmostEqual(mass.sum()/4e17, 1.)


class LocatedEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = dict(width=48, height=24, mesh_level=4, coast_geometry_level=2, plate_count=4,
                      mechanics_nodes=128, seed=37, physics_profile='reviewed_v1',
                      primordial_subduction={'enabled': True},
                      primordial_ocean={'enabled': True, 'continental_attachments':
                                        [dict(region_index=1, continental_plate_uid=3)]},
                      slab_allocation='fed_v1')
        cls.uniform = native_engine.Simulation(dict(config, slab_allocation='uniform'))
        cls.world = native_engine.Simulation(config)
        cls.stepped = deepcopy(cls.world)
        cls.stepped.step(1.)
        cls.stepped.step(1.)

    def test_initial_located_pull_equals_the_uniform_declaration(self):
        # Inherited slabs are declared uniform along their traces, so the
        # located law starts from exactly the same line forces.
        o1, l1, _ = slab_memory.line_load(self.uniform)
        o2, l2, _ = slab_memory.line_load(self.world)
        real = np.asarray(self.world.bl) > 1e-9   # zero-length contacts carry no force
        np.testing.assert_array_equal(o1[real], o2[real])
        np.testing.assert_allclose(l1[real], l2[real], rtol=1e-9)

    def test_inventory_conserved_through_evolution(self):
        s = self.stepped
        self.assertEqual(s.slab_allocation_version, 1)
        for row in s.trench_systems:
            slab_memory.validate_row(row, require_mass=True)
        total = sum(r['slab_retained_excess_mass_kg'] for r in s.trench_systems)
        anchored = sum(a['excess_mass_kg'] for r in s.trench_systems for a in r.get('slab_anchors', []))
        self.assertAlmostEqual(anchored/total, 1., places=9)
        fed = [r for r in s.trench_systems if r['slab_fed_area_km2'] > 0.]
        self.assertTrue(fed)

    def test_pull_follows_the_slab_not_the_whole_trace(self):
        s = self.stepped
        owners, load, _ = slab_memory.line_load(s)
        ids = np.asarray(s.trench_id)
        varied = 0
        for row in s.trench_systems:
            edges = np.flatnonzero((ids == row['id']) & (owners >= 0))
            if len(edges) > 2 and np.ptp(load[edges]) > 1e-6*load[edges].max():
                varied += 1
        self.assertGreater(varied, 0)


if __name__ == '__main__':
    unittest.main()
