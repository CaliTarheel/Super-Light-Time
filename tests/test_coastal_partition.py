"""Ocean continuations of continental fractures, including recorded fixtures."""
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from coastal_partition import extend_continental_partition, bounded_continental_partition
from fracture import make_fracture
from raster_engine import Simulation, DEFAULT_CONFIG, RADIUS_KM


def geometry(width, height):
    rows = np.pi / 2 - np.arange(height + 1) * np.pi / height
    cells = np.arange(width * height).reshape(height, width)
    area = np.repeat((np.sin(rows[:-1]) - np.sin(rows[1:])) * 2 * np.pi / width * RADIUS_KM**2, width)
    a = np.r_[cells.ravel(), cells[:-1].ravel()]
    b = np.r_[np.roll(cells, -1, axis=1).ravel(), cells[1:].ravel()]
    length = np.r_[np.full(width * height, np.pi * RADIUS_KM / height),
                   np.repeat(2 * np.pi * RADIUS_KM / width * np.cos(rows[1:-1]), width)]
    return area, a, b, length


def load_fixture(time):
    with np.load(Path(__file__).parent / 'fixtures' / f'continental_ocean_cut_{time}.npz') as data:
        return {key: data[key].copy() for key in data.files}


class CoastalPartitionTests(unittest.TestCase):
    def test_recorded_194_myr_cut_shortens_ocean_break_without_changing_continent(self):
        data = load_fixture(194)
        width, height = int(data['width']), int(data['height'])
        region, detail = bounded_continental_partition(
            data['parent'], data['continental'], data['child_side'], width, height, *geometry(width, height))
        np.testing.assert_array_equal(region[data['continental']], data['child_side'][data['continental']])
        self.assertFalse(np.any(region & ~data['parent']))
        self.assertEqual(detail['method'], 'coastal shortest-path continuation')
        self.assertLess(detail['ocean_interface_km'], detail['original_ocean_interface_km'] * .25)
        self.assertLessEqual(detail['new_disconnected_area_fraction'], .025)
        self.assertGreater(detail['ocean_cells_reassigned'], 1000)

    def test_recorded_358_myr_cut_falls_back_when_coasts_need_a_longer_link(self):
        data = load_fixture(358)
        width, height = int(data['width']), int(data['height'])
        region, detail = bounded_continental_partition(
            data['parent'], data['continental'], data['child_side'], width, height, *geometry(width, height))
        np.testing.assert_array_equal(region, data['child_side'])
        self.assertEqual(detail['method'], 'original curved cut')
        self.assertGreaterEqual(detail['candidate_ocean_interface_km'], detail['original_ocean_interface_km'])

    def test_existing_other_plate_is_a_barrier_and_unattached_ocean_stays_parent(self):
        parent = np.zeros((12, 24), bool)
        parent[2:9, 2:12] = True
        parent[3:7, 17:21] = True
        continent = np.zeros_like(parent)
        continent[3:8, 3:6] = True
        child = np.zeros_like(parent)
        child[3:5, 3:6] = True
        child[3:7, 17:21] = True  # Old infinite cut crosses this unconnected ocean.
        region = extend_continental_partition(parent, continent, child, 24, 12).reshape(12, 24)
        self.assertFalse(np.any(region[~parent]))
        self.assertFalse(np.any(region[3:7, 17:21]))
        np.testing.assert_array_equal(region[continent], child[continent])

    def test_longitude_seam_does_not_change_the_continuation(self):
        parent = np.ones((12, 24), bool)
        continent = np.zeros_like(parent)
        continent[3:9, [0, 1, 22, 23]] = True
        child = np.zeros_like(parent)
        child[3:6] = True
        base = extend_continental_partition(parent, continent, child, 24, 12).reshape(12, 24)
        moved = extend_continental_partition(np.roll(parent, 7, 1), np.roll(continent, 7, 1),
                                            np.roll(child, 7, 1), 24, 12).reshape(12, 24)
        np.testing.assert_array_equal(moved, np.roll(base, 7, 1))

    def test_polar_continuation_reaches_the_same_cap_not_the_opposite_pole(self):
        parent = np.ones((8, 16), bool)
        continent = np.zeros_like(parent)
        continent[0, 0] = continent[-1, 0] = True
        child = np.zeros_like(parent)
        child[0, 0] = True
        region = extend_continental_partition(parent, continent, child, 16, 8).reshape(8, 16)
        self.assertTrue(region[0, 8])
        self.assertFalse(region[-1, 8])

    def test_all_continental_crack_is_unchanged(self):
        parent = np.ones((8, 16), bool)
        child = np.zeros_like(parent)
        child[:, :8] = True
        region, detail = bounded_continental_partition(parent, parent, child, 16, 8, *geometry(16, 8))
        np.testing.assert_array_equal(region, child.ravel())
        self.assertEqual(detail['ocean_cells_reassigned'], 0)

    @staticmethod
    def simulation():
        width, height = 64, 32
        simulation = Simulation(dict(DEFAULT_CONFIG, width=width, height=height, seed=37, plate_count=4),
                                {'width': width, 'height': height, 'crust': np.ones(width * height, np.uint8)})
        simulation.active[:] = False
        simulation.active[0] = True
        simulation.plate[:] = simulation.parcel_plate[:] = simulation.trace_plate[:] = 0
        simulation.support[:] = 0
        simulation.support[0] = 1
        simulation.t = 200
        return simulation

    def test_hidden_material_and_craton_owners_keep_the_original_whole_patch_cut(self):
        simulation = self.simulation()
        # Subthreshold display deliberately hides some buoyant material. The
        # new ocean continuation must never override its material-side test.
        simulation.crust[np.abs(simulation.xyz[:, 2]) > .4] = 0
        crack = make_fracture(np.array([1., 0., 0.]), np.array([0., 1., 0.]), 7)
        parcel_side, trace_side = simulation._material_cut(crack)
        selected_patch = simulation.parcel_patch[np.flatnonzero(parcel_side)[0]]
        craton = simulation.parcel_patch == selected_patch
        simulation.parcel_craton[craton] = 42
        simulation.kind[craton] = 2
        before = {name: getattr(simulation, name).copy() for name in ('mass', 'kind', 'pos', 'parcel_patch', 'parcel_craton', 'trace_id', 'trace_xyz')}
        choice = dict(crack=crack, fraction=float(np.sum(simulation.mass[parcel_side]) / simulation.mass.sum()),
                      grid_distance=crack.signed_distance(simulation.xyz), disconnected_fraction=0.)
        with patch.object(simulation, '_choose_fracture', return_value=choice):
            self.assertTrue(simulation._split(0))
        q = int(simulation.events[-1]['plate_ids'][1])
        np.testing.assert_array_equal(simulation.parcel_plate, np.where(parcel_side, q, 0))
        np.testing.assert_array_equal(simulation.trace_plate, np.where(trace_side, q, 0))
        for name, expected in before.items():
            np.testing.assert_array_equal(getattr(simulation, name), expected)
        self.assertEqual(len(np.unique(simulation.parcel_plate[craton])), 1)
        np.testing.assert_allclose(simulation.support.sum(axis=0), 1)
        self.assertIn('ocean_continuation', simulation.events[-1]['details'])

    def test_explicit_ocean_fracture_does_not_use_continental_continuation(self):
        simulation = self.simulation()
        simulation.crust[:] = 0
        crack = make_fracture(np.array([1., 0., 0.]), np.array([0., 1., 0.]), 9)
        choice = dict(crack=crack, fraction=.5, grid_distance=crack.signed_distance(simulation.xyz), disconnected_fraction=0.)
        with patch.object(simulation, '_choose_fracture', return_value=choice), \
                patch('coastal_partition.bounded_continental_partition', side_effect=AssertionError('Continental-only rule used in ocean')):
            self.assertTrue(simulation._split_ocean(0))
        self.assertEqual(simulation.events[-1]['details']['setting'], 'oceanic')


if __name__ == '__main__':
    unittest.main()
