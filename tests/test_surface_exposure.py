"""Surface identity is continuous only where its own material still exists."""
from copy import deepcopy
import unittest

import numpy as np

from crust_transport import deposit, footprints
import geology_snapshot
import structure_engine
from surface_exposure import resolve
from raster_engine import Simulation, _rotate, _xyz
from tests.test_local_accretion import world


def overlapping_sheets(width=64):
    """Two unbroken finite-footprint sheets with tiny alternating area excess."""
    height = width // 2
    cells = np.arange(width * height)
    s = world(width, height, source=cells, target=cells)
    count = 4 * s.n
    # Match actual continental creation: four distinct quarter-cell samples
    # with their exact spherical areas, rather than four coincident centres.
    row, col = np.divmod(np.repeat(cells, 4), width)
    oy = np.tile([.25, .25, .75, .75], s.n)
    ox = np.tile([.25, .75, .25, .75], s.n)
    positions = _xyz((col+ox)*2*np.pi/width-np.pi,
                     np.pi/2-(row+oy)*np.pi/height)
    top = np.pi/2-(row+np.where(oy < .5, 0., .5))*np.pi/height
    masses = (np.sin(top)-np.sin(top-.5*np.pi/height))*np.pi/width*6371.**2
    s.pos = np.concatenate((positions, positions))
    s.mass = np.tile(masses, 2)
    s.parcel_east, s.parcel_extent = footprints(s.pos, width, height)
    s.parcel_plate = np.repeat([0, 1], count).astype(np.int16)
    s.parcel_patch = np.repeat(np.arange(2*s.n) + 1000, 4)
    s.kind[:] = 1
    s.relief[:count] = 420.
    s.relief[count:] = 1420.
    y, x = np.divmod(cells, width)
    excess = np.repeat(np.where((x+y) % 2, .025, -.025), 4)
    s.mass[:count] *= 1. + excess
    s.mass[count:] *= 1. - excess
    s.plate = (s.xyz[:, 0] < 0).astype(np.int16)
    s.age = np.zeros(s.n)
    structure_engine.initialize_parcels(s)
    return s


def independent_deposit(s):
    return deposit(
        s.pos, s.parcel_east, s.parcel_extent, s.mass, s.kind,
        structure_engine.material_height(s.kind, s.relief), s.suture,
        s.parcel_plate, s.w, s.h, s.xyz, s.cell_area,
        s._sample_coordinates, extra_fields=geology_snapshot.deposit_fields(s))


class SurfaceExposureTests(unittest.TestCase):
    def test_minor_area_alternation_does_not_cut_two_coherent_sheets(self):
        h, w = 16, 32
        y, x = np.indices((h, w))
        previous = (x >= w//2).astype(np.int16).ravel()
        dominant = ((x+y) % 2).astype(np.int16).ravel()
        occupancy = {0: np.arange(h*w), 1: np.arange(h*w)}
        original = [previous.copy(), dominant.copy(), deepcopy(occupancy)]
        actual = resolve(previous, dominant, np.ones(h*w, bool), occupancy)
        np.testing.assert_array_equal(actual, previous)
        self.assertGreater(np.count_nonzero(actual != dominant), h*w//3)
        np.testing.assert_array_equal(previous, original[0])
        np.testing.assert_array_equal(dominant, original[1])
        for key in occupancy:
            np.testing.assert_array_equal(occupancy[key], original[2][key])

    def test_lost_thin_and_newly_exposed_material_yield_to_present_sheet(self):
        # Columns 0/1 retain an independently visible owner; 2/3 have lost
        # that footprint. At 4, two subthreshold sheets jointly make land,
        # but neither receives a historical surface-ownership privilege.
        previous = np.array([0, 1, 0, 1, 0, 2], np.int16)
        dominant = np.array([1, 0, 1, 0, 1, 0], np.int16)
        occupancy = {0: np.array([0, 1, 3]), 1: np.array([0, 1, 2])}
        actual = resolve(previous, dominant,
                         np.array([1, 1, 1, 1, 1, 0], bool), occupancy)
        np.testing.assert_array_equal(actual, [0, 1, 1, 0, 1, 2])

    def test_global_raster_preserves_material_height_and_coherent_surface(self):
        s = overlapping_sheets()
        previous = s.plate.copy()
        reference = independent_deposit(s)
        self.assertGreater(np.count_nonzero(reference[6] != previous), s.n//3)
        material_fields = ('pos', 'parcel_east', 'parcel_extent', 'mass',
                           'kind', 'relief', 'suture', 'parcel_plate',
                           'parcel_patch', 'parcel_craton')
        before = {name: getattr(s, name).copy() for name in material_fields}
        columns = deepcopy(s.structure)
        Simulation._rasterize(s)
        np.testing.assert_array_equal(s.plate, previous)
        np.testing.assert_array_equal(s.land_mass, reference[0])
        np.testing.assert_array_equal(s.land_height, reference[1]/reference[0])
        self.assertAlmostEqual(float(s.land_mass.sum()/s.mass.sum()), 1., places=13)
        self.assertTrue(np.all(s.crust != 0))
        np.testing.assert_array_equal(s.support.sum(axis=0), np.ones(s.n))
        np.testing.assert_array_equal(s.support[s.plate, np.arange(s.n)], np.ones(s.n))
        for name, old in before.items():
            np.testing.assert_array_equal(getattr(s, name), old, err_msg=name)
        for name, old in columns.items():
            np.testing.assert_array_equal(s.structure[name], old, err_msg=name)
        # Repeated projection cannot manufacture transfers or alter the choice.
        Simulation._rasterize(s)
        np.testing.assert_array_equal(s.plate, previous)
        np.testing.assert_array_equal(s.land_height, reference[1]/reference[0])

    def test_actual_raster_does_not_pin_new_arc_beneath_absent_old_owner(self):
        s = overlapping_sheets()
        count = len(s.pos)//2
        s.mass[:count] *= .05  # Residual material remains, below visibility.
        s.kind[count:] = 3
        s.plate[:] = 0
        structure_engine.initialize_parcels(s)
        mass = s.mass.copy()
        owner = s.parcel_plate.copy()
        Simulation._rasterize(s)
        self.assertEqual(len(s._owner_occupancy[0]), 0)
        self.assertEqual(len(s._owner_occupancy[1]), s.n)
        np.testing.assert_array_equal(s.plate, np.ones(s.n, np.int16))
        np.testing.assert_array_equal(s.crust, np.full(s.n, 3, np.uint8))
        np.testing.assert_array_equal(s.mass, mass)
        np.testing.assert_array_equal(s.parcel_plate, owner)

    def test_rotated_finite_sheets_cover_seam_and_both_polar_caps(self):
        for rotation in ([0., 0., np.pi], [0., np.pi/2, 0.],
                         [0., -np.pi/2, 0.]):
            with self.subTest(rotation=rotation):
                s = overlapping_sheets()
                vector = np.asarray(rotation)
                s.pos = _rotate(s.pos, vector)
                s.parcel_east = _rotate(s.parcel_east, vector)
                normal = _rotate(np.array([[1., 0., 0.]]), vector)[0]
                s.plate = (s.xyz @ normal < 0).astype(np.int16)
                previous = s.plate.copy()
                total = s.mass.sum()
                Simulation._rasterize(s)
                # This is a physical rotation of finite material, not just
                # permutation of raster indices at the poles or date line.
                self.assertTrue(np.all(s.crust != 0))
                for owner in (0, 1):
                    self.assertEqual(len(s._owner_occupancy[owner]), s.n)
                np.testing.assert_array_equal(s.plate, previous)
                self.assertAlmostEqual(float(s.land_mass.sum()/total), 1., places=13)
                self.assertTrue(np.isfinite(s.land_height).all())


if __name__ == '__main__':
    unittest.main()
