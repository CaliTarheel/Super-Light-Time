"""Rigid transport checks against known spherical shapes, not terrain images."""
import unittest

import numpy as np

from raster_engine import Simulation, _rotate, _xyz


def spherical_cap(width=128, centre=(1., 0., 0.), channel=False):
    height = width // 2
    lon, lat = np.meshgrid((np.arange(width)+.5)*2*np.pi/width-np.pi,
                          np.pi/2-(np.arange(height)+.5)*np.pi/height)
    points = _xyz(lon.ravel(), lat.ravel())
    distance = np.arccos(np.clip(points @ np.asarray(centre), -1, 1))
    crust = np.where(distance < np.radians(35), 1, 0).astype(np.uint8)
    crust[distance < np.radians(15)] = 2
    if channel:
        # A three-cell-wide ocean passage through the initial continent.
        crust[np.abs(points[:, 1]) < np.sin(1.5*np.pi/height)] = 0
    sim = Simulation(dict(width=width, height=height, duration_myr=2),
                     dict(width=width, height=height, crust=crust))
    sim.parcel_plate[:] = 0
    sim.plate[:] = 0
    sim.support[:] = 0
    sim.support[0] = 1
    return sim


def rotate_material(sim, vector):
    sim.omega[:] = vector
    sim._advect(1)
    sim._rasterize()


class PolarTransportTests(unittest.TestCase):
    def assert_coherent_cap(self, sim, back, centre=(1., 0., 0.)):
        distance = np.arccos(np.clip(back @ np.asarray(centre), -1, 1))
        margin = 2*np.pi/sim.h
        interior = distance < np.radians(35)-margin
        craton = distance < np.radians(15)-margin
        far_ocean = distance > np.radians(35)+margin
        self.assertTrue(np.all(sim.crust[interior] > 0), "interior was perforated")
        self.assertTrue(np.all(sim.crust[craton] == 2), "craton was perforated")
        self.assertTrue(np.all(sim.crust[far_ocean] == 0), "invented remote land")
        self.assertTrue(np.all(sim.plate[interior] == 0), "unanchored interior owners")
        self.assertAlmostEqual(sim.land_mass.sum()/sim.original_mass, 1, places=12)
        self.assertAlmostEqual(sim.mass[sim.kind == 2].sum()/sim.original_craton_mass, 1, places=13)
        np.testing.assert_allclose(np.linalg.norm(sim.pos, axis=1), 1, atol=2e-14)
        np.testing.assert_allclose(np.sum(sim.pos*sim.parcel_east, axis=1), 0, atol=2e-14)
        # Raster coastlines may move by a subcell; material area cannot change.
        self.assertAlmostEqual(sim.cell_area[sim.crust > 0].sum()/sim.original_mass,
                               1, delta=.035)

    def test_equator_antimeridian_and_both_poles_keep_solid_interiors(self):
        for angle in (0, 70, 90, 110, -70, -90, -110, 180):
            with self.subTest(angle=angle):
                sim = spherical_cap()
                vector = np.array([0., -np.radians(angle), 0.])
                rotate_material(sim, vector)
                self.assert_coherent_cap(sim, _rotate(sim.xyz, -vector))

    def test_high_resolution_polar_cap_area_and_coverage(self):
        sim = spherical_cap(256)
        vector = np.array([0., -np.pi/2, 0.])
        rotate_material(sim, vector)
        self.assert_coherent_cap(sim, _rotate(sim.xyz, -vector))
        self.assertAlmostEqual(sim.cell_area[sim.crust > 0].sum()/sim.original_mass,
                               1, delta=.015)

    def test_oblique_rotations_keep_an_entire_continental_sphere_solid(self):
        # A solid sphere has no coastline uncertainty: any ocean cell is a
        # transport hole. Exercise all source and destination latitudes at once,
        # including a small oblique turn that gave the lowest coverage in the
        # seeded rotation probe, a roughly quarter-turn, and a near half-turn.
        width, height = 128, 64
        sim = Simulation(dict(width=width, height=height, duration_myr=2),
                         dict(width=width, height=height,
                              crust=np.ones(width*height, dtype=np.uint8)))
        sim.parcel_plate[:] = 0
        sim.plate[:] = 0
        sim.support[:] = 0
        sim.support[0] = 1
        sim.active[:] = False
        sim.active[0] = True
        positions, axes = sim.pos.copy(), sim.parcel_east.copy()
        vectors = ((-.0268839968, -.0312785548, .0484295682),
                   (-1.5433020467, .3660946016, -.2425796335),
                   (2.9148491556, .6401333590, .8104884510))
        for vector in vectors:
            with self.subTest(vector=vector):
                sim.pos, sim.parcel_east = positions.copy(), axes.copy()
                rotate_material(sim, np.asarray(vector))
                self.assertTrue(np.all(sim.crust > 0), "solid sphere was perforated")
                self.assertTrue(np.all(sim.plate == 0), "material lost its owner")
                self.assertTrue(np.isfinite(sim.density).all())
                self.assertAlmostEqual(sim.land_mass.sum()/sim.original_mass, 1, places=12)
                np.testing.assert_allclose(np.linalg.norm(sim.pos, axis=1), 1, atol=2e-14)
                np.testing.assert_allclose(np.sum(sim.pos*sim.parcel_east, axis=1), 0,
                                           atol=2e-14)

    def test_original_polar_wedges_can_travel_to_equator(self):
        sim = spherical_cap(128, centre=(0., 0., 1.))
        vector = np.array([0., np.pi/2, 0.])
        rotate_material(sim, vector)
        self.assert_coherent_cap(sim, _rotate(sim.xyz, -vector), centre=(0., 0., 1.))

    def test_roundtrip_does_not_accumulate_continental_remapping_loss(self):
        sim = spherical_cap()
        original = sim.crust.copy()
        positions, axes = sim.pos.copy(), sim.parcel_east.copy()
        for direction in (1, -1):
            for _ in range(12):
                rotate_material(sim, np.array([0., direction*np.pi/12, 0.]))
        np.testing.assert_allclose(sim.pos, positions, atol=3e-14)
        np.testing.assert_allclose(sim.parcel_east, axes, atol=3e-14)
        np.testing.assert_array_equal(sim.crust, original)
        self.assert_coherent_cap(sim, sim.xyz)

    def test_genuine_ocean_channel_survives_polar_crossing(self):
        for angle in (90, -90):
            with self.subTest(angle=angle):
                sim = spherical_cap(128, channel=True)
                vector = np.array([0., -np.radians(angle), 0.])
                rotate_material(sim, vector)
                back = _rotate(sim.xyz, -vector)
                deep_inside = back[:, 0] > np.cos(np.radians(25))
                channel = deep_inside & (np.abs(back[:, 1]) < np.sin(.3*np.pi/sim.h))
                self.assertGreater(np.count_nonzero(channel), 30)
                self.assertTrue(np.all(sim.crust[channel] == 0), "remap bridged the real channel")
                self.assertAlmostEqual(sim.land_mass.sum()/sim.original_mass, 1, places=12)

    def test_polar_lookup_continues_same_cap_at_opposite_longitude(self):
        sim = spherical_cap()
        points = np.array([[0., 0., 1.], [0., 0., -1.]])
        lookup = sim._sample_coordinates(points)
        np.testing.assert_allclose(sum(weight for _, weight in lookup), 1)
        for index, _ in lookup:
            self.assertLess(index[0], sim.w)
            self.assertGreaterEqual(index[1], sim.n-sim.w)
        np.testing.assert_allclose(sim._sample(sim.xyz[:, 0], lookup), 0, atol=1e-14)
        np.testing.assert_allclose(sim._sample(sim.xyz[:, 1], lookup), 0, atol=1e-14)

    def test_arc_addition_and_coalescing_keep_footprints_aligned(self):
        sim = spherical_cap()
        cells = np.flatnonzero(sim.crust == 0)[sim.w:sim.w+2]
        sim.steps = 1
        sim._add_arc_crust(cells, np.array([100., 200.]))
        arcs = np.flatnonzero(sim.kind == 3)
        sim.pos[arcs] = sim.pos[arcs[0]]
        sim._coalesce_arcs()
        self.assertEqual(len(sim.mass), len(sim.parcel_east))
        self.assertEqual(len(sim.mass), len(sim.parcel_extent))
        self.assertEqual(np.count_nonzero(sim.kind == 3), 1)
        self.assertAlmostEqual(sim.mass[sim.kind == 3].sum(), 300)
        rotate_material(sim, np.array([0., -.7, 0.]))
        self.assertTrue(np.isfinite(sim.density).all())
        self.assertAlmostEqual(sim.land_mass.sum()/sim.mass.sum(), 1, places=12)


if __name__ == '__main__':
    unittest.main()
