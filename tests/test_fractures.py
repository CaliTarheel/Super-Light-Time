"""Initial geometry contracts: curved cracks, connected crust and intact cratons."""
import unittest

import numpy as np

from fracture import barriers_from_grid, component_labels, make_fracture
from raster_engine import Simulation, _xyz


class FractureTests(unittest.TestCase):
    def test_generated_continent_has_one_ocean_and_intact_connected_cratons(self):
        sim = Simulation(dict(width=128, height=64, duration_myr=2))
        self.assertEqual(len(np.unique(sim.plate[sim.initial_crust == 0])), 1)
        ocean_slot = int(sim.plate[sim.initial_crust == 0][0])
        self.assertEqual(sim.initial_ocean_plate_uid, sim.plate_uid[ocean_slot])
        labels, count = component_labels(sim.initial_crust == 2, sim.w, sim.h)
        self.assertGreaterEqual(count, 5)
        for label in range(count):
            self.assertEqual(len(np.unique(sim.plate[labels == label])), 1)
        self.assertGreaterEqual(len(sim.initial_fractures), 5)
        for p in np.unique(sim.plate[sim.initial_crust > 0]):
            _, count = component_labels((sim.plate == p) & (sim.initial_crust > 0), sim.w, sim.h)
            self.assertEqual(count, 1, f'new disconnected continental shard on plate {p}')
        repeated = Simulation(dict(width=128, height=64, duration_myr=2))
        np.testing.assert_array_equal(sim.plate, repeated.plate)
        changed = Simulation(dict(width=128, height=64, seed=93, duration_myr=2))
        self.assertFalse(np.array_equal(sim.plate, changed.plate))

    def test_crack_bypasses_a_craton_that_a_straight_cut_would_bisect(self):
        lon, lat = np.meshgrid(np.linspace(-.3, .3, 51), np.linspace(-.3, .3, 51))
        xyz = _xyz(lon.ravel(), lat.ravel())
        craton = xyz[:, 0] > np.cos(.17)
        normal = np.array([0., 0., 1.])
        self.assertTrue(np.any(xyz[craton] @ normal < 0))
        self.assertTrue(np.any(xyz[craton] @ normal > 0))
        cut = make_fracture([1., 0., 0.], normal, 12, barriers=[(np.array([1., 0., 0.]), .18)])
        signs = cut.signed_distance(xyz[craton]) > 0
        self.assertTrue(np.all(signs) or not np.any(signs))
        # The bypass follows a rounded route, rather than laying a straight
        # shelf across the cap. Curvature remains visible along its shoulders.
        along = np.linspace(-.4,.4,401)
        offset = cut.offset_at(along)
        slope = np.gradient(offset,along)
        self.assertGreater(float(np.std(slope)), .15)
        self.assertGreater(float(np.ptp(offset)), .05)
        self.assertLess(np.count_nonzero(np.abs(np.diff(offset)) < 1e-8), 3)
        seam = _xyz(np.array([-np.pi+1e-8, np.pi-1e-8]), np.array([.3, .3]))
        self.assertAlmostEqual(*cut.signed_distance(seam), places=6)

    def test_spherical_components_wrap_seam_without_joining_opposite_poles(self):
        mask = np.zeros((24, 48), bool)
        mask[3:7, :3] = mask[3:7, -3:] = True
        mask[0, 8] = mask[0, 32] = True
        mask[-1, 8] = mask[-1, 32] = True
        labels, count = component_labels(mask, 48, 24)
        labels = labels.reshape(24, 48)
        self.assertEqual(count, 3)
        self.assertEqual(labels[4, 0], labels[4, 47])
        self.assertEqual(labels[0, 8], labels[0, 32])
        self.assertNotEqual(labels[0, 8], labels[-1, 8])

    def test_uniform_ocean_or_craton_is_not_forcibly_fragmented(self):
        for crust in (0, 2):
            with self.subTest(crust=crust):
                sim = Simulation(dict(width=48, height=24, duration_myr=2),
                                 dict(width=48, height=24, crust=np.full(48*24, crust)))
                self.assertEqual(sim.count, 1)
                self.assertEqual(np.count_nonzero(sim.active), 1)
                self.assertEqual(sim.initial_ocean_plate_uid, 1 if crust == 0 else None)
                sim.step(2)
                self.assertTrue(np.isfinite(sim.snapshot()['elevation']).all())

    def test_curves_turn_smoothly_at_knots_and_at_the_spherical_seam(self):
        cut = make_fracture([1,0,0],[0,0,1],40)
        epsilon = 1e-6
        left = (cut.offset_at(cut.knots)-cut.offset_at(cut.knots-epsilon))/epsilon
        right = (cut.offset_at(cut.knots+epsilon)-cut.offset_at(cut.knots))/epsilon
        self.assertLess(float(np.max(np.abs(left-right))), .001)
        self.assertAlmostEqual(float(cut.offset_at(-np.pi)),float(cut.offset_at(np.pi)),places=14)
        middle = (cut.knots+np.roll(cut.knots,-1))/2
        middle = middle[:-1]
        line = (cut.offsets[:-1]+cut.offsets[1:])/2
        self.assertGreater(np.count_nonzero(np.abs(cut.offset_at(middle)-line) > 1e-5),100)

    def test_curve_contains_broad_bends_and_small_meanders_without_grid_noise(self):
        cut = make_fracture([1,0,0],[0,0,1],19)
        along = np.linspace(-np.pi,np.pi,4096,endpoint=False)
        values = cut.offset_at(along)
        spectrum = np.abs(np.fft.rfft(values))/len(values)
        self.assertGreater(float(np.linalg.norm(spectrum[1:6])), .035)
        self.assertGreater(float(np.linalg.norm(spectrum[10:40])), .004)
        self.assertLess(float(np.linalg.norm(spectrum[100:])), .0005)
        repeated = make_fracture([1,0,0],[0,0,1],19)
        np.testing.assert_array_equal(values,repeated.offset_at(along))
        changed = make_fracture([1,0,0],[0,0,1],20)
        self.assertFalse(np.array_equal(values,changed.offset_at(along)))

    def test_rounded_bypass_preserves_entire_caps_at_varied_spherical_orientations(self):
        # Dense cap samples exercise interpolation between the compact knots,
        # including a cap whose bypass crosses the periodic chart seam.
        for longitude, latitude, radius in ((0.,0.,.18),(.5,.11,.24),
                                             (-.6,-.16,.29),(np.pi-.04,.04,.21)):
            point = _xyz(np.array([longitude]),np.array([latitude]))[0]
            east = np.array([-np.sin(longitude),np.cos(longitude),0.])
            north = np.cross(point,east)
            angle, radial = np.meshgrid(np.linspace(-np.pi,np.pi,240,endpoint=False),
                                        np.linspace(0,radius,20))
            samples = (np.cos(radial.ravel())[:,None]*point +
                       np.sin(radial.ravel())[:,None]*(np.cos(angle.ravel())[:,None]*east+
                                                       np.sin(angle.ravel())[:,None]*north))
            for seed in (1,12,47):
                with self.subTest(longitude=longitude,latitude=latitude,seed=seed):
                    cut = make_fracture([1,0,0],[0,0,1],seed,barriers=[(point,radius)])
                    signs = cut.signed_distance(samples) > 0
                    self.assertTrue(np.all(signs) or not np.any(signs))

    def test_large_grid_classifier_matches_bounded_material_batches(self):
        rng = np.random.default_rng(4)
        points = rng.normal(size=(70_000,3))
        points /= np.linalg.norm(points,axis=1)[:,None]
        cut = make_fracture([1,0,0],[0,0,1],19)
        whole = cut.signed_distance(points)
        batches = np.concatenate([cut.signed_distance(points[i:i+10_000])
                                  for i in range(0,len(points),10_000)])
        np.testing.assert_allclose(whole,batches,atol=2e-15,rtol=0)
        np.testing.assert_array_equal(whole > 0,batches > 0)


if __name__ == '__main__':
    unittest.main()
