from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from orientation import orient_frame, reproject_raster
from terrain import TerrainCancelled, _coordinates, build_terrain


class TerrainOrientationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)

    @staticmethod
    def frame():
        width, height = 32, 16
        lon, lat = np.meshgrid((np.arange(width)+.5)*2*np.pi/width-np.pi,
                              np.pi/2-(np.arange(height)+.5)*np.pi/height)
        z = (4000*np.cos(lat)*np.cos(lon)-1000+800*np.sin(3*lon)*np.cos(lat)**2).astype(np.float32)
        return dict(width=width, height=height, time_myr=280., elevation=z.reshape(-1),
                    crust=np.where(z > 1800, 2, np.where(z > -200, 1, 0)).astype(np.uint8).reshape(-1),
                    boundary=np.where(np.abs(z-1800) < 250, 4, 0).astype(np.uint8).reshape(-1),
                    age=np.full(width*height, 60., np.float32))

    def test_identity_and_column_yaw_preserve_the_finished_terrain_exactly(self):
        frame = self.frame()
        frozen = deepcopy(frame)
        for name, orientation in (('canonical', None), ('identity', dict(yaw=360, pitch=-720)),
                                  ('yaw', dict(yaw=90))):
            build_terrain(frame, self.path/name, width=128, seed=83, orientation=orientation, band_rows=7)
        canonical = np.load(self.path/'canonical'/'elevation_m.npy')
        np.testing.assert_array_equal(canonical, np.load(self.path/'identity'/'elevation_m.npy'))
        np.testing.assert_array_equal(np.roll(canonical, 32, axis=1), np.load(self.path/'yaw'/'elevation_m.npy'))
        for filename in ('elevation_m.npy', 'heightmap_16bit.png', 'preview.png'):
            self.assertEqual((self.path/'canonical'/filename).read_bytes(), (self.path/'identity'/filename).read_bytes())
        for key in ('elevation', 'crust', 'boundary', 'age'):
            np.testing.assert_array_equal(frame[key], frozen[key])

    def test_arbitrary_rotation_moves_existing_noise_and_output_statistics(self):
        frame = self.frame()
        orientation = dict(yaw=132, pitch=-78, roll=23)
        build_terrain(frame, self.path/'canonical', width=128, seed=91)
        progress = []
        metadata = build_terrain(frame, self.path/'rotated', width=128, seed=91,
                                 orientation=orientation, progress=progress.append)
        original = np.load(self.path/'canonical'/'elevation_m.npy')
        actual = np.load(self.path/'rotated'/'elevation_m.npy')
        expected = np.empty_like(original)
        reproject_raster(original, expected, orientation)
        np.testing.assert_array_equal(actual, expected)
        # Regenerating world-coordinate noise after moving the coarse source
        # would visibly replace the user's fine terrain. Guard against that.
        build_terrain(orient_frame(frame, orientation), self.path/'wrong_order', width=128, seed=91)
        wrong = np.load(self.path/'wrong_order'/'elevation_m.npy')
        self.assertGreater(float(np.abs(actual-wrong).mean()), 10.)
        self.assertEqual(progress, sorted(progress))
        self.assertEqual(progress[-1], 1.)
        self.assertTrue(any(.65 < value <= .8 for value in progress))
        _, lat, _, _, _ = _coordinates(128, 64, 0, 64)
        weights = np.cos(lat).astype(np.float64)
        denominator = weights.sum()*128
        self.assertEqual(metadata['stats']['minimum_m'], float(actual.min()))
        self.assertEqual(metadata['stats']['maximum_m'], float(actual.max()))
        self.assertAlmostEqual(metadata['stats']['area_weighted_mean_m'], float((actual*weights).sum()/denominator), places=10)
        self.assertAlmostEqual(metadata['stats']['land_area_fraction'], float(((actual >= 0)*weights).sum()/denominator), places=13)
        self.assertEqual(metadata['orientation'], orientation)
        self.assertEqual(metadata['time_myr'], 280.)
        with Image.open(self.path/'rotated'/'heightmap_16bit.png') as image:
            self.assertLessEqual(float(np.abs(np.asarray(image).astype(np.float64)-12000-actual).max()), .5)
        self.assertFalse((self.path/'rotated'/'.orientation_work.npy').exists())

    def test_rotation_cancellation_closes_and_removes_only_its_temporary_file(self):
        output = self.path/'cancelled'
        output.mkdir()
        (output/'caller-notes.txt').write_text('Keep this file.')
        progress = []
        with self.assertRaises(TerrainCancelled):
            build_terrain(self.frame(), output, width=512, orientation=dict(pitch=75),
                          progress=progress.append, cancel=lambda: bool(progress and progress[-1] > .65))
        self.assertGreater(progress[-1], .65)
        self.assertFalse((output/'terrain_metadata.json').exists())
        self.assertFalse((output/'.orientation_work.npy').exists())
        self.assertEqual((output/'caller-notes.txt').read_text(), 'Keep this file.')
        # On Windows these operations fail if the cancelled pass leaked mappings.
        (output/'elevation_m.npy').unlink()
        with self.assertRaises(ValueError):
            build_terrain(self.frame(), self.path/'invalid', width=64, orientation=dict(pitch=float('nan')))
        self.assertFalse((self.path/'invalid').exists())


if __name__ == '__main__':
    unittest.main()
