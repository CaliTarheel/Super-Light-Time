"""Export checks independent of the simulation's numerical implementation."""
import io
import unittest
import numpy as np
from PIL import Image
from server import png_bytes, boundary_geojson, validate_initial


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.frame = dict(width=4, height=2, time_myr=1000,
                          elevation=np.array([-11000, -4500, -1, 0, 1, 1800, 4999, 9000], dtype=np.float32),
                          boundary=np.array([0, 1, 2, 3, 4, 5, 0, 0]),
                          plate=np.array([0, 0, 1, 1, 2, 2, 2, 2]))

    def test_heightmap_preserves_negative_elevations_and_fixed_datum(self):
        image = np.array(Image.open(io.BytesIO(png_bytes(self.frame)))).astype(np.int32)
        np.testing.assert_array_equal(image.reshape(-1) - 12000, self.frame['elevation'])
        self.assertEqual(image.shape, (2, 4))

    def test_boundary_georeferencing_and_north_to_south_rows(self):
        features = boundary_geojson(self.frame)['features']
        self.assertEqual(len(features), 5)
        self.assertEqual(features[0]['geometry']['coordinates'], [-45, 45])
        self.assertEqual(features[-1]['geometry']['coordinates'], [-45, -45])
        self.assertEqual(features[-1]['properties']['type'], 'continental rift')

    def test_initial_map_rejects_silent_resolution_change(self):
        with self.assertRaises(ValueError):
            validate_initial(dict(width=4, height=2, crust=[0]*8), dict(width=8, height=4))

    def test_initial_map_rejects_unknown_crust_and_nan(self):
        for invalid in (3, -1, float('nan'), .5):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                validate_initial(dict(width=4, height=2, crust=[invalid]+[0]*7), dict(width=4, height=2))


if __name__ == '__main__':
    unittest.main()
