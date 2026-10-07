"""Export, conditioning and seam contracts for procedural terrain refinement."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from terrain import TerrainCancelled, _coordinates, _noise3, _prepare, _sample, _sampling, build_terrain, color_tile


class TerrainTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / 'tmp'
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)

    def frame(self, kind=1, elevation=1500):
        w, h = 32, 16
        return dict(width=w, height=h, time_myr=42, index=21, run_id='fixture',
                    source_engine_sha256='fixture-engine',
                    elevation=np.full(w*h, elevation, dtype=np.float32),
                    crust=np.full(w*h, kind, dtype=np.uint8),
                    plate=np.ones(w*h, dtype=np.int32),
                    age=np.full(w*h, 90, dtype=np.float32),
                    boundary=np.zeros(w*h, dtype=np.uint8))

    def test_export_roundtrip_and_provenance(self):
        frame = self.frame()
        progress = []
        result = build_terrain(frame, self.path / 'export', width=128, progress=progress.append)
        z = np.load(self.path / 'export' / 'elevation_m.npy', allow_pickle=False)
        with Image.open(self.path / 'export' / 'heightmap_16bit.png') as picture:
            png = np.asarray(picture).astype(np.float64) - 12000
        self.assertEqual(z.shape, (64, 128))
        self.assertEqual(z.dtype, np.float32)
        self.assertLessEqual(float(np.abs(png - z).max()), .5)
        self.assertTrue(np.isfinite(z).all())
        with Image.open(self.path / 'export' / 'preview.png') as picture:
            self.assertEqual(picture.size, (128, 64))
        self.assertEqual(result['source']['run_id'], 'fixture')
        self.assertEqual(result['source']['source_engine_sha256'], 'fixture-engine')
        self.assertEqual(result['source_width'], 32)
        self.assertEqual(result['resolution']['tectonic_grid'], [32, 16])
        self.assertFalse(result['history']['fine_features_material_tracked'])
        self.assertFalse(result['downstream']['gospl_ready_forcing'])
        self.assertEqual(progress[-1], 1.)
        self.assertTrue(all(a <= b for a, b in zip(progress, progress[1:])))
        self.assertEqual(json.loads((self.path / 'export' / 'terrain_metadata.json').read_text())['state'], 'complete')

    def test_partition_invariance_and_epoch_stability(self):
        frame = self.frame()
        for name, band in (('one', 1), ('seven', 7)):
            build_terrain(frame, self.path / name, width=128, seed=789, band_rows=band)
        np.testing.assert_array_equal(np.load(self.path / 'one' / 'elevation_m.npy'),
                                      np.load(self.path / 'seven' / 'elevation_m.npy'))
        frame['time_myr'] = 500
        frame['index'] = 250
        build_terrain(frame, self.path / 'later', width=128, seed=789, band_rows=13)
        np.testing.assert_array_equal(np.load(self.path / 'one' / 'elevation_m.npy'),
                                      np.load(self.path / 'later' / 'elevation_m.npy'))
        build_terrain(frame, self.path / 'other_seed', width=128, seed=790)
        self.assertFalse(np.array_equal(np.load(self.path / 'one' / 'elevation_m.npy'),
                                        np.load(self.path / 'other_seed' / 'elevation_m.npy')))

    def test_cratons_are_quieter_than_mountainous_continental_crust(self):
        for name, kind in (('craton', 2), ('continent', 1)):
            build_terrain(self.frame(kind=kind, elevation=2500), self.path / name, width=256)
        craton = np.load(self.path / 'craton' / 'elevation_m.npy')
        continent = np.load(self.path / 'continent' / 'elevation_m.npy')
        self.assertLess(float(craton.std()), float(continent.std()) * .35)
        self.assertGreater(float(continent.std()), 30)
        self.assertLess(abs(float(continent.mean()) - 2500), 10)

    def test_boundary_and_material_history_condition_relief(self):
        frame = self.frame(elevation=700)
        base_fields, count = _prepare(frame)
        self.assertEqual(count, 0)
        frame['boundary'][7*32+15:7*32+18] = 4
        lon, lat = np.deg2rad([-5.625, 5.625])
        frame.update(trace_xyz=np.array([[np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]]),
                     trace_suture=np.array([.9]), trace_uplift_m=np.array([12000.]))
        conditioned, count = _prepare(frame)
        self.assertEqual(count, 1)
        self.assertGreater(float(conditioned['suture'].max()), .1)
        self.assertGreater(float(conditioned['deformation'].max()), .1)
        self.assertGreater(float(conditioned['collision'].max()), .1)
        self.assertEqual(float(base_fields['collision'].max()), 0)

    def test_disabled_detail_is_spherical_bilinear_baseline(self):
        frame = self.frame()
        lon = (np.arange(32) + .5) * 2*np.pi/32 - np.pi
        lat = np.pi/2 - (np.arange(16) + .5) * np.pi/16
        source = 1700*np.cos(lat[:, None])*np.cos(lon[None, :]) - 1200
        frame['elevation'] = source.astype(np.float32).ravel()
        build_terrain(frame, self.path / 'zero', width=128, detail=0)
        lon, lat, _, _, _ = _coordinates(128, 64, 0, 64)
        expected = _sample(source.astype(np.float32), _sampling(lon, lat, 32, 16))
        np.testing.assert_array_equal(np.load(self.path / 'zero' / 'elevation_m.npy'), expected)

    def test_spherical_noise_and_sampling_have_no_dateline_discontinuity(self):
        # +/-180 degrees are the same point; near-pole longitude variation also
        # tends to zero, unlike noise generated in raw equirectangular pixels.
        lons = np.deg2rad(np.array([-180., 180., -90., 90.]))
        lat = np.full(4, np.pi/2 - 1e-8)
        n = _noise3(np.cos(lat)*np.cos(lons), np.cos(lat)*np.sin(lons), np.sin(lat), 2048, 12)
        self.assertLess(float(np.ptp(n)), .0001)
        fields, _ = _prepare(self.frame())
        sample = _sample(fields['elevation'], _sampling(lons, np.zeros(4), 32, 16))
        self.assertEqual(sample[0], sample[1])
        # A non-uniform source polar ring must converge to one pole value.
        fields['elevation'][0] = np.arange(32, dtype=np.float32) * 100
        polar_sample = _sample(fields['elevation'], _sampling(lons, np.full(4, np.pi/2), 32, 16))
        np.testing.assert_allclose(polar_sample, np.full(4, fields['elevation'][0].mean()), atol=.002)

    def test_abyssal_texture_cannot_create_islands(self):
        build_terrain(self.frame(kind=0, elevation=-4000), self.path / 'ocean', width=128, detail=2)
        z = np.load(self.path / 'ocean' / 'elevation_m.npy')
        self.assertLess(float(z.max()), -3000)
        self.assertGreater(float(z.std()), 1)

    def test_cancellation_has_no_completion_marker_and_existing_output_is_protected(self):
        calls = 0
        def stop():
            nonlocal calls
            calls += 1
            return calls >= 3
        with self.assertRaises(TerrainCancelled):
            build_terrain(self.frame(), self.path / 'cancel', width=128, cancel=stop, band_rows=2)
        self.assertFalse((self.path / 'cancel' / 'terrain_metadata.json').exists())
        with self.assertRaises(ValueError):
            build_terrain(self.frame(), self.path / 'cancel', width=128)

    def test_tile_pixels_and_edge_dimensions(self):
        z = np.arange(64*128, dtype=np.float32).reshape(64, 128) - 4000
        np.save(self.path / 'z.npy', z)
        tile = color_tile(self.path / 'z.npy', 111, 54, size=32)
        self.assertEqual(tile.size, (17, 10))
        np.testing.assert_array_equal(np.asarray(tile), np.asarray(color_tile(z, 111, 54, size=32)))
        with self.assertRaises(ValueError):
            color_tile(z, 128, 0)

    def test_bad_inputs_fail_before_output(self):
        for args in ({'width': 1600}, {'detail': float('nan')}, {'detail': 3}, {'band_rows': 0}, {'seed': 1.5}):
            with self.assertRaises(ValueError):
                build_terrain(self.frame(), self.path / 'invalid', **args)
        frame = self.frame()
        frame['elevation'][0] = float('nan')
        with self.assertRaises(ValueError):
            build_terrain(frame, self.path / 'invalid', width=128)
        self.assertFalse((self.path / 'invalid').exists())


if __name__ == '__main__':
    unittest.main()
