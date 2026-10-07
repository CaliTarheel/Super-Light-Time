"""Temporary thermal support is exported as motion, never fictitious erosion."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

import gospl_export as export
from orientation import rotation_matrix
from ridge_interaction import sample_ridge_effects
from tests.test_gospl_export import grid_points


def thermal_frame(time, *, width=128, relief=0., loss=0., supported=False, center=None):
    height = width//2
    points = grid_points(width, height)
    if center is None:
        center = points[height//2*width+width//2]
    episode = dict(id=1, center=np.asarray(center).tolist(), overriding_plate_uid=71,
                   incoming_plate_uids=[81, 91], started_myr=0., peak_myr=8., end_myr=32.,
                   last_seen_myr=0., radius_km=450., thermal_peak_m=400., volcanic_budget_m=200.)
    frame = dict(width=width, height=height, time_myr=float(time),
                 plates=[dict(id=0, uid=71, name='Thermal fixture', angular_velocity=[0., 0., 0.])],
                 plate=np.zeros(len(points), np.int32), crust=np.ones(len(points), np.uint8),
                 boundary=np.zeros(len(points), np.uint8), age=np.zeros(len(points), np.float32),
                 ridge_episodes=[episode] if time < 32 else [])
    thermal = sample_ridge_effects(frame['ridge_episodes'], points,
                                  np.full(len(points), 71), time)['thermal_support_m']
    frame['elevation'] = (220.+relief+thermal).astype(np.float32)
    count = len(points) if supported else 0
    frame.update(trace_id=np.arange(count, dtype=np.int64), trace_xyz=points[:count],
                 trace_plate_uid=np.full(count, 71, np.int64),
                 trace_erosion_m=np.full(count, loss), trace_ridge_uplift_m=np.zeros(count))
    return frame


class RidgeGosplTests(unittest.TestCase):
    def test_unsupported_grid_correction_excludes_thermal_but_keeps_internal_relief(self):
        first, second = thermal_frame(8., relief=125.), thermal_frame(40., relief=125.)
        a, b = export.matched_traces(first, second)
        loss, support, _, _ = export.erosion_correction(first, second, a, b, dict(erosion=1.))
        expected = 125.*(1-np.exp(-32./180.))
        np.testing.assert_allclose(loss, expected, atol=1e-5)
        self.assertFalse(np.any(support))
        points = grid_points(128, 64)
        fields, _ = export.interval_forcing(points, first, second, dict(erosion=1.))
        delta = second['elevation']-first['elevation']
        np.testing.assert_allclose(fields['geometric_rate']*32_000_000., delta, atol=.0001)
        np.testing.assert_allclose(fields['upsub']*32_000_000., delta+expected, atol=.0001)

    def test_owner_mismatch_fallback_subtracts_interpolated_source_thermal_raster(self):
        first = thermal_frame(8., width=512)
        width, height = first['width'], first['height']
        cell = height//2*width+width//2
        point = grid_points(width, height)[cell].copy()
        # Sample inside the original cell but away from its center, so an
        # analytic dome evaluation differs from the saved bilinear overlay.
        angle = .3*2*np.pi/width
        point = point*np.cos(angle)+np.array([-point[1], point[0], 0.])*np.sin(angle)
        point /= np.linalg.norm(point)
        coarse_cell = int(export.cells_at(point[None], 256, 128)[0])
        coarse_xyz = grid_points(256, 128)[coarse_cell]
        source_cell = int(export.cells_at(coarse_xyz[None], width, height)[0])
        self.assertNotEqual(source_cell, int(export.cells_at(point[None], width, height)[0]))
        first['plate'][source_cell], first['crust'][source_cell] = 1, 0
        first['plates'].append(dict(id=1, uid=72, name='Adjacent owner', angular_velocity=[0., 0., 0.]))
        points = grid_points(width, height)
        uids = np.where(first['plate'] == 0, 71, 72)
        overlay = sample_ridge_effects(first['ridge_episodes'], points, uids, 8.)['thermal_support_m']
        first['elevation'] = (220.+overlay).astype(np.float32)
        second = deepcopy(first)
        second.update(time_myr=40., ridge_episodes=[], elevation=np.full(len(points), 220., np.float32))
        interpolated = export.sample(overlay.reshape(height, width), point[None])[0]
        analytic = sample_ridge_effects(first['ridge_episodes'], point[None], np.array([71]), 8.)['thermal_support_m'][0]
        self.assertGreater(abs(float(analytic)-interpolated), .1)
        fields, _ = export.interval_forcing(point[None], first, second, dict(erosion=1.))
        self.assertEqual(int(fields['correction_supported'][0]), 0)
        self.assertAlmostEqual(float(fields['relaxation_correction'][0])*32_000_000., 0., delta=1e-5)
        self.assertAlmostEqual(float(fields['upsub'][0])*32_000_000., -interpolated, delta=.0001)

    def test_recorded_volcanic_relief_is_not_added_twice_and_legacy_is_unchanged(self):
        first = thermal_frame(8., relief=600., loss=0., supported=True)
        second = thermal_frame(10., relief=675., loss=25., supported=True)
        second['trace_ridge_uplift_m'][:] = 100.
        points = grid_points(128, 64)
        fields, _ = export.interval_forcing(points, first, second, dict(erosion=1.))
        expected = second['elevation']-first['elevation']+25.
        np.testing.assert_allclose(fields['upsub']*2_000_000., expected, atol=.0002)
        np.testing.assert_allclose(fields['relaxation_correction']*2_000_000., 25., atol=1e-5)
        self.assertTrue(np.all(fields['correction_supported']))
        first.pop('ridge_episodes')
        second.pop('ridge_episodes')
        legacy, _ = export.interval_forcing(points, first, second, dict(erosion=1.))
        np.testing.assert_array_equal(legacy['upsub'], fields['upsub'])
        self.assertIsNone(export.recorded_thermal_grid(first))

    def test_oriented_native_package_captures_the_sampler_and_preserves_forcing_basis(self):
        angles = dict(yaw=37., pitch=21., roll=-13.)
        matrix = rotation_matrix(angles)
        vertices, _ = export.icosphere(1)
        first = thermal_frame(8., center=vertices[0]@matrix.T)
        second = thermal_frame(40., center=vertices[0]@matrix.T)
        scratch = Path(__file__).resolve().parents[1]/'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory)
            source = root/'run'
            source.mkdir()
            payload = b'# Static ridge export fixture\n'
            (source/'engine.py').write_bytes(payload)
            manifest = dict(run_id='ridge-fixture', config=dict(erosion=1., dt_myr=2.),
                            frames=[dict(time_myr=8.), dict(time_myr=40.)],
                            engine_sha256=hashlib.sha256(payload).hexdigest(), auxiliary_sources_sha256={})
            for i, frame in enumerate((first, second)):
                arrays = {key: value for key, value in frame.items() if isinstance(value, np.ndarray)}
                metadata = {key: value for key, value in frame.items() if key not in arrays}
                np.savez_compressed(source/f'frame_{i:04d}.npz', **arrays)
                (source/f'frame_{i:04d}.json').write_text(json.dumps(metadata))
            destination = root/'package'
            metadata = export.build_history(source, destination, manifest, subdivisions=1,
                                            dt_years=100000, orientation=angles)
            expected, _ = export.interval_forcing(vertices@matrix.T, first, second, dict(erosion=1., dt_myr=2.))
            with np.load(destination/'input/forcing_0000.npz', allow_pickle=False) as actual:
                np.testing.assert_array_equal(actual['upsub'], expected['upsub'])
                np.testing.assert_allclose(actual['hdisp'], expected['hdisp']@matrix, atol=1e-20)
                self.assertGreater(abs(float(actual['upsub'][0]))*32_000_000., 100.)
            captured = destination/'source/ridge_interaction.py'
            self.assertEqual(captured.read_bytes(), export.RIDGE_INTERACTION_SOURCE)
            self.assertEqual(metadata['ridge_interaction_sha256'], hashlib.sha256(captured.read_bytes()).hexdigest())
            with zipfile.ZipFile(destination/'gospl-history.zip') as archive:
                self.assertEqual(archive.read('source/ridge_interaction.py'), captured.read_bytes())


if __name__ == '__main__':
    unittest.main()
