"""Coordinate covariance across reviewed history, saved grids and goSPL forcing."""
import hashlib
import io
import json
import unittest
import zipfile

import numpy as np

from tests import test_history as history_fixture
from tests import test_gospl_export as gospl_fixture
from orientation import inverse_cells, normalize_orientation, orient_metadata, rotation_matrix
from gospl_export import RADIUS_M, load_frame, sample


class HistoryOrientationTests(unittest.TestCase):
    setUp = history_fixture.HistoryTests.setUp
    coordinates = history_fixture.HistoryTests.coordinates
    xyz = history_fixture.HistoryTests.xyz
    snapshot = history_fixture.HistoryTests.snapshot
    write_run = history_fixture.HistoryTests.write_run

    def test_rotated_inspection_follows_same_material_and_matching_grid(self):
        orientation = dict(yaw=90)
        destination = self.marker_cells[3]+2
        original = self.manager.history(3, self.marker_cells[3], 'tracked')
        result = self.manager.history(3, destination, 'tracked', orientation)
        self.assertEqual(result['trace_id'], original['trace_id'])
        self.assertEqual(result['selection']['cell'], destination)
        self.assertEqual(result['selection']['source_cell'], self.marker_cells[3])
        np.testing.assert_allclose([[p['lon'], p['lat']] for p in result['points']],
                                   [[p['lon'], p['lat']] for p in orient_metadata(original, orientation)['points']])
        self.assertEqual(self.manager.frame(3, orientation)['crust'][destination], 2)

    def test_export_rotates_all_epochs_and_traces_without_rewriting_sources(self):
        path = self.manager.path()
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()}
        angles = dict(yaw=37, pitch=73, roll=-22)
        matrix = rotation_matrix(angles)
        with zipfile.ZipFile(io.BytesIO(self.manager.export(3, angles))) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(json.loads(archive.read('orientation.json'))['orientation'], normalize_orientation(angles))
            exported = np.load(io.BytesIO(archive.read('elevation_m.npy')))
            np.testing.assert_array_equal(exported.ravel(), self.manager.frame(3, angles)['elevation'])
            for i in range(4):
                original = self.snapshot(i, True)
                with np.load(io.BytesIO(archive.read(f'history/frame_{i:04d}.npz'))) as data:
                    np.testing.assert_allclose(data['trace_xyz'], original['trace_xyz']@matrix, atol=1e-14)
                    mapping = inverse_cells(np.arange(32), 8, 4, angles)
                    np.testing.assert_array_equal(data['crust'], original['crust'][mapping])
                    np.testing.assert_array_equal(data['trace_id'], original['trace_id'])
                header = json.loads(archive.read(f'history/frame_{i:04d}.json'))
                self.assertEqual(header['orientation'], normalize_orientation(angles))
                expected = orient_metadata(original['events'], angles)
                np.testing.assert_allclose([[e['lon'], e['lat']] for e in header['events']],
                                           [[e['lon'], e['lat']] for e in expected])
        self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()})


class GosplOrientationTests(unittest.TestCase):
    setUp = gospl_fixture.NativePackageTests.setUp
    write_frame = gospl_fixture.NativePackageTests.write_frame
    build = gospl_fixture.NativePackageTests.build

    def test_rotation_preserves_known_rigid_motion_and_vertical_forcing(self):
        # The fixture rotates its continent +90 degrees about the original pole
        # over each interval and accumulates exactly 100 m of corrected uplift.
        angles = dict(yaw=29, pitch=67, roll=-19)
        matrix = rotation_matrix(angles)
        output, metadata = self.build(orientation=angles)
        with np.load(output/'input/mesh.npz') as mesh:
            vertices = mesh['v'].copy()
            initial = mesh['z'].copy()
        first = load_frame(self.run, 0)
        np.testing.assert_allclose(initial, sample(first['elevation'].reshape(16, 32), vertices/RADIUS_M@matrix.T))
        with np.load(output/'input/forcing_0000.npz') as forcing:
            destination = vertices + forcing['hdisp'].astype(float)*2_000_000.
            expected = vertices@matrix.T@gospl_fixture.QUARTER_TURN@matrix
            np.testing.assert_allclose(destination, expected, atol=1., rtol=0)
            np.testing.assert_allclose(forcing['upsub']*2_000_000., 100., atol=2e-5)
            np.testing.assert_allclose(np.linalg.norm(destination, axis=1), RADIUS_M, atol=1., rtol=0)
        self.assertEqual(metadata['orientation'], normalize_orientation(angles))
        self.assertEqual(hashlib.sha256((output/'source/orientation.py').read_bytes()).hexdigest(), metadata['orientation_sha256'])

    def test_tilted_mesh_receives_nonuniform_starting_geography(self):
        # An analytic smooth globe supplies an independent geographic oracle.
        source_points = gospl_fixture.grid_points()
        source_heights = 1000 + 500*source_points[:, 0] + 100*source_points[:, 2]
        self.write_frame(0, gospl_fixture.synthetic_frame(10., field=source_heights))
        angles = dict(yaw=-51, pitch=83, roll=37)
        output, _ = self.build(orientation=angles)
        with np.load(output/'input/mesh.npz') as mesh:
            original_positions = mesh['v']/RADIUS_M@rotation_matrix(angles).T
            expected = 1000 + 500*original_positions[:, 0] + 100*original_positions[:, 2]
            np.testing.assert_allclose(mesh['z'], expected, atol=8., rtol=0)
            self.assertGreater(float(mesh['z'].std()), 200.)


if __name__ == '__main__':
    unittest.main()
