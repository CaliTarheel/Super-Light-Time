"""Native geometry, motion and column budgets survive the goSPL adapter."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

import gospl_export as export
import mesh_geometry
import native_frame_sampling as sampling


def triangle(center, radius=.002):
    center = np.asarray(center, float)
    center /= np.linalg.norm(center)
    axis = np.array([1., 0, 0]) if abs(center[0]) < .9 else np.array([0., 0, 1.])
    u = np.cross(center, axis); u /= np.linalg.norm(u)
    v = np.cross(center, u)
    angles = np.arange(3)*2*np.pi/3
    points = center+radius*(np.cos(angles)[:, None]*u+np.sin(angles)[:, None]*v)
    return points/np.linalg.norm(points, axis=1)[:, None]


def frame(time=0., centers=((0., 0., 1.),), heights=None, rates=None):
    vertices, faces = export.icosphere(1)
    n = len(faces); m = len(centers)
    material = np.concatenate([triangle(center) for center in centers]) if m else np.empty((0, 3))
    heights = np.full(m, 900.) if heights is None else np.asarray(heights, float)
    rates = np.full(m, 7.) if rates is None else np.asarray(rates, float)
    return dict(mesh_version=1, structure_version=1, width=16, height=8, time_myr=float(time),
                plates=[dict(id=0, uid=10, angular_velocity=[0., 0., 0.]),
                        dict(id=1, uid=20, angular_velocity=[0., 0., 0.])],
                elevation=np.full(128, -123.), plate=np.zeros(128, np.int32),
                crust=np.zeros(128, np.uint8), boundary=np.zeros(128, np.uint8),
                mesh_vertices=vertices, mesh_faces=faces,
                mesh_area_km2=mesh_geometry.spherical_area(vertices, faces),
                mesh_plate=np.zeros(n, np.int32), mesh_crust=np.zeros(n, np.uint8),
                mesh_age_myr=np.full(n, 60.), mesh_ocean_relief_m=np.full(n, 150.),
                mesh_elevation_m=np.full(n, 8888.), mesh_boundary=np.full(n, 3, np.uint8),
                material_vertices=material, material_faces=np.arange(m*3).reshape(m, 3),
                material_face_id=np.arange(m, dtype=np.int64)+100,
                material_owner=np.ones(m, np.int32), material_kind=np.full(m, 3, np.uint8),
                material_height_m=heights, material_erosion_rate_m_myr=rates,
                trace_id=np.arange(m*3, dtype=np.int64), trace_xyz=material.copy(),
                trace_patch=np.repeat(np.arange(m)+100, 3),
                trace_plate_uid=np.full(m*3, 20, np.int64),
                trace_erosion_m=np.zeros(m*3))


def write_frame(path, index, data):
    arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    metadata = {k: v for k, v in data.items() if k not in arrays}
    np.savez_compressed(path/f'frame_{index:04d}.npz', **arrays)
    (path/f'frame_{index:04d}.json').write_text(json.dumps(metadata), encoding='utf-8')


class NativeSamplingTests(unittest.TestCase):
    def test_tiny_arc_at_pole_and_seam_uses_actual_containment_and_material_owner(self):
        centers = np.array([[0., 0., 1.], [-1., 0., 0.]])
        source = frame(centers=centers, heights=[900., 2100.])
        points = np.concatenate((centers, [[0., 1., 0.], [0., 0., -1.]]))
        with patch.object(export, 'cells_at', side_effect=AssertionError('No raster lookup')):
            result = sampling.sample_frame(source, points)
        np.testing.assert_array_equal(result['crust'], [3, 3, 0, 0])
        np.testing.assert_array_equal(result['plate'], [1, 1, 0, 0])
        np.testing.assert_array_equal(result['material_face'], [0, 1, -1, -1])
        np.testing.assert_allclose(result['elevation'][:2], [900., 2100.])
        ocean = -5651.+2473.*np.exp(-.0278*60.)+150.
        np.testing.assert_allclose(result['elevation'][2:], ocean)
        np.testing.assert_array_equal(result['boundary'], 3)
        # Both actual islands are absent from every display crust pixel.
        self.assertEqual(int(source['crust'].sum()), 0)

    def test_exposure_prefers_control_owner_then_persistent_face_id(self):
        source = frame(centers=[[0, 0, 1]]*3, heights=[800., 1300., 2100.])
        source['material_owner'][:] = [1, 0, 1]
        source['material_face_id'][:] = [6, 99, 2]
        result = sampling.sample_frame(source, [[0., 0., 1.]])
        self.assertEqual(result['elevation'][0], 1300.)
        source['mesh_plate'][:] = 1
        result = sampling.sample_frame(source, [[0., 0., 1.]])
        self.assertEqual(result['elevation'][0], 2100.)

    def test_ocean_reconstruction_does_not_mix_other_owner_or_land_columns(self):
        source = frame(centers=[])
        centers = source['mesh_vertices'][source['mesh_faces']].sum(axis=1)
        centers /= np.linalg.norm(centers, axis=1)[:, None]
        chosen = 9
        source['mesh_plate'][:] = 1
        source['mesh_plate'][chosen] = 0
        source['mesh_age_myr'][:] = 0.
        source['mesh_age_myr'][chosen] = 60.
        source['mesh_ocean_relief_m'][:] = 5000.
        source['mesh_ocean_relief_m'][chosen] = 150.
        result = sampling.sample_frame(source, centers[[chosen]])
        self.assertAlmostEqual(result['elevation'][0], -5651.+2473.*np.exp(-.0278*60.)+150.)

    def test_ocean_native_reconstruction_is_continuous_inside_one_owner(self):
        source = frame(centers=[])
        centers = source['mesh_vertices'][source['mesh_faces']].sum(axis=1)
        centers /= np.linalg.norm(centers, axis=1)[:, None]
        source['mesh_ocean_relief_m'] = centers[:, 0]*500.
        context = sampling.prepare(source)
        # Cross a native edge by tiny increments; reconstructed values meet.
        edge = mesh_geometry.geometry(source['mesh_vertices'], source['mesh_faces'])
        middle = edge['edge_mid'][4]; normal = edge['edge_normal'][4]
        points = middle+np.array([-1e-8, 1e-8])[:, None]*normal
        points /= np.linalg.norm(points, axis=1)[:, None]
        result = sampling.sample_frame(source, points, context)
        self.assertLess(abs(np.diff(result['elevation'])[0]), .0001)

    def test_geometry_sampling_is_globe_rotation_equivariant(self):
        source = frame(centers=[[0, 0, 1], [-1, 0, 0]], heights=[820., 1900.])
        points = np.array([[0., 0., 1.], [-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
        initial = sampling.sample_frame(source, points)
        matrix = export.rotation([.31, -.83, .52], 1.7)
        rotated = deepcopy(source)
        for key in ('mesh_vertices', 'material_vertices'):
            rotated[key] = rotated[key]@matrix
        result = sampling.sample_frame(rotated, points@matrix)
        for name in ('elevation', 'plate', 'crust', 'material_face', 'erosion_rate_m_myr'):
            np.testing.assert_allclose(result[name], initial[name], atol=1e-9, err_msg=name)

    def test_thermal_support_added_exactly_once_for_selected_material_owner(self):
        source = frame(time=2.)
        episode = dict(started_myr=0., peak_myr=2., end_myr=12., center=[0., 0., 1.],
                       radius_km=1000., thermal_peak_m=500., volcanic_budget_m=200., overriding_plate_uid=20)
        source['ridge_episodes'] = [episode]
        result = sampling.sample_frame(source, np.array([[0., 0., 1.], [0., 1., 0.]]))
        self.assertEqual(result['elevation'][0], 1400.)
        self.assertEqual(result['thermal_support_m'][0], 500.)
        self.assertEqual(result['erosion_rate_m_myr'][0], 7.)
        self.assertEqual(result['thermal_support_m'][1], 0.)
        # Native centroid heights already contain support and are deliberately
        # poisoned in the fixture: sampling them would add it a second time.
        self.assertTrue(np.all(source['mesh_elevation_m'] == 8888.))

    def test_marker_erosion_is_per_material_face_with_rate_fallback_and_uid_guard(self):
        first = frame(centers=[[0, 0, 1], [-1, 0, 0]], rates=[7., 11.])
        second = deepcopy(first); second['time_myr'] = 2.
        second['trace_erosion_m'][:] = [30., 60., 90., 1000., 1000., 1000.]
        first['trace_plate_uid'][3:] = 10  # A wrong owner cannot donate its loss.
        a, b = export.matched_traces(first, second)
        loss, support = sampling.material_erosion_correction(first, second, a, b)
        np.testing.assert_array_equal(loss, [60., 22.])
        np.testing.assert_array_equal(support, [True, False])
        first['trace_erosion_m'][:3] = 100.
        loss, _ = sampling.material_erosion_correction(first, second, a, b)
        self.assertEqual(loss[0], -40.)  # Preserve the recorded signed counter.

    def test_advected_material_and_erosion_close_budget_without_any_raster_lookup(self):
        first = frame(centers=[[0, 0, 1], [-1, 0, 0]])
        second = deepcopy(first); second['time_myr'] = 2.
        matrix = export.rotation([0., .45, .1], 2.)
        second['material_vertices'] = second['material_vertices']@matrix
        second['trace_xyz'] = second['trace_xyz']@matrix
        second['material_height_m'] += [-60., 140.]
        second['trace_erosion_m'][:] = 60.
        points = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
        with patch.object(export, 'sample', side_effect=AssertionError('No display elevations')), \
                patch.object(export, 'cells_at', side_effect=AssertionError('No display owners')):
            fields, info = export.interval_forcing(points, first, second, {'dt_myr': 2.})
        expected = (points[:2]@matrix-points[:2])*export.RADIUS_M/2e6
        np.testing.assert_allclose(fields['hdisp'][:2], expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(fields['upsub'], [0., 200./2e6, 0.], atol=1e-10)
        np.testing.assert_allclose(fields['relaxation_correction'], [60./2e6, 60./2e6, 0.], atol=1e-11)
        np.testing.assert_array_equal(fields['correction_supported'], [1, 1, 0])
        self.assertEqual(info['relaxation_fallback_nodes'], 0)

    def test_review_grid_changes_cannot_change_native_export_initial_or_forcing(self):
        first = frame(); second = deepcopy(first); second['time_myr'] = 2.
        second['material_height_m'] += 100.
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            outputs = []
            for i, shape in enumerate(((16, 8), (40, 20))):
                run = base/f'run{i}'; run.mkdir()
                for index, original in enumerate((first, second)):
                    saved = deepcopy(original); saved['width'], saved['height'] = shape
                    n = shape[0]*shape[1]
                    saved['elevation'] = np.full(n, -9000. if i else 9000.)
                    saved['plate'] = np.full(n, i, np.int32)
                    saved['crust'] = np.full(n, i, np.uint8)
                    saved['boundary'] = np.full(n, i, np.uint8)
                    write_frame(run, index, saved)
                manifest = dict(run_id='native-grid-independence', config={'dt_myr': 2.},
                                frames=[{'time_myr': 0.}, {'time_myr': 2.}])
                output = base/f'export{i}'
                with patch.object(export, 'sample', side_effect=AssertionError('No raster sample')), \
                        patch.object(export, 'cells_at', side_effect=AssertionError('No raster cells')):
                    export.build_history(run, output, manifest, subdivisions=1, dt_years=100000)
                outputs.append(output)
            for name in ('mesh.npz', 'forcing_0000.npz'):
                with np.load(outputs[0]/'input'/name) as a, np.load(outputs[1]/'input'/name) as b:
                    for key in a.files:
                        np.testing.assert_array_equal(a[key], b[key], err_msg=name+':'+key)
            metadata = json.loads((outputs[0]/'export_metadata.json').read_text())
            for name, digest in metadata['native_sampling_sources_sha256'].items():
                self.assertEqual(hashlib.sha256((outputs[0]/'source'/name).read_bytes()).hexdigest(), digest)
            self.assertTrue({'native_frame_sampling.py', 'mesh_geometry.py', 'mesh_transport.py',
                             'material_reconstruction.py', 'native_boundary_geometry.py', 'material_transport.py',
                             'arc_surface.py', 'continental_margin.py', 'continental_contacts.py'}.issubset(
                                 metadata['native_sampling_sources_sha256']))
            # Captured exporter/helper imports must resolve without the live
            # application's module directory on sys.path.
            subprocess.run([sys.executable, '-I', '-c',
                            "import sys; sys.path.insert(0, '.'); import gospl_export; "
                            "assert {'continental_margin.py','continental_contacts.py'}.issubset(gospl_export.NATIVE_SAMPLING_SOURCES)"],
                           cwd=outputs[0]/'source', check=True, capture_output=True, text=True)

    def test_missing_native_fields_fail_instead_of_silently_sampling_display(self):
        source = frame()
        del source['mesh_ocean_relief_m']
        with self.assertRaisesRegex(ValueError, 'mesh_ocean_relief_m'):
            sampling.sample_frame(source, np.array([[0., 0., 1.]]))
        first = frame(); second = deepcopy(first); second['time_myr'] = 2.
        del second['mesh_version']
        with self.assertRaisesRegex(ValueError, 'cannot mix'):
            export.interval_forcing(np.array([[0., 0., 1.]]), first, second, {})


if __name__ == '__main__':
    unittest.main()
