"""Normal closure survives a transform display label and simultaneous sliding."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np

import normal_partition as partition
import native_subduction
import trench_history as history
import backarc
import slab_memory as slab
from ridge_geometry import rotate
from tests.test_native_subduction_oracle import fixture as capture_fixture
from tests._native_spreading_oracle import polygon_area
from tests.test_trench_history import fixture as trench_fixture, step


def oblique(*, geometry=1, maturity=1., split=1):
    s, axis, transported = capture_fixture(split=split, maturity=maturity)
    s.t = 0.
    s.bl = np.array([1200.])
    s.bcode[:] = 3
    s.normal_speed[:] = -5.
    s.omega[0] = [0., 20./6371., 2.5/6371.]
    s.omega[1] = -s.omega[0]
    s.native_subduction_geometry_version = geometry
    partition.upgrade(s)
    return s, axis, transported


class NormalPartitionTests(unittest.TestCase):
    def test_oblique_finite_capture_matches_independent_spherical_area(self):
        for geometry in (0, 1):
            for split in (1, 2, 4):
                with self.subTest(geometry=geometry, split=split):
                    s, axis, support = oblique(geometry=geometry, split=split)
                    incoming = rotate(axis, s.omega[0]*2.)
                    over = rotate(axis, s.omega[1]*2.)
                    expected = polygon_area(np.array([incoming[0], incoming[1], over[1], over[0]]))
                    removed = native_subduction.removal(s, support, 2.)
                    area = float(removed.sum(axis=0) @ s.cell_area)
                    self.assertGreater(area, 10000.)
                    self.assertAlmostEqual(area, expected, delta=1e-5)
                    np.testing.assert_array_equal(removed[1:], 0.)
                    s.bcode[:] = 2
                    np.testing.assert_allclose(native_subduction.removal(s, support, 2.), removed, atol=1e-12)

    def test_maturity_still_limits_capture_and_legacy_label_gate_is_preserved(self):
        areas = []
        for maturity in (0., .5, 1.):
            s, _, support = oblique(maturity=maturity)
            areas.append(float(native_subduction.removal(s, support, 2.).sum(axis=0) @ s.cell_area))
        self.assertEqual(areas[0], 0.)
        self.assertAlmostEqual(areas[1], .5*areas[2], places=7)
        s.normal_partition_version = 0
        np.testing.assert_array_equal(native_subduction.removal(s, support, 2.), 0.)

    def test_transform_capture_feeds_the_conservative_slab_inventory(self):
        s, _, support = oblique()
        s.slab_memory_version = slab.VERSION
        s.trench_id = np.ones(len(s.ba), int)
        s.age = np.full(s.n, 80.)
        s.plate_uid = np.arange(len(s.support))+11
        s.trench_systems = [dict(id=1, phase='mature', length_km=1200.,
            downgoing_plate_uid=11, overriding_plate_uid=12)]
        slab.ensure(s.trench_systems[0], conservative=True)
        native_subduction.removal(s, support, 2.)
        record = s.native_subduction_diagnostics
        self.assertGreater(record['removed_area_km2'], 0.)
        self.assertAlmostEqual(sum(r['area_km2'] for r in record['removed_area_by_trench']),
                               record['removed_area_km2'], places=8)
        s.t = 2.
        slab.advance(s, 2.)
        self.assertGreater(s.trench_systems[0][slab.RETAINED_MASS_FIELD], 0.)
        slab.validate_row(s.trench_systems[0], require_mass=True)

    def test_arc_loading_and_collision_shortening_ignore_display_label(self):
        import material_forcing
        for continental in (False, True):
            s = trench_fixture()
            s.bcode[:] = 3
            s.normal_speed[:] = -5.
            if continental:
                s.crust[:] = 1
                s.down[:] = -1
            partition.upgrade(s)
            belts = material_forcing.boundary_belts(s)
            self.assertEqual(len(belts['length_km']), len(s.ba))
            np.testing.assert_allclose(belts['values'][:, 0 if continental else 2],
                                       5. if continental else 6.25)

    def test_version_and_provenance_survive_checkpoint(self):
        from pathlib import Path
        import tempfile
        import checkpoint
        s = SimpleNamespace(t=0., config={}, rng=np.random.default_rng(123), native_subduction_version=1)
        partition.upgrade(s)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.npz'
            checkpoint.write_checkpoint(path, s, {'config': {}}, {})
            restored, _ = checkpoint.read_checkpoint(path, None, SimpleNamespace)
        self.assertEqual(partition.snapshot(restored), partition.snapshot(s))

    def test_foreland_target_survives_transform_label_with_the_same_mountain_load(self):
        import structure_engine
        s = trench_fixture(fronts=((31, 32, 63),))
        s.crust[:] = 1
        s.land_relief = np.full(s.n, 1780.)  # 2 km mountain load.
        s.normal_speed[:] = -5.
        s.bcode[:] = 4
        angle = 250./6371.
        points = np.cos(angle)*s.bmid + np.sin(angle)*s.bn
        sample = lambda: structure_engine._target(points, np.array([0]), np.array([1]),
                                                  structure_engine._foreland_inputs(s))
        expected = sample()
        np.testing.assert_allclose(expected, [375.], atol=1e-8)
        s.bcode[:] = 3
        np.testing.assert_array_equal(sample(), 0.)  # Explicit legacy contract.
        partition.upgrade(s)
        np.testing.assert_allclose(sample(), expected, atol=1e-8)
        for speed, length, continental in ((0., 300., True), (5., 300., True),
                                            (-5., 0., True), (-5., 300., False)):
            s.normal_speed[:] = speed; s.bl[:] = length
            s.crust[s.bb] = int(continental)
            np.testing.assert_array_equal(sample(), 0.)

    def test_rift_inversion_uses_actual_normal_motion_and_preserves_owner_and_budget_gates(self):
        from tests.test_rift_inversion import boundary, samples
        from rift_inversion import inversion_increment
        s = boundary(closing=5., code=3)
        s.t = 0.
        s.ba = np.array([0]); s.bb = np.array([1]); s.crust = np.array([1, 1])
        s.normal_speed = np.array([100.])  # Recomputed Euler motion is authoritative here.
        s.omega[0, 1] = 40./6371.
        self.assertEqual(inversion_increment(s, *samples(), 2.)[0], 0.)
        partition.upgrade(s)
        expected = -600.*np.expm1(-3.75*2./600.)
        for owner in (0, 1):
            self.assertAlmostEqual(inversion_increment(s, *samples(owner=owner), 2.)[0], expected, places=11)
        self.assertEqual(inversion_increment(s, *samples(owner=2), 2.)[0], 0.)
        self.assertEqual(inversion_increment(s, *samples(extension=0.), 2.)[0], 0.)
        s.crust[0] = 0  # Valid descending ocean loads the overrider only.
        self.assertEqual(inversion_increment(s, *samples(owner=0), 2.)[0], 0.)
        self.assertAlmostEqual(inversion_increment(s, *samples(owner=1), 2.)[0], expected, places=11)
        s.down[:] = -1
        self.assertEqual(inversion_increment(s, *samples(owner=1), 2.)[0], 0.)
        s.crust[:] = 1
        s.normal_speed[:] = -5.  # Stale closure cannot supply load after motion stops.
        for closing in (0., -5.):
            s.omega[0, 2] = closing/6371.
            self.assertEqual(inversion_increment(s, *samples(), 2.)[0], 0.)

    def test_contact_attachment_diagnostic_does_not_discard_oblique_collision(self):
        import eclogite_sink
        s = trench_fixture()
        s.crust[:] = 1; s.normal_speed[:] = -5.; s.bcode[:] = 3
        partition.upgrade(s)
        value = eclogite_sink._slab_attachment(s, dict(top_owner=0, under_owner=1),
                                              np.full(len(s.ba), .7))
        self.assertAlmostEqual(value, .7)
        s.normal_speed[:] = 0.
        self.assertEqual(eclogite_sink._slab_attachment(s, dict(top_owner=0, under_owner=1),
                                                      np.full(len(s.ba), .7)), 0.)

    def test_opening_and_pure_sliding_do_not_consume_ocean(self):
        for geometry in (0, 1):
            for opening in (0., 2.5/6371.):
                s, _, support = oblique(geometry=geometry)
                s.omega[0, 2] = -opening
                s.omega[1] = -s.omega[0]
                s.normal_speed[:] = 2.*opening*6371.
                removed = native_subduction.removal(s, support, 2.)
                self.assertLess(float(removed.sum(axis=0) @ s.cell_area), 1e-7)

    def test_five_km_myr_closure_accumulates_500_km_despite_transform_label(self):
        s = trench_fixture()
        s.bcode[:] = 3
        s.normal_speed[:] = -5.
        partition.upgrade(s)
        history.initialize(s)
        self.assertEqual(len(s.trench_systems), 1)
        self.assertEqual(s.trench_systems[0]['maturity'], 0.)
        for _ in range(5):
            step(s)
        self.assertAlmostEqual(s.trench_systems[0]['maturity'], .5)
        for i in range(45):
            s.bcode[:] = 2 if i % 2 else 3
            step(s)
        row = s.trench_systems[0]
        self.assertAlmostEqual(row['shortening_km'], 500.)
        self.assertEqual(row['maturity'], 1.)
        self.assertEqual(row['episode'], 1)
        self.assertEqual(len(s.trench_systems), 1)

    def test_nonclosing_motion_does_not_initiate_a_trench(self):
        for speed in (0., 5.):
            s = trench_fixture()
            s.bcode[:] = 3
            s.normal_speed[:] = speed
            partition.upgrade(s)
            history.initialize(s)
            step(s)
            self.assertEqual(s.trench_systems, [])

    def test_native_boundary_populates_polarity_under_transform_label(self):
        from tests.test_native_processes import ocean_fixture
        s = ocean_fixture(level=2, two=True)
        s._boundaries()
        edge = np.flatnonzero(s.bl > 1.)[0]
        r, normal = s.bmid[edge], s.bn[edge]
        tangent = np.cross(r, normal)
        p, q = s.bp[edge], s.bq[edge]
        s.omega[q] = np.cross(r, -5.*normal+40.*tangent)/6371.
        s._boundaries()
        self.assertEqual(s.bcode[edge], 3)
        self.assertEqual(s.down[edge], -1)
        partition.upgrade(s)
        s._boundaries()
        self.assertEqual(s.bcode[edge], 3)
        self.assertAlmostEqual(s.normal_speed[edge], -5.)
        self.assertIn(s.down[edge], (p, q))

    def test_backarc_does_not_reapply_obliquity_threshold(self):
        s = trench_fixture(fronts=((31, 32, 63),))
        r, n = s.bmid[0], s.bn[0]
        s.omega[1] = np.cross(r, -5.*n+40.*np.cross(r, n))/6371.
        s.normal_speed[:] = -5.
        # Avoid lifecycle state: this test isolates eligibility geometry.
        s.bcode[:] = 2
        self.assertEqual(len(backarc.trench_edges(s, 0, 1)), 0)
        s.bcode[:] = 3
        partition.upgrade(s)
        np.testing.assert_array_equal(backarc.trench_edges(s, 0, 1), [0])

    def test_version_validation_and_idempotent_migration(self):
        s = SimpleNamespace(t=12., native_subduction_version=1)
        report = partition.upgrade(s)
        self.assertEqual(partition.upgrade(s), report)
        frame = dict(native_subduction_version=1, **partition.snapshot(s))
        partition.validate_frame(frame)
        self.assertEqual(partition.snapshot(SimpleNamespace(**deepcopy(frame))), partition.snapshot(s))
        for bad in (True, 2, -1, '1', 1.5):
            with self.assertRaises(ValueError):
                partition.enabled(SimpleNamespace(normal_partition_version=bad))
        with self.assertRaises(ValueError):
            partition.validate_frame(dict(normal_partition_version=1))


if __name__ == '__main__':
    unittest.main()
