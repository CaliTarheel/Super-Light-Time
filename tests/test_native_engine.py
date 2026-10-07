"""Integrated contracts for the native mesh edition, independent of display."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from native_engine import Simulation, make_initial
from checkpoint import write_checkpoint, read_checkpoint
import mesh_history
import raster_engine
from orientation import orient_frame, rotation_matrix


def world(**updates):
    config = dict(width=96, height=48, mesh_level=2, duration_myr=8, dt_myr=2,
                  snapshot_myr=2, plate_count=4, mechanics_nodes=128, seed=37)
    config.update(updates)
    initial = make_initial(dict(width=192, height=96, seed=37, plate_count=4))
    return Simulation(config, initial)


class NativeEngineTests(unittest.TestCase):
    def test_mantle_and_force_equations_are_retained(self):
        self.assertIs(Simulation._forces, raster_engine.Simulation._forces)

    def test_tiny_arc_keeps_physical_area_in_control_mesh(self):
        s = Simulation(dict(width=96, height=48, mesh_level=2, plate_count=4),
                       dict(width=48, height=24, crust=np.zeros(48*24, np.uint8)))
        # Historical tiny-patch representation fixture; profile1 formation
        # and conserved pending capacity have separate moving-source tests.
        s.native_arc_birth_profile_version=0
        s._add_arc_crust(np.array([12]), np.array([1000.]))
        s._rasterize()
        self.assertAlmostEqual(s.mass.sum(), 1000., places=5)
        self.assertAlmostEqual(s.land_mass.sum(), 1000., places=5)
        self.assertLess(s.density.max(), .01)
        self.assertFalse(np.any(s.crust > 0))
        # Unresolved in the control classification does not erase geometry.
        face, _ = s._exposed_faces(s.pos, s.parcel_plate)
        np.testing.assert_array_equal(face, np.arange(len(s.mass)))

    def test_detached_subcell_island_keeps_its_own_persistent_domain(self):
        s = Simulation(dict(width=96, height=48, mesh_level=2, plate_count=4),
                       dict(width=48, height=24, crust=np.zeros(48*24, np.uint8)))
        # Retain this pre-profile tiny-island domain fixture and its assertions.
        s.native_arc_birth_profile_version=0
        s._add_arc_crust(np.array([170]), np.array([1000.]))
        s.active[1] = True
        s.plate_uid[1] = 2
        s.names[1] = 'Surrounding ocean'
        s.plate[:] = 1
        s.plate[0] = 0
        s.support[:] = 0.
        s.support[s.plate, np.arange(s.n)] = 1.
        s._rasterize(); s._update_surface_domains()
        island = int(s.material_domain[0])
        self.assertNotEqual(island, int(s.domain[0]))
        by_id = {row['uid']: row for row in s.domains}
        self.assertEqual(by_id[island]['plate_id'], 0)
        self.assertIn('fragment', by_id[island]['name'])
        s._update_surface_domains()
        self.assertEqual(int(s.material_domain[0]), island)

    def test_geometry_and_native_snapshot_are_independent_of_display_size(self):
        a, b = world(), world(width=192, height=96)
        for name in ('xyz', 'plate', 'crust', 'age', 'pos', 'mass', 'omega', 'mantle', 'parcel_patch'):
            np.testing.assert_array_equal(getattr(a,name), getattr(b,name), err_msg=name)
        first, second = a.snapshot(), b.snapshot()
        self.assertEqual(len(first['elevation']), 96*48)
        self.assertEqual(len(second['elevation']), 192*96)
        self.assertEqual(a.n, 320)
        for name in mesh_history.DTYPES:
            # Version-gated LIP/foundering fields are absent when disabled.
            # Compare presence as well as values; arrays() checks required fields.
            self.assertEqual(name in first, name in second, msg=name)
            if name in first:
                np.testing.assert_array_equal(first[name], second[name], err_msg=name)
        mesh_history.arrays(first)
        mesh_history.arrays(second)

    def test_evolution_does_not_read_back_the_display_grid(self):
        a, b = world(), world(width=192, height=96)
        for _ in range(3):
            a.step(); b.step()
            # Different snapshot cadences must not affect engine state either.
            b.snapshot()
            for name in ('plate', 'crust', 'age', 'pos', 'mass', 'omega', 'mantle', 'support'):
                np.testing.assert_array_equal(getattr(a,name), getattr(b,name), err_msg=name)
        for name in a.structure:
            np.testing.assert_array_equal(a.structure[name], b.structure[name], err_msg=name)

    def test_native_and_material_geometry_rotate_with_export(self):
        frame = world().snapshot()
        pose = dict(yaw=42., pitch=65., roll=-17.)
        rotated = orient_frame(frame, pose)
        rotation = rotation_matrix(pose)
        for name in ('mesh_vertices', 'material_vertices', 'trace_xyz'):
            np.testing.assert_allclose(rotated[name], frame[name]@rotation, atol=1e-14)
        for name in ('mesh_faces', 'material_faces', 'material_face_id', 'mesh_area_km2'):
            np.testing.assert_array_equal(rotated[name], frame[name])
        self.assertTrue(frame['boundary_segments'])
        for before, after in zip(frame['boundary_segments'], rotated['boundary_segments']):
            np.testing.assert_allclose(after['geometry_xyz'], np.asarray(before['geometry_xyz'])@rotation, atol=1e-14)
            np.testing.assert_allclose(after['normal'], np.asarray(before['normal'])@rotation, atol=1e-14)
            self.assertEqual(before['down'], after['down'])
        mesh_history.arrays(rotated)

    def test_history_validator_rejects_detached_geometry_or_nonfinite_state(self):
        frame = world().snapshot()
        frame['material_faces'] = frame['material_faces'].copy()
        frame['material_faces'][0,0] = len(frame['material_vertices'])
        with self.assertRaises(ValueError):
            mesh_history.arrays(frame)
        frame = world().snapshot()
        frame['mesh_age_myr'][0] = np.nan
        with self.assertRaises(ValueError):
            mesh_history.arrays(frame)


if __name__ == '__main__':
    unittest.main()
