"""Highland 65 initial plate ownership is distinct from crust geometry."""
from copy import deepcopy
import unittest

import numpy as np

import native_engine
import raster_engine
import server
import initial_worlds
from fracture import component_labels


class InitialPlateTopologyTests(unittest.TestCase):
    def test_highland50_connected_ocean_at_native_import_resolution(self):
        config = self.config(seed=37, width=192, height=96)
        initial = initial_worlds.make_highland65(config, continental_fraction=.50)
        imported = dict(config, width=256, height=128)
        world = raster_engine.Simulation(imported, deepcopy(initial))
        diagnostic = world.initial_plate_topology_diagnostics
        self.assertTrue(diagnostic['enabled'])
        self.assertEqual(diagnostic['independent_ocean_components'], 1)
        self.assertGreater(diagnostic['attached_ocean_fraction'], 0.)
        self.assertLessEqual(diagnostic['attached_ocean_fraction'], .45)
        self.assertLess(diagnostic['realized_apron_width_km'], 650./8)
        self.assertEqual(diagnostic['requested_apron_width_km'], 650.)
        self.assertEqual(diagnostic['requested_passive_coast_fraction'], .65)
        plain_initial = deepcopy(initial)
        plain_initial.pop('initial_plate_topology')
        plain = raster_engine.Simulation(imported, plain_initial)
        np.testing.assert_array_equal(world.initial_crust, plain.initial_crust)
        for slot in diagnostic['mixed_plate_slots']:
            self.assertTrue(np.any((world.plate == slot) & (world.initial_crust > 0)))
            self.assertTrue(np.any((world.plate == slot) & (world.initial_crust == 0)))

    def config(self, **updates):
        value = dict(width=128, height=64, plate_count=12, seed=41,
                     mechanics_nodes=128, duration_myr=10., dt_myr=2., snapshot_myr=2.)
        value.update(updates)
        return value

    def test_highland65_creates_mixed_plates_and_real_passive_margins(self):
        config = self.config()
        initial = raster_engine.make_initial(config, preset='highland65')
        authored = raster_engine.Simulation(config, deepcopy(initial))
        plain_initial = deepcopy(initial)
        plain_initial.pop('initial_plate_topology')
        plain = raster_engine.Simulation(config, plain_initial)

        diagnostic = authored.initial_plate_topology_diagnostics
        self.assertTrue(diagnostic['enabled'])
        self.assertEqual(diagnostic['independent_ocean_components'], 1)
        self.assertGreater(diagnostic['mixed_plate_count'], 0)
        self.assertGreater(diagnostic['passive_margin_length_km'], 0.)
        self.assertGreater(diagnostic['active_coast_length_km'], 0.)
        self.assertGreater(diagnostic['attached_ocean_fraction'], 0.)
        self.assertLessEqual(diagnostic['attached_ocean_fraction'], .45+1e-12)
        self.assertGreater(diagnostic['independent_ocean_fraction'], .54)
        self.assertEqual(authored.count, plain.count)
        np.testing.assert_array_equal(authored.initial_crust, plain.initial_crust)

        ocean = authored.initial_crust == 0
        attached = ocean & (authored.plate != 0)
        self.assertTrue(np.any(attached))
        self.assertTrue(np.all(authored.initial_crust[attached] == 0))
        _, components = component_labels(ocean & (authored.plate == 0), authored.w, authored.h)
        self.assertEqual(components, 1)

        a, b = authored.edge_a, authored.edge_b
        coast = (authored.initial_crust[a] > 0) ^ (authored.initial_crust[b] > 0)
        self.assertTrue(np.any(authored.plate[a[coast]] == authored.plate[b[coast]]))
        self.assertTrue(np.any(authored.plate[a[coast]] != authored.plate[b[coast]]))
        self.assertTrue(np.all(plain.plate[a[coast]] != plain.plate[b[coast]]))

    def test_highland65_plate_topology_is_deterministic(self):
        config = self.config(seed=93)
        initial = raster_engine.make_initial(config, preset='highland65')
        first = raster_engine.Simulation(config, deepcopy(initial))
        second = raster_engine.Simulation(config, deepcopy(initial))
        np.testing.assert_array_equal(first.plate, second.plate)
        self.assertEqual(first.initial_plate_topology_diagnostics,
                         second.initial_plate_topology_diagnostics)

    def test_uniform_world_skips_topology_without_failing(self):
        config = self.config(width=64, height=32)
        topology = dict(version=1, mode='passive_margin_aprons', seed=7,
                        passive_coast_fraction=.65, apron_width_km=650.,
                        max_attached_ocean_fraction=.45)
        initial = dict(width=64, height=32, crust=np.zeros(64*32, np.uint8),
                       initial_plate_topology=topology)
        world = raster_engine.Simulation(config, initial)
        self.assertFalse(world.initial_plate_topology_diagnostics['enabled'])

    def test_server_validation_preserves_topology_provenance(self):
        config = native_engine.validate_config(self.config())
        initial = native_engine.make_initial(config, preset='highland65')
        validated = server.validate_initial(initial, config)
        self.assertEqual(validated['initial_plate_topology'], initial['initial_plate_topology'])
        self.assertEqual(validated['initial_subduction'], initial['initial_subduction'])
        self.assertEqual(validated['continental_lifecycle'], initial['continental_lifecycle'])
        self.assertEqual(validated['rift_traction'], initial['rift_traction'])

    def test_explicit_primordial_ocean_overrides_preset_attachment(self):
        config = self.config(width=64, height=32, mesh_level=3, coast_geometry_level=3,
            physics_profile='reviewed_v1',
            primordial_ocean=dict(enabled=True, plate_count=3,
                                  half_spreading_rate_cm_yr=2., maximum_age_myr=180.))
        initial = native_engine.make_initial(config, preset='highland65')
        world = native_engine.Simulation(config, initial)
        self.assertTrue(world.initial_plate_topology_overridden_by_primordial_ocean)
        self.assertFalse(world.initial_plate_topology_diagnostics['enabled'])
        self.assertEqual(world.primordial_ocean_diagnostics['requested_plate_count'], 3)
        self.assertEqual(world.primordial_ocean_diagnostics['continental_attachments'], [])


if __name__ == '__main__':
    unittest.main()
