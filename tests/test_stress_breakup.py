"""Topology behavior contracts for the worldbuilding stress closure.

These are controlled boundary-loading scenarios, not Earth calibration.
"""
import unittest
from unittest.mock import patch

import numpy as np

from raster_engine import DEFAULT_CONFIG, Simulation


def ring_world():
    config = dict(DEFAULT_CONFIG, width=96, height=48, plate_count=8,
                  duration_myr=10, dt_myr=2, snapshot_myr=2, seed=24)
    simulation = Simulation(config, {"width": 96, "height": 48,
                                     "crust": np.ones(96 * 48, np.uint8)})
    # One large surrounding plate and two small independent plate islands.
    simulation.active[:] = False
    simulation.active[:3] = True
    simulation.plate[:] = 0
    simulation.plate[simulation.xyz[:, 0] > .86] = 1
    simulation.plate[simulation.xyz[:, 1] > .86] = 2
    simulation.parcel_plate[:] = simulation.plate[simulation._indices(simulation.pos)]
    simulation.trace_plate[:] = simulation.plate[simulation._indices(simulation.trace_xyz)]
    simulation.support[:] = 0
    simulation.support[simulation.plate, np.arange(simulation.n)] = 1
    simulation.born[:] = 0
    simulation.t = 200
    simulation.steps = 1
    simulation.omega[:] = 0
    simulation._boundaries()
    return simulation


def loading(simulation, plate=0):
    area = np.bincount(simulation.plate, weights=simulation.cell_area, minlength=simulation.capacity)
    mass = np.bincount(simulation.parcel_plate, weights=simulation.mass, minlength=simulation.capacity)
    perimeter = np.bincount(np.concatenate((simulation.bp, simulation.bq)),
                            weights=np.tile(simulation.bl, 2), minlength=simulation.capacity)
    return simulation._internal_loading(plate, area, mass, perimeter)[0]


class InternalStressTests(unittest.TestCase):
    def test_rigid_surrounding_plate_does_not_rupture_from_size_or_enclosure(self):
        simulation = ring_world()
        simulation.omega[:] = [.003, -.004, .002]
        simulation._boundaries()
        diagnostic = loading(simulation)
        self.assertEqual(diagnostic["surrounded_neighbors"], 2)
        self.assertEqual(diagnostic["load_proxy"], 0.)
        simulation.rift_clock[0] = 30
        with patch.object(simulation, "_split") as split, patch.object(simulation, "_split_ocean") as ocean_split:
            for _ in range(200):
                simulation.t += 2
                simulation._topology(2)
            split.assert_not_called()
            ocean_split.assert_not_called()
        self.assertLess(simulation.rift_clock[0], 4)

    def test_legacy_continental_clock_cannot_trigger_an_undeveloped_rift(self):
        simulation = ring_world()
        simulation.omega[1] = [0., 0., .012]
        simulation.omega[2] = [0., 0., -.012]
        simulation._boundaries()
        diagnostic = loading(simulation)
        self.assertGreater(diagnostic["incompatible_speed_km_myr"], 1)
        self.assertGreater(diagnostic["load_proxy"], .5)
        mantle = simulation.mantle.copy()
        simulation.rift_clock[:3] = 1e6
        simulation.ocean_rift_clock[:3] = 1e6
        with patch.object(simulation, "_split") as split, patch.object(simulation, "_split_ocean") as ocean_split:
            for _ in range(40):
                simulation.t += 2
                simulation._topology(2)
            split.assert_not_called()
            ocean_split.assert_not_called()
        self.assertLess(simulation.rift_clock[0], 1e6)
        self.assertFalse(any(row['phase'] == 'broken_through' for row in simulation.rift_systems))
        np.testing.assert_array_equal(simulation.mantle, mantle)

    def test_original_ocean_identity_cannot_acquire_an_automatic_rift_timer(self):
        simulation = ring_world()
        simulation.initial_ocean_plate_uid = int(simulation.plate_uid[0])
        simulation.omega[1] = [0., 0., .012]
        simulation.omega[2] = [0., 0., -.012]
        simulation._boundaries()
        simulation.rift_clock[0] = 300
        with patch.object(simulation, "_split") as split, patch.object(simulation, "_split_ocean") as ocean_split:
            simulation._topology(2)
            self.assertTrue(simulation.stress_diagnostics[0]["initial_ocean_protected"])
            self.assertFalse(simulation.stress_diagnostics[0]["eligible"])
            self.assertTrue(all(call.args[0] != 0 for call in split.call_args_list))
            self.assertTrue(all(call.args[0] != 0 for call in ocean_split.call_args_list))
        # Protection belongs to that historical identity, not its reusable slot.
        simulation.plate_uid[0] += 1000
        simulation._topology(2)
        self.assertFalse(simulation.stress_diagnostics[0]["initial_ocean_protected"])

    def test_welded_boundary_owners_do_not_leave_phantom_internal_loading(self):
        simulation = ring_world()
        simulation.omega[1] = [0., 0., .012]
        simulation.omega[2] = [0., 0., -.012]
        simulation._boundaries()
        self.assertGreater(loading(simulation)["load_proxy"], .5)
        # This is the real step ordering: welding changes parcel/grid owners
        # after boundaries were cached; topology precedes their final refresh.
        simulation._weld(1, 0)
        simulation._weld(2, 0)
        self.assertGreater(len(simulation.bp), 0)
        self.assertTrue(np.all(simulation.plate == 0))
        simulation.rift_clock[0] = 30.
        with patch.object(simulation, "_split") as split:
            simulation._topology(2)
            split.assert_not_called()
        self.assertEqual(simulation.stress_diagnostics[0]["load_proxy"], 0.)
        self.assertEqual(simulation.stress_diagnostics[0]["surrounded_neighbors"], 0)
        self.assertLess(simulation.rift_clock[0], 30.)

    def test_rupture_preserves_material_and_mantle_opening_rule_and_records_loading(self):
        simulation = ring_world()
        mass, kinds, positions = simulation.mass.copy(), simulation.kind.copy(), simulation.pos.copy()
        ids, trace_xyz = simulation.trace_id.copy(), simulation.trace_xyz.copy()
        p, q = 0, int(np.flatnonzero(~simulation.active)[0])
        old_mantle = simulation.mantle[p].copy()
        old_omega = simulation.omega[p].copy()
        diagnostic = dict(load_proxy=2., accumulated_load_myr=76., rupture_threshold_myr=75.)
        self.assertTrue(simulation._split(p, loading=diagnostic))
        np.testing.assert_array_equal(simulation.mass, mass)
        np.testing.assert_array_equal(simulation.kind, kinds)
        np.testing.assert_array_equal(simulation.pos, positions)
        np.testing.assert_array_equal(simulation.trace_id, ids)
        np.testing.assert_array_equal(simulation.trace_xyz, trace_xyz)
        self.assertTrue(np.any(simulation.parcel_plate == q))
        self.assertEqual(simulation.plate_parent_uid[q], simulation.plate_uid[p])
        event = simulation.events[-1]
        self.assertEqual(event["details"]["loading"], diagnostic)
        self.assertIn("curved", event["details"]["fracture_geometry"])
        self.assertLessEqual(event["details"]["new_disconnected_area_fraction"], .025)
        frac = event["details"]["new_plate_mass_fraction"]
        # Existing opposite opening formulas preserve the weighted mean motion
        # and inherited mantle component through the split.
        np.testing.assert_allclose(simulation.omega[p]*(1-frac)+simulation.omega[q]*frac,
                                   old_omega, atol=1e-15)
        np.testing.assert_allclose(simulation.mantle[p]*(1-frac)+simulation.mantle[q]*frac,
                                   old_mantle, atol=1e-15)
        self.assertEqual(simulation.born[p], simulation.t)
        self.assertEqual(simulation.born[q], simulation.t)
        self.assertEqual(simulation.rift_clock[p], 0.)

    def test_fracture_routes_around_a_coherent_craton(self):
        simulation = ring_world()
        center = np.array([0., -.8, .6])
        craton = (simulation.xyz @ center > .965) & (simulation.plate == 0)
        simulation.crust[craton] = 2
        parcel_craton = craton[simulation._indices(simulation.pos)]
        simulation.kind[parcel_craton] = 2
        trace_craton = craton[simulation._indices(simulation.trace_xyz)]
        simulation.trace_kind[trace_craton] = 2
        self.assertGreater(np.count_nonzero(parcel_craton), 10)
        self.assertTrue(simulation._split(0, preferred=np.array([0., 1., 0.])))
        self.assertEqual(len(np.unique(simulation.plate[craton])), 1)
        self.assertEqual(len(np.unique(simulation.parcel_plate[parcel_craton])), 1)
        self.assertEqual(len(np.unique(simulation.trace_plate[trace_craton])), 1)


if __name__ == "__main__":
    unittest.main()
