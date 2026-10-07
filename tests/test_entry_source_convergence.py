"""Numerical source-split check for a complete inherited oceanic trench."""
import unittest

import numpy as np

import entry_regions
import plate_balance
import slab_memory
import slab_tether_history
import slab_tether_local
import slab_tether_native
import trench_history
from tests.test_automatic_entry import automatic_native_world
from tests.test_slab_tether import neck


def complete_front_world():
    state = automatic_native_world()
    state.kind[:] = 1
    state.native_arc_birth_profile_version = 1
    state.native_arc_footprint_version = 1

    # The short-entry fixture starts with only four inherited trench edges.
    # Define the *whole* closing half of this oceanic interface as the initial
    # condition; otherwise a source-split comparison measures fresh initiation
    # and redistribution of an arbitrarily short initial slab.
    model = plate_balance.Balance(state, 1., slab_tethers=True)
    model.solve()
    state.omega = model.rotation()
    state._boundaries()
    edges = np.flatnonzero((state.normal_speed < 0.) & (state.bmid[:, 0] > 0.))
    assert len(edges) == 12
    state.trench_systems = []
    state.next_trench_id = 1
    row = trench_history._birth(state, edges, np.zeros(len(state.ba), int))
    row.update(phase='mature', maturity=1., active_myr=10., shortening_km=100.)
    area = row['length_km'] * 300.
    mass = area * 1e6 * float(slab_memory.excess_mass_per_area_kg_m2(80.))
    row.update(slab_fed_area_km2=area, slab_retained_area_km2=area,
               slab_retained_buoyancy_area_km2=area,
               slab_initial_excess_mass_kg=mass, slab_retained_excess_mass_kg=mass)
    slab_memory.refresh_line_load(row)
    slab_tether_history.initialize(row, neck(damage=0., failure_opening_m=100000.))
    slab_tether_local.localize(row, [row['center']], [1.], support_radius_km=15000.)
    state.omega[:] = 0.
    state._boundaries()
    trench_history.prepare(state)
    return state


class EntrySourceConvergenceTests(unittest.TestCase):
    def test_complete_front_keeps_trench_identity_and_converges_entry(self):
        results = []
        for source_step in (.0005, .00025, .000125):
            state = complete_front_world()
            report = slab_tether_native.advance(
                state, .001, new_neck=neck(failure_opening_m=100000.),
                support_radius_km=15000., max_source_step_myr=source_step,
                absolute_tolerance=1e-6, relative_tolerance=1e-5)
            self.assertEqual(len(state.trench_systems), 1)
            self.assertEqual(state.trench_systems[0]['phase'], 'mature')
            self.assertEqual(len(report['source_intervals']), round(.001/source_step))
            # Moving-hinge version 1 carries the trench with its upper plate;
            # the heuristic forearc sweep no longer runs on this path.
            self.assertEqual(plate_balance.subduction_response_version(state), 1)
            self.assertEqual(state.process_totals['trench_swept_km2'], 0.)
            slab_memory.validate_row(state.trench_systems[0], require_mass=True)
            potential = entry_regions.frozen(state)
            sample = potential.evaluate(state.material_surface['vertices'],
                                        state.material_surface['faces'],
                                        potential.volumes, potential.sheets,
                                        radius=potential.radius)
            results.append((entry_regions.energy_j(state),
                            float(sample['corner_entry_coordinate_m'].max())))
            self.assertLess(abs(state.material_column_budget['residual_km3']) /
                            state.material_column_budget['after_columns_volume_km3'], 1e-9)
        energy = np.array([item[0] for item in results])
        penetration = np.array([item[1] for item in results])
        self.assertGreater(energy.min(), 0.)
        self.assertGreater(penetration.min(), 0.)
        self.assertLess(np.ptp(energy) / energy.mean(), .002)
        self.assertLess(np.ptp(penetration) / penetration.mean(), .001)


if __name__ == '__main__':
    unittest.main()
