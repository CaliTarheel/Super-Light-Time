import unittest

import numpy as np

from plate_topology import (assign_material_components, domain_components,
                            plan_domain_repairs)
from raster_engine import Simulation


def sphere(width, height):
    lon, lat = np.meshgrid((np.arange(width)+.5)*2*np.pi/width-np.pi,
                           np.pi/2-(np.arange(height)+.5)*np.pi/height)
    xyz = np.column_stack((np.cos(lat).ravel()*np.cos(lon).ravel(),
                           np.cos(lat).ravel()*np.sin(lon).ravel(), np.sin(lat).ravel()))
    edges = np.pi/2-np.arange(height+1)*np.pi/height
    area = np.repeat(np.sin(edges[:-1])-np.sin(edges[1:]), width)*2*np.pi/width
    return xyz, area


class PlateConnectivityTests(unittest.TestCase):
    def test_full_domains_do_not_split_islands_on_connected_ocean(self):
        w, h = 16, 8
        plate = np.zeros((h, w), np.int16)
        plate[:, 7:9] = 1
        crust = np.zeros_like(plate)
        crust[2, 1] = crust[5, 13] = 1
        _, area = sphere(w, h)
        plan = plan_domain_repairs(plate, crust, w, h, area)
        self.assertEqual(len(plan.component_owner), 2)
        self.assertEqual(plan.fragments, [])
        self.assertEqual(plan.absorptions, [])

    def test_wrap_and_separate_polar_caps(self):
        w, h = 12, 6
        plate = np.zeros((h, w), np.int16)
        plate[2, 0] = plate[2, -1] = 3
        plate[0, 2] = plate[0, 8] = 4
        plate[-1, 2] = plate[-1, 8] = 4
        labels, owner = domain_components(plate, w, h)
        grid = labels.reshape(h, w)
        self.assertEqual(grid[2, 0], grid[2, -1])
        self.assertEqual(grid[0, 2], grid[0, 8])
        self.assertEqual(grid[-1, 2], grid[-1, 8])
        self.assertNotEqual(grid[0, 2], grid[-1, 2])
        np.testing.assert_array_equal(owner[labels], plate.ravel())

    def test_odd_width_polar_continuation_overlapping_columns(self):
        plate = np.zeros((5, 9), np.int16)
        plate[0, 0] = plate[0, 4] = plate[0, 5] = 2
        labels, _ = domain_components(plate, 9, 5)
        self.assertEqual(len(set(labels[[0, 4, 5]])), 1)

    def test_detached_material_gets_fragment_and_only_empty_crumb_is_absorbed(self):
        w, h = 32, 16
        plate = np.zeros((h, w), np.int16)
        plate[4:10, 2:9] = 1
        plate[7:10, 20:24] = 1
        plate[5, 15] = 1
        crust = np.zeros_like(plate)
        crust[5:9, 3:8] = 2
        crust[8, 21] = 1
        xyz, area = sphere(w, h)
        cells = np.array([6*w+4, 8*w+21])
        masses = np.array([4., 2.])
        plan = plan_domain_repairs(plate, crust, w, h, area, parcel_cells=cells,
                                   parcel_plate=np.ones(2, np.int16), parcel_mass=masses,
                                   parcel_xyz=xyz[cells], xyz=xyz)
        self.assertEqual(len(plan.fragments), 1)
        self.assertEqual(len(plan.absorptions), 1)
        fragment = plan.fragments[0]
        self.assertEqual(fragment['parent_slot'], 1)
        self.assertEqual(fragment['material_mass_km2'], 2.)
        self.assertEqual(plan.labels[8*w+21], fragment['component'])
        absorption = plan.absorptions[0]
        self.assertEqual(plan.labels[5*w+15], absorption['component'])
        self.assertEqual(plan.component_owner[absorption['target_component']], 0)
        self.assertGreater(absorption['shared_boundary_radians'], 0)
        np.testing.assert_array_equal(plate[7:10, 20:24], 1)
        np.testing.assert_array_equal(masses, [4., 2.])

    def test_fragment_capacity_never_reassigns_buoyant_material(self):
        w, h = 24, 12
        plate = np.zeros((h, w), np.int16)
        plate[2:7, 2:7] = 1
        plate[7:10, 16:20] = 1
        crust = (plate == 1).astype(np.uint8)
        _, area = sphere(w, h)
        plan = plan_domain_repairs(plate, crust, w, h, area, available_slots=0)
        self.assertEqual(plan.fragments, [])
        self.assertEqual(plan.absorptions, [])
        self.assertEqual(len(plan.deferred), 1)
        self.assertIn('capacity', plan.deferred[0]['reason'])
        self.assertEqual(plan.deferred[0]['parent_slot'], 1)
        self.assertGreater(plan.deferred[0]['land_area_km2'], 0)

    def test_overlapping_parcel_maps_only_to_local_same_owner(self):
        w, h = 24, 12
        xyz, area = sphere(w, h)
        plate = np.zeros((h, w), np.int16)
        plate[4:8, 3:7] = 1
        plate[5:8, 17:20] = 1
        labels, owner = domain_components(plate, w, h)
        cells = np.array([6*w+7, 6*w+16, 1*w+11])
        target = assign_material_components(labels, owner, w, h, cells,
                                            np.ones(3, np.int16), xyz[cells], xyz)
        self.assertEqual(target[0], labels[6*w+6])
        self.assertEqual(target[1], labels[6*w+17])
        self.assertEqual(target[2], -1)
        plan = plan_domain_repairs(plate, np.zeros_like(plate), w, h, area,
                                   parcel_cells=cells, parcel_plate=np.ones(3, np.int16),
                                   parcel_mass=np.array([10., 5., 3.]), parcel_xyz=xyz[cells], xyz=xyz)
        self.assertEqual(plan.unresolved_mass_by_owner, {1: 3.})
        self.assertAlmostEqual(plan.material_mass.sum(), 15.)

    def test_polar_parcel_neighborhood_uses_reflected_longitude(self):
        w, h = 24, 12
        plate = np.zeros((h, w), np.int16)
        plate[0, 12] = 1
        labels, owner = domain_components(plate, w, h)
        xyz, _ = sphere(w, h)
        result = assign_material_components(labels, owner, w, h, np.array([0]),
                                            np.array([1]), xyz[[0]], xyz)
        self.assertEqual(result[0], labels[12])

    def test_polar_material_at_quarter_turn_is_still_locally_adjacent(self):
        w, h = 48, 24
        plate = np.zeros((h, w), np.int16)
        plate[0, 12] = 1
        labels, owner = domain_components(plate, w, h)
        xyz, _ = sphere(w, h)
        result = assign_material_components(labels, owner, w, h, np.array([0]),
                                            np.array([1]), xyz[[0]], xyz)
        self.assertEqual(result[0], labels[12])
        # The same longitude gap at the equator is not a local relationship.
        plate[:] = 0
        plate[h//2, 12] = 1
        labels, owner = domain_components(plate, w, h)
        cell = h//2*w
        result = assign_material_components(labels, owner, w, h, np.array([cell]),
                                            np.array([1]), xyz[[cell]], xyz)
        self.assertEqual(result[0], -1)

    def test_absorption_chains_are_local_and_acyclic(self):
        w, h = 32, 16
        plate = np.zeros((h, w), np.int16)
        plate[3:7, 2:6] = 1
        plate[8:12, 2:6] = 2
        plate[7, 16] = 1
        plate[7, 17] = 2
        _, area = sphere(w, h)
        plan = plan_domain_repairs(plate, np.zeros_like(plate), w, h, area)
        links = {row['component']: row['target_component'] for row in plan.absorptions}
        adjacent = {tuple(pair) for pair in plan.adjacency}
        for source, target in links.items():
            self.assertIn(tuple(sorted((source, target))), adjacent)
            seen = {source}
            while target in links:
                self.assertNotIn(target, seen)
                seen.add(target)
                target = links[target]

    def test_initial_ocean_pinched_by_consumption_can_gain_fragment_identity(self):
        w, h = 24, 12
        plate = np.ones((h, w), np.int16)
        plate[2:7, 2:8] = 0
        plate[7:10, 16:20] = 0
        _, area = sphere(w, h)
        plan = plan_domain_repairs(plate, np.zeros_like(plate), w, h, area)
        self.assertEqual(len(plan.fragments), 1)
        self.assertEqual(plan.fragments[0]['parent_slot'], 0)
        protected = plan_domain_repairs(plate, np.zeros_like(plate), w, h, area, protected_slots=(0,))
        self.assertEqual(protected.fragments, [])

    def test_concave_craton_follows_material_component_mode_not_ocean_centroid(self):
        w, h = 48, 24
        yy, xx = np.mgrid[:h, :w]
        radius = ((xx-24)/8)**2+((yy-12)/6)**2
        ring = (radius < 1) & (radius > .40)
        # A narrow raster ownership gap divides the visible ring. Its true
        # cratonic parcels still share one protected material-group identity.
        ring[:, 24] = False
        plate = ring.astype(np.int16)
        crust = (ring*2).astype(np.uint8)
        xyz, area = sphere(w, h)
        cells = np.flatnonzero(ring)
        masses = np.where(cells % w < 24, 2., 1.)
        # Include a currently overlapping parcel whose center is entirely in
        # the enclosed ocean. Its valid group mates resolve its affiliation.
        cells = np.r_[cells, 12*w+24]
        masses = np.r_[masses, .1]
        owners = np.ones(len(cells), np.int16)
        plain = plan_domain_repairs(plate, crust, w, h, area, parcel_cells=cells,
                                    parcel_plate=owners, parcel_mass=masses,
                                    parcel_xyz=xyz[cells], xyz=xyz)
        valid_components = np.unique(plain.parcel_components[plain.parcel_components >= 0])
        self.assertEqual(len(valid_components), 2)
        self.assertEqual(plate[12, 24], 0)
        self.assertEqual(plain.parcel_components[-1], -1)
        chosen = plain.parcel_components[np.flatnonzero(cells % w < 24)[0]]
        grouped = plan_domain_repairs(plate, crust, w, h, area, parcel_cells=cells,
                                      parcel_plate=owners, parcel_mass=masses,
                                      parcel_xyz=xyz[cells], xyz=xyz,
                                      parcel_groups=np.full(len(cells), 7))
        np.testing.assert_array_equal(grouped.parcel_components, chosen)
        self.assertAlmostEqual(grouped.material_mass[chosen], masses.sum())
        self.assertEqual(grouped.unresolved_mass_by_owner, {})
        ungrouped = plan_domain_repairs(plate, crust, w, h, area, parcel_cells=cells,
                                        parcel_plate=owners, parcel_mass=masses,
                                        parcel_xyz=xyz[cells], xyz=xyz,
                                        parcel_groups=np.full(len(cells), -1))
        np.testing.assert_array_equal(ungrouped.parcel_components, plain.parcel_components)


class EngineDomainRepairTests(unittest.TestCase):
    def simulation(self, material=True, crust_override=None):
        w, h = 64, 32
        crust = np.zeros((h, w), np.uint8)
        if material:
            crust[8:19, 5:19] = 1
            crust[20:25, 40:48] = 2
        if crust_override is not None:
            crust = crust_override.copy()
        sim = Simulation(dict(width=w, height=h, plate_count=4, duration_myr=2),
                         initial=dict(width=w, height=h, crust=crust.ravel()))
        sim.plate = np.where(crust.ravel() > 0, 1, 0).astype(np.int16)
        sim.active[:] = False
        sim.active[:2] = True
        sim.plate_uid[:] = 0
        sim.plate_uid[:2] = [1, 2]
        sim.plate_generation[:] = sim.active.astype(np.int32)
        sim.plate_parent_uid[:] = 0
        sim.next_plate_uid = 3
        sim.count = 2
        sim.names[:2] = ['Primordial ocean', 'Test parent']
        sim.parcel_plate[:] = 1
        sim.trace_plate[:] = 1
        sim.crust = crust.ravel()
        self.set_support(sim)
        return sim

    @staticmethod
    def set_support(sim):
        sim.support[:] = 0
        sim.support[sim.plate, np.arange(sim.n)] = 1

    def test_material_fragment_preserves_motion_material_and_history(self):
        sim = self.simulation()
        sim.omega[1] = [.001, -.002, .003]
        sim.mantle[1] = [-.003, .005, .007]
        sim.polarity[1, 0] = sim.polarity[0, 1] = 1
        sim.t = 100
        sim.born[1] = 42
        original = {name: getattr(sim, name).copy() for name in
                    ('pos', 'mass', 'kind', 'trace_id', 'trace_xyz', 'trace_origin_kind')}
        parcel_before, trace_before = sim.parcel_plate.copy(), sim.trace_plate.copy()
        sim._repair_plate_domains()
        fragments = [e for e in sim.events if e['type'] == 'plate_fragment']
        self.assertEqual(len(fragments), 1)
        q = int(sim.plate[22*sim.w+44])
        self.assertNotEqual(q, 1)
        self.assertNotEqual(sim.plate_uid[q], sim.plate_uid[1])
        self.assertEqual(sim.plate_parent_uid[q], sim.plate_uid[1])
        self.assertEqual(sim.plate_created[q], 100)
        np.testing.assert_array_equal(sim.omega[q], sim.omega[1])
        np.testing.assert_array_equal(sim.mantle[q], sim.mantle[1])
        self.assertEqual(sim.polarity[q, 0], q)
        self.assertEqual(sim.polarity[0, q], q)
        self.assertEqual(sim.born[q], sim.born[1])
        self.assertGreater(sim.count, q)
        self.assertGreater(np.count_nonzero(sim.parcel_plate != parcel_before), 0)
        self.assertGreater(np.count_nonzero(sim.trace_plate != trace_before), 0)
        for name, value in original.items():
            np.testing.assert_array_equal(getattr(sim, name), value)
        labels, owners = domain_components(sim.plate, sim.w, sim.h)
        self.assertEqual(len(owners), len(np.unique(owners)))
        before = (sim.plate.copy(), sim.parcel_plate.copy(), sim.trace_plate.copy(), len(sim.events))
        sim._repair_plate_domains()
        np.testing.assert_array_equal(sim.plate, before[0])
        np.testing.assert_array_equal(sim.parcel_plate, before[1])
        np.testing.assert_array_equal(sim.trace_plate, before[2])
        self.assertEqual(len(sim.events), before[3])

    def test_pinched_primordial_ocean_gets_fragment_without_opening_force(self):
        sim = self.simulation(material=False)
        grid = sim.plate.reshape(sim.h, sim.w)
        grid[:] = 1
        grid[7:18, 5:19] = 0
        grid[21:26, 39:47] = 0
        self.set_support(sim)
        sim.omega[0] = [.005, .002, -.001]
        sim.mantle[0] = [.006, -.003, .009]
        sim.initial_ocean_plate_uid = int(sim.plate_uid[0])
        sim._repair_plate_domains()
        q = int(sim.plate[23*sim.w+43])
        self.assertNotEqual(q, 0)
        self.assertEqual(sim.plate_parent_uid[q], sim.initial_ocean_plate_uid)
        np.testing.assert_array_equal(sim.omega[q], sim.omega[0])
        np.testing.assert_array_equal(sim.mantle[q], sim.mantle[0])
        self.assertEqual(sim.process_totals['rift_events'], 0)
        event = [e for e in sim.events if e['type'] == 'plate_fragment'][-1]
        self.assertFalse(event['details']['opening_impulse'])

    def test_polar_and_seam_continuity_do_not_create_false_fragment(self):
        sim = self.simulation(material=False)
        grid = sim.plate.reshape(sim.h, sim.w)
        grid[:] = 0
        grid[:8, :3] = 1
        grid[:8, -3:] = 1
        grid[:8, sim.w//2-2:sim.w//2+2] = 1
        self.set_support(sim)
        before = sim.plate.copy()
        sim._repair_plate_domains()
        np.testing.assert_array_equal(sim.plate, before)
        self.assertEqual(sim.domain_diagnostics['detached_created'], 0)

    def test_capacity_grows_in_blocks_and_preserves_every_plate_array(self):
        sim = self.simulation()
        old = sim.capacity
        names = ('active', 'rift_clock', 'ocean_rift_clock', 'born', 'plate_uid',
                 'plate_generation', 'plate_created', 'plate_parent_uid', 'omega',
                 'mantle', 'centres', 'polarity', 'collision_clock', 'support')
        before = {name: getattr(sim, name).copy() for name in names}
        sim._ensure_plate_capacity(old+9)
        self.assertEqual(sim.capacity, old+16)
        self.assertEqual(len(sim.names), sim.capacity)
        for name, expected in before.items():
            actual = getattr(sim, name)
            prefix = actual[:old, :old] if name in ('polarity', 'collision_clock') else actual[:old]
            np.testing.assert_array_equal(prefix, expected)
        self.assertTrue(np.all(sim.polarity[old:] == -1))
        self.assertTrue(np.all(sim.polarity[:, old:] == -1))
        self.assertFalse(np.any(sim.active[old:]))
        self.assertFalse(np.any(sim.support[old:]))

    def test_tiny_buoyant_fragment_is_not_absorbed_by_empty_ocean_cleanup(self):
        crust = np.zeros((32, 64), np.uint8)
        crust[8:19, 5:19] = 1
        crust[0, 20] = 2
        sim = self.simulation(crust_override=crust)
        sim._repair_plate_domains()
        q = int(sim.plate[20])
        self.assertNotEqual(q, 1)
        self.assertLess(sim.mass[sim.parcel_plate == q].sum(), sim.earth_area*.0001)
        uid = int(sim.plate_uid[q])
        sim.steps = 20
        sim._boundaries()
        sim._topology(2)
        self.assertTrue(sim.active[q])
        self.assertEqual(sim.plate_uid[q], uid)
        self.assertTrue(np.any(sim.parcel_plate == q))


if __name__ == '__main__':
    unittest.main()
