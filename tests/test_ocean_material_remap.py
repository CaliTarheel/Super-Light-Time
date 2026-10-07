"""Conservative source measure differs from occupied destination area."""
from copy import deepcopy
import json
import unittest

import numpy as np

import mesh_geometry
import ocean_material_remap as remap


def unit(value):
    value = np.asarray(value, float)
    return value/np.linalg.norm(value, axis=-1, keepdims=True)


def octahedron():
    vertices = np.concatenate((np.eye(3), -np.eye(3)))
    faces = []
    for a in (0, 3):
        for b in (1, 4):
            for c in (2, 5):
                face = [a, b, c]
                if np.linalg.det(vertices[face]) < 0: face = [a, c, b]
                faces.append(face)
    return mesh_geometry.geometry(vertices, np.asarray(faces))


def patch():
    return unit([[[1., -.2, -.1], [1., .3, -.1], [1., -.2, .4]]])


def plan_for(source=None, mapped=None, fractions=None, owners=None, ids=None, mesh=None):
    source = patch() if source is None else source
    return remap.build_plan(octahedron() if mesh is None else mesh, source,
        source if mapped is None else mapped,
        np.full(len(source), 81) if owners is None else owners,
        np.arange(len(source)) if ids is None else ids,
        np.ones(len(source)) if fractions is None else fractions)


class OceanMaterialRemapTests(unittest.TestCase):
    def test_identity_has_unit_material_fraction_and_no_source_creation(self):
        mesh = octahedron(); source = mesh['vertices'][mesh['faces']]
        plan = plan_for(source=source, mesh=mesh)
        area = np.asarray(plan['source_area_m2'])
        result = remap.apply(plan, area*12e3, extensive={'heat_j': area*3e4}, units={'heat_j': 'J'})
        np.testing.assert_allclose(result['occupied_area_m2'], area, rtol=2e-13)
        np.testing.assert_allclose(result['volume_m3'], area*12e3, rtol=2e-13)
        np.testing.assert_allclose(result['thickness_m'], 12e3, rtol=2e-13)
        np.testing.assert_allclose(result['cell_total_occupied_fraction'], 1., rtol=2e-13)
        self.assertFalse(result['diagnostics']['sources_created'])

    def test_analytic_rigid_rotation_permutes_octahedral_cells_without_mass_change(self):
        mesh = octahedron(); source = mesh['vertices'][mesh['faces']]
        # Exact coordinate-cycle rotation, independent of a production Euler helper.
        mapped = source[..., [2, 0, 1]]
        plan = plan_for(source=source, mapped=mapped, mesh=mesh)
        values = np.arange(1, len(source)+1)*1e12
        result = remap.apply(plan, values)
        destination = np.argmax(unit(mapped.sum(axis=1))@mesh['xyz'].T, axis=1)
        expected = np.empty(len(source)); expected[destination] = values
        np.testing.assert_allclose(result['volume_m3'], expected, rtol=2e-13)
        np.testing.assert_allclose(plan['mapped_area_m2'], plan['source_area_m2'], rtol=2e-13)

    def test_general_rigid_rotation_on_icosphere_closes_every_donor(self):
        mesh = mesh_geometry.icosphere(1); source = mesh['vertices'][mesh['faces']]
        angle = .193
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                             [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        plan = plan_for(source=source, mapped=source@rotation.T, mesh=mesh)
        np.testing.assert_allclose(plan['source_fraction_residual'], 0., rtol=0., atol=2e-13)
        np.testing.assert_allclose(plan['mapped_area_m2'], plan['source_area_m2'], rtol=2e-13)
        result = remap.apply(plan, np.asarray(plan['source_area_m2'])*50e3)
        np.testing.assert_allclose(result['cell_total_occupied_fraction'], 1., rtol=2e-13)
        np.testing.assert_allclose(result['thickness_m'], 50e3, rtol=2e-13)

    def test_divergent_expansion_changes_area_and_thickness_not_volume_or_energy(self):
        source = patch(); mapped = unit(source*np.array([1., 2., 1.7]))
        plan = plan_for(source=source, mapped=mapped)
        old_area, new_area = plan['source_area_m2'][0], plan['mapped_area_m2'][0]
        self.assertGreater(new_area, old_area*2.)
        volume = old_area*80e3
        result = remap.apply(plan, [volume], extensive={'fracture_capacity_j': [7e20],
            'fracture_spent_j': [2e19], 'heat_j': [5e18]},
            units={'fracture_capacity_j': 'J', 'fracture_spent_j': 'J', 'heat_j': 'J'})
        self.assertAlmostEqual(sum(result['volume_m3'])/volume, 1., places=11)
        self.assertAlmostEqual(sum(result['occupied_area_m2'])/new_area, 1., places=11)
        self.assertLess(sum(result['volume_m3'])/sum(result['occupied_area_m2']), 40e3)
        for name, expected in (('fracture_capacity_j', 7e20), ('fracture_spent_j', 2e19), ('heat_j', 5e18)):
            self.assertAlmostEqual(sum(result['extensive'][name])/expected, 1., places=11)
        self.assertFalse(result['diagnostics']['fracture_capacity_reinitialized'])

    def test_projective_pullback_preserves_orthant_material_fractions(self):
        # Equal absolute corner coordinates give equal normalization factors
        # after this diagonal transformation, so the corner-defined map is
        # exactly the same global projective diagonal map.
        source = unit([[[1., -.3, -.2], [1., .3, -.2], [1., -.3, .2]]])
        identity = plan_for(source=source)
        expanded = plan_for(source=source, mapped=unit(source*np.array([1., 2., .6])))
        # A positive diagonal projective map leaves every coordinate-plane
        # boundary fixed. Each receiving orthant therefore contains precisely
        # the same source material, despite different spherical area fractions.
        self.assertEqual(identity['receiving_index'], expanded['receiving_index'])
        np.testing.assert_allclose(expanded['donor_material_fraction'], identity['donor_material_fraction'],
                                   rtol=2e-11, atol=1e-13)
        area_fraction = np.asarray(expanded['destination_overlap_area_m2'])/expanded['mapped_area_m2'][0]
        self.assertGreater(np.max(np.abs(area_fraction-expanded['donor_material_fraction'])), 1e-3)

    def test_partial_multiple_owner_occupancy_keeps_separate_inventories(self):
        source = np.repeat(patch(), 2, axis=0)
        plan = plan_for(source=source, fractions=[.25, .75], owners=[81, 902], ids=[7, 7])
        area = plan['source_area_m2'][0]
        result = remap.apply(plan, [area*.25*10e3, area*.75*20e3],
            extensive={'spent_j': [2e12, 9e12]}, units={'spent_j': 'J'})
        owners = np.asarray(result['owner_uids'])
        for owner, fraction, thickness, energy in ((81, .25, 10e3, 2e12), (902, .75, 20e3, 9e12)):
            take = owners == owner
            self.assertAlmostEqual(np.asarray(result['occupied_area_m2'])[take].sum()/(area*fraction), 1., places=11)
            np.testing.assert_allclose(np.asarray(result['thickness_m'])[take], thickness, rtol=2e-11)
            self.assertAlmostEqual(np.asarray(result['extensive']['spent_j'])[take].sum()/energy, 1., places=11)
        self.assertFalse(result['diagnostics']['occupancy_normalized'])
        self.assertEqual(result['source_inventory']['owner_uids'], [81, 902])
        self.assertEqual(result['source_inventory']['donor_ids'], [7, 7])

    def test_cross_owner_excess_is_preserved_and_reported_not_removed(self):
        mesh = octahedron(); triangle = mesh['vertices'][mesh['faces'][:1]]
        plan = plan_for(source=np.repeat(triangle, 2, axis=0), fractions=[.8, .8], owners=[1, 2], mesh=mesh)
        result = remap.apply(plan, [4e12, 6e12])
        self.assertAlmostEqual(max(result['cell_total_occupied_fraction']), 1.6, places=12)
        self.assertGreater(result['diagnostics']['unresolved_cell_excess_area_m2'], 0.)
        self.assertAlmostEqual(sum(result['volume_m3']), 1e13, delta=1.)
        self.assertFalse(result['diagnostics']['pointwise_cross_owner_capacity_checked'])

    def test_restart_json_roundtrip_and_repeated_apply_are_identical_and_read_only(self):
        plan = plan_for(mapped=unit(patch()*[1., 1.4, .8])); before = deepcopy(plan)
        restored = json.loads(json.dumps(plan))
        first = remap.apply(plan, [5e15], extensive={'budget_j': [8e12]}, units={'budget_j': 'J'})
        second = remap.apply(restored, [5e15], extensive={'budget_j': [8e12]}, units={'budget_j': 'J'})
        self.assertEqual(first, second); self.assertEqual(plan, before)
        restored['donor_material_fraction'][0] *= .5
        with self.assertRaisesRegex(ValueError, 'changed after geometric validation'):
            remap.apply(restored, [5e15])

    def test_inversion_and_same_owner_endpoint_overlap_are_rejected(self):
        source = patch()
        with self.assertRaisesRegex(ValueError, 'inverted endpoint'):
            plan_for(mapped=source[:, [0, 2, 1]])
        with self.assertRaisesRegex(ValueError, 'Source geometry has positive same-owner overlap'):
            plan_for(source=np.repeat(source, 2, axis=0))
        other = source.copy(); other[..., 1] += .7; other = unit(other)
        with self.assertRaisesRegex(ValueError, 'Mapped geometry has positive same-owner overlap'):
            plan_for(source=np.concatenate((source, other)), mapped=np.repeat(source, 2, axis=0))

    def test_resource_incomplete_sphere_and_inventory_errors_are_explicit(self):
        source = patch(); mesh = octahedron()
        with self.assertRaisesRegex(ValueError, 'resource limit'):
            remap.build_plan(mesh, source, source, [1], [2], [1.], max_pair_evaluations=1)
        mesh['faces'] = mesh['faces'][:-1]
        with self.assertRaises(ValueError): plan_for(mesh=mesh)
        plan = plan_for(fractions=[0.])
        with self.assertRaisesRegex(ValueError, 'unoccupied donor'):
            remap.apply(plan, [1.])
        with self.assertRaisesRegex(ValueError, 'explicit matching unit'):
            remap.apply(plan, [0.], extensive={'heat': [1.]})
        with self.assertRaises(ValueError): remap.apply(plan, [-1.])


if __name__ == '__main__':
    unittest.main()
