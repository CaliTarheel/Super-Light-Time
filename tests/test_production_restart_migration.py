"""Conservation and independent stack-temperature oracles at a restart boundary."""
from copy import deepcopy
import json
import unittest
import numpy as np

import dense_crust
import eclogite_sink
import phase_evolution
import production_restart_migration as migration
from tests.test_eclogite_sink import stack


def source(**kwargs):
    s = stack(**kwargs)
    eclogite_sink.upgrade_inventory(s)
    eclogite_sink.upgrade_depth_integration(s)
    s.t = 146.
    s.steps = 73
    s.config.update(dt_myr=2., physics_profile='legacy')
    s.rng = np.random.default_rng(1984)
    s.events = [dict(time_myr=140., type='historical')]
    s.plate_balance_version = s.plate_resistance_version = 1
    s.material_mechanics_version = s.same_sheet_nonpenetration_version = 1
    s.subduction_response_version = 0
    s.physics_profile_version = 0
    s.erosion_relief_version = 1
    s.foreland_loading_enabled = False
    s.trench_retreat_speed = np.zeros(3)
    s.deformation_diagnostics = dict(model='historical fixture; no accepted successor solve')
    return s


class ProductionRestartMigrationTests(unittest.TestCase):
    def test_reference_bath_uses_real_partial_burial_before_averaging(self):
        s = source(lower_km=60., upper_km=40., alone_km=70., thin_km=30.)
        before = migration._arrays(vars(s))
        original_rng = deepcopy(s.rng.bit_generator.state)
        actual, report = migration.migrate_production(s)
        # Independent two-region oracle: 0C + 15C/km at 3/4 own-column depth,
        # plus the whole upper 40km column only on its actual spherical overlap.
        overlap = s._collision_overlap
        covered = float(overlap['area_km2'].sum())/s.material_surface['area_km2'][1]
        expected = np.array([450., (1.-covered)*675.+covered*1275., 787.5, 337.5])
        np.testing.assert_allclose(dense_crust.temperature(actual.structure), expected, rtol=2e-12)
        np.testing.assert_allclose(dense_crust.temperature(actual.trace_structure), expected, rtol=2e-12)
        self.assertEqual(migration._arrays(vars(s)), before)
        self.assertEqual(s.rng.bit_generator.state, original_rng)
        for path, recorded in before.items():
            self.assertEqual(migration._arrays(vars(actual))[path], recorded, path)
        self.assertEqual(actual.config, s.config)
        self.assertEqual(actual.events, s.events)
        self.assertEqual((actual.t, actual.steps), (146., 73))
        self.assertFalse(np.shares_memory(actual.mass, s.mass))
        self.assertEqual(actual.physics_profile_version, 0)
        self.assertEqual(actual.subduction_response_version, 1)
        self.assertEqual(actual.erosion_relief_version, 2)
        self.assertEqual(actual.retained_phase_floor_version, 2)
        self.assertEqual(actual.same_sheet_nonpenetration_version, 1)
        self.assertEqual(actual.adaptive_timestep_version, 1)
        self.assertEqual(actual.complete_contact_version, 1)
        self.assertEqual(report['thermal_initial_condition']['buried_material_faces'], 1)
        self.assertFalse(report['thermal_initial_condition']['inherited_thermal_history_reconstructed'])
        for column in (actual.structure, actual.trace_structure):
            np.testing.assert_array_equal(column[dense_crust.DENSE], 0.)
            np.testing.assert_array_equal(column[dense_crust.CONVERTED], 0.)
            np.testing.assert_array_equal(column[dense_crust.RETURNED], 0.)
            dense_crust.validate(column)
        json.dumps(report, allow_nan=False)

    def test_unburied_sensitivity_changes_only_new_heat_ledgers(self):
        s = source()
        reference, _ = migration.migrate_production(s)
        unburied, report = migration.migrate_production(s, temperature_policy='unburied_reference')
        np.testing.assert_allclose(dense_crust.temperature(unburied.structure), [450., 675., 787.5, 337.5])
        self.assertGreater(dense_crust.temperature(reference.structure)[1],
                           dense_crust.temperature(unburied.structure)[1])
        self.assertEqual(report['thermal_initial_condition']['policy'], 'unburied_reference')
        self.assertEqual(report['thermal_initial_condition']['formula'],
                         'min(T_mantle, T_surface + geotherm * depth_fraction * own_thickness_km)')
        self.assertEqual(report['thermal_initial_condition']['maximum_applied_burial_temperature_increment_c'], 0.)
        different = [path for path, value in migration._arrays(vars(reference)).items()
                     if migration._arrays(vars(unburied)).get(path) != value]
        self.assertEqual(set(different), {
            'state/structure/'+dense_crust.HEAT, 'state/structure/'+dense_crust.HEAT_BASELINE,
            'state/trace_structure/'+dense_crust.HEAT, 'state/trace_structure/'+dense_crust.HEAT_BASELINE})

    def test_bath_caps_each_region_before_face_average(self):
        s = source(lower_km=80., upper_km=80.)
        actual, _ = migration.migrate_production(s)
        fraction = s._collision_overlap['area_km2'].sum()/s.material_surface['area_km2'][1]
        expected = (1.-fraction)*900.+fraction*1300.
        self.assertAlmostEqual(dense_crust.temperature(actual.structure)[1], expected, places=8)
        self.assertLess(float(dense_crust.temperature(actual.structure).max()), 1300.)

    def test_moving_hinge_and_geometry_prerequisites_fail_without_mutation(self):
        for attribute, value, pattern in [
            ('plate_resistance_version', 0, 'plate_resistance_version'),
            ('plate_resistance_version', 1., 'plate_resistance_version'),
            ('same_sheet_nonpenetration_version', 0, 'same_sheet_nonpenetration_version'),
            ('trench_retreat_speed', np.array([0., .1]), 'zero heuristic'),
            ('t', 148., '146 Myr'),
            ('continental_entry_regions', {'version': 1}, 'experimental')]:
            with self.subTest(attribute=attribute):
                s = source()
                setattr(s, attribute, value)
                before = migration._arrays(vars(s))
                with self.assertRaisesRegex(ValueError, pattern):
                    migration.migrate_production(s)
                self.assertEqual(migration._arrays(vars(s)), before)
                self.assertFalse(hasattr(s, 'retained_dense_crust_version'))

    def test_same_sheet_intersection_cannot_be_authorized_by_flag_alone(self):
        s = source()
        s.parcel_collision_sheet[0] = s.parcel_collision_sheet[1]
        with self.assertRaisesRegex(ValueError, 'zero same-sheet overlap'):
            migration.migrate_production(s)
        self.assertFalse(hasattr(s, 'retained_dense_crust_version'))

    def test_invalid_stack_order_rejects_atomic_migration(self):
        s = source()
        s.collision_contacts = []
        before = migration._arrays(vars(s))
        with self.assertRaisesRegex(ValueError, 'vertical order'):
            migration.migrate_production(s)
        self.assertEqual(migration._arrays(vars(s)), before)
        self.assertFalse(hasattr(s, 'retained_dense_crust_version'))

    def test_snapshot_identifies_actual_activation_without_fresh_profile(self):
        s, receipt = migration.migrate_production(source())
        view = migration.snapshot(s)
        self.assertEqual(view['subduction_response_version'], 1)
        self.assertEqual(view['retained_dense_crust_version'], 1)
        self.assertEqual(view['retained_phase_floor_version'], 2)
        self.assertEqual(view['production_restart_migration'], receipt)
        self.assertEqual(view['retained_phase_profile']['profile'], 'production_restart_reference_bath')
        self.assertEqual(s.physics_profile_version, 0)
        with self.assertRaisesRegex(ValueError, 'already recorded'):
            migration.migrate_production(s)


if __name__ == '__main__':
    unittest.main()
