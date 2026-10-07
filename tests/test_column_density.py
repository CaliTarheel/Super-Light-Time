"""Density gravity checked by independently integrating vertical mass moments."""
from types import SimpleNamespace
import unittest
import numpy as np
import column_density as density
import gravitational_relaxation as gravity
from tests.test_gravitational_relaxation import (
    fixture, sphere_polygon_area, planar_clip, unit, rotation)
from tests import test_gravitational_relaxation as oracle


def moment(layers):
    # Layers are physical (thickness km, density kg/m3), bottom to top.
    # Set the Airy-compensated bottom from total mass per area, then integrate
    # each layer's rho*z and subtract the displaced mantle's rho*z.
    mass = sum(h*rho for h, rho in layers)
    bottom = -mass/3300.
    value = .5*3300.*bottom**2
    for h, rho in layers:
        value += rho*((bottom+h)**2-bottom**2)/2.
        bottom += h
    return value


def independent(points, faces, volumes, reference, sheets, profile):
    triangles = points[faces]
    area = np.array([sphere_polygon_area(tri) for tri in triangles])
    height = volumes/area
    fraction = profile['dense_fraction']
    layers = [[(h*f, 3450.), (h*(1.-f), 2800.)] for h, f in zip(height, fraction)]
    physical = np.array([moment(row) for row in layers])
    factor = 2800.*(1.-2800./3300.)
    # Retain the pre-existing balanced-reference background, with its changed
    # material coefficient. This is separate from the literal mass moments.
    total = float(np.sum(area*physical*(height-reference)**2/height**2))
    for a in range(len(faces)):
        for b in range(a+1, len(faces)):
            if sheets[a] == sheets[b]:
                continue
            xy = planar_clip(triangles[a, :, :2]/triangles[a, :, 2, None],
                             triangles[b, :, :2]/triangles[b, :, 2, None])
            if not len(xy):
                continue
            overlap = sphere_polygon_area(unit(np.column_stack((xy, np.ones(len(xy))))))
            lower, upper = (b, a) if sheets[b] in profile['sheet_order'].get(int(sheets[a]), ()) else (a, b)
            total += overlap*(moment(layers[lower]+layers[upper])-physical[a]-physical[b])
    return total/factor


class ColumnDensityTests(unittest.TestCase):
    def profile(self, fraction=(.15, .45), reverse=False):
        return dict(dense_fraction=np.array(fraction), sheet_order={2: {7}} if reverse else {7: {2}})

    def test_layer_integral_and_fixed_volume_virtual_work(self):
        points, faces, h, ref, sheets = fixture()
        volume = np.array([sphere_polygon_area(tri) for tri in points[faces]])*h
        rng = np.random.default_rng(58)
        for reverse in (False, True):
            profile = self.profile(reverse=reverse)
            value, gradient, _ = gravity.energy_gradient(points, faces, h, ref, sheets, density_profile=profile)
            self.assertAlmostEqual(value/independent(points, faces, volume, ref, sheets, profile), 1., places=12)
            for _ in range(3):
                motion = rng.normal(size=points.shape)
                motion -= points*np.sum(points*motion, axis=1)[:, None]
                epsilon = 2e-7
                numerical = (independent(unit(points+epsilon*motion), faces, volume, ref, sheets, profile)
                             -independent(unit(points-epsilon*motion), faces, volume, ref, sheets, profile))/(2*epsilon)
                analytic = float(np.sum(gradient*motion))
                self.assertAlmostEqual(analytic/numerical, 1., places=6)

    def test_zero_phase_recovers_legacy_and_common_rotation_has_zero_work(self):
        points, faces, h, ref, sheets = fixture()
        old_value, old_gradient, _ = gravity.energy_gradient(points, faces, h, ref, sheets)
        value, gradient, _ = gravity.energy_gradient(points, faces, h, ref, sheets, density_profile=self.profile((0., 0.)))
        self.assertAlmostEqual(value/old_value, 1., places=14)
        np.testing.assert_allclose(gradient, old_gradient, rtol=2e-14, atol=1e-6)
        profile = self.profile()
        _, gradient, _ = gravity.energy_gradient(points, faces, h, ref, sheets, density_profile=profile)
        torque = np.sum(np.cross(points, gradient), axis=0)
        self.assertLess(np.linalg.norm(torque)/np.linalg.norm(gradient), 2e-14)
        matrix = rotation()
        value, moved, _ = gravity.energy_gradient(points@matrix.T, faces, h, ref, sheets, density_profile=profile)
        original, _, _ = gravity.energy_gradient(points, faces, h, ref, sheets, density_profile=profile)
        self.assertAlmostEqual(value/original, 1., places=12)
        np.testing.assert_allclose(moved, gradient@matrix.T, rtol=2e-12, atol=1e-5)

    def test_layer_order_changes_energy_and_missing_or_cyclic_order_is_rejected(self):
        points, faces, h, ref, sheets = fixture()
        first = gravity.energy_gradient(points, faces, h, ref, sheets, density_profile=self.profile())[0]
        second = gravity.energy_gradient(points, faces, h, ref, sheets, density_profile=self.profile(reverse=True))[0]
        self.assertGreater(abs(first-second), .1*abs(first))
        for order in ({}, {2: {7}, 7: {2}}):
            with self.assertRaises(ValueError):
                gravity.energy_gradient(points, faces, h, ref, sheets,
                    density_profile=dict(dense_fraction=np.array([.2, .3]), sheet_order=order))

    def test_physical_dense_layer_instability_is_not_clamped_to_buoyant_crust(self):
        own, _ = density.weights(np.array([1.]), np.array([2]), [], [], {})
        expected = 3450.*(1.-3450./3300.)/(2800.*(1.-2800./3300.))
        self.assertAlmostEqual(own[0], expected)
        self.assertLess(own[0], 0.)

    def test_relaxation_uses_same_density_energy_as_acceptance_and_diagnostics(self):
        points, faces, h, ref, sheets = fixture()
        area = np.array([sphere_polygon_area(tri) for tri in points[faces]])
        profile = self.profile()
        result, report = oracle.GravitationalEvolutionTests().advance(points, faces, area*h, ref, sheets, area, .5,
                                                              density_profile=profile)
        self.assertLess(report['energy_after_km4'], report['energy_before_km4'])
        independent_final = independent(result, faces, area*h, ref, sheets, profile)
        self.assertAlmostEqual(report['energy_after_km4']/independent_final, 1., places=12)

    def test_simulation_profile_requires_explicit_inventory_and_survives_strain(self):
        state = dict(thickness_km=np.array([40.]), area_factor=np.array([1.]),
                     **{density.DENSE_VOLUME_FIELD: np.array([10.])})
        s = SimpleNamespace(retained_dense_crust_version=1, structure=state, collision_contacts=[])
        np.testing.assert_allclose(density.options(s)['density_profile']['dense_fraction'], [.25])
        state['thickness_km'] *= 2.
        state['area_factor'] /= 2.
        np.testing.assert_allclose(density.options(s)['density_profile']['dense_fraction'], [.25])
        del state[density.DENSE_VOLUME_FIELD]
        with self.assertRaises(ValueError): density.options(s)
        self.assertEqual(density.options(SimpleNamespace()), {})

    def test_plate_force_reads_the_same_ordered_phase_energy(self):
        import mesh_coverage
        import plate_balance
        import native_material_evolution
        points, faces, h, ref, sheets = fixture()
        area = np.array([sphere_polygon_area(tri) for tri in points[faces]])
        profile = self.profile()
        owners = np.repeat([0, 1], 3)
        s = SimpleNamespace(retained_dense_crust_version=1, parcel_collision_sheet=sheets,
            collision_contacts=[dict(top_sheet=7, under_sheet=2)],
            structure=dict(thickness_km=h, reference_thickness_km=ref, area_factor=np.ones(2),
                           **{density.DENSE_VOLUME_FIELD: h*profile['dense_fraction']}),
            material_surface=dict(vertices=points, faces=faces, area_km2=area, vertex_owner=owners),
            _collision_overlap=mesh_coverage.material_overlaps(points, faces, sheets))
        model = plate_balance.Balance.__new__(plate_balance.Balance)
        model.s, model.size, model.slot, model.notes = s, 6, {0: 0, 1: 1}, {}
        torque = model._collision_torque().reshape(2, 3)
        _, gradient, _ = gravity.energy_gradient(points, faces, h, ref, sheets, density_profile=profile)
        physical = gravity.rigid_mode_partition(points, faces, area, gradient, owners)['torque_n_m']
        np.testing.assert_allclose(torque, physical*(plate_balance.CM_YR_M_S/plate_balance.RADIUS_M), rtol=2e-14)
        self.assertLess(np.linalg.norm(torque.sum(axis=0))/np.linalg.norm(torque), 2e-13)
        expected = independent(points, faces, area*h, ref, sheets, profile)
        self.assertAlmostEqual(native_material_evolution.reference_energy_state(s)/expected, 1., places=12)

    def test_subdivision_does_not_change_energy_or_integrated_plate_torque(self):
        points, faces, h, ref, sheets = fixture()
        profile = self.profile()
        value, gradient, _ = gravity.energy_gradient(points, faces, h, ref, sheets, density_profile=profile)
        expected = np.array([np.cross(points[:3], gradient[:3]).sum(axis=0),
                             np.cross(points[3:], gradient[3:]).sum(axis=0)])
        middle = unit(points[:3].sum(axis=0))
        refined = np.vstack((points, middle))
        triangles = np.array([[0, 1, 6], [1, 2, 6], [2, 0, 6], [3, 4, 5]])
        take = np.array([0, 0, 0, 1])
        profile['dense_fraction'] = profile['dense_fraction'][take]
        actual, derivative, _ = gravity.energy_gradient(refined, triangles, h[take], ref[take], sheets[take],
                                                       density_profile=profile)
        self.assertAlmostEqual(actual/value, 1., places=12)
        moment = np.cross(refined, derivative)
        torque = np.array([moment[[0, 1, 2, 6]].sum(axis=0), moment[3:6].sum(axis=0)])
        np.testing.assert_allclose(torque, expected, rtol=2e-11, atol=1e-4)


if __name__ == '__main__': unittest.main()
