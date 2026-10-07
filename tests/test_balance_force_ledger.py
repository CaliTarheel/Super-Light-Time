"""Physical quadrature/work and finite-coordinate regressions for force export."""
from copy import deepcopy
import itertools
import math
import unittest
import numpy as np
import plate_balance as pb
import balance_force_ledger as ledger
from tests.test_moving_hinge_tether import moving_fixture
from tests.test_slab_force_ledger import opposed_fixture
from tests.test_slab_tether_forces import fixture


def uniform_basis(model, extra=0):
    result = {}
    for p in model.plates:
        field = np.zeros((len(model.s.xyz), 3, model.size+extra))
        field[:, :, 3*model.slot[p]:3*model.slot[p]+3] = np.eye(3)
        result[p] = field
    return result


def fractional_octants():
    """Octant triangles have an independent closed-form rotational integral."""
    s = moving_fixture()
    vertices = np.vstack((np.eye(3), -np.eye(3)))
    signs = np.array(list(itertools.product((-1., 1.), repeat=3)))
    faces = []
    for sign in signs:
        face = np.array([i if sign[i] > 0. else i+3 for i in range(3)])
        if np.linalg.det(vertices[face]) < 0.:
            face = face[[0, 2, 1]]
        faces.append(face)
    s.native_mesh = dict(vertices=vertices, faces=np.array(faces))
    s.xyz = signs/math.sqrt(3.)
    s.cell_area = np.full(8, 500.)
    s.plate = np.repeat([0, 1], 4)
    s.crust = np.zeros(8, int); s.age = np.full(8, 150.)
    s.support = np.array([np.where(s.plate == 0, .8, .2), np.where(s.plate == 0, .2, .8)])
    s.bl[:] = 0.
    metric = np.full((8, 3, 3), -2./(3.*math.pi))
    metric *= signs[:, :, None]*signs[:, None, :]
    for i in range(3):
        metric[:, i, i] = 2./3.
    return s, metric


def fractional_only_octants():
    """A transported owner retains real area but wins no control centre."""
    s, metric = fractional_octants()
    s.plate[:] = 0
    s.support[1] = np.arange(1., 9.)/20.
    s.support[0] = 1.-s.support[1]
    return s, metric


def localized_opposed():
    """Separate rim control neighborhoods retain the two incoming ports."""
    s = opposed_fixture()
    points = []
    for center in s.bmid:
        tangent = np.cross([0., 0., 1.], center)
        for d in (-.006, 0., .006):
            for z in (-.004, .004):
                point = center+d*tangent+np.array([0., 0., z])
                points.append(point/np.linalg.norm(point))
    # Incoming rim controls plus two well-conditioned upper footprints.
    s.xyz = np.vstack((points, np.eye(3), -np.eye(3), np.eye(3), -np.eye(3)))
    s.plate = np.r_[np.zeros(12, int), np.ones(6, int), np.full(6, 2, int)]
    s.cell_area = np.full(24, 500.)
    s.crust = np.zeros(24, int); s.age = np.full(24, 150.)
    s.ba = np.array([0, 6]); s.bb = np.array([12, 18])
    return s


class CompleteForceLedgerTests(unittest.TestCase):
    def test_all_current_terms_and_arbitrary_motion_match_full_gradient(self):
        rng = np.random.default_rng(314)
        for s in (fixture(), moving_fixture()):
            s.bmid[1] = [0., 1., 0.]; s.bn[1] = [0., 0., 1.]
            for attached in (False, True):
                model = pb.Balance(s, 1., slab_tethers=attached)
                for x in rng.normal(size=(4, model.size))*12.:
                    result = ledger.export(model, x)
                    expected = -model._evaluate(x, result['delta'])[1]
                    np.testing.assert_allclose(result['generalized_force'], expected, rtol=1e-11, atol=1e-4)
                    for p, entry in result['plates'].items():
                        np.testing.assert_array_equal(s.plate[entry['cells']], p)
                        np.testing.assert_allclose(entry['cell_generalized_force'].sum(axis=0),
                            expected[3*model.slot[p]:3*model.slot[p]+3], rtol=1e-11, atol=1e-4)
                    compiled = ledger.compile_mode(result, uniform_basis(model))
                    value, gradient, hessian = compiled.evaluate(x)
                    original = model._evaluate(x, result['delta'])
                    self.assertAlmostEqual(value/original[0], 1., places=11)
                    np.testing.assert_allclose(gradient, original[1], rtol=1e-11, atol=1e-4)
                    np.testing.assert_allclose(hessian, original[2], rtol=1e-11, atol=1e-4)
                    self.assertEqual(compiled._resistance_work(x, result['delta']),
                                     model._resistance_work(x, result['delta']))
                    self.assertFalse(result['diagnostics']['equilibrium_reaction_added'])

    def test_fractional_basal_integral_matches_analytic_octants_and_keeps_off_owned(self):
        s, metric = fractional_octants()
        model = pb.Balance(s, 1., slab_tethers=True)
        x = np.array([2., -4., 8., -3., 7., 5.])
        result = ledger.export(model, x)
        factor = pb.ASTHENOSPHERE_DRAG_PA_S_M*pb.KEEL[0]*500.e6*pb.CM_YR_M_S**2
        for p, entry in result['plates'].items():
            block = factor*np.einsum('c,cij->ij', s.support[p], metric)
            expected = -block@x[3*p:3*p+3]
            np.testing.assert_allclose(entry['cell_generalized_force'].sum(axis=0), expected,
                                       rtol=1e-13, atol=1e-6)
            self.assertEqual(entry['diagnostics']['off_owned_basal_contributions'], 4)
            self.assertAlmostEqual(entry['diagnostics']['off_owned_basal_support_weight'], .8)
            self.assertGreater(np.linalg.norm(entry['diagnostics']['off_owned_basal_torque_n_m']), 0.)
            self.assertEqual(len(entry['cells']), 4)
        np.testing.assert_allclose(pb._cell_rotation_metric(s.native_mesh), metric, rtol=1e-14, atol=1e-14)
        basis = uniform_basis(model, 1)
        basis[0][:4, 2, -1] = np.array([.1, .2, .5, -.2])
        compiled = ledger.compile_mode(result, basis)
        owned = ledger.compile_mode(result, {p: basis[p][entry['cells']]
                                    for p, entry in result['plates'].items()})
        np.testing.assert_array_equal(owned.stiffness, compiled.stiffness)
        np.testing.assert_array_equal(owned.torque, compiled.torque)
        z = np.r_[x, 0.]
        gradient = compiled.evaluate(z)[1]
        independent = sum(float(np.sum(entry['cell_generalized_force']*basis[p][entry['cells'], :, -1]))
                          for p, entry in result['plates'].items())
        self.assertAlmostEqual(-gradient[-1]/independent, 1., places=12)
        # Resistance must change when the daughter differential speed changes.
        moved = z.copy(); moved[-1] = 10.
        self.assertGreater(compiled.evaluate(moved)[2][-1, -1], 0.)
        self.assertNotEqual(compiled.evaluate(moved)[1][-1], gradient[-1])

    def test_fractional_only_owner_keeps_exact_finite_basal_work_and_rigid_modes(self):
        s, metric = fractional_only_octants()
        before = deepcopy(s)
        model = pb.Balance(s, 1., slab_tethers=True)
        factor = pb.ASTHENOSPHERE_DRAG_PA_S_M*pb.KEEL[0]*500.e6*pb.CM_YR_M_S**2
        rng = np.random.default_rng(617)
        for x in rng.normal(size=(3, model.size))*9.:
            result = ledger.export(model, x)
            entry = result['plates'][1]
            np.testing.assert_array_equal(entry['cells'], np.flatnonzero(s.support[1] > 0.))
            self.assertFalse(np.any(s.plate[entry['cells']] == 1))
            oracle = -factor*np.einsum('c,cij->ij', s.support[1], metric)@x[3:6]
            np.testing.assert_allclose(entry['cell_generalized_force'].sum(axis=0), oracle,
                                       rtol=1e-13, atol=1e-5)
            self.assertEqual(entry['diagnostics']['categorical_owned_controls'], 0)
            self.assertEqual(entry['diagnostics']['fractional_only_controls'], 8)
            self.assertEqual(result['diagnostics']['fractional_only_plate_slots'], [1])
            self.assertEqual(entry['diagnostics']['off_owned_basal_contributions'], 8)
            self.assertAlmostEqual(entry['diagnostics']['off_owned_basal_support_weight'],
                                   float(s.support[1].sum()))
            basis = uniform_basis(model)
            for prepared in (basis, {p: basis[p][row['cells']]
                                    for p, row in result['plates'].items()}):
                compiled = ledger.compile_mode(result, prepared)
                for found, wanted in zip(compiled.evaluate(x), model._evaluate(x, result['delta'])[:3]):
                    np.testing.assert_allclose(found, wanted, rtol=1e-12, atol=1e-3)
                self.assertEqual(compiled._resistance_work(x, result['delta']),
                                 model._resistance_work(x, result['delta']))
        for name in ('plate', 'support', 'active', 'omega', 'xyz', 'cell_area'):
            np.testing.assert_array_equal(getattr(s, name), getattr(before, name))

    def test_fractional_only_lift_uses_actual_area_fraction_and_same_adjoint(self):
        s, _ = fractional_only_octants()
        model = pb.Balance(s, 1., slab_tethers=True)
        spatial = ledger._Spatial(model, 400.)
        port = spatial.port(1, np.array([.8, -.2, .5]))
        weights = s.cell_area[port.cells]*s.support[1, port.cells]
        weights /= weights.sum()
        metric = np.einsum('c,cij->ij', weights, spatial.metrics[port.cells])
        oracle = (weights[:, None, None]*spatial.metrics[port.cells])@np.linalg.inv(metric)
        np.testing.assert_allclose(port.matrices, oracle, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(port.matrices.sum(axis=0), np.eye(3), rtol=1e-13, atol=1e-13)
        uniform_weights = s.cell_area[port.cells]/s.cell_area[port.cells].sum()
        wrong_metric = np.einsum('c,cij->ij', uniform_weights, spatial.metrics[port.cells])
        wrong = (uniform_weights[:, None, None]*spatial.metrics[port.cells])@np.linalg.inv(wrong_metric)
        self.assertGreater(np.linalg.norm(port.matrices-wrong), .01)
        rng = np.random.default_rng(811)
        field = rng.normal(size=(len(s.xyz), 3))
        force = rng.normal(size=3)
        lifted = np.einsum('cij,j->ci', port.matrices, force)
        self.assertAlmostEqual(float(np.sum(lifted*field[port.cells])),
                               float(force@port.motion(field)), places=13)

    def test_fractional_only_rigid_force_is_rotation_covariant(self):
        s, _ = fractional_only_octants()
        rotated = deepcopy(s)
        axis = np.array([2., -1., 3.]); axis /= np.linalg.norm(axis)
        cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]],
                          [-axis[1], axis[0], 0.]])
        rotation = (np.eye(3)*math.cos(.713)+np.outer(axis, axis)*(1.-math.cos(.713))
                    +cross*math.sin(.713))
        for name in ('xyz', 'bmid', 'bn'):
            setattr(rotated, name, getattr(rotated, name)@rotation.T)
        rotated.native_mesh['vertices'] = rotated.native_mesh['vertices']@rotation.T
        x = np.array([2., -4., 8., -3., 7., 5.])
        other = (x.reshape(-1, 3)@rotation.T).ravel()
        base = ledger.export(pb.Balance(s, 1., slab_tethers=True), x)
        turned = ledger.export(pb.Balance(rotated, 1., slab_tethers=True), other)
        np.testing.assert_allclose(turned['generalized_force'].reshape(-1, 3),
                                   base['generalized_force'].reshape(-1, 3)@rotation.T,
                                   rtol=1e-12, atol=1e-4)
        self.assertEqual(turned['diagnostics']['fractional_only_plate_slots'], [1])

    def test_fractional_only_owner_retains_common_mode_without_a_new_cut_domain(self):
        import breakup_mode
        import force_rifting
        from tests.test_effective_subduction_forces import fixture as lite_fixture
        s = lite_fixture(plate_count=3)
        s.support = (np.arange(3)[:, None] == s.plate).astype(float)
        old = s.plate == 2
        s.plate[old] = 0
        s.support[0, old] = .75
        s.support[2, old] = .25
        model = pb.Balance(s, 1.)
        x = np.arange(model.size, dtype=float)*.4-.9
        report = ledger.export(model, x)
        self.assertNotIn(2, force_rifting._loaded_plates(s, model, ledger=report))
        cells = report['plates'][0]['cells']
        piece = np.arange(len(cells)) < len(cells)//2
        basis, _ = breakup_mode._basis(s, model, report, 0, cells, piece, np.array([0., 0., 1.]))
        self.assertTrue(np.all(basis[2][:, :, -1] == 0.))
        for row in basis[2][:, :, 6:9]:
            np.testing.assert_array_equal(row, np.eye(3))
        compiled = ledger.compile_mode(report, basis)
        z = np.r_[x, 0.]
        got = compiled.evaluate(z)
        expected = model._evaluate(x, report['delta'])
        np.testing.assert_allclose(got[0], expected[0], rtol=1e-12)
        np.testing.assert_allclose(got[1][:-1], expected[1], rtol=1e-12, atol=1e-3)
        np.testing.assert_allclose(got[2][:-1, :-1], expected[2], rtol=1e-12, atol=1e-3)

    def test_fractional_stencil_does_not_duplicate_categorical_ridge_sources(self):
        s, _ = fractional_only_octants()
        faces = s.native_mesh['faces']
        s.native_mesh['edge_faces'] = np.array([(i, j) for i in range(len(faces))
            for j in range(i+1, len(faces)) if len(set(faces[i]) & set(faces[j])) == 2])
        s.age = np.array([10., 20., 40., 80., 15., 35., 60., 100.])
        s.cell_area = np.arange(8)*70.+500.
        force = pb.ridge_force_density(s)
        self.assertGreater(np.linalg.norm(force), 0.)
        lever = pb.CM_YR_M_S*np.cross(s.xyz, force)*(s.cell_area*1e6)[:, None]
        self.assertGreater(np.linalg.norm(lever.sum(axis=0)), 1.)
        model = pb.Balance(s, 1., slab_tethers=True)
        np.testing.assert_array_equal(model.drivers['ridge'][3:6], np.zeros(3))
        result = ledger.export(model, np.array([2., -4., 8., -3., 7., 5.]))
        ridge = [term for term in result['local_terms'] if term.kind == 'ridge_driver']
        self.assertEqual(len(ridge), np.count_nonzero(np.any(lever != 0., axis=1)))
        self.assertTrue(all(term.owners == (0,) for term in ridge))
        self.assertNotIn('ridge_driver', result['plates'][1]['localcomponents_n_m'])
        np.testing.assert_allclose(sum((term.drive for term in ridge), np.zeros(3)),
                                   model.drivers['ridge'][:3], rtol=1e-14)
        compiled = ledger.compile_mode(result, uniform_basis(model))
        x = np.array([-5., 3., 9., 2., -7., 4.])
        for got, wanted in zip(compiled.evaluate(x), model._evaluate(x, result['delta'])[:3]):
            np.testing.assert_allclose(got, wanted, rtol=1e-12, atol=1e-3)

    def test_missing_real_support_and_singular_fractional_lifts_still_fail_closed(self):
        s, _ = fractional_only_octants()
        s.support[1] = 0.
        spatial = ledger._Spatial(pb.Balance(s, 1., slab_tethers=True), 400.)
        with self.assertRaisesRegex(ledger.UnsupportedForceLedger, 'no owned native controls.*positive fractional support'):
            spatial.port(1, np.array([1., 0., 0.]))
        s, _ = fractional_only_octants()
        del s.native_mesh
        s.xyz[:] = [1., 0., 0.]
        spatial = ledger._Spatial(pb.Balance(s, 1., slab_tethers=True), 400.)
        with self.assertRaisesRegex(ledger.UnsupportedForceLedger, 'singular/ill-conditioned'):
            spatial.port(1, np.array([1., 0., 0.]))

    def test_opposing_slabs_have_finite_spatial_work_despite_cancelled_rigid_drive(self):
        s = localized_opposed()
        model = pb.Balance(s, 1., slab_tethers=True)
        x = np.zeros(model.size)
        result = ledger.export(model, x, band_km=100.)
        incoming = result['plates'][0]
        attached = incoming['localcomponents_n_m']['attached_slab']
        self.assertLess(np.linalg.norm(attached.sum(axis=0))/np.linalg.norm(attached), 1e-12)
        basis = uniform_basis(model, 1)
        basis[0][:6, 2, -1] = .5; basis[0][6:12, 2, -1] = -.5
        compiled = ledger.compile_mode(result, basis)
        z = np.zeros(model.size+1)
        exact = float(np.sum(incoming['cell_generalized_force']*basis[0][incoming['cells'], :, -1]))
        self.assertGreater(exact, 0.)
        self.assertAlmostEqual(-compiled.evaluate(z)[1][-1]/exact, 1., places=12)
        self.assertGreater(np.linalg.norm(incoming['torques_n_m']), 0.)

    def test_unknown_assembly_or_nonlocal_terms_fail_without_residual_repair(self):
        model = pb.Balance(moving_fixture(), 1., slab_tethers=True)
        model.stiffness[0, 0] += 1e20
        with self.assertRaisesRegex(ledger.UnsupportedForceLedger, 'does not cover current assembly'):
            ledger.export(model, np.zeros(model.size))

    def test_material_contact_interface_gpe_and_both_weld_coordinates_keep_work(self):
        from tests.test_local_weld_traction import fixture as contact_fixture
        from mesh_geometry import icosphere
        import collision_interface
        for version in (0, 1):
            s = contact_fixture()
            native = icosphere(0)
            n = len(native['faces'])
            s.native_mesh = native; s.xyz = native['xyz']; s.cell_area = native['area_km2']
            s.plate = np.repeat([0, 1], n//2); s.crust = np.zeros(n, int); s.age = np.full(n, 100.)
            s.bp = s.bq = s.ba = s.bb = np.zeros(0, int)
            s.bl = np.zeros(0); s.bmid = s.bn = np.zeros((0, 3))
            s.trench_systems = []; s.suture_weld_coordinate_version = version
            s.plate_resistance_version = 1
            collision_interface.upgrade(s)
            model = pb.Balance(s, 1.)
            x = np.array([.2, -.3, .4, -.7, .9, -.6])
            result = ledger.export(model, x)
            self.assertIn('collision_interface', result['plates'][0]['localcomponents_n_m'])
            self.assertIn('collision_gpe_driver', result['plates'][0]['localcomponents_n_m'])
            self.assertIn('suture_weld', result['plates'][0]['localcomponents_n_m'])
            compiled = ledger.compile_mode(result, uniform_basis(model))
            expected = model._evaluate(x, result['delta'])
            actual = compiled.evaluate(x)
            for found, wanted in zip(actual, expected[:3]):
                np.testing.assert_allclose(found, wanted, rtol=2e-10, atol=.01)
            for kind in ('collision_interface', 'suture_weld'):
                common = np.tile([.1, -.4, .7], 2)
                work = sum(float(np.sum(entry['localcomponents_n_m'][kind]
                    *common[3*model.slot[p]:3*model.slot[p]+3])) for p, entry in result['plates'].items())
                scale = sum(float(np.abs(entry['localcomponents_n_m'][kind]).sum())
                            for entry in result['plates'].values())
                if scale:
                    self.assertLess(abs(work)/scale, 1e-10)
                else:
                    self.assertEqual(work, 0.)
        model = pb.Balance(moving_fixture(), 1., slab_tethers=True)
        model.elements.append(dict(kind='custom_entry', shape='negative',
            rows=(np.ones((1, model.size)),), coefficient=np.ones(1), scale=np.ones(1), edges=None))
        with self.assertRaisesRegex(ledger.UnsupportedForceLedger, 'no physical geometry'):
            ledger.export(model, np.zeros(model.size))

    def test_reader_does_not_change_geometry_or_assembled_source_and_reports_json(self):
        import json
        s = moving_fixture(); model = pb.Balance(s, 1., slab_tethers=True)
        state = deepcopy(s); matrix = model.stiffness.copy(); drive = model.torque.copy()
        input_motion = np.arange(model.size, dtype=float)
        result = ledger.export(model, input_motion, band_km=None)
        np.testing.assert_array_equal(result['motion_coordinates'], np.arange(model.size, dtype=float))
        result['motion_coordinates'][:] = -99.
        np.testing.assert_array_equal(input_motion, np.arange(model.size, dtype=float))
        np.testing.assert_array_equal(s.omega, state.omega)
        np.testing.assert_array_equal(model.stiffness, matrix); np.testing.assert_array_equal(model.torque, drive)
        for key in ('xyz', 'plate', 'crust', 'age', 'cell_area', 'omega'):
            np.testing.assert_array_equal(getattr(s, key), getattr(state, key))
        self.assertEqual(s.trench_systems, state.trench_systems)
        json.dumps(result['diagnostics'], allow_nan=False)
        for entry in result['plates'].values():
            json.dumps(entry['diagnostics'], allow_nan=False)


if __name__ == '__main__':
    unittest.main()
