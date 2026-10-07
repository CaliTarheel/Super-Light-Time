"""Matrix-free assembly parity, work and resource-growth tests; no solve claim."""
import unittest
import gc
import hashlib
import json
from pathlib import Path
import time
import tracemalloc

import numpy as np

import mesh_geometry
from benchmarks.collision_architecture import shared_contact as dense
from benchmarks.collision_architecture import sparse_shared_contact as sparse
from tests.test_shared_contact import fixture, relative, unit


def spherical_fixture(level):
    mesh = mesh_geometry.icosphere(level); p = mesh['vertices']
    return dict(points=p, faces=mesh['faces'], vertex_plate=np.zeros(len(p), int),
        radius_m=6371e3, basal_drag_pa_s_per_m=1e15, viscosity_pa_s=2e21,
        sheet_thickness_m=4e4, basal_reference_velocity_m_s=np.zeros_like(p),
        other_plate_rotational_drag_n_m_s=np.eye(3)*1e39,
        external_torques_n_m=np.array([[1e25, -2e25, .5e25]]),
        external_nodal_forces_n=np.zeros_like(p), contacts=[])


class SparseSharedContact(unittest.TestCase):
    def test_all_local_maps_and_hessian_match_reviewed_dense_reference(self):
        cfg = fixture(); cfg['basal_drag_pa_s_per_m'] *= np.arange(1, 7)
        cfg['basal_reference_velocity_m_s'] = np.cross([.2e-9, -.4e-9, .1e-9], cfg['points'])
        a = dense.assemble(**cfg); b = sparse.assemble(**cfg, face_chunk_size=1)
        for name in ('velocity_map', 'strain_design', 'gauge_matrix', 'contact_matrix', 'hessian_n_s_m'):
            self.assertLess(relative(a[name], b[name].to_dense()), 2e-14, name)
        for name in ('load_n', 'mantle_load_n', 'nodal_load_n', 'torque_load_n', 'nodal_area_m2'):
            self.assertLess(relative(a[name], b[name]), 2e-14, name)
        rng = np.random.default_rng(323)
        for _ in range(5):
            y = rng.normal(size=len(a['load_n']))*1e-9
            va, ga = dense.evaluate(a, y); vb, gb = sparse.evaluate(b, y)
            self.assertLess(abs(va-vb)/max(abs(va), abs(vb)), 2e-14)
            self.assertLess(relative(ga, gb), 2e-14)

    def test_local_transpose_and_joint_cross_blocks_are_work_conjugate(self):
        system = sparse.assemble(**fixture())
        rng = np.random.default_rng(513); size = len(system['load_n'])
        for name in ('velocity_map', 'strain_design', 'gauge_matrix', 'contact_matrix'):
            operator = system[name]; y = rng.normal(size=size); dual = rng.normal(size=operator.shape[0])
            first = float(dual@(operator@y)); second = float(y@operator.rmatvec(dual))
            self.assertLess(abs(first-second)/max(abs(first), abs(second), 1.), 2e-14, name)
        plate = np.r_[rng.normal(size=6), np.zeros(size-6)]
        sheet = np.r_[np.zeros(6), rng.normal(size=size-6)]
        a = float(plate@(system['hessian_n_s_m']@sheet))
        b = float(sheet@(system['hessian_n_s_m']@plate))
        self.assertLess(abs(a-b)/max(abs(a), abs(b)), 2e-14)

    def test_common_rotation_and_moving_mantle_physical_work(self):
        cfg = fixture(); omega = np.array([.2, -.3, .5])*1e-9
        cfg['basal_reference_velocity_m_s'] = np.cross(omega, cfg['points'])
        system = sparse.assemble(**cfg); y = np.r_[omega, omega, np.zeros(12)]
        self.assertLess(np.linalg.norm(system['contact_matrix']@y)/np.linalg.norm(omega), 2e-15)
        power = sparse.work(system, y)
        self.assertLess(power['basal_dissipation_w'], 1e-20)
        self.assertLess(power['viscous_dissipation_w'], 1e-20)
        # V=M cancels large basal terms in K*y-f. Compare against their
        # absolute products, rather than demanding relative precision in a
        # much smaller remainder. This is an arithmetic test, not a solve gate.
        products = abs(y)@abs(system['hessian_n_s_m']@y)+abs(y)@abs(system['load_n'])
        self.assertLess(abs(power['unsolved_power_defect_w']-power['generalized_gradient_work_w']),
            128*np.finfo(float).eps*products)

    def test_more_than_256_unknowns_and_linear_retained_storage_growth(self):
        measurements = []
        for level in (2, 3):
            system = sparse.assemble(**spherical_fixture(level))
            report = sparse.storage_report(system); measurements.append(report)
            self.assertGreater(report['unknowns'], 256)
            y = np.random.default_rng(level).normal(size=report['unknowns'])*1e-9
            power = sparse.work(system, y)
            self.assertGreaterEqual(power['viscous_dissipation_w'], 0.)
            self.assertTrue(np.isfinite(system['hessian_n_s_m']@y).all())
            with self.assertRaisesRegex(ValueError, 'small parity'):
                system['hessian_n_s_m'].to_dense()
        a, b = measurements
        ratio = b['retained_numpy_bytes']/a['retained_numpy_bytes']
        self.assertGreater(ratio, 3.5); self.assertLess(ratio, 4.2)
        self.assertLess(b['retained_numpy_bytes'], b['hypothetical_dense_hessian_bytes'])

    def test_chunk_size_does_not_change_assembled_operator(self):
        cfg = spherical_fixture(1)
        a = sparse.assemble(**cfg, face_chunk_size=3)
        b = sparse.assemble(**cfg, face_chunk_size=2048)
        np.testing.assert_array_equal(a['strain_design'].blocks, b['strain_design'].blocks)
        y = np.random.default_rng(87).normal(size=len(a['load_n']))*1e-9
        np.testing.assert_array_equal(a['hessian_n_s_m']@y, b['hessian_n_s_m']@y)

    def test_icosphere_operator_matches_dense_in_absolute_si_units(self):
        cfg = spherical_fixture(1); a = dense.assemble(**cfg); b = sparse.assemble(**cfg)
        y = np.random.default_rng(874).normal(size=len(a['load_n']))*1e-9
        self.assertLess(relative(a['hessian_n_s_m']@y, b['hessian_n_s_m']@y), 2e-14)
        expected = np.linalg.norm(a['strain_design']@y)**2
        self.assertLess(abs(sparse.work(b, y)['viscous_dissipation_w']-expected)/expected, 2e-14)

    def test_invalid_local_geometry_and_passivity_are_rejected(self):
        for edit in ('owner', 'drag', 'mantle', 'chunk'):
            cfg = fixture()
            if edit == 'owner': cfg['vertex_plate'][0] = 1
            elif edit == 'drag': cfg['other_plate_rotational_drag_n_m_s'][1, 1] = -1e20
            elif edit == 'mantle': cfg['basal_reference_velocity_m_s'][0] = cfg['points'][0]*1e-10
            else: cfg['face_chunk_size'] = 0
            with self.assertRaises(ValueError, msg=edit): sparse.assemble(**cfg)

    def test_hundreds_of_paired_rows_have_sparse_conjugate_actions(self):
        # Manufactured duplicate spherical sheets exercise row storage/scatter;
        # this is not a geological contact-admission or basal-allocation model.
        cfg = spherical_fixture(2); p = cfg['points']; f = cfg['faces']; count = len(p)
        cfg['points'] = np.r_[p, p]; cfg['faces'] = np.r_[f, f+count]
        cfg['vertex_plate'] = np.r_[np.zeros(count, int), np.ones(count, int)]
        cfg['basal_reference_velocity_m_s'] = np.zeros((2*count, 3))
        cfg['external_nodal_forces_n'] = np.zeros((2*count, 3))
        cfg['external_torques_n_m'] = np.r_[cfg['external_torques_n_m'], -cfg['external_torques_n_m']]
        cfg['other_plate_rotational_drag_n_m_s'] = np.eye(6)*1e39
        samples = []
        for i, face in enumerate(f):
            c = unit(p[face].sum(axis=0)); axis = np.eye(3)[np.argmin(abs(c))]
            samples.append(dense.ContactSample(i, i+len(f), c, unit(np.cross(c, axis)), 100.))
        cfg['contacts'] = samples; system = sparse.assemble(**cfg)
        self.assertEqual(len(samples), 320); self.assertGreater(len(system['load_n']), 256)
        self.assertEqual(system['contact_matrix'].blocks.shape, (640, 1, 9))
        rng = np.random.default_rng(80); y = rng.normal(size=len(system['load_n']))*1e-9
        dual = rng.normal(size=system['nplate']+len(samples))*1e18
        first = float(dual@sparse.constraint_action(system, y))
        second = float(y@sparse.constraint_transpose(system, dual))
        self.assertLess(abs(first-second)/max(abs(first), abs(second)), 2e-14)
        y[:] = 0.; y[:3] = y[3:6] = [.2e-9, -.4e-9, .1e-9]
        self.assertLess(np.max(abs(system['contact_matrix']@y)), 2e-24)

    def test_rotated_physical_fields_preserve_energy(self):
        cfg = fixture(); a = sparse.assemble(**cfg)
        rng = np.random.default_rng(109); y = rng.normal(size=len(a['load_n']))*1e-9
        residual = np.einsum('nia,na->ni', a['tangent_basis'], y[6:].reshape(-1, 2))
        axis = unit([.4, -.2, .7]); angle = .52; skew = dense._cross_matrix(axis)
        q = np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*(skew@skew)
        cfg['points'] = cfg['points']@q.T
        cfg['external_torques_n_m'] = cfg['external_torques_n_m']@q.T
        cfg['contacts'] = [dense.ContactSample(s.first_face, s.second_face,
            q@s.point, q@s.normal, s.length_m) for s in cfg['contacts']]
        b = sparse.assemble(**cfg)
        rotated = np.r_[(y[:6].reshape(-1, 3)@q.T).ravel(),
            np.einsum('nia,ni->na', b['tangent_basis'], residual@q.T).ravel()]
        one, two = sparse.evaluate(a, y)[0], sparse.evaluate(b, rotated)[0]
        self.assertLess(abs(one-two)/max(abs(one), abs(two)), 2e-13)
        np.testing.assert_allclose(a['contact_matrix']@y, b['contact_matrix']@rotated, rtol=2e-13, atol=1e-24)


def benchmark(output_path, levels=(3, 4, 5, 6)):
    """Explicit resource probe; never run implicitly by the unit test suite."""
    records = []
    for level in levels:
        gc.collect(); cfg = spherical_fixture(level)
        tracemalloc.start(); start = time.perf_counter()
        system = sparse.assemble(**cfg)
        elapsed = time.perf_counter()-start
        current, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        report = sparse.storage_report(system)
        y = np.random.default_rng(900+level).normal(size=report['unknowns'])*1e-9
        start = time.perf_counter()
        for _ in range(10): result = system['hessian_n_s_m']@y
        action_seconds = (time.perf_counter()-start)/10
        powers = sparse.work(system, y)
        report.update(level=level, assembly_seconds=elapsed, mean_matvec_seconds=action_seconds,
            tracemalloc_current_bytes=current, tracemalloc_peak_bytes=peak,
            finite_action=bool(np.isfinite(result).all()), power=powers,
            contact_samples=0, plate_count=1)
        records.append(report); del system, cfg, y, result
    source = Path(sparse.__file__)
    output = dict(source_path=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        numpy_version=np.__version__, levels=records,
        scope='Frozen single-sheet spherical assembly and Hessian action only; no solve or trajectory.',
        memory_scope='tracemalloc tracked allocations during assembly, not operating-system peak RSS; retained array buffers reported separately.',
        dense_comparison='Hypothetical dense Hessian storage is arithmetic only; a large dense matrix was never allocated.')
    Path(output_path).write_text(json.dumps(output, indent=2)+'\n', encoding='utf-8')
    return output


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == '--benchmark-output':
        print(json.dumps(benchmark(sys.argv[2]), indent=2))
    else:
        unittest.main()
