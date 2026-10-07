"""Finite attachment identity/remap tests; no native attachment or rupture law."""
from copy import deepcopy
from fractions import Fraction
from io import BytesIO
import json
import unittest
import warnings

import numpy as np

import adaptive_material
import checkpoint
import mesh_coverage
from native_material_adaptivity import _charts
from benchmarks.collision_architecture import attachment_transaction as tx


def leaves():
    return dict(face_id=np.array([10, 20]), root_id=np.array([7, 8]),
                reference_corners=np.array([np.eye(3), np.eye(3)]), owner_uid=np.array([41, 41]))


def store(corners=None, receiver=None):
    points = np.eye(3) if corners is None else np.asarray(corners, float)
    return dict(version=1, scope=tx.SCOPE, bonds=[dict(bond_id=1, creation_event_id=51,
        created_myr=102., source=dict(root_id=7, corners=points.copy()),
        receiver=dict(root_id=8, corners=points.copy() if receiver is None else np.asarray(receiver, float)))])


def refined_leaf_state():
    vertices = np.array([[1., 0., 0.], [1., .1, 0.], [1., 0., .1]])
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    faces = np.array([[0, 1, 2]])
    # Use the actual native conforming refinement and homogeneous chart map.
    mapping = adaptive_material.refine(vertices, faces, np.array([41]), np.array([1]),
        desired_edge_km=100., face_ids=np.array([10]))
    chart = _charts(np.eye(3)[None], mapping)
    return dict(face_id=np.array([100, 101, 102, 103, 20]), root_id=np.array([7, 7, 7, 7, 8]),
        reference_corners=np.concatenate((chart, np.eye(3)[None])), owner_uid=np.full(5, 41))


class FiniteAttachmentTransactions(unittest.TestCase):
    def test_nonreal_and_object_chart_types_fail_before_coercion(self):
        invalid = [np.eye(3, dtype=object), np.eye(3, dtype=complex)*(1.+3.j),
                   np.eye(3, dtype=bool), np.eye(3).astype(str), np.eye(3).astype('S8'),
                   np.eye(3, dtype=np.int64).astype('datetime64[D]')]
        if np.dtype(np.longdouble).itemsize > 8:
            invalid.append(np.eye(3, dtype=np.longdouble))
        for array in invalid:
            for side in ('source', 'receiver', 'leaf'):
                with self.subTest(dtype=str(array.dtype), side=side):
                    record = store(); state = leaves()
                    if side == 'leaf':
                        state['reference_corners'] = np.stack((array, array))
                    else:
                        record['bonds'][0][side]['corners'] = array
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        with self.assertRaisesRegex(ValueError, 'real integer'):
                            tx.prepare_transaction(record, state, state)
                    self.assertEqual(caught, [], 'Invalid coordinates must not reach a lossy numeric cast.')
        with self.assertRaisesRegex(ValueError, 'serialization-safe'):
            tx._signature(store(np.eye(3)), dict(leaves(), reference_corners=np.ones((2, 3, 3), object)))

    def test_accepted_chart_types_have_stable_signatures_and_checkpoint_roundtrip(self):
        for dtype in (np.float16, np.float32, np.float64, np.int32, np.int64, np.uint64):
            with self.subTest(dtype=dtype):
                first = store(); second = store()
                state = leaves()
                state['reference_corners'] = state['reference_corners'].astype(dtype)
                for record in (first, second):
                    for side in ('source', 'receiver'):
                        record['bonds'][0][side]['corners'] = np.eye(3, dtype=dtype)
                a = tx.prepare_transaction(first, state, state)
                b = tx.prepare_transaction(second, deepcopy(state), deepcopy(state))
                self.assertEqual(a['current_signature'], b['current_signature'])
                arrays = {}; encoded = checkpoint._encode([a['store'], state], arrays)
                archive_bytes = BytesIO(); np.savez(archive_bytes, **arrays); archive_bytes.seek(0)
                with np.load(archive_bytes, allow_pickle=False) as archive:
                    restored, restored_leaves = checkpoint._decode(json.loads(json.dumps(encoded)), archive)
                repeated = tx.prepare_transaction(restored, restored_leaves, restored_leaves)
                self.assertEqual(a['current_signature'], repeated['current_signature'])
                self.assertEqual(a['current_incidence'], repeated['current_incidence'])
                self.assertEqual(restored['bonds'][0]['source']['corners'].dtype, np.dtype(dtype))
        # Large accepted integer coordinates must not silently round through a
        # float64 conversion before their rational geometry is audited.
        large = np.eye(3, dtype=np.uint64); large[0, 0] = 2**53+1
        self.assertEqual(tx._matrix(large, 'Integer witness')[0][0], Fraction(2**53+1))

    def test_real_transfer_plans_cannot_cut_an_explicitly_authored_intact_bond(self):
        # These are actual native contact/planner/application operators. The
        # fixture is held at fixed geometry and has authored mature histories;
        # this external guard is not a force-balanced or integrated trajectory.
        from tests.test_terrane_attachment import (
            three_body_fixture, next_plans, apply_and_sync, preserved_material)
        simulation = three_body_fixture()
        first, = next_plans(simulation)
        self.assertEqual((first['source'], first['target']), (0, 1))
        triangles = simulation.material_surface['vertices'][simulation.material_surface['faces']]
        witness = None
        for source in first['parcel_indices']:
            for receiver in np.flatnonzero(simulation.parcel_plate == first['target']):
                polygon = mesh_coverage.clip_triangle(triangles[source], triangles[receiver])
                if len(polygon) < 3 or mesh_coverage._polygon_area(polygon, 6371.) <= 0:
                    continue
                center = polygon.sum(axis=0); center /= np.linalg.norm(center)
                patch = .9*center+.1*polygon[:3]
                patch /= np.linalg.norm(patch, axis=1)[:, None]
                a = np.linalg.solve(triangles[source].T, patch.T).T
                b = np.linalg.solve(triangles[receiver].T, patch.T).T
                # A strict interior finite witness avoids any boundary repair
                # or positive-area threshold serving as an admission rule.
                if np.linalg.det(a) > 0 and np.linalg.det(b) > 0 and np.all(a > 0) and np.all(b > 0):
                    witness = (int(source), int(receiver), a, b, patch)
                    break
            if witness is not None:
                break
        self.assertIsNotNone(witness)
        source, receiver, a, b, patch = witness
        self.assertGreater(mesh_coverage._polygon_area(patch, 6371.), 0.)
        apply_and_sync(simulation, first)
        self.assertEqual(simulation.t, 102.)
        # EXPLICIT diagnostic assertion made after the real first docking:
        # this small engaged material patch is now an intact bond. The primitive
        # does not infer or physically authorize this assertion from overlap.
        bond = dict(version=1, scope=tx.SCOPE, bonds=[dict(bond_id=1,
            creation_event_id=1, created_myr=simulation.t,
            source=dict(root_id=int(simulation.parcel_patch[source]), corners=a),
            receiver=dict(root_id=int(simulation.parcel_patch[receiver]), corners=b))])

        def current_leaves():
            return dict(face_id=simulation.parcel_patch.copy(),
                root_id=simulation.material_lineage['root_id'].copy(),
                reference_corners=np.tile(np.eye(3), (len(simulation.mass), 1, 1)),
                owner_uid=simulation.plate_uid[simulation.parcel_plate].copy())

        def array_snapshot(value, path=()):
            # Include every reachable array, including nested physical columns,
            # mesh, histories and planning caches, without copying opaque trees
            # or functions. Counters/events are checked separately below.
            if isinstance(value, np.ndarray):
                return {path: value.copy()}
            result = {}
            if isinstance(value, dict):
                for key, item in value.items():
                    result.update(array_snapshot(item, path+(repr(key),)))
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    result.update(array_snapshot(item, path+(str(index),)))
            return result

        now = current_leaves()
        tx.prepare_transaction(bond, now, now)
        physical_at_docking = preserved_material(simulation)
        counter_at_docking = simulation.process_totals['accreted_km2']
        for epoch in (104., 106., 108.):
            previous_time = simulation.t
            plan, = next_plans(simulation)
            self.assertEqual(simulation.t, previous_time+2.)
            self.assertEqual(simulation.t, epoch)
            self.assertEqual((plan['source'], plan['target']), (1, 0))
            np.testing.assert_array_equal(plan['parcel_indices'], np.arange(8))
            # Take the snapshot AFTER planning. next_plans intentionally ages
            # contact and maturity records; no rollback of that is claimed.
            before_arrays = array_snapshot(vars(simulation))
            before_counters = deepcopy(simulation.process_totals)
            before_events = deepcopy(simulation.events)
            now = current_leaves(); proposed = deepcopy(now)
            proposed['owner_uid'][plan['parcel_indices']] = plan['target_uid']
            with self.assertRaisesRegex(ValueError, 'owner cut'):
                tx.prepare_transaction(bond, now, proposed)
            after_arrays = array_snapshot(vars(simulation))
            self.assertEqual(before_arrays.keys(), after_arrays.keys())
            for key, expected in before_arrays.items():
                np.testing.assert_array_equal(after_arrays[key], expected, err_msg=str(key))
            self.assertEqual(simulation.process_totals, before_counters)
            self.assertEqual(tx._canonical(simulation.events), tx._canonical(before_events))
            # No apply_accretion call follows a rejected external preflight.
            for name, expected in physical_at_docking.items():
                np.testing.assert_array_equal(preserved_material(simulation)[name], expected, err_msg=name)
            self.assertEqual(simulation.process_totals['accreted_km2'], counter_at_docking)
            self.assertEqual(len(simulation.events), 1)
        self.assertEqual(counter_at_docking, simulation.mass[:8].sum())

    def test_small_patch_does_not_bind_remote_same_root_siblings(self):
        state = refined_leaf_state()
        bond = store([[.9, .05, .05], [.8, .15, .05], [.8, .05, .15]])
        prepared = tx.prepare_transaction(bond, leaves(), state)
        pairs = prepared['proposed_incidence'][0]['paired_cells']
        self.assertEqual({row['source_face_id'] for row in pairs}, {100})
        # A remote equal-root sibling changes owner without severing the bond.
        changed = deepcopy(state); changed['owner_uid'][3] = 52
        tx.prepare_transaction(bond, state, changed)

    def test_patch_interior_selects_child_missed_by_vertex_only_mapping(self):
        bond = store([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]])
        prepared = tx.prepare_transaction(bond, leaves(), refined_leaf_state())
        rows = prepared['proposed_incidence'][0]['paired_cells']
        self.assertEqual({r['source_face_id'] for r in rows}, {100, 101, 102, 103})
        self.assertEqual(sum(Fraction(*map(int, r['fraction_exact'])) for r in rows), 1)

    def test_partial_footprint_survives_exact_parent_return_and_checkpoint(self):
        bond = store([[.9, .05, .05], [.8, .15, .05], [.8, .05, .15]])
        proposed = tx.prepare_transaction(bond, leaves(), refined_leaf_state())
        arrays = {}; encoded = checkpoint._encode(proposed['store'], arrays)
        archive_bytes = BytesIO(); np.savez(archive_bytes, **arrays); archive_bytes.seek(0)
        with np.load(archive_bytes, allow_pickle=False) as archive:
            restored = checkpoint._decode(json.loads(json.dumps(encoded)), archive)
        backward = tx.prepare_transaction(restored, refined_leaf_state(), leaves())
        np.testing.assert_array_equal(backward['store']['bonds'][0]['source']['corners'],
                                      bond['bonds'][0]['source']['corners'])
        self.assertEqual(backward['proposed_incidence'], proposed['current_incidence'])

    def test_three_actual_refinement_levels_preserve_full_paired_domain(self):
        vertices = np.array([[1., 0., 0.], [1., .1, 0.], [1., 0., .1]])
        vertices /= np.linalg.norm(vertices, axis=1)[:, None]
        faces = np.array([[0, 1, 2]]); charts = np.eye(3)[None]
        for expected_count in (4, 16, 64):
            mapping = adaptive_material.refine(vertices, faces, np.full(len(faces), 41),
                np.ones(len(faces), int), desired_edge_km=1., face_ids=np.arange(len(faces))+100)
            charts = _charts(charts, mapping)
            vertices, faces = mapping['vertices'], mapping['faces']
            self.assertEqual(len(faces), expected_count)
            candidate = dict(face_id=np.r_[np.arange(len(faces))+1000, 20],
                root_id=np.r_[np.full(len(faces), 7), 8],
                reference_corners=np.r_[charts, np.eye(3)[None]], owner_uid=np.full(len(faces)+1, 41))
            answer = tx.prepare_transaction(store(), leaves(), candidate)
            rows = answer['proposed_incidence'][0]['paired_cells']
            self.assertEqual(len(rows), expected_count)
            self.assertEqual(sum(Fraction(*map(int, row['fraction_exact'])) for row in rows), 1)

    def test_paired_correspondence_not_independent_endpoint_coverage(self):
        state = refined_leaf_state()
        # Two valid individually covered endpoint footprints with a nontrivial
        # source/receiver correspondence must generate the actual joint cells.
        receiver_chart = state['reference_corners'][:4].copy()
        both = dict(face_id=np.r_[state['face_id'][:4], [200, 201, 202, 203]],
            root_id=np.array([7]*4+[8]*4), reference_corners=np.r_[receiver_chart, receiver_chart],
            owner_uid=np.full(8, 41))
        record = store(np.eye(3), np.eye(3)[[1, 2, 0]])
        answer = tx.prepare_transaction(record, leaves(), both)
        rows = answer['proposed_incidence'][0]['paired_cells']
        self.assertEqual(len(rows), 4)
        self.assertEqual({(r['source_face_id'], r['receiver_face_id']) for r in rows},
                         {(100, 201), (101, 202), (102, 200), (103, 203)})
        # A matched physical partition can carry both endpoints together;
        # mismatching one corresponding child is an unsupported finite cut.
        both['owner_uid'][0] = both['owner_uid'][5] = 52
        tx.prepare_transaction(record, leaves(), both)
        both['owner_uid'][5] = 41
        with self.assertRaisesRegex(ValueError, 'owner cut'):
            tx.prepare_transaction(record, leaves(), both)

    def test_owner_cut_rejected_before_any_input_changes(self):
        before = leaves(); after = deepcopy(before); after['owner_uid'][0] = 52
        bond = store(); snapshot = deepcopy((bond, before, after))
        with self.assertRaisesRegex(ValueError, 'owner cut'):
            tx.prepare_transaction(bond, before, after)
        self.assertEqual(tx._canonical((bond, before, after)), tx._canonical(snapshot))
        # Both endpoints moving together needs no invented bond release.
        after['owner_uid'][:] = 52
        result = tx.prepare_transaction(bond, before, after)
        self.assertEqual(result['proposed_incidence'][0]['paired_cells'][0]['owner_uid'], 52)

    def test_missing_and_duplicate_leaf_coverage_fail_without_area_floor(self):
        state = refined_leaf_state(); bond = store()
        missing = {key: value[[0, 1, 2, 4]].copy() for key, value in state.items()}
        with self.assertRaisesRegex(ValueError, 'missing'):
            tx.prepare_transaction(bond, leaves(), missing)
        duplicate = {key: np.concatenate((value, value[:1])) for key, value in state.items()}
        duplicate['face_id'][-1] = 999
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            tx.prepare_transaction(bond, leaves(), duplicate)
        tiny = store([[1.-2e-12, 1e-12, 1e-12], [1.-3e-12, 2e-12, 1e-12],
                      [1.-3e-12, 1e-12, 2e-12]])
        result = tx.prepare_transaction(tiny, leaves(), state)
        self.assertEqual(len(result['proposed_incidence'][0]['paired_cells']), 1)

    def test_restart_zero_time_and_stale_preflight_are_explicit(self):
        state = leaves(); bond = store()
        staged = tx.prepare_transaction(bond, state, state)
        self.assertEqual(tx._canonical(tx.validate_prepared(staged, bond, state, state)), tx._canonical(bond))
        staged['store']['bonds'][0]['creation_event_id'] += 1
        with self.assertRaisesRegex(ValueError, 'modified'):
            tx.validate_prepared(staged, bond, state, state)
        staged = tx.prepare_transaction(bond, state, state)
        changed = deepcopy(state); changed['owner_uid'][:] = 52
        with self.assertRaisesRegex(ValueError, 'stale'):
            tx.validate_prepared(staged, bond, state, changed)

    def test_no_bond_is_inferred_and_zero_area_or_legacy_schema_rejected(self):
        state = leaves(); empty = dict(version=1, scope=tx.SCOPE, bonds=[])
        changed = deepcopy(state); changed['owner_uid'][0] = 52
        self.assertEqual(tx.prepare_transaction(empty, state, changed)['proposed_incidence'], [])
        for invalid in ({}, dict(version=0, scope=tx.SCOPE, bonds=[]), store(np.ones((3, 3)))):
            with self.assertRaises(ValueError):
                tx.prepare_transaction(invalid, state, state)


if __name__ == '__main__':
    unittest.main()
