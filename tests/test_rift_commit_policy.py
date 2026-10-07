"""Rift commit policy 1: accretion elsewhere no longer vetoes a continental cut.

Local accretion runs between the loading update and the commit of each step.
Commit policy 0 (every saved world) refuses all cuts of the step when any
rift-mesh node or link anywhere changed. Policy 1 carries the pending solve to
the current mesh by persistent keys and defers only cuts whose own loading unit
changed. The integrated fixture is the loaded continent of
test_progressive_rifting, stopped on the step whose commit breaks through.
"""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import progressive_rifting as rifts
import rift_mesh
from tests.test_progressive_rifting import loaded_continent, load_step


def ready_state():
    """The loaded continent just before the commit that breaks it."""
    s = loaded_continent()
    for _ in range(200):
        load_step(s)
        trial = deepcopy(s)
        if rifts.commit(trial):
            return s
    raise AssertionError('The loaded continent never broke through.')


def outcome(s):
    return {key: getattr(s, key).copy() for key in ('plate', 'parcel_plate', 'omega', 'active', 'plate_uid')}


def with_extra_pending_node(s, uid, *, base=None):
    """Pending work also held an isolated node that has since left the mesh.

    Equivalent to a terrane of plate ``uid`` transferred away by accretion
    after the loading update: its (base, uid) key is gone from the current
    material mesh while every other key is unchanged.
    """
    pending = s.rift_pending
    mesh = dict(pending['mesh'])
    base = int(np.max(mesh['bases']))+1000 if base is None else base
    mesh['bases'] = np.r_[mesh['bases'], base]
    mesh['owner_uids'] = np.r_[mesh['owner_uids'], uid]
    pending['mesh'] = mesh
    pending['velocity'] = np.vstack((pending['velocity'], np.zeros(3)))


def shuffled_pending(s, seed=3):
    """Reorder pending nodes and links, keeping every keyed value."""
    rng = np.random.default_rng(seed)
    pending = s.rift_pending
    mesh = dict(pending['mesh'])
    count = len(mesh['bases'])
    order = rng.permutation(count)
    inverse = np.empty(count, np.int64)
    inverse[order] = np.arange(count)
    for key in ('bases', 'owner_uids'):
        mesh[key] = np.asarray(mesh[key])[order]
    pending['velocity'] = np.asarray(pending['velocity'])[order]
    edges = inverse[np.asarray(mesh['edges'])]
    links = rng.permutation(len(edges))
    # Flip some endpoint orders too: the link key sorts its base pair.
    edges = edges[links]
    edges[::2] = edges[::2, ::-1]
    mesh['edges'] = edges
    for key in ('damage', 'strain', 'extension_km', 'edge_extension'):
        if key in pending:
            pending[key] = np.asarray(pending[key])[links]
    pending['mesh'] = mesh


def realized_calibration(threshold):
    return dict(damage_strain_measure=rifts.REALIZED_STRAIN_MEASURE, rupture_strain_threshold=threshold)


class RiftCommitPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ready = ready_state()
        baseline = deepcopy(cls.ready)
        assert rifts.commit(baseline)
        cls.baseline = outcome(baseline)
        cls.cut_uid = int(cls.ready.plate_uid[0])
        cls.other_uid = int(cls.ready.plate_uid[1])

    def state(self, version):
        s = deepcopy(self.ready)
        if version:
            s.rift_commit_version = version
            # The fixture was loaded under policy 0; a policy-1 update also
            # records the material image of the mesh it loaded.
            s.rift_pending.update(rifts.material_image(s, s.rift_pending['mesh']))
        return s

    def cut_component_node(self, s, patches=2):
        """A cut-plate node inside the failing component with several patches."""
        pending = s.rift_pending
        mesh = pending['mesh']
        edges = np.asarray(mesh['edges'], np.int64)
        labels = rift_mesh._components(len(mesh['bases']), edges, np.asarray(mesh['owner_uids']))
        failing = labels[edges[np.argmax(pending['damage']), 0]]
        for node in np.flatnonzero((labels == failing) & (np.asarray(mesh['owner_uids']) == self.cut_uid)):
            ids = np.unique(s.parcel_patch[np.asarray(mesh['parcel_node']) == node])
            if len(ids) >= patches:
                return int(node), ids
        raise AssertionError('No multi-patch node in the failing component.')

    def assert_baseline(self, s):
        for key, value in outcome(s).items():
            np.testing.assert_array_equal(value, self.baseline[key], err_msg=key)

    def test_version_zero_still_refuses_every_cut_after_any_mesh_change(self):
        s = self.state(0)
        self.assertEqual(rifts.rift_commit_version(s), 0)
        before = deepcopy(s.rift_mechanics)
        with_extra_pending_node(s, self.other_uid)
        with patch.object(rifts.rift_mesh, 'coherent_cut', side_effect=AssertionError('policy 0 vetoes')):
            self.assertFalse(rifts.commit(s))
        # Saved worlds gain no commit diagnostics or frame keys.
        self.assertEqual(s.rift_mechanics, before)
        self.assertNotIn('rift_commit_version', rifts.snapshot(s))
        unchanged = self.state(0)
        self.assertTrue(rifts.commit(unchanged))
        self.assert_baseline(unchanged)

    def test_version_one_with_an_unchanged_mesh_is_version_zero(self):
        s = self.state(1)
        self.assertTrue(rifts.commit(s))
        self.assert_baseline(s)
        mechanics = s.rift_mechanics
        self.assertFalse(mechanics['commit_mesh_changed'])
        self.assertEqual(mechanics['commit_changed_plate_uids'], [])
        self.assertEqual(mechanics['commit_new_links'], 0)
        self.assertEqual(mechanics['commit_carried_links'], len(s.rift_pending['mesh']['edges']))
        self.assertEqual(mechanics['commit_reassigned_patches'], 0)
        self.assertEqual(mechanics['commit_deferred_cuts'], 0)

    def test_policy_one_work_without_its_material_image_is_refused_like_policy_zero(self):
        image_keys = ('material_patch_ids', 'material_patch_bases', 'material_patch_owner_uids')
        s = self.state(1)
        for key in image_keys:
            del s.rift_pending[key]
        with_extra_pending_node(s, self.other_uid)
        before = deepcopy(s.rift_mechanics)
        self.assertFalse(rifts.commit(s))
        self.assertEqual(s.rift_mechanics, before)
        unchanged = self.state(1)
        for key in image_keys:
            del unchanged.rift_pending[key]
        self.assertTrue(rifts.commit(unchanged))
        self.assert_baseline(unchanged)

    def test_a_real_partial_transfer_from_the_cut_plate_defers_its_cut(self):
        # Local accretion moves one patch of a multi-patch node of the cut
        # plate to another plate after loading. The donor keeps its (base,
        # uid) key with less material, so keys alone would miss it.
        threshold = rifts._rupture_calibration(self.ready)['rupture_strain_threshold']
        for realized in (False, True):
            with self.subTest(realized=realized):
                s = self.state(1)
                node, ids = self.cut_component_node(s)
                key = (int(s.rift_pending['mesh']['bases'][node]), self.cut_uid)
                s.parcel_plate[s.parcel_patch == ids[0]] = 1
                current = rifts.rift_material.refresh(s)
                self.assertIn(key, rifts._node_keys(current))
                self.assertEqual(rifts.changed_owner_uids(s.rift_pending['mesh'], current), [self.other_uid])
                calibration = rifts._rupture_calibration(self.ready)
                if realized:
                    s.rift_pending['damage_strain_measure'] = rifts.REALIZED_STRAIN_MEASURE
                    calibration = realized_calibration(threshold)
                with patch.object(rifts, '_rupture_calibration', return_value=calibration), \
                        patch.object(rifts, 'splits_protected_groups',
                                     side_effect=AssertionError('a deferred cut is not evaluated')):
                    self.assertFalse(rifts.commit(s))
                mechanics = s.rift_mechanics
                self.assertTrue(mechanics['commit_mesh_changed'])
                self.assertEqual(mechanics['commit_changed_plate_uids'], sorted((self.cut_uid, self.other_uid)))
                self.assertEqual(mechanics['commit_reassigned_patches'], 1)
                self.assertEqual(mechanics['commit_deferred_cuts'], 1)
                self.assertEqual(mechanics['commit_deferred_plate_uids'], [self.cut_uid])
                frame = rifts.snapshot(s)
                frame['rift_mechanics'].update(
                    rifts._commit_labels(s), damage_strain_measure=calibration['damage_strain_measure'],
                    commit_loading_unit=rifts.REALIZED_LOADING_UNIT if realized else rifts.PROXY_LOADING_UNIT)
                rifts.validate_frame(frame)
                # Policy 0 refuses on the receiver's new key.
                old = deepcopy(self.ready)
                old.parcel_plate[old.parcel_patch == ids[0]] = 1
                self.assertFalse(rifts.commit(old))

    def test_a_real_transfer_that_links_new_nodes_counts_new_links(self):
        # Two linked nodes of the cut plate each lose a patch to plate 1: the
        # receiver gains two nodes and the link between them, which has no
        # pending work and so cannot fail this step.
        s = self.state(1)
        mesh = s.rift_pending['mesh']
        parcel_node = np.asarray(mesh['parcel_node'])
        patches = {}
        for node in range(len(mesh['bases'])):
            ids = np.unique(s.parcel_patch[parcel_node == node])
            if len(ids) >= 2:
                patches[node] = ids
        pair = next((int(a), int(b)) for a, b in np.asarray(mesh['edges'])
                    if int(a) in patches and int(b) in patches)
        for node in pair:
            s.parcel_plate[s.parcel_patch == patches[node][0]] = 1
        self.assertFalse(rifts.commit(s))
        mechanics = s.rift_mechanics
        self.assertEqual(mechanics['commit_changed_plate_uids'], sorted((self.cut_uid, self.other_uid)))
        self.assertEqual(mechanics['commit_reassigned_patches'], 2)
        self.assertGreaterEqual(mechanics['commit_new_links'], 1)
        self.assertEqual(mechanics['commit_carried_links'], len(mesh['edges']))
        self.assertEqual(mechanics['commit_deferred_plate_uids'], [self.cut_uid])

    def test_accretion_on_another_plate_no_longer_vetoes_a_valid_cut(self):
        s = self.state(1)
        with_extra_pending_node(s, self.other_uid)
        self.assertTrue(rifts.commit(s))
        self.assert_baseline(s)
        mechanics = s.rift_mechanics
        self.assertTrue(mechanics['commit_mesh_changed'])
        self.assertEqual(mechanics['commit_changed_plate_uids'], [self.other_uid])
        self.assertEqual(mechanics['commit_new_links'], 0)
        self.assertEqual(mechanics['commit_deferred_cuts'], 0)
        self.assertEqual(mechanics['commit_deferred_plate_uids'], [])

    def test_accretion_on_the_plate_being_cut_defers_that_cut(self):
        # Legacy membrane loading is gathered plate-wide, so any node the cut
        # plate gained or lost defers its cut to the next solved step.
        s = self.state(1)
        before = outcome(s)
        with_extra_pending_node(s, self.cut_uid)
        with patch.object(rifts, 'splits_protected_groups',
                          side_effect=AssertionError('a deferred cut is not evaluated')):
            self.assertFalse(rifts.commit(s))
        for key, value in outcome(s).items():
            np.testing.assert_array_equal(value, before[key], err_msg=key)
        mechanics = s.rift_mechanics
        self.assertEqual(mechanics['commit_changed_plate_uids'], [self.cut_uid])
        self.assertEqual(mechanics['commit_deferred_cuts'], 1)
        self.assertEqual(mechanics['commit_deferred_plate_uids'], [self.cut_uid])

    def test_realized_links_defer_only_the_changed_component(self):
        # Realized links measure their own lengthening, so an island of the cut
        # plate that changed does not defer a cut of an untouched component.
        threshold = rifts._rupture_calibration(self.ready)['rupture_strain_threshold']
        for extra_uid in (self.cut_uid, self.other_uid):
            with self.subTest(extra_uid=extra_uid):
                s = self.state(1)
                s.rift_pending['damage_strain_measure'] = rifts.REALIZED_STRAIN_MEASURE
                with_extra_pending_node(s, extra_uid)
                with patch.object(rifts, '_rupture_calibration', return_value=realized_calibration(threshold)):
                    self.assertTrue(rifts.commit(s))
                self.assert_baseline(s)
                self.assertEqual(s.rift_mechanics['commit_changed_plate_uids'], [extra_uid])
                self.assertEqual(s.rift_mechanics['commit_deferred_cuts'], 0)
        # A change inside the cut component itself still defers it: one of its
        # links was never solved (its pending link is missing).
        s = self.state(1)
        s.rift_pending['damage_strain_measure'] = rifts.REALIZED_STRAIN_MEASURE
        pending = s.rift_pending
        mesh = dict(pending['mesh'])
        intact = np.flatnonzero(pending['damage'] < rifts.RUPTURE_DAMAGE)[0]
        keep = np.arange(len(mesh['edges'])) != intact
        mesh['edges'] = np.asarray(mesh['edges'])[keep]
        for key in ('damage', 'strain', 'extension_km', 'edge_extension'):
            pending[key] = np.asarray(pending[key])[keep]
        pending['mesh'] = mesh
        with patch.object(rifts, '_rupture_calibration', return_value=realized_calibration(threshold)):
            self.assertFalse(rifts.commit(s))
        self.assertEqual(s.rift_mechanics['commit_new_links'], 1)
        self.assertEqual(s.rift_mechanics['commit_deferred_plate_uids'], [self.cut_uid])

    def test_remapping_by_key_after_reordering_is_exact(self):
        s = self.state(1)
        shuffled_pending(s)
        self.assertFalse(np.array_equal(s.rift_pending['mesh']['bases'], self.ready.rift_pending['mesh']['bases']))
        self.assertTrue(rifts.commit(s))
        self.assert_baseline(s)
        # A pure reordering changes no key or material: nothing is recorded.
        self.assertFalse(s.rift_mechanics['commit_mesh_changed'])
        self.assertEqual(s.rift_mechanics['commit_changed_plate_uids'], [])
        self.assertEqual(s.rift_mechanics['commit_new_links'], 0)
        self.assertEqual(s.rift_mechanics['commit_carried_links'], len(s.rift_pending['mesh']['edges']))
        # Policy 0 cannot see past a pure reordering.
        old = self.state(0)
        shuffled_pending(old)
        self.assertFalse(rifts.commit(old))

    def test_remap_pending_maps_every_key_and_nothing_else(self):
        pending = dict(bases=np.array([10, 11, 12, 13, 40]), owner_uids=np.array([7, 7, 7, 7, 7]),
                       edges=np.array([[0, 1], [1, 2], [2, 3]]))
        # Reordered, one node moved to uid 8, one new juvenile node and link.
        current = dict(bases=np.array([13, 12, 11, 10, 40, 41]), owner_uids=np.array([7, 7, 7, 7, 8, 7]),
                       edges=np.array([[2, 3], [0, 1], [2, 1], [0, 5]]))
        node, edge = rifts.remap_pending(pending, current)
        np.testing.assert_array_equal(node, [3, 2, 1, 0, -1, -1])
        np.testing.assert_array_equal(edge, [0, 2, 1, -1])
        self.assertEqual(rifts.changed_owner_uids(pending, current), [7, 8])
        realized = rifts.unchanged_loading(pending, current, True)
        # Nodes 0-3 gained the link to new node 41, so their component changed.
        np.testing.assert_array_equal(realized, [False]*6)
        current['edges'] = current['edges'][:3]
        current['bases'], current['owner_uids'] = current['bases'][:5], current['owner_uids'][:5]
        # Only the island moved: the chain is untouched for realized links but
        # its plate lost a node for the plate-wide legacy membrane.
        np.testing.assert_array_equal(rifts.unchanged_loading(pending, current, True), [True]*4+[False])
        np.testing.assert_array_equal(rifts.unchanged_loading(pending, current, False), [False]*5)
        self.assertEqual(rifts.changed_owner_uids(pending, current), [7, 8])

    def test_partial_transfers_mark_both_nodes_through_the_material_image(self):
        # Base 12 held patches 100 and 101 on uid 7; patch 101 moved to uid 8,
        # which already held a node on base 40 and on base 12.
        mesh = dict(bases=np.array([10, 11, 12, 40, 12]), owner_uids=np.array([7, 7, 7, 8, 8]),
                    edges=np.array([[0, 1], [1, 2]]))
        pending = dict(material_patch_ids=np.array([98, 99, 100, 101, 200, 201]),
                       material_patch_bases=np.array([10, 11, 12, 12, 40, 12]),
                       material_patch_owner_uids=np.array([7, 7, 7, 7, 8, 8]))
        image = dict(material_patch_ids=np.array([98, 99, 100, 101, 200, 201, 300]),
                     material_patch_bases=np.array([10, 11, 12, 12, 40, 12, 41]),
                     material_patch_owner_uids=np.array([7, 7, 7, 8, 8, 8, 8]))
        keys, count = rifts.reassigned_material(pending, image)
        # New patch 300 is not a transfer between nodes.
        self.assertEqual((keys, count), ({(12, 7), (12, 8)}, 1))
        # Same key sets before and after: only the material image sees it.
        self.assertEqual(rifts.changed_owner_uids(mesh, mesh), [])
        self.assertEqual(rifts.changed_owner_uids(mesh, mesh, keys), [7, 8])
        np.testing.assert_array_equal(rifts.unchanged_loading(mesh, mesh, False, keys), [False]*5)
        # Realized: the donor's whole chain and the receiver's base-12 node
        # change; the receiver's untouched island on base 40 does not.
        np.testing.assert_array_equal(rifts.unchanged_loading(mesh, mesh, True, keys),
                                      [False, False, False, True, False])
        np.testing.assert_array_equal(rifts.unchanged_loading(mesh, mesh, True), [True]*5)
        self.assertEqual(rifts.reassigned_material(pending, pending), (set(), 0))

    def test_version_reader_rejects_bools_and_unknown_values(self):
        self.assertEqual(rifts.rift_commit_version(SimpleNamespace()), 0)
        self.assertEqual(rifts.rift_commit_version(SimpleNamespace(rift_commit_version=np.int64(1))), 1)
        for invalid in (True, np.bool_(True), 2, -1, 1., '1', None):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    rifts.rift_commit_version(SimpleNamespace(rift_commit_version=invalid))
                s = SimpleNamespace(rift_commit_version=invalid, rift_pending=None)
                with self.assertRaises(ValueError):
                    rifts.commit(s)

    def test_frame_labels_and_validation(self):
        s = self.state(1)
        s.rift_mechanics.update(rifts._commit_labels(s))
        frame = rifts.snapshot(s)
        self.assertEqual(frame['rift_commit_version'], 1)
        self.assertEqual(frame['rift_mechanics']['commit_loading_unit'], rifts.PROXY_LOADING_UNIT)
        rifts.validate_frame(frame)
        with_extra_pending_node(s, self.cut_uid)
        rifts.commit(s)
        good = rifts.snapshot(s)
        self.assertEqual(good['rift_mechanics']['commit_deferred_plate_uids'], [self.cut_uid])
        rifts.validate_frame(good)
        old = rifts.snapshot(self.state(0))
        self.assertNotIn('rift_commit_version', old)
        rifts.validate_frame(old)
        mechanics = good['rift_mechanics']
        bad = [dict(good, rift_commit_version=value) for value in (True, 2, 1., '1')]
        bad.append({key: value for key, value in good.items() if key != 'rift_commit_version'})
        bad.append(dict(good, rift_mechanics={k: v for k, v in mechanics.items() if k != 'commit_policy'}))
        bad.append(dict(good, rift_mechanics=dict(mechanics, commit_loading_unit=rifts.REALIZED_LOADING_UNIT)))
        bad.append(dict(good, rift_mechanics=dict(mechanics, commit_policy='other')))
        bad.append(dict(good, rift_mechanics=dict(mechanics, commit_unknown=1)))
        bad.append(dict(good, rift_mechanics={k: v for k, v in mechanics.items() if k != 'commit_new_links'}))
        bad.append(dict(good, rift_mechanics=dict(mechanics, commit_deferred_cuts=-1)))
        bad.append(dict(good, rift_mechanics=dict(mechanics, commit_deferred_cuts=True)))
        bad.append(dict(good, rift_mechanics=dict(mechanics, commit_deferred_cuts=0)))
        bad.append(dict(good, rift_mechanics={k: v for k, v in mechanics.items() if k != 'commit_reassigned_patches'}))
        bad.append(dict(good, rift_mechanics=dict(mechanics, commit_reassigned_patches=-1)))
        bad.append(dict(good, rift_mechanics=dict(mechanics, commit_changed_plate_uids=[], commit_deferred_cuts=0,
                                                  commit_deferred_plate_uids=[])))
        bad.append(dict(good, rift_mechanics=dict(mechanics, commit_mesh_changed=False, commit_changed_plate_uids=[],
                                                  commit_new_links=0, commit_deferred_cuts=0,
                                                  commit_deferred_plate_uids=[], commit_reassigned_patches=1)))
        bad.append(dict(good, rift_mechanics=dict(mechanics, commit_mesh_changed=1)))
        bad.append(dict(good, rift_mechanics=dict(mechanics, commit_mesh_changed=False)))
        bad.append(dict(good, rift_mechanics=dict(mechanics, commit_deferred_plate_uids=[self.other_uid])))
        bad.append(dict(good, rift_mechanics=dict(mechanics, commit_changed_plate_uids=(self.cut_uid,))))
        bad.append(dict(good, rift_mechanics=dict(mechanics, commit_changed_plate_uids=[self.cut_uid]*2)))
        bad.append(dict(old, rift_mechanics=dict(old['rift_mechanics'], commit_mesh_changed=False)))
        bad.append(dict(good, rift_mechanics=None))
        for index, frame in enumerate(bad):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    rifts.validate_frame(frame)


if __name__ == '__main__':
    unittest.main()
