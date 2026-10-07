"""Engine mesh membership follows material instead of a geographic raster."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from checkpoint import read_checkpoint, write_checkpoint
from rift_material import refresh
from rift_mesh import assign_cells


def indices(points, width, height):
    lon = np.arctan2(points[:, 1], points[:, 0])
    lat = np.arcsin(np.clip(points[:, 2], -1, 1))
    col = np.floor((lon+np.pi)*width/(2*np.pi)).astype(int) % width
    row = np.clip(np.floor((np.pi/2-lat)*height/np.pi).astype(int), 0, height-1)
    return row*width+col


def fixture(width=96, height=48, cells=None, kinds=None):
    if cells is None:
        y, x = np.meshgrid(np.arange(19, 27), np.arange(37, 56), indexing='ij')
        cells = (y*width+x).ravel()
    cells = np.asarray(cells)
    row, col = np.divmod(cells, width)
    lon = (col+.5)*2*np.pi/width-np.pi
    lat = np.pi/2-(row+.5)*np.pi/height
    pos = np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))
    kind = np.ones(len(cells), np.uint8) if kinds is None else np.asarray(kinds, np.uint8)
    s = SimpleNamespace(w=width, h=height, config={'mechanics_nodes': 128}, t=0.,
                        pos=pos, mass=np.full(len(cells), 10_000.), kind=kind,
                        parcel_plate=np.zeros(len(cells), np.int32),
                        parcel_patch=np.arange(100, 100+len(cells), dtype=np.int64),
                        parcel_cell=cells.copy(), parcel_craton=np.full(len(cells), -1),
                        suture=np.linspace(.1, .4, len(cells)),
                        plate_uid=np.array([51, 70, 99]),
                        trace_patch=np.arange(100, 100+len(cells), dtype=np.int64),
                        trace_kind=kind.copy(), trace_plate=np.zeros(len(cells), np.int32),
                        rng=np.random.default_rng(3))
    s.structure = dict(thickness_km=np.linspace(28., 43., len(cells)),
                       reference_thickness_km=np.full(len(cells), 35.),
                       rift_heat_m=np.linspace(0., 500., len(cells)))
    return s


def turn(s):
    angle = 1.65
    matrix = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                       [-np.sin(angle), 0, np.cos(angle)]])
    s.pos = s.pos @ matrix.T
    s.parcel_cell = indices(s.pos, s.w, s.h)
    return matrix


class MaterialMeshAdapterTests(unittest.TestCase):
    def test_reference_membership_survives_polar_crossing_without_topology_rebuild(self):
        s = fixture()
        before = refresh(s)
        matrix = turn(s)
        after = refresh(s)
        np.testing.assert_array_equal(before['bases'], after['bases'])
        np.testing.assert_array_equal(before['parcel_node'], after['parcel_node'])
        np.testing.assert_array_equal(before['edges'], after['edges'])
        np.testing.assert_allclose(before['xyz'] @ matrix.T, after['xyz'], atol=3e-15)
        self.assertEqual(before['topology_revision'], after['topology_revision'])
        self.assertFalse(np.array_equal(assign_cells(before['xyz'], 128), assign_cells(after['xyz'], 128)))

    def test_whole_material_patches_share_one_node_and_weighted_properties(self):
        s = fixture()
        original = copy.deepcopy(s)
        for name in ('pos', 'mass', 'kind', 'parcel_plate', 'parcel_patch', 'parcel_cell', 'parcel_craton', 'suture'):
            values = getattr(s, name)
            setattr(s, name, np.concatenate([values]*4))
        s.mass *= np.tile([.1, .2, .3, .4], (len(original.mass), 1)).T.ravel()
        s.structure = {name: np.tile(value, 4) for name, value in s.structure.items()}
        nodes = refresh(s)
        for offset in (1, 2, 3):
            np.testing.assert_array_equal(nodes['parcel_node'][:len(original.mass)],
                                          nodes['parcel_node'][offset*len(original.mass):(offset+1)*len(original.mass)])
        self.assertAlmostEqual(nodes['area'].sum(), original.mass.sum())
        self.assertAlmostEqual(np.sum(nodes['thickness']*nodes['area']),
                               np.sum(original.structure['thickness_km']*original.mass))
        np.testing.assert_array_equal(nodes['trace_node'], nodes['parcel_node'][:len(original.mass)])

    def test_array_reordering_changes_no_material_identity_or_topology(self):
        s = fixture()
        before = refresh(s)
        order = np.random.default_rng(8).permutation(len(s.mass))
        for name in ('pos', 'mass', 'kind', 'parcel_plate', 'parcel_patch', 'parcel_cell', 'parcel_craton', 'suture'):
            setattr(s, name, getattr(s, name)[order])
        s.structure = {name: value[order] for name, value in s.structure.items()}
        after = refresh(s)
        for name in ('bases', 'edges', 'trace_node'):
            np.testing.assert_array_equal(before[name], after[name])
        for name in ('xyz', 'area', 'suture', 'thickness', 'heat'):
            np.testing.assert_allclose(before[name], after[name])
        np.testing.assert_array_equal(before['parcel_node'][order], after['parcel_node'])
        self.assertEqual(before['topology_revision'], after['topology_revision'])

    def test_disconnected_islands_inside_one_reference_cell_are_separate_nodes(self):
        width, height = 4096, 2048
        # Both pairs fit comfortably in one 128-site reference region. Two
        # intervening unoccupied columns represent a real narrow ocean gap.
        cells = np.array([1024*width+2047, 1024*width+2048,
                          1024*width+2052, 1024*width+2053])
        s = fixture(width, height, cells)
        self.assertEqual(len(np.unique(assign_cells(s.pos, 128))), 1)
        nodes = refresh(s)
        self.assertEqual(len(nodes['bases']), 2)
        self.assertEqual(len(nodes['edges']), 0)
        self.assertEqual(nodes['parcel_node'][0], nodes['parcel_node'][1])
        self.assertNotEqual(nodes['parcel_node'][1], nodes['parcel_node'][2])

    def test_topology_split_preserves_base_and_filters_cross_owner_links(self):
        s = fixture()
        before = refresh(s)
        parent_node = np.argmax(np.bincount(before['parcel_node']))
        selected = np.flatnonzero(before['parcel_node'] == parent_node)[::2]
        s.parcel_plate[selected] = 1
        s.trace_plate[selected] = 1
        after = refresh(s)
        for index in selected:
            self.assertEqual(before['bases'][before['parcel_node'][index]],
                             after['bases'][after['parcel_node'][index]])
        self.assertGreater(len(after['bases']), len(before['bases']))
        self.assertTrue(np.all(after['owners'][after['edges'][:, 0]] == after['owners'][after['edges'][:, 1]]))
        self.assertEqual(after['topology_revision'], before['topology_revision']+1)

    def test_slot_reuse_uses_uid_and_refreshes_current_identity(self):
        s = fixture()
        before = refresh(s)
        s.plate_uid[0] = 501
        after = refresh(s)
        np.testing.assert_array_equal(before['bases'], after['bases'])
        self.assertTrue(np.all(after['owner_uids'] == 501))
        self.assertEqual(after['topology_revision'], before['topology_revision']+1)

    def test_juvenile_material_enters_only_on_accretion(self):
        s = fixture()
        selected = np.arange(12)
        s.kind[selected] = s.trace_kind[selected] = 3
        before = refresh(s)
        self.assertTrue(np.all(before['parcel_node'][selected] == -1))
        self.assertTrue(np.all(before['trace_node'][selected] == -1))
        old_base = dict(zip(s.rift_material['patch_ids'], s.rift_material['patch_bases']))
        s.kind[selected] = s.trace_kind[selected] = 1
        after = refresh(s)
        self.assertTrue(np.all(after['parcel_node'][selected] >= 0))
        for patch, base in old_base.items():
            index = np.flatnonzero(s.parcel_patch == patch)[0]
            self.assertEqual(after['bases'][after['parcel_node'][index]], base)
        self.assertEqual(after['topology_revision'], before['topology_revision']+1)

    def test_arc_array_coalescence_does_not_disturb_existing_membership(self):
        s = fixture()
        s.kind[-8:] = s.trace_kind[-8:] = 3
        before = refresh(s)
        keep = np.arange(len(s.mass)-6)
        for name in ('pos', 'mass', 'kind', 'parcel_plate', 'parcel_patch', 'parcel_cell', 'parcel_craton', 'suture'):
            setattr(s, name, getattr(s, name)[keep])
        s.structure = {name: value[keep] for name, value in s.structure.items()}
        after = refresh(s)
        np.testing.assert_array_equal(before['parcel_node'][keep], after['parcel_node'])
        np.testing.assert_array_equal(before['trace_node'], after['trace_node'])
        self.assertEqual(before['topology_revision'], after['topology_revision'])

    def test_new_same_owner_contact_connects_accreted_blocks(self):
        s = fixture()
        s.parcel_plate[len(s.mass)//2:] = 1
        s.trace_plate[len(s.mass)//2:] = 1
        before = refresh(s)
        self.assertTrue(np.all(before['owners'][before['edges'][:, 0]] == before['owners'][before['edges'][:, 1]]))
        s.parcel_plate[:] = s.trace_plate[:] = 0
        after = refresh(s)
        left = set(before['bases'][before['owners'] == 0])
        right = set(before['bases'][before['owners'] == 1])
        self.assertTrue(any((int(a) in left and int(b) in right) or (int(b) in left and int(a) in right)
                            for a, b in after['edge_bases']))

    def test_no_contacts_over_wide_empty_ocean(self):
        width, height = 96, 48
        cells = np.r_[np.arange(20, 25)+20*width, np.arange(65, 70)+30*width]
        s = fixture(width, height, cells)
        nodes = refresh(s)
        left = set(nodes['parcel_node'][:5])
        right = set(nodes['parcel_node'][5:])
        self.assertFalse(any((a in left and b in right) or (b in left and a in right)
                             for a, b in nodes['edges']))

    def test_trace_owner_disagreement_stays_unassigned(self):
        s = fixture()
        s.trace_plate[:4] = 1
        nodes = refresh(s)
        self.assertTrue(np.all(nodes['trace_node'][:4] == -1))

    def test_state_is_exactly_checkpoint_safe(self):
        s = fixture()
        refresh(s)
        turn(s)
        expected = refresh(s)
        compatibility = dict(engine_sha256='fixture', auxiliary_sources_sha256={}, numpy_version=np.__version__)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.npz'
            write_checkpoint(path, s, {'config': s.config}, compatibility)
            restored, _ = read_checkpoint(path, compatibility, SimpleNamespace)
            actual = refresh(restored)
        for key in expected:
            np.testing.assert_array_equal(expected[key], actual[key], err_msg=key)

    def test_empty_continent_and_budget_change(self):
        s = fixture(cells=[])
        nodes = refresh(s)
        self.assertEqual(nodes['xyz'].shape, (0, 3))
        self.assertEqual(nodes['edges'].shape, (0, 2))
        s.config['mechanics_nodes'] = 512
        with self.assertRaises(ValueError):
            refresh(s)


if __name__ == '__main__':
    unittest.main()
