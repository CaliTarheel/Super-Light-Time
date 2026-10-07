"""Material attachment tests, including the previously observed ownership loop.

The miniature fixture exercises actual contact/planning/application operators.
It holds geometry fixed and authors mature directed exposure; it is not a
force-balanced trajectory or a calibrated geological docking time experiment.
"""
from copy import deepcopy
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import collision_contacts
import local_accretion as local
import localized_accretion as welding
import material_surface
import mesh_geometry
from tests.test_collision_contacts import rectangle


def three_body_fixture():
    """Two large hosts and one smaller separate terrane, initially on host A."""
    mesh = mesh_geometry.icosphere(2)
    surface = material_surface.initialize_surface(
        *rectangle(-200., 80., 0, width=180., nx=3, ny=3))
    material_surface.append_surface(surface,
        *rectangle(-80., 300., 1, width=180., nx=3, ny=3))
    material_surface.append_surface(surface,
        *rectangle(-600., -10., 0, width=180., nx=3, ny=3))
    m, n = len(surface['faces']), len(mesh['faces'])
    owner, kind = surface['face_owner'].copy(), surface['face_kind'].copy()
    pos, patches = material_surface.face_centres(surface), surface['face_id'].copy()
    locator = mesh_geometry.build_locator(mesh['vertices'], mesh['faces'])
    s = SimpleNamespace(native_mesh=mesh, material_surface=surface, n=n,
        xyz=mesh['xyz'].copy(), cell_area=mesh['area_km2'].copy(), pos=pos,
        parcel_patch=patches, parcel_plate=owner,
        mass=surface['reference_area_km2'].copy(), kind=kind,
        parcel_craton=np.full(m, -1, int), material_lineage={'root_id':patches.copy()},
        relief=np.full(m, 500.), suture=np.zeros(m),
        trace_patch=patches.copy(), trace_plate=owner.copy(), trace_xyz=pos.copy(),
        trace_kind=kind.copy(), trace_origin_kind=kind.copy(),
        trace_relief_m=np.full(m, 500.), trace_suture=np.zeros(m), trace_adjustment_m=np.zeros(m),
        structure={'thickness_km':np.full(m, 35.), 'thermal_history_sentinel':np.arange(m, dtype=float)},
        trace_structure={'thickness_km':np.full(m, 35.)},
        plate=np.full(n, 2), crust=np.zeros(n, np.uint8),
        _indices=lambda p:mesh_geometry.locate_points(p, locator)[0],
        active=np.ones(3, bool), plate_uid=np.array([41, 52, 63]), born=np.zeros(3), t=100.,
        omega=np.array([[0.,0.,.003], [0.,0.,-.003], [0.,0.,0.]]), mantle=np.zeros((3,3)),
        names=['Host A', 'Host B', 'Ocean'], events=[], ridge_episodes=[], backarc_basins=[],
        process_totals={'accreted_km2':0.}, ba=np.empty(0,int), bb=np.empty(0,int),
        bp=np.empty(0,int), bq=np.empty(0,int), bmid=np.empty((0,3)), bn=np.empty((0,3)),
        bcode=np.empty(0,np.uint8), bl=np.empty(0))
    s.support = np.zeros((3,n)); s.support[2] = 1.
    def record(self, kind, description, key=None, *, plates=(), xyz=None, details=None):
        self.events.append(dict(type=kind, time_myr=self.t, details=deepcopy(details or {})))
    s._record = MethodType(record, s)
    collision_contacts.refresh(s)
    hits = material_surface.sample_surface(surface, s.xyz)
    s.plate[hits['query_index']] = owner[hits['face_index']]
    s.crust[hits['query_index']] = kind[hits['face_index']]
    s.support[:] = 0.; s.support[s.plate,np.arange(n)] = 1.
    welding.initialize(s)
    # Real saved evidence has mature loading for this same material in both
    # host directions. Unrelated host material deliberately has no mature dose.
    keys = sorted((a,b,int(root)) for a,b in ((41,52),(52,41)) for root in patches[:8])
    state = s.accretion_welding_state
    for i,name in enumerate(welding.TABLE[:3]):
        state[name] = np.array([key[i] for key in keys], np.int64)
    state['loading'] = np.full(len(keys), 30.)
    state['active_weight'] = np.zeros(len(keys))
    state['geometry_signature'] = welding.geometry_signature(s)
    return s


def next_plans(s):
    s.t += 2.
    collision_contacts.refresh(s, 2.)
    return local.plan_accretions(s, 2.)


def apply_and_sync(s, plan):
    if not local.apply_accretion(s, plan):
        raise AssertionError('A freshly accepted docking plan was refused.')
    return material_surface.reassign_owners(s.material_surface, s.parcel_plate)


def preserved_material(s):
    names = ('mass','pos','parcel_patch','kind','relief','trace_xyz',
             'trace_patch','trace_kind','trace_relief_m')
    data = {name:np.asarray(getattr(s,name)).copy() for name in names}
    data.update({'surface_'+name:np.asarray(s.material_surface[name]).copy()
                 for name in ('vertices','faces','reference_area_km2','face_id')})
    data.update({'structure_'+name:np.asarray(value).copy() for name,value in s.structure.items()})
    data['material_roots'] = s.material_lineage['root_id'].copy()
    area = material_surface.spherical_face_areas(s.material_surface['vertices'],
        s.material_surface['faces'], s.material_surface['radius_km'])
    data['inventory'] = np.array([s.mass.sum(), area.sum(), area@s.structure['thickness_km']])
    return data


class TerraneLegacyCompatibilityTests(unittest.TestCase):
    def test_missing_policy_preserves_the_real_repeated_transfer_baseline(self):
        s = three_body_fixture()
        before = preserved_material(s)
        contact_ids = []
        for source,target in ((0,1),(1,0),(0,1),(1,0)):
            plans = next_plans(s)
            self.assertEqual(len(plans), 1)
            plan = plans[0]
            self.assertEqual((plan['source'],plan['target']), (source,target))
            np.testing.assert_array_equal(plan['parcel_indices'], np.arange(8))
            contact_ids.append(plan['contact_id'])
            apply_and_sync(s, plan)
            for name,value in preserved_material(s).items():
                np.testing.assert_array_equal(value, before[name], err_msg=name)
        self.assertEqual(len(set(contact_ids)), 4)
        self.assertEqual(s.process_totals['accreted_km2'], 4.*s.mass[:8].sum())
        self.assertEqual([e['type'] for e in s.events], ['accretion']*4)

    def test_cold_reverse_memory_blocks_immediate_return(self):
        s = three_body_fixture()
        state = s.accretion_welding_state
        state['loading'][state['source_uid']==52] = 0.
        plan, = next_plans(s)
        apply_and_sync(s, plan)
        self.assertEqual(next_plans(s), [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
