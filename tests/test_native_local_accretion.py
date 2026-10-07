"""Native terranes follow actual connected material, not projected occupancy."""
from copy import deepcopy
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import local_accretion as local
import material_surface as material
import mesh_geometry as geometry
from ridge_geometry import rotate


def neighborhood(mesh, seed, depth, excluded):
    result={int(seed)}
    for _ in range(depth):
        result |= {int(n) for cell in list(result) for n in mesh['face_neighbors'][cell]
                   if n >= 0 and int(n) not in excluded}
    return result-set(excluded)


def world(*, private=False):
    mesh=geometry.icosphere(2); count=len(mesh['faces'])
    first=0; remote=int(np.argmin(mesh['xyz']@mesh['xyz'][first]))
    source_a={first,int(mesh['face_neighbors'][first,0])}
    source_b={remote,int(mesh['face_neighbors'][remote,0])}
    source=source_a|source_b
    target_a=next(int(n) for n in mesh['face_neighbors'][first] if n not in source)
    target_b=next(int(n) for n in mesh['face_neighbors'][remote] if n not in source)
    receiver=neighborhood(mesh,target_a,2,source)|neighborhood(mesh,target_b,2,source)
    owners=np.full(count,2,np.int16); kinds=np.zeros(count,np.uint8)
    owners[list(source)]=0; owners[list(receiver)]=1
    kinds[list(source|receiver)]=1
    vertices,faces=mesh['vertices'],mesh['faces']
    if private:
        vertices=vertices[faces].reshape(-1,3)
        faces=np.arange(count*3).reshape(-1,3)
    surface=material.initialize_surface(vertices,faces,owners,kinds)
    pos=material.face_centres(surface); p=surface['face_id'].copy(); m=len(p)
    locator=geometry.build_locator(mesh['vertices'],mesh['faces'])
    def indices(points): return geometry.locate_points(points,locator)[0]
    s=SimpleNamespace(native_mesh=mesh,material_surface=surface,n=count,xyz=mesh['xyz'].copy(),
        cell_area=mesh['area_km2'].copy(),pos=pos,parcel_patch=p,
        parcel_plate=surface['face_owner'].copy(),mass=surface['reference_area_km2'].copy(),
        kind=surface['face_kind'].copy(),parcel_craton=np.full(m,-1,int),
        trace_patch=p.copy(),trace_plate=surface['face_owner'].copy(),trace_xyz=pos.copy(),
        trace_kind=surface['face_kind'].copy(),trace_origin_kind=surface['face_kind'].copy(),
        relief=np.full(m,500.),suture=np.full(m,.2),trace_relief_m=np.full(m,500.),
        trace_suture=np.full(m,.2),trace_adjustment_m=np.zeros(m),
        plate=owners.copy(),crust=kinds.copy(),_indices=indices,
        active=np.ones(3,bool),plate_uid=np.array([41,52,63]),born=np.zeros(3),t=100.,
        omega=np.zeros((3,3)),mantle=np.array([[1.,2.,3.],[4.,5.,6.],[0.,0.,0.]]),
        names=['Incoming','Receiver','Ocean'],events=[],ridge_episodes=[],backarc_basins=[],
        process_totals={'accreted_km2':0.},source_patches=(source_a,source_b),
        contact_pairs=((first,target_a),(remote,target_b)),
        structure={'thickness_km':np.full(m,35.)},trace_structure={'thickness_km':np.full(m,35.)})
    s.support=np.zeros((3,count)); s.support[s.plate,np.arange(count)]=1.
    def record(self,kind,description,key=None,*,plates=(),xyz=None,details=None):
        self.events.append(dict(type=kind,details=deepcopy(details or {})))
    s._record=MethodType(record,s)
    cache(s)
    contact(s,0)
    return s


def cache(s):
    hits=material.sample_surface(s.material_surface,s.xyz)
    s._material_occupancy_hits=hits
    s._owner_occupancy={int(p):np.unique(hits['query_index'][s.parcel_plate[hits['face_index']]==p])
                        for p in np.unique(s.parcel_plate)}
    s._owner_occupancy_signature=local.occupancy_signature(s)
    s.exposed_material=np.full(s.n,-1,int)
    s.exposed_material[hits['query_index']]=hits['face_index']


def contact(s, which):
    a,b=s.contact_pairs[which]
    s.ba=np.array([a]); s.bb=np.array([b]); s.bp=s.plate[s.ba]; s.bq=s.plate[s.bb]
    mid=s.xyz[a]+s.xyz[b]; mid/=np.linalg.norm(mid)
    normal=s.xyz[b]-s.xyz[a]; normal-=mid*np.dot(normal,mid); normal/=np.linalg.norm(normal)
    s.bmid=mid[None,:];s.bn=normal[None,:];s.bcode=np.array([4],np.uint8);s.bl=np.array([5000.])
    axis=np.cross(mid,normal)*20./6371.
    s.omega[0],s.omega[1]=axis,-axis


def mature(s):
    for _ in range(30):
        s.t+=2.
        plans=local.plan_accretions(s,2.)
        if plans:return plans[0]
    raise AssertionError('Sustained prescribed native contact did not mature.')


class NativeLocalAccretionTests(unittest.TestCase):
    def test_shared_material_edges_define_the_two_remote_terranes(self):
        s=world()
        with patch.object(local,'deposit',side_effect=AssertionError('Raster deposition is forbidden')):
            result=local._material_components(s,0)
        groups={frozenset(row['patch_ids']) for row in result['components'].values()}
        self.assertEqual(groups,{frozenset(p) for p in s.source_patches})
        for row in result['components'].values():
            np.testing.assert_array_equal(row['footprint_cells'],np.sort(row['patch_ids']))
            self.assertAlmostEqual(row['area_km2'],float(s.mass[row['parcel_indices']].sum()))

    def test_adjacent_projected_private_triangles_never_become_a_connected_terrane(self):
        shared,private=world(),world(private=True)
        np.testing.assert_array_equal(shared._owner_occupancy[0],private._owner_occupancy[0])
        result=local._material_components(private,0)
        self.assertEqual(len(result['components']),4)
        self.assertTrue(all(len(row['parcel_indices'])==1 for row in result['components'].values()))

    def test_explicit_craton_protection_keeps_disconnected_inherited_group_whole(self):
        s=world();s.parcel_craton[s.parcel_plate==0]=17
        result=local._material_components(s,0)
        self.assertEqual(len(result['components']),1)
        row=next(iter(result['components'].values()))
        np.testing.assert_array_equal(row['parcel_indices'],np.flatnonzero(s.parcel_plate==0))
        plan=mature(s)
        np.testing.assert_array_equal(plan['parcel_indices'],row['parcel_indices'])

    def test_control_cell_alias_or_overlap_does_not_merge_disconnected_material(self):
        s=world(); source_id=min(s.source_patches[0]); f=int(np.flatnonzero(s.parcel_patch==source_id)[0])
        triangle=s.material_surface['vertices'][s.material_surface['faces'][f]]
        material.append_surface(s.material_surface,triangle,np.array([[0,1,2]]),np.array([0]),np.array([3]),face_id=np.array([9001]))
        s.pos=material.face_centres(s.material_surface)
        s.mass=np.r_[s.mass,s.material_surface['reference_area_km2'][-1]]
        s.parcel_patch=np.r_[s.parcel_patch,9001];s.parcel_plate=np.r_[s.parcel_plate,0]
        s.parcel_craton=np.r_[s.parcel_craton,-1];s.kind=np.r_[s.kind,3]
        result=local._material_components(s,0)
        self.assertEqual(len(result['components']),3)
        overlapping=[row for row in result['components'].values() if source_id in row['footprint_cells']]
        self.assertEqual(len(overlapping),2)
        self.assertTrue(any(np.array_equal(row['patch_ids'],[9001]) for row in overlapping))

    def test_exact_fallback_does_not_fill_a_control_cell_from_a_nearby_patch_centre(self):
        s=world(private=True); source_id=min(s.source_patches[0]); f=int(np.flatnonzero(s.parcel_patch==source_id)[0])
        indices=s.material_surface['faces'][f]
        triangle=s.material_surface['vertices'][indices]
        center=.8*triangle[0]+.1*triangle[1]+.1*triangle[2];center/=np.linalg.norm(center)
        small=.98*center+.02*triangle;small/=np.linalg.norm(small,axis=1)[:,None]
        s.material_surface['vertices'][indices]=small;material.refresh_geometry(s.material_surface)
        s.pos=material.face_centres(s.material_surface)
        self.assertEqual(s._indices(s.pos[[f]])[0],source_id)
        with patch.object(local,'deposit',side_effect=AssertionError('No raster gap repair')):
            result=local._material_components(s,0)
        self.assertEqual(result['labels'][source_id],-1)
        row=next(row for row in result['components'].values() if source_id in row['patch_ids'])
        self.assertEqual(len(row['footprint_cells']),0)

    def test_native_exact_hit_cache_is_reused_then_invalidated_by_geometry_and_owner_contents(self):
        s=world()
        with patch.object(material,'sample_surface',side_effect=AssertionError('A current exact cache should be reused')):
            self.assertIsNotNone(local._material_components(s,0))
        old=s._owner_occupancy_signature
        s.material_surface['vertices']=rotate(s.material_surface['vertices'],[0.,.004,0.])
        self.assertNotEqual(local.occupancy_signature(s),old)
        with patch.object(material,'sample_surface',wraps=material.sample_surface) as query:
            local._material_components(s,0)
            self.assertEqual(query.call_count,1)
        old=local.occupancy_signature(s);s.parcel_plate[0]=2
        self.assertNotEqual(local.occupancy_signature(s),old)

    def test_rigid_motion_across_poles_keeps_material_components_and_area(self):
        s=world();before=local._material_components(s,0)
        expected={frozenset(row['patch_ids']):row['area_km2'] for row in before['components'].values()}
        material.advect_surface(s.material_surface,np.tile([0.,np.pi/4,0.],(3,1)),2.)
        s.pos=material.face_centres(s.material_surface)
        after=local._material_components(s,0)
        self.assertEqual({frozenset(row['patch_ids']):row['area_km2'] for row in after['components'].values()},expected)

    def test_local_native_contact_matures_without_borrowing_remote_clock_or_selecting_remote_material(self):
        s=world()
        for _ in range(4):s.t+=2.;self.assertEqual(local.plan_accretions(s,2.),[])
        first=s.local_accretion_contacts[0]['id']
        contact(s,1);s.t+=2.;self.assertEqual(local.plan_accretions(s,2.),[])
        self.assertEqual(len(s.local_accretion_contacts),2)
        self.assertNotEqual(s.local_accretion_contacts[-1]['id'],first)
        plan=mature(s)
        self.assertEqual(set(plan['patch_ids']),s.source_patches[1])
        self.assertEqual((plan['source'],plan['target']),(0,1))
        self.assertEqual(set(plan['surface_cells']),s.source_patches[1])

    def test_native_application_preserves_remote_material_geometry_and_all_column_state(self):
        s=world();plan=mature(s)
        remote=(s.parcel_plate==0)&~np.isin(np.arange(len(s.mass)),plan['parcel_indices'])
        arrays={key:getattr(s,key).copy() for key in ('pos','mass','parcel_patch','parcel_craton','parcel_plate','kind','relief')}
        surface=deepcopy(s.material_surface);columns=deepcopy(s.structure)
        self.assertTrue(local.apply_accretion(s,plan))
        for key in ('pos','mass','parcel_patch','parcel_craton'):
            np.testing.assert_array_equal(getattr(s,key),arrays[key])
        for key in ('parcel_plate','kind','relief'):
            np.testing.assert_array_equal(getattr(s,key)[remote],arrays[key][remote])
        np.testing.assert_array_equal(s.material_surface['vertices'],surface['vertices'])
        np.testing.assert_array_equal(s.material_surface['faces'],surface['faces'])
        np.testing.assert_array_equal(s.structure['thickness_km'],columns['thickness_km'])
        np.testing.assert_array_equal(s.parcel_plate[plan['parcel_indices']],1)
        self.assertFalse(local.apply_accretion(s,plan))
        self.assertEqual(s.process_totals['accreted_km2'],plan['area_km2'])
        # The engine performs this synchronization after applying the plan.
        material.reassign_owners(s.material_surface,s.parcel_plate)
        np.testing.assert_array_equal(s.material_surface['vertices'][s.material_surface['faces']],
                                       surface['vertices'][surface['faces']])


if __name__=='__main__':unittest.main()
