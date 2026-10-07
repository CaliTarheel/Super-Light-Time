"""Continental mechanics contacts are actual shared moving material edges."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np

import material_surface as surface
import mesh_geometry
import native_rift_material
import rift_material
import progressive_rifting
import structure_engine


def fixture(vertices, faces, owners=None, kinds=None, budget=4):
    count=len(faces)
    owners=np.zeros(count,np.int32) if owners is None else np.asarray(owners,np.int32)
    kinds=np.ones(count,np.uint8) if kinds is None else np.asarray(kinds,np.uint8)
    material=surface.initialize_surface(vertices,faces,owners,kinds)
    s=SimpleNamespace(material_surface=material,config={'mechanics_nodes':budget},
        plate_uid=np.arange(101,101+int(owners.max(initial=0))+1),
        pos=surface.face_centres(material),mass=material['reference_area_km2'].copy(),
        kind=material['face_kind'].copy(),parcel_plate=material['face_owner'].copy(),
        parcel_patch=material['face_id'].copy(),relief=np.zeros(len(material['faces'])),
        suture=np.full(len(material['faces']),.2),_indices=Mock(side_effect=AssertionError('No raster contact sampling')))
    structure_engine.initialize_parcels(s)
    s.trace_patch=s.parcel_patch.copy();s.trace_kind=s.kind.copy();s.trace_plate=s.parcel_plate.copy()
    return s


def triangles():
    points=np.array([[1.,-.03,-.03],[1.,.03,-.03],[1.,-.03,.03],[1.,.03,.03]])
    points/=np.linalg.norm(points,axis=1)[:,None]
    return points,np.array([[0,1,2],[1,3,2]])


class NativeRiftMaterialTests(unittest.TestCase):
    def test_exact_shared_edges_define_contact_and_point_touch_is_not_contact(self):
        faces=np.array([[0,1,2],[2,1,3],[3,4,5]])
        np.testing.assert_array_equal(native_rift_material.shared_face_edges(faces),[[0,1]])
        np.testing.assert_array_equal(native_rift_material.shared_face_edges(np.empty((0,3),int)),np.empty((0,2),int))

    def test_disconnected_coincident_material_is_never_grouped_from_overlap(self):
        points,faces=triangles()
        vertices=np.vstack((points[:3],points[:3]))
        s=fixture(vertices,np.array([[0,1,2],[3,4,5]]))
        result=rift_material.refresh(s)
        self.assertEqual(len(result['xyz']),2)
        self.assertEqual(len(result['edges']),0)
        self.assertNotEqual(*result['bases'])
        s._indices.assert_not_called()

    def test_shared_material_groups_follow_rotation_without_changing_membership(self):
        native=mesh_geometry.icosphere(1)
        s=fixture(native['vertices'],native['faces'],budget=32)
        before=rift_material.refresh(s)
        area=before['area'].sum();revision=before['topology_revision']
        surface.advect_surface(s.material_surface,np.array([[.01,-.02,.03]]),17.)
        s.pos=surface.face_centres(s.material_surface)
        after=rift_material.refresh(s)
        for key in ('bases','owners','owner_uids','parcel_node','trace_node','edges'):
            np.testing.assert_array_equal(before[key],after[key],err_msg=key)
        self.assertEqual(after['topology_revision'],revision)
        self.assertAlmostEqual(after['area'].sum(),area,places=7)
        self.assertFalse(np.allclose(before['xyz'],after['xyz']))
        s._indices.assert_not_called()

    def test_reassignment_separates_owner_sides_and_never_restitches_old_fault(self):
        points,faces=triangles()
        s=fixture(points,faces,budget=1024)
        s.plate_uid=np.array([101,102])
        before=rift_material.refresh(s)
        s.parcel_plate[1]=1;s.trace_plate[1]=1
        surface.reassign_owners(s.material_surface,s.parcel_plate)
        separate=rift_material.refresh(s)
        self.assertEqual(len(separate['edges']),0)
        self.assertEqual(set(separate['owner_uids']),{101,102})
        # Reuniting motion owners alone does not recreate deleted shared
        # vertices; an explicit geometrical weld would be required.
        s.parcel_plate[1]=0;s.trace_plate[1]=0
        surface.reassign_owners(s.material_surface,s.parcel_plate)
        reunited=rift_material.refresh(s)
        self.assertEqual(len(reunited['edges']),0)
        self.assertEqual(len(reunited['xyz']),2)
        np.testing.assert_array_equal(before['area'].sum(),reunited['area'].sum())

    def test_arc_becomes_eligible_only_after_accretion_and_uses_actual_contacts(self):
        points,faces=triangles()
        s=fixture(points,faces,kinds=[1,3],budget=1024)
        before=rift_material.refresh(s)
        self.assertEqual(before['parcel_node'][1],-1)
        s.kind[1]=s.trace_kind[1]=1;s.material_surface['face_kind'][1]=1
        after=rift_material.refresh(s)
        self.assertGreaterEqual(after['parcel_node'][1],0)
        self.assertEqual(len(after['edges']),1)
        self.assertAlmostEqual(after['area'].sum(),s.mass.sum(),places=8)

    def test_reference_area_and_column_means_are_preserved(self):
        native=mesh_geometry.icosphere(1)
        kinds=np.where(native['xyz'][:,2]>.5,2,1).astype(np.uint8)
        s=fixture(native['vertices'],native['faces'],kinds=kinds,budget=16)
        result=rift_material.refresh(s)
        self.assertAlmostEqual(result['area'].sum(),s.mass.sum(),places=6)
        expected=np.sum(s.mass*s.structure['thickness_km'])
        self.assertAlmostEqual(np.sum(result['area']*result['thickness']),expected,places=4)
        np.testing.assert_array_equal(result['parcel_node'],result['trace_node'])
        self.assertEqual(s.rift_material['backend'],'native_material_triangles')

    def test_native_component_check_uses_control_adjacency_without_map_dimensions(self):
        s=SimpleNamespace(native_mesh={'edge_faces':np.array([[0,1],[1,2],[2,3],[3,4],[4,5]])},cell_area=np.ones(6))
        self.assertAlmostEqual(progressive_rifting._connected_fraction(s,np.ones(6,bool),np.array([1,1,0,0,1,0],bool)),1/3)
        self.assertEqual(progressive_rifting._connected_fraction(s,np.ones(6,bool),np.array([1,1,1,0,0,0],bool)),0.)

    def test_misaligned_face_ids_are_rejected(self):
        points,faces=triangles();s=fixture(points,faces)
        s.parcel_patch=s.parcel_patch[::-1]
        with self.assertRaises(ValueError):rift_material.refresh(s)

    def test_empty_continental_surface_returns_empty_mechanics(self):
        points,faces=triangles();s=fixture(points,faces,kinds=[3,3])
        result=rift_material.refresh(s)
        self.assertEqual(result['xyz'].shape,(0,3))
        self.assertEqual(result['edges'].shape,(0,2))
        np.testing.assert_array_equal(result['parcel_node'],-1)


if __name__=='__main__':unittest.main()
