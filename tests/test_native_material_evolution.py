"""Shared material transport keeps markers, charts, and columns consistent."""
from types import SimpleNamespace
from copy import deepcopy
import unittest
import numpy as np
import material_surface
import native_material_evolution as evolution
import crust_inventory
import dense_crust
import crustal_structure as columns
from crustal_structure import initialize_structure
from ridge_geometry import rotate


def fixture():
    vertices=np.array([[1., -.02, -.02], [1., .02, -.02], [1., .02, .02], [1., -.02, .02]])
    mesh=material_surface.initialize_surface(vertices, np.array([[0,1,2], [0,2,3]]),
        np.array([0,0]),np.array([2,1]),face_id=np.array([12,13]))
    points=material_surface.face_centres(mesh)
    return SimpleNamespace(material_surface=mesh,parcel_patch=mesh['face_id'].copy(),
        mass=mesh['reference_area_km2'].copy(),parcel_plate=np.array([0,0]),pos=points.copy(),
        # A protected anchor makes the neighbouring mobile face an actual
        # deforming margin; an entirely detached island should remain rigid.
        structure=initialize_structure([2,1],[220.,220.]),config=dict(deforming_regions=1,deformation_width_km=400.),
        omega=np.array([[0.,0.,.002],[0.,0.,-.002]]),bmid=np.array([[1.,0.,0.]]),
        bn=np.array([[0.,1.,0.]]),bl=np.array([1000.]),bp=np.array([0]),bq=np.array([1]),
        rift_tangent=np.tile([0.,0.,1.],(2,1)),trace_xyz=points.copy(),trace_patch=np.array([12,13]),
        trace_rift_tangent=np.tile([0.,0.,1.],(2,1)))


class EvolutionTests(unittest.TestCase):
    def test_markers_follow_same_nonrigid_map_as_face_corners(self):
        s=fixture(); old=deepcopy(s)
        evolution.advect(s,2.)
        np.testing.assert_allclose(s.trace_xyz,material_surface.face_centres(s.material_surface),atol=1e-14)
        np.testing.assert_allclose(s.trace_geometric_log_area,s.geometric_log_area,atol=0.)
        self.assertGreater(np.linalg.norm(s.trace_xyz-rotate(old.trace_xyz,old.omega[0]*2.)),1e-8)
        np.testing.assert_array_equal(s.material_lineage['root_id'],[12,13])
        np.testing.assert_array_equal(s.material_lineage['reference_corners'],np.tile(np.eye(3),(2,1,1)))

    def test_all_vertices_incident_to_a_craton_remain_exact_rigid(self):
        s=fixture();s.material_surface['face_kind'][:]=2
        original=s.material_surface['vertices'].copy()
        evolution.advect(s,2.)
        np.testing.assert_allclose(s.material_surface['vertices'],rotate(original,s.omega[0]*2.),atol=1e-15)
        np.testing.assert_array_equal(s.geometric_log_area,[0.,0.])

    def test_lineage_birth_and_reordering_preserve_existing_charts(self):
        s=fixture();evolution.ensure_lineage(s)
        s.material_lineage['reference_corners'][0]*=1.3
        s.parcel_patch=np.array([13,99,12]);evolution.ensure_lineage(s)
        np.testing.assert_array_equal(s.material_lineage['root_id'],[13,99,12])
        np.testing.assert_allclose(s.material_lineage['reference_corners'][2],1.3*np.eye(3))
        np.testing.assert_array_equal(s.material_lineage['reference_corners'][1],np.eye(3))

    def test_juvenile_gravity_reference_follows_exact_phase_adjusted_floor(self):
        s=fixture();s.native_arc_birth_profile_version=1;s.parcel_arc_id=np.array([1,0])
        s.structure['thickness_km'][0]=40.
        s.structure['reference_thickness_km'][0]=40.
        crust_inventory.initialize(s.structure)
        dense_crust.initialize(s.structure,[900.,900.])
        dense_crust.advance(s.structure,[1.,0.],[900.,900.],[10.,10.],1000.,
                            minimum_thickness_km=8.,detachment_fraction=0.)
        reference=evolution.gravitational_reference(s)
        self.assertAlmostEqual(float(reference[0]),float(columns.conserved_volume_floor(s.structure)[0]),places=13)
        self.assertGreater(abs(float(reference[0]-columns.fixed_area_floor(s.structure)[0])),1.)
        self.assertAlmostEqual(float(reference[1]),float(s.structure['reference_thickness_km'][1]),places=13)


if __name__ == '__main__':unittest.main()
