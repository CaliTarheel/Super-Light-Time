"""Finite trench endpoint clipping has conserved area and reciprocal work."""
import unittest

import numpy as np

import continental_entry
import entry_depth
import entry_regions
import finite_entry_arc as arc
import mesh_coverage
import material_surface
import phase_evolution
import dense_crust
from ridge_geometry import rotate
from tests.test_automatic_entry import approaching_face
from tests.test_entry_phase_depth import world as phase_world


def unit(points):
    points=np.asarray(points,float)
    return points/np.linalg.norm(points,axis=-1,keepdims=True)


def geometry():
    points=unit([[1.,-.01,-.015],[1.,.012,-.007],[1.,.015,.01]])
    normal=np.array([[0.,1.,0.]])
    midpoint=np.array([[1.,0.,0.]])
    return points,np.array([[0,1,2]]),normal,midpoint


def energy(points,normal,midpoint,half=50.):
    return continental_entry.evaluate(points,np.array([[0,1,2]]),np.array([1.]),
        np.array([2700e9]),normal,np.array([50.]),finite_midpoints=midpoint,
        finite_half_lengths_km=np.array([half]))


class FiniteEntryArcTests(unittest.TestCase):
    def test_entry_stack_ledger_conserves_area_when_upper_face_is_subdivided(self):
        points,_,normal,midpoint=geometry()
        points=rotate(points,[0.,0.,.03])
        area=float(material_surface.spherical_face_areas(points,np.array([[0,1,2]]))[0])
        halves=unit(points+np.roll(points,-1,axis=0))
        vertices=np.vstack((points,points,halves))
        faces=np.array([[0,1,2],[3,6,8],[6,4,7],[8,7,5],[6,7,8]])
        sheets=np.array([1,2,2,2,2])
        fraction=continental_entry.reference_entry_fractions(points[None],normal,
            radius_km=6371.,finite_midpoints=midpoint,
            finite_half_lengths_km=np.array([500.]))
        self.assertEqual(float(fraction[0]),1.)
        ledger=continental_entry.entry_stack_overlap_ledger(vertices,faces,sheets,
            np.array([0]),normal,fraction,radius_km=6371.,finite_midpoints=midpoint,
            finite_half_lengths_km=np.array([500.]))
        self.assertEqual({row['stack_face'] for row in ledger},{1,2,3,4})
        self.assertTrue(all(row['entered_face']==0 and row['area_km2']>0. for row in ledger))
        np.testing.assert_allclose(sum(row['area_km2'] for row in ledger),area,rtol=2e-12)
        ordered=continental_entry.ordered_entry_stack_depth_ledger(vertices,faces,sheets,
            np.array([0]),normal,np.array([50.]),fraction,{2:{1}},radius_km=6371.,
            finite_midpoints=midpoint,finite_half_lengths_km=np.array([500.]))
        self.assertTrue(all(row['entered_is_lower'] for row in ordered))
        np.testing.assert_allclose(sum(row['area_km2'] for row in ordered),area,rtol=2e-12)
        self.assertTrue(continental_entry.active_stack_overlap(vertices,faces,sheets,
            np.array([0]),normal,fraction,radius_km=6371.,finite_midpoints=midpoint,
            finite_half_lengths_km=np.array([500.])))

    def test_affine_clip_derivatives_and_reference_subdivision(self):
        q=np.array([-12.,8.,23.]);left=np.array([-.4,.3,.8]);right=np.array([.9,.3,-.2])
        mean,dq,dl,dr,fraction,_=arc.integrate(q,left,right)
        self.assertGreater(mean,0.);self.assertGreater(fraction,0.);self.assertLess(fraction,1.)
        for index,(field,derivative) in enumerate(((q,dq),(left,dl),(right,dr))):
            for corner in range(3):
                step=1e-6
                plus=[q.copy(),left.copy(),right.copy()]
                minus=[q.copy(),left.copy(),right.copy()]
                plus[index][corner]+=step;minus[index][corner]-=step
                predicted=(arc.integrate(*plus)[0]-arc.integrate(*minus)[0])/(2*step)
                np.testing.assert_allclose(predicted,derivative[corner],rtol=1e-8,atol=3e-9)
        self.assertAlmostEqual(float(dq.sum()),fraction,places=12)
        vertices=np.eye(3)
        mids=(vertices+np.roll(vertices,-1,axis=0))/2.
        children=np.array([[vertices[0],mids[0],mids[2]],
                           [mids[0],vertices[1],mids[1]],
                           [mids[2],mids[1],vertices[2]],
                           [mids[0],mids[1],mids[2]]])
        total=0.;part=0.
        for child in children:
            value,_,_,_,area,_=arc.integrate(child@q,child@left,child@right)
            total+=value/4.;part+=area/4.
        np.testing.assert_allclose(total,mean,rtol=2e-13,atol=1e-13)
        np.testing.assert_allclose(part,fraction,rtol=2e-13,atol=1e-13)

    def test_finite_entry_torque_matches_energy_derivative_and_rigid_symmetry(self):
        points,_,normal,midpoint=geometry()
        result=energy(points,normal,midpoint)
        self.assertGreater(result['energy_j'],0.)
        self.assertGreater(result['entered_reference_fraction'][0],0.)
        self.assertLess(result['entered_reference_fraction'][0],1.)
        scale=max(np.linalg.norm(result['entering_potential_torque_n_m'][0]),1.)
        self.assertLess(np.linalg.norm(result['entering_potential_torque_n_m'][0]
                            +result['hinge_potential_torque_n_m'][0])/scale,2e-13)
        for axis in np.eye(3):
            delta=2e-7
            incoming=(energy(rotate(points,axis*delta),normal,midpoint)['energy_j']
                     -energy(rotate(points,-axis*delta),normal,midpoint)['energy_j'])/(2*delta)
            upper=(energy(points,rotate(normal,axis*delta),rotate(midpoint,axis*delta))['energy_j']
                  -energy(points,rotate(normal,-axis*delta),rotate(midpoint,-axis*delta))['energy_j'])/(2*delta)
            np.testing.assert_allclose(incoming,axis@result['entering_potential_torque_n_m'][0],rtol=3e-7,atol=1e6)
            np.testing.assert_allclose(upper,axis@result['hinge_potential_torque_n_m'][0],rtol=3e-7,atol=1e6)

    def test_endpoint_sliver_is_continuous_and_outside_has_zero_force(self):
        points,_,normal,midpoint=geometry()
        fully_inside=energy(points,normal,midpoint,half=500.)
        finite=energy(points,normal,midpoint,half=50.)
        self.assertGreater(fully_inside['energy_j'],finite['energy_j'])
        outside=energy(rotate(points,[0.,.3,0.]),normal,midpoint)
        self.assertEqual(outside['energy_j'],0.)
        np.testing.assert_array_equal(outside['vertex_gradient_j'],0.)
        for offset in (0.,.0001,.0002):
            moved=rotate(points,[0.,offset,0.])
            value=energy(moved,normal,midpoint)['energy_j']
            self.assertGreaterEqual(value,0.)
        # A finite arc wider than the face reduces exactly to the unbounded law.
        original=continental_entry.evaluate(points,np.array([[0,1,2]]),np.array([1.]),
            np.array([2700e9]),normal,np.array([50.]))
        np.testing.assert_allclose(fully_inside['energy_j'],original['energy_j'],rtol=2e-14)

    def test_stack_outside_inactive_arc_has_no_entry_load_but_active_stack_fails(self):
        points,_,normal,midpoint=geometry()
        outside=rotate(points,[0.,.3,0.])
        vertices=np.vstack((outside,outside))
        faces=np.array([[0,1,2],[3,4,5]])
        potential=continental_entry.FrozenPotential(vertices,faces,np.array([1.,1.]),
            np.array([2700e9,2700e9]),np.array([1,2]),np.array([0]),normal,
            np.array([50.]),finite_midpoints=midpoint,finite_half_lengths_km=np.array([50.]))
        self.assertEqual(potential.evaluate(vertices,faces,potential.volumes,
            potential.sheets,radius=potential.radius)['energy_j'],0.)
        entered=np.vstack((points,points))
        with self.assertRaisesRegex(continental_entry.EntryGeometryError,
                                    'Active entry domain intersects'):
            potential.evaluate(entered,faces,potential.volumes,potential.sheets,
                               radius=potential.radius)

    def test_stack_on_inactive_part_of_entering_face_keeps_disjoint_loads(self):
        points,_,normal,midpoint=geometry()
        inactive=unit(np.array([[.95,.05,0.],[.8,.1,.1],[.95,0.,.05]])@points)
        vertices=np.vstack((points,inactive))
        faces=np.array([[0,1,2],[3,4,5]])
        overlap=mesh_coverage.material_overlaps(vertices,faces,np.array([1,2]))
        self.assertGreater(overlap['area_km2'].sum(),0.)
        potential=continental_entry.FrozenPotential(vertices,faces,np.array([1.,1.]),
            np.array([2700e9,2700e9]),np.array([1,2]),np.array([0]),normal,
            np.array([50.]),finite_midpoints=midpoint,finite_half_lengths_km=np.array([50.]))
        disjoint=potential.evaluate(vertices,faces,potential.volumes,potential.sheets,
                                    radius=potential.radius)
        self.assertEqual(continental_entry.entry_stack_overlap_ledger(vertices,faces,
            potential.sheets,potential.indices,potential.normals,
            disjoint['entered_reference_fraction'],radius_km=potential.radius,
            finite_midpoints=potential.finite_midpoints,
            finite_half_lengths_km=potential.finite_half_lengths_km),[])
        self.assertGreater(disjoint['energy_j'],0.)
        self.assertGreater(disjoint['entered_reference_fraction'][0],0.)
        rotated=continental_entry.FrozenPotential(rotate(vertices,[.3,-.2,.1]),faces,
            np.array([1.,1.]),np.array([2700e9,2700e9]),np.array([1,2]),
            np.array([0]),rotate(normal,[.3,-.2,.1]),np.array([50.]),
            finite_midpoints=rotate(midpoint,[.3,-.2,.1]),
            finite_half_lengths_km=np.array([50.]))
        np.testing.assert_allclose(rotated.evaluate(rotate(vertices,[.3,-.2,.1]),faces,
            rotated.volumes,rotated.sheets,radius=rotated.radius)['energy_j'],
            disjoint['energy_j'],rtol=2e-13)
        beyond_endpoint=unit(np.array([[.02,.02,.96],[.05,.02,.93],[.02,.05,.93]])@points)
        end_disjoint=potential.evaluate(np.vstack((points,beyond_endpoint)),faces,
                                        potential.volumes,potential.sheets,radius=potential.radius)
        self.assertEqual(end_disjoint['energy_j'],disjoint['energy_j'])
        active=unit(np.array([[.05,.9,.05],[.05,.7,.25],[.2,.7,.1]])@points)
        active_vertices=np.vstack((points,active))
        ledger=continental_entry.entry_stack_overlap_ledger(active_vertices,faces,
            potential.sheets,potential.indices,potential.normals,
            disjoint['entered_reference_fraction'],radius_km=potential.radius,
            finite_midpoints=potential.finite_midpoints,
            finite_half_lengths_km=potential.finite_half_lengths_km)
        self.assertGreater(sum(row['area_km2'] for row in ledger),0.)
        with self.assertRaisesRegex(continental_entry.EntryGeometryError,
                                    'Active entry domain intersects'):
            potential.evaluate(active_vertices,faces,potential.volumes,
                               potential.sheets,radius=potential.radius)

    def test_source_depth_partition_has_zero_atom_and_matching_first_moment(self):
        state=approaching_face();entry_regions.admit_approaching(state,.01)
        potential=entry_regions.frozen(state)
        points=rotate(state.material_surface['vertices'],[0.,0.,.03])
        result=potential.evaluate(points,state.material_surface['faces'],potential.volumes,
            potential.sheets,radius=potential.radius)
        normal=potential.normals[0];middle=potential.finite_midpoints[0]
        left,right,_,_=arc.endpoint_planes(normal,middle,potential.finite_half_lengths_km[0],potential.radius)
        triangle=points[state.material_surface['faces'][0]]
        q=result['corner_entry_coordinate_m'][0]
        _,_,_,_,fraction,pieces=arc.integrate(q,triangle@left,triangle@right)
        self.assertGreater(fraction,0.);self.assertLess(fraction,1.)
        mean=0.;weight=1.-fraction
        for area,bary in pieces:
            x,w,_=entry_depth.quadrature(bary@q,lambda z:z[:,None])
            mean+=area*float(w@x);weight+=area*float(w.sum())
        self.assertAlmostEqual(weight,1.,places=12)
        self.assertAlmostEqual(mean/result['mean_entry_depth_m'][0],
                               1./np.sin(np.deg2rad(50.)),places=12)

    def test_actual_phase_source_uses_endpoint_partition_and_conserves_column(self):
        state=phase_world()
        row=state.continental_entry_regions['regions'][0]
        triangle=state.material_surface['vertices'][state.material_surface['faces'][0]]
        normal=np.asarray(row['hinge_normal'])
        midpoint=triangle.sum(axis=0)
        midpoint-=normal*(midpoint@normal)
        midpoint/=np.linalg.norm(midpoint)
        row.update(finite_midpoint=midpoint,finite_half_length_km=1.)
        prepared=phase_evolution.prepare(state)
        partition=prepared['entry_depth']['finite_pieces'][0]
        self.assertGreater(partition['zero_fraction'],0.)
        self.assertLess(partition['zero_fraction'],1.)
        actual,report,_=phase_evolution.evolve(state,state.structure,.001,prepared=prepared)
        dense_crust.validate(actual)
        self.assertGreater(report['diagnostics']['entry_depth']['panels'],0)
        self.assertLessEqual(report['diagnostics']['entry_depth']['maximum_estimated_normalized_error'],1.)
        # The same clipped local distribution supplies mechanics and sources.
        potential=entry_regions.frozen(state)
        mechanical=potential.evaluate(state.material_surface['vertices'],
            state.material_surface['faces'],potential.volumes,potential.sheets,radius=potential.radius)
        source_mean=sum(area*float(corners.mean()) for area,corners in partition['triangles'])
        np.testing.assert_allclose(source_mean*1000.,mechanical['mean_entry_depth_m'][0],rtol=2e-13)
        marker=int(np.flatnonzero(state.trace_patch==state.parcel_patch[0])[0])
        state.trace_xyz[marker]=triangle[0]
        self.assertLess(np.min(prepared['entry_depth']['finite_planes'][0]@triangle[0]),0.)
        source=phase_evolution.evolve(state,state.trace_structure,.1,trace=True,prepared=prepared)[0]
        del state.entry_phase_depth
        ordinary=phase_evolution.evolve(state,state.trace_structure,.1,trace=True)[0]
        for key in source:
            np.testing.assert_array_equal(source[key][marker],ordinary[key][marker])


if __name__=='__main__':unittest.main()
