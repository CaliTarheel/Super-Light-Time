"""Independent column, spherical-gradient and source-capacity acceptance checks."""
from pathlib import Path
import sys
import unittest
from copy import deepcopy
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT)]
import arc_birth_profile as profile
import arc_cohort_footprint as cohorts
import arc_source_cohorts as history
import native_arc_material as arcs
import arc_surface
import material_surface
import material_reconstruction
import crust_inventory
import crustal_structure as columns
import dense_crust
from test_native_processes import ocean_fixture
from test_arc_source_geometry import add,positions_in_cell
from test_arc_source_cohorts import fixture as source_fixture,observe
from ridge_geometry import rotate


class ArcBirthProfileTests(unittest.TestCase):
    def test_actual_columns_determine_freeboard_and_exact_juvenile_volume(self):
        for amount in (100.,10000.,100000.):
            patch=arcs._patch([1.,0.,0.],[0.,1.,0.],amount)
            for basement in (-2000.,-5500.,-15000.):
                born=profile.birth(patch,amount,basement)
                thickness=born['thickness_km'];area=born['area_km2']
                air=basement*(3300.-1030.)/3300.+(3300.-2800.)/3300.*1000.*thickness
                expected=np.where(air<0,air*3300./(3300.-1030.),air)
                np.testing.assert_allclose(born['height_m'],expected,rtol=2e-15,atol=1e-10)
                self.assertAlmostEqual(float(area@thickness),25.*amount,delta=max(1e-7,25.*amount*1e-14))
                self.assertTrue(np.all((thickness>=8.)&(thickness<=75.)))
                self.assertAlmostEqual(born['diagnostics']['reference_self_GPE_created_km4'],
                    float(.5*area@((thickness-8.)**2)),places=6)
        self.assertLess(profile.birth(patch,amount,-15000.)['height_m'].max(),0.)

    def test_gradient_bound_matches_independent_dense_spherical_derivatives(self):
        patch=arcs._patch([1.,0.,0.],[0.,1.,0.],10000.)
        born=profile.birth(patch,10000.,-5500.)
        measured=profile.measure(patch['vertices'],patch['faces'],born['area_km2'],born['height_m'],-5500.)
        maximum=0.
        # The exact P1 quotient is checked by central tangent finite differences,
        # including the one-sided limiting values at all triangle vertices.
        bary=np.vstack((np.eye(3),np.ones((1,3))/3.,[[.001,.499,.5],[.2,.7,.1]]))
        for face in patch['faces']:
            triangle=patch['vertices'][face];h=measured['nodal_uplift_m'][face]
            points=arcs._unit(bary@triangle)
            for point in points:
                e=arcs._unit(np.cross(point,[0.,0.,1.]));n=np.cross(point,e)
                values=[]
                for tangent in (e,n):
                    eps=2e-8
                    nearby=np.array([arcs._unit(point+eps*tangent),arcs._unit(point-eps*tangent)])
                    weights=np.linalg.solve(triangle.T,nearby.T).T
                    weights/=weights.sum(axis=1)[:,None]
                    values.append(float(np.diff(weights@h)[0])/(-2*eps*6371000.))
                maximum=max(maximum,float(np.linalg.norm(values)))
        self.assertAlmostEqual(maximum,measured['maximum_grade'],delta=maximum*1e-7)

    def test_rotation_and_subdivision_preserve_reconstructed_profile_gradient(self):
        patch=arcs._patch([1.,0.,0.],[0.,1.,0.],10000.)
        born=profile.birth(patch,10000.,-5500.)
        measured=profile.measure(patch['vertices'],patch['faces'],born['area_km2'],born['height_m'],-5500.)
        vertices=patch['vertices'];faces=patch['faces'];heights=measured['nodal_uplift_m']
        angle=.73;rotation=np.array([[np.cos(angle),0.,np.sin(angle)],[0.,1.,0.],[-np.sin(angle),0.,np.cos(angle)]])
        np.testing.assert_allclose(profile.face_gradient_bounds(vertices@rotation,faces,heights),
                                   measured['face_grade'],rtol=1e-11,atol=1e-12)
        # Refinement samples the SAME radial-P1 field. The child basis must
        # converge to that physical field, rather than recreate a higher summit.
        all_vertices=[];all_faces=[];all_height=[]
        for face in faces:
            tri=vertices[face];h=heights[face]
            bary=np.vstack((np.eye(3),[[.5,.5,0.],[0.,.5,.5],[.5,0.,.5]]))
            new=arcs._unit(bary@tri)
            weights=np.linalg.solve(tri.T,new.T).T;weights/=weights.sum(axis=1)[:,None]
            base=len(all_vertices);all_vertices.extend(new);all_height.extend(weights@h)
            all_faces.extend(np.array([[0,3,5],[3,1,4],[5,4,2],[3,4,5]])+base)
        new_max=profile.face_gradient_bounds(np.asarray(all_vertices),np.asarray(all_faces),np.asarray(all_height)).max()
        self.assertAlmostEqual(new_max,measured['maximum_grade'],delta=measured['maximum_grade']*2e-5)

    def test_capacity_keeps_volume_pending_without_altering_geometry_or_display(self):
        s=ocean_fixture();s.native_arc_birth_profile_version=1;point=positions_in_cell(s)[:1]
        vertices=s.material_surface['vertices'].copy()
        report=add(s,point,[100.])
        self.assertEqual(report['added_area_km2'],0.)
        self.assertEqual(report['pending_magma_volume_km3'],2500.)
        np.testing.assert_array_equal(s.material_surface['vertices'],vertices)
        np.testing.assert_array_equal(s.native_arc_pending['xyz'],point)
        self.assertEqual(report['emplacement_geometry']['geometry_evaluations'],0)
        report=add(s,point,[9900.])
        self.assertEqual(len(s.mass),24)
        self.assertEqual(report['pending_area_km2'],0.)
        self.assertAlmostEqual(float(s.material_surface['area_km2']@s.structure['thickness_km']),250000.,places=6)
        np.testing.assert_allclose(s.structure['reference_elevation_m'],
            profile.freeboard(s.structure['thickness_km'],s.parcel_arc_basal_m),atol=1e-10)
        self.assertLessEqual(report['birth_profile_capacity']['maximum_accepted_birth_grade'],np.tan(np.deg2rad(20.)))

    def test_fixed_point_time_partitions_keep_source_and_actual_volume(self):
        for amounts in ([12000.],[6000.,6000.],[3000.]*4):
            s=ocean_fixture();s.native_arc_birth_profile_version=1;point=positions_in_cell(s)[:1]
            for amount in amounts:report=add(s,point,[amount])
            self.assertGreater(float(s.mass.sum()),0.)
            self.assertAlmostEqual(float(s.mass.sum())+report['pending_area_km2'],12000.,places=8)
            actual=float(s.material_surface['area_km2']@s.structure['thickness_km'])
            self.assertAlmostEqual(actual+report['pending_magma_volume_km3'],300000.,places=6)

    def test_legacy_profile_and_unmarked_world_are_not_reinterpreted(self):
        s=ocean_fixture();s.native_arc_birth_profile_version=0
        point=positions_in_cell(s)[:1];report=add(s,point,[100.])
        self.assertEqual(report['added_area_km2'],100.)
        self.assertNotIn('birth_profile_capacity',report)
        self.assertNotIn('arc_birth_profile_version',arcs.snapshot_fields(s))
        self.assertEqual(float(arc_surface.birth_profile([0.],-5500.)['height_m'][0]),970.)

    def test_growth_admits_phase_thinned_column_by_restored_capacity(self):
        vertices=arcs._unit(np.array([[1.,-.01,-.01],[1.,.01,-.01],[1.,0.,.01]]))
        mesh=material_surface.initialize_surface(vertices,np.array([[0,1,2]]),
            np.array([0]),np.array([3]),face_id=np.array([1]))
        state=columns.initialize_structure([3],[0.])
        state['thickness_km'][:]=8.01;state['reference_thickness_km'][:]=8.01
        crust_inventory.initialize(state);dense_crust.initialize(state,[900.])
        dense_crust.advance(state,[1.],[900.],[10.],2.,minimum_thickness_km=8.)
        self.assertLess(float(state['thickness_km'][0]),8.)
        old=float(mesh['area_km2'][0]);added=old*1e-4
        plan=dict(vertices=mesh['vertices'].copy(),vertex_ids=np.arange(len(mesh['vertices'])),
                  area_km2=np.array([old+added]),added_area_km2=np.array([added]))
        mixed=(old*state['thickness_km'][0]+added*profile.SOURCE_COLUMN_KM)/(old+added)
        self.assertLess(float(mixed),8.)
        s=type('World',(),{})();s.material_surface=mesh;s.structure=state
        s.mass=mesh['reference_area_km2'].copy();s.parcel_arc_basal_m=np.array([-5500.])
        report=profile.growth(s,np.array([0]),plan)
        self.assertTrue(report['admissible'],report)

    def test_profile_schema_rejects_versions_units_and_relaxed_capacity(self):
        s=ocean_fixture();s.native_arc_birth_profile_version=1;point=positions_in_cell(s)[:1]
        report=add(s,point,[10000.])
        frame=dict(arcs.snapshot_fields(s),arc_material_diagnostics=report,time_myr=2.)
        profile.validate_frame(frame)
        mutations=[lambda x:x.pop('arc_birth_profile_version'),
            lambda x:x.update(arc_birth_profile_version=True),lambda x:x.update(arc_birth_profile_version=1.),
            lambda x:x.update(arc_birth_profile_version=2),lambda x:x.pop('arc_birth_profile'),
            lambda x:x['arc_birth_profile'].update(version=1.),
            lambda x:x['arc_birth_profile'].update(source_column_km=20.),
            lambda x:x['arc_birth_profile'].update(mechanical_reference_column_km=25.),
            lambda x:x['arc_birth_profile'].update(maximum_constructive_slope_deg=80.),
            lambda x:x['arc_material_diagnostics']['birth_profile_capacity'].update(maximum_accepted_birth_grade=2.),
            lambda x:x['arc_material_diagnostics']['emplacement_geometry']['sources'][0]['profile_capacity'].update(maximum_allowed_grade=2.),
            lambda x:x['arc_material_diagnostics']['emplacement_geometry']['sources'][0]['profile_capacity'].update(admissible=False)]
        for mutate in mutations:
            changed=deepcopy(frame);mutate(changed)
            with self.subTest(mutation=mutate):
                with self.assertRaises(ValueError):profile.validate_frame(changed)

    def test_finely_partitioned_connected_sources_fund_one_containing_footprint(self):
        angles=np.linspace(-10.,10.,100)/6371.
        points=np.column_stack((np.cos(angles),np.sin(angles),np.zeros(len(angles))))
        area=np.full(100,100.)
        group=dict(indices=list(range(100)),anchor_index=49,links=[[i,i+1] for i in range(99)])
        factory=lambda anchor,amount:arcs._patch(points[anchor],[0.,1.,0.],amount)
        capacity=lambda patch,anchor,amount:profile.birth(patch,amount,-5500.)['diagnostics']['admissible']
        selected=cohorts.local_groups(points,area,group,factory,capacity)
        self.assertEqual(len(selected),1)
        self.assertEqual(len(selected[0]['indices']),100)
        take=selected[0]['indices'];patch=factory(selected[0]['anchor_index'],float(area[take].sum()))
        self.assertTrue(cohorts.contained(patch,points[take]).all())
        np.testing.assert_array_equal(patch['vertices'][0],points[49])

    def test_close_ends_of_distant_u_shaped_source_do_not_borrow_a_cohort(self):
        angles=np.array([0.,500.,5.])/6371.
        points=np.column_stack((np.cos(angles),np.sin(angles),np.zeros(3)))
        area=np.full(3,4000.)
        group=dict(indices=[0,1,2],anchor_index=0,links=[[0,1],[1,2]])
        factory=lambda anchor,amount:arcs._patch(points[anchor],[0.,1.,0.],amount)
        capacity=lambda patch,anchor,amount:profile.birth(patch,amount,-5500.)['diagnostics']['admissible']
        selected=cohorts.local_groups(points,area,group,factory,capacity)
        self.assertEqual(sorted(row['indices'] for row in selected),[[0],[1],[2]])

    def test_actual_source_history_and_front_subdivision_emit_conserved_material(self):
        totals=[]
        for n in (1,2,4,8):
            split=np.linspace(-10./6371.,10./6371.,n+1)
            observed,points=source_fixture(list(zip(split[:-1],split[1:])))
            s=ocean_fixture();s.native_arc_birth_profile_version=1
            # This is an actual finite connected trace, not a hand-labelled
            # shared trench ID. The producer determines its source links.
            observed.plate_uid[0]=s.plate_uid[0]
            observed.trench_systems[0].update(overriding_plate_uid=int(s.plate_uid[0]))
            amounts=np.full(n,10000./n)
            provenance=observe(observed,points,amounts)
            report=arcs.add_arc_crust(s,s._indices(points),amounts,positions=points,
                                     owners=np.zeros(n,int),source_provenance=provenance)
            self.assertEqual(len(s.mass),24)
            self.assertEqual(report['pending_area_km2'],0.)
            self.assertEqual(len(s.native_arc_source_placements),n)
            self.assertAlmostEqual(sum(r['area_km2'] for r in s.native_arc_source_placements),10000.,places=8)
            self.assertTrue(cohorts.contained(s.material_surface,points).all())
            totals.append([float(s.mass.sum()),float(s.material_surface['area_km2']@s.structure['thickness_km'])])
        np.testing.assert_allclose(totals,np.tile(totals[0],(len(totals),1)),rtol=1e-10,atol=1e-6)

    def test_actual_moving_vent_forms_finite_crust_for_both_time_partitions(self):
        outcomes=[]
        for dt in (1.,2.):
            s=ocean_fixture();s.native_arc_birth_profile_version=1
            observed,original_point=source_fixture(((-.0016,.0016),))
            observed.plate_uid[0]=s.plate_uid[0]
            observed.trench_systems[0].update(overriding_plate_uid=int(s.plate_uid[0]))
            original_trace=deepcopy(observed.native_boundary_geometry)
            s.omega[:]=observed.omega[0]
            supplied=0.;birth_time=None
            for time in np.arange(dt,20.+dt/2,dt):
                history.advect(s,dt);s.t=float(time);observed.t=float(time)
                if hasattr(s,'arc_source_cohort_state'):
                    observed.arc_source_cohort_state=s.arc_source_cohort_state
                for name in ('segments_start','segments_end'):
                    observed.native_boundary_geometry[name]=arcs._unit(rotate(original_trace[name],s.omega[0]*time))
                point=arcs._unit(rotate(original_point,s.omega[0]*time))
                rows=observe(observed,point,[500.*dt]);s.arc_source_cohort_state=observed.arc_source_cohort_state
                report=arcs.add_arc_crust(s,s._indices(point),[500.*dt],positions=point,
                                         owners=np.array([0]),source_provenance=rows)
                supplied+=500.*dt
                actual=float(s.material_surface['area_km2']@s.structure['thickness_km'])
                self.assertAlmostEqual(actual+report['pending_magma_volume_km3'],supplied*25.,places=6)
                if report['new_faces']:
                    birth_time=float(time)
                    self.assertTrue(cohorts.contained(s.material_surface,
                        np.asarray([r['geometry_xyz'][0] for r in s.native_arc_source_placements])).all())
                    break
            self.assertIsNotNone(birth_time,'The actual moving vent must produce physical crust, not perpetual pending.')
            self.assertEqual(len(s.mass),24)
            outcomes.append((birth_time,float(s.mass.sum()),actual))
        self.assertLessEqual(abs(outcomes[0][0]-outcomes[1][0]),2.)
        np.testing.assert_allclose(np.asarray(outcomes)[:,1:],np.tile(outcomes[0][1:],(2,1)),rtol=1e-10,atol=1e-6)

    def test_actual_temporal_u_front_cannot_fund_an_edifice_across_unrelated_arms(self):
        path=np.array([[-.028,.025],[-.028,0.],[-.028,-.06],[.028,-.06],[.028,0.],[.028,.025]])
        observed,_=source_fixture([(-.03,.03)]*5)
        s=ocean_fixture();s.native_arc_birth_profile_version=1
        observed.plate_uid[0]=s.plate_uid[0]
        observed.trench_systems[0].update(overriding_plate_uid=int(s.plate_uid[0]))
        vertices=arcs._unit(np.column_stack((np.ones(len(path)),path)))
        observed.native_boundary_geometry.update(segments_start=vertices[:-1],segments_end=vertices[1:])
        middle=(path[:-1]+path[1:])*.5
        middle+=np.array([[.03,0.],[.03,0.],[0.,.03],[-.03,0.],[-.03,0.]])
        points=arcs._unit(np.column_stack((np.ones(len(middle)),middle)))
        upper_origins=set()
        for time in (0.,2.):
            s.t=time;observed.t=time
            if hasattr(s,'arc_source_cohort_state'):observed.arc_source_cohort_state=s.arc_source_cohort_state
            rows=observe(observed,points,np.full(5,2000.));s.arc_source_cohort_state=observed.arc_source_cohort_state
            upper_origins.update(rows[i]['origin_id'] for i in (0,4))
            report=arcs.add_arc_crust(s,s._indices(points),np.full(5,2000.),positions=points,
                                     owners=np.zeros(5,int),source_provenance=rows)
        spent={row['source_provenance']['origin_id'] for row in s.native_arc_source_placements}
        self.assertFalse(upper_origins&spent,'The temporal source graph must not bridge opposite arms.')
        self.assertGreater(float(s.mass.sum()),0.,'The legitimate connected lower source region still forms crust.')
        self.assertAlmostEqual(float(s.mass.sum())+report['pending_area_km2'],20000.,places=7)
        self.assertAlmostEqual(float(s.material_surface['area_km2']@s.structure['thickness_km'])+
                               report['pending_magma_volume_km3'],500000.,places=5)


if __name__=='__main__':unittest.main(verbosity=2)
