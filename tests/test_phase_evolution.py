"""Hydrostatic, rheological and local-stack tests for retained phase forcing."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np
import crust_inventory as inventory
import dense_crust as phase
import phase_evolution as evolution
import collision_surface
import eclogite_sink
import crustal_structure as columns
import structure_engine
from deforming_regions import IncompleteContactStepError
from unittest.mock import patch
from tests.test_dense_crust import state, advance
from tests.test_eclogite_sink import stack


def world(**kwargs):
    s=stack(**kwargs)
    eclogite_sink.upgrade_inventory(s)
    for column in (s.structure,s.trace_structure):phase.initialize(column,1000.)
    s.retained_dense_crust_version=1
    s.retained_phase_parameters=evolution.parameters(dict(surface_temperature_c=1000.,
        mantle_temperature_c=1000.,geotherm_c_per_km=0.,supporting_yield_stress_pa=1e9))
    evolution.upgrade_floor(s)
    return s


class PhaseEvolutionTests(unittest.TestCase):
    def test_self_overlap_rejects_the_complete_coupled_timestep(self):
        s=world()
        # Two geometrically distinct upper faces are assigned to one physical
        # collision sheet and made to cover the same positive-area region.  The
        # phase model must not guess which copy supplies the overburden.
        s.parcel_collision_sheet[2]=s.parcel_collision_sheet[0]
        upper=np.array([0,2]);lower=np.array([1,1])
        weight=float(s.material_surface['area_km2'][1])*.1
        pairs=(upper,lower,np.full(2,weight),np.ones(2,int))
        region=(s.material_surface['vertices'][s.material_surface['faces'][1]],(0,1))
        def partition(triangles,face,selected,upper_faces,face_area,radius):
            if face == 1:
                np.testing.assert_array_equal(selected,[0,1])
                return [region],np.array([weight]),0.
            self.assertEqual(len(selected),0)
            return [(triangles[face],tuple())],np.array([face_area]),0.
        with patch.object(eclogite_sink,'_depth_pairs',return_value=pairs), \
             patch.object(evolution.burial_depth,'partition_face',side_effect=partition):
            with self.assertRaisesRegex(IncompleteContactStepError,'complete coupled timestep'):
                evolution.prepare(s)

    def test_floor_policy_migration_is_explicit_atomic_and_idempotent(self):
        s=world()
        del s.retained_phase_floor_version
        del s.retained_phase_floor_migration
        s.t=56.
        s.rng=np.random.default_rng(17)
        before_keys=set(vars(s))
        before_structure=deepcopy(s.structure)
        before_trace=deepcopy(s.trace_structure)
        before_relief=s.relief.copy()
        before_trace_relief=s.trace_relief_m.copy()
        before_rng=deepcopy(s.rng.bit_generator.state)
        zeros=np.zeros(s.n)
        with self.assertRaisesRegex(ValueError,'upgrade_floor'):
            evolution.prepare(s)
        with self.assertRaisesRegex(ValueError,'upgrade_floor'):
            evolution.evolve(s,s.structure,.25)
        with self.assertRaisesRegex(ValueError,'upgrade_floor'):
            structure_engine.deform(s,zeros,zeros,zeros,.25)
        self.assertEqual(set(vars(s)),before_keys)
        for current,old in ((s.structure,before_structure),(s.trace_structure,before_trace)):
            for key in old:np.testing.assert_array_equal(current[key],old[key])
        np.testing.assert_array_equal(s.relief,before_relief)
        np.testing.assert_array_equal(s.trace_relief_m,before_trace_relief)
        self.assertEqual(s.rng.bit_generator.state,before_rng)

        report=evolution.upgrade_floor(s)
        self.assertEqual(report['from_version'],1)
        self.assertEqual(report['to_version'],evolution.FLOOR_VERSION)
        self.assertEqual(report['time_myr'],56.)
        self.assertFalse(report['physical_state_changed'])
        self.assertEqual(evolution.upgrade_floor(s),report)
        self.assertEqual(set(vars(s))-before_keys,
                         {'retained_phase_floor_version','retained_phase_floor_migration'})
        for current,old in ((s.structure,before_structure),(s.trace_structure,before_trace)):
            for key in old:np.testing.assert_array_equal(current[key],old[key])
        self.assertEqual(s.rng.bit_generator.state,before_rng)

        for marker in (True,0,3):
            invalid=deepcopy(s)
            invalid.retained_phase_floor_version=marker
            with self.assertRaises(ValueError):evolution.floor_version(invalid)
        future=deepcopy(s)
        future.retained_phase_floor_migration['time_myr']=57.
        with self.assertRaisesRegex(ValueError,'after the current world time'):
            evolution.floor_version(future)
        with self.assertRaises(ValueError):
            evolution.floor_version(SimpleNamespace(retained_dense_crust_version=0,
                retained_phase_floor_version=evolution.FLOOR_VERSION))

    def test_negative_dense_support_survives_saved_sampling_and_foreland_input(self):
        import native_frame_sampling,foreland_loading
        from tests.test_collision_surface_acceptance import saved
        s=world(cover=1.)
        s.foundering_depth_version=1
        advance(s.structure,[0.,1.,0.,0.],1000.,10.,100.,detachment_fraction=0.)
        # Expose the dense base by eroding ordinary upper material first.
        removed=np.zeros(s.n)
        removed[1]=s.structure['thickness_km'][1]-s.structure[phase.DENSE][1]
        inventory.erode(s.structure,removed)
        s.structure['thickness_km']-=removed
        phase.validate(s.structure)
        collision_surface.refresh(s)
        self.assertLess(s.parcel_collision_support_m[0],0.)
        frame=saved(s)
        frame.update(eclogite_sink.snapshot_fields(s))
        prepared=native_frame_sampling.prepare(frame)
        np.testing.assert_array_equal(prepared['material_collision_support_m'],s.parcel_collision_support_m)
        result=native_frame_sampling.sample_frame(frame,s.pos,prepared=prepared)
        self.assertTrue(np.isfinite(result['elevation']).all())
        for row in s.collision_contacts:row.update(top_owner=0,under_owner=1)
        foreland_loading.prepare_native(s)

    def test_first_contact_order_is_staged_and_rejected_geometry_leaves_no_record(self):
        import column_density,gravitational_relaxation,material_surface,collision_contacts
        from ridge_geometry import rotate
        s=world()
        touching=s.material_surface['vertices'].copy()
        s.material_surface['vertices'][:3]=rotate(touching[:3],np.array([0.,0.,.4]))
        material_surface.refresh_geometry(s.material_surface)
        s.collision_contacts=[];s.next_collision_contact_id=1
        volumes=s.material_surface['area_km2']*s.structure['thickness_km']
        admission=column_density.ContactAdmission(s,volumes)
        candidate=admission.prepare(touching)
        self.assertEqual(len(candidate[0].collision_contacts),1)
        self.assertEqual(s.collision_contacts,[])
        self.assertEqual(admission.profile['sheet_order'],{})
        # Discarding that candidate must leave even a subsequent non-contact
        # candidate empty; acceptance is a separate transaction.
        self.assertEqual(admission.prepare(s.material_surface['vertices'])[0].collision_contacts,[])
        admission.accept(candidate)
        points,report=gravitational_relaxation.relax(touching,s.material_surface['faces'],
            volumes,s.structure['reference_thickness_km'],s.parcel_collision_sheet,.1,
            rigid_mask=np.ones(len(touching),bool),minimum_area_km2=s.mass*.5,
            maximum_area_km2=s.mass*2.,density_profile=admission.profile,density_admission=admission)
        self.assertEqual(s.collision_contacts,[])
        admission.publish(s)
        self.assertEqual(len(s.collision_contacts),1)
        order=deepcopy(s.collision_contacts[0])
        s.material_surface['vertices']=points
        material_surface.refresh_geometry(s.material_surface)
        s.omega=np.zeros((len(s.plate_uid),3))
        collision_contacts.refresh(s,0.)
        self.assertEqual(s.collision_contacts[0]['top_sheet'],order['top_sheet'])
        self.assertEqual(s.collision_contacts[0]['under_sheet'],order['under_sheet'])
        self.assertEqual(report['accepted_time_fraction'],1.)

    def test_first_contact_score_includes_water_loading_after_strain(self):
        import column_density,material_surface
        s=world()
        s.collision_contacts=[];s.next_collision_contact_id=1
        mesh=s.material_surface
        trial=mesh['vertices'].copy()
        trial[:3,1:]*=1.8
        trial[:3]/=np.linalg.norm(trial[:3],axis=1,keepdims=True)
        old_area=mesh['area_km2']
        new_area=material_surface.spherical_face_areas(trial,mesh['faces'])
        projected=deepcopy(s.structure)
        projected['thickness_km']*=old_area/new_area
        projected['area_factor']*=new_area/old_area
        expected=columns.elevation(projected)
        self.assertLess(expected[0],0.)
        admission=column_density.ContactAdmission(s,old_area*s.structure['thickness_km'])
        candidate=admission.prepare(trial)
        scores=candidate[0].collision_contacts[0]['polarity_height_scores_m']
        self.assertAlmostEqual(scores['1'],expected[0],places=9)

    def test_native_step_checkpoint_and_saved_frame_preserve_phase_continuation(self):
        from pathlib import Path
        import tempfile,checkpoint,mesh_history
        from native_engine import Simulation
        from tests.test_native_engine import world as native_world
        s=native_world(erosion=0.)
        s.plate_balance_version=1
        evolution.upgrade(s,1000.,constitutive_parameters=dict(reaction_pressure_pa=5e8,
            surface_temperature_c=1000.,mantle_temperature_c=1000.,geotherm_c_per_km=0.))
        s.step(.25)
        phase.validate(s.structure)
        self.assertGreater(float(s.mass@s.structure[phase.DENSE]),0.)
        budget=s.material_column_budget
        self.assertLess(abs(budget['phase_mass_residual_kg'])/budget['after_columns_mass_kg'],1e-12)
        self.assertLess(abs(budget['residual_km3'])/budget['after_columns_volume_km3'],1e-12)
        mesh_history.arrays(s.snapshot())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.npz'
            checkpoint.write_checkpoint(path,s,dict(config=s.config),{})
            resumed,_=checkpoint.read_checkpoint(path,None,Simulation)
        s.step(.25);resumed.step(.25)
        for key in s.structure:np.testing.assert_array_equal(s.structure[key],resumed.structure[key])
        np.testing.assert_array_equal(s.omega,resumed.omega)
        mesh_history.arrays(resumed.snapshot())

    def test_full_structure_step_conserves_mass_tracks_returns_and_local_marker_history(self):
        s=world(erosion=0.)
        initial_mass=float(s.mass@phase.mass_volume(s.structure))
        initial_height=s.trace_relief_m.copy()
        initial_volume=float(s.mass@(s.structure['thickness_km']*s.structure['area_factor']))
        for step in range(8):
            s.t+=2.
            if step==3:s.retained_phase_parameters['supporting_yield_stress_pa']=0.
            structure_engine.deform(s,np.zeros(s.n),np.zeros(s.n),np.zeros(s.n),2.)
            for state in (s.structure,s.trace_structure):phase.validate(state)
            actual=float(s.mass@(phase.mass_volume(s.structure)+s.structure[inventory.RETURNED]))
            self.assertAlmostEqual(actual/initial_mass,1.,places=12)
        dense_return=float(s.mass@s.structure[phase.RETURNED])
        self.assertGreater(dense_return,0.)
        self.assertAlmostEqual(s.mantle_return_km3['total']/dense_return,1.,places=12)
        contraction=float(s.mass@s.structure[phase.CONVERTED])*(1./phase.RATIO-1.)
        volume=float(s.mass@(s.structure['thickness_km']*s.structure['area_factor']))
        self.assertAlmostEqual((volume+dense_return+contraction)/initial_volume,1.,places=12)
        np.testing.assert_allclose(s.trace_relief_m-initial_height,
            s.trace_uplift_m-s.trace_extension_m-s.trace_erosion_m+s.trace_adjustment_m,atol=2e-10)
        np.testing.assert_array_equal(s.trace_erosion_m,0.)
        # The lower marker is under the covering sheet; its face also contains
        # uncovered regions. Local thermodynamics must distinguish these.
        self.assertGreater(s.trace_structure[phase.CONVERTED][1],s.structure[phase.CONVERTED][1])

    def test_explicit_migration_is_atomic_and_idempotent(self):
        s=stack()
        s.plate_balance_version=s.material_mechanics_version=1
        before=deepcopy(s.structure)
        with self.assertRaises(ValueError):evolution.upgrade(s,np.nan)
        self.assertFalse(hasattr(s,'retained_dense_crust_version'))
        for key in before:np.testing.assert_array_equal(before[key],s.structure[key])
        report=evolution.upgrade(s,[900.,1000.,1100.,500.])
        self.assertEqual(evolution.upgrade(s,0.),report)
        self.assertEqual(s.retained_phase_floor_version,evolution.FLOOR_VERSION)
        np.testing.assert_allclose(phase.temperature(s.structure),[900.,1000.,1100.,500.])
        np.testing.assert_array_equal(phase.mass_volume(s.structure),before['thickness_km']*before['area_factor'])
        np.testing.assert_array_equal(s.structure[phase.DENSE],0.)
        self.assertEqual(s.foundering_depth_version,1)

    def test_frame_round_trip_and_validation_include_phase_and_heat(self):
        s=stack();s.plate_balance_version=s.material_mechanics_version=1
        evolution.upgrade(s,1000.)
        structure_engine.deform(s,np.zeros(s.n),np.zeros(s.n),np.zeros(s.n),2.)
        frame=dict(material_faces=s.material_surface['faces'],**eclogite_sink.snapshot_fields(s))
        eclogite_sink.validate_frame(frame)
        for field in (phase.DENSE,phase.HEAT,inventory.REMAINING):
            np.testing.assert_array_equal(frame['material_'+field],s.structure[field])
        broken=deepcopy(frame);broken['material_'+phase.HEAT][0]+=1.
        with self.assertRaises(ValueError):eclogite_sink.validate_frame(broken)
        broken=deepcopy(frame);del broken['retained_dense_crust_version']
        with self.assertRaises(ValueError):eclogite_sink.validate_frame(broken)
        broken=deepcopy(frame);broken['material_crust_temperature_c']+=1.
        with self.assertRaises(ValueError):eclogite_sink.validate_frame(broken)
        broken=deepcopy(frame);broken['retained_phase_floor_version']=True
        with self.assertRaises(ValueError):eclogite_sink.validate_frame(broken)
        legacy=deepcopy(frame)
        del legacy['retained_phase_floor_version']
        del legacy['retained_phase_floor_migration']
        eclogite_sink.validate_frame(legacy)
        legacy['retained_phase_floor_version']=1
        eclogite_sink.validate_frame(legacy)
        broken=deepcopy(frame);del broken['retained_phase_floor_migration']
        with self.assertRaises(ValueError):eclogite_sink.validate_frame(broken)
        frame.update(material_actual_area_km2=s.material_surface['area_km2'],material_reference_area_km2=s.mass)
        eclogite_sink.validate_frame(frame)
        broken=deepcopy(frame);broken['material_actual_area_km2']*=1.01
        with self.assertRaises(ValueError):eclogite_sink.validate_frame(broken)

    def test_pressure_threshold_is_hydrostatic_and_excludes_dense_base_from_new_reaction(self):
        s=state(thickness=40.,temperature=1000.)
        controls=evolution.parameters()
        pressure=1.5e9
        forcing=evolution.conditions(s,[20.],[20.*2800.*1000.],controls)
        cutoff=(pressure/9.81-20.*2800.*1000.)/(2800.*1000.)
        np.testing.assert_allclose(forcing['eligible_fraction'],[(40.-cutoff)/40.])
        advance(s,1.,1000.,10.,5.,detachment_fraction=0.)
        dense=s[phase.DENSE][0]
        ordinary=s['thickness_km'][0]-dense
        forcing=evolution.conditions(s,[20.],[20.*2800.*1000.],controls)
        np.testing.assert_allclose(forcing['eligible_fraction'],[max(ordinary-cutoff,0.)/ordinary])

    def test_reference_heating_time_is_set_by_the_thickened_column(self):
        # The geotherm of a doubled crust relaxes over tens of Myr. A 20 km
        # conductive length made the reference first-mode time 1.3 Myr, so a
        # cold underthrust sheet reached the 600 C reaction temperature at once.
        controls=evolution.parameters()
        s=state(thickness=35.,temperature=300.)
        forcing=evolution.conditions(s,[35.],[35.*2800.*1000.],controls)
        tau=float(np.asarray(forcing['thermal_tau_myr']).ravel()[0])
        expected=(controls['diffusion_length_km']*1e3)**2/(np.pi**2*1e-6)/(365.25*86400.*1e6)
        self.assertAlmostEqual(tau,expected,places=12)
        self.assertGreater(tau,10.)
        self.assertLess(tau,30.)
        # Buried under 35 km, the bath is ~860 C. Crossing 600 C from 300 C
        # takes -tau*ln((860-600)/(860-300)) ~ 0.77 tau, i.e. >10 Myr.
        bath=float(np.asarray(forcing['bath_temperature_c']).ravel()[0])
        crossing=-tau*np.log((bath-phase.REACTION_TEMPERATURE_C)/(bath-300.))
        self.assertGreater(crossing,10.)

    def test_draining_requires_overstress_and_is_suppressed_by_cold_viscosity(self):
        s=state(temperature=1000.)
        advance(s,1.,1000.,10.,10.,detachment_fraction=0.)
        controls=evolution.parameters(dict(activation_energy_j_mol=200000.,supporting_yield_stress_pa=5e6))
        forcing=evolution.conditions(s,[0.],[0.],controls)
        stress=150.*9.81*s[phase.DENSE][0]*1000.
        expected=max(stress-5e6,0.)/(3.*1e21)*(365.25*86400.*1e6)
        np.testing.assert_allclose(forcing['drainage_rate_myr'],[expected],rtol=2e-13)
        cold=deepcopy(s)
        cold[phase.HEAT]=phase.mass_volume(cold)*400.
        cold[phase.HEAT_BATH]+=cold[phase.HEAT]-s[phase.HEAT]
        chill=evolution.conditions(cold,[0.],[0.],controls)
        self.assertLess(chill['drainage_rate_myr'][0],expected/1e6)
        strong=evolution.conditions(s,[0.],[0.],evolution.parameters(dict(supporting_yield_stress_pa=1e9)))
        np.testing.assert_array_equal(strong['drainage_rate_myr'],0.)

    def test_local_covered_regions_react_when_average_overburden_would_not(self):
        s=world(lower_km=30.,upper_km=40.,alone_km=30.,thin_km=30.,cover=.45)
        prepared=evolution.prepare(s)
        covered=[r for r in prepared['regions'] if r['face']==1 and len(r['upper'])]
        fraction=sum(r['weight'] for r in covered)
        self.assertLess((30.+40.*fraction)*2800.*1000.*9.81,1.5e9)
        merged,result,_=evolution.evolve(s,s.structure,2.,prepared=prepared)
        self.assertGreater(merged[phase.CONVERTED][1],0.)
        self.assertEqual(merged[phase.CONVERTED][2],0.)
        self.assertEqual(merged[phase.CONVERTED][3],0.)
        before=float(s.mass@phase.mass_volume(s.structure))
        after=float(s.mass@phase.mass_volume(merged))
        self.assertAlmostEqual(after/before,1.,places=13)
        self.assertGreater(float(result['contraction_km'][1]),0.)

    def test_zero_reserve_column_completes_the_step_with_reserve_limited_return(self):
        # Face 3 is born on the 8 km mass floor like a juvenile arc sliver and
        # carries a dense root above the overstress threshold. The source step
        # must complete: return is limited to the (zero) reserve, the dense
        # phase stays, every ledger closes, and the limit is reported.
        s=world(thin_km=8.)
        advance(s.structure,[0.,0.,0.,1.],1000.,10.,60.,detachment_fraction=0.)
        advance(s.trace_structure,[0.,0.,0.,1.],1000.,10.,60.,detachment_fraction=0.)
        self.assertGreater(float(s.structure[phase.DENSE][3]/s.structure['area_factor'][3]),3.5)
        np.testing.assert_allclose(phase.restored_thickness(s.structure)[3],8.,rtol=0,atol=1e-9)
        s.retained_phase_parameters=evolution.parameters(dict(surface_temperature_c=1000.,
            mantle_temperature_c=1000.,geotherm_c_per_km=0.,supporting_yield_stress_pa=0.))
        forcing=evolution.conditions(s.structure,np.zeros(4),np.zeros(4),s.retained_phase_parameters)
        self.assertGreater(float(forcing['drainage_rate_myr'][3]),.01)
        before=float(s.mass@(phase.mass_volume(s.structure)+s.structure[inventory.RETURNED]))
        merged,result,_=evolution.evolve(s,s.structure,2.)
        phase.validate(merged)
        self.assertGreaterEqual(float(phase.restored_thickness(merged)[3]),8.-phase.FLOOR_TOLERANCE_KM)
        np.testing.assert_allclose(phase.restored_thickness(merged)[3],8.,rtol=0,atol=1e-9)
        self.assertEqual(float(result['returned_km'][3]),0.)
        self.assertGreater(float(result['withheld_equivalent_km'][3]),0.)
        self.assertTrue(bool(result['reserve_limited_faces'][3]))
        self.assertEqual(int(result['reserve_limited_faces'].sum()),1)
        diagnostics=result['diagnostics']
        self.assertEqual(diagnostics['reserve_limited_faces'],1)
        self.assertGreaterEqual(diagnostics['reserve_limited_regions'],1)
        self.assertGreater(diagnostics['reserve_withheld_return_km3'],0.)
        self.assertEqual(diagnostics['drainage_reserve_policy'],evolution.DRAINAGE_RESERVE_POLICY)
        # The thick isolated root (face 2) still drains normally.
        self.assertGreater(float(result['returned_km'][2]),0.)
        self.assertFalse(bool(result['reserve_limited_faces'][2]))
        after=float(s.mass@(phase.mass_volume(merged)+merged[inventory.RETURNED]))
        self.assertAlmostEqual(after/before,1.,places=13)
        self.assertGreaterEqual(float(merged[phase.DENSE][3]),float(s.structure[phase.DENSE][3])*.999)
        # Markers use the same closure.
        traced,trace_result,_=evolution.evolve(s,s.trace_structure,2.,trace=True,prepared=result['prepared'])
        phase.validate(traced)
        self.assertEqual(float(trace_result['returned_km'][3]),0.)
        self.assertTrue(bool(trace_result['reserve_limited_faces'][3]))
        self.assertEqual(trace_result['diagnostics']['reserve_withheld_return_km3'],0.)

    def test_full_thermal_phase_interval_converges_with_internal_step_refinement(self):
        s=world()
        s.retained_phase_parameters=evolution.parameters(dict(surface_temperature_c=1000.,
            mantle_temperature_c=1000.,geotherm_c_per_km=0.,supporting_yield_stress_pa=0.))
        results=[]
        for dt in (.5,.25,.125,.0625):
            s.retained_phase_parameters['maximum_substep_myr']=dt
            merged,report,_=evolution.evolve(s,s.structure,5.)
            results.append(merged[phase.DENSE].copy())
            phase.validate(merged)
        errors=[np.linalg.norm(x-results[-1]) for x in results[:-1]]
        self.assertGreater(errors[0],errors[1])
        self.assertGreater(errors[1],errors[2])
        self.assertLess(errors[1]/np.linalg.norm(results[-1]),.02)

    def test_prepared_thermal_forcing_is_coeval_and_does_not_mutate_sources(self):
        s=world()
        prepared=evolution.prepare(s)
        before=deepcopy(s.structure)
        first,_,_=evolution.evolve(s,s.trace_structure,2.,trace=True,prepared=prepared)
        s.structure['thickness_km']*=.9
        second,_,_=evolution.evolve(s,s.trace_structure,2.,trace=True,prepared=prepared)
        for key in first:np.testing.assert_array_equal(first[key],second[key])
        for key in before:
            if key!='thickness_km':np.testing.assert_array_equal(s.structure[key],before[key])

    def test_dense_lower_layer_reduces_upper_support_and_detachment_restores_it(self):
        s=world(lower_km=60.,upper_km=40.,cover=1.)
        collision_surface.refresh(s)
        before=s.parcel_collision_support_m.copy()
        old=s.structure['thickness_km'].copy()
        advance(s.structure,np.array([0.,1.,0.,0.]),1000.,10.,2.,detachment_fraction=0.)
        collision_surface.refresh(s)
        reduction=before[0]-s.parcel_collision_support_m[0]
        expected=(old[1]-s.structure['thickness_km'][1])*1000.
        self.assertAlmostEqual(reduction/expected,1.,places=12)
        before=s.parcel_collision_support_m.copy()
        report=advance(s.structure,0.,1000.,10.,5.)
        collision_surface.refresh(s)
        self.assertAlmostEqual((s.parcel_collision_support_m[0]-before[0])/
            (report['returned_km'][1]*150./3300.*1000.),1.,places=12)


if __name__=='__main__':unittest.main()
