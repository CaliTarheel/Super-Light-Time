"""Independent coupled Stokes/damage event and restart oracles."""
from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import math
import pickle
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import checkpoint
import plate_balance as pb
import slab_memory as slab
import slab_tether_history as history
import slab_tether_local as local
from slab_tether import SECONDS_PER_MYR
from slab_tether_evolution import integrate_frozen_geometry
from tests.test_slab_tether_forces import fixture


@contextmanager
def linear_forces():
    # Isolate a physically linear two-body problem while using real assembly
    # and force acceptance. Retain basal and slab resistance, remove friction.
    with patch.multiple(pb,ASTHENOSPHERE_DRAG_PA_S_M=1e19,HINGE_VISCOSITY_PA_S=0.,
                        MEGATHRUST_N_PER_M=0.,TRANSFORM_SHEAR_PA=0.,COMPRESSIVE_N_PER_M=0.):
        yield


def world(damage=.3):
    s=fixture(damage=damage,edges=(1000.,));s.plate_resistance_version=1
    row=s.trench_systems[0]
    row[history.FIELD][0]['neck']['failure_opening_m']=1000.
    local.localize(row,s.bmid,[1.],support_radius_km=1000.)
    s.t=42.;s.rng=np.random.default_rng(97)
    return s


def oracle(s):
    """Analytic integral of the independently eliminated 2x2 Stokes system.

    B*v = Cn*(u-v), W = Cs*u+Cn*(u-v). Hence
    dt/dd = delta/W * [Cs + C0*(1+Cs/B)*(1-d)^2].
    """
    row=s.trench_systems[0];neck=row[history.FIELD][0]['neck']
    width=1e6
    basal=2.*1e19*1000.*1e6
    mantle=3e22*(300.*math.sin(math.radians(50.))/660.)*width
    intact=4.*neck['viscosity_pa_s']*neck['thickness_m']/neck['length_m']*width
    weight=row[slab.RETAINED_MASS_FIELD]*9.81*math.sin(math.radians(50.))
    initial=neck['damage'];opening=neck['failure_opening_m']
    def elapsed(d):
        return opening/weight/SECONDS_PER_MYR*(mantle*(d-initial)+intact*(1.+mantle/basal)*
                                               ((1.-initial)**3-(1.-d)**3)/3.)
    def speed(d):
        cn=intact*(1.-d)**2
        return cn*weight/(basal*(mantle+cn)+cn*mantle)
    return elapsed,speed


class CoupledSlabEvolutionTests(unittest.TestCase):
    def test_coupled_rupture_time_matches_independent_integral_and_consumes_remainder(self):
        s=world();elapsed,_=oracle(s);failure=elapsed(1.)
        before=pickle.dumps(s)
        with linear_forces():
            result,report=integrate_frozen_geometry(s,2.*failure,max_step_myr=failure,
                                                   absolute_tolerance=1e-10,relative_tolerance=1e-9)
        self.assertEqual(pickle.dumps(s),before)
        self.assertEqual(len(report['events']),1)
        self.assertAlmostEqual(report['events'][0]['time_myr'],failure,delta=2e-7*failure)
        self.assertEqual(report['advanced_dt_myr'],2.*failure)
        self.assertEqual(report['remaining_dt_myr'],0.)
        self.assertEqual(report['intervals'][-1]['end_myr'],2.*failure)
        self.assertLess(report['events'][0]['time_myr'],report['advanced_dt_myr'])
        self.assertEqual(result.t,42.)
        self.assertEqual(result.trench_systems[0][slab.RETAINED_MASS_FIELD],0.)
        self.assertEqual(result.trench_systems[0]['slab_retired_excess_mass_kg'],s.trench_systems[0][slab.RETAINED_MASS_FIELD])
        np.testing.assert_array_equal(result.omega,0.)
        self.assertLessEqual(report['maximum_scaled_force_residual'],pb.FORCE_RELATIVE_TOLERANCE)

    def test_pre_event_damage_and_velocity_follow_coupled_analytic_solution(self):
        s=world();elapsed,speed=oracle(s)
        target=.8;duration=elapsed(target)
        with linear_forces():
            result,report=integrate_frozen_geometry(s,duration,max_step_myr=duration,
                                                   absolute_tolerance=1e-10,relative_tolerance=1e-9)
        actual=result.trench_systems[0][history.FIELD][0]['neck']['damage']
        self.assertAlmostEqual(actual,target,delta=3e-8)
        v=result.omega[0,2]*pb.RADIUS_M/SECONDS_PER_MYR
        self.assertAlmostEqual(v/speed(target),1.,delta=4e-7)
        self.assertGreater(abs(v-speed(.3))/speed(.3),.01)
        self.assertEqual(report['events'],[])
        self.assertGreater(report['rejected_trials'],0)

    def test_error_refinement_improves_event_time(self):
        s=world();elapsed,_=oracle(s);failure=elapsed(1.)
        errors=[]
        with linear_forces():
            for tolerance in (1e-4,1e-8):
                _,report=integrate_frozen_geometry(s,1.1*failure,max_step_myr=failure,
                                                    absolute_tolerance=tolerance,relative_tolerance=tolerance)
                errors.append(abs(report['events'][0]['time_myr']-failure))
        self.assertLess(errors[1],errors[0]*.05)
        self.assertLess(errors[1]/failure,1e-5)

    def test_checkpoint_continuation_preserves_coupled_trajectory_and_event_time(self):
        s=world();elapsed,_=oracle(s);failure=elapsed(1.)
        with linear_forces():
            whole,complete=integrate_frozen_geometry(s,1.5*failure,max_step_myr=failure/10.)
            first,a=integrate_frozen_geometry(s,.4*failure,max_step_myr=failure/10.)
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'state.npz'
                checkpoint.write_checkpoint(path,first,dict(config={}),{})
                resumed,_=checkpoint.read_checkpoint(path,None,SimpleNamespace)
            end,b=integrate_frozen_geometry(resumed,1.1*failure,max_step_myr=failure/10.)
        self.assertEqual(end.trench_systems,whole.trench_systems)
        self.assertAlmostEqual(.4*failure+b['events'][0]['time_myr'],complete['events'][0]['time_myr'],delta=2e-6*failure)
        np.testing.assert_allclose(a['rotation_integral_rad']+b['rotation_integral_rad'],complete['rotation_integral_rad'],rtol=2e-6,atol=1e-12)

    def test_force_solve_budget_failure_does_not_commit_damage_mass_rng_or_time(self):
        s=world();before=pickle.dumps(s)
        with linear_forces(),self.assertRaisesRegex(ValueError,'budget exhausted'):
            integrate_frozen_geometry(s,1.,max_force_solves=2)
        self.assertEqual(pickle.dumps(s),before)

    def test_events_are_global_across_distinct_trench_rows(self):
        from tests.test_slab_tether_local import world as two_patches
        s=two_patches();s.plate_resistance_version=1;row=s.trench_systems[0]
        row[history.FIELD][0]['neck']['damage']=.9
        for c in row[history.FIELD]:c['neck']['failure_opening_m']=1000.
        child={k:deepcopy(row[k]) for k in ('phase','maturity','downgoing_plate_uid','overriding_plate_uid')}
        child.update(id=2,length_km=600.)
        slab.partition(row,child,.6,channel_fractions=[0.,1.])
        row['length_km']=400.;slab.refresh_line_load(row)
        s.trench_systems.append(child);s.trench_id[1]=2
        with linear_forces():
            result,report=integrate_frozen_geometry(s,2.,max_step_myr=.2)
        self.assertEqual([e['trench_id'] for e in report['events']],[1,2])
        self.assertLess(report['events'][0]['time_myr'],report['events'][1]['time_myr'])
        self.assertEqual(sum(r[slab.RETAINED_MASS_FIELD] for r in result.trench_systems),0.)
        for previous,current in zip(report['intervals'],report['intervals'][1:]):
            self.assertEqual(previous['end_myr'],current['start_myr'])

    def test_actual_nonlinear_balance_is_resolved_at_each_damage_stage(self):
        s=world(damage=.6)
        result,report=integrate_frozen_geometry(s,.02,max_step_myr=.01)
        self.assertGreater(result.trench_systems[0][history.FIELD][0]['neck']['damage'],.6)
        self.assertLessEqual(report['maximum_scaled_force_residual'],pb.FORCE_RELATIVE_TOLERANCE)
        np.testing.assert_array_equal(result.xyz,s.xyz)
        self.assertEqual(result.rng.bit_generator.state,s.rng.bit_generator.state)

    def test_invalid_controls_fail_without_silently_shortening_interval(self):
        s=world()
        for dt,kwargs in ((0.,{}),(-1.,{}),(np.inf,{}),(1.,dict(max_step_myr=0.)),
                           (1.,dict(relative_tolerance=0.)),(1.,dict(max_force_solves=True))):
            with self.assertRaises(ValueError):integrate_frozen_geometry(s,dt,**kwargs)
        s.plate_resistance_version=0
        with self.assertRaisesRegex(ValueError,'passive'):
            integrate_frozen_geometry(s,.1)


if __name__=='__main__':unittest.main()
