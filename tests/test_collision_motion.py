"""Coupled contact-motion proxies: no thermodynamic-energy claims."""
from copy import deepcopy
from itertools import permutations
from types import SimpleNamespace
import unittest
import numpy as np
import collision_contacts
from orientation import orient_frame, rotation_matrix


def world():
    area=np.array([100.,130.,160.])
    contacts=[dict(id=1,state='active',top_owner=0,under_owner=1,overlap_area_km2=80.,center=[1.,0.,0.],normal=[0.,1.,0.]),
              dict(id=2,state='active',top_owner=1,under_owner=2,overlap_area_km2=100.,center=[1.,0.,0.],normal=[0.,.8,.6]),
              dict(id=3,state='active',top_owner=0,under_owner=2,overlap_area_km2=70.,center=[0.,1.,0.],normal=[-1.,0.,0.])]
    return SimpleNamespace(parcel_patch=np.arange(3,dtype=np.int64),parcel_plate=np.arange(3),mass=area.copy(),
        parcel_collision_sheet=np.arange(1,4,dtype=np.int64),parcel_exposed_fraction=np.ones(3),
        parcel_burial_myr=np.zeros(3),parcel_collision_suture=np.zeros(3),collision_contacts=contacts,
        material_surface=dict(area_km2=area,faces=np.arange(9).reshape(3,3)),
        omega=np.array([[.002,.001,.004],[-.001,-.002,-.002],[.003,.002,-.008]]))


def split(row,count):
    result=[]
    for i in range(count):
        piece=deepcopy(row);piece['id']=100+10*row['id']+i;piece['overlap_area_km2']/=count;result.append(piece)
    return result


class CollisionMotionTests(unittest.TestCase):
    def test_contact_order_is_invariant(self):
        source=world();results=[]
        for order in permutations(source.collision_contacts):
            s=deepcopy(source);s.collision_contacts=list(deepcopy(order));collision_contacts.resist_motion(s,2.)
            results.append(s.omega)
        for value in results[1:]:np.testing.assert_allclose(value,results[0],atol=2e-17,rtol=0)

    def test_identical_front_subdivision_preserves_capped_and_coupled_motion(self):
        for dt in (.25,2.,4.,10.):
            one=world();many=deepcopy(one)
            many.collision_contacts=[piece for row in reversed(many.collision_contacts) for piece in split(row,4)]
            collision_contacts.resist_motion(one,dt);collision_contacts.resist_motion(many,dt)
            np.testing.assert_allclose(many.omega,one.omega,rtol=0,atol=2e-17)

    def test_local_fronts_match_equivalent_explicit_fronts_in_coupled_solve(self):
        nested=world();explicit=deepcopy(nested)
        a,b=deepcopy(nested.collision_contacts[0]),deepcopy(nested.collision_contacts[0])
        a['overlap_area_km2']=40.;b.update(id=17,overlap_area_km2=40.,center=[0.,1.,0.],normal=[1.,0.,0.])
        # This pair has simultaneous opening/closing patches. Its aggregate
        # zero normal must not discard either physical local constraint.
        nested.collision_contacts[0].update(normal=[0.,0.,0.],local_fronts=[a,b])
        explicit.collision_contacts=[a,b]+explicit.collision_contacts[1:]
        collision_contacts.resist_motion(nested,2.);collision_contacts.resist_motion(explicit,2.)
        np.testing.assert_allclose(nested.omega,explicit.omega,atol=2e-17,rtol=0)
        self.assertEqual(nested.collision_resistance_diagnostics['input_active_contacts'],4)
        self.assertEqual(nested.collision_resistance_diagnostics['distinct_motion_fronts'],4)

    def test_quiet_parent_or_explicit_empty_fronts_cannot_apply_stale_motion(self):
        for quiet in (False,True):
            s=world();before=s.omega.copy()
            for row in s.collision_contacts:
                child=deepcopy(row)
                row['local_fronts']=[child] if quiet else []
                if quiet:row['state']='quiet'
            collision_contacts.resist_motion(s,2.)
            np.testing.assert_array_equal(s.omega,before)
            self.assertEqual(s.collision_resistance_diagnostics['input_active_contacts'],0)

    def test_the_retired_velocity_relaxation_law_stays_retired(self):
        """resist_motion must not move omega, in the cases that used to prove it did.

        The weld is now a plastic element that plate_balance assembles on each pair's
        overlap-area rate, inside the inertia-free solve. Five tests here pinned the old
        velocity law: its exponential formula and 20%-per-step cap, its coupled active set,
        its damping proxy, that proxy's dissipation, and the timestep convergence of the
        integrator. They were retired WITH the law rather than left asserting behaviour the
        engine no longer has, and this replaces them with the inverse claim over the same
        three scenarios. The weld's own physics is covered by tests/test_suture_weld.py and
        measured against a real checkpoint by
        validation/balance-weld-sink-20260914/acceptance.py.
        """
        def closing_pair():
            s=world();s.collision_contacts=s.collision_contacts[:1];return s

        def initially_open_front():
            s=world();s.collision_contacts=s.collision_contacts[:2]
            s.omega[:]=0.;s.omega[1,2]=.0001;s.omega[2,2]=-.01;return s

        def balanced_motion():
            s=world();s.material_surface['area_km2'][:]=128.;s.mass[:]=128.
            s.omega[:]=np.array([[2.,1.,4.],[-2.,-1.,-4.],[0.,0.,0.]])/1024.;return s

        for label,build in (('closing pair',closing_pair),
                            ('initially open front',initially_open_front),
                            ('balanced motion',balanced_motion)):
            for dt in (.1,2.,10.):
                s=build();before=s.omega.copy()
                collision_contacts.resist_motion(s,dt)
                np.testing.assert_array_equal(s.omega,before,
                    err_msg=f'{label}: resist_motion moved omega at dt={dt}')
                self.assertIn('contacts',s.collision_resistance_diagnostics)

    def test_opening_and_shared_rigid_rotation_are_bitwise_unchanged(self):
        for common in (False,True):
            s=world();s.collision_contacts=s.collision_contacts[:1]
            if common:s.omega[:]=[.002,-.003,.007]
            else:s.omega=-s.omega
            before=s.omega.copy();collision_contacts.resist_motion(s,2.)
            np.testing.assert_array_equal(s.omega,before)

    def test_rotation_covariance(self):
        source=world();rotated=deepcopy(source);matrix=rotation_matrix(dict(yaw=171.,pitch=84.,roll=-39.))
        rotated.omega=rotated.omega@matrix
        for row in rotated.collision_contacts:
            row['center']=np.asarray(row['center'])@matrix;row['normal']=np.asarray(row['normal'])@matrix
        collision_contacts.resist_motion(source,2.);collision_contacts.resist_motion(rotated,2.)
        np.testing.assert_allclose(rotated.omega,source.omega@matrix,rtol=0,atol=4e-17)

    def test_saved_motion_diagnostics_rotate_vectors_and_preserve_all_scalars(self):
        angles=dict(yaw=131.,pitch=-82.,roll=24.);matrix=rotation_matrix(angles)
        vector_keys={'angular_motion_proxy_before','angular_motion_proxy_after'}
        for balanced in (False,True):
            s=world()
            if balanced:
                s.material_surface['area_km2'][:]=128.;s.mass[:]=128.
                s.omega[:]=np.array([[2.,1.,4.],[-2.,-1.,-4.],[0.,0.,0.]])/1024.
            collision_contacts.resist_motion(s,2.)
            frame=dict(width=8,height=4,collision_resistance_diagnostics=deepcopy(s.collision_resistance_diagnostics),
                       collision_contacts=[dict(id=7,local_fronts=[dict(id=3,center=[1.,0.,0.],
                                                normal=[0.,1.,0.],overlap_area_km2=17.)])])
            original=deepcopy(frame);result=orient_frame(frame,angles)
            for key in vector_keys:
                np.testing.assert_allclose(result['collision_resistance_diagnostics'][key],
                    np.asarray(frame['collision_resistance_diagnostics'][key])@matrix,rtol=0,atol=1e-30)
            scalars=lambda row:{key:value for key,value in row.items() if key not in vector_keys}
            self.assertEqual(scalars(result['collision_resistance_diagnostics']),scalars(frame['collision_resistance_diagnostics']))
            nested=result['collision_contacts'][0]['local_fronts'][0]
            np.testing.assert_array_equal(nested['center'],np.array([1.,0.,0.])@matrix)
            np.testing.assert_array_equal(nested['normal'],np.array([0.,1.,0.])@matrix)
            self.assertEqual(nested['overlap_area_km2'],17.)
            self.assertEqual(frame,original)

    def test_distinct_locations_with_same_angular_axis_are_not_merged(self):
        s=world();s.collision_contacts=s.collision_contacts[:1]
        other=deepcopy(s.collision_contacts[0]);other.update(id=99,center=[0.,1.,0.],normal=[-1.,0.,0.])
        s.collision_contacts.append(other);collision_contacts.resist_motion(s,2.)
        self.assertEqual(s.collision_resistance_diagnostics['distinct_motion_fronts'],2)

    def test_quiet_contacts_do_not_change_motion_and_bad_input_does_not_commit(self):
        s=world()
        for row in s.collision_contacts:row['state']='quiet'
        before=s.omega.copy();collision_contacts.resist_motion(s,2.)
        np.testing.assert_array_equal(s.omega,before)
        self.assertEqual(s.collision_resistance_diagnostics['input_active_contacts'],0)
        s=world();before=s.omega.copy();s.collision_contacts[0]['overlap_area_km2']=np.nan
        with self.assertRaises(ValueError):collision_contacts.resist_motion(s,2.)
        np.testing.assert_array_equal(s.omega,before)

    def test_repeated_calls_are_path_independent(self):
        """What replaces the retired integrator's timestep-convergence test.

        That test asserted a coarse timestep carried a discretisation error which a finer
        one reduced -- a property of integrating a velocity law. There is no integration
        here any more, so the meaningful claim is the stronger one: the result does not
        depend on the step size or on how many times the census is taken.
        """
        def run(step):
            s=world()
            for _ in range(round(2./step)):collision_contacts.resist_motion(s,step)
            return s.omega
        np.testing.assert_array_equal(run(.25),run(2.))
        np.testing.assert_array_equal(run(2./256),world().omega)


if __name__=='__main__':unittest.main()
