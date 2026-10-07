"""Persistent entry identity, native stage epochs, remeshing and restart."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import checkpoint
import entry_regions as regions
import material_surface
import native_material_adaptivity as adaptivity
from ridge_geometry import rotate
from tests.test_slab_tether_native import world as slab_world,advance
from tests.test_native_material_adaptivity import world as adaptive_world


def unit(x):
    x=np.asarray(x,float);return x/np.linalg.norm(x,axis=-1,keepdims=True)


def install(s,index=0):
    s.plate_balance_version=s.plate_resistance_version=s.material_mechanics_version=1
    s.config['deforming_regions']=1
    row=s.trench_systems[0]
    center=unit(s.material_surface['vertices'][s.material_surface['faces'][index]].sum(axis=0))
    normal=unit(np.cross(center,[.3,.2,.8])+.002*center)
    regions.initialize(s,[dict(face_ids=s.parcel_patch[index:index+1],trench_id=row['id'],
        overriding_plate_uid=row['overriding_plate_uid'],hinge_normal=normal,dip_degrees=50.)])
    return normal


def world():
    s=slab_world();down=s.trench_systems[0]['downgoing_plate_uid']
    owner=int(np.flatnonzero(s.plate_uid==down)[0])
    cells=np.flatnonzero((s.plate==owner)&(np.abs(s.xyz[:,1])>.5))
    s._add_arc_crust(cells[:1],np.array([1000.]))
    s._rasterize();s._boundaries()
    install(s)
    return s


def remesh_world():
    s=adaptive_world();s.active=np.ones(2,bool)
    s.trench_systems=[dict(id=1,downgoing_plate_uid=31,overriding_plate_uid=42)]
    install(s);s.omega[:]=0.
    return s


class EntryRegionTests(unittest.TestCase):
    def test_hinge_moves_with_overrider_once_and_stale_epoch_is_rejected(self):
        s=remesh_world();s.omega[1]=[.001,-.002,.003]
        initial=s.continental_entry_regions['regions'][0]['hinge_normal'].copy()
        expected=rotate(initial[None],s.omega[1]*.2)[0]
        spec=regions.specification(s,advance_myr=.2)
        np.testing.assert_array_equal(spec['hinge_normals'][0],expected)
        np.testing.assert_array_equal(s.continental_entry_regions['regions'][0]['hinge_normal'],initial)
        regions.advance_hinges(s,.2)
        with self.assertRaisesRegex(ValueError,'epochs disagree'):regions.advance_hinges(s,.2)
        s.t+=.2
        np.testing.assert_array_equal(regions.specification(s)['hinge_normals'][0],expected)

    def test_real_refine_and_coarsen_preserve_membership_and_its_existing_inventory(self):
        s=remesh_world();before=regions.energy_j(s)
        volume=float(np.sum(s.mass[s.parcel_entry_region>0]*s.structure['thickness_km'][s.parcel_entry_region>0]))
        original=s.parcel_patch.copy();self.assertTrue(adaptivity.adapt(s,2.))
        selected=s.parcel_entry_region>0
        self.assertEqual(int(selected.sum()),4)
        np.testing.assert_array_equal(s.material_lineage['root_id'][selected],1000)
        self.assertAlmostEqual(float(np.sum(s.mass[selected]*s.structure['area_factor'][selected]*s.structure['thickness_km'][selected]))/volume,1.,places=13)
        self.assertAlmostEqual((regions.energy_j(s)-before-s.continental_entry_regions['remesh_energy_change_j'])/abs(before),0.,places=13)
        regions.advance_hinges(s,40.);s.t+=40.
        s.material_deformation=dict(face_weight=np.zeros(len(s.mass)),face_rigid=np.ones(len(s.mass),bool),face_strain=np.zeros(len(s.mass)))
        self.assertTrue(adaptivity.adapt(s,2.))
        np.testing.assert_array_equal(s.parcel_patch,original)
        self.assertEqual(int((s.parcel_entry_region>0).sum()),1)
        self.assertAlmostEqual(regions.energy_j(s)/before,1.,places=12)

    def test_different_local_hinge_histories_cannot_be_averaged_by_coarsening(self):
        s=remesh_world();adaptivity.adapt(s,2.)
        selected=np.flatnonzero(s.parcel_entry_region>0)
        second=deepcopy(s.continental_entry_regions['regions'][0]);second['id']=2
        second['hinge_normal']=rotate(second['hinge_normal'][None],[.001,0.,0.])[0]
        s.continental_entry_regions['regions'].append(second)
        s.parcel_entry_region[selected[0]]=2
        regions.advance_hinges(s,40.);s.t+=40.
        s.material_deformation=dict(face_weight=np.zeros(len(s.mass)),face_rigid=np.ones(len(s.mass),bool),face_strain=np.zeros(len(s.mass)))
        adaptivity.adapt(s,2.)
        self.assertEqual(int(np.count_nonzero(s.material_lineage['root_id']==1000)),4)
        self.assertEqual(int(np.count_nonzero(s.parcel_entry_region==2)),1)

    def test_native_interval_uses_matching_hinge_epoch_and_books_sources_separately(self):
        s=world();initial=deepcopy(s.continental_entry_regions);old_ids=s.parcel_patch.copy()
        with self.assertRaisesRegex(ValueError,'atomic coupled'):s.step(.00001)
        result=advance(s,.00001,max_source_step_myr=.00001)
        self.assertTrue(result['persistent_entry_regions'])
        self.assertEqual(s.continental_entry_regions['epoch_myr'],s.t)
        self.assertGreater(np.linalg.norm(s.continental_entry_regions['regions'][0]['hinge_normal']-initial['regions'][0]['hinge_normal']),0.)
        new=~np.isin(s.parcel_patch,old_ids)
        np.testing.assert_array_equal(s.parcel_entry_region[new],0)
        for row in result['source_intervals']:
            self.assertAlmostEqual((row['entry_energy_after_j']-row['entry_energy_after_motion_j']
                -row['entry_source_energy_change_j']-row['entry_remesh_energy_change_j'])/max(abs(row['entry_energy_after_j']),1.),0.,places=13)
        gravity=s.deformation_diagnostics['gravitational_relaxation']
        self.assertIn('entry_hinge_rigid_work_j',gravity)
        self.assertLessEqual(gravity['energy_after_km4'],gravity['energy_before_km4']+1.)

    def test_native_checkpoint_continuation_preserves_domains_and_solution(self):
        from native_engine import Simulation
        s=world();advance(s,.000005,max_source_step_myr=.000005)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'entry.npz'
            checkpoint.write_checkpoint(path,s,dict(config=s.config),{})
            resumed,_=checkpoint.read_checkpoint(path,None,Simulation)
        report=advance(s,.000005,max_source_step_myr=.000005)
        replay=advance(resumed,.000005,max_source_step_myr=.000005)
        self.assertEqual(report,replay)
        np.testing.assert_array_equal(s.material_surface['vertices'],resumed.material_surface['vertices'])
        np.testing.assert_array_equal(s.parcel_entry_region,resumed.parcel_entry_region)
        np.testing.assert_array_equal(s.omega,resumed.omega)
        for a,b in zip(s.continental_entry_regions['regions'],resumed.continental_entry_regions['regions']):
            np.testing.assert_array_equal(a['hinge_normal'],b['hinge_normal'])

    def test_failed_native_source_rolls_back_entry_epoch_and_material_together(self):
        from native_engine import Simulation
        s=world();before=deepcopy(s.__dict__);original=Simulation._advance_step
        def fail(staged,dt,**kwargs):
            original(staged,dt,**kwargs)
            raise ValueError('failure after entry transport')
        with patch.object(Simulation,'_advance_step',fail),self.assertRaisesRegex(ValueError,'failure after entry'):
            advance(s,.000005,max_source_step_myr=.000005)
        self.assertEqual(s.t,before['t'])
        self.assertEqual(s.continental_entry_regions['epoch_myr'],before['continental_entry_regions']['epoch_myr'])
        np.testing.assert_array_equal(s.material_surface['vertices'],before['material_surface']['vertices'])
        np.testing.assert_array_equal(s.parcel_entry_region,before['parcel_entry_region'])
        np.testing.assert_array_equal(s.continental_entry_regions['regions'][0]['hinge_normal'],
            before['continental_entry_regions']['regions'][0]['hinge_normal'])

    def test_native_sheet_receives_end_epoch_hinge_and_the_force_inventory(self):
        import native_material_evolution as material
        import gravitational_relaxation as gravity
        s=world();original=material.advect;relax=gravity.relax;seen=[]
        def audit(staged,dt,**kwargs):
            row=staged.continental_entry_regions['regions'][0]
            slot=int(np.flatnonzero(staged.plate_uid==row['overriding_plate_uid'])[0])
            spin=staged.omega[slot]*dt;angle=np.linalg.norm(spin)
            normal=np.asarray(row['hinge_normal']);axis=spin/angle
            # Independent Rodrigues rotation at the end of prescribed motion.
            expected=normal*np.cos(angle)+np.cross(axis,normal)*np.sin(angle)+axis*np.dot(axis,normal)*(1.-np.cos(angle))
            expected_volume=staged.mass*staged.structure['area_factor']*staged.structure['thickness_km']
            def inspect(*args,**controls):
                potential=controls['entry_potential']
                np.testing.assert_allclose(potential.normals[0],expected,rtol=0.,atol=3e-16)
                np.testing.assert_array_equal(args[2],expected_volume)
                self.assertEqual(staged.continental_entry_regions['epoch_myr'],staged.t)
                seen.append(angle)
                return relax(*args,**controls)
            with patch.object(gravity,'relax',inspect):return original(staged,dt,**kwargs)
        with patch.object(material,'advect',audit):advance(s,.00001,max_source_step_myr=.00001)
        self.assertTrue(seen);self.assertTrue(all(angle>0 for angle in seen))

    def test_registration_failure_and_unmapped_material_loss_do_not_reset_history(self):
        s=remesh_world();old=deepcopy(s.continental_entry_regions)
        with self.assertRaisesRegex(ValueError,'already initialized'):regions.initialize(s,[])
        np.testing.assert_array_equal(old['face_ids'],s.continental_entry_regions['face_ids'])
        s.parcel_patch=s.parcel_patch[1:]
        with self.assertRaisesRegex(ValueError,'disappeared'):regions.ensure_fields(s)
        np.testing.assert_array_equal(old['face_ids'],s.continental_entry_regions['face_ids'])


if __name__=='__main__':unittest.main()
