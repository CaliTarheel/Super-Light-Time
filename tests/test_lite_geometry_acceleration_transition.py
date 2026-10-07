"""A source-only accelerator must not import an unreviewed or altered world."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import checkpoint
import lite_geometry_acceleration_transition as transition
import native_engine


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


class GeometryAccelerationTransitionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.parent = self.root / 'preserved-current-parent'
        self.parent.mkdir()
        self.old = {name: b'VALUE = 1\n' for name in transition.HELPERS}
        self.old.update({f'helper_{index:03d}.py': b'VALUE = 3\n' for index in range(171)})
        self.new = dict(self.old)
        for index, name in enumerate(transition.HELPERS):
            self.new[name] = f'VALUE = {index+2}\n'.encode()
        self.engine = ('\n'.join('import '+name[:-3] for name in self.old)+'\n').encode()
        self.prior = dict(engine_sha256=sha(self.engine), auxiliary_sources_sha256={name:sha(value) for name,value in self.old.items()},
            numpy_version=np.__version__, flow_backend={'name':'fixture','sha256':'flow'}, material_backend={'name':'fixture','sha256':'material'})
        self.current = dict(deepcopy(self.prior), auxiliary_sources_sha256={name:sha(value) for name,value in self.new.items()})
        (self.parent / 'engine.py').write_bytes(self.engine)
        for name,payload in self.old.items():
            (self.parent/name).write_bytes(payload)
        config = dict(seed=37, width=512, height=256, dt_myr=1., snapshot_myr=2., duration_myr=1000.,
            enhanced_rifting=False, effective_subduction=dict(enabled=True, force_n_per_m=3e13,
                initiation={'enabled':True}, carrier_traction={'version':1,'enabled':True,'alpha':.1}),
            force_limit_rifting={'enabled':True,'search_policy':{'version':1,'mode':'bounded_axes','max_axes':10}})
        self.original = SimpleNamespace(t=72., steps=73, config=config, rng=np.random.default_rng(37),
            initial_arrays={'artwork':np.arange(12,dtype=np.float32),'owners':np.array([2,7],np.int32)},
            omega=np.arange(6,dtype=np.float64).reshape(2,3), plate_uid=np.array([2,7],np.int32),
            native_boundary_geometry={'segments_start':np.eye(3),'segments_end':np.roll(np.eye(3),1,axis=0),
                'contact_index':np.array([0,1,2],np.int32),'diagnostics':{'accepted_prior_geometry':True}},
            native_locator={'candidates':np.array([2,0,1],np.int32),'keys':np.arange(3,dtype=np.int64)},
            bn=np.eye(3), bmid=np.roll(np.eye(3),1,axis=0), bl=np.array([30.,40.,50.]),
            bcode=np.array([2,3,5],np.uint8), normal_speed=np.array([-2.,0.,3.]),
            material_surface={'vertices':np.eye(3),'faces':np.array([[0,1,2]],np.int32),
                'reference_area_km2':np.array([18.]),'area_km2':np.array([17.])},
            effective_subduction_initiation={'version':1,'activation_myr':25.,'epoch_myr':72.,
                'candidates':[{'id':2,'compression_myr':12.,'shortening_km':110.}],'births':[]},
            effective_subduction_carrier_traction={'version':1,'activation_myr':43.,'parameters':deepcopy(config['effective_subduction']['carrier_traction'])},
            force_rifting_version=1, force_rifting_state={'2':{'work_j':4e20,'opening_km':21.,'supported_elapsed_myr':5.}},
            force_rifting_policy={'activation_myr':43.,'epoch_myr':72.,'parameters':deepcopy(config['force_limit_rifting'])},
            backarc_systems=[{'id':1,'loaded_km':55.,'last_attempt_myr':70.}],
            rift_systems=[{'id':3,'_bonds':{(1,2),(2,3)}}], events=[{'type':'accepted-old','time_myr':71.}],
            event_keys={(i,'accepted-old') for i in range(12)})
        self.manifest = dict(run_id='logical-live-parent',title='Asein Lite',state='paused',can_resume=True,config=deepcopy(config),
            time_myr=72.,integration_time_myr=72.,checkpoint_time_myr=72.,next_output_myr=74.,duration_myr=1000.,
            frame_count=3,frames=[{'index':0,'time_myr':0.},{'index':1,'time_myr':70.},{'index':2,'time_myr':72.}],
            source_transition={'kind':'earlier_source_only','epoch_myr':64.,'inherited_frame_count':1},
            frame_sources=[{'run_id':'predecessor','inherited':True,'parent_frame_source':{'run_id':'older',
                'parent_frame_source':{'run_id':'original-world'}}}],**deepcopy(self.prior))
        for row in self.manifest['frames']:
            stem=self.parent/f"frame_{row['index']:04d}"
            np.savez_compressed(stem.with_suffix('.npz'),elevation=np.arange(12,dtype=np.float32),native_owner=np.array([2,7]))
            stem.with_suffix('.json').write_text(json.dumps(row))
        (self.parent/'inherited-parent').mkdir()
        (self.parent/'inherited-parent/manifest.json').write_bytes(b'{"run_id":"older"}\n')
        (self.parent/'checkpoint.source-boundary.npz').write_bytes(b'earlier immutable source proof')
        (self.parent/'initial.json').write_bytes(b'{"artwork":"exact original Asein"}\n')
        (self.parent/'config.json').write_text(json.dumps(config))
        self.publish_parent()
        self.preservation=self.root/'preservation.json';self.seal=self.root/'seal.json';self.source_review=self.root/'source-review.json'
        self.acceleration_review=self.root/'acceleration-review.json';self.authority=self.root/'authority.json'
        self.proofs=[]
        for role in ('regression','measured-equivalence'):
            path=self.root/(role+'.json');transition._write_json(path,{'role':role,'passed':True})
            self.proofs.append({'path':path.name,'sha256':transition._file_sha(path)})
        self.source_commit='4'*40
        self.authority_data=dict(kind=transition.AUTHORITY_KIND,version=1,parent_run_id=self.manifest['run_id'],
            source_commit=self.source_commit,authorized=True,request=transition.AUTHORITY_REQUEST,
            authorization_source='Human request in this chat',physical_policy_change_authorized=False,scope=transition.AUTHORITY_SCOPE)
        transition._write_json(self.authority,self.authority_data)
        self.current_context=transition._current_context
        item=patch.object(transition,'_current_context',side_effect=lambda approved:(deepcopy(self.current),self.engine,dict(self.new)))
        item.start();self.addCleanup(item.stop)
        item=patch.object(transition,'REVIEWED_APPROVAL',None);item.start();self.addCleanup(item.stop)
        self.approve()

    def publish_parent(self):
        checkpoint.write_checkpoint(self.parent/'checkpoint.npz',self.original,self.manifest,self.prior)
        (self.parent/'manifest.json').write_text(json.dumps(self.manifest))

    def approve(self, *, source_changes=None, review_changes=None, authority_changes=None):
        cp=transition._file_sha(self.parent/'checkpoint.npz')
        inventory=transition._files(self.parent)
        transition._write_json(self.preservation,dict(run_id=self.manifest['run_id'],checkpoint_sha256=cp,
            all_copied_files_exact=True,file_count=len(inventory),files=[{'path':name,'sha256':value} for name,value in inventory.items()]))
        transition._write_json(self.seal,self.current['auxiliary_sources_sha256'])
        authority=dict(self.authority_data,parent_checkpoint_sha256=cp,epoch_myr=self.original.t)
        authority.update(authority_changes or {});transition._write_json(self.authority,authority)
        changed={name:{'parent_sha256':sha(self.old[name]),'child_sha256':sha(self.new[name])} for name in transition.HELPERS}
        review=dict(kind=transition.ACCELERATION_REVIEW_KIND,version=1,ready=True,parent_run_id=self.manifest['run_id'],
            parent_checkpoint_sha256=cp,epoch_myr=self.original.t,changed_helpers=changed,
            runtime_source_seal_sha256=transition._file_sha(self.seal),captured_regression_passed=True,captured_equivalence_passed=True,
            geometry_outputs_exact=True,performance_measured=True,performance_improved=True,physical_laws_unchanged=True,
            physical_guards_unchanged=True,resolution_unchanged=True,validation_receipts=deepcopy(self.proofs))
        review.update(review_changes or {});transition._write_json(self.acceleration_review,review)
        source=dict(kind=transition.SOURCE_REVIEW_KIND,version=1,ready=True,parent_run_id=self.manifest['run_id'],
            parent_checkpoint_sha256=cp,epoch_myr=self.original.t,boundary_kind='healthy_accepted_step_pause',
            parent_preservation_sha256=transition._file_sha(self.preservation),runtime_source_seal_sha256=transition._file_sha(self.seal),
            parent_compatibility_sha256=sha(transition._json_bytes(self.prior)),child_compatibility_sha256=sha(transition._json_bytes(self.current)),
            changed_helpers=changed,added_helpers={},removed_helpers=[],helper_count=173,
            acceleration_review_sha256=transition._file_sha(self.acceleration_review),authority_sha256=transition._file_sha(self.authority),
            configuration_unchanged=True,state_import='entire typed state exact; cached geometry retained')
        source.update(source_changes or {});transition._write_json(self.source_review,source)
        self.approval=transition.TransitionApproval(self.manifest['run_id'],self.source_commit,float(self.original.t),cp,
            self.preservation,transition._file_sha(self.preservation),self.seal,transition._file_sha(self.seal),
            self.source_review,transition._file_sha(self.source_review),self.acceleration_review,transition._file_sha(self.acceleration_review),
            self.authority,transition._file_sha(self.authority),tuple((name,sha(self.old[name]),sha(self.new[name])) for name in transition.HELPERS),True)
        transition.REVIEWED_APPROVAL=self.approval

    def plan(self):
        return transition.plan_transition(self.parent)

    def test_dynamic_boundary_full_state_rng_clocks_cache_scheduler_and_recursive_frames_exact(self):
        before=transition._files(self.parent)
        sentinels=[patch.object(native_engine.Simulation,name,side_effect=AssertionError(name))
            for name in ('__init__','step','_boundaries','_rasterize','_forces')]
        for item in sentinels:item.start();self.addCleanup(item.stop)
        child,receipt=transition.create_transition(self.plan(),output_root=self.root,run_id='child')
        restored,saved=checkpoint.read_checkpoint(child/'checkpoint.npz',self.current,native_engine.Simulation)
        self.assertEqual(transition._fingerprint(vars(restored)),transition._fingerprint(vars(self.original)))
        self.assertEqual(receipt['epoch_myr'],72.);self.assertEqual(saved['next_output_myr'],74.)
        self.assertEqual(saved['frames'],self.manifest['frames']);self.assertEqual(saved['config'],self.manifest['config'])
        self.assertEqual(saved['frame_sources'][0]['parent_frame_source'],self.manifest['frame_sources'][0])
        self.assertEqual(saved['frame_sources'][0]['run_id'],self.manifest['run_id'])
        self.assertEqual(transition._files(self.parent),before);self.assertEqual(transition._files(child/'inherited-parent'),before)
        self.assertEqual((child/'checkpoint.source-boundary.npz').read_bytes(),(child/'checkpoint.npz').read_bytes())
        self.assertFalse(receipt['model_constructed']);self.assertFalse(receipt['model_stepped'])
        self.assertFalse(receipt['geometry_rebuilt_at_import']);self.assertEqual(receipt['physical_time_advanced_myr'],0.)
        self.assertEqual(json.loads((child/'manifest.json').read_text()),saved)

    def test_disabled_unready_unlocked_mutable_or_wrong_helper_approval_rejected(self):
        with patch.object(transition,'REVIEWED_APPROVAL',None),self.assertRaisesRegex(ValueError,'not sealed'):self.plan()
        with patch.object(transition,'REVIEWED_APPROVAL',replace(self.approval,review_ready=False)),self.assertRaises(ValueError):self.plan()
        with self.assertRaises(ValueError):transition.plan_transition(self.parent,approval=replace(self.approval))
        with self.assertRaises(FrozenInstanceError):self.approval.review_ready=False
        for change in ({'parent_epoch_myr':float('nan')},{'helper_delta':list(self.approval.helper_delta)},
                {'helper_delta':self.approval.helper_delta[:1]},{'helper_delta':tuple(reversed(self.approval.helper_delta))}):
            with self.subTest(change=change),patch.object(transition,'REVIEWED_APPROVAL',replace(self.approval,**change)),self.assertRaises(ValueError):self.plan()

    def test_authority_exact_parent_pause_and_computational_scope_required(self):
        for change in ({'authorized':False},{'request':'Observe only'},{'source_commit':'0'*40},
                {'parent_run_id':'other'},{'epoch_myr':69.},{'parent_checkpoint_sha256':'0'*64},
                {'physical_policy_change_authorized':True},{'scope':'lower resolution'}):
            self.approve(authority_changes=change)
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'authority'):self.plan()

    def test_measured_equivalence_and_immutable_distinct_proofs_required(self):
        for change in ({'ready':False},{'captured_equivalence_passed':False},{'performance_improved':False},
                {'geometry_outputs_exact':False},{'physical_guards_unchanged':False},{'physical_laws_unchanged':False},
                {'resolution_unchanged':False},{'validation_receipts':[]},{'validation_receipts':[self.proofs[0]]*2}):
            self.approve(review_changes=change)
            with self.subTest(change=change),self.assertRaises(ValueError):self.plan()
        self.approve();(self.root/self.proofs[0]['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'validation evidence'):self.plan()

    def test_source_delta_scope_backend_and_full_import_closure_exact(self):
        baseline=deepcopy(self.new);compat=deepcopy(self.current)
        for kind in ('one_unchanged','third_helper','extra_helper','backend','engine'):
            self.new=deepcopy(baseline);self.current=deepcopy(compat)
            if kind=='one_unchanged':self.new[transition.HELPERS[0]]=self.old[transition.HELPERS[0]]
            if kind=='third_helper':self.new['helper_000.py']=b'VALUE = 9\n'
            if kind=='extra_helper':self.new['extra.py']=b'VALUE = 9\n'
            self.current['auxiliary_sources_sha256']={name:sha(value) for name,value in self.new.items()}
            if kind=='backend':self.current['material_backend']={'name':'other'}
            if kind=='engine':self.current['engine_sha256']='0'*64
            self.approve()
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.plan()
        self.new=baseline;self.current=compat;self.approve()
        (self.parent/'helper_000.py').write_text('VALUE = 999\n');self.approve()
        with self.assertRaisesRegex(ValueError,'helper source'):self.plan()

    def test_generation_scheduler_config_frame_and_lineage_tampering_rejected(self):
        baseline=deepcopy(self.manifest)
        for change in ({'state':'running'},{'state':'pausing'},{'title':'Other model'},{'error':'stopped'},
                {'checkpoint_time_myr':71.},{'integration_time_myr':73.},{'next_output_myr':72.},
                {'frame_count':2},{'frame_sources':[]},{'config':dict(self.original.config,width=256)}):
            external=dict(deepcopy(baseline),**change)
            (self.parent/'manifest.json').write_text(json.dumps(external));self.approve()
            with self.subTest(change=change),self.assertRaises(ValueError):self.plan()
        (self.parent/'manifest.json').write_text(json.dumps(baseline));self.approve()
        row=self.parent/'frame_0002.json';row.write_text('{"index":2,"time_myr":69}')
        self.approve()
        with self.assertRaisesRegex(ValueError,'inherited frame'):self.plan()

    def test_import_cannot_reset_rng_cache_policy_clocks_or_controls(self):
        def corrupt(kind):
            def mutate(original):
                child=deepcopy(original)
                if kind=='rng':child.rng.random()
                if kind=='cache':child.native_locator['candidates'][0]=0
                if kind=='clock':child.effective_subduction_initiation['candidates'][0]['compression_myr']=0.
                if kind=='config':child.config['effective_subduction']['force_n_per_m']=5e12
                return child
            return mutate
        for kind in ('rng','cache','clock','config'):
            with self.subTest(kind=kind),patch.object(transition,'_derive',side_effect=corrupt(kind)),self.assertRaisesRegex(ValueError,'typed array'):self.plan()

    def test_parent_race_mutable_plan_and_failed_publication_never_publish_child(self):
        plan=self.plan();plan.derived.native_locator['keys'][0]=999
        with self.assertRaisesRegex(ValueError,'mutable plan'):transition.create_transition(plan,output_root=self.root,run_id='altered')
        self.assertFalse((self.root/'altered').exists())
        plan=self.plan();(self.parent/'unexpected.log').write_text('parent changed')
        with self.assertRaises(ValueError):transition.create_transition(plan,output_root=self.root,run_id='raced')
        self.assertFalse((self.root/'raced').exists())
        self.approve();plan=self.plan()
        verify=transition._verify_child_artifacts
        def sabotage(plan, child, manifest, receipt):
            (child/'checkpoint.source-boundary.npz').write_bytes(b'tampered')
            return verify(plan,child,manifest,receipt)
        with patch.object(transition,'_verify_child_artifacts',side_effect=sabotage),self.assertRaisesRegex(ValueError,'child artifact'):
            transition.create_transition(plan,output_root=self.root,run_id='failed')
        self.assertFalse((self.root/'failed').exists())
        self.assertTrue(list(self.root.glob('.src-*')))

    def test_deep_windows_history_short_stage_exact_copy_and_public_path_refusal(self):
        run_id='20261002-214356-15f8c4'
        relative=Path(*(['inherited-parent']*4))/'profiling'/'4b6b0354abdf4bbdb90e87baa966f805.jsonl'
        old_stage=self.root/('.'+run_id+'.source-pending-'+'a'*32)
        padding=max(0,260-len(str(old_stage/'inherited-parent'/relative)))
        relative=relative.with_name('x'*padding+relative.name)
        source=self.parent/relative
        source.parent.mkdir(parents=True)
        source.write_bytes(b'Exact archived profiler bytes, with all nested history retained.\n')
        self.assertGreaterEqual(len(str(old_stage/'inherited-parent'/relative)),260)
        self.assertLess(len(str(source)),260)
        self.approve();before=transition._files(self.parent);plan=self.plan()
        copytree=transition.shutil.copytree
        destinations=[]
        def bounded_copy(src,dst,*args,**kwargs):
            # Model the Windows ordinary-path copy limitation even on other OSes.
            if Path(src)==self.parent:
                destinations.append(Path(dst))
                for name in before:
                    target=Path(dst)/name
                    self.assertLess(len(str(target)),260)
                    self.assertLess(len(str(target.parent)),248)
            return copytree(src,dst,*args,**kwargs)
        with patch.object(transition,'_WINDOWS_PATHS',True),patch.object(transition.shutil,'copytree',side_effect=bounded_copy):
            child,receipt=transition.create_transition(plan,output_root=self.root,run_id=run_id)
        self.assertEqual(len(destinations),1)
        self.assertRegex(destinations[0].parent.name,r'^\.src-[0-9a-f]{12}$')
        self.assertFalse(destinations[0].parent.exists())
        self.assertEqual(child,self.root/run_id)
        self.assertEqual(transition._files(child/'inherited-parent'),before)
        self.assertEqual(transition._files(self.parent),before)
        self.assertEqual((child/'inherited-parent'/relative).read_bytes(),source.read_bytes())
        self.assertEqual(json.loads((child/'manifest.json').read_text())['run_id'],run_id)
        restored,_=checkpoint.read_checkpoint(child/'checkpoint.npz',self.current,native_engine.Simulation)
        self.assertEqual(transition._fingerprint(vars(restored)),transition._fingerprint(vars(self.original)))
        self.assertEqual(receipt['physical_time_advanced_myr'],0.)
        self.assertFalse(list(self.root.glob('.src-*')))
        # The allowed 80-character identity would make the final tree too long.
        # Refuse before copying or publishing, even though the short stage fits.
        unsupported_id='r'*80
        self.assertGreaterEqual(len(str(self.root/unsupported_id/'inherited-parent'/relative)),260)
        with patch.object(transition,'_WINDOWS_PATHS',True),patch.object(transition.shutil,'copytree') as copying, \
                self.assertRaisesRegex(ValueError,'Windows path budget'):
            transition.create_transition(plan,output_root=self.root,run_id=unsupported_id)
        copying.assert_not_called()
        self.assertFalse((self.root/unsupported_id).exists())
        self.assertFalse(list(self.root.glob('.src-*')))
        # Atomic checkpoint serialization has a longer temporary basename.
        long_stage=self.root/('s'*max(1,212-len(str(self.root))-1))
        short_public=self.root/'short'
        child_names={'checkpoint.npz','checkpoint.source-boundary.npz'}
        with patch.object(transition,'_WINDOWS_PATHS',True):
            transition._check_publication_paths(self.parent,long_stage,short_public,{'a':'hash'},child_names)
            with self.assertRaisesRegex(ValueError,'Windows path budget'):
                transition._check_publication_paths(self.parent,long_stage,short_public,{'a':'hash'},
                    child_names|{'checkpoint.npz.'+'0'*32+'.tmp'})

    def test_source_review_binding_and_current_bytes_seal(self):
        for change in ({'epoch_myr':69.},{'boundary_kind':'durable_stopped'},{'authority_sha256':'0'*64},
                {'runtime_source_seal_sha256':'0'*64},{'parent_preservation_sha256':'0'*64},
                {'state_import':'rebuild caches'},{'configuration_unchanged':False}):
            self.approve(source_changes=change)
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'source delta review'):self.plan()
        self.approve();runtime=self.root/'runtime';runtime.mkdir();(runtime/'tectonics.py').write_bytes(self.engine)
        for name,value in self.new.items():(runtime/name).write_bytes(value)
        with patch.object(transition,'ROOT',runtime),patch.object(transition.server.SimulationManager,'compatibility',return_value=deepcopy(self.current)), \
                patch.object(transition.server,'ENGINE_SOURCE',self.engine),patch.object(transition.server,'AUXILIARY_SOURCES',dict(self.new)):
            self.assertEqual(self.current_context(self.approval),(self.current,self.engine,self.new))
            (runtime/transition.HELPERS[0]).write_text('VALUE = 99\n')
            with self.assertRaisesRegex(ValueError,'runtime files changed'):self.current_context(self.approval)


if __name__ == '__main__':
    unittest.main()
