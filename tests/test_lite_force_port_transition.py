"""Source-only imports retain enabled policy clocks and accepted cached geometry.

The fixtures use constructorless typed checkpoints and a complete 173-helper
AST closure. No simulation interval or production server is started here.
"""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import checkpoint
import lite_force_port_transition as transition
import native_engine


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


class ForcePortSourceTransitionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.parent = self.root / 'current-feedback-parent'
        self.parent.mkdir()
        self.old = {transition.HELPER: b'VALUE = 1\n', 'boundary_consumer.py': b'import balance_force_ledger\n'}
        self.old.update({f'helper_{index:03d}.py': b'VALUE = 3\n' for index in range(171)})
        self.new = dict(self.old, **{transition.HELPER: b'VALUE = 2\n'})
        self.engine = ('import boundary_consumer\n' + '\n'.join('import '+name[:-3] for name in self.old if name.startswith('helper_'))+'\n').encode()
        self.prior = dict(engine_sha256=sha(self.engine), auxiliary_sources_sha256={name:sha(value) for name,value in self.old.items()},
            numpy_version=np.__version__, flow_backend={'name':'fixture','sha256':'flow'}, material_backend={'name':'fixture','sha256':'material'})
        self.current = dict(deepcopy(self.prior), auxiliary_sources_sha256={name:sha(value) for name,value in self.new.items()})
        (self.parent / 'engine.py').write_bytes(self.engine)
        for name,payload in self.old.items():
            (self.parent/name).write_bytes(payload)
        config = dict(seed=37, width=512, height=256, dt_myr=1., snapshot_myr=2., duration_myr=1000.,
            enhanced_rifting=False, effective_subduction=dict(enabled=True, force_n_per_m=3e13,
                initiation={'enabled':True}, carrier_traction={'version':1,'enabled':True,'alpha':.1}),
            force_limit_rifting={'enabled':True,'inherited_weakness':True,'search_policy':{'version':1,'mode':'bounded_axes','max_axes':10}})
        self.original = SimpleNamespace(t=64., steps=55, config=config, rng=np.random.default_rng(37),
            initial_arrays={'artwork':np.arange(12,dtype=np.float32),'owners':np.array([2,7],np.int32)},
            omega=np.arange(6,dtype=np.float64).reshape(2,3), plate_uid=np.array([2,7],np.int32),
            native_boundary_geometry={'segments_start':np.eye(3),'segments_end':np.roll(np.eye(3),1,axis=0),
                'contact_index':np.array([0,1,2],np.int32),'diagnostics':{'fallback_edges':2,'accepted_prior_geometry':True}},
            bn=np.eye(3), bmid=np.roll(np.eye(3),1,axis=0), bl=np.array([30.,40.,50.]),
            bcode=np.array([2,3,5],np.uint8), normal_speed=np.array([-2.,0.,3.]),
            material_surface={'vertices':np.eye(3),'faces':np.array([[0,1,2]],np.int32),
                'reference_area_km2':np.array([18.]),'area_km2':np.array([17.])},
            effective_subduction_initiation={'version':1,'activation_myr':25.,'epoch_myr':64.,
                'candidates':[{'id':2,'compression_myr':12.,'shortening_km':110.}],'births':[]},
            effective_subduction_carrier_traction={'version':1,'activation_myr':43.,'parameters':deepcopy(config['effective_subduction']['carrier_traction'])},
            force_rifting_version=1,force_rifting_state={'2':{'work_j':4e20,'opening_km':21.,'supported_elapsed_myr':5.}},
            force_rifting_policy={'activation_myr':43.,'epoch_myr':64.,'parameters':deepcopy(config['force_limit_rifting'])},
            force_rifting_diagnostics={'checks':12,'commits':0,'refusals':[]},
            backarc_systems=[{'id':1,'loaded_km':65.,'last_attempt_myr':62.}], rift_systems=[{'id':3,'_bonds':{(1,2),(2,3)}}],
            trench_systems=[{'id':4,'intake_km':33.}], events=[{'type':'accepted-old','time_myr':63.}],event_keys={(i,'accepted-old') for i in range(12)})
        self.manifest = dict(run_id=self.parent.name,title='Asein Lite',state='interrupted',can_resume=True,config=deepcopy(config),
            time_myr=64.,integration_time_myr=64.,checkpoint_time_myr=64.,next_output_myr=66.,duration_myr=1000.,
            frame_count=3,frames=[{'index':0,'time_myr':0.},{'index':1,'time_myr':62.},{'index':2,'time_myr':64.}],
            source_transition={'kind':'lite_mechanical_feedback_source_transition','epoch_myr':43.,'inherited_frame_count':1},
            frame_sources=[{'run_id':'feedback-predecessor','inherited':True,'parent_frame_source':{'run_id':'initiation-predecessor',
                'parent_frame_source':{'run_id':'original-world'}}}],**deepcopy(self.prior))
        for row in self.manifest['frames']:
            stem=self.parent/f"frame_{row['index']:04d}"
            np.savez_compressed(stem.with_suffix('.npz'),elevation=np.arange(12,dtype=np.float32),native_owner=np.array([2,7]))
            stem.with_suffix('.json').write_text(json.dumps(row))
        (self.parent/'inherited-parent').mkdir()
        (self.parent/'inherited-parent/manifest.json').write_bytes(b'{"run_id":"feedback-predecessor"}\n')
        (self.parent/'checkpoint.policy-boundary.npz').write_bytes(b'earlier immutable policy proof')
        (self.parent/'initial.json').write_bytes(b'{"artwork":"exact original Asein"}\n')
        (self.parent/'config.json').write_text(json.dumps(config))
        self.publish_parent()
        self.preservation=self.root/'preservation.json';self.seal=self.root/'seal.json';self.source_review=self.root/'source-review.json'
        self.repair_review=self.root/'repair-review.json';self.authority=self.root/'authority.json'
        self.stop_evidence=self.root/'stop-evidence.json'
        self.proofs=[]
        for role in ('captured-regression','future-interval'):
            path=self.root/(role+'.json')
            transition._write_json(path,{'role':role,'passed':True})
            self.proofs.append({'path':path.name,'sha256':transition._file_sha(path)})
        self.authority_data=dict(kind=transition.AUTHORITY_KIND,version=1,parent_run_id=self.parent.name,
            source_commit=transition.PARENT_SOURCE_COMMIT,parent_checkpoint_sha256=transition._file_sha(self.parent/'checkpoint.npz'),
            authorization_source='Human-user heartbeat instructions in this chat',authorized=True,new_physical_policy_authorized=False,
            instruction='If unexpectedly stopped, preserve manifest/source closure/checkpoint/frames/error; diagnose implementation bugs '
                'within authorized laws, add meaningful regression coverage and relevant checks, and resume only through truthful '
                'supported explicit source transitions or preserved fresh audited histories. Never weaken guards, forge compatibility '
                'or repeatedly retry an unfixed failure.',
            scope='Repair only the numerical finite-force ledger representation of existing fractional native support; preserve forces, '
                'coefficients, resistance, physical laws, material/ownership, guards, all inherited typed state, history and scheduling. '
                'Validate before truthful explicit source-only stopped-checkpoint continuation.')
        transition._write_json(self.authority,self.authority_data)
        self.saved_context=transition._current_context
        self.patches=[patch.object(transition,'PARENT_HELPER_SHA256',sha(self.old[transition.HELPER])),
            patch.object(transition,'CHILD_HELPER_SHA256',sha(self.new[transition.HELPER])),
            patch.object(transition,'_current_context',side_effect=lambda approved:(deepcopy(self.current),self.engine,dict(self.new))),
            patch.object(transition,'REVIEWED_APPROVAL',None)]
        for item in self.patches:
            item.start();self.addCleanup(item.stop)
        self.approve()

    def publish_parent(self):
        checkpoint.write_checkpoint(self.parent/'checkpoint.npz',self.original,self.manifest,self.prior)
        self.external=dict(deepcopy(self.manifest),state='error',
            error='UnsupportedForceLedger: Plate 9 has a force port but no owned native controls.',next_output_myr=56.)
        (self.parent/'manifest.json').write_text(json.dumps(self.external))


    def approve(self, *, source_changes=None, repair_changes=None):
        transition._write_json(self.stop_evidence,self.external)
        self.authority_data['parent_checkpoint_sha256']=transition._file_sha(self.parent/'checkpoint.npz')
        transition._write_json(self.authority,self.authority_data)
        transition._write_json(self.preservation,dict(kind='asein_lite_complete_failed_history_preservation',run_id=self.parent.name,
            all_copied_files_exact=True,checkpoint_sha256=transition._file_sha(self.parent/'checkpoint.npz'),
            checkpoint_time_myr=64.,public_frame_myr=self.external['time_myr'],error=self.external['error'],
            file_count=len(transition._files(self.parent)),files=[{'path':name,'sha256':value} for name,value in transition._files(self.parent).items()]))
        transition._write_json(self.seal,self.current['auxiliary_sources_sha256'])
        self.repair_data=dict(kind=transition.REPAIR_REVIEW_KIND,version=1,ready=True,parent_run_id=self.parent.name,
            parent_checkpoint_sha256=transition._file_sha(self.parent/'checkpoint.npz'),helper=transition.HELPER,
            parent_sha256=transition.PARENT_HELPER_SHA256,child_sha256=transition.CHILD_HELPER_SHA256,
            physical_guards_unchanged=True,physical_laws_unchanged=True,virtual_work_preserved=True,
            stop_evidence_sha256=transition._file_sha(self.stop_evidence),
            captured_regression_passed=True,future_interval_passed=True,validation_receipts=deepcopy(self.proofs))
        self.repair_data.update(repair_changes or {})
        transition._write_json(self.repair_review,self.repair_data)
        self.source_data=dict(kind=transition.SOURCE_REVIEW_KIND,version=1,ready=True,parent_run_id=self.parent.name,
            parent_checkpoint_sha256=transition._file_sha(self.parent/'checkpoint.npz'),
            parent_compatibility_sha256=sha(transition._json_bytes(self.prior)),child_compatibility_sha256=sha(transition._json_bytes(self.current)),
            changed_helpers={name:{'parent_sha256':sha(self.old[name]),'child_sha256':sha(self.new[name])}
                for name in self.old.keys() & self.new.keys() if self.old[name]!=self.new[name]},
            added_helpers={name:sha(self.new[name]) for name in self.new.keys()-self.old.keys()},removed_helpers=sorted(self.old.keys()-self.new.keys()),
            helper_count=173,repair_review_sha256=transition._file_sha(self.repair_review),authority_sha256=transition._file_sha(self.authority),
            boundary_kind='durable_stopped',stop_evidence_sha256=transition._file_sha(self.stop_evidence),
            configuration_unchanged=True,state_import='entire typed state exact; cached geometry retained')
        self.source_data.update(source_changes or {})
        transition._write_json(self.source_review,self.source_data)
        self.approval=transition.TransitionApproval(self.parent.name,transition._file_sha(self.parent/'checkpoint.npz'),
            self.preservation,transition._file_sha(self.preservation),self.seal,transition._file_sha(self.seal),
            self.source_review,transition._file_sha(self.source_review),self.repair_review,transition._file_sha(self.repair_review),
            self.authority,transition._file_sha(self.authority),self.stop_evidence,transition._file_sha(self.stop_evidence),review_ready=True)
        transition.REVIEWED_APPROVAL=self.approval

    def plan(self):
        return transition.plan_transition(self.parent)

    def test_complete_typed_state_cache_rng_policies_initial_and_recursive_frames_exact(self):
        before=transition._files(self.parent)
        with patch.object(native_engine.Simulation,'__init__',side_effect=AssertionError('constructed')), \
                patch.object(native_engine.Simulation,'_boundaries',side_effect=AssertionError('cache rebuilt')):
            child,receipt=transition.create_transition(self.plan(),run_id='child')
        restored,saved=checkpoint.read_checkpoint(child/'checkpoint.npz',self.current,native_engine.Simulation)
        self.assertEqual(transition._fingerprint(vars(restored)),transition._fingerprint(vars(self.original)))
        self.assertEqual(saved['config'],self.manifest['config']);self.assertEqual(saved['next_output_myr'],66.)
        self.assertEqual(saved['frames'],self.manifest['frames']);self.assertEqual(transition._files(self.parent),before)
        self.assertEqual(transition._files(child/'inherited-parent'),before)
        self.assertEqual(saved['frame_sources'][0]['run_id'],self.parent.name)
        self.assertEqual(saved['frame_sources'][0]['parent_frame_source'],self.manifest['frame_sources'][0])
        self.assertIsNone(saved['frame_sources'][2]['parent_frame_source'])
        for name in ('initial.json','config.json'):
            self.assertEqual((child/name).read_bytes(),(self.parent/name).read_bytes())
        for index in range(3):
            for suffix in ('.npz','.json'):
                name=f'frame_{index:04d}{suffix}';self.assertEqual((child/name).read_bytes(),(self.parent/name).read_bytes())
        self.assertEqual((child/'checkpoint.source-boundary.npz').read_bytes(),(child/'checkpoint.npz').read_bytes())
        self.assertFalse(receipt['geometry_rebuilt_at_import']);self.assertFalse(receipt['physical_policies_added'])
        self.assertTrue(receipt['cached_boundary_geometry_unchanged']);self.assertEqual(receipt['physical_time_advanced_myr'],0.)
        self.assertIn('predecessor cache',receipt['first_source_solve_geometry'])
        self.assertEqual(json.loads((child/'manifest.json').read_text()),saved)
        self.assertEqual(saved['state'],'paused');self.assertNotIn('error',saved)
        self.assertEqual(json.loads((child/'stop-evidence.json').read_text()),self.external)
        self.assertEqual(receipt['predecessor_embedded_state'],'interrupted')
        self.assertEqual(receipt['predecessor_external_state'],'error')
        self.assertEqual(saved['durable_source_recovery']['error_evidence_sha256'],self.approval.stop_evidence_sha256)
        self.assertEqual(json.loads((child/'inherited-parent/manifest.json').read_text())['state'],'error')

    def test_new_locked_ready_approval_and_exact_authority_are_required(self):
        for approval in (None,replace(self.approval,review_ready=False)):
            with patch.object(transition,'REVIEWED_APPROVAL',approval),self.assertRaisesRegex(ValueError,'not sealed'):
                self.plan()
        with self.assertRaisesRegex(ValueError,'not sealed'):
            transition.plan_transition(self.parent,approval=replace(self.approval))
        with self.assertRaises(FrozenInstanceError):
            self.approval.review_ready=False
        initial_authority=deepcopy(self.authority_data)
        for change in ({'authorized':False},{'instruction':'Observe only'},{'parent_run_id':'other-run'},
                {'source_commit':'other-source'},{'scope':'new physical policy'},{'new_physical_policy_authorized':True}):
            self.authority_data=dict(initial_authority,**change);self.approve()
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'authority'):
                self.plan()

    def test_preserved_copy_directory_name_does_not_replace_sealed_logical_run_identity(self):
        original_id=self.approval.parent_run_id
        archive=self.root/'immutable-preservation';archive.mkdir()
        relocated=archive/'run';self.parent.rename(relocated);self.parent=relocated
        child,receipt=transition.create_transition(self.plan(),output_root=self.root,run_id='preserved-source-child')
        saved=json.loads((child/'manifest.json').read_text())
        self.assertEqual(receipt['parent_run_id'],original_id)
        self.assertEqual(saved['branch_origin']['run_id'],original_id)
        self.assertEqual(saved['source_transition']['parent_run_id'],original_id)
        self.assertTrue(all(row['run_id']==original_id for row in saved['frame_sources']))
        self.assertEqual(json.loads((child/'inherited-parent/manifest.json').read_text())['run_id'],original_id)

    def test_missing_failed_duplicate_or_tampered_new_proofs_rejected(self):
        for changes in ({'ready':False},{'future_interval_passed':False},{'physical_guards_unchanged':False},
                {'physical_laws_unchanged':False},{'virtual_work_preserved':False},{'stop_evidence_sha256':'0'*64},
                {'captured_regression_passed':False},{'validation_receipts':[]},{'validation_receipts':[self.proofs[0]]*2}):
            self.approve(repair_changes=changes)
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                self.plan()
        self.approve();(self.root/self.proofs[0]['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'validation evidence'):
            self.plan()

    def test_same_source_additional_helper_backend_and_engine_changes_rejected(self):
        original_new=deepcopy(self.new);original_current=deepcopy(self.current)
        for change in ('same','third','added','backend','engine'):
            self.new=deepcopy(original_new);self.current=deepcopy(original_current)
            if change=='same':self.new[transition.HELPER]=self.old[transition.HELPER]
            if change=='third':self.new['helper_000.py']=b'VALUE = 9\n'
            if change=='added':self.new['other.py']=b'VALUE = 1\n'
            self.current['auxiliary_sources_sha256']={name:sha(value) for name,value in self.new.items()}
            if change=='backend':self.current['flow_backend']={'name':'changed'}
            if change=='engine':self.current['engine_sha256']='0'*64
            self.approve()
            with self.subTest(change=change),self.assertRaises(ValueError):self.plan()

    def test_stopped_generation_scheduler_config_history_and_sealed_error_rejected(self):
        original=deepcopy(self.manifest)
        for changes in ({'state':'running'},{'state':'paused'},{'error':'forged accepted error'},
                {'checkpoint_time_myr':63.},{'integration_time_myr':65.},{'time_myr':65.},
                {'next_output_myr':64.},{'config':dict(self.original.config,dt_myr=2.)},{'frame_count':2}):
            self.manifest=dict(deepcopy(original),**changes)
            self.publish_parent();self.approve()
            with self.subTest(changes=changes),self.assertRaises(ValueError):self.plan()
        self.manifest=deepcopy(original);self.publish_parent();self.approve()
        # The stale external 56 clock remains evidence; accepted embedded 66
        # is imported and is never silently replaced by a guessed cadence.
        self.assertEqual(self.plan().receipt['next_output_myr'],66.)
        self.assertEqual(self.plan().receipt['predecessor_external_next_output_myr'],56.)
        for changes in ({'state':'paused'},{'error':None},{'integration_time_myr':65.},
                {'checkpoint_time_myr':63.},{'time_myr':65.},{'next_output_myr':58.}):
            self.publish_parent();self.approve()
            bad=dict(self.external,**changes)
            (self.parent/'manifest.json').write_text(json.dumps(bad))
            # Reseal copied bytes to exercise generation validation rather
            # than merely the preserved-file mismatch.
            transition._write_json(self.preservation,dict(kind='asein_lite_complete_failed_history_preservation',
                run_id=self.parent.name,all_copied_files_exact=True,checkpoint_sha256=transition._file_sha(self.parent/'checkpoint.npz'),
                checkpoint_time_myr=64.,public_frame_myr=64.,error=self.external['error'],
                file_count=len(transition._files(self.parent)),
                files=[{'path':name,'sha256':value} for name,value in transition._files(self.parent).items()]))
            self.approval=replace(self.approval,parent_preservation_sha256=transition._file_sha(self.preservation))
            transition.REVIEWED_APPROVAL=self.approval
            with self.subTest(external=changes),self.assertRaises(ValueError):self.plan()
        self.publish_parent();self.approve();transition._write_json(self.stop_evidence,dict(self.external,error='different trial failure'))
        with self.assertRaisesRegex(ValueError,'stopped status evidence'):self.plan()


    def test_rng_config_cached_geometry_and_existing_policy_mutations_are_rejected(self):
        def mutation(name):
            def alter(original):
                state=deepcopy(original)
                if name=='rng':state.rng.random()
                if name=='cache':state.bl[0]+=1.
                if name=='config':state.config['effective_subduction']['force_n_per_m']=5e13
                if name=='clock':state.force_rifting_policy['epoch_myr']=65.
                if name=='candidate':state.effective_subduction_initiation['candidates'][0]['compression_myr']=0.
                return state
            return alter
        for name in ('rng','cache','config','clock','candidate'):
            with self.subTest(name=name),patch.object(transition,'_derive',side_effect=mutation(name)),self.assertRaisesRegex(ValueError,'typed array'):
                self.plan()

    def test_parent_checkpoint_frame_closure_and_lineage_tampering_rejected(self):
        cp=(self.parent/'checkpoint.npz').read_bytes()
        (self.parent/'checkpoint.npz').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'checkpoint'):self.plan()
        (self.parent/'checkpoint.npz').write_bytes(cp)
        (self.parent/'helper_000.py').write_bytes(b'VALUE = 10\n');self.approve()
        with self.assertRaisesRegex(ValueError,'helper source'):self.plan()
        (self.parent/'helper_000.py').write_bytes(self.old['helper_000.py'])
        self.old['unreachable.py']=b'VALUE = 1\n';self.new['unreachable.py']=self.old['unreachable.py']
        (self.parent/'unreachable.py').write_bytes(self.old['unreachable.py'])
        self.prior['auxiliary_sources_sha256']['unreachable.py']=sha(self.old['unreachable.py'])
        self.publish_parent();self.approve()
        with self.assertRaisesRegex(ValueError,'173-helper'):self.plan()
        # Equal closure size cannot disguise a missing reachable source or an
        # unrelated helper substituted into its compatibility record.
        self.prior['auxiliary_sources_sha256'].pop('helper_170.py')
        self.old.pop('helper_170.py');self.new.pop('helper_170.py')
        self.current['auxiliary_sources_sha256']={name:sha(value) for name,value in self.new.items()}
        self.publish_parent();self.approve()
        with self.assertRaisesRegex(ValueError,'complete saved import closure'):self.plan()

    def test_source_review_binding_and_current_source_seal_reject_tamper(self):
        for changes in ({'repair_review_sha256':'0'*64},{'authority_sha256':'0'*64},{'configuration_unchanged':False},
                {'boundary_kind':'paused'},{'stop_evidence_sha256':'0'*64},
                {'state_import':'rebuild boundary caches'},{'helper_count':172},{'kind':'old feature approval'}):
            self.approve(source_changes=changes)
            with self.subTest(changes=changes),self.assertRaisesRegex(ValueError,'source delta review'):self.plan()
        self.approve();runtime=self.root/'runtime';runtime.mkdir();(runtime/'tectonics.py').write_bytes(self.engine)
        for name,value in self.new.items():(runtime/name).write_bytes(value)
        with patch.object(transition,'ROOT',runtime),patch.object(transition.server.SimulationManager,'compatibility',return_value=self.current), \
                patch.object(transition.server,'ENGINE_SOURCE',self.engine),patch.object(transition.server,'AUXILIARY_SOURCES',self.new):
            self.assertEqual(self.saved_context(self.approval),(self.current,self.engine,self.new))
            (runtime/transition.HELPER).write_bytes(b'VALUE = 99\n')
            with self.assertRaisesRegex(ValueError,'after local source capture'):self.saved_context(self.approval)

    def test_mutable_plan_and_after_plan_changes_fail_without_publication(self):
        for mutate in ('cache','manifest','receipt','helper'):
            plan=self.plan()
            if mutate=='cache':plan.derived.bn[0,0]=4.
            if mutate=='manifest':plan.manifest['next_output_myr']=68.
            if mutate=='receipt':plan.receipt['entire_typed_state_exact']=False
            if mutate=='helper':plan.helper_sources[transition.HELPER]=b'VALUE = 99\n'
            with self.subTest(mutate=mutate),self.assertRaisesRegex(ValueError,'mutable plan'):
                transition.create_transition(plan,run_id='mutated-'+mutate)
            self.assertFalse((self.root/('mutated-'+mutate)).exists())
        plan=self.plan();(self.parent/'frame_0002.json').write_text('{}')
        with self.assertRaises(ValueError):transition.create_transition(plan,run_id='changed-parent')
        self.assertFalse((self.root/'changed-parent').exists())

    def test_readback_failure_and_nested_child_never_publish_or_touch_parent(self):
        before=transition._files(self.parent);plan=self.plan()
        real=checkpoint.read_checkpoint
        def read(path,*args,**kwargs):
            if Path(path).parent.name=='unpublished':raise ValueError('injected readback failure')
            return real(path,*args,**kwargs)
        with patch.object(checkpoint,'read_checkpoint',side_effect=read),self.assertRaisesRegex(ValueError,'injected'):
            transition.create_transition(plan,run_id='unpublished')
        self.assertFalse((self.root/'unpublished/manifest.json').exists());self.assertEqual(transition._files(self.parent),before)
        with self.assertRaisesRegex(ValueError,'disjoint'):transition.create_transition(plan,output_root=self.parent,run_id='nested')
        self.assertFalse((self.parent/'nested').exists())

    def test_corrupt_copied_and_written_child_artifacts_never_publish(self):
        before=transition._files(self.parent)
        original_copy=transition.shutil.copy2
        original_write=Path.write_bytes
        names=('engine.py',transition.HELPER,'initial.json','config.json',
            'parent-preservation.json','runtime-source-seal.json','source-delta-review.json',
            'repair-review.json','application-authority.json','stop-evidence.json','checkpoint.source-boundary.npz')
        for index,name in enumerate(names):
            child=self.root/('corrupt-artifact-'+str(index));injected=[]
            def copy(source,destination,*args,**kwargs):
                result=original_copy(source,destination,*args,**kwargs)
                if Path(destination)==child/name:
                    original_write(Path(destination),b'injected corrupt child copy');injected.append(True)
                return result
            def write(path,payload):
                if path==child/name:
                    injected.append(True);payload=b'injected corrupt child source'
                return original_write(path,payload)
            with self.subTest(name=name),patch.object(transition.shutil,'copy2',side_effect=copy), \
                    patch.object(Path,'write_bytes',write),self.assertRaisesRegex(ValueError,'child artifact'):
                transition.create_transition(self.plan(),run_id=child.name)
            self.assertTrue(injected);self.assertFalse((child/'manifest.json').exists())
            self.assertEqual(transition._files(self.parent),before)

    def test_checkpoint_after_readback_and_source_receipt_write_corruption_never_publish(self):
        before=transition._files(self.parent);original_copy=transition.shutil.copy2
        child=self.root/'corrupt-primary-after-readback';injected=[]
        def copy(source,destination,*args,**kwargs):
            result=original_copy(source,destination,*args,**kwargs)
            if Path(destination)==child/'checkpoint.source-boundary.npz':
                (child/'checkpoint.npz').write_bytes(b'corrupt after valid readback');injected.append(True)
            return result
        with patch.object(transition.shutil,'copy2',side_effect=copy),self.assertRaisesRegex(ValueError,'child artifact'):
            transition.create_transition(self.plan(),run_id=child.name)
        self.assertTrue(injected);self.assertFalse((child/'manifest.json').exists())
        child=self.root/'corrupt-source-receipt';original_json=transition._write_json
        def write(path,value):
            original_json(path,value)
            if path==child/'source-transition.json':path.write_bytes(b'{}\n')
        with patch.object(transition,'_write_json',side_effect=write),self.assertRaisesRegex(ValueError,'receipt readback'):
            transition.create_transition(self.plan(),run_id=child.name)
        self.assertFalse((child/'manifest.json').exists());self.assertEqual(transition._files(self.parent),before)

    def test_relative_proofs_survive_external_replacements_and_windows_collisions_fail(self):
        child,receipt=transition.create_transition(self.plan(),run_id='retained-proofs')
        for row in receipt['validation_evidence']:
            self.assertEqual(row['storage'],'child_relative_exact_copy');self.assertEqual(transition._file_sha(child/row['child_file']),row['sha256'])
            Path(row['source_path']).write_text('{}')
        self.repair_review.write_text('{}')
        repair=json.loads((child/'repair-review.json').read_text())
        for row in repair['validation_receipts']:self.assertEqual(transition._file_sha(child/row['path']),row['sha256'])
        for row in self.proofs:transition._write_json(self.root/row['path'],{'role':Path(row['path']).stem,'passed':True})
        for index,name in enumerate(('MANIFEST.JSON','FRAME_9999.JSON','source-transition.json.tmp')):
            transition._write_json(self.root/name,{'reserved':True})
            self.proofs[0]={'path':name,'sha256':transition._file_sha(self.root/name)};self.approve()
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'basename collides'):
                transition.create_transition(self.plan(),run_id='collision-'+str(index))
            self.assertFalse((self.root/('collision-'+str(index))/'manifest.json').exists())

    def test_cli_plans_only_and_never_prints_nested_source_provenance(self):
        output=io.StringIO()
        with patch('sys.argv',['transition','--parent',str(self.parent)]),patch('sys.stdout',output), \
                patch.object(transition,'create_transition',side_effect=AssertionError('published')):
            transition.main()
        summary=json.loads(output.getvalue());self.assertEqual(summary['state'],'read_only_plan_validated')
        self.assertNotIn('parent_files_sha256',summary);self.assertLess(len(output.getvalue()),1000)


if __name__=='__main__':unittest.main()
