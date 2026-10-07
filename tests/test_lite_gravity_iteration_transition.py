"""Synthetic stopped histories; never run a real coupled interval or server."""
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
import lite_gravity_iteration_transition as transition
import native_engine


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


class GravityIterationTransitionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'tmp')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / 'output/runs'
        self.output.mkdir(parents=True)
        self.parent = self.root / 'preserved-parent'
        self.parent.mkdir()
        self.old = {name: b'VALUE = 1\n' for name in transition.HELPERS}
        self.old.update({f'helper_{i:03d}.py': b'VALUE = 3\n' for i in range(171)})
        self.new = dict(self.old)
        for i, name in enumerate(transition.HELPERS):
            self.new[name] = f'VALUE = {i + 2}\n'.encode()
        self.engine = ('\n'.join('import ' + name[:-3] for name in self.old) + '\n').encode()
        self.prior = dict(engine_sha256=sha(self.engine), auxiliary_sources_sha256={name:sha(data) for name,data in self.old.items()},
            numpy_version=np.__version__, flow_backend={'fixture':'flow'}, material_backend={'fixture':'material'})
        self.current = dict(deepcopy(self.prior), auxiliary_sources_sha256={name:sha(data) for name,data in self.new.items()})
        self.frames = [{'index':i, 'time_myr':float(epoch)} for i,epoch in enumerate(sorted([*range(0,95,2),25]))]
        config = dict(seed=37,width=512,height=256,dt_myr=1.,snapshot_myr=2.,duration_myr=1000.,enhanced_rifting=False,
            effective_subduction={'force_n_per_m':3e13,'carrier_traction':{'version':1,'alpha':.1}},
            force_limit_rifting={'inherited_weakness':True,'search_policy':{'mode':'bounded_axes','max_axes':10}})
        self.original = SimpleNamespace(t=95.,steps=97,config=config,rng=np.random.default_rng(37),
            omega=np.arange(6,dtype=np.float64).reshape(2,3),native_locator={'keys':np.arange(3,dtype=np.int64)},
            native_boundary_geometry={'segments':np.eye(3),'diagnostics':{'accepted_epoch':95.}},
            material_surface={'vertices':np.eye(3),'faces':np.array([[0,1,2]],np.int32),'reference_area_km2':np.array([18.])},
            effective_subduction_initiation={'activation_myr':25.,'candidates':[{'compression_myr':12.,'shortening_km':110.}]},
            force_rifting_policy={'activation_myr':43.,'epoch_myr':95.},force_rifting_state={'2':{'work_j':4e20,'opening_km':21.}},
            events=[{'time_myr':94.,'type':'accepted_old'}],event_keys={(i,'accepted_old') for i in range(8)})
        count = len(transition._fingerprint(vars(self.original))['arrays'])
        self.original.fixture_arrays = [np.array([i],dtype=np.int32) for i in range(463-count)]
        self.manifest = dict(run_id=transition.PARENT_RUN_ID,title='Asein Lite',state='interrupted',can_resume=True,
            config=deepcopy(config),time_myr=95.,integration_time_myr=95.,checkpoint_time_myr=95.,next_output_myr=96.,
            duration_myr=1000.,frame_count=49,frames=deepcopy(self.frames),source_transition={'kind':'earlier_source','inherited_frame_count':38},
            frame_sources=[],**deepcopy(self.prior))
        ancestor = self.parent / 'inherited-parent'
        older = ancestor / 'inherited-parent'
        self.source_files(self.parent,49)
        self.source_files(ancestor,38)
        self.source_files(older,2)
        older_manifest = dict(run_id='older',frame_count=2,frames=self.frames[:2],**deepcopy(self.prior))
        transition._write_json(older/'manifest.json',older_manifest)
        ancestor_manifest = dict(run_id='ancestor',frame_count=38,frames=self.frames[:38],
            frame_sources=[self.lineage(older,'older',i,None) for i in range(2)],**deepcopy(self.prior))
        transition._write_json(ancestor/'manifest.json',ancestor_manifest)
        self.manifest['frame_sources'] = [self.lineage(ancestor,'ancestor',i,
            ancestor_manifest['frame_sources'][i] if i<2 else None) for i in range(38)]
        (self.parent/'config.json').write_text(json.dumps(config))
        (self.parent/'initial.json').write_bytes(b'{"original_artwork":"fixture"}\n')
        (self.parent/'checkpoint.source-boundary.npz').write_bytes(b'older immutable evidence')
        self.publish_parent()
        for i in range(1508-len(transition._files(self.parent))):
            (self.parent/f'preserved_evidence_{i:04d}.txt').write_bytes(f'exact old evidence {i}'.encode())
        self.paths = {key:self.root/(key+'.json') for key in ('preservation','seal','source-review','repair-review','authority','stop-evidence','original-capture','original-operator-audit')}
        self.proofs = []
        for role in transition.VALIDATION_ROLES:
            path=self.root/(role+'.json');transition._write_json(path,{'synthetic_fixture':True,'role':role,'passed':True})
            self.proofs.append({'role':role,'path':path.name,'sha256':transition._file_sha(path)})
        self.saved_context = transition._current_context
        for item in (patch.object(transition,'ROOT',self.root),patch.object(transition,'REVIEWED_APPROVAL',None),
                patch.object(transition,'PARENT_CHECKPOINT_SHA256','0'*64),patch.object(transition,'ORIGINAL_CAPTURE_SHA256','0'*64),
                patch.object(transition,'ORIGINAL_OPERATOR_AUDIT_SHA256','0'*64),
                patch.object(transition,'_current_context',side_effect=lambda approved:(deepcopy(self.current),self.engine,dict(self.new)))):
            item.start();self.addCleanup(item.stop)
        self.approve()

    def source_files(self,folder,count):
        folder.mkdir(parents=True,exist_ok=True)
        (folder/'engine.py').write_bytes(self.engine)
        for name,payload in self.old.items():(folder/name).write_bytes(payload)
        for row in self.frames[:count]:
            stem=folder/f"frame_{row['index']:04d}"
            np.savez_compressed(stem.with_suffix('.npz'),elevation=np.arange(12,dtype=np.float32))
            stem.with_suffix('.json').write_text(json.dumps(row))

    def lineage(self,folder,run_id,index,prior):
        return dict(index=index,time_myr=self.frames[index]['time_myr'],run_id=run_id,inherited=True,
            source_manifest='inherited-parent/manifest.json',source_manifest_sha256=transition._file_sha(folder/'manifest.json'),
            engine_sha256=self.prior['engine_sha256'],auxiliary_sources_sha256=deepcopy(self.prior['auxiliary_sources_sha256']),
            files_sha256={suffix:transition._file_sha(folder/f'frame_{index:04d}{suffix}') for suffix in ('.npz','.json')},
            parent_frame_source=deepcopy(prior))

    def publish_parent(self):
        checkpoint.write_checkpoint(self.parent/'checkpoint.npz',self.original,self.manifest,self.prior)
        self.external=dict(deepcopy(self.manifest),state='error',time_myr=94.,next_output_myr=74.,can_resume=False,error=transition.ORIGINAL_ERROR)
        transition._write_json(self.parent/'manifest.json',self.external)

    def approve(self, *, review_changes=None, source_changes=None, authority_changes=None, capture_changes=None):
        cp=transition._file_sha(self.parent/'checkpoint.npz');transition.PARENT_CHECKPOINT_SHA256=cp
        inventory=transition._files(self.parent)
        transition._write_json(self.paths['preservation'],dict(kind='asein_lite_complete_failed_history_preservation',
            run_id=transition.PARENT_RUN_ID,checkpoint_sha256=cp,checkpoint_time_myr=95.,public_frame_myr=94.,
            error=self.external['error'],error_signature=transition.ORIGINAL_ERROR_SIGNATURE,all_copied_files_exact=True,
            source_closure_verified=True,helper_count=173,file_count=len(inventory),files=[{'path':name,'sha256':value} for name,value in inventory.items()]))
        transition._write_json(self.paths['stop-evidence'],self.external)
        transition._write_json(self.paths['seal'],self.current['auxiliary_sources_sha256'])
        authority=dict(kind=transition.AUTHORITY_KIND,version=1,parent_run_id=transition.PARENT_RUN_ID,source_commit=transition.PARENT_SOURCE_COMMIT,
            parent_checkpoint_sha256=cp,epoch_myr=95.,authorized=True,authorization_source='Synthetic fixture of existing human authority',
            new_physical_policy_authorized=False,instruction=transition.AUTHORITY_INSTRUCTION,scope=transition.AUTHORITY_SCOPE)
        authority.update(authority_changes or {});transition._write_json(self.paths['authority'],authority)
        fp=transition._fingerprint(vars(self.original))
        capture=dict(kind='private_rejected_interval_predictor_capture',run_id=transition.PARENT_RUN_ID,checkpoint_sha256=cp,start_myr=95.,
            requested_dt_myr=1.,outcome='captured_original_failure',captured_failure=True,constructor_called=False,import_state_exact=True,
            imported_array_count=463,state_fingerprint=fp['sha256'],private_state_fingerprint_after=fp['sha256'],entire_private_state_rolled_back=True,
            production_files_unchanged=True,production_file_count=1508,source_closure_unchanged=True,runtime_source_changed=False,
            physics_or_guards_changed=False,production_restarted=False,durable_step_published=False)
        capture.update(capture_changes or {});transition._write_json(self.paths['original-capture'],capture)
        audit=dict(kind='frozen_rejected_gravity_predictor_operator_audit',run_id=transition.PARENT_RUN_ID,solve_iterations_run=0,
            coupled_simulation_imported=False,physical_law_or_guard_changed=False,original_iterations=2048,original_tolerance=1e-8,
            status='completed_read_only_operator_audit_of_original_failure',checks_total=1,checks_passed=1)
        transition._write_json(self.paths['original-operator-audit'],audit)
        transition.ORIGINAL_CAPTURE_SHA256=transition._file_sha(self.paths['original-capture'])
        transition.ORIGINAL_OPERATOR_AUDIT_SHA256=transition._file_sha(self.paths['original-operator-audit'])
        delta={name:{'parent_sha256':sha(self.old[name]),'child_sha256':sha(self.new[name])} for name in transition.HELPERS}
        review=dict(kind=transition.REPAIR_REVIEW_KIND,version=1,ready=True,parent_run_id=transition.PARENT_RUN_ID,
            parent_checkpoint_sha256=cp,epoch_myr=95.,changed_helpers=delta,runtime_source_seal_sha256=transition._file_sha(self.paths['seal']),
            stop_evidence_sha256=transition._file_sha(self.paths['stop-evidence']),original_error_signature=transition.ORIGINAL_ERROR_SIGNATURE,
            original_capture_sha256=transition.ORIGINAL_CAPTURE_SHA256,original_operator_audit_sha256=transition.ORIGINAL_OPERATOR_AUDIT_SHA256,
            original_iteration_budget=2048,maximum_iteration_budget=4096,stationarity_tolerance=1e-8,original_budget_prefix_exact=True,
            unconstrained_gravity_only=True,bounded_inverse_budgets_unchanged=True,stationarity_gate_unchanged=True,captured_regression_passed=True,
            future_interval_passed=True,physical_laws_unchanged=True,physical_guards_unchanged=True,resolution_unchanged=True,validation_receipts=deepcopy(self.proofs))
        review.update(review_changes or {});transition._write_json(self.paths['repair-review'],review)
        source=dict(kind=transition.SOURCE_REVIEW_KIND,version=1,ready=True,parent_run_id=transition.PARENT_RUN_ID,parent_checkpoint_sha256=cp,
            epoch_myr=95.,boundary_kind='durable_stopped',stop_evidence_sha256=transition._file_sha(self.paths['stop-evidence']),
            parent_preservation_sha256=transition._file_sha(self.paths['preservation']),runtime_source_seal_sha256=transition._file_sha(self.paths['seal']),
            parent_compatibility_sha256=sha(transition._json_bytes(self.prior)),child_compatibility_sha256=sha(transition._json_bytes(self.current)),
            changed_helpers=delta,added_helpers={},removed_helpers=[],helper_count=173,repair_review_sha256=transition._file_sha(self.paths['repair-review']),
            authority_sha256=transition._file_sha(self.paths['authority']),configuration_unchanged=True,state_import='entire typed state exact; cached geometry retained')
        source.update(source_changes or {});transition._write_json(self.paths['source-review'],source)
        self.approval=transition.TransitionApproval(transition.PARENT_RUN_ID,transition.PARENT_SOURCE_COMMIT,95.,cp,
            self.paths['preservation'],transition._file_sha(self.paths['preservation']),self.paths['seal'],transition._file_sha(self.paths['seal']),
            self.paths['source-review'],transition._file_sha(self.paths['source-review']),self.paths['repair-review'],transition._file_sha(self.paths['repair-review']),
            self.paths['authority'],transition._file_sha(self.paths['authority']),self.paths['stop-evidence'],transition._file_sha(self.paths['stop-evidence']),
            self.paths['original-capture'],transition.ORIGINAL_CAPTURE_SHA256,self.paths['original-operator-audit'],transition.ORIGINAL_OPERATOR_AUDIT_SHA256,
            tuple((name,sha(self.old[name]),sha(self.new[name])) for name in transition.HELPERS),True)

    def plan(self):
        return transition.plan_transition(self.parent,approval=self.approval)

    def create(self,plan=None,run_id='child'):
        return transition.create_transition(plan or self.plan(),output_root=self.output,run_id=run_id)

    def test_exact463_arrays_rng_cache_clocks49_recursive_frames_and_stale_external_schedule(self):
        before=transition._files(self.parent)
        sentinels=[patch.object(native_engine.Simulation,name,side_effect=AssertionError(name))
            for name in ('__init__','step','_boundaries','_rasterize','_forces')]
        for item in sentinels:item.start();self.addCleanup(item.stop)
        child,receipt=self.create()
        restored,saved=checkpoint.read_checkpoint(child/'checkpoint.npz',self.current,native_engine.Simulation)
        self.assertEqual(transition._fingerprint(vars(restored)),transition._fingerprint(vars(self.original)))
        self.assertEqual(receipt['preexisting_array_count'],463);self.assertEqual(receipt['parent_file_count'],1508)
        self.assertEqual(saved['next_output_myr'],96.);self.assertEqual(receipt['predecessor_external_next_output_myr'],74.)
        self.assertEqual(receipt['predecessor_public_time_myr'],94.);self.assertEqual(saved['state'],'paused')
        self.assertEqual(saved['source_transition']['boundary_kind'],'durable_stopped');self.assertNotIn('error',saved)
        self.assertEqual(saved['frames'],self.frames);self.assertEqual(len(saved['frame_sources']),49)
        self.assertEqual(saved['frame_sources'][0]['parent_frame_source'],self.manifest['frame_sources'][0])
        self.assertIsNone(saved['frame_sources'][48]['parent_frame_source'])
        self.assertEqual(transition._files(self.parent),before);self.assertEqual(transition._files(child/'inherited-parent'),before)
        self.assertEqual((child/'checkpoint.npz').read_bytes(),(child/'checkpoint.source-boundary.npz').read_bytes())
        self.assertEqual(json.loads((child/'manifest.json').read_text()),saved)
        self.assertFalse(receipt['model_constructed']);self.assertFalse(receipt['model_stepped']);self.assertFalse(receipt['geometry_rebuilt_at_import'])
        self.assertEqual(receipt['physical_time_advanced_myr'],0.)
        self.assertFalse(list(self.output.glob('.src-*')))

    def test_no_default_approval_exact_immutable_pins_and_two_ordered_deltas(self):
        with self.assertRaisesRegex(ValueError,'not sealed'):transition.plan_transition(self.parent)
        with self.assertRaises(FrozenInstanceError):self.approval.review_ready=False
        for change in ({'review_ready':False},{'parent_epoch_myr':94.},{'parent_run_id':'other'},
                {'original_capture_sha256':'0'*64},{'helper_delta':self.approval.helper_delta[:1]},
                {'helper_delta':tuple(reversed(self.approval.helper_delta))}):
            with self.subTest(change=change),self.assertRaises(ValueError):transition.plan_transition(self.parent,approval=replace(self.approval,**change))

    def test_regression_alone_unchanged_guard_prefix_budget_and_distinct_four_receipts_required(self):
        for change in ({'future_interval_passed':False},{'captured_regression_passed':False},{'original_budget_prefix_exact':False},
                {'stationarity_tolerance':1e-7},{'maximum_iteration_budget':8192},{'bounded_inverse_budgets_unchanged':False},
                {'unconstrained_gravity_only':False},{'physical_guards_unchanged':False},{'validation_receipts':self.proofs[:1]},
                {'validation_receipts':[self.proofs[0],self.proofs[0]]}):
            self.approve(review_changes=change)
            with self.subTest(change=change),self.assertRaises(ValueError):self.plan()
        self.approve();(self.root/self.proofs[0]['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'validation evidence'):self.plan()

    def test_original_capture_rollback_and_audit_cannot_be_relabelled_as_regression(self):
        for change in ({'entire_private_state_rolled_back':False},{'production_files_unchanged':False},
                {'imported_array_count':462},{'runtime_source_changed':True}):
            self.approve(capture_changes=change)
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'original diagnostic capture'):self.plan()
        self.approve()
        proofs=deepcopy(self.proofs);proofs[0].update(path=str(self.paths['original-capture']),sha256=self.approval.original_capture_sha256)
        self.approve(review_changes={'validation_receipts':proofs})
        with self.assertRaisesRegex(ValueError,'must be distinct'):self.plan()

    def test_wrong_authority_source_review_or_unknown_kind_is_refused(self):
        for change in ({'kind':'lite_geometry_acceleration_equivalence_performance_review'},
                {'original_error_signature':'0'*64},{'physical_laws_unchanged':False}):
            self.approve(review_changes=change)
            with self.subTest(change=change),self.assertRaises(ValueError):self.plan()
        for change in ({'boundary_kind':'healthy_accepted_step_pause'},{'configuration_unchanged':False},{'parent_preservation_sha256':'0'*64}):
            self.approve(source_changes=change)
            with self.subTest(change=change),self.assertRaises(ValueError):self.plan()
        self.approve(authority_changes={'new_physical_policy_authorized':True})
        with self.assertRaisesRegex(ValueError,'authority'):self.plan()

    def test_only_two_current_helpers_and_full173_import_closure_are_allowed(self):
        baseline=deepcopy(self.new);compat=deepcopy(self.current)
        for kind in ('same_source','third_helper','extra_helper','backend','engine'):
            self.new=deepcopy(baseline);self.current=deepcopy(compat)
            if kind=='same_source':self.new[transition.HELPERS[0]]=self.old[transition.HELPERS[0]]
            if kind=='third_helper':self.new['helper_000.py']=b'VALUE = 9\n'
            if kind=='extra_helper':self.new['extra.py']=b'VALUE = 9\n'
            self.current['auxiliary_sources_sha256']={name:sha(data) for name,data in self.new.items()}
            if kind=='backend':self.current['flow_backend']={'changed':True}
            if kind=='engine':self.current['engine_sha256']='0'*64
            self.approve()
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.plan()
        self.new=baseline;self.current=compat;self.approve()
        (self.parent/'helper_000.py').write_bytes(b'damaged saved helper');self.approve()
        with self.assertRaisesRegex(ValueError,'parent helper source'):self.plan()

    def test_generation_frame_recursive_chain_and_exact_embedded_schedule_tampering_fail(self):
        baseline=deepcopy(self.external)
        for change in ({'state':'paused'},{'error':'different failure'},{'title':'Other'},
                {'checkpoint_time_myr':94.},{'time_myr':95.},{'next_output_myr':96.},{'frame_count':48},{'frame_sources':[]}):
            self.external=dict(deepcopy(baseline),**change);transition._write_json(self.parent/'manifest.json',self.external);self.approve()
            with self.subTest(change=change),self.assertRaises(ValueError):self.plan()
        self.external=baseline;transition._write_json(self.parent/'manifest.json',baseline);self.approve()
        (self.parent/'inherited-parent/inherited-parent/frame_0000.npz').write_bytes(b'ancestor corruption');self.approve()
        with self.assertRaisesRegex(ValueError,'recursive accepted frame'):self.plan()

    def test_import_cannot_modify_rng_cache_clocks_config_or_any_array(self):
        def corrupt(kind):
            def mutate(original):
                result=deepcopy(original)
                if kind=='rng':result.rng.random()
                if kind=='cache':result.native_locator['keys'][0]=999
                if kind=='clock':result.force_rifting_policy['epoch_myr']=0.
                if kind=='config':result.config['effective_subduction']['force_n_per_m']=5e12
                if kind=='array':result.fixture_arrays[-1][0]=-1
                return result
            return mutate
        for kind in ('rng','cache','clock','config','array'):
            with self.subTest(kind=kind),patch.object(transition,'_derive',side_effect=corrupt(kind)),self.assertRaisesRegex(ValueError,'typed array'):self.plan()

    def test_mutable_plan_parent_race_and_corrupted_stage_never_publish(self):
        plan=self.plan();plan.derived.fixture_arrays[0][0]=-1
        with self.assertRaisesRegex(ValueError,'mutable plan'):self.create(plan)
        self.assertFalse((self.output/'child').exists())
        plan=self.plan();(self.parent/'new-unsealed.log').write_text('parent raced')
        with self.assertRaises(ValueError):self.create(plan)
        (self.parent/'new-unsealed.log').unlink();self.approve()
        verify=transition._verify_child_artifacts
        def corrupt(plan,child,manifest,receipt):
            (child/'checkpoint.source-boundary.npz').write_bytes(b'bad proof')
            return verify(plan,child,manifest,receipt)
        with patch.object(transition,'_verify_child_artifacts',side_effect=corrupt),self.assertRaisesRegex(ValueError,'child artifact'):self.create()
        self.assertFalse((self.output/'child').exists());self.assertTrue(list(self.output.glob('.src-*')))

    def test_explicit_isolated_root_windows_final_and_uuid_temp_path_preflight(self):
        plan=self.plan()
        with self.assertRaisesRegex(ValueError,'explicit'):transition.create_transition(plan,run_id='child')
        with self.assertRaisesRegex(ValueError,'isolated'):transition.create_transition(plan,output_root=self.root,run_id='child')
        relative=Path(*(['inherited-parent']*5))/'profiling'/'4b6b0354abdf4bbdb90e87baa966f805.jsonl'
        deep=self.parent/relative;deep.parent.mkdir(parents=True);deep.write_bytes(b'unchanged deep historical evidence')
        next(self.parent.glob('preserved_evidence_*.txt')).unlink()
        self.approve();plan=self.plan()
        with patch.object(transition,'_WINDOWS_PATHS',True),patch.object(transition.shutil,'copytree') as copying, \
                self.assertRaisesRegex(ValueError,'Windows path budget'):
            transition.create_transition(plan,output_root=self.output,run_id='x'*80)
        copying.assert_not_called();self.assertFalse((self.output/('x'*80)).exists())
        long_stage=self.root/('s'*max(1,212-len(str(self.root))-1))
        with patch.object(transition,'_WINDOWS_PATHS',True):
            transition._check_publication_paths(self.parent,long_stage,self.output/'short',{'a':'hash'},{'checkpoint.npz'})
            with self.assertRaisesRegex(ValueError,'Windows path budget'):
                transition._check_publication_paths(self.parent,long_stage,self.output/'short',{'a':'hash'},
                    {'checkpoint.npz.'+'0'*32+'.tmp'})

    def test_fresh_runtime_file_seal_is_checked_before_using_cached_server_capture(self):
        (self.root/'tectonics.py').write_bytes(self.engine)
        for name,payload in self.new.items():(self.root/name).write_bytes(payload)
        with patch.object(transition.server.SimulationManager,'compatibility',return_value=deepcopy(self.current)), \
                patch.object(transition.server,'ENGINE_SOURCE',self.engine),patch.object(transition.server,'AUXILIARY_SOURCES',dict(self.new)):
            self.assertEqual(self.saved_context(self.approval),(self.current,self.engine,self.new))
            (self.root/transition.HELPERS[0]).write_text('VALUE = 999\n')
            with self.assertRaisesRegex(ValueError,'runtime files changed'):self.saved_context(self.approval)


if __name__ == '__main__':
    unittest.main()
