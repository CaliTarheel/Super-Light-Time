"""Chunk split numerics, durable transactions and native completion guards."""
from decimal import Decimal as D,localcontext
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from benchmarks.collision_architecture import chunked_basal as cb
from benchmarks.collision_architecture.framed_basal_map import FramedBlockMap
from benchmarks.collision_architecture import finite_basal_operator as finite
from benchmarks.collision_architecture.native_basal_domain import NativeBasalBlocker
from tests import test_framed_basal_map as frame_test
from tests import test_native_basal_domain as native_test


def chunk(seed=1,count=1,unknowns=9):
    rng=np.random.default_rng(seed)
    blocks=rng.normal(size=(count,9,9));columns=rng.integers(0,unknowns,size=(count,9))
    hi,lo=frame_test.frame()
    design=FramedBlockMap(blocks,np.arange(count*9).reshape(count,9),columns,(count*9,unknowns),
        np.tile(hi,(count,1,1)),np.tile(lo,(count,1,1)))
    return cb.FactorChunk(design,rng.normal(size=count*9))


def store(path,*,create=False,**kw):
    args=dict(binding={'source':'test-source','state':'test-state','policy':'explicit'},
        expected_control_ids=[0,1],unknowns=9,max_chunk_bytes=100000,max_store_bytes=1000000)
    args.update(kw)
    return cb.FactorStore(path,create=create,**args)


def full_map(chunks):
    blocks=np.concatenate([c.design.blocks for c in chunks]);n=len(blocks)
    return FramedBlockMap(blocks,np.arange(9*n).reshape(n,9),np.concatenate([c.design.columns for c in chunks]),
        (9*n,chunks[0].design.shape[1]),np.concatenate([c.design.frame_hi for c in chunks]),
        np.concatenate([c.design.frame_lo for c in chunks]))


def native_build(domain,system,path,**kwargs):
    options=dict(mantle_omega_rad_s=np.array([1.,-2.,3.])*1e-15,
        quadrature_relative_tolerance=2e-10,max_order=64,max_chunk_bytes=1000000,
        max_store_bytes=10000000,controls_per_batch=2)
    options.update(kwargs)
    return cb.build_native(domain,system,path,**options)


class ChunkedBasalTests(unittest.TestCase):
    def test_chunked_actions_diagonal_and_absolute_bounds_match_full_factor(self):
        chunks=(chunk(1,2),chunk(2,3),chunk(3,1));design=cb.ChunkedFramedMap(chunks);full=full_map(chunks)
        y=np.linspace(-.3,.7,9);v=np.linspace(-.5,.2,54)
        np.testing.assert_allclose(design@y,full@y,rtol=2e-15,atol=1e-15)
        np.testing.assert_allclose(design.rmatvec(v),full.rmatvec(v),rtol=2e-15,atol=1e-15)
        np.testing.assert_allclose(design.gram_action(y),full.rmatvec(full@y),rtol=3e-15,atol=1e-14)
        np.testing.assert_allclose(design.diagonal(),full.diagonal(),rtol=3e-15,atol=1e-14)
        dense=full.to_dense()
        self.assertTrue(np.all(design.absolute_action(y)>=abs(dense).T@(abs(dense)@abs(y))))
        self.assertTrue(np.all(design.absolute_forward(y)>=abs(dense)@abs(y)))
        self.assertEqual(design.storage_bytes,sum(c.storage_bytes for c in chunks))

    def test_saved_split_centroid_retains_low_words_across_chunks(self):
        t=frame_test;chunks=[]
        for b,h,l in zip(t.SPLIT_BLOCKS,t.SPLIT_HI,t.SPLIT_LO):
            blocks=np.zeros((1,9,9));blocks[0,:3,:3]=b
            d=FramedBlockMap(blocks,np.arange(9)[None],np.arange(9)[None],(9,9),h[None],l[None])
            chunks.append(cb.FactorChunk(d,np.zeros(9)))
        values=np.zeros((2,9));values[:,:3]=t.SPLIT_INPUT
        actual=cb.ChunkedFramedMap(chunks).rmatvec(values.ravel())
        rounded=sum((c.design.rmatvec(v) for c,v in zip(chunks,values)),np.zeros(9))
        with localcontext() as ctx:
            ctx.prec=140;truth=[D(0)]*3
            for b,h,l,v in zip(t.SPLIT_BLOCKS,t.SPLIT_HI,t.SPLIT_LO,t.SPLIT_INPUT):
                local=[t.dot(list(map(t.dec,col)),list(map(t.dec,v))) for col in b.T]
                q=[[t.dec(h[i,j])+t.dec(l[i,j]) for j in range(3)] for i in range(3)]
                truth=[a+t.dot(row,local) for a,row in zip(truth,q)]
            errors=[abs(t.dec(a)-r)/abs(r) for a,r in zip(actual,truth)]
            lost=[abs(t.dec(a)-r)/abs(r) for a,r in zip(rounded,truth)]
            self.assertLess(max(errors),D('4e-16'));self.assertGreater(max(lost),D('1e-10'))
        # Sparse transpose exchange allocates only referenced unique indices.
        indices,hi,lo=chunks[0].design.rmatvec_contributions(values[0])
        self.assertEqual(len(indices),9)
        np.testing.assert_array_equal(hi+lo,chunks[0].design.rmatvec(values[0]))

    def test_streaming_component_matches_existing_mantle_and_work_protocol(self):
        _,system=native_test.system(native_test.state())
        chunks=(chunk(2),chunk(5));design=cb.ChunkedFramedMap(chunks)
        streamed=cb.ChunkedFiniteBasal(system,design)
        full=finite.FiniteBasal(system,full_map(chunks),np.concatenate([c.reference for c in chunks]),[])
        y=np.linspace(-.3,.7,9)
        np.testing.assert_allclose(streamed.load_n,full.load_n,rtol=1e-14,atol=1e-14)
        np.testing.assert_allclose(streamed.evaluate(y)[1],full.evaluate(y)[1],rtol=1e-14,atol=1e-14)
        self.assertAlmostEqual(streamed.evaluate(y)[0],full.evaluate(y)[0],places=12)
        for k,v in full.work(y).items():self.assertAlmostEqual(streamed.work(y)[k]/v,1.,places=13)
        replacement=finite.replace_basal(system,streamed,other_drag_excludes_allocated_basal=True)
        np.testing.assert_allclose(replacement['hessian_n_s_m']@y,streamed.hessian_n_s_m@y,rtol=1e-14)
        self.assertFalse(streamed.reference_factor_sqrt_w.flags.writeable)
        with localcontext() as context:
            context.prec=100;expected=D(0)
            for c in chunks:
                for a,b in zip(c.design.absolute_forward(y),abs(c.reference)):
                    expected+=(D.from_float(float(a))+D.from_float(float(b)))**2
            self.assertGreaterEqual(D.from_float(streamed.work(y)['arithmetic_constituent_work_bound_w']),expected)

    def test_actual_interface_sparse_solve_accepts_chunked_basal_protocol(self):
        from tests.test_finite_basal_operator import FiniteBasalOperatorTests
        original=finite.build
        def replacement(system,*args,**kwargs):
            component=original(system,*args,**kwargs);d=component.basal_design;chunks=[]
            # Exact partition of existing factor rows into two maps. Zeros
            # retain9x9 local storage without duplicating physical resistance.
            for start,end in ((0,4),(4,9)):
                blocks=np.zeros_like(d.blocks);blocks[:,start:end]=d.blocks[:,start:end]
                reference=np.zeros_like(component.reference_factor_sqrt_w).reshape(-1,9)
                reference[:,start:end]=component.reference_factor_sqrt_w.reshape(-1,9)[:,start:end]
                design=FramedBlockMap(blocks,d.rows,d.columns,d.shape,d.frame_hi,d.frame_lo)
                chunks.append(cb.FactorChunk(design,reference.ravel()))
            return cb.ChunkedFiniteBasal(system,cb.ChunkedFramedMap(chunks))
        with patch.object(finite,'build',side_effect=replacement):
            FiniteBasalOperatorTests().test_actual_finite_basal_weld_interface_and_sparse_solver_share_ledger()

    def test_rows_are_disjoint_and_memory_mutation_rejected(self):
        c=chunk();d=c.design
        bad=FramedBlockMap(d.blocks,np.zeros_like(d.rows),d.columns,d.shape,d.frame_hi,d.frame_lo)
        with self.assertRaisesRegex(ValueError,'rows'):cb.FactorChunk(bad,c.reference)
        design=cb.ChunkedFramedMap([c]);d.blocks.setflags(write=True);d.blocks[0,0,0]+=1
        with self.assertRaisesRegex(ValueError,'changed'):design@np.ones(9)
        empty=FramedBlockMap(np.empty((0,9,9)),np.empty((0,9),int),np.empty((0,9),int),(0,9),
            np.empty((0,3,3)),np.empty((0,3,3)))
        with self.assertRaises(ValueError):cb.FactorChunk(empty,np.empty(0))

    def test_durable_partial_resume_seal_and_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=chunk(1),chunk(2);s=store(tmp,create=True);s.append([0],a,{'cell':0})
            with self.assertRaisesRegex(ValueError,'Partial'):s.factor_map()
            with self.assertRaisesRegex(ValueError,'every expected'):s.seal({'complete':True})
            reopened=store(tmp);self.assertEqual(reopened.completed_controls,(0,))
            reopened.append([1],b,{'cell':1});reopened.seal({'controls_complete':True})
            with self.assertRaisesRegex(ValueError,'immutable'):reopened.append([1],b,{})
            x=np.arange(9)*.01
            np.testing.assert_array_equal(reopened.factor_map()@x,cb.ChunkedFramedMap([a,b])@x)
            np.testing.assert_allclose(reopened.factor_map().gram_action(x),cb.ChunkedFramedMap([a,b]).gram_action(x),rtol=2e-15)
            self.assertLess(reopened.factor_map().storage_bytes,a.storage_bytes+b.storage_bytes)

    def test_stale_bindings_duplicate_and_foreign_controls_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=store(tmp,create=True);s.append([0],chunk(),{})
            for ids in ([0],[1,1],[2],[True]):
                with self.assertRaises(ValueError):s.append(ids,chunk(),{})
            with self.assertRaisesRegex(ValueError,'binding'):store(tmp,binding={'source':'changed'})
            with self.assertRaises(ValueError):store(tmp,expected_control_ids=[0,2])
            stale=store(tmp);s.append([1],chunk(),{})
            with self.assertRaisesRegex(ValueError,'changed'):list(stale.iter_batches())

    def test_altered_file_missing_member_and_mutable_descriptor_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=store(tmp,create=True);s.append([0],chunk(),{});s.append([1],chunk(2),{});s.seal({})
            path=Path(tmp)/s._manifest['chunks'][0]['file'];data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
            with self.assertRaisesRegex(ValueError,'checksum'):s.factor_map().gram_action(np.ones(9))
        with tempfile.TemporaryDirectory() as tmp:
            s=store(tmp,create=True);s.append([0],chunk(),{})
            s._manifest['chunks'][0]['control_ids']=[1]
            with self.assertRaisesRegex(ValueError,'changed'):list(s.iter_batches())

    def test_crash_orphan_is_ignored_and_prior_completed_batch_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=store(tmp,create=True);s.append([0],chunk(),{})
            with patch.object(cb,'_atomic_json',side_effect=OSError('injected manifest interruption')):
                with self.assertRaises(OSError):s.append([1],chunk(2),{})
            reopened=store(tmp);self.assertEqual(reopened.completed_controls,(0,))
            self.assertEqual(len(list(Path(tmp).glob('chunk-*.npz'))),2)
            reopened.append([1],chunk(2),{});reopened.seal({})
            self.assertEqual(len(list(Path(tmp).glob('chunk-*.npz'))),3)
            self.assertEqual(len(reopened._manifest['chunks']),2)

    def test_resource_limit_keeps_completed_work_without_partial_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=store(tmp,create=True,max_chunk_bytes=10000);s.append([0],chunk(),{})
            before=(Path(tmp)/'manifest.json').read_bytes()
            with self.assertRaisesRegex(ValueError,'resource limit'):s.append([1],chunk(count=20),{})
            self.assertEqual((Path(tmp)/'manifest.json').read_bytes(),before)
            self.assertEqual(s.completed_controls,(0,))
            with self.assertRaises(ValueError):s.factor_map()
        with tempfile.TemporaryDirectory() as tmp:
            s=store(tmp,create=True,max_store_bytes=20000)
            (Path(tmp)/('chunk-'+'a'*32+'.npz.tmp')).write_bytes(b'0'*19000)
            with self.assertRaisesRegex(ValueError,'resource limit'):s.append([0],chunk(),{})
            self.assertEqual(s.completed_controls,())

    def test_native_interruption_resume_preserves_completed_cells_and_global_law(self):
        with tempfile.TemporaryDirectory() as tmp:
            domain,system=native_test.system(native_test.state());visited=[]
            original=domain._cell
            def count(index):visited.append(index);return original(index)
            with patch.object(domain,'_cell',side_effect=count):
                progress=native_build(domain,system,tmp,stop_after_batches=1)
                self.assertFalse(progress.global_allocation_complete);self.assertEqual(progress.completed_controls,(0,1))
                result=native_build(domain,system,tmp)
            self.assertEqual(visited,list(range(8)));self.assertTrue(result.certificate['global_allocation_complete'])
            baseline=native_test.build(domain,system);y=np.linspace(-.003,.002,9)
            for key,value in baseline.component.work(y).items():
                self.assertAlmostEqual(result.component.work(y)[key]/value,1.,places=12)
            np.testing.assert_allclose(result.component.hessian_n_s_m@y,baseline.component.hessian_n_s_m@y,rtol=5e-15,atol=1.)
            # A sealed reopen does not rebuild even one expensive control.
            with patch.object(domain,'_cell',side_effect=AssertionError('must reuse completed factors')):
                again=native_build(domain,system,tmp)
            self.assertEqual(again.certificate['manifest_sha256'],result.certificate['manifest_sha256'])
            for bad in ('2e-10',np.array([2e-10]),True):
                with self.assertRaises(ValueError):native_build(domain,system,tmp,quadrature_relative_tolerance=bad)

    def test_native_failure_keeps_journal_and_cannot_skip_its_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            domain,system=native_test.system(native_test.state())
            first=native_build(domain,system,tmp,stop_after_batches=1)
            with patch.object(domain,'_cell',side_effect=NativeBasalBlocker('missing order',{'control_id':2})):
                with self.assertRaisesRegex(NativeBasalBlocker,'missing order'):native_build(domain,system,tmp)
            manifest=json.loads((Path(tmp)/'manifest.json').read_text())
            self.assertEqual(manifest['state'],'BUILDING');self.assertIsNone(manifest['certificate'])
            self.assertEqual([i for row in manifest['chunks'] for i in row['control_ids']],list(first.completed_controls))
            self.assertEqual(json.loads((Path(tmp)/'failure.json').read_text())['witness']['original_witness']['control_id'],2)

    def test_native_actual_unordered_stack_remains_hard_failure(self):
        s=native_test.state(3,mixed=True)
        s.collision_contacts=[dict(top_sheet=11,under_sheet=10),dict(top_sheet=12,under_sheet=10)]
        domain,system=native_test.system(s)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(NativeBasalBlocker,'unique saved layer order'):native_build(domain,system,tmp)
            manifest=json.loads((Path(tmp)/'manifest.json').read_text())
            self.assertEqual(manifest['chunks'],[]);self.assertEqual(manifest['state'],'BUILDING')

    def test_native_changed_state_and_source_prevent_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=native_test.state();domain,system=native_test.system(s)
            native_build(domain,system,tmp,stop_after_batches=1)
            with patch.object(cb,'_source_binding',return_value={'files':{'changed':'source'}}):
                with self.assertRaisesRegex(ValueError,'binding'):native_build(domain,system,tmp)
            s.crust[0]=1
            with self.assertRaisesRegex(ValueError,'changed'):native_build(domain,system,tmp)

    def test_midbuild_source_change_cannot_commit_or_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            domain,system=native_test.system(native_test.state());original=cb._source_binding()
            with patch.object(cb,'_source_binding',side_effect=[original,{'changed':'during computation'}]):
                with self.assertRaisesRegex(ValueError,'source changed'):native_build(domain,system,tmp)
            manifest=json.loads((Path(tmp)/'manifest.json').read_text())
            self.assertEqual(manifest['state'],'BUILDING');self.assertEqual(manifest['chunks'],[])

    def test_corrupt_prior_batch_rejects_before_computing_more_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            domain,system=native_test.system(native_test.state())
            native_build(domain,system,tmp,stop_after_batches=1)
            manifest=json.loads((Path(tmp)/'manifest.json').read_text());p=Path(tmp)/manifest['chunks'][0]['file']
            p.write_bytes(b'invalid')
            with patch.object(domain,'_cell',side_effect=AssertionError('must verify committed work first')):
                with self.assertRaisesRegex(ValueError,'byte length'):native_build(domain,system,tmp)
        with tempfile.TemporaryDirectory() as tmp:
            domain,system=native_test.system(native_test.state());native_build(domain,system,tmp,stop_after_batches=1)
            original=cb.FactorStore._load
            def bad_summary(store,row):
                chunk,metadata=original(store,row);metadata['area_m2'][0]*=1.1
                return chunk,metadata
            with patch.object(cb.FactorStore,'_load',bad_summary),patch.object(domain,'_cell',side_effect=AssertionError('must verify summaries first')):
                with self.assertRaisesRegex(ValueError,'individual measure'):native_build(domain,system,tmp)

    def test_finite_failure_saves_piece_chart_and_quadrature_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            domain,system=native_test.system(native_test.state())
            with self.assertRaisesRegex(RuntimeError,'quadrature did not converge'):
                native_build(domain,system,tmp,quadrature_relative_tolerance=1e-16,max_order=16)
            record=json.loads((Path(tmp)/'failure.json').read_text())['witness']
            self.assertEqual(record['control_id'],0);self.assertEqual(record['piece_index'],0)
            self.assertEqual(len(record['exact_chart_polygon']),3)
            self.assertEqual(record['quadrature_orders'],[8,16]);self.assertTrue(record['quadrature_records'])


if __name__=='__main__':unittest.main()
