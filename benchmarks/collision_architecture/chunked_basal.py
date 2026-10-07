"""Restartable research basal factors; unchanged geometry and resistance laws.

Chunks have disjoint consecutive factor rows and a shared generalized space.
Only a sealed, complete native build produces a global native component.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import uuid
import zipfile

import numpy as np

from . import finite_basal_operator as finite
from .framed_basal_map import FramedBlockMap, _add_pairs, _checked, _up
from .joint_resistance import _system_signature


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def _hash(data): return hashlib.sha256(data).hexdigest()


def _integer(value,name,minimum=0):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<minimum:
        raise ValueError(f'{name} requires an integer >= {minimum}.')
    return int(value)


def _ids(values,name):
    result=tuple(_integer(v,name) for v in values)
    if not result or tuple(sorted(set(result)))!=result:
        raise ValueError(f'{name} must be a nonempty sorted unique integer sequence.')
    return result


def _array_digest(*arrays):
    digest=hashlib.sha256()
    for a in arrays:
        digest.update(str((a.shape,a.dtype.str)).encode());digest.update(a.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class FactorChunk:
    design: FramedBlockMap
    reference: np.ndarray

    def __post_init__(self):
        d=self.design
        if (not isinstance(d,FramedBlockMap) or not len(d.blocks)
                or not np.array_equal(d.rows,np.arange(d.rows.size).reshape(d.rows.shape))):
            raise ValueError('Chunk rows must be disjoint consecutive local factor rows.')
        if d.shape[0]!=d.rows.size:
            raise ValueError('Chunk output shape must contain exactly its factor rows.')
        reference=finite._vector(self.reference,d.shape[0]).copy();reference.setflags(write=False)
        object.__setattr__(self,'reference',reference)
        object.__setattr__(self,'_signature',self.signature())

    def signature(self):
        d=self.design
        return _array_digest(d.blocks,d.rows,d.columns,d.frame_hi,d.frame_lo,self.reference)

    def validate(self):
        if self.signature()!=self._signature:
            raise ValueError('An immutable factor chunk changed after validation.')

    @property
    def storage_bytes(self): return self.design.storage_bytes+self.reference.nbytes


class _MemoryChunks:
    def __init__(self,chunks):
        self.chunks=tuple(chunks)
        if not self.chunks or not all(isinstance(c,FactorChunk) for c in self.chunks):
            raise ValueError('At least one validated factor chunk is required.')
        sizes={c.design.shape[1] for c in self.chunks}
        if len(sizes)!=1: raise ValueError('All chunks must share the generalized space.')
        self.shape=(sum(c.design.shape[0] for c in self.chunks),sizes.pop())
        self.storage_bytes=sum(c.storage_bytes for c in self.chunks)

    def __iter__(self):
        for c in self.chunks:
            c.validate();yield c


class ChunkedFramedMap:
    """Logical row concatenation; physical factor arrays are never concatenated."""
    def __init__(self,chunks):
        self._provider=chunks if isinstance(chunks,_DiskChunks) else _MemoryChunks(chunks)
        self.shape=self._provider.shape

    @property
    def storage_bytes(self): return self._provider.storage_bytes

    def iter_chunks(self): return iter(self._provider)

    @_checked
    def __matmul__(self,value):
        y=finite._vector(value,self.shape[1]); result=np.empty(self.shape[0]);at=0
        for c in self.iter_chunks():
            n=c.design.shape[0];result[at:at+n]=c.design@y;at+=n
        return result

    def _transpose(self,values):
        hi=np.zeros(self.shape[1]);lo=hi.copy()
        for c,value in values:
            indices,ch,cl=c.design.rmatvec_contributions(value)
            hi[indices],lo[indices]=_add_pairs(hi[indices],lo[indices],ch,cl)
        result=hi+lo;finite._finite(result)
        return result

    @_checked
    def rmatvec(self,value):
        v=finite._vector(value,self.shape[0])
        def values():
            at=0
            for c in self.iter_chunks():
                n=c.design.shape[0];yield c,v[at:at+n];at+=n
        return self._transpose(values())

    @_checked
    def gram_action(self,value):
        y=finite._vector(value,self.shape[1])
        return self._transpose((c,c.design@y) for c in self.iter_chunks())

    @_checked
    def diagonal(self):
        hi=np.zeros(self.shape[1]);lo=hi.copy()
        for c in self.iter_chunks():
            # Local duplicate columns coalesce in the existing map. Distinct
            # chunks have distinct factor rows, hence no cross-row terms.
            value=c.design.diagonal();hi,lo=_add_pairs(hi,lo,value,np.zeros_like(value))
        result=hi+lo;finite._finite(result);return result

    @_checked
    def absolute_forward(self,value):
        y=finite._vector(value,self.shape[1]);result=np.empty(self.shape[0]);at=0
        for c in self.iter_chunks():
            n=c.design.shape[0];result[at:at+n]=c.design.absolute_forward(y);at+=n
        return result

    @_checked
    def absolute_action(self,value):
        y=finite._vector(value,self.shape[1]);result=np.zeros(self.shape[1])
        for c in self.iter_chunks(): result=_up(result+c.design.absolute_action(y))
        return result

    def to_dense(self):
        if self.shape[1]>256 or math.prod(self.shape)>2_000_000:
            raise ValueError('Dense conversion is limited to small parity audits.')
        return np.column_stack([self@v for v in np.eye(self.shape[1])])


@dataclass(frozen=True)
class _ChunkedGram:
    design: ChunkedFramedMap
    @property
    def shape(self): return (self.design.shape[1],)*2
    def __matmul__(self,value): return self.design.gram_action(value)
    def to_dense(self):
        if self.shape[0]>256: raise ValueError('Dense conversion is limited to small parity audits.')
        return np.column_stack([self@v for v in np.eye(self.shape[0])])


class ChunkedFiniteBasal:
    """FiniteBasal replacement protocol with fused streaming physical work."""
    def __init__(self,system,design,*,validate_source=None):
        if not isinstance(design,ChunkedFramedMap) or design.shape[1]!=len(system['load_n']):
            raise ValueError('Chunked basal map and system differ.')
        self.system_signature=_system_signature(system)
        self.basal_design=design;self._validate_source=validate_source
        self.hessian_n_s_m=_ChunkedGram(design)
        self.diagonal_n_s_m=design.diagonal()
        self.load_n=design._transpose((c,c.reference) for c in design.iter_chunks())
        self.constant_w=.5*math.fsum(float(c.reference@c.reference) for c in design.iter_chunks())
        finite._finite(self.diagonal_n_s_m,self.load_n,self.constant_w)
        self.diagonal_n_s_m.setflags(write=False);self.load_n.setflags(write=False)

    @property
    def reference_factor_sqrt_w(self):
        """Explicit diagnostic materialization; solver operations never call it."""
        result=np.empty(self.basal_design.shape[0]);at=0
        for c in self.basal_design.iter_chunks():
            n=len(c.reference);result[at:at+n]=c.reference;at+=n
        result.setflags(write=False);return result

    def validate_system(self,system):
        if self._validate_source is not None: self._validate_source(system)
        if _system_signature(system)!=self.system_signature:
            raise ValueError('Chunked basal operator belongs to a different joint geometry.')

    def absolute_action(self,value): return self.basal_design.absolute_action(value)

    @_checked
    def evaluate(self,value):
        y=finite._vector(value,self.basal_design.shape[1]);amounts=[]
        def values():
            for c in self.basal_design.iter_chunks():
                slip=c.design@y-c.reference;amounts.append(float(slip@slip));yield c,slip
        gradient=self.basal_design._transpose(values())
        potential=.5*math.fsum(amounts);finite._finite(potential,gradient);return potential,gradient

    @_checked
    def work(self,value):
        y=finite._vector(value,self.basal_design.shape[1]);amounts=[];upper=np.array(0.)
        for c in self.basal_design.iter_chunks():
            speed=c.design@y;slip=speed-c.reference
            bound=_up(c.design.absolute_forward(y)+abs(c.reference))
            gamma=len(bound)*np.finfo(float).eps
            if gamma>=.25:raise ValueError('Too many terms for the positive constituent bound.')
            chunk_upper=_up(np.sum(_up(bound*bound))/(1.-gamma))
            upper=_up(upper+chunk_upper)
            amounts.append((float(slip@slip),-float(slip@c.reference),float(slip@speed)))
        diss,mantle,gradient=(math.fsum(row[i] for row in amounts) for i in range(3))
        result=dict(basal_dissipation_w=diss,basal_reference_input_power_w=mantle,
            generalized_gradient_work_w=gradient,objective_w=.5*diss,arithmetic_constituent_work_bound_w=float(upper))
        finite._finite(*result.values());return result


@contextmanager
def _writer(path):
    """OS lock is released on process exit; no stale-PID lock guessing."""
    with open(path/'.writer.lock','a+b') as stream:
        if stream.seek(0,2)==0: stream.write(b'0');stream.flush()
        stream.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: yield
        finally:
            stream.seek(0)
            if os.name=='nt': msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def _atomic_json(path,value):
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with open(temp,'xb') as f: f.write(_json(value));f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


class FactorStore:
    """Single-writer, checksummed batches with explicit incomplete state."""
    def __init__(self,path,*,binding,expected_control_ids,unknowns,max_chunk_bytes,max_store_bytes,create=False):
        self.path=Path(path).resolve()
        self._base=dict(schema=1,binding=json.loads(_json(binding)),
            expected_controls=list(_ids(expected_control_ids,'expected controls')),
            unknowns=_integer(unknowns,'unknowns',1),max_chunk_bytes=_integer(max_chunk_bytes,'max_chunk_bytes',1),
            max_store_bytes=_integer(max_store_bytes,'max_store_bytes',1))
        self._base_bytes=_json(self._base)
        if create:
            self.path.mkdir(parents=True,exist_ok=True)
            with _writer(self.path):
                if (self.path/'manifest.json').exists(): raise ValueError('Store already exists; open explicitly to resume.')
                _atomic_json(self.path/'manifest.json',dict(self._base,state='BUILDING',chunks=[],certificate=None))
        self._raw,self._manifest=self._read()

    def _read(self):
        path=self.path/'manifest.json'
        if path.stat().st_size>1_000_000+1024*len(self._base['expected_controls']):
            raise ValueError('Manifest exceeds the declared control-universe bound.')
        raw=path.read_bytes();m=json.loads(raw)
        if _json({k:m.get(k) for k in self._base})!=self._base_bytes:
            raise ValueError('Store source/state/policy/configuration binding differs.')
        if set(m)!=set(self._base)|{'state','chunks','certificate'} or m['state'] not in ('BUILDING','SEALED'):
            raise ValueError('Invalid store schema or state.')
        completed=set();seen_files=set();total=0
        for row in m['chunks']:
            if set(row)!={'file','sha256','bytes','blocks','control_ids'}:
                raise ValueError('Invalid chunk descriptor.')
            ids=_ids(row['control_ids'],'chunk controls')
            if not set(ids)<=set(self._base['expected_controls']) or completed.intersection(ids):
                raise ValueError('Chunk controls are missing, foreign or duplicated.')
            if (not isinstance(row['file'],str) or re.fullmatch(r'chunk-[0-9a-f]{32}\.npz',row['file']) is None
                    or row['file'] in seen_files or re.fullmatch(r'[0-9a-f]{64}',row['sha256']) is None):
                raise ValueError('Invalid chunk path or content hash.')
            size=_integer(row['bytes'],'chunk bytes',1);_integer(row['blocks'],'blocks',1)
            if size>self._base['max_chunk_bytes']: raise ValueError('Chunk exceeds resource limit.')
            completed.update(ids);seen_files.add(row['file']);total+=size
        if total>self._base['max_store_bytes']: raise ValueError('Store exceeds resource limit.')
        if m['state']=='SEALED' and completed!=set(self._base['expected_controls']):
            raise ValueError('A partial store cannot be sealed.')
        if m['state']=='BUILDING' and m['certificate'] is not None: raise ValueError('Incomplete store has a completion certificate.')
        return raw,m

    def _unchanged(self):
        raw,_=self._read()
        if (raw!=self._raw or _json(self._manifest)!=_json(json.loads(raw))
                or _json(self._base)!=self._base_bytes):
            raise ValueError('Store manifest or prepared binding changed; reopen to resume or verify.')

    @property
    def completed_controls(self):
        return tuple(sorted(i for row in self._manifest['chunks'] for i in row['control_ids']))

    @property
    def sealed(self): return self._manifest['state']=='SEALED'

    @property
    def manifest_sha256(self): return _hash(self._raw)

    def _load(self,row):
        path=self.path/row['file']
        if path.resolve().parent!=self.path: raise ValueError('Chunk leaves its bound store directory.')
        with open(path,'rb') as f:
            if os.fstat(f.fileno()).st_size!=row['bytes']: raise ValueError('Chunk byte length changed.')
            # One bounded immutable byte snapshot closes the hash/load race;
            # no file is parsed again after verifying different bytes.
            data=f.read(row['bytes']+1)
        if len(data)!=row['bytes'] or _hash(data)!=row['sha256']: raise ValueError('Chunk checksum mismatch.')
        with io.BytesIO(data) as f:
            n=row['blocks'];shapes={'blocks':(n,9,9),'columns':(n,9),'frame_hi':(n,3,3),
                'frame_lo':(n,3,3),'reference':(9*n,),'metadata':None}
            with zipfile.ZipFile(f) as z:
                if sorted(z.namelist())!=sorted(k+'.npy' for k in shapes): raise ValueError('Unexpected archive members.')
                if sum(info.file_size for info in z.infolist())>self._base['max_chunk_bytes']:
                    raise ValueError('Uncompressed archive exceeds the factor resource limit.')
                for name,shape in shapes.items():
                    info=z.getinfo(name+'.npy')
                    if info.file_size>self._base['max_chunk_bytes']: raise ValueError('Archive member exceeds resource limit.')
                    with z.open(info) as part:
                        version=np.lib.format.read_magic(part)
                        if version not in ((1,0),(2,0)): raise ValueError('Unsupported array header.')
                        reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
                        actual,fortran,dtype=reader(part)
                        expected=np.dtype('uint8' if name=='metadata' else 'int64' if name=='columns' else 'float64')
                        if (dtype!=expected or fortran or (shape is not None and actual!=shape)
                                or (shape is None and len(actual)!=1)
                                or part.tell()+math.prod(actual)*dtype.itemsize!=info.file_size):
                            raise ValueError('Archive array type/shape/payload is inconsistent.')
            f.seek(0)
            with np.load(f,allow_pickle=False) as a:
                metadata=json.loads(a['metadata'].tobytes())
                d=FramedBlockMap(a['blocks'],np.arange(9*n).reshape(n,9),a['columns'],
                    (9*n,self._base['unknowns']),a['frame_hi'],a['frame_lo'])
                chunk=FactorChunk(d,a['reference'])
        return chunk,metadata

    def iter_batches(self):
        self._unchanged()
        for row in self._manifest['chunks']: yield row,*self._load(row)
        self._unchanged()

    def append(self,control_ids,chunk,metadata):
        ids=_ids(control_ids,'chunk controls');chunk.validate()
        if chunk.design.shape[1]!=self._base['unknowns']: raise ValueError('Chunk has different generalized space.')
        encoded=_json(metadata)
        with _writer(self.path):
            self._unchanged()
            if self.sealed: raise ValueError('A sealed store is immutable.')
            if not set(ids)<=set(self._base['expected_controls']) or set(ids).intersection(self.completed_controls):
                raise ValueError('Chunk controls overlap completed work or leave the expected universe.')
            d=chunk.design
            arrays=dict(blocks=d.blocks,columns=d.columns,frame_hi=d.frame_hi,frame_lo=d.frame_lo,
                        reference=chunk.reference,metadata=np.frombuffer(encoded,dtype=np.uint8))
            estimate=sum(a.nbytes for a in arrays.values())+4096
            # Include both uncommitted complete archives and interrupted .tmp
            # files; neither is completed work, but both consume disk space.
            used=sum(p.stat().st_size for p in self.path.glob('chunk-*.npz*'))
            if estimate>self._base['max_chunk_bytes'] or used+estimate>self._base['max_store_bytes']:
                raise ValueError('Factor resource limit reached; completed batches remain intact.')
            name='chunk-'+uuid.uuid4().hex+'.npz';temp=self.path/(name+'.tmp')
            with open(temp,'xb') as f: np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
            size=temp.stat().st_size
            if size>self._base['max_chunk_bytes'] or used+size>self._base['max_store_bytes']:
                raise ValueError('Serialized factor resource limit reached; completed batches remain intact.')
            digest=hashlib.sha256()
            with open(temp,'rb') as f:
                for part in iter(lambda:f.read(1024*1024),b''):digest.update(part)
            os.replace(temp,self.path/name)
            row=dict(file=name,sha256=digest.hexdigest(),bytes=size,blocks=len(d.blocks),control_ids=list(ids))
            manifest=dict(self._manifest,chunks=self._manifest['chunks']+[row])
            _atomic_json(self.path/'manifest.json',manifest)
            self._raw,self._manifest=self._read()

    def seal(self,certificate):
        with _writer(self.path):
            self._unchanged()
            if self.sealed: raise ValueError('Store is already sealed.')
            if self.completed_controls!=tuple(self._base['expected_controls']):
                raise ValueError('Cannot seal until every expected unique control is completed.')
            for _ in self.iter_batches(): pass
            _atomic_json(self.path/'manifest.json',dict(self._manifest,state='SEALED',certificate=json.loads(_json(certificate))))
            self._raw,self._manifest=self._read()

    def factor_map(self):
        if not self.sealed: raise ValueError('Partial factor store has no complete operator.')
        return ChunkedFramedMap(_DiskChunks(self))


class _DiskChunks:
    def __init__(self,store):
        self.store=store
        self.shape=(9*sum(row['blocks'] for row in store._manifest['chunks']),store._base['unknowns'])
        # Resident provider metadata only. Disk bytes and peak loaded-chunk
        # payload are reported separately; neither is a measured process RSS.
        self.storage_bytes=len(store._raw)
        self.disk_bytes=sum(row['bytes'] for row in store._manifest['chunks'])
        self.maximum_chunk_file_bytes=max((row['bytes'] for row in store._manifest['chunks']),default=0)
    def __iter__(self):
        for _,chunk,_ in self.store.iter_batches(): yield chunk


def _source_binding():
    root=Path(__file__).resolve().parents[2]
    paths=sorted(set(root.glob('*.py'))|set((root/'benchmarks').rglob('*.py')))
    return dict(files={p.relative_to(root).as_posix():_hash(p.read_bytes()) for p in paths},
                python=list(sys.version_info[:3]),numpy=np.__version__)


@dataclass(frozen=True)
class NativeBuildProgress:
    path: str
    completed_controls: tuple
    expected_control_count: int
    manifest_sha256: str
    global_allocation_complete: bool=False


def _failure_witness(error,control_ids):
    """Preserve bounded geometry/order evidence from the unchanged integrator."""
    result=dict(original_witness=getattr(error,'witness',None),control_ids=list(control_ids))
    tb=error.__traceback__
    while tb is not None:
        frame=tb.tb_frame
        if Path(frame.f_code.co_filename).resolve()==Path(finite.__file__).resolve():
            local=frame.f_locals
            if frame.f_code.co_name=='_build':
                index=local.get('cell_index')
                if isinstance(index,int) and 0<=index<len(control_ids):result['control_id']=control_ids[index]
                if 'piece_index' in local:result['piece_index']=int(local['piece_index'])
                piece=local.get('piece',{})
                result['piece']={k:piece[k] for k in ('bottom_face','owner','area_m2') if k in piece}
                geometry=local.get('geometry')
                if geometry is not None:
                    result['exact_chart_polygon']=[[str(v) for v in p] for p in geometry.chart_polygon]
                    result['control_triangle']=geometry.control_triangle.tolist()
            if frame.f_code.co_name=='_material_factor':
                result['quadrature_records']=local.get('records',[])
                result['quadrature_orders']=list(local.get('orders',()))
                result['quadrature_relative_tolerance']=float(local['tolerance'])
                result['material_vertex_indices']=np.asarray(local['face']).tolist()
        tb=tb.tb_next
    return result


def _validate_native_summary(domain,row,summary,tolerance):
    required={'control_ids','seen_face_ids','area_m2','positive_piece_count',
        'maximum_cell_metric_closure','per_owner_legacy_metric_change_m2','integration_diagnostics'}
    if (not isinstance(summary,dict) or set(summary)!=required
            or summary['control_ids']!=row['control_ids']):
        raise ValueError('Native batch summaries do not match their completed controls.')
    seen=tuple(_integer(i,'covered face ID') for i in summary['seen_face_ids'])
    if tuple(sorted(set(seen)))!=seen or not set(seen)<=set(domain.face_id_to_index):
        raise ValueError('Stored native material coverage has invalid identity.')
    _integer(summary['positive_piece_count'],'positive piece count',1)
    measured=finite._real(summary['area_m2'],'stored native cell areas')
    expected=domain._data['saved_area_m2'][row['control_ids']]
    closure=finite._real(summary['maximum_cell_metric_closure'],'stored native metric closure')
    if (measured.shape!=expected.shape or np.any(measured<=0.)
            or np.any(abs(measured-expected)>2e-10*expected)
            or closure.shape!=() or closure<0. or closure>2e-10):
        raise ValueError('Stored native cells fail their individual measure/work closure.')
    deltas=summary['per_owner_legacy_metric_change_m2']
    if set(deltas)!={str(k) for k in domain.owner_to_plate_slot}:
        raise ValueError('Stored native owner diagnostic identities differ.')
    for value in deltas.values():
        if finite._real(value,'stored owner metric change').shape!=(3,3):
            raise ValueError('Stored owner metric change has an invalid shape.')
    records=summary['integration_diagnostics']
    if not isinstance(records,list) or len(records)!=row['blocks']:
        raise ValueError('Stored integration diagnostics do not cover every factor.')
    for record in records:
        local=_integer(record['cell'],'diagnostic local cell')
        if (local>=len(row['control_ids']) or record['control_id']!=row['control_ids'][local]
                or record['policy']!=domain.allocation_policy):
            raise ValueError('Stored integration diagnostic control/policy differs.')
        _integer(record['piece'],'diagnostic piece')
        finite._positive(record['area_m2'],'diagnostic area')
        if record['kind']=='bottom-material':
            if record['bottom_face'] not in domain.face_id_to_index or not record['quadrature']:
                raise ValueError('Stored material integration diagnostic is incomplete.')
            last=record['quadrature'][-1]
            for key in ('operator_relative_change','area_relative_error',
                        'full_rigid_metric_relative_error','compressed_rigid_metric_relative_error'):
                value=finite._real(last[key],'stored quadrature acceptance')
                if value.shape!=() or value<0. or value>tolerance:
                    raise ValueError('Stored material integration does not meet its acceptance gate.')
        elif record['kind']=='uncovered-rigid':
            if record['owner'] not in domain.owner_to_plate_slot:
                raise ValueError('Stored uncovered factor has an invalid owner.')
            if finite._positive(record['fraction'],'diagnostic fraction')>1.:
                raise ValueError('Stored uncovered factor has an invalid support fraction.')
        else:raise ValueError('Stored integration diagnostic has an unknown region type.')


def build_native(domain,system,path,*,mantle_omega_rad_s,quadrature_relative_tolerance,max_order,
                 max_chunk_bytes,max_store_bytes,controls_per_batch=16,stop_after_batches=None):
    """Resume exact accepted batches; partial progress never exposes an operator.

    Per-cell geometry/ordering and all finite quadrature gates are unchanged.
    Resource caps bound retained batches, not one indivisible cell's transient
    exact partition scratch; a completed oversized cell is rejected, not cut.
    """
    from .native_basal_domain import NativeBasalBuild,NativeBasalBlocker
    domain.validate_source();domain.validate_system(system)
    controls_per_batch=_integer(controls_per_batch,'controls_per_batch',1)
    if stop_after_batches is not None: stop_after_batches=_integer(stop_after_batches,'stop_after_batches',1)
    tolerance=finite._positive(quadrature_relative_tolerance,'quadrature_relative_tolerance')
    order=_integer(max_order,'max_order',1)
    if tolerance>=.01 or order not in (16,32,64,128):raise ValueError('Finite quadrature settings are outside their supported domain.')
    mantle=finite._real(mantle_omega_rad_s,'mantle_omega_rad_s')
    if mantle.shape!=(3,): raise ValueError('Explicit mantle Euler reference must have three components.')
    count=len(domain._controls)
    binding=dict(source=_source_binding(),state=domain.source_signature,derived=domain._prepared_signature,
        system=_system_signature(system),allocation_policy=domain.allocation_policy,
        mantle_omega_rad_s=mantle.tolist(),quadrature_relative_tolerance=tolerance,
        max_order=order,controls_per_batch=controls_per_batch)
    def validate(current):
        domain.validate_source();domain.validate_system(current)
        if _source_binding()!=binding['source']:raise ValueError('Basal numerical source changed.')
    create=not (Path(path)/'manifest.json').exists()
    store=FactorStore(path,binding=binding,expected_control_ids=range(count),unknowns=len(system['load_n']),
        max_chunk_bytes=max_chunk_bytes,max_store_bytes=max_store_bytes,create=create)
    # Verify saved work before spending time on any new exact partition.
    for row,_,summary in store.iter_batches():_validate_native_summary(domain,row,summary,tolerance)
    completed=set(store.completed_controls);remaining=[i for i in range(count) if i not in completed]
    batches=0
    for at in range(0,len(remaining),controls_per_batch):
        ids=remaining[at:at+controls_per_batch]
        try:
            allocations=[domain._cell(i) for i in ids]
            component=finite.build(system,allocations,domain.face_id_to_index,
                owner_to_plate_slot=domain.owner_to_plate_slot,mantle_omega_rad_s=mantle,
                quadrature_relative_tolerance=quadrature_relative_tolerance,max_order=max_order)
            seen=sorted({f for a in allocations for p in a['pieces'] for f in p['cover_faces']})
            summary=dict(control_ids=ids,seen_face_ids=seen,
                area_m2=[math.fsum(p['area_m2'] for p in a['pieces']) for a in allocations],
                positive_piece_count=sum(len(a['pieces']) for a in allocations),
                maximum_cell_metric_closure=max(max(a['scaled_metric_closure_error'],a['scaled_common_rotation_closure_error']) for a in allocations),
                per_owner_legacy_metric_change_m2={str(owner):sum((a['per_owner_legacy_metric_change_m2'][owner] for a in allocations),np.zeros((3,3))).tolist() for owner in domain.owner_to_plate_slot})
            summary['integration_diagnostics']=[dict(row,control_id=ids[row['cell']]) for row in component.diagnostics]
            validate(system)
            store.append(ids,FactorChunk(component.basal_design,component.reference_factor_sqrt_w),summary)
            del component,allocations,summary
        except (ValueError,RuntimeError) as error:
            _atomic_json(store.path/'failure.json',dict(control_ids=ids,error=str(error),
                witness=_failure_witness(error,ids),manifest_sha256=store.manifest_sha256))
            raise
        batches+=1
        if stop_after_batches is not None and batches>=stop_after_batches and at+len(ids)<len(remaining):
            return NativeBuildProgress(str(store.path),store.completed_controls,count,store.manifest_sha256)
    seen=set();areas=[];piece_count=0;largest=0.;deltas={str(k):np.zeros((3,3)) for k in domain.owner_to_plate_slot}
    for row,_,summary in store.iter_batches():
        _validate_native_summary(domain,row,summary,tolerance)
        seen.update(summary['seen_face_ids']);areas.extend(summary['area_m2']);piece_count+=summary['positive_piece_count']
        largest=max(largest,summary['maximum_cell_metric_closure'])
        for owner in deltas:
            deltas[owner]+=summary['per_owner_legacy_metric_change_m2'][owner]
    total=math.fsum(domain._data['saved_area_m2']);area_sum=math.fsum(areas)
    if seen!=set(domain.face_id_to_index) or abs(area_sum-total)>2e-10*total or largest>2e-10:
        raise ValueError('Complete native material/area/metric closure was not established.')
    validate(system)
    certificate=dict(domain.control_certificate,global_allocation_complete=True,source_signature=domain.source_signature,
        allocation_policy=domain.allocation_policy,active_native_slots=domain.active_native_slots.tolist(),
        material_faces_covered=len(seen),positive_piece_count=piece_count,native_area_m2=total,
        area_closure_error_m2=area_sum-total,maximum_cell_metric_closure=largest,
        per_owner_legacy_metric_change_m2={k:v.tolist() for k,v in deltas.items()},
        candidate_policy='unchanged native exact positive candidate partition; no area floor',
        source_binding=binding['source'])
    if not store.sealed:store.seal(certificate)
    elif _json(store._manifest['certificate'])!=_json(certificate):raise ValueError('Saved completion certificate differs.')
    result=ChunkedFiniteBasal(system,store.factor_map(),validate_source=validate)
    validate(system)
    return NativeBasalBuild(result,dict(certificate,manifest_sha256=store.manifest_sha256,store_path=str(store.path)))
