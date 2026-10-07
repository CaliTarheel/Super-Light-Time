"""Retire observer-only fine solver traces at an explicit coarse source boundary.

Detailed historical evidence stays in the original accepted checkpoint. No
geometry, force/slab/material ledger, area credit, RNG, clock or config is changed.
"""
from copy import deepcopy
import hashlib,json,math,re
import numpy as np
VERSION=1
FIELD='coarse_diagnostic_compaction'

def _value(value):
    if isinstance(value,np.ndarray):
        if value.dtype.hasobject:raise ValueError('Object diagnostic arrays unsupported')
        return dict(kind='array',dtype=value.dtype.str,shape=list(value.shape),sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
    if isinstance(value,np.generic):return _value(value.item())
    if value is None or type(value) in (bool,int,str):return value
    if type(value) is float:
        if not math.isfinite(value):raise ValueError('Nonfinite diagnostic scalar')
        return dict(kind='float',hex=value.hex())
    if type(value) is dict:return dict(kind='dict',items=[[_value(k),_value(v)] for k,v in value.items()])
    if type(value) in (list,tuple):return dict(kind=type(value).__name__,items=[_value(v) for v in value])
    raise ValueError('Unsupported diagnostic value: '+type(value).__name__)

def _digest(value):
    return hashlib.sha256(json.dumps(_value(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def _scalars(value):
    result={}
    for name,item in value.items():
        if not isinstance(name,str):continue
        if isinstance(item,np.generic):item=item.item()
        if item is None or type(item) in (bool,int) or type(item) is float and math.isfinite(item):result[name]=item
        elif type(item) is str:result[name]=item[:2000]
    return result

def compact(s,source_receipt):
    """Mutate only two observer attributes after validating all source evidence."""
    if not isinstance(source_receipt,dict):raise ValueError('Detailed historical source reference required')
    pin=source_receipt.get('checkpoint_sha256');path=source_receipt.get('checkpoint_file');run=source_receipt.get('run_id')
    if not (type(pin) is str and re.fullmatch('[0-9a-f]{64}',pin) and type(path) is str and path and type(run) is str and run):
        raise ValueError('Exact historical checkpoint/run reference required')
    epoch=float(s.t);steps=int(s.steps)
    if not math.isfinite(epoch) or epoch<0 or steps<0:raise ValueError('Invalid accepted epoch')
    source=dict(run_id=run,checkpoint_file=path,checkpoint_sha256=pin,diagnostic_field='deformation_diagnostics')
    prior=getattr(s,FIELD,None)
    if prior is not None:
        if (not isinstance(prior,dict) or prior.get('version')!=VERSION or prior.get('epoch_myr')!=epoch
                or prior.get('detailed_source')!=source):raise ValueError('Different diagnostic compaction already recorded')
        return deepcopy(prior)
    original=getattr(s,'deformation_diagnostics',None)
    if not isinstance(original,dict):raise ValueError('Historical deformation diagnostics must be a dictionary')
    old_boundary=getattr(s,'production_restart_migration',{})
    if isinstance(old_boundary,dict) and old_boundary.get('epoch_myr')==epoch:
        raise ValueError('An active earlier restart-boundary diagnostic contract needs explicit migration')
    summary=dict(epoch_myr=epoch,integration_steps=steps,top_level_scalars=_scalars(original),
        nested_scalar_summaries={name:_scalars(value) for name,value in original.items()
            if isinstance(name,str) and isinstance(value,dict)},
        top_level_sequence_lengths={name:len(value) for name,value in original.items()
            if isinstance(name,str) and isinstance(value,(list,tuple))})
    report=dict(version=VERSION,epoch_myr=epoch,integration_steps=steps,physical_time_advanced_myr=0.,
        detailed_source=source,previous_diagnostic_sha256=_digest(original),summary=summary,
        scope='Observer-only deformation/solver trace replaced by explicit coarse initialization view; detailed original accepted checkpoint retained.',
        mutated_fields=['deformation_diagnostics',FIELD],mechanical_ledgers_changed=False,
        geometry_changed=False,config_changed=False,rng_changed=False,
        diagnostic_digest_scope='Typed diagnostic value tree and array bytes; not an alias graph.')
    current=dict(model='rigid material transport',deforming_vertices=0,initialized=True,
        physical_time_advanced_myr=0.,mechanical_solve_performed=False,
        interpretation='Coarse-mode source initialization, not a completed successor interval.',
        inherited_epoch_summary=deepcopy(summary),historical_details_reference=deepcopy(source))
    # All hashing/validation precedes the two observer assignments.
    s.deformation_diagnostics=current
    setattr(s,FIELD,report)
    return deepcopy(report)
