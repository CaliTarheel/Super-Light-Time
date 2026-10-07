"""Conservative neck channels carried by optional slab-memory records.

Channels preserve separate constitutive histories; they are never replaced by
an average damage value. These helpers do not enable a new force/polarity policy.
"""
from copy import deepcopy
from dataclasses import asdict
import math
import numpy as np
from slab_tether import Neck

FIELD='slab_tether_channels'
VERSION_FIELD='slab_tether_history_version'
INVENTORIES=('retained_area_km2','retained_buoyancy_area_km2','retained_excess_mass_kg')
EXTENSIVE=('reference_width_m',*INVENTORIES,'detached_area_km2','detached_excess_mass_kg')


def validate(row):
    if FIELD not in row and VERSION_FIELD not in row:return
    version=row.get(VERSION_FIELD)
    if isinstance(version,(bool,np.bool_)) or not isinstance(version,(int,np.integer)) or version not in (1,2) or FIELD not in row:
        raise ValueError('Slab neck channels require their explicit history version.')
    if any('slab_'+key not in row for key in INVENTORIES) or 'slab_retired_excess_mass_kg' not in row:
        raise ValueError('Slab neck history requires a conservative mass inventory.')
    channels=row[FIELD]
    import slab_tether_local as local
    extensive=local.EXTENSIVE if version==2 else EXTENSIVE
    geometry=set(local.GEOMETRY) if version==2 else set()
    if not isinstance(channels,list):raise ValueError('Slab neck channels must be explicit records.')
    for channel in channels:
        schema=set(extensive)|{'neck'}|geometry
        allowed=(schema,schema-{local.INITIAL_AREA}) if version==2 else (schema,)
        if not isinstance(channel,dict) or set(channel) not in allowed:
            raise ValueError('Slab neck channel schema is incomplete.')
        Neck(**channel['neck'])
        values=[channel.get(k,0.) if version==2 and k==local.INITIAL_AREA else channel[k] for k in extensive]
        if any(isinstance(value,(bool,np.bool_)) or not np.isscalar(value)
               or not np.isfinite(value) or value<0 for value in values):
            raise ValueError('Slab neck channel inventories must be finite and nonnegative.')
        if channel['reference_width_m']==0. and any(channel[k]>0 for k in INVENTORIES):
            raise ValueError('A retained slab neck requires a positive reference trace width.')
        if channel['neck']['damage']==1. and any(channel[k]>0 for k in INVENTORIES):
            raise ValueError('Ruptured slab inventory must be retired in the same transaction.')
        if channel['retained_buoyancy_area_km2']>1.8*channel['retained_area_km2']+1e-7:
            raise ValueError('Slab neck buoyancy proxy exceeds its admitted area bound.')
        if version==2:local.validate_channel(channel)
    for key in (local.LEDGER if version==2 else INVENTORIES):
        represented=math.fsum(c.get(key,0.) if version==2 and key==local.INITIAL_AREA else c[key] for c in channels)
        total=float(row.get('slab_'+key,0.) if version==2 and key==local.INITIAL_AREA else row['slab_'+key])
        if not math.isclose(represented,total,rel_tol=2e-11,abs_tol=1e-7 if 'area' in key else 1.):
            raise ValueError('Slab neck channels disagree with conserved '+key)
    for key in ('area_km2','excess_mass_kg'):
        detached=math.fsum(c['detached_'+key] for c in channels)
        retired=float(row['slab_retired_'+key])
        if detached>retired+max(1e-7,retired*2e-11):
            raise ValueError('Detached neck material must remain in the slab retirement ledger.')


def initialize(row,neck):
    """Explicit current-state baseline; no reconstructed damage or past rupture."""
    import slab_memory as slab
    slab.validate_row(row,require_mass=True)
    if FIELD in row:return deepcopy(row[FIELD])
    width=float(row.get('length_km',0.))*1000.
    if not math.isfinite(width) or width<=0 or neck.damage==1.:
        raise ValueError('Initialize an attached neck on an explicit positive trace width.')
    channel=dict(neck=asdict(neck),reference_width_m=width,detached_area_km2=0.,detached_excess_mass_kg=0.)
    channel.update({key:float(row['slab_'+key]) for key in INVENTORIES})
    staged=dict(row,**{FIELD:[channel],VERSION_FIELD:1})
    validate(staged)
    row[FIELD]=[channel]
    row[VERSION_FIELD]=1
    return deepcopy(row[FIELD])


def partition_data(parent,child,fraction,channel_fractions=None):
    """Stage channel shares BEFORE the containing slab inventories are split."""
    if FIELD not in parent and FIELD not in child:return None
    if FIELD not in parent or child.get(FIELD):
        raise ValueError('Slab neck split needs a channel parent and empty daughter.')
    validate(parent)
    import slab_tether_local as local
    if local.enabled(parent):
        shares=np.asarray(channel_fractions,float)
        if shares.shape!=(len(parent[FIELD]),) or not np.isfinite(shares).all() or np.any((shares<0)|(shares>1)):
            raise ValueError('Local neck splits require a resolved fraction for every patch.')
        extensive=local.EXTENSIVE
    else:
        shares=np.full(len(parent[FIELD]),fraction)
        extensive=EXTENSIVE
    left,right=deepcopy(parent[FIELD]),deepcopy(parent[FIELD])
    for a,b,share in zip(left,right,shares):
        for key in extensive:
            value=a.get(key,0.) if local.enabled(parent) and key==local.INITIAL_AREA else a[key]
            b[key]=value*share
            a[key]=value-b[key]
    return left,right


def join_data(source,target):
    """Stage a channel union; never average damage, force or rheology."""
    if FIELD not in source and FIELD not in target:return None
    if FIELD not in source or FIELD not in target:
        raise ValueError('Cannot silently join neck-enabled and uninitialized slab histories.')
    if source.get(VERSION_FIELD)!=target.get(VERSION_FIELD):
        raise ValueError('Cannot mix local and unlocalized slab neck histories on join.')
    validate(source);validate(target)
    return deepcopy(target[FIELD])+deepcopy(source[FIELD])


def advance(row,decay,retained_area_gain,retained_buoyancy_gain,retained_mass_gain):
    """Apply the SAME exact source-step retention as the containing slab row.

    Call after the parent row has advanced. Feed enters continuing channels in
    proportion to their reference width. A ruptured channel cannot reattach by
    receiving feed; creating a new system requires a new explicit neck history.
    """
    if FIELD not in row:return
    import slab_tether_local as local
    if local.enabled(row):
        if any(v!=0 for v in (retained_area_gain,retained_buoyancy_gain,retained_mass_gain)):
            raise ValueError('Local neck feeding requires resolved capture provenance; uniform trench feed is unsupported.')
        if not np.isfinite(decay) or not 0<=decay<=1:
            raise ValueError('Slab retention requires bounded finite decay.')
        staged=deepcopy(row)
        local.retire(staged,decay)
        validate(staged)
        row[FIELD]=staged[FIELD]
        return
    gains=np.array([retained_area_gain,retained_buoyancy_gain,retained_mass_gain])
    if not 0<=decay<=1 or not np.isfinite(decay) or not np.isfinite(gains).all() or np.any(gains<0):
        raise ValueError('Slab neck retention requires bounded decay and nonnegative admitted gains.')
    channels=deepcopy(row[FIELD])
    widths=np.array([c['reference_width_m'] if c['neck']['damage']<1. else 0. for c in channels])
    total=math.fsum(widths)
    if np.any(gains>0) and total<=0:
        raise ValueError('New slab feed after rupture requires an explicitly initiated neck.')
    for channel,width in zip(channels,widths):
        share=width/total if total>0 else 0.
        for key,gain in zip(INVENTORIES,gains):channel[key]=channel[key]*decay+share*float(gain)
    staged=dict(row,**{FIELD:channels})
    validate(staged)
    row[FIELD]=channels


def rupture(row,index,failed_neck):
    """Retire a mechanically failed channel atomically without losing its mass.

    The caller has localized the event with slab_tether.advance. This transaction
    marks that channel fully damaged and transfers its area/excess mass from the
    retained to the retired ledger. It does not authorize a polarity reversal.
    """
    import slab_memory as slab
    slab.validate_row(row,require_mass=True)
    if (isinstance(index,(bool,np.bool_)) or not isinstance(index,(int,np.integer))
            or index<0 or index>=len(row.get(FIELD,[]))):
        raise ValueError('Slab rupture needs an existing channel index.')
    staged=deepcopy(row)
    channel=staged[FIELD][int(index)]
    failed=asdict(failed_neck)
    if failed['damage']!=1. or any(failed[k]!=channel['neck'][k] for k in failed if k!='damage'):
        raise ValueError('Slab rupture requires the failed constitutive state of this same neck.')
    area=channel['retained_area_km2'];mass=channel['retained_excess_mass_kg']
    for key in INVENTORIES:
        channel[key]=0.
        # Channels are the extensive partition. Re-sum survivors instead of
        # repeatedly subtracting from the row: cancellation otherwise leaves
        # a spurious signed remnant when the last of many channels ruptures.
        staged['slab_'+key]=math.fsum(c[key] for c in staged[FIELD])
    staged['slab_retired_area_km2']+=area
    staged['slab_retired_excess_mass_kg']+=mass
    channel['detached_area_km2']+=area
    channel['detached_excess_mass_kg']+=mass
    import slab_tether_local as local
    if local.enabled(staged):
        channel['retired_area_km2']+=area
        channel['retired_excess_mass_kg']+=mass
    channel['neck']=failed
    slab.refresh_line_load(staged)
    slab.validate_row(staged,require_mass=True)
    row.clear();row.update(staged)
    return dict(detached_area_km2=area,detached_excess_mass_kg=mass)
