"""Explicit future-only numerical accuracy and persistent dimensional area credit.

These are numerical admission budgets, not changes to force, column, speed or
shape laws. Plain column arrays are checkpointed; immutable policy objects are
derived for a single mechanical transaction and are never saved in Simulation.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np

VERSION = 1
KIND = 'material_identity_numerical_accuracy'
OUTER_RELATIVE_TOLERANCE = 1e-6
INVERSE_RELATIVE_TOLERANCE = 1e-6
ABSOLUTE_AREA_CAP_KM2 = 2.
REFERENCE_FACE_AREA_KM2 = 95.
REFERENCE = 'area_accuracy_reference_km2'
BUDGET = 'area_accuracy_budget_km2'
SPENT = 'area_accuracy_spent_km2'
AREA = 'area_accuracy_current_area_km2'
FIELDS = (REFERENCE, BUDGET, SPENT, AREA)
EXTENSIVE_FIELDS = (REFERENCE, BUDGET, SPENT)
FRAME_FIELDS={'material_'+name:name for name in FIELDS}
RESTORED_FRAME_FIELD='material_area_accuracy_restored_thickness_km'
ARRAY_FIELDS={name:np.float64 for name in FRAME_FIELDS}
ARRAY_FIELDS[RESTORED_FRAME_FIELD]=np.float64


def version(s):
    value = getattr(s, 'numerical_accuracy_version', 0)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value not in (0, VERSION):
        raise ValueError('Unsupported material numerical accuracy policy.')
    return int(value)


def present(state):
    found = [name in state for name in FIELDS]
    if any(found) and not all(found):
        raise ValueError('Incomplete dimensional column accuracy metadata.')
    return all(found)


def allowance(reference):
    return np.minimum(ABSOLUTE_AREA_CAP_KM2,
        ABSOLUTE_AREA_CAP_KM2/REFERENCE_FACE_AREA_KM2*np.asarray(reference, float))


def _array(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError('Invalid dimensional numerical accuracy array: '+name)
    # A bytes-backed array cannot be made writeable by a borrower.
    return np.frombuffer(array.tobytes(), dtype=np.float64)


def nominal_miss(area, minimum, maximum):
    return np.maximum(np.asarray(minimum)-np.asarray(area), 0.) + np.maximum(np.asarray(area)-np.asarray(maximum), 0.)


@dataclass(frozen=True)
class NumericalPolicy:
    reference_area_km2: np.ndarray
    budget_km2: np.ndarray
    spent_km2: np.ndarray
    current_area_km2: np.ndarray
    minimum_area_km2: np.ndarray
    maximum_area_km2: np.ndarray
    reference_identity: str
    outer_tolerance: float = OUTER_RELATIVE_TOLERANCE
    inverse_tolerance: float = INVERSE_RELATIVE_TOLERANCE
    # Contact transaction only. These stage anchors are not column ledger fields,
    # are never checkpointed, and must survive every regional/substep advance.
    hard_minimum_area_km2: np.ndarray | None = None
    hard_maximum_area_km2: np.ndarray | None = None
    hard_arithmetic_area_km2: np.ndarray | None = None
    hard_stage_area_km2: np.ndarray | None = None

    def __post_init__(self):
        for name in ('reference_area_km2','budget_km2','spent_km2','current_area_km2','minimum_area_km2','maximum_area_km2'):
            object.__setattr__(self, name, _array(getattr(self, name), name))
        if (self.hard_minimum_area_km2 is None) != (self.hard_maximum_area_km2 is None):
            raise ValueError('Hard contact area anchors must be supplied together.')
        if self.hard_minimum_area_km2 is not None:
            if self.hard_stage_area_km2 is None:
                raise ValueError('Hard contact anchors require their original full-stage area.')
            for name in ('hard_minimum_area_km2','hard_maximum_area_km2'):
                object.__setattr__(self,name,_array(getattr(self,name),name))
            object.__setattr__(self,'hard_stage_area_km2',_array(self.hard_stage_area_km2,'hard_stage_area_km2'))
            margin=(np.zeros_like(self.current_area_km2) if self.hard_arithmetic_area_km2 is None
                else self.hard_arithmetic_area_km2)
            object.__setattr__(self,'hard_arithmetic_area_km2',_array(margin,'hard_arithmetic_area_km2'))
        elif self.hard_arithmetic_area_km2 is not None or self.hard_stage_area_km2 is not None:
            raise ValueError('Hard contact arithmetic margin requires fixed stage anchors.')
        self.validate(len(self.reference_area_km2))

    def validate(self, count):
        if (type(count) is not int or count < 0 or not isinstance(self.reference_identity, str) or not self.reference_identity
            or self.outer_tolerance != OUTER_RELATIVE_TOLERANCE or self.inverse_tolerance != INVERSE_RELATIVE_TOLERANCE):
            raise ValueError('Invalid numerical accuracy identity or declared constants.')
        arrays = (self.reference_area_km2,self.budget_km2,self.spent_km2,self.current_area_km2,self.minimum_area_km2,self.maximum_area_km2)
        if (any(value.shape != (count,) for value in arrays)
            or np.any(self.reference_area_km2 <= 0) or np.any(self.current_area_km2 <= 0)
            or np.any(self.minimum_area_km2 <= 0) or np.any(self.maximum_area_km2 < self.minimum_area_km2)
            or np.any(self.budget_km2 <= 0) or np.any(self.budget_km2 > allowance(self.reference_area_km2))
            or np.any(self.spent_km2 < 0) or np.any(self.spent_km2 > self.budget_km2)):
            raise ValueError('Numerical area credit is invalid or has been renewed beyond its material allocation.')
        if np.any(self.baseline_miss_km2 > self.budget_km2):
            raise ValueError('Starting material geometry exceeds its persistent dimensional area budget.')
        if self.has_hard_bounds:
            if (any(getattr(self,name).shape!=(count,) for name in
                    ('hard_minimum_area_km2','hard_maximum_area_km2','hard_arithmetic_area_km2','hard_stage_area_km2'))
                or np.any(self.hard_minimum_area_km2<=0)
                or np.any(self.hard_maximum_area_km2<self.hard_minimum_area_km2)
                or np.any(self.hard_arithmetic_area_km2<0)
                or np.any(self.hard_stage_area_km2<=0)
                or np.any(self.hard_minimum_area_km2>self.hard_stage_area_km2)
                or np.any(self.hard_maximum_area_km2<self.hard_stage_area_km2)
                or np.any(self.hard_arithmetic_area_km2>1e-12*self.hard_stage_area_km2)
                or np.any(self.hard_minimum_area_km2-self.hard_arithmetic_area_km2<=0)):
                raise ValueError('Invalid fixed contact strain interval or arithmetic margin.')
            if not np.all(self.hard_feasible_mask(self.current_area_km2)):
                raise ValueError('Contact geometry exceeds its fixed full-stage strain interval.')
        return self

    @property
    def has_hard_bounds(self):
        return self.hard_minimum_area_km2 is not None

    def _hard_keywords(self, index=None):
        if not self.has_hard_bounds:return {}
        return {name:getattr(self,name) if index is None else getattr(self,name)[index]
            for name in ('hard_minimum_area_km2','hard_maximum_area_km2','hard_arithmetic_area_km2','hard_stage_area_km2')}

    def with_hard_bounds(self, minimum, maximum, *, arithmetic_area=None):
        if self.has_hard_bounds:
            raise ValueError('Contact full-stage strain anchors cannot be reset in a substep.')
        return NumericalPolicy(self.reference_area_km2,self.budget_km2,self.spent_km2,
            self.current_area_km2,self.minimum_area_km2,self.maximum_area_km2,self.reference_identity,
            hard_minimum_area_km2=minimum,hard_maximum_area_km2=maximum,
            hard_arithmetic_area_km2=arithmetic_area,hard_stage_area_km2=self.current_area_km2)

    def hard_feasible_mask(self, area):
        area=np.asarray(area,float)
        if area.shape!=self.budget_km2.shape:raise ValueError('Hard contact bounds require aligned material identities.')
        if not self.has_hard_bounds:return np.ones(area.shape,dtype=bool)
        return (np.isfinite(area)&(area>=self.hard_minimum_area_km2-self.hard_arithmetic_area_km2)
            &(area<=self.hard_maximum_area_km2+self.hard_arithmetic_area_km2))

    def effective_bounds(self, minimum=None, maximum=None):
        minimum=self.minimum_area_km2 if minimum is None else minimum
        maximum=self.maximum_area_km2 if maximum is None else maximum
        _,minimum,maximum=self._values(self.current_area_km2,minimum,maximum)
        lower=minimum-self.allowance_km2;upper=maximum+self.allowance_km2
        if self.has_hard_bounds:
            lower=np.maximum(lower,self.hard_minimum_area_km2-self.hard_arithmetic_area_km2)
            upper=np.minimum(upper,self.hard_maximum_area_km2+self.hard_arithmetic_area_km2)
        if np.any(upper<lower):raise ValueError('Contact nominal admission and fixed strain interval have an empty intersection.')
        return lower,upper

    @property
    def baseline_miss_km2(self):
        return nominal_miss(self.current_area_km2,self.minimum_area_km2,self.maximum_area_km2)

    @property
    def allowance_km2(self):
        return np.minimum(self.budget_km2, self.baseline_miss_km2+self.budget_km2-self.spent_km2)

    def allowance_for(self, indices):
        return self.allowance_km2[np.asarray(indices)]

    def subset(self, indices):
        index = np.asarray(indices)
        if index.ndim != 1 or index.dtype.kind not in 'iu' or np.any(index<0) or np.any(index>=len(self.budget_km2)):
            raise ValueError('Area policy subset requires exact aligned material indices.')
        return NumericalPolicy(*(getattr(self,name)[index] for name in ('reference_area_km2','budget_km2','spent_km2',
            'current_area_km2','minimum_area_km2','maximum_area_km2')), reference_identity=self.reference_identity,
            **self._hard_keywords(index))

    def _values(self, area, minimum, maximum):
        values = tuple(np.asarray(value, float) for value in (area,minimum,maximum))
        if any(value.shape != self.budget_km2.shape for value in values):
            raise ValueError('Area admission must align with fixed material reference identities.')
        if self.has_hard_bounds and (not np.array_equal(values[1],self.minimum_area_km2)
                or not np.array_equal(values[2],self.maximum_area_km2)):
            raise ValueError('Contact admission must retain original nominal bounds; effective bounds cannot receive credit twice.')
        return values

    def feasible_mask(self, area, minimum, maximum):
        area,minimum,maximum=self._values(area,minimum,maximum)
        return (np.isfinite(area)&np.isfinite(minimum)&np.isfinite(maximum)&(area>0)&(minimum>0)&(maximum>=minimum)
            & (nominal_miss(area,minimum,maximum)<=self.allowance_km2)
            & self.hard_feasible_mask(area))

    def feasible(self, area, minimum, maximum):
        return bool(np.all(self.feasible_mask(area,minimum,maximum)))

    def violation_in_budgets(self, area, minimum, maximum):
        area,minimum,maximum=self._values(area,minimum,maximum)
        if not np.isfinite(area).all() or np.any(area<=0):return float('inf')
        miss=nominal_miss(area,minimum,maximum)
        return float(np.divide(miss,self.allowance_km2,out=np.where(miss==0,0.,np.inf),
            where=self.allowance_km2>0).max(initial=0.))

    def advance(self, area, minimum=None, maximum=None):
        minimum=self.minimum_area_km2 if minimum is None else minimum
        maximum=self.maximum_area_km2 if maximum is None else maximum
        area,minimum,maximum=self._values(area,minimum,maximum)
        if not self.feasible(area,minimum,maximum):
            raise ValueError('Accepted geometry exhausted its persistent dimensional area budget.')
        spent=self.spent_km2+np.maximum(nominal_miss(area,minimum,maximum)-self.baseline_miss_km2,0.)
        return NumericalPolicy(self.reference_area_km2,self.budget_km2,spent,area,minimum,maximum,self.reference_identity,
            **self._hard_keywords())

    def evidence(self, area, minimum, maximum, *, include_arrays=True):
        area,minimum,maximum=self._values(area,minimum,maximum)
        miss=nominal_miss(area,minimum,maximum)
        increment=np.maximum(miss-self.baseline_miss_km2,0.)
        signed=np.where(area<minimum,area-minimum,np.where(area>maximum,area-maximum,0.))
        result=dict(kind=KIND,version=VERSION,outer_relative_tolerance=self.outer_tolerance,
            inverse_relative_tolerance=self.inverse_tolerance,absolute_area_cap_km2=ABSOLUTE_AREA_CAP_KM2,
            reference_face_area_km2=REFERENCE_FACE_AREA_KM2,reference_identity=self.reference_identity,
            reference_area_sha256=hashlib.sha256(self.reference_area_km2.tobytes()).hexdigest(),
            fixed_reference_area_km2=self.reference_area_km2.tolist(),allocated_budget_km2=self.budget_km2.tolist(),
            previous_spent_km2=self.spent_km2.tolist(),baseline_nominal_miss_km2=self.baseline_miss_km2.tolist(),
            permitted_nominal_miss_km2=self.allowance_km2.tolist(),nominal_miss_km2=miss.tolist(),
            signed_nominal_miss_km2=signed.tolist(),spent_increment_km2=increment.tolist(),
            cumulative_spent_km2=(self.spent_km2+increment).tolist(),
            maximum_nominal_miss_km2=float(miss.max(initial=0.)),total_nominal_miss_km2=float(miss.sum()),
            signed_total_nominal_miss_km2=float(signed.sum()),total_cumulative_spent_km2=float((self.spent_km2+increment).sum()),
            original_physical_bounds_unchanged=True,numerical_policy_changed=True,
            cumulative_budget_renewed=False,admissible=self.feasible(area,minimum,maximum))
        result['maximum_budget_fraction']=self.violation_in_budgets(area,minimum,maximum)
        result['allocated_budget_total_km2']=float(self.budget_km2.sum())
        result['input_face_count']=len(area)
        hard_names=()
        if self.has_hard_bounds:
            effective_minimum,effective_maximum=self.effective_bounds(minimum,maximum)
            result.update(contact_strain_anchor_scope='fixed original full contact stage; never reanchored by serial substeps',
                contact_arithmetic_margin_scope='fixed original legacy ratio arithmetic allowance: 1e-12 times stage area',
                contact_hard_strain_admissible=bool(np.all(self.hard_feasible_mask(area))),
                contact_maximum_hard_lower_excess_km2=float(np.maximum(
                    self.hard_minimum_area_km2-self.hard_arithmetic_area_km2-area,0.).max(initial=0.)),
                contact_maximum_hard_upper_excess_km2=float(np.maximum(
                    area-self.hard_maximum_area_km2-self.hard_arithmetic_area_km2,0.).max(initial=0.)),
                contact_effective_bounds_scope='original nominal ledger admission intersected once with independent hard strain interval',
                contact_actual_area_km2=area.tolist(),
                hard_stage_area_km2=self.hard_stage_area_km2.tolist(),
                hard_minimum_area_km2=self.hard_minimum_area_km2.tolist(),
                hard_maximum_area_km2=self.hard_maximum_area_km2.tolist(),
                hard_arithmetic_area_km2=self.hard_arithmetic_area_km2.tolist(),
                effective_minimum_area_km2=effective_minimum.tolist(),effective_maximum_area_km2=effective_maximum.tolist())
            hard_names=('contact_actual_area_km2','hard_stage_area_km2','hard_minimum_area_km2','hard_maximum_area_km2','hard_arithmetic_area_km2',
                'effective_minimum_area_km2','effective_maximum_area_km2')
        if not include_arrays:
            names=('fixed_reference_area_km2','allocated_budget_km2','previous_spent_km2',
                'baseline_nominal_miss_km2','permitted_nominal_miss_km2','nominal_miss_km2',
                'signed_nominal_miss_km2','spent_increment_km2','cumulative_spent_km2')
            for name in names+hard_names:
                value=np.asarray(result.pop(name),dtype=np.float64)
                result[name+'_sha256']=hashlib.sha256(value.tobytes()).hexdigest()
        return result


def initialize_columns(state, area):
    if present(state):raise ValueError('Numerical accuracy metadata must be initialized exactly once.')
    area=np.asarray(area,float)
    if area.shape!=np.asarray(state['thickness_km']).shape or not np.isfinite(area).all() or np.any(area<=0):
        raise ValueError('Numerical accuracy requires actual positive material areas.')
    state.update({REFERENCE:area.copy(),BUDGET:allowance(area).copy(),SPENT:np.zeros_like(area),AREA:area.copy()})


def validate_columns(state, restored, minimum_thickness, maximum_thickness):
    if not present(state):return False
    shape=np.asarray(state['thickness_km']).shape
    if any(np.asarray(state[name]).shape!=shape or not np.isfinite(state[name]).all() for name in FIELDS):
        raise ValueError('Invalid dimensional column accuracy metadata.')
    ref,budget,spent,area=(np.asarray(state[name],float) for name in FIELDS)
    if (np.any(ref<=0) or np.any(area<=0) or np.any(budget<=0) or np.any(budget>allowance(ref))
        or np.any(spent<0) or np.any(spent>budget) or np.any(np.asarray(state['thickness_km'])<=0)
        or not np.isfinite(restored).all() or np.any(np.asarray(restored)<=0)):
        raise ValueError('Column numerical area budget is invalid or renewed.')
    minimum=area*np.asarray(state['thickness_km'])/maximum_thickness
    maximum=area*np.asarray(restored)/minimum_thickness
    if np.any(nominal_miss(area,minimum,maximum)>budget):
        raise ValueError('Exact column volume exceeds its dimensional nominal-area budget.')
    return True


def policy(s, minimum, maximum, *, current_area=None):
    if not version(s):return None
    state=s.structure
    if not present(state):raise ValueError('Explicit numerical migration is missing its material ledger.')
    return NumericalPolicy(state[REFERENCE],state[BUDGET],state[SPENT],
        state[AREA] if current_area is None else current_area,minimum,maximum,
        'persistent material lineage from '+s.numerical_accuracy_migration['parent_checkpoint_sha256'])


def migrate(s, *, boundary_myr, parent_checkpoint_sha256, authorization_sha256):
    """Constructorless deterministic additions only; caller publishes the transition."""
    if version(s) or any(hasattr(s,name) for name in ('numerical_accuracy_migration',)):
        raise ValueError('Numerical accuracy migration is a one-time declared transition.')
    for value in (parent_checkpoint_sha256,authorization_sha256):
        if not isinstance(value,str) or len(value)!=64 or any(ch not in '0123456789abcdef' for ch in value):
            raise ValueError('Numerical migration requires exact external SHA256 pins.')
    if not np.isfinite(boundary_myr) or float(getattr(s,'t',-1))!=float(boundary_myr):
        raise ValueError('Numerical migration must bind the exact accepted boundary.')
    ids=np.asarray(s.parcel_patch);trace_ids=np.asarray(s.trace_patch)
    order=np.argsort(ids);found=np.searchsorted(ids[order],trace_ids)
    if np.any(found>=len(ids)) or np.any(ids[order[found]]!=trace_ids):
        raise ValueError('Numerical migration requires retained material marker identities.')
    if present(s.structure) or present(s.trace_structure):
        raise ValueError('Preexisting numerical area fields require their original declared policy.')
    initialize_columns(s.structure,s.material_surface['area_km2'])
    initialize_columns(s.trace_structure,np.asarray(s.material_surface['area_km2'])[order[found]])
    s.numerical_accuracy_version=VERSION
    s.numerical_accuracy_migration=dict(kind=KIND,version=VERSION,boundary_myr=float(boundary_myr),
        parent_checkpoint_sha256=parent_checkpoint_sha256,authorization_sha256=authorization_sha256,
        fields=list(FIELDS),reference='actual accepted boundary face area; extensive lineage allocation',
        numerical_budget_borrowed=False,preexisting_typed_arrays_changed=False)
    return dict(s.numerical_accuracy_migration)


def endpoint_columns(state, final_area, spent):
    """Metadata for the same accepted geometry as exact V/A; no physical writes."""
    result={name:np.asarray(state[name],float).copy() for name in FIELDS}
    result[AREA]=np.asarray(final_area,float).copy()
    result[SPENT]=np.asarray(spent,float).copy()
    return result


def accepted_contact_policy(before, result, final_area):
    if before is None:return None
    spent=result.get('diagnostics',{}).get('numerical_accuracy_final_spent_km2')
    if spent is None:raise ValueError('Accepted contact response omitted cumulative material area spending.')
    spent=np.asarray(spent,float)
    if np.any(spent<before.spent_km2):raise ValueError('Accepted contact response refunded cumulative material area spending.')
    if not before.feasible(final_area,before.minimum_area_km2,before.maximum_area_km2):
        raise ValueError('Accepted contact response exceeded its remaining material area credit.')
    minimum_spent=before.spent_km2+np.maximum(nominal_miss(final_area,before.minimum_area_km2,
        before.maximum_area_km2)-before.baseline_miss_km2,0.)
    if np.any(spent<minimum_spent):raise ValueError('Contact response failed to charge its accepted outward nominal-area miss.')
    return NumericalPolicy(before.reference_area_km2,before.budget_km2,spent,final_area,
        before.minimum_area_km2,before.maximum_area_km2,before.reference_identity)


def remap_columns(old, new, mapping):
    """Transfer identity credit extensively; an incompatible merge is refused."""
    import adaptive_material
    if not present(old):raise ValueError('Numerical adaptation requires the original identity ledger.')
    for name in EXTENSIVE_FIELDS:new[name]=adaptive_material.transfer_extensive(old[name],mapping)
    for name in EXTENSIVE_FIELDS:
        if not np.isclose(np.sum(new[name]),np.sum(old[name]),rtol=2e-13,atol=1e-12):
            raise ValueError('Adaptation changed persistent numerical area inventory: '+name)
    return bool(np.all(new[BUDGET]<=allowance(new[REFERENCE])) and np.all(new[SPENT]<=new[BUDGET]))


def column_endpoint(s, *, trace=False):
    if not version(s):return None
    row=s.material_deformation.get('numerical_accuracy_endpoint')
    if not isinstance(row,dict):raise ValueError('Active geometric columns lack their accepted numerical area ledger.')
    if trace:
        ids=np.asarray(s.parcel_patch);order=np.argsort(ids)
        found=np.searchsorted(ids[order],s.trace_patch)
        if np.any(found>=len(ids)) or np.any(ids[order[found]]!=s.trace_patch):
            raise ValueError('Accuracy columns require exact marker ancestry.')
        index=order[found]
        result={name:np.asarray(s.structure[name])[index].copy() for name in (REFERENCE,BUDGET)}
        result[AREA]=np.asarray(row[AREA])[index].copy()
        result[SPENT]=np.asarray(row[SPENT])[index].copy()
        return result
    return endpoint_columns(s.structure,row[AREA],row[SPENT])


def validate_simulation(s):
    """Read-only current-state checks for migration and accepted-step witnesses."""
    if not version(s):return dict(version=0,state='legacy')
    import crustal_structure as columns
    state=columns._state(s.structure);traces=columns._state(s.trace_structure)
    area=np.asarray(s.material_surface['area_km2'],float)
    ids=np.asarray(s.parcel_patch);order=np.argsort(ids);found=np.searchsorted(ids[order],s.trace_patch)
    if np.any(found>=len(ids)) or np.any(ids[order[found]]!=s.trace_patch):
        raise ValueError('Numerical area ledger lost a material marker identity.')
    index=order[found]
    if not np.allclose(state[AREA],area,rtol=2e-12,atol=0.) or not np.allclose(traces[AREA],area[index],rtol=2e-12,atol=0.):
        raise ValueError('Column numerical area ledger differs from its accepted material geometry.')
    for name in (REFERENCE,BUDGET,SPENT):
        if not np.array_equal(traces[name],state[name][index]):
            raise ValueError('Marker numerical area credit differs from its containing material identity.')
    physical=area*state['thickness_km'];represented=np.asarray(s.mass)*state['area_factor']*state['thickness_km']
    geometric_volume=float(area@state['thickness_km'])
    reference_volume=float(np.asarray(s.mass)@(state['area_factor']*state['thickness_km']))
    volume_relative=(geometric_volume-reference_volume)/max(abs(geometric_volume),abs(reference_volume),1.)
    # Historical accepted skinny faces retain tiny differences between their
    # measured area and mass*area_factor. Do not normalize that saved state or
    # replace the existing aggregate conservation predicate with a newly
    # stronger per-face equivalence claim. The active writer separately checks
    # each actual V/A transaction in both geometric and reference measures.
    if not np.isfinite(volume_relative) or abs(volume_relative)>=1e-12:
        raise ValueError('Numerical accuracy column metadata conceals an aggregate physical/reference volume mismatch.')
    minimum=physical/columns.MAX_THICKNESS_KM
    maximum=area*columns.restored_thickness(state)/columns.MIN_THICKNESS_KM
    miss=nominal_miss(area,minimum,maximum)
    return dict(kind=KIND,version=VERSION,material_faces=len(ids),material_markers=len(index),
        fixed_reference_area_km2=float(state[REFERENCE].sum()),allocated_budget_km2=float(state[BUDGET].sum()),
        cumulative_spent_km2=float(state[SPENT].sum()),maximum_nominal_miss_km2=float(miss.max(initial=0.)),
        total_nominal_miss_km2=float(miss.sum()),
        maximum_physical_reference_volume_difference_km3=float(np.abs(physical-represented).max(initial=0.)),
        maximum_physical_reference_volume_relative_difference=float(np.divide(np.abs(physical-represented),
            np.maximum(np.abs(physical),np.abs(represented)),out=np.zeros_like(physical),
            where=np.maximum(np.abs(physical),np.abs(represented))>0).max(initial=0.)),
        physical_reference_volume_relative_residual=volume_relative,
        physical_reference_volume_tolerance=1e-12,physical_reference_volume_scope='original aggregate conservation predicate',
        physical_volume_km3=geometric_volume,reference_volume_km3=reference_volume,
        nominal_thickness_limits_km=[columns.MIN_THICKNESS_KM,columns.MAX_THICKNESS_KM],
        volume_clipped=False,cumulative_budget_renewed=False)


def snapshot(s):
    if not version(s):return {}
    state=s.structure
    import crustal_structure as columns
    return dict(numerical_accuracy_version=VERSION,numerical_accuracy_migration=dict(s.numerical_accuracy_migration),
        material_area_accuracy_reference_km2=state[REFERENCE].copy(),
        material_area_accuracy_budget_km2=state[BUDGET].copy(),
        material_area_accuracy_spent_km2=state[SPENT].copy(),
        material_area_accuracy_current_area_km2=state[AREA].copy(),
        material_area_accuracy_restored_thickness_km=columns.restored_thickness(state).copy(),
        numerical_accuracy_diagnostics=getattr(s,'numerical_accuracy_diagnostics',None))


def validate_evidence(row):
    if (not isinstance(row,dict) or row.get('kind')!=KIND or type(row.get('version')) is not int or row['version']!=VERSION
        or row.get('outer_relative_tolerance')!=OUTER_RELATIVE_TOLERANCE
        or row.get('inverse_relative_tolerance')!=INVERSE_RELATIVE_TOLERANCE
        or row.get('absolute_area_cap_km2')!=ABSOLUTE_AREA_CAP_KM2
        or row.get('reference_face_area_km2')!=REFERENCE_FACE_AREA_KM2
        or row.get('original_physical_bounds_unchanged') is not True
        or row.get('cumulative_budget_renewed') is not False or row.get('admissible') is not True):
        raise ValueError('Invalid declared dimensional numerical accuracy evidence.')
    for name in ('maximum_budget_fraction','total_cumulative_spent_km2','allocated_budget_total_km2'):
        if type(row.get(name)) not in (int,float) or not np.isfinite(row[name]) or row[name]<0:
            raise ValueError('Invalid dimensional accuracy scalar: '+name)
    if row['maximum_budget_fraction']>1 or row['total_cumulative_spent_km2']>row['allocated_budget_total_km2']:
        raise ValueError('Saved numerical area budget was exceeded.')
    return row


def validate_frame(frame):
    marker=frame.get('numerical_accuracy_version',0)
    if type(marker) is not int or marker not in (0,VERSION):raise ValueError('Invalid saved numerical accuracy policy marker.')
    found=set(ARRAY_FIELDS).intersection(frame)
    if marker==0:
        if found or 'numerical_accuracy_migration' in frame:raise ValueError('Undeclared saved dimensional numerical metadata.')
        return
    if found!=set(ARRAY_FIELDS):raise ValueError('Saved numerical accuracy policy omits its complete material ledger.')
    count=len(frame['material_faces'])
    values={name:np.asarray(frame[name],float) for name in ARRAY_FIELDS}
    if any(value.shape!=(count,) or not np.isfinite(value).all() for value in values.values()):
        raise ValueError('Saved dimensional accuracy fields do not align with material identities.')
    state={field:values[name] for name,field in FRAME_FIELDS.items()}
    state['thickness_km']=np.asarray(frame['material_crustal_thickness_km'],float)
    import dense_crust
    expected_restored=state['thickness_km']
    if frame.get('retained_dense_crust_version',0)==1:
        phase={name:np.asarray(frame[field],float) for field,name in dense_crust.FRAME_FIELDS.items()}
        expected_restored=dense_crust.restored_thickness(phase)
    if not np.array_equal(values[RESTORED_FRAME_FIELD],expected_restored):
        raise ValueError('Saved dimensional accuracy uses a different physical restored column.')
    validate_columns(state,values[RESTORED_FRAME_FIELD],8.,75.)
    if not np.allclose(state[AREA],np.asarray(frame['material_actual_area_km2']),rtol=2e-12,atol=0.):
        raise ValueError('Saved numerical material area differs from its accepted geometry.')
    migration=frame.get('numerical_accuracy_migration')
    if (not isinstance(migration,dict) or migration.get('kind')!=KIND or migration.get('version')!=VERSION
        or migration.get('fields')!=list(FIELDS) or migration.get('preexisting_typed_arrays_changed') is not False):
        raise ValueError('Saved numerical policy lacks its declared boundary migration.')
