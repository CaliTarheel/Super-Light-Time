"""Reduced, dissipative connection between an incoming plate and a sinking slab.

This module is a constitutive component, not a subduction/polarity policy. It
does not feed, delete or move rock, guess an inherited strength, or start a clock
when a continental label appears. Callers must supply the slab's actual force,
mantle drag and neck properties, and integrate it with their mass/force ledgers.

For one metre of trench, u is down-dip slab speed and v is plate inlet speed.
The Rayleigh functional is .5*Cs*u^2 + .5*Cn*(u-v)^2 - W*u. Eliminating u gives
plate force alpha*(W-Cs*v), alpha=Cn/(Cs+Cn). The driving force and resisting
coefficient MUST both receive alpha. Damping a solved velocity is not equivalent.

Cn=4*eta*h/L*(1-d)^2 is a reduced plane-strain viscous neck. d is irreversible
tensile opening divided by a supplied failure opening. This damage law is an
explicit phenomenological rupture closure, not a resolved neck instability or
a calibrated prediction of slab breakoff. It must not be enabled without the
associated geometry, mass transfer and inherited-polarity policy.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import math
import numpy as np

SECONDS_PER_MYR=365.25*86400.*1e6


@dataclass(frozen=True)
class Neck:
    viscosity_pa_s: float
    thickness_m: float
    length_m: float
    failure_opening_m: float
    damage: float=0.

    def __post_init__(self):
        for name in ('viscosity_pa_s','thickness_m','length_m','failure_opening_m'):
            value=getattr(self,name)
            if isinstance(value,(bool,np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value<=0:
                raise ValueError('Slab neck requires an explicit positive finite '+name)
        if isinstance(self.damage,(bool,np.bool_)) or not np.isscalar(self.damage) or not np.isfinite(self.damage) or not 0<=self.damage<=1:
            raise ValueError('Slab neck damage must be a finite fraction.')
        if not math.isfinite(self.intact_coefficient_pa_s) or self.intact_coefficient_pa_s<=0:
            raise ValueError('Slab neck resistance overflows its constitutive range.')

    @property
    def intact_coefficient_pa_s(self):
        return 4.*self.viscosity_pa_s*(self.thickness_m/self.length_m)

    @property
    def coefficient_pa_s(self):
        return self.intact_coefficient_pa_s*(1.-self.damage)**2


def _controls(weight_n_per_m,mantle_drag_pa_s,inlet_speed_m_s=None):
    values=[weight_n_per_m,mantle_drag_pa_s]
    if inlet_speed_m_s is not None:values.append(inlet_speed_m_s)
    if any(isinstance(v,(bool,np.bool_)) or not np.isscalar(v) or not np.isfinite(v) for v in values):
        raise ValueError('Slab force, drag and inlet speed must be finite scalars.')
    if mantle_drag_pa_s<=0:
        raise ValueError('The sinking slab requires positive mantle drag.')


def condensed(neck,weight_n_per_m,mantle_drag_pa_s):
    """Affine plate traction after eliminating the independent slab velocity.

    A signed W permits a positively buoyant down-dip inventory. The caller must
    derive it from conserved material and avoid counting the same buoyancy in
    both this path and an existing column gravitational functional.
    """
    _controls(weight_n_per_m,mantle_drag_pa_s)
    c=neck.coefficient_pa_s
    scale=max(mantle_drag_pa_s,c)
    alpha=(c/scale)/(mantle_drag_pa_s/scale+c/scale)
    return dict(drive_n_per_m=alpha*weight_n_per_m,
                drag_pa_s=alpha*mantle_drag_pa_s,transmission_fraction=alpha)


def solve(neck,weight_n_per_m,mantle_drag_pa_s,inlet_speed_m_s):
    """Exact fixed-state force balance and power audit, all per metre of trench."""
    _controls(weight_n_per_m,mantle_drag_pa_s,inlet_speed_m_s)
    terms=condensed(neck,weight_n_per_m,mantle_drag_pa_s)
    alpha=terms['transmission_fraction']
    c=neck.coefficient_pa_s
    scale=max(mantle_drag_pa_s,c)
    denominator=mantle_drag_pa_s/scale+c/scale
    u=(weight_n_per_m/scale)/denominator+alpha*inlet_speed_m_s
    extension=(weight_n_per_m/scale-(mantle_drag_pa_s/scale)*inlet_speed_m_s)/denominator
    traction=terms['drive_n_per_m']-terms['drag_pa_s']*inlet_speed_m_s
    mantle_power=mantle_drag_pa_s*u*u
    neck_power=traction*extension
    weight_power=weight_n_per_m*u
    plate_power=traction*inlet_speed_m_s
    residual=weight_power-plate_power-mantle_power-neck_power
    result=dict(**terms,slab_speed_m_s=u,neck_extension_speed_m_s=extension,
        plate_traction_n_per_m=traction,mantle_dissipation_w_per_m=mantle_power,
        neck_dissipation_w_per_m=neck_power,weight_power_w_per_m=weight_power,
        plate_power_w_per_m=plate_power,power_residual_w_per_m=residual,
        force_residual_n_per_m=weight_n_per_m-mantle_drag_pa_s*u-traction)
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError('Slab tether solution exceeds its finite constitutive range.')
    return result


def parallel(necks,weights_n_per_m,mantle_drags_pa_s,widths_m):
    """Sum independent neck channels without averaging nonlinear damage.

    The returned drive is N and drag is N s/m, for channels with the SAME inlet
    velocity/direction. Different local directions must be assembled separately
    by the plate solver. Width subdivision at unchanged material properties is
    invariant. Retained mass and channel provenance belong to the caller.
    """
    necks=tuple(necks)
    arrays=[np.asarray(value,float) for value in (weights_n_per_m,mantle_drags_pa_s,widths_m)]
    if any(value.shape!=(len(necks),) or not np.isfinite(value).all() for value in arrays):
        raise ValueError('Parallel slab neck channels must have aligned finite properties.')
    weights,drags,widths=arrays
    if np.any(widths<0):raise ValueError('Slab neck channel widths cannot be negative.')
    drive=[];resistance=[]
    for material,weight,drag,width in zip(necks,weights,drags,widths):
        terms=condensed(material,float(weight),float(drag))
        drive.append(width*terms['drive_n_per_m'])
        resistance.append(width*terms['drag_pa_s'])
    result=dict(drive_n=math.fsum(drive),drag_n_s_per_m=math.fsum(resistance))
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError('Integrated slab neck channels exceed their finite constitutive range.')
    return result


def advance(neck,weight_n_per_m,mantle_drag_pa_s,inlet_speed_m_s,dt_myr):
    """Exact damage continuation for frozen external force/drag/inlet speed.

    The failure time is a consequence of accumulated positive relative opening,
    not a continental-arrival delay. Compression never heals damage. Returns a
    new immutable Neck and an event/power report, without mutating the input.
    The caller must split/re-solve the coupled plate step at a rupture event;
    this routine does not silently keep pre-breakoff plate forces for the rest
    of the interval or reattach a detached slab.
    """
    _controls(weight_n_per_m,mantle_drag_pa_s,inlet_speed_m_s)
    if isinstance(dt_myr,(bool,np.bool_)) or not np.isfinite(dt_myr) or dt_myr<0:
        raise ValueError('Slab damage requires a finite nonnegative elapsed interval.')
    initial=solve(neck,weight_n_per_m,mantle_drag_pa_s,inlet_speed_m_s)
    drive=weight_n_per_m-mantle_drag_pa_s*inlet_speed_m_s
    if neck.damage==1. or drive<=0. or dt_myr==0.:
        return neck,dict(ruptured=False,rupture_time_myr=None,advanced_dt_myr=float(dt_myr),
            remaining_dt_myr=0.,damage_increment=0.,opening_m=0.,initial=initial,final=initial)
    intact=neck.intact_coefficient_pa_s
    rest=1.-neck.damage
    # Integrate dt/dd = delta_crit*(Cs+C0*(1-d)^2)/(W-Cs*v).
    time_scale=neck.failure_opening_m/drive/SECONDS_PER_MYR
    def elapsed(increment):
        # Difference of cubes factored to avoid cancellation at small steps.
        return time_scale*increment*(mantle_drag_pa_s+
            intact*(rest*rest-rest*increment+increment*increment/3.))
    failure_time=elapsed(rest)
    if not math.isfinite(failure_time) or failure_time<=0.:
        raise ValueError('Slab damage integration exceeds its constitutive range.')
    rupture=dt_myr>=failure_time
    if rupture:
        increment=rest
    else:
        # Prefix-average resistance is at least its average over the full
        # remaining opening, so dt/failure_time bounds the fractional damage.
        # This also retains relative resolution for extremely small steps.
        low,high=0.,rest*(dt_myr/failure_time)
        # Monotone cubic; bisection is bounded and cannot overshoot rupture.
        for _ in range(80):
            middle=.5*(low+high)
            if elapsed(middle)<dt_myr:low=middle
            else:high=middle
        increment=.5*(low+high)
    damaged=replace(neck,damage=1. if rupture else neck.damage+increment)
    advanced=min(dt_myr,failure_time)
    return damaged,dict(ruptured=rupture,rupture_time_myr=failure_time if rupture else None,
        advanced_dt_myr=float(advanced),remaining_dt_myr=float(dt_myr-advanced),
        damage_increment=float(increment),opening_m=float(increment*neck.failure_opening_m),
        initial=initial,final=solve(damaged,weight_n_per_m,mantle_drag_pa_s,inlet_speed_m_s))
