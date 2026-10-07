"""Finite retained slab proxy: admitted area and conserved excess mass.

Actual admitted ocean area enters an exponentially retiring inventory. The same
feed, weighted by the thermal density contrast the engine already uses for
bathymetry, carries an excess mass per metre of trench, S_k in kg/m. That is the
state a force balance can read: g*S_k is a line force in N/m, so slab pull stops
being a dimensionless label multiplier and becomes the weight of what has
actually gone down. Version 2 carries an extensive excess-mass ledger and
derives line density from it; version 1 retains the historical density law.
Kinematic initiation maturity still admits consumption, so
the basal driver can start a trench without inventing an initial slab or
creating a feeding deadlock.

Retention here follows RETENTION_MYR. The existing trench policy still has
kinematic shutdown timers; it does not yet implement mechanical breakoff.
Optional neck histories preserve a future mechanical policy's inventories,
but do not enable that policy or replace the current force assembly.
The inventory is not a resolved slab-depth profile. Force readers use a uniform
down-dip column proxy and only couple its upper-mantle portion to the plate.
Deeper mass stays in the conserved inventory until exponential retirement;
reading the force never deletes mass or invents a mantle-return transaction.
"""
import math
from copy import deepcopy
import numpy as np

VERSION = 2
RETENTION_MYR = 50.
DEVELOPED_LENGTH_KM = 100.
# Upper-mantle-only pull is a reduced coupling assumption (Conrad &
# Lithgow-Bertelloni 2002), not a claim that deeper slab mass disappears.
MAX_SLAB_DEPTH_KM = 660.
# The one dip in this engine. Nothing solves or stores a slab dip anywhere, and
# global compilations put the shallow (0-125 km) mean at 35-55 degrees with a
# very wide spread (Lallemand et al. 2005; Syracuse & Abers 2006). 50 degrees is
# the middle of that band and is the number every consumer must quote.
SUBDUCTION_DIP_DEG = 50.


def upper_mantle_length_km(dip_deg=None):
    """Convert vertical depth to down-dip length for the fixed-dip proxy."""
    dip = SUBDUCTION_DIP_DEG if dip_deg is None else float(dip_deg)
    if not math.isfinite(dip) or not 0. < dip <= 90.:
        raise ValueError('Slab dip must be finite and in (0, 90] degrees.')
    return MAX_SLAB_DEPTH_KM/math.sin(math.radians(dip))


MAX_SLAB_LENGTH_KM = upper_mantle_length_km()
MANTLE_DENSITY_KG_M3 = 3300.
WATER_DENSITY_KG_M3 = 1030.
# Ridge-crest datum of the same GDH1-style subsidence law the display uses
# (raster_engine.snapshot). Thermal subsidence measured from the crest is what
# isostasy converts into excess mass.
RIDGE_CREST_DEPTH_M = 2600.
FIELDS = ('slab_fed_area_km2', 'slab_retained_area_km2',
          'slab_retained_buoyancy_area_km2', 'slab_retired_area_km2')
INITIAL_AREA_FIELD = 'slab_initial_area_km2'
LINE_LOAD_FIELD = 'slab_line_load_kg_per_m'
MASS_FIELDS = ('slab_initial_excess_mass_kg', 'slab_fed_excess_mass_kg',
               'slab_retained_excess_mass_kg', 'slab_retired_excess_mass_kg')
RETAINED_MASS_FIELD = 'slab_retained_excess_mass_kg'


def enabled(s):
    return getattr(s, 'slab_memory_version', 0) in (1, VERSION)


def conservative(s):
    return getattr(s, 'slab_memory_version', 0) == VERSION


def thermal_subsidence_m(age_myr):
    """GDH1-style subsidence below the ridge crest, the display's own law.

    raster_engine.snapshot draws bathymetry from exactly this curve: half-space
    cooling under 20 Myr giving way to the exponential plate-cooling tail. Using
    a second, independent cooling law for the slab's mass would let the world's
    oceans and the world's slab pull disagree about the same lithosphere.
    """
    age = np.maximum(np.asarray(age_myr, float), 0.)
    depth = np.where(age < 20., -2600.-365.*np.sqrt(age), -5651.+2473.*np.exp(-.0278*age))
    return -depth-RIDGE_CREST_DEPTH_M


def excess_mass_per_area_kg_m2(age_myr):
    """Isostatic excess mass of cooled lithosphere, kg per square metre.

    A column that has subsided d below the ridge crest has replaced d of mantle
    with d of water, so its excess mass per unit area is (rho_m - rho_w) d. This
    is the same statement as the thermal buoyancy integral, expressed through the
    quantity the model actually carries. It saturates near 6.9e6 kg/m2 at plate
    ages beyond ~150 Myr, which is why an unrealistically old ocean (this world's
    mean age is 185 Myr) does not produce an unbounded slab pull.
    """
    return (MANTLE_DENSITY_KG_M3-WATER_DENSITY_KG_M3)*thermal_subsidence_m(age_myr)


# Line load of a full upper-mantle slab of thermally saturated lithosphere. Used
# only to normalise strength() back into the dimensionless [0, 1.8] view that
# trench_dynamics.retreat_speed and backarc consume; no force reads it.
SATURATED_LINE_LOAD_KG_PER_M = float(excess_mass_per_area_kg_m2(400.)*MAX_SLAB_LENGTH_KM*1e3)
# Compatibility check for historical v1 inventories, which used 660 as a
# length. Correcting the physical reference must not relax this old validator.
LEGACY_MAX_LINE_LOAD_KG_PER_M = float(4.*excess_mass_per_area_kg_m2(400.)*660.*1e3)


def buoyancy_to_age_myr(buoyancy):
    """Invert trench_history.ocean_buoyancy so the feed's provenance is reusable.

    ocean_buoyancy is min(sqrt(age/80), 1.8), so age = 80*b^2 up to its 259 Myr
    saturation. The mass law above is already flat there (3,049 of 3,051 m of
    subsidence), so the inversion's ceiling costs nothing physical: it only means
    a 500 Myr slab is treated as a 259 Myr one, which it is, thermally.
    """
    value = np.clip(np.asarray(buoyancy, float), 0., 1.8)
    return 80.*value*value


def ensure(row, *, conservative=False):
    for key in FIELDS:
        row.setdefault(key, 0.)
    row.setdefault('slab_attachment', 1.)
    row.setdefault(LINE_LOAD_FIELD, 0.)
    if conservative:
        row.setdefault(INITIAL_AREA_FIELD, 0.)
        if any(key in row for key in MASS_FIELDS):
            validate_row(row, require_mass=True)
        else:
            # Only new, empty records may acquire the new ledger implicitly.
            # An old loaded record requires upgrade_mass_inventory().
            if row[LINE_LOAD_FIELD] or any(row[key] for key in FIELDS):
                raise ValueError('Existing slab history requires explicit mass-inventory migration.')
            row.update(dict.fromkeys(MASS_FIELDS, 0.))


def _has_mass(row):
    return any(key in row for key in MASS_FIELDS)


def _length_m(row):
    length = float(row.get('length_km', 0.))*1e3
    if not math.isfinite(length) or length < 0.:
        raise ValueError('Slab trace length must be finite and nonnegative.')
    return length


def line_density(row):
    """Diagnostic load on the stored trace; excess mass is authoritative in v2."""
    if not _has_mass(row):
        return float(row.get(LINE_LOAD_FIELD, 0.))
    length = _length_m(row)
    return row[RETAINED_MASS_FIELD]/length if length > 0. else 0.


def refresh_line_load(row):
    """Refresh a derived diagnostic after rematching, never change inventory."""
    if _has_mass(row):
        row[LINE_LOAD_FIELD] = line_density(row)


def upgrade_mass_inventory(s):
    """Explicit, atomic v1 -> v2 conversion of *currently represented* mass.

    The baseline is saved load times saved trace length. It preserves that
    boundary's represented mass, including any pre-existing bookkeeping error;
    it cannot reconstruct unrecorded thermal/feed history. No world is advanced.
    """
    version = getattr(s, 'slab_memory_version', 0)
    if version not in (1, VERSION):
        raise ValueError('Slab mass migration requires a version-1 or version-2 inventory.')
    staged = deepcopy(s.trench_systems)
    for row in staged:
        validate_row(row, require_mass=version == VERSION)
        if version == VERSION:
            continue
        if _has_mass(row):
            raise ValueError('Version-1 slab state already contains mass fields; refusing mixed migration.')
        length = _length_m(row)
        load = float(row.get(LINE_LOAD_FIELD, 0.))
        if load > 0. and length <= 0.:
            raise ValueError('A loaded legacy slab requires a positive recorded trace length.')
        mass = load*length
        row.update(dict.fromkeys(MASS_FIELDS, 0.))
        row['slab_initial_excess_mass_kg'] = row[RETAINED_MASS_FIELD] = mass
        refresh_line_load(row)
        validate_row(row, require_mass=True)
    report = dict(from_version=version, to_version=VERSION,
        already_converted=version == VERSION,
        retained_excess_mass_kg=math.fsum(r[RETAINED_MASS_FIELD] for r in staged),
        provenance='Boundary baseline = saved line load times saved trace length; historical errors are not reconstructed.')
    if version != VERSION:
        s.trench_systems = staged
        s.slab_memory_version = VERSION
        s.slab_mass_migration = deepcopy(report)
    return report


def validate_row(row, *, require_mass=False):
    values=np.array([row.get(key,0.) for key in FIELDS],float)
    attachment=float(row.get('slab_attachment',1.))
    load=float(row.get(LINE_LOAD_FIELD,0.))
    if not np.isfinite(values).all() or np.any(values<0) or not math.isfinite(attachment) or not 0<=attachment<=1:
        raise ValueError('Retained slab state requires nonnegative finite inventories and attachment in [0,1].')
    mass_state = _has_mass(row)
    if not math.isfinite(load) or load<0:
        raise ValueError('Retained slab line load must be finite and nonnegative.')
    if not mass_state and load>LEGACY_MAX_LINE_LOAD_KG_PER_M:
        raise ValueError('Legacy slab line load exceeds its historical bound.')
    fed,retained,buoyancy,retired=values
    initial_area = float(row.get(INITIAL_AREA_FIELD, 0.))
    if not math.isfinite(initial_area) or initial_area < 0. or (initial_area and not mass_state):
        raise ValueError('Inherited slab area requires a finite nonnegative conservative baseline.')
    tolerance=max(1e-7,(initial_area+fed)*2e-11)
    if abs(initial_area+fed-retained-retired)>tolerance or buoyancy>1.8*retained+tolerance:
        raise ValueError('Retained slab state does not conserve its admitted inventory or buoyancy bound.')
    if require_mass or mass_state:
        if not all(key in row for key in MASS_FIELDS):
            raise ValueError('Conservative slab state requires the complete excess-mass inventory.')
        mass = np.array([row[key] for key in MASS_FIELDS], float)
        if not np.isfinite(mass).all() or np.any(mass < 0.):
            raise ValueError('Slab excess-mass inventories must be finite and nonnegative.')
        initial, mass_fed, mass_retained, mass_retired = mass
        if abs(initial+mass_fed-mass_retained-mass_retired) > max(1., (initial+mass_fed)*2e-11):
            raise ValueError('Slab excess-mass inventory does not close.')
        expected = line_density(row)
        if not math.isfinite(expected) or not math.isclose(load, expected, rel_tol=2e-11, abs_tol=1e-7):
            raise ValueError('Slab line load disagrees with conserved mass and recorded trace length.')
    import slab_tether_history
    slab_tether_history.validate(row)
    import slab_anchors
    slab_anchors.validate(row)


def partition(parent, child, fraction, *, channel_fractions=None, split_points=None):
    """Conserve each extensive inventory when a local history separates.

    With located slab anchors and ``split_points`` (child trace, parent trace
    midpoints), the retained slab divides by where it actually lies; the
    historical area and mass ledgers follow the same area and mass shares.
    """
    if parent is child:
        raise ValueError('A slab cannot be partitioned into itself.')
    ensure(parent)
    validate_row(parent)
    import slab_anchors
    if slab_anchors.carries(parent):
        if split_points is None:
            raise ValueError('Located slab anchors need the trace points of both pieces to divide.')
        area_share, mass_share = slab_anchors.partition(parent, child, *split_points)
        area_keys = FIELDS + ((INITIAL_AREA_FIELD,) if INITIAL_AREA_FIELD in parent else ())
        for keys, share in ((area_keys, area_share), (MASS_FIELDS if _has_mass(parent) else (), mass_share)):
            for key in keys:
                child[key] = parent[key]*share
                parent[key] -= child[key]
        # Exact complements: the anchors are the retained inventory.
        for key, total in (('slab_retained_area_km2', 'area_km2'), (RETAINED_MASS_FIELD, 'excess_mass_kg')):
            for record in (parent, child):
                record[key] = math.fsum(a[total] for a in record[slab_anchors.FIELD])
        for record in (parent, child):
            if _has_mass(record):
                record['slab_retired_excess_mass_kg'] = max(0., record['slab_initial_excess_mass_kg']
                    + record['slab_fed_excess_mass_kg']-record[RETAINED_MASS_FIELD])
            record['slab_retired_area_km2'] = max(0., record.get(INITIAL_AREA_FIELD, 0.)
                + record['slab_fed_area_km2']-record['slab_retained_area_km2'])
        child['slab_attachment'] = parent['slab_attachment']
        child[LINE_LOAD_FIELD] = parent[LINE_LOAD_FIELD]
        refresh_line_load(parent)
        refresh_line_load(child)
        return
    fraction = float(fraction)
    if not math.isfinite(fraction) or not 0. <= fraction <= 1.:
        raise ValueError('Slab partition fraction must lie in [0,1].')
    import slab_tether_history as neck_history
    channels=neck_history.partition_data(parent,child,fraction,channel_fractions)
    import slab_tether_local as local
    if local.enabled(parent):
        left,right=deepcopy(parent),deepcopy(child)
        left[neck_history.FIELD],right[neck_history.FIELD]=channels
        right[neck_history.VERSION_FIELD]=local.VERSION
        for record in (left,right):
            # Every label follows its local patch share. A global trace-length
            # fraction can differ from the initial area actually transferred.
            local.aggregate(record)
            record['slab_attachment']=parent['slab_attachment']
            refresh_line_load(record)
            validate_row(record,require_mass=True)
        parent.clear();parent.update(left)
        child.clear();child.update(right)
        return
    for key in FIELDS + (MASS_FIELDS if _has_mass(parent) else ()) + ((INITIAL_AREA_FIELD,) if INITIAL_AREA_FIELD in parent else ()):
        child[key] = parent[key]*fraction
        parent[key] -= child[key]
    child['slab_attachment'] = parent['slab_attachment']
    # Legacy density is inherited. V2 derives density from each mass share and
    # refreshes it again when trench_history commits the new trace lengths.
    child[LINE_LOAD_FIELD] = parent[LINE_LOAD_FIELD]
    refresh_line_load(parent)
    refresh_line_load(child)
    if channels is not None:
        parent[neck_history.FIELD],child[neck_history.FIELD]=channels
        child[neck_history.VERSION_FIELD]=1
        neck_history.validate(parent);neck_history.validate(child)


def join(source, target):
    if source is target:
        raise ValueError('A slab cannot be joined into itself.')
    ensure(source)
    ensure(target)
    validate_row(source)
    validate_row(target)
    mass_state = _has_mass(source)
    if mass_state != _has_mass(target):
        raise ValueError('Cannot join legacy and conservative slab inventories.')
    import slab_tether_history as neck_history
    channels=neck_history.join_data(source,target)
    a, b = source['slab_retained_area_km2'], target['slab_retained_area_km2']
    if a+b > 0:
        target['slab_attachment'] = (a*source['slab_attachment']+b*target['slab_attachment'])/(a+b)
        if not mass_state:
            target[LINE_LOAD_FIELD] = (a*source[LINE_LOAD_FIELD]+b*target[LINE_LOAD_FIELD])/(a+b)
    if INITIAL_AREA_FIELD in source or INITIAL_AREA_FIELD in target:
        source.setdefault(INITIAL_AREA_FIELD, 0.)
        target.setdefault(INITIAL_AREA_FIELD, 0.)
    for key in FIELDS + (MASS_FIELDS if mass_state else ()) + ((INITIAL_AREA_FIELD,) if INITIAL_AREA_FIELD in source else ()):
        target[key] += source[key]
        source[key] = 0.
    import slab_anchors
    if slab_anchors.carries(source) or slab_anchors.carries(target):
        source.setdefault(slab_anchors.FIELD, []); target.setdefault(slab_anchors.FIELD, [])
        slab_anchors.join(source, target)
        slab_anchors._match_totals(target)
    source[LINE_LOAD_FIELD] = 0.
    refresh_line_load(target)
    if channels is not None:
        source[neck_history.FIELD]=[];target[neck_history.FIELD]=channels
        neck_history.validate(source);neck_history.validate(target)


def bootstrap_slab_state(s):
    """Give already-running trenches the line load their history implies.

    Called once by a migration script, never by the engine. A run that predates
    S_k has no record of the mass it put down a trench, only of how much
    shortening and area that trench accumulated. Reconstruct the full retained
    mass from the recorded area and its thermal proxy. The force reader then
    applies the same depth window as it does to natively fed slabs; clipping the
    baseline here too would count the depth window twice. Positive loads are left
    alone, so the helper is idempotent and safe to re-run.
    """
    if not enabled(s):
        return dict(version=VERSION, bootstrapped=0, skipped=0)
    done, skipped = 0, 0
    for row in s.trench_systems:
        ensure(row)
        validate_row(row, require_mass=conservative(s))
        if row[LINE_LOAD_FIELD] > 0. or row['phase'] in ('shutdown', 'joined'):
            skipped += 1
            continue
        retained = float(row.get('slab_retained_area_km2', 0.))
        buoyancy = float(row.get('slab_retained_buoyancy_area_km2', 0.))
        shortening = float(row.get('shortening_km', 0.) or 0.)
        if retained <= 0. or shortening <= 0.:
            skipped += 1
            continue
        import slab_tether_history
        if slab_tether_history.FIELD in row:
            raise ValueError('Reconstruct slab mass before explicitly initializing its neck histories.')
        age = buoyancy_to_age_myr(buoyancy/max(retained, 1e-30))
        trace_km = _length_m(row)/1e3
        if trace_km <= 0.:
            raise ValueError('Slab bootstrap requires a positive recorded trace length.')
        # Area/trace is an along-slab length, not vertical depth. The retained
        # area already accounts for retirement; do not recreate retired mass.
        length_km = retained/trace_km
        row[LINE_LOAD_FIELD] = float(excess_mass_per_area_kg_m2(age)*length_km*1e3)
        if conservative(s):
            mass = row[LINE_LOAD_FIELD]*_length_m(row)
            row['slab_initial_excess_mass_kg'] += mass
            row[RETAINED_MASS_FIELD] += mass
            refresh_line_load(row)
        validate_row(row)
        done += 1
    s.slab_bootstrap_diagnostics = dict(version=s.slab_memory_version, bootstrapped=done, skipped=skipped,
        dip_deg=SUBDUCTION_DIP_DEG, maximum_slab_depth_km=MAX_SLAB_DEPTH_KM,
        maximum_slab_length_km=upper_mantle_length_km(),
        saturated_line_load_kg_per_m=SATURATED_LINE_LOAD_KG_PER_M,
        source='retained area per trace length, weighed by the retained inventory cooling age; force depth window applied on read')
    return s.slab_bootstrap_diagnostics


def advance(s, dt):
    """Consume the committed source-step sink record exactly once, before rematch."""
    if not enabled(s):
        return
    dt = float(dt)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('Slab inventory needs finite positive elapsed time.')
    end = float(s.t)
    previous = getattr(s, '_slab_last_update_myr', None)
    if previous is not None and end <= previous:
        return
    record = s.native_subduction_diagnostics
    if abs(record['step_end_myr']-end) > 1e-9 or abs(record['step_duration_myr']-dt) > 1e-9:
        raise ValueError('Slab feeding must use the just-completed physical removal record.')
    feeds = record.get('removed_area_by_trench')
    if feeds is None:
        raise ValueError('A slab-memory run requires local accepted-removal provenance.')
    rows = {row['id']: row for row in s.trench_systems}
    for row in rows.values():validate_row(row, require_mass=conservative(s))
    fed = {}
    for entry in feeds:
        ident = int(entry['trench_id'])
        amount, buoyancy = float(entry['area_km2']), float(entry['buoyancy_area_km2'])
        if (ident not in rows or not np.isfinite([amount, buoyancy]).all()
                or amount < 0 or not 0 <= buoyancy <= 1.8*amount+1e-8):
            raise ValueError('Actual slab feed requires a known local trench and bounded buoyancy area.')
        row = rows[ident]
        if (int(entry['downgoing_plate_uid']) != row['downgoing_plate_uid']
                or row['phase'] == 'joined' or ident in fed):
            raise ValueError('Slab feed has stale polarity, joined identity or duplicate attribution.')
        fed[ident] = (amount, buoyancy)
    total = math.fsum(value[0] for value in fed.values())
    if abs(total-record['removed_area_km2']) > max(1e-6, total*2e-11):
        raise ValueError('Local slab feed must close to the existing actual ocean sink.')
    decay = math.exp(-dt/RETENTION_MYR)
    # Exact constant-input integration over this source step, not an impulse
    # that changes the retained inventory merely by halving the timestep.
    retain_feed = -math.expm1(-dt/RETENTION_MYR)*RETENTION_MYR/dt
    import slab_anchors
    anchored = slab_anchors.enabled(s)
    if anchored:
        slab_anchors.ensure(s)
    staged = deepcopy(s.trench_systems)
    feed_records={entry['trench_id']:entry for entry in feeds}
    for row in staged:
        ensure(row)
        if row['phase'] == 'joined':
            continue
        import slab_tether_local as local_necks
        if local_necks.enabled(row):
            local_necks.advance_source(row,feed_records.get(row['id']),decay,retain_feed)
            refresh_line_load(row)
            validate_row(row,require_mass=True)
            continue
        amount, buoyancy = fed.get(row['id'], (0., 0.))
        old = row['slab_retained_area_km2']
        new = old*decay+amount*retain_feed
        row['slab_fed_area_km2'] += amount
        row['slab_retained_area_km2'] = new
        row['slab_retained_buoyancy_area_km2'] = row['slab_retained_buoyancy_area_km2']*decay+buoyancy*retain_feed
        row['slab_retired_area_km2'] += old+amount-new
        if conservative(s):
            # Integrate mass before deriving density. Trace rematching, splitting
            # and joining cannot create or destroy this extensive inventory.
            gain_mass = (float(excess_mass_per_area_kg_m2(buoyancy_to_age_myr(buoyancy/amount)))*amount*1e6
                         if amount > 0. else 0.)
            old_mass = row[RETAINED_MASS_FIELD]
            new_mass = old_mass*decay+gain_mass*retain_feed
            row['slab_fed_excess_mass_kg'] += gain_mass
            row[RETAINED_MASS_FIELD] = new_mass
            row['slab_retired_excess_mass_kg'] += old_mass+gain_mass-new_mass
            refresh_line_load(row)
            import slab_tether_history
            slab_tether_history.advance(row,decay,amount*retain_feed,buoyancy*retain_feed,gain_mass*retain_feed)
            if anchored and slab_anchors.carries(row):
                located = [(np.asarray(p['xyz'], float), p['area_km2'], p['mass_weight_kg'])
                           for p in feed_records.get(row['id'], {}).get('located', [])]
                if amount > 0. and not located:
                    raise ValueError('Located slab allocation needs the feed contacts of every accepted slab.')
                slab_anchors.advance(row, decay, retain_feed, located, amount, gain_mass)
        else:
            # Preserve the historical per-length integration only for v1.
            length_m = max(float(row.get('length_km', 0.) or 0.), 1e-9)*1e3
            gain = 0.
            if amount > 0. and length_m > 1e-3:
                age = buoyancy_to_age_myr(buoyancy/amount)
                gain = float(excess_mass_per_area_kg_m2(age)*(amount*1e6/length_m))
            row[LINE_LOAD_FIELD] = row[LINE_LOAD_FIELD]*decay+gain*retain_feed
        validate_row(row)
    s.trench_systems = staged
    s._slab_last_update_myr = end
    s.slab_memory_diagnostics = dict(version=s.slab_memory_version, step_end_myr=end,
        step_fed_area_km2=total, retention_myr=RETENTION_MYR,
        developed_length_km=DEVELOPED_LENGTH_KM, dip_deg=SUBDUCTION_DIP_DEG,
        saturated_line_load_kg_per_m=SATURATED_LINE_LOAD_KG_PER_M,
        fed_area_km2=math.fsum(row['slab_fed_area_km2'] for row in staged),
        retained_area_km2=math.fsum(row['slab_retained_area_km2'] for row in staged),
        retired_area_km2=math.fsum(row['slab_retired_area_km2'] for row in staged),
        maximum_line_load_kg_per_m=max((row[LINE_LOAD_FIELD] for row in staged), default=0.),
        source='actual unique native removal after maturity and available-overlap caps',
        limitation='retained area, line load and cooling-age proxy; no slab depth, dip, temperature or mantle flow solve')
    if conservative(s):
        s.slab_memory_diagnostics.update({key:math.fsum(row[key] for row in staged) for key in MASS_FIELDS})
        s.slab_memory_diagnostics[INITIAL_AREA_FIELD] = math.fsum(row.get(INITIAL_AREA_FIELD, 0.) for row in staged)


def coupled_fraction(row, trace_length_km=None):
    """Uniform-column share above the depth boundary; leave ledgers untouched.

    Use the same current attached trace for mass and area allocation. A shrinking
    trace must not amplify pull after the effective column fills the depth window.
    No represented area means no geometrically supported pulling column.
    """
    length = float(row.get('length_km', 0.) if trace_length_km is None else trace_length_km)
    area = float(row.get('slab_retained_area_km2', 0.))
    if length <= 0. or area <= 0.:
        return 0.
    return min(1., upper_mantle_length_km()*length/area)


def slab_length_km(row, trace_length_km=None):
    """Effective pulling down-dip extent, not the full inventory's extent."""
    length = float(row.get('length_km', 0.) if trace_length_km is None else trace_length_km)
    if length <= 0.:
        return 0.
    return min(float(row.get('slab_retained_area_km2', 0.))/length, upper_mantle_length_km())


def strength(row):
    """Normalised S_k/S_sat view retained for the kinematic rollback consumers."""
    validate_row(row)
    if row['phase'] in ('shutdown', 'joined'):
        return 0.
    load = line_density(row)*coupled_fraction(row)
    if load <= 0.:
        return 0.
    return min(1.8, load/SATURATED_LINE_LOAD_KG_PER_M)*min(1., float(row['maturity']))


def pull_state(s):
    """Retained pull can briefly oppose collision shutdown; never starts new sinks."""
    owners = np.full(len(s.ba), -1, int)
    weight = np.zeros(len(s.ba))
    active = {int(s.plate_uid[p]): int(p) for p in np.flatnonzero(s.active)}
    ids = np.asarray(getattr(s, 'trench_id', np.zeros(len(s.ba), int)))
    for row in s.trench_systems:
        validate_row(row, require_mass=conservative(s))
        down, over = active.get(row['downgoing_plate_uid']), active.get(row['overriding_plate_uid'])
        value = strength(row)
        if value <= 0 or down is None or over is None:
            continue
        # No bcode gate. An attached slab pulls because it exists, not because
        # this step's kinematic label happened to land on 2, 3 or 4; gating a
        # force on a label computed from the velocity the force produces is what
        # made the old law discontinuous in omega.
        valid = ((ids == row['id'])
                 & (((s.bp == down) & (s.bq == over)) | ((s.bp == over) & (s.bq == down))))
        owners[valid], weight[valid] = down, value
    return owners, weight


def line_load(s):
    """Per boundary edge: downgoing owner, S_k in kg/m, and retained slab length.

    The force balance sums g*S_k*sin(dip) over every attached edge. In v2,
    retained mass AND area are allocated across the currently matched edges.
    Only the uniform column's share above MAX_SLAB_DEPTH_KM transmits pull;
    the rest remains in the inventory. Subdivision cannot change this share,
    but actual trace shortening can put more of the column below the window.
    """
    count = len(s.ba)
    owners = np.full(count, -1, int)
    load = np.zeros(count)
    length = np.zeros(count)
    active = {int(s.plate_uid[p]): int(p) for p in np.flatnonzero(s.active)}
    ids = np.asarray(getattr(s, 'trench_id', np.zeros(count, int)))
    for row in s.trench_systems:
        validate_row(row, require_mass=conservative(s))
        down, over = active.get(row['downgoing_plate_uid']), active.get(row['overriding_plate_uid'])
        if down is None or over is None or row['phase'] in ('shutdown', 'joined'):
            continue
        valid = ((ids == row['id'])
                 & (((s.bp == down) & (s.bq == over)) | ((s.bp == over) & (s.bq == down))))
        import slab_anchors
        if conservative(s) and slab_anchors.carries(row) and np.any(valid):
            # Located allocation: each edge pulls with the slab that went down
            # near it, through its own upper-mantle window.
            edges = np.flatnonzero(valid)
            length_km = np.asarray(s.bl, float)[edges]
            mass, area, _ = slab_anchors.edge_loads(row, edges, np.asarray(s.bmid)[edges], length_km)
            with np.errstate(divide='ignore', invalid='ignore'):
                column_km = np.where(length_km > 0., area/np.maximum(length_km, 1e-30), 0.)
                coupled = np.where(column_km > 0., np.minimum(1., upper_mantle_length_km()/np.maximum(column_km, 1e-30)), 0.)
                value = np.where(length_km > 0., mass/np.maximum(length_km*1e3, 1e-30), 0.)*coupled
            if not np.isfinite(value).all():
                raise ValueError('Located slab allocation produced a nonfinite line load.')
            pulled = value > 0.
            owners[edges[pulled]] = down
            load[edges[pulled]] = value[pulled]
            length[edges[pulled]] = np.minimum(column_km[pulled], upper_mantle_length_km())
            continue
        if conservative(s):
            lengths = np.asarray(s.bl, float)[valid]*1e3
            if not np.isfinite(lengths).all() or np.any(lengths < 0.):
                raise ValueError('Attached slab edges require finite nonnegative lengths.')
            attached_length = float(lengths.sum())
            value = row[RETAINED_MASS_FIELD]/attached_length if attached_length > 0. else 0.
            trace_km = attached_length/1e3
        else:
            value = float(row.get(LINE_LOAD_FIELD, 0.))
            trace_km = float(row.get('length_km', 0.))
        value *= coupled_fraction(row, trace_km)
        if not math.isfinite(value):
            raise ValueError('Slab mass allocation produced a nonfinite line load.')
        if value <= 0.:
            continue
        owners[valid] = down
        load[valid] = value
        length[valid] = slab_length_km(row, trace_km)
    return owners, load, length


def snapshot(s):
    if not enabled(s):
        return {}
    for row in s.trench_systems:validate_row(row, require_mass=conservative(s))
    diagnostics = deepcopy(getattr(s, 'slab_memory_diagnostics',
                    dict(version=s.slab_memory_version, state='initialized', fed_area_km2=0.,
                         retained_area_km2=0., retired_area_km2=0., retention_myr=RETENTION_MYR,
                         developed_length_km=DEVELOPED_LENGTH_KM,
                         saturated_line_load_kg_per_m=SATURATED_LINE_LOAD_KG_PER_M)))
    if conservative(s):
        diagnostics.update(version=VERSION,
            **{key:math.fsum(row[key] for row in s.trench_systems) for key in MASS_FIELDS})
        diagnostics[INITIAL_AREA_FIELD] = math.fsum(row.get(INITIAL_AREA_FIELD, 0.) for row in s.trench_systems)
        for field in ('fed_area_km2', 'retained_area_km2', 'retired_area_km2'):
            diagnostics[field] = math.fsum(row['slab_'+field] for row in s.trench_systems)
        if hasattr(s, 'slab_mass_migration'):
            diagnostics['mass_migration'] = deepcopy(s.slab_mass_migration)
    return dict(slab_memory_version=s.slab_memory_version, slab_memory_diagnostics=diagnostics)
