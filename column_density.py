"""Isostatic gravitational coefficients for a retained dense basal phase.

Each material column has dense crust at its base and ordinary crust above it.
At fixed phase volumes their thickness fractions are unchanged by lateral strain.
The coefficients are derived from integral(rho*g*z dz) relative to mantle, with
local Airy compensation. They are not an additional body-force prescription.
"""
import numpy as np

RHO_CRUST = 2800.
RHO_DENSE = 3450.
RHO_MANTLE = 3300.
REFERENCE_COEFFICIENT = RHO_CRUST*(1.-RHO_CRUST/RHO_MANTLE)
DENSE_VOLUME_FIELD = 'dense_crust_km_per_reference_km2'
# Largest distinct-sheet overlap that may lack a persistent layer order. Only
# ContactAdmission records an order, and only for motion; arc emplacement admits
# up to max(1e-9, 1e-12*area) km2 of new overlap measured by polygon clipping,
# while mesh_coverage counts any overlap above 1e-8 km2 by a different routine.
# Run SEP21T stopped at 85 Myr on one such pair: arcs 38 and 93 on block 02
# touching with 1.83e-8 km2 (0.018 m2) of overlap. Below this floor an
# unordered pair is edge contact at any map resolution; it takes the mean of
# its two stack orders. Larger unordered overlap, or a contradictory order,
# still raises. PHYSICS_REPAIRS §17.
UNORDERED_OVERLAP_FLOOR_KM2 = 1.


def options(s):
    """One profile shared by plate torques, sheet relaxation and energy audits.

    The phase model must explicitly supply retained physical dense volume per
    reference area. Reading this function cannot initialize a missing phase.
    """
    version = getattr(s, 'retained_dense_crust_version', 0)
    if isinstance(version, (bool, np.bool_)) or not isinstance(version, (int, np.integer)) or version not in (0, 1):
        raise ValueError('Unsupported retained dense-crust version.')
    if not version:
        return {}
    import collision_surface
    state = s.structure
    if DENSE_VOLUME_FIELD not in state:
        raise ValueError('Density-dependent gravity requires retained dense-crust inventory.')
    volume = np.asarray(state['thickness_km'])*np.asarray(state['area_factor'])
    dense = np.asarray(state[DENSE_VOLUME_FIELD])
    fraction = dense/volume
    if (dense.shape != volume.shape or not np.isfinite(volume).all() or np.any(volume <= 0)
            or not np.isfinite(fraction).all() or np.any((fraction < 0) | (fraction > 1))):
        raise ValueError('Retained dense volume exceeds its physical material column.')
    return dict(density_profile=dict(dense_fraction=fraction,
        sheet_order=collision_surface.descendants(s.collision_contacts)))


def weights(fraction, sheets, first, second, order, area_km2=None):
    """Return self and ordered cross terms, divided by the legacy density factor.

    For dense thickness D and ordinary thickness B (T=B+D), the self term is
    g/2 * [rho_d D^2 + rho_c B^2 + 2 rho_c B D - M^2/rho_m], where
    M=rho_d D+rho_c B is mass per area. A lower column i and upper column j
    contribute g * [M_j*T_i - M_i*M_j/rho_m]. All lengths here are normalized
    by T. Multiplying by A*T^2 or overlap*T_i*T_j restores the geometry.
    """
    fraction = np.asarray(fraction, float)
    sheets = np.asarray(sheets)
    if (fraction.shape != sheets.shape or fraction.ndim != 1
            or not np.isfinite(fraction).all() or np.any((fraction < 0) | (fraction > 1))):
        raise ValueError('Dense basal fractions must align with material sheets and lie in [0,1].')
    density = RHO_CRUST*(1.-fraction)+RHO_DENSE*fraction
    self_weight = (RHO_DENSE*fraction**2+RHO_CRUST*(1.-fraction)**2
                   +2.*RHO_CRUST*fraction*(1.-fraction)-density**2/RHO_MANTLE)/REFERENCE_COEFFICIENT
    if area_km2 is not None:
        area_km2 = np.asarray(area_km2, float)
        if area_km2.shape != (len(first),) or not np.isfinite(area_km2).all() or np.any(area_km2 < 0):
            raise ValueError('Overlap areas must align with overlap pairs and be finite and non-negative.')
    def stacked(upper, lower):
        return density[upper]*(1.-density[lower]/RHO_MANTLE)/REFERENCE_COEFFICIENT
    pair_weight = np.empty(len(first))
    for k, (a, b) in enumerate(zip(first, second)):
        sa, sb = int(sheets[a]), int(sheets[b])
        a_above = sb in order.get(sa, ())
        b_above = sa in order.get(sb, ())
        if a_above == b_above:
            if (not a_above and area_km2 is not None
                    and area_km2[k] <= UNORDERED_OVERLAP_FLOOR_KM2):
                pair_weight[k] = .5*(stacked(a, b)+stacked(b, a))
                continue
            raise ValueError('Density-dependent overlap requires a unique persistent layer order.')
        upper, lower = (a, b) if a_above else (b, a)
        pair_weight[k] = stacked(upper, lower)
    return self_weight, pair_weight


def coefficients(profile, sheets, first, second, area_km2=None):
    if profile is None:
        return np.ones(len(sheets)), np.ones(len(first))
    return weights(profile['dense_fraction'], sheets, first, second, profile['sheet_order'], area_km2)


class ContactAdmission:
    """Stage first-contact order on candidate geometry, commit only acceptance.

    Uses the same contacting-height policy as the contact ledger. Existing order
    is immutable. Rejected line-search candidates never acquire persistent
    contacts; the caller publishes accepted records only after motion succeeds.
    """
    def __init__(self,s,volumes_km3):
        from copy import copy,deepcopy
        import structure_engine
        self.state=copy(s)
        self.state.collision_contacts=deepcopy(s.collision_contacts)
        self.profile=options(s)['density_profile']
        self.volumes=np.asarray(volumes_km3)
        self.height=structure_engine.material_height(s.kind,s.relief).copy()
        self.thickness=np.asarray(s.structure['thickness_km']).copy()
        self.density=RHO_CRUST+(RHO_DENSE-RHO_CRUST)*self.profile['dense_fraction']

    def prepare(self,points):
        from copy import copy,deepcopy
        import collision_contacts,collision_surface,mesh_coverage,material_surface
        import crustal_structure as columns
        staged=copy(self.state)
        staged.collision_contacts=deepcopy(self.state.collision_contacts)
        surface=staged.material_surface
        radius=surface.get('radius_km',6371.)
        sheets=staged.parcel_collision_sheet
        overlap=mesh_coverage.material_overlaps(points,surface['faces'],sheets,radius_km=radius)
        first,second,area=(overlap[k] for k in ('first','second','area_km2'))
        air_change=(self.volumes/material_surface.spherical_face_areas(points,surface['faces'],radius)
                    -self.thickness)*(1.-self.density/RHO_MANTLE)*1000.
        heights=columns._water_load(columns._air_height(self.height)+air_change)
        records={tuple(sorted((r['top_sheet'],r['under_sheet']))) for r in staged.collision_contacts}
        pairs=np.sort(np.column_stack((sheets[first],sheets[second])),axis=1)
        unique,inverse=np.unique(pairs,axis=0,return_inverse=True)
        for index,(one,two) in enumerate(unique):
            if (int(one),int(two)) in records:continue
            take=inverse==index
            collision_contacts.admit_contact(staged,one,two,first[take],second[take],area[take],heights)
        profile=dict(dense_fraction=self.profile['dense_fraction'],
                     sheet_order=collision_surface.descendants(staged.collision_contacts))
        return staged,profile,overlap

    def accept(self,candidate):
        self.state,self.profile,self.overlap=candidate

    def publish(self,s):
        s.collision_contacts=self.state.collision_contacts
        s.next_collision_contact_id=self.state.next_collision_contact_id
