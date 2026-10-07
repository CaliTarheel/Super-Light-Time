"""Material-attached, progressive continental rifts for the reduced model.

Boundary velocity contrasts are a proxy for loading, not stresses in pascals.
A spherical relative-strength membrane distributes that loading. Only tensile
strain damages its links. No plate-wide timer or prescribed opening rotation
can create a continental daughter through this path.
"""
from copy import deepcopy
import math

import numpy as np

import rift_material
import rift_mechanics
import rift_mesh
import structure_engine
import enhanced_rifting
import rift_traction
from fracture import component_labels
from material_geometry import splits_protected_groups

RADIUS_KM = 6371.
MODEL = 'local material extension and damage'
# Mode-independent breakthrough guards, named so a caller can retune them
# rather than editing literals inside commit(). The accumulated extension gate
# depends on the rupture criterion version and on which measure the mechanics
# mode stores; see _rupture_calibration.
RUPTURE_DAMAGE = .95
RUPTURE_OPENING_KM_MYR = .02
# The legacy membrane's unresolved-strain partition: the share of solved
# loading-proxy opening that counts as realized extension of the material.
# The same share already charges legacy damage strain and column thinning.
# It is itself an uncalibrated reduced-model partition, not a measured
# fraction, so a legacy gate expressed through it is only as physical as it.
PROXY_REALIZED_FRACTION = .25
# The two damage-strain measures, one per mechanics mode, and the extension
# each mode supplies to rupture criterion 1.
REALIZED_STRAIN_MEASURE = 'realized native material-link logarithmic extension'
PROXY_STRAIN_MEASURE = 'quarter-partitioned positive local-loading extension proxy'
# Bond extension adds positive lengthening only; shortening never reduces it,
# so a link keeps kilometres opened before any later inversion.
REALIZED_EXTENSION_MEASURE = 'gross positive realized link lengthening, never reduced by shortening (km)'
PROXY_EXTENSION_MEASURE = 'quarter of gross positive local-loading proxy opening, never reduced by shortening (km)'
# Rupture criterion 1 (fresh reviewed worlds). Version 0 keeps the recorded
# mode-specific strain gates of saved worlds and checkpoints.
RUPTURE_CRITERION_VERSION = 1
# Total realized horizontal extension a link must accumulate before breakup.
#
# Breakup is reached after a finite extension across the rift, a length that
# does not depend on how finely the rift is sampled. Restorations of the
# magma-poor Iberia-Newfoundland conjugate margins need 229-256 km to breakup
# (Sutra et al., 2013, G-cubed 14); the magma-assisted Main Ethiopian Rift
# reached incipient breakup after about 60-80 km (Corti, 2009, Earth-Sci. Rev.
# 96). Necking domains are about 55 km wide per margin (Chenin et al., 2025).
# A pure-shear neck that thins crust to zero conserves crustal area,
# T0*W0 = T0*(W0+E)/2, so its extension E equals its original width W0: two
# ~55 km necking domains need at least ~55 km. 100 km lies between the
# magma-assisted and magma-poor cases. (That both earlier strain gates implied
# about 103 km at the resolutions they were set on, enhanced 0.35 at 246 km
# links and legacy 0.15 at 687 km, is a consistency check, not the reason.)
#
# Typical mechanics links run from about 246 km (8192 nodes) to about 1,970 km
# (128 nodes), all longer than an Earth neck, so each link's own extension is
# compared directly. A mesh with links shorter than the neck (budgets above
# about 59,000 nodes) would need the series form E_b*min(1, L0/W0) instead.
# The model's own realized deformation localizes over deforming belts of
# deformation_width_km (default 400 km) with 100 km smoothing, not over a
# 55 km neck; the gate is length independent by construction, but whether the
# kilometres a rift reaches are the same across node budgets and belt widths
# still needs the multi-resolution comparison of gap G121.
BREAKUP_EXTENSION_KM = 100.
RUPTURE_CRITERION = 'accumulated realized breakup extension (km)'
# Criterion-1 labels each mechanics mode must carry together.
_CRITERION_ONE_MODES = {
    REALIZED_STRAIN_MEASURE: (1., REALIZED_EXTENSION_MEASURE),
    PROXY_STRAIN_MEASURE: (PROXY_REALIZED_FRACTION, PROXY_EXTENSION_MEASURE),
}


# Rift commit policy 1 (fresh reviewed worlds). Local accretion runs between the
# loading update and the commit of every step. Version 0, the policy of every
# saved world, refuses every continental cut of the step when any rift-mesh
# node or link anywhere changed in between, so one terrane transfer on any
# plate vetoed breakup worldwide. Version 1 carries the pending solve over to
# the current mesh by the persistent keys rift_bonds use, and defers only cuts
# whose own loading was gathered on material that has since changed; see
# commit(). The unit that loading was gathered on follows from each mechanics
# mode: realized links measure their own lengthening, so a connected material
# component is independent of every other; the legacy membrane assigns each
# plate boundary face to its nearest node of that plate, so the whole plate is.
RIFT_COMMIT_VERSION = 1
RIFT_COMMIT_POLICY = 'defer only cuts whose loading unit changed after loading'
REALIZED_LOADING_UNIT = 'connected material component'
PROXY_LOADING_UNIT = 'plate'


def rift_commit_version(s):
    version = getattr(s, 'rift_commit_version', 0)
    if (isinstance(version, (bool, np.bool_)) or not isinstance(version, (int, np.integer))
            or version not in (0, RIFT_COMMIT_VERSION)):
        raise ValueError('Unsupported continental rift commit policy version.')
    return int(version)


def _commit_labels(s):
    """Rift mechanics labels of the commit policy; none for saved worlds."""
    if not rift_commit_version(s):
        return {}
    return dict(commit_policy_version=RIFT_COMMIT_VERSION, commit_policy=RIFT_COMMIT_POLICY,
                commit_loading_unit=REALIZED_LOADING_UNIT if enhanced_rifting.enabled(s) else PROXY_LOADING_UNIT)


def rupture_criterion_version(s):
    version = getattr(s, 'rupture_criterion_version', 0)
    if (isinstance(version, (bool, np.bool_)) or not isinstance(version, (int, np.integer))
            or version not in (0, RUPTURE_CRITERION_VERSION)):
        raise ValueError('Unsupported continental rupture criterion version.')
    return int(version)


def _rupture_calibration(s):
    """Keep the coarse loading proxy distinct from realized material strain.

    Version 0 gates are numerical strain thresholds, not measured lithosphere
    failure constants; the kilometres they require grow with link length.
    Version 1 gates realized extension in km, the same at every mesh budget.
    Neither changes the existing accumulation or converts history.
    """
    if rupture_criterion_version(s) == 0:
        # Version 0 is exactly the per-run strain gate of PR #188.
        if enhanced_rifting.enabled(s):
            # An experiment keeps the realized-strain gate it was started with: a
            # run records its own value, and only an unrecorded one takes the
            # reviewed default. Run SEP21T was started at .30 and silently moved
            # to .35 when this calibration arrived in a source migration.
            threshold = getattr(s, 'realized_rupture_strain_threshold', None)
            if threshold is None:
                threshold = .35
            elif (isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, (int, float, np.floating))
                    or not 0. < float(threshold) < 1.):
                raise ValueError('A recorded realized rupture strain threshold must be a number in (0, 1).')
            return dict(damage_strain_measure=REALIZED_STRAIN_MEASURE,
                        rupture_strain_threshold=float(threshold))
        return dict(damage_strain_measure=PROXY_STRAIN_MEASURE,
                    rupture_strain_threshold=.15)
    # Version 1 has no strain threshold, so a recorded per-run strain value
    # (realized_rupture_strain_threshold) has nothing to select here.
    enhanced = enhanced_rifting.enabled(s)
    return dict(damage_strain_measure=REALIZED_STRAIN_MEASURE if enhanced else PROXY_STRAIN_MEASURE,
                rupture_criterion_version=RUPTURE_CRITERION_VERSION,
                rupture_criterion=RUPTURE_CRITERION,
                breakup_extension_km=BREAKUP_EXTENSION_KM,
                realized_extension_fraction=1. if enhanced else PROXY_REALIZED_FRACTION,
                rupture_extension_measure=REALIZED_EXTENSION_MEASURE if enhanced else PROXY_EXTENSION_MEASURE)


def _unit(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-20)


def initialize(s):
    s.rift_bonds = {}
    s.rift_systems = []
    s.next_rift_system_id = 1
    s.rift_mechanics = dict(model=MODEL, mesh_nodes=0, mesh_edges=0,
                            target_nodes=int(s.config['mechanics_nodes']),
                            solver_iterations=0, residual=0., converged=True,
                            **_rupture_calibration(s), **_commit_labels(s))
    mesh = rift_material.refresh(s)
    s.rift_mechanics.update(mesh_nodes=len(mesh['xyz']), mesh_edges=len(mesh['edges']))


def advect(s, dt, rotate):
    for row in s.rift_systems:
        slots = np.flatnonzero(s.active & (s.plate_uid == row['plate_uid']))
        if len(slots) and row['geometry_xyz']:
            row['geometry_xyz'] = rotate(np.asarray(row['geometry_xyz']), s.omega[slots[0]]*dt).tolist()


def _edge_keys(mesh):
    return [(*sorted((int(mesh['bases'][a]), int(mesh['bases'][b]))), int(mesh['owner_uids'][a]))
            for a, b in mesh['edges']]


def _node_keys(mesh):
    return list(zip(np.asarray(mesh['bases']).astype(np.int64).tolist(),
                    np.asarray(mesh['owner_uids']).astype(np.int64).tolist()))


def remap_pending(pending_mesh, current):
    """Index the pending solve by the current mesh's persistent keys.

    Returns (node, edge): for every current node the pending node with the same
    (base, owner uid), and for every current link the pending link with the
    same _edge_keys key (sorted base pair, owner uid), or -1 where the pending
    solve has none. Keys are the ones rift_bonds persist under, so arrays that
    were merely reordered (a juvenile parcel, another owner's new node) map
    back exactly.
    """
    old_nodes = {key: index for index, key in enumerate(_node_keys(pending_mesh))}
    old_edges = {key: index for index, key in enumerate(_edge_keys(pending_mesh))}
    node = np.array([old_nodes.get(key, -1) for key in _node_keys(current)], np.int64)
    edge = np.array([old_edges.get(key, -1) for key in _edge_keys(current)], np.int64)
    return node, edge


def material_image(s, mesh):
    """Each continental patch's node key at this refresh, for commit policy 1.

    A node key (base, owner uid) can survive a partial ownership transfer with
    less material in it, so comparing keys alone cannot see a donor node that
    kept its key. Patch IDs are persistent material identities (a raster patch
    never spans two owners; a native patch is one face), so the patch -> node
    key image records which material each node's loading was gathered on.
    """
    parcel_node = np.asarray(mesh['parcel_node'])
    selected = parcel_node >= 0
    patches, first = np.unique(np.asarray(s.parcel_patch, np.int64)[selected], return_index=True)
    nodes = parcel_node[selected][first]
    return dict(material_patch_ids=patches,
                material_patch_bases=np.asarray(mesh['bases'], np.int64)[nodes],
                material_patch_owner_uids=np.asarray(mesh['owner_uids'], np.int64)[nodes])


def reassigned_material(pending, current_image):
    """Node keys whose material membership changed since loading.

    Returns (keys, count): the pending and current node keys of every patch
    present at both times whose node key differs, and the number of such
    patches. A patch that appears or disappears (new juvenile or arc crust,
    consumed material) is not a transfer between nodes; any node it creates or
    removes is already a key change.
    """
    old = np.asarray(pending['material_patch_ids'], np.int64)
    new = np.asarray(current_image['material_patch_ids'], np.int64)
    _, i, j = np.intersect1d(old, new, assume_unique=True, return_indices=True)
    old_bases = np.asarray(pending['material_patch_bases'], np.int64)[i]
    old_uids = np.asarray(pending['material_patch_owner_uids'], np.int64)[i]
    new_bases = np.asarray(current_image['material_patch_bases'], np.int64)[j]
    new_uids = np.asarray(current_image['material_patch_owner_uids'], np.int64)[j]
    moved = (old_bases != new_bases) | (old_uids != new_uids)
    keys = (set(zip(old_bases[moved].tolist(), old_uids[moved].tolist()))
            | set(zip(new_bases[moved].tolist(), new_uids[moved].tolist())))
    return keys, int(np.count_nonzero(moved))


def _component_signatures(mesh):
    """Label each node's linked component and key that component's material.

    A signature is the component's complete node-key and link-key sets, so two
    components compare equal only when nothing was added, removed or relinked.
    """
    count = len(mesh['bases'])
    edges = np.asarray(mesh['edges'], np.int64).reshape(-1, 2)
    labels = rift_mesh._components(count, edges, np.asarray(mesh['owner_uids'])).tolist()
    nodes, links = {}, {}
    for key, label in zip(_node_keys(mesh), labels):
        nodes.setdefault(label, set()).add(key)
    for key, a in zip(_edge_keys(mesh), edges[:, 0].tolist()):
        links.setdefault(labels[a], set()).add(key)
    return labels, {label: (frozenset(members), frozenset(links.get(label, ())))
                    for label, members in nodes.items()}


def changed_owner_uids(pending_mesh, current, reassigned=()):
    """Plate UIDs that gained or lost any node key, link key or node material.

    ``reassigned`` holds node keys whose material membership changed (see
    reassigned_material); without it only key changes are visible.
    """
    nodes = set(_node_keys(pending_mesh)) ^ set(_node_keys(current))
    edges = set(_edge_keys(pending_mesh)) ^ set(_edge_keys(current))
    return sorted({int(uid) for _, uid in nodes} | {int(key[2]) for key in edges}
                  | {int(uid) for _, uid in reassigned})


def unchanged_loading(pending_mesh, current, realized, reassigned=()):
    """True for current nodes whose pending loading was gathered on this material.

    Realized mechanics measure each link's own lengthening and each node's own
    non-rigid motion. rift_material links only same-owner nodes, so nothing
    couples two components: a component whose node and link keys are exactly
    one pending component's, and none of whose nodes gained or lost material
    to another node, was measured on the same material, whatever happened to
    another island of its plate. The legacy membrane instead assigns every
    boundary face of a plate to that plate's nearest node, placed at its
    material centroid, so a node or node material gained or lost anywhere on
    the plate can move load between its components; there the whole plate
    must be unchanged.
    """
    if not realized:
        changed = changed_owner_uids(pending_mesh, current, reassigned)
        return ~np.isin(np.asarray(current['owner_uids'], np.int64), np.asarray(changed, np.int64))
    labels, signatures = _component_signatures(current)
    solved = set(_component_signatures(pending_mesh)[1].values())
    same = {label for label, signature in signatures.items() if signature in solved}
    for key, label in zip(_node_keys(current), labels):
        if key in reassigned:
            same.discard(label)
    return np.array([label in same for label in labels], bool)


def _inherit_bonds(s, keys):
    """Copy reference-link inheritance into independently owned working links.

    A partially accreted reference node can now have two owners. Each link must
    evolve independently, even though both inherit the old reference weakness.
    New ownership copies the most recently observed same-base link once; it
    does not alias its dictionary or continually share later damage. In mixed
    nodes this remains a coarse inheritance rule, not patch-scale fault stress.
    """
    missing = [key for key in keys if key not in s.rift_bonds]
    if not missing:
        return
    previous = {}
    for key, row in s.rift_bonds.items():
        pair = key[:2]
        candidate = (float(row.get('last_seen_myr', 0.)), key)
        if pair not in previous or candidate > previous[pair][0]:
            previous[pair] = (candidate, row)
    for key in missing:
        donor = previous.get(key[:2])
        ancestry = getattr(s, 'rift_material', {}).get('base_parent', {})
        if donor is None and ancestry:
            pair = list(key[:2])
            while any(base in ancestry for base in pair):
                pair = sorted(ancestry.get(base, base) for base in pair)
                donor = previous.get(tuple(pair))
                if donor is not None:
                    break
        if donor is not None:
            s.rift_bonds[key] = dict(donor[1], inherited_from=donor[0][1])


def _node_max(values, edges, count):
    out = np.zeros(count)
    if len(edges):
        np.maximum.at(out, edges[:, 0], values)
        np.maximum.at(out, edges[:, 1], values)
    return out


def relative_strength(thickness, reference, craton, suture, heat, damage):
    """Dimensionless resistance: roots resist; thinning/heat/inheritance weaken."""
    return np.clip((np.maximum(thickness, 8.)/np.maximum(reference, 8.))**1.5
                   * (1+5*np.asarray(craton))
                   / (1+1.2*np.asarray(suture)+np.asarray(heat)/800.+2.5*np.asarray(damage)),
                   .06, 12.)


def evolve_damage(damage, strain, strength, dt, *, inherited=0., compression=0.):
    """Tension grows damage; cooling/closure can quiet a rift, never open it.

    Strain is a quarter-partitioned positive loading proxy in legacy mechanics,
    or realized positive logarithmic material-link extension in enhanced mode.
    Cumulative tensile history remains separate from healing damage; it is not
    current net stretch after compression. Coefficients are worldbuilding
    closures, not calibrated laboratory failure constants.
    """
    strain = np.maximum(np.asarray(strain), 0.)
    old = np.asarray(damage)
    # Both healing terms used to be vestigial, so damage could only ratchet up.
    # A 1200 Myr e-folding time removes 0.17% per 2 Myr step -- 15% across a whole
    # 200 Myr history -- against a growth term carrying (1+2*old)/(.55*strength),
    # which is roughly 11x stronger for weak damaged crust. Damage therefore pinned
    # at 1.0 everywhere it had ever been tensile, the three-way failure criterion
    # collapsed to its extension term alone, and abandoned rifts stayed permanently
    # weak instead of cooling and strengthening.
    #
    # 250 Myr is a lithospheric thermal-relaxation scale rather than a geological
    # eternity, and the compression coefficient is raised to match the growth
    # coefficient so that closing a rift quiets it about as fast as opening one
    # damages it. Still worldbuilding closures, not laboratory constants.
    healed = old*np.exp(-dt/250.-np.maximum(compression, 0.)*10.)
    growth = strain*(1+2*old)*(1+.8*np.asarray(inherited)) / (.55*np.maximum(strength, .1))
    return np.clip(healed+growth, 0., 1.)


def boundary_loading(s, mesh):
    """Local soft constraints from existing force-derived boundary motions.

    Integrate face length at nearby same-owner material nodes. Empty interiors
    are unconstrained, not pinned to zero. Distant ocean boundaries do not act
    as direct loads on a remote continental island.
    """
    count = len(mesh['xyz'])
    loads, lengths = np.zeros((count, 3)), np.zeros(count)
    valid = s._valid_loading_edges()
    for p in np.unique(mesh['owners']):
        nodes = np.flatnonzero(mesh['owners'] == p)
        faces = np.flatnonzero(valid & ((s.bp == p) | (s.bq == p)))
        if not len(nodes) or not len(faces):
            continue
        xyz = mesh['xyz'][nodes]
        reach = np.maximum(400., 2*np.sqrt(mesh['area'][nodes]/np.pi))
        for start in range(0, len(faces), 256):
            f = faces[start:start+256]
            dots = s.bmid[f] @ xyz.T
            nearest = dots.argmax(axis=1)
            distance = RADIUS_KM*np.arccos(np.clip(dots[np.arange(len(f)), nearest], -1, 1))
            accepted = distance <= reach[nearest]
            f, nearest = f[accepted], nearest[accepted]
            if not len(f):
                continue
            outward = s.bn[f]*np.where(s.bp[f] == p, 1., -1.)[:, None]
            other = np.where(s.bp[f] == p, s.bq[f], s.bp[f])
            relative = np.cross(s.omega[other]-s.omega[p], s.bmid[f])*RADIUS_KM
            normal = np.sum(relative*outward, axis=1)
            shear = relative-outward*normal[:, None]
            drive = outward*(np.maximum(normal, 0.)-.6*np.maximum(-normal, 0.))[:, None]+.25*shear
            target = nodes[nearest]
            np.add.at(lengths, target, s.bl[f])
            np.add.at(loads, target, drive*s.bl[f, None])
    loads /= np.maximum(lengths[:, None], 1e-20)
    loads *= float(s.config['rift_strength'])
    # Length normalization makes subdividing a raster boundary approximately
    # neutral. This remains a discrete mechanical approximation across meshes.
    return loads, np.minimum(lengths/800., 3.)


def _inherited_alignment(s, mesh):
    """Prefer reopening across an inherited rift, not lengthwise along it."""
    n, edges = len(mesh['xyz']), mesh['edges']
    nodes = mesh['parcel_node']
    valid = (nodes >= 0) & (s.rift_id > 0)
    # Strike is an AXIS: t and -t describe the same inherited fault. Average
    # its second moment rather than vectors which could cancel each other.
    weight = np.bincount(nodes[valid], weights=s.mass[valid], minlength=n)
    moment = np.zeros((n, 3, 3))
    for i in range(3):
        for j in range(3):
            moment[:, i, j] = np.bincount(nodes[valid],
                weights=s.mass[valid]*s.rift_tangent[valid, i]*s.rift_tangent[valid, j], minlength=n)/np.maximum(weight, 1e-20)
    a, b = edges.T
    cosine = np.sum(mesh['xyz'][a]*mesh['xyz'][b], axis=1)
    da = _unit(mesh['xyz'][b]-mesh['xyz'][a]*cosine[:, None])
    db = _unit(mesh['xyz'][a]-mesh['xyz'][b]*cosine[:, None])
    weak_a = (1-np.einsum('ni,nij,nj->n', da, moment[a], da))*(weight[a] > 0)
    weak_b = (1-np.einsum('ni,nij,nj->n', db, moment[b], db))*(weight[b] > 0)
    return np.clip(np.maximum(weak_a, weak_b), 0., 1.)


def _transition(s, row, phase, reason, *, event=None):
    row['phase'] = phase
    row['history'].append(dict(time_myr=float(s.t), phase=phase, reason=reason))
    p = np.flatnonzero(s.active & (s.plate_uid == row['plate_uid']))
    s._record(event or {'incipient': 'rift_initiating', 'active': 'rift_active',
                       'failed': 'rift_failed', 'broken_through': 'rift_breakthrough'}[phase],
              f"Continental rift {row['id']}: {reason}", plates=tuple(map(int, p)),
              xyz=np.sum(np.asarray(row['geometry_xyz']), axis=0) if row['geometry_xyz'] else None,
              details=dict(rift_system_id=row['id'], phase=phase, mechanism=MODEL))


def _track(s, mesh, keys, damage, strain, extension, strength):
    """Track connected damaged corridors by enduring material-link overlap."""
    edges, n = mesh['edges'], len(mesh['xyz'])
    visible = damage >= .12
    # A rift severs parallel cross-belt springs which need not share endpoints.
    # Join damaged links through their adjacent endpoint nodes (the mesh dual),
    # rather than requiring those cross-belt springs themselves to form a chain.
    touched = np.zeros(n, bool)
    touched[edges[visible].ravel()] = True
    labels = rift_mesh._components(n, edges[touched[edges[:, 0]] & touched[edges[:, 1]]], mesh['owners'])
    claimed = set()
    for label in np.unique(labels[edges[visible].ravel()] if np.any(visible) else []):
        members = np.flatnonzero(visible & (labels[edges[:, 0]] == label))
        if len(members) < 2:
            continue
        nodes = np.unique(edges[members])
        uid = int(s.plate_uid[mesh['owners'][nodes[0]]])
        keyset = {keys[i] for i in members}
        candidates = [(len(keyset.intersection(row['_bonds'])), row['id'], row)
                      for row in s.rift_systems if row['phase'] != 'broken_through'
                      and row['id'] not in claimed and row['plate_uid'] == uid]
        overlap, _, row = max(candidates, key=lambda x: (x[0], -x[1])) if candidates else (0, 0, None)
        new = overlap == 0
        if new:
            row = dict(id=s.next_rift_system_id, plate_uid=uid, phase='incipient',
                       started_myr=float(s.t), last_active_myr=float(s.t), extension_km=0.,
                       peak_damage=0., mean_strength=1., geometry_xyz=[], history=[], _bonds=set())
            s.next_rift_system_id += 1
            s.rift_systems.append(row)
        claimed.add(row['id'])
        row['_bonds'].update(keyset)
        # Connected link midpoints are anchors, never an ordered coastline.
        mids = _unit(mesh['xyz'][edges[members, 0]]+mesh['xyz'][edges[members, 1]])
        take = np.linspace(0, len(mids)-1, min(96, len(mids))).astype(int)
        row['geometry_xyz'] = mids[take].tolist()
        row['peak_damage'] = max(row['peak_damage'], float(damage[members].max()))
        row['extension_km'] = max(row['extension_km'], float(np.max(extension[members])))
        row['mean_strength'] = float(np.average(strength[nodes], weights=mesh['area'][nodes]))
        active = bool(np.max(strain[members]) > 1e-5)
        if new:
            _transition(s, row, 'incipient', 'local extension is developing a persistent damaged belt.')
        if active:
            row['last_active_myr'] = float(s.t)
            if row['phase'] == 'failed':
                _transition(s, row, 'active', 'renewed extension is reopening inherited weakness.', event='rift_reactivated')
            elif row['phase'] == 'incipient' and damage[members].max() >= .35:
                _transition(s, row, 'active', 'continued extension is thinning and weakening the belt.')
        elif s.t-row['last_active_myr'] >= 20 and row['phase'] != 'failed':
            _transition(s, row, 'failed', 'opening has been quiet for 20 Myr; the weakened basin remains.')
    for row in s.rift_systems:
        if row['id'] not in claimed and row['phase'] not in ('failed', 'broken_through') and s.t-row['last_active_myr'] >= 20:
            _transition(s, row, 'failed', 'the connected active belt has stalled or lost its loading.')


def _seed_scars(s, mesh, keys, damage):
    """Give developing basins the existing inversion-capable material memory."""
    edges, xyz = mesh['edges'], mesh['xyz']
    for row in s.rift_systems:
        if row['phase'] in ('failed', 'broken_through'):
            continue
        members = np.array([i for i, key in enumerate(keys) if key in row['_bonds'] and damage[i] >= .12], int)
        if not len(members):
            continue
        nodes = np.unique(edges[members])
        if 'source_rift_id' not in row:
            slots = np.flatnonzero(s.active & (s.plate_uid == row['plate_uid']))
            if not len(slots):
                continue
            row['source_rift_id'] = s._new_rift_record('progressive continental rift', tuple(map(int, slots)), xyz[nodes].sum(axis=0))
        tangent = np.zeros_like(xyz)
        # Use the strongest incident link's transverse axis. Equivalent strikes
        # with opposite signs must not cancel when a new scar is seeded.
        for i in members[np.argsort(damage[members], kind='stable')]:
            a, b = edges[i]
            cross = _unit(np.cross(xyz[a], xyz[b]))
            tangent[a] = cross
            tangent[b] = cross
        tangent = _unit(tangent)
        for prefix, mapping, kinds, points in (('', mesh['parcel_node'], s.kind, s.pos),
                                              ('trace_', mesh['trace_node'], s.trace_kind, s.trace_xyz)):
            ids = getattr(s, prefix+'rift_id')
            selected = (mapping >= 0) & np.isin(mapping, nodes) & (ids < 0) & (kinds != 3)
            ids[selected] = row['source_rift_id']
            getattr(s, prefix+'rift_birth_myr')[selected] = s.t
            direction = tangent[mapping[selected]]
            local = points[selected]
            getattr(s, prefix+'rift_tangent')[selected] = _unit(direction-local*np.sum(direction*local, axis=1)[:, None])


def _save_properties(s, mesh, damage, strength):
    mapping = mesh['parcel_node']
    valid = mapping >= 0
    ids, first = np.unique(s.parcel_patch[valid], return_index=True)
    nodes = mapping[valid][first]
    s.rift_properties = dict(patch=ids, damage=damage[nodes], strength=strength[nodes])
    # Record the scar while parcel-aligned damage is in hand. This is the once-per
    # step persistence point; material_fields() derives the same quantity but is a
    # read-only diagnostic and must not accumulate state.
    if enhanced_rifting.enabled(s):
        aligned = np.zeros(len(s.mass))
        aligned[valid] = damage[mapping[valid]]
        enhanced_rifting.accumulate_scar(s, aligned)


def material_fields(s):
    damage = np.zeros(len(s.mass))
    strength = relative_strength(s.structure['thickness_km'], s.structure['reference_thickness_km'],
                                 s.kind == 2, s.suture, s.structure['rift_heat_m'], damage)
    state = getattr(s, 'rift_properties', None)
    if state is not None and len(state['patch']):
        idx = np.searchsorted(state['patch'], s.parcel_patch)
        valid = idx < len(state['patch'])
        valid[valid] &= state['patch'][idx[valid]] == s.parcel_patch[valid]
        damage[valid] = state['damage'][idx[valid]]
        strength[valid] = state['strength'][idx[valid]]
    if enhanced_rifting.enabled(s) and hasattr(s,'parcel_rift_seed_weakness'):
        enhanced_rifting.ensure_fields(s)
        strength=enhanced_rifting.resistance(s,damage)
    return dict(rift_damage=damage, rift_strength_relative=strength)


def update(s, dt, boundary_extension, *, realized_extension=None):
    """Advance mechanics once, after ordinary columns and before accretion."""
    mesh = rift_material.refresh(s)
    # Commit policy 1 also needs the material each node is loaded on, taken
    # from the same refresh as the node mapping.
    image = material_image(s, mesh) if rift_commit_version(s) else None
    xyz, edges = mesh['xyz'], mesh['edges']
    keys = _edge_keys(mesh)
    _inherit_bonds(s, keys)
    old = np.array([s.rift_bonds.get(key, {}).get('damage', 0.) for key in keys])
    accumulated = np.array([s.rift_bonds.get(key, {}).get('strain', 0.) for key in keys])
    extension = np.array([s.rift_bonds.get(key, {}).get('extension_km', 0.) for key in keys])
    strength = relative_strength(mesh['thickness'], mesh['reference_thickness'], mesh['craton'],
                                 mesh['suture'], mesh['heat'], _node_max(old, edges, len(xyz)))
    if enhanced_rifting.enabled(s):
        strength=enhanced_rifting.node_mean(s,mesh,enhanced_rifting.material_viscosity(s))
    cap = max(120, int(math.ceil(10*math.sqrt(len(xyz)))))
    traction_report=dict(version=rift_traction.VERSION,enabled=False,applied=False)
    loading_source='realized native material motion' if enhanced_rifting.enabled(s) else 'boundary-motion proxy'
    if enhanced_rifting.enabled(s):
        a,b=edges.T
        length=RADIUS_KM*np.arctan2(np.linalg.norm(np.cross(xyz[a],xyz[b]),axis=1),np.sum(xyz[a]*xyz[b],axis=1))
        response=dict(edge_length_km=length,edge_extension=np.zeros(len(edges)),velocity=np.zeros_like(xyz),
            iterations=0,relative_residual=0.,converged=True,component_count=len(np.unique(mesh['owners'])))
        enhanced_rifting.realized_response(s,mesh,response,dt)
    else:
        motion_loads,motion_weights=boundary_loading(s,mesh)
        if rift_traction.enabled(s):
            traction_loads,traction_weights,traction_report=rift_traction.loading(s,mesh)
            fraction=float(s.config['rift_traction']['boundary_motion_fraction'])
            motion_weights=motion_weights*fraction
            combined=motion_weights+traction_weights
            loads=np.zeros_like(motion_loads)
            valid=combined>0.
            loads[valid]=(motion_loads[valid]*motion_weights[valid,None]
                          +traction_loads[valid]*traction_weights[valid,None])/combined[valid,None]
            weights=np.minimum(combined,3.)
            traction_report=dict(traction_report,applied=True,
                legacy_boundary_motion_fraction=fraction,
                traction_soft_weight_sum=float(traction_weights.sum()),
                boundary_soft_weight_sum=float(motion_weights.sum()))
            loading_source='conserved slab traction + reduced boundary-motion proxy'
        else:
            loads,weights=motion_loads,motion_weights
        response = rift_mechanics.solve_loading(xyz, edges, strength, loads, load_weights=weights,
                                                iterations=cap, tolerance=1e-5)
    s.rift_mechanics = dict(model=MODEL, mesh_nodes=len(xyz), mesh_edges=len(edges),
                            loading_source=loading_source, traction_loading=traction_report,
                            target_nodes=int(s.config['mechanics_nodes']),
                            solver_iterations=response['iterations'], iteration_limit=cap,
                            residual=response['relative_residual'], converged=response['converged'],
                            component_count=response['component_count'],
                            max_local_extension_km_myr=float(np.max(response['edge_extension'], initial=0.)),
                            **_rupture_calibration(s), **_commit_labels(s))
    a, b = edges.T
    length = response['edge_length_km']
    # Same unresolved strain partition as the existing crustal columns.
    tensile = np.maximum(response['edge_extension'], 0.)
    strain = np.minimum(PROXY_REALIZED_FRACTION*tensile/np.maximum(length, 1.), .08)*dt
    compression = PROXY_REALIZED_FRACTION*np.maximum(-response['edge_extension'], 0.)/np.maximum(length, 1.)*dt
    realized = enhanced_rifting.realized_response(s,mesh,response,dt)
    if realized is not None:
        strain,compression=realized
        tensile=np.maximum(response['edge_extension'],0.)
    # Do not turn an inadequately solved deformation field into a new plate.
    reliable = response['converged'] or response['relative_residual'] < .01
    if realized is not None:reliable=True
    if not reliable:
        strain[:] = 0.
    damage = evolve_damage(old, strain, .5*(strength[a]+strength[b]), dt,
                           inherited=_inherited_alignment(s, mesh), compression=compression)
    accumulated += strain
    s.rift_mechanics['max_bond_extension_strain'] = float(accumulated.max(initial=0.))
    extension += np.where(strain > 0, tensile*dt, 0.)
    if 'breakup_extension_km' in s.rift_mechanics:
        s.rift_mechanics['max_bond_realized_extension_km'] = float(
            s.rift_mechanics['realized_extension_fraction']*extension.max(initial=0.))
    for i, key in enumerate(keys):
        s.rift_bonds.setdefault(key, {}).update(damage=float(damage[i]), strain=float(accumulated[i]),
                                              extension_km=float(extension[i]), last_seen_myr=float(s.t))
    _track(s, mesh, keys, damage, strain, extension, strength)
    _seed_scars(s, mesh, keys, damage)
    # Fault-normal tension can coexist with equal perpendicular shortening.
    # Damage follows that local tension, but column volume responds only to
    # actual areal dilation: the trace of the resolved tangent strain tensor.
    # Taking the largest positive incident link would thin shear zones and
    # compressed material, creating arbitrary low patches at mesh-node edges.
    areal = rift_mechanics.areal_strain_rate(xyz, edges, response['edge_extension'])
    node_strain = np.minimum(PROXY_REALIZED_FRACTION*np.maximum(areal['rate'], 0.), .08)*dt
    if not reliable:
        node_strain[:] = 0.
    s.rift_mechanics.update(column_strain_measure='positive tangent strain trace (areal extension)',
                            areal_strain_supported_nodes=int(areal['supported'].sum()),
                            areal_strain_supported_area_fraction=float(
                                mesh['area'][areal['supported']].sum()/max(mesh['area'].sum(), 1e-20)),
                            max_areal_extension_per_myr=float(np.maximum(areal['rate'], 0.).max(initial=0.)))
    # Ordinary boundary extension has already been applied. Interior mechanics
    # may replace a smaller estimate, but must not charge it a second time.
    increments = []
    for index, (mapping, kinds, cells) in enumerate(((mesh['parcel_node'], s.kind, s.parcel_cell),
                                                    (mesh['trace_node'], s.trace_kind, s._indices(s.trace_xyz)))):
        total = np.zeros(len(mapping))
        selected = mapping >= 0
        total[selected] = node_strain[mapping[selected]]
        existing = (np.asarray(realized_extension[index]) if realized_extension is not None else
                    PROXY_REALIZED_FRACTION*np.where(kinds == 2, .35, 1.)*boundary_extension[cells]/400.*dt)
        increments.append(np.maximum(total-existing, 0.))
    if getattr(s, 'geometric_log_area', None) is None:
        structure_engine.extend_interior(s, *increments)
    else:
        # The native deforming surface owns actual areal strain. This reduced
        # response still drives damage, strength and possible rupture, but may
        # not thin an unmoved column in addition to the realized geometry.
        s.rift_mechanics['column_strain_measure'] = 'realized spherical material face area change'
    _save_properties(s, mesh, _node_max(damage, edges, len(xyz)), strength)
    # Plain state only: exact checkpoints never need to deserialize a solver.
    s.rift_pending = dict(mesh=mesh, damage=damage, strain=accumulated, extension_km=extension.copy(),
                          velocity=response['velocity'], edge_extension=response['edge_extension'], reliable=bool(reliable),
                          damage_strain_measure=s.rift_mechanics['damage_strain_measure'],
                          rupture_criterion_version=rupture_criterion_version(s))
    if image is not None:
        s.rift_pending.update(image)


class DevelopedBelt:
    """Transient smooth spherical side classifier from the developed material.

    Used only to continue a material cut into its surrounding ocean support.
    Actual continental patches transfer by their graph component, never by
    resampling this map classifier or by drawing another random fracture.
    """
    def __init__(self, xyz, positive, area):
        self.xyz, self.positive = xyz, positive
        self.center = _unit(np.sum(xyz*area[:, None], axis=0))
        ca = _unit(np.sum(xyz[~positive]*area[~positive, None], axis=0))
        cb = _unit(np.sum(xyz[positive]*area[positive, None], axis=0))
        self.normal = _unit(cb-ca-self.center*np.dot(cb-ca, self.center))

    def signed_distance(self, points):
        result = np.empty(len(points))
        for start in range(0, len(points), 1024):
            dots = np.asarray(points[start:start+1024])@self.xyz.T
            da = np.arccos(np.clip(np.max(dots[:, ~self.positive], axis=1), -1, 1))
            db = np.arccos(np.clip(np.max(dots[:, self.positive], axis=1), -1, 1))
            result[start:start+1024] = (da-db)*.5
        return result


def daughter_rotations(xyz, area, velocity, positive):
    """Best rigid responses of two daughters, conserving the common motion."""
    normal = np.eye(3)[None, :, :]-xyz[:, :, None]*xyz[:, None, :]
    matrices, moments, rotations = [], [], []
    for selected in (~positive, positive):
        matrix = np.sum(normal[selected]*area[selected, None, None], axis=0)
        moment = np.sum(np.cross(xyz[selected], velocity[selected])*area[selected, None], axis=0)/RADIUS_KM
        matrices.append(matrix)
        moments.append(moment)
        rotations.append(np.linalg.pinv(matrix, rcond=1e-10)@moment)
    common = np.linalg.pinv(matrices[0]+matrices[1], rcond=1e-10)@(matrices[0]@rotations[0]+matrices[1]@rotations[1])
    return np.asarray(rotations)-common


def _connected_fraction(s, mask, side):
    if hasattr(s, 'native_mesh'):
        from mesh_geometry import connected_components
        label_components = lambda selected: connected_components(np.asarray(selected, bool), s.native_mesh['edge_faces'])
    else:
        label_components = lambda selected: component_labels(selected, s.w, s.h)
    labels, count = label_components(mask)
    fragmented = 0.
    for parent in range(count):
        for selected in (mask & side & (labels == parent), mask & ~side & (labels == parent)):
            parts, number = label_components(selected)
            if number > 1:
                areas = np.bincount(parts[selected], weights=s.cell_area[selected], minlength=number)
                fragmented += float(areas.sum()-areas.max())
    return fragmented/max(float(s.cell_area[mask].sum()), 1.)


def partition_viability(s, mask, side):
    """Require resolved daughter interiors, independent of planet/parent size.

    Each *newly divided* connected parent component must retain a connected
    main piece on both sides, with at least three positive-area control cells
    and one cell whose complete edge-neighbour ring is also in that piece.
    That interior-ring test rejects sub-resolution ribbons without imposing
    a kilometre or parent-fraction floor. Refining the control mesh can resolve
    a previously inadmissible microcontinent. It does not resolve a material
    sliver that remains smaller than the control mesh.

    Existing separate islands do not count as fractures or as extra width for
    a newly cut piece. The small projection-fragment tolerance is measured
    against EACH side of EACH divided component, never the whole parent.
    Material connectivity, craton integrity and current opening remain the
    responsibility of the material cut/commit guards.
    """
    mask, side = np.asarray(mask, bool), np.asarray(side, bool)
    area = np.asarray(s.cell_area, float)
    result = dict(viable=False, model='resolved daughter interiors',
                  min_positive_cells=3, max_allowed_fragment_fraction=.025,
                  divided_parent_components=0, max_daughter_fragment_fraction=0.,
                  min_interior_cells=0, reason='invalid partition')
    if (mask.shape != area.shape or side.shape != mask.shape or mask.ndim != 1
            or np.any(side & ~mask) or not np.any(side) or not np.any(mask & ~side)
            or not np.isfinite(area[mask]).all() or np.any(area[mask] <= 0)):
        return result
    if hasattr(s, 'native_mesh'):
        from mesh_geometry import connected_components
        edges = np.asarray(s.native_mesh['edge_faces'], np.int32)
        label_components = lambda selected: connected_components(selected, edges)
    else:
        # Match component_labels, including same-pole antipodal continuation.
        cells = np.arange(len(mask)).reshape(s.h, s.w)
        edges = np.vstack((np.column_stack((cells.ravel(), np.roll(cells, -1, axis=1).ravel())),
                           np.column_stack((cells[:-1].ravel(), cells[1:].ravel())),
                           np.column_stack((cells[[0, -1]].ravel(),
                                            np.roll(cells[[0, -1]], s.w//2, axis=1).ravel()))))
        label_components = lambda selected: component_labels(selected, s.w, s.h)
    labels, count = label_components(mask)
    a, b = edges.T
    degree = np.bincount(edges.ravel(), minlength=len(mask))
    interior_counts = []
    for parent in range(count):
        component = mask & (labels == parent)
        positive = component & side
        negative = component & ~side
        if not np.any(positive) or not np.any(negative):
            continue
        result['divided_parent_components'] += 1
        for selected in (negative, positive):
            parts, number = label_components(selected)
            areas = np.bincount(parts[selected], weights=area[selected], minlength=number)
            largest = int(np.argmax(areas))
            fragmented = max(0., float((areas.sum()-areas[largest])/areas.sum()))
            result['max_daughter_fragment_fraction'] = max(result['max_daughter_fragment_fraction'], fragmented)
            if fragmented > result['max_allowed_fragment_fraction']:
                result['reason'] = 'daughter projection is disconnected'
                return result
            main = selected & (parts == largest)
            if np.count_nonzero(main) < result['min_positive_cells']:
                result['reason'] = 'daughter has too few resolved control cells'
                return result
            boundary = main[a] != main[b]
            interior = main & (degree > 0)
            interior[edges[boundary].ravel()] = False
            interior_counts.append(int(np.count_nonzero(interior)))
            if not interior_counts[-1]:
                result['reason'] = 'daughter has no resolved interior width'
                return result
    if not result['divided_parent_components']:
        result['reason'] = 'partition only reassigns existing disconnected islands'
        return result
    result.update(viable=True, min_interior_cells=min(interior_counts), reason='resolved connected daughters')
    return result


def _consolidate_projection(s, mask, distance, *, locked=None):
    """Remove only already-admissible projection fragments, without moving rock.

    The original partition must pass the existing per-daughter area and resolved
    interior checks. Both sides' fragments are selected from that same original
    partition, so one relabeling cannot make another eligible. A final check can
    refuse the complete proposal; callers never receive a partial cleanup.
    """
    mask = np.asarray(mask, bool)
    original = np.asarray(distance)
    report = dict(status='refused', reason='invalid projection distances', committed=False,
                  reassigned_cells=0, reassigned_area_km2=0., before=None, after=None)
    if (original.ndim != 1 or original.shape != mask.shape
            or not np.issubdtype(original.dtype, np.floating)
            or not np.isfinite(original[mask]).all()):
        return original.copy(), report
    side = original > 0
    before = partition_viability(s, mask, mask & side)
    report['before'] = before
    if not before['viable']:
        report['reason'] = 'original partition: '+before['reason']
        return original.copy(), report

    if hasattr(s, 'native_mesh'):
        from mesh_geometry import connected_components
        label_components = lambda selected: connected_components(selected, s.native_mesh['edge_faces'])
    else:
        label_components = lambda selected: component_labels(selected, s.w, s.h)
    labels, count = label_components(mask)
    flip = np.zeros(len(mask), bool)
    for parent in range(count):
        component = mask & (labels == parent)
        positive, negative = component & side, component & ~side
        # An existing island wholly on one side is not a projection fragment
        # of a newly divided parent, however small its area.
        if not np.any(positive) or not np.any(negative):
            continue
        for selected in (negative, positive):
            parts, number = label_components(selected)
            if number > 1:
                areas = np.bincount(parts[selected], weights=s.cell_area[selected], minlength=number)
                flip |= selected & (parts != int(np.argmax(areas)))
    if not np.any(flip):
        report.update(status='unchanged', reason='no projection fragments', after=deepcopy(before))
        return original.copy(), report

    # Projected cells containing the parent's material keep the assignment
    # established by the material cut, even when their projected area is tiny.
    # Neither native nor raster split commit can be relied on to restore a
    # material-majority sign after cleanup has overwritten it.
    if locked is not None:
        locked = np.asarray(locked, bool)
        if locked.shape != mask.shape or np.any(flip & locked):
            report['reason'] = 'cleanup would change a material-owned cell'
            return original.copy(), report

    cleaned = original.copy()
    magnitude_floor = max(1e-8, np.finfo(original.dtype).tiny)
    cleaned[flip] = np.maximum(np.abs(original[flip]), magnitude_floor)*np.where(side[flip], -1., 1.)
    after = partition_viability(s, mask, mask & (cleaned > 0))
    report['after'] = after
    if not after['viable']:
        report['reason'] = 'cleaned partition: '+after['reason']
        return original.copy(), report
    report.update(status='accepted', reason='small fragments removed from an already viable partition',
                  reassigned_cells=int(np.count_nonzero(flip)),
                  reassigned_area_km2=float(np.sum(s.cell_area[flip])))
    return cleaned, report


def commit(s):
    """Accept at most one connected, craton-safe continental breakthrough."""
    policy = rift_commit_version(s)
    pending = getattr(s, 'rift_pending', None)
    if (not pending or not pending['reliable']
            or (not np.any(~s.active) and not hasattr(s, 'native_mesh'))):
        return False
    mesh = pending['mesh']
    calibration = _rupture_calibration(s)
    # Fresh work cannot be reinterpreted after a mode change. Older pending
    # records lack this annotation; their recorded configuration selects the
    # measure, while source compatibility remains the checkpoint's authority.
    if pending.get('damage_strain_measure', calibration['damage_strain_measure']) != calibration['damage_strain_measure']:
        return False
    # Likewise a rupture criterion is never applied to work gathered under
    # another; records older than the annotation use the state's criterion.
    version = rupture_criterion_version(s)
    if pending.get('rupture_criterion_version', version) != version:
        return False
    if version == 0:
        # Damage alone cannot tear unextended or already compressed lithosphere.
        failed = ((pending['damage'] >= RUPTURE_DAMAGE)
                  & (pending['strain'] >= calibration['rupture_strain_threshold'])
                  & (pending['edge_extension'] > RUPTURE_OPENING_KM_MYR))
    else:
        # Realized kilometres of breakup extension, independent of link length.
        # Pending work without its extension waits for the next solved step.
        extension = pending.get('extension_km')
        if extension is None:
            return False
        failed = ((pending['damage'] >= RUPTURE_DAMAGE)
                  & (calibration['realized_extension_fraction']*np.asarray(extension, float)
                     >= BREAKUP_EXTENSION_KM)
                  & (pending['edge_extension'] > RUPTURE_OPENING_KM_MYR))
    if not np.any(failed):
        return False
    current = rift_material.refresh(s)
    velocity = unchanged = None
    # New juvenile parcels and coalescence can reorder arrays without changing
    # the continental mesh. Refresh their mappings. If accretion changed the
    # actual mechanical ownership, wait for its newly solved loading next step.
    arrays_differ = (not np.array_equal(current['bases'], mesh['bases'])
                     or not np.array_equal(current['owner_uids'], mesh['owner_uids'])
                     or not np.array_equal(current['edges'], mesh['edges']))
    if policy and 'material_patch_ids' in pending:
        # Commit policy 1: carry the solve over by persistent key. A current
        # link without a pending counterpart cannot fail this step, and a cut
        # whose loading unit gained or lost node keys, link keys or node
        # material (a partial transfer keeps the donor's key) is deferred to
        # its next solved loading.
        reassigned, moved_patches = reassigned_material(pending, material_image(s, current))
        changed = changed_owner_uids(mesh, current, reassigned)
        carried_links, new_links = int(len(current['edges'])), 0
        if arrays_differ:
            node, edge = remap_pending(mesh, current)
            carried = np.zeros(len(edge), bool)
            carried[edge >= 0] = failed[edge[edge >= 0]]
            failed = carried
            velocity = np.zeros((len(node), 3))
            velocity[node >= 0] = np.asarray(pending['velocity'])[node[node >= 0]]
            carried_links, new_links = int(np.count_nonzero(edge >= 0)), int(np.count_nonzero(edge < 0))
        if changed:
            unchanged = unchanged_loading(mesh, current,
                                          calibration['damage_strain_measure'] == REALIZED_STRAIN_MEASURE,
                                          reassigned)
        s.rift_mechanics.update(commit_mesh_changed=bool(changed), commit_changed_plate_uids=changed,
                                commit_carried_links=carried_links, commit_new_links=new_links,
                                commit_reassigned_patches=moved_patches,
                                commit_deferred_cuts=0, commit_deferred_plate_uids=[])
    elif arrays_differ:
        # Commit policy 0: any change anywhere refuses every cut. Policy-1
        # work gathered without its material image is treated the same way.
        return False
    mesh = current
    cuts = rift_mesh.coherent_cut(len(mesh['xyz']), mesh['edges'], failed,
                                  mesh['owners'], area=mesh['area'], min_nodes=3, min_fraction=0.)
    import backarc
    protected = backarc.protected_hosts(s)
    initial_ocean_uids = set(getattr(s, 'initial_ocean_plate_uids',
                                    (getattr(s, 'initial_ocean_plate_uid', None),)))
    for cut in cuts:
        p = cut['owner']
        if (not s.active[p] or int(s.plate_uid[p]) in protected or len(cut['components']) != 2
                or int(s.plate_uid[p]) in initial_ocean_uids):
            continue
        nodes = cut['parent_nodes']
        if unchanged is not None and not np.all(unchanged[nodes]):
            # Its loading was gathered on other material; next step re-solves it.
            s.rift_mechanics['commit_deferred_cuts'] += 1
            uids = s.rift_mechanics['commit_deferred_plate_uids']
            if int(s.plate_uid[p]) not in uids:
                uids.append(int(s.plate_uid[p]))
            continue
        material = mesh['parcel_node']
        child = min(cut['components'], key=lambda c: mesh['area'][c].sum())
        positive = np.isin(nodes, child)
        select = s.parcel_plate == p
        parcel_side = np.isin(material, child)
        if splits_protected_groups(s.parcel_craton[select], parcel_side[select]):
            continue
        child_mass = float(s.mass[select & parcel_side].sum())
        retained_mass = float(s.mass[select & ~parcel_side].sum())
        if not np.isfinite([child_mass, retained_mass]).all() or min(child_mass, retained_mass) <= 0:
            continue
        frac = child_mass/(child_mass+retained_mass)
        crack = DevelopedBelt(mesh['xyz'][nodes], positive, mesh['area'][nodes])
        distance = crack.signed_distance(s.xyz)
        mask = s.plate == p
        weight = np.bincount(s.parcel_cell[select], weights=s.mass[select], minlength=s.n)
        positive_weight = np.bincount(s.parcel_cell[select & parcel_side], weights=s.mass[select & parcel_side], minlength=s.n)
        material_cells = mask & (s.crust > 0) & (weight > 0)
        distance[material_cells] = np.maximum(np.abs(distance[material_cells]), 1e-8)*np.where(positive_weight[material_cells] > .5*weight[material_cells], 1., -1.)
        distance, cleanup = _consolidate_projection(s, mask, distance, locked=mask & (weight > 0))
        # This is the latest projection attempt in this mechanics update. It
        # is preserved by snapshot/checkpoint, including reasons for refusal.
        s.rift_mechanics['projection_cleanup'] = deepcopy(cleanup)
        if cleanup['status'] == 'refused':
            continue
        fragmented = _connected_fraction(s, mask, distance > 0)
        viability = cleanup['after']
        rotations = daughter_rotations(mesh['xyz'][nodes], mesh['area'][nodes], (pending['velocity'] if velocity is None else velocity)[nodes], positive)
        # A completed damage belt still needs present differential opening.
        relative = np.cross(rotations[1]-rotations[0], crack.center)
        if np.dot(relative, crack.normal)*RADIUS_KM <= .01:
            continue
        chosen = dict(crack=crack, fraction=frac, grid_distance=distance,
                      disconnected_fraction=fragmented, daughter_viability=viability, parcel_side=parcel_side,
                      trace_side=np.isin(mesh['trace_node'], child), rotations=rotations)
        keys = {(*sorted((int(mesh['bases'][a]), int(mesh['bases'][b]))), int(mesh['owner_uids'][a])) for a, b in cut['failed_edges']}
        rows = [row for row in s.rift_systems if row['plate_uid'] == int(s.plate_uid[p]) and row['phase'] != 'broken_through']
        row = max(rows, key=lambda row: len(keys.intersection(row['_bonds'])), default=None)
        if row is None:
            continue
        old_uid = s.next_plate_uid
        if s._split(p, chosen=chosen, progressive=True, loading=dict(rift_system_id=row['id'], model=MODEL)):
            cleanup['committed'] = True
            s.rift_mechanics['projection_cleanup'] = deepcopy(cleanup)
            row['new_plate_uid'] = old_uid
            row['projection_cleanup'] = deepcopy(cleanup)
            _transition(s, row, 'broken_through', 'a continuously damaged and extended belt separated two viable continental blocks.')
            return True
    return False


def snapshot(s):
    rows = [{key: deepcopy(value) for key, value in row.items() if not key.startswith('_')}
            for row in s.rift_systems]
    result = dict(rift_systems=rows, rift_mechanics=deepcopy(s.rift_mechanics))
    if rupture_criterion_version(s):
        result['rupture_criterion_version'] = RUPTURE_CRITERION_VERSION
    if rift_commit_version(s):
        result['rift_commit_version'] = RIFT_COMMIT_VERSION
    return result


_COMMIT_COUNTS = ('commit_carried_links', 'commit_new_links', 'commit_reassigned_patches', 'commit_deferred_cuts')
_COMMIT_UID_LISTS = ('commit_changed_plate_uids', 'commit_deferred_plate_uids')
_COMMIT_EVALUATION = ('commit_mesh_changed',)+_COMMIT_COUNTS+_COMMIT_UID_LISTS


def _validate_commit_frame(frame, mechanics):
    """A saved frame's commit policy and its diagnostics must agree."""
    version = frame.get('rift_commit_version', 0)
    if (isinstance(version, (bool, np.bool_)) or not isinstance(version, (int, np.integer))
            or version not in (0, RIFT_COMMIT_VERSION)):
        raise ValueError('Unsupported continental rift commit policy version.')
    labels = {key for key in (mechanics or {}) if key.startswith('commit_')}
    if not version:
        if labels:
            raise ValueError('Rift commit diagnostics require their commit policy version.')
        return
    if mechanics is None:
        raise ValueError('A rift commit policy frame requires its rift mechanics labels.')
    if (mechanics.get('commit_policy_version') != version or mechanics.get('commit_policy') != RIFT_COMMIT_POLICY
            or mechanics.get('commit_loading_unit') not in (REALIZED_LOADING_UNIT, PROXY_LOADING_UNIT)):
        raise ValueError('Saved rift commit labels disagree with the frame commit policy.')
    unit = (REALIZED_LOADING_UNIT if mechanics.get('damage_strain_measure') == REALIZED_STRAIN_MEASURE
            else PROXY_LOADING_UNIT)
    if mechanics['commit_loading_unit'] != unit:
        raise ValueError('Saved rift commit loading unit disagrees with the mechanics mode.')
    if labels - {'commit_policy_version', 'commit_policy', 'commit_loading_unit', *_COMMIT_EVALUATION}:
        raise ValueError('Saved rift mechanics carry an unknown commit diagnostic.')
    evaluated = [key in mechanics for key in _COMMIT_EVALUATION]
    if not any(evaluated):
        return
    if not all(evaluated):
        raise ValueError('Saved rift commit diagnostics are incomplete.')
    if not isinstance(mechanics['commit_mesh_changed'], (bool, np.bool_)):
        raise ValueError('Saved rift commit mesh change must be a boolean.')
    for key in _COMMIT_COUNTS:
        value = mechanics[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError('Saved rift commit counts must be nonnegative integers.')
    for key in _COMMIT_UID_LISTS:
        value = mechanics[key]
        if (not isinstance(value, list) or len(set(value)) != len(value)
                or any(isinstance(uid, (bool, np.bool_)) or not isinstance(uid, (int, np.integer)) for uid in value)):
            raise ValueError('Saved rift commit plate UIDs must be a list of distinct integers.')
    if not mechanics['commit_mesh_changed'] and (mechanics['commit_new_links'] or mechanics['commit_changed_plate_uids']
                                                 or mechanics['commit_reassigned_patches']
                                                 or mechanics['commit_deferred_cuts']):
        raise ValueError('An unchanged rift mesh cannot record changed links, plates, material or deferrals.')
    if mechanics['commit_mesh_changed'] and not mechanics['commit_changed_plate_uids']:
        raise ValueError('A changed rift mesh must record the plates whose keys or material changed.')
    if not set(mechanics['commit_deferred_plate_uids']) <= set(mechanics['commit_changed_plate_uids']):
        raise ValueError('A deferred rift cut must belong to a changed plate.')
    if bool(mechanics['commit_deferred_cuts']) != bool(mechanics['commit_deferred_plate_uids']):
        raise ValueError('Saved rift commit deferrals and their plates disagree.')


def validate_frame(frame):
    """A saved frame's rupture criterion, commit policy and labels must agree."""
    mechanics = frame.get('rift_mechanics')
    if mechanics is not None and not isinstance(mechanics, dict):
        raise ValueError('Saved rift mechanics must be a mapping.')
    _validate_commit_frame(frame, mechanics)
    version = frame.get('rupture_criterion_version', 0)
    if (isinstance(version, (bool, np.bool_)) or not isinstance(version, (int, np.integer))
            or version not in (0, RUPTURE_CRITERION_VERSION)):
        raise ValueError('Unsupported continental rupture criterion version.')
    mechanics = frame.get('rift_mechanics')
    if mechanics is None:
        if version:
            raise ValueError('A rupture criterion frame requires its rift mechanics labels.')
        return version
    if not isinstance(mechanics, dict):
        raise ValueError('Saved rift mechanics must be a mapping.')
    if not version:
        if 'breakup_extension_km' in mechanics or 'rupture_criterion_version' in mechanics:
            raise ValueError('Breakup-extension labels require their rupture criterion version.')
        return version
    extension = mechanics.get('breakup_extension_km')
    fraction = mechanics.get('realized_extension_fraction')
    if (isinstance(extension, (bool, np.bool_)) or not isinstance(extension, (int, float, np.integer, np.floating))
            or not np.isfinite(extension) or extension <= 0):
        raise ValueError('Saved breakup extension must be a positive finite length in km.')
    if (isinstance(fraction, (bool, np.bool_)) or not isinstance(fraction, (int, float, np.integer, np.floating))
            or float(fraction) not in (1., PROXY_REALIZED_FRACTION)):
        raise ValueError('Saved realized-extension fraction must be 1 or the proxy partition.')
    if 'rupture_strain_threshold' in mechanics:
        raise ValueError('A breakup-extension frame cannot also record a strain threshold.')
    if mechanics.get('rupture_criterion') != RUPTURE_CRITERION:
        raise ValueError('A breakup-extension frame must name its rupture criterion.')
    # The fraction and the extension measure must belong to the recorded mode.
    mode = _CRITERION_ONE_MODES.get(mechanics.get('damage_strain_measure'))
    if mode is None:
        raise ValueError('A breakup-extension frame must record a known damage strain measure.')
    if float(fraction) != mode[0] or mechanics.get('rupture_extension_measure') != mode[1]:
        raise ValueError('Saved realized-extension labels disagree with the mechanics mode.')
    if mechanics.get('rupture_criterion_version', version) != version:
        raise ValueError('Saved rift mechanics disagree with the frame rupture criterion.')
    return version
