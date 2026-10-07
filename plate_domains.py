"""Persistent names for visible, connected domains of moving tectonic plates.

A visible domain is an observation of the raster, not an additional force-bearing
plate. Occlusion can temporarily separate visible regions of one coherent crustal
body. This tracker records that distinction without tearing material or creating
new motion, and without reusing a retired identity.
"""
from dataclasses import dataclass

import numpy as np

from plate_topology import domain_components


@dataclass
class DomainState:
    domain: np.ndarray
    domains: list
    created: list
    retired_uids: list


class PersistentDomainTracker:
    """Track connected-domain identities by advected, area-weighted overlap."""

    def __init__(self, width, height):
        self.width, self.height = int(width), int(height)
        self._previous = None
        self._records = {}
        self._seen_plate_uids = set()
        self._next_uid = 1

    def update(self, plate, cell_area, plate_uids, names, time_myr,
               previous_cells=None):
        """Return current identities without changing any tectonic input.

        ``previous_cells`` maps each destination cell to its inverse-advected
        source index in the previous raster. -1 means no prior correspondence.
        Only domains with the same persistent kinematic plate UID can match.
        The strongest available overlap wins, one old ID per new component;
        ties prefer the older domain ID, then the lower component index.

        The first domain of each kinematic plate uses its plate name. Additional
        domains use unique Fragment names. A split child's source_domain_uid
        identifies its previous visible parent. When domains reunite, the
        strongest overlap retains its identity and the others retire. A domain
        that disappears and later reappears receives a new identity.
        """
        plate = np.asarray(plate).reshape(-1)
        cell_area = np.asarray(cell_area, dtype=float).reshape(-1)
        n = self.width*self.height
        if len(plate) != n or len(cell_area) != n:
            raise ValueError('plate and cell_area must match the tracker dimensions')
        if not np.all(np.isfinite(cell_area)) or np.any(cell_area <= 0):
            raise ValueError('cell_area must contain finite positive areas')
        labels, owners = domain_components(plate, self.width, self.height)
        parent_uids = np.asarray(plate_uids, np.int64)[owners]
        areas = np.bincount(labels, weights=cell_area, minlength=len(owners))
        assigned = np.zeros(len(owners), np.int32)
        best_source = np.zeros(len(owners), np.int32)
        old_current = set() if self._previous is None else set(np.unique(self._previous).tolist())
        if self._previous is not None:
            if previous_cells is None:
                previous = self._previous
            else:
                source = np.asarray(previous_cells, np.int64).reshape(-1)
                if len(source) != n or np.any(source < -1) or np.any(source >= n):
                    raise ValueError('previous_cells must contain source indices or -1')
                previous = np.zeros(n, np.int32)
                valid = source >= 0
                previous[valid] = self._previous[source[valid]]
            uid_owner = np.zeros(self._next_uid, np.int64)
            for uid, record in self._records.items():
                uid_owner[uid] = record['parent_plate_uid']
            valid = (previous > 0) & (uid_owner[previous] == parent_uids[labels])
            if np.any(valid):
                pairs, inverse = np.unique(np.column_stack((labels[valid], previous[valid])),
                                           axis=0, return_inverse=True)
                weights = np.bincount(inverse, weights=cell_area[valid], minlength=len(pairs))
                order = np.lexsort((pairs[:, 0], pairs[:, 1], -weights))
                used = set()
                for index in order:
                    component, uid = map(int, pairs[index])
                    if best_source[component] == 0:
                        best_source[component] = uid
                    if assigned[component] == 0 and uid not in used:
                        assigned[component] = uid
                        used.add(uid)
        created = []
        # Largest first makes the principal region the named plate's primary
        # domain when a new plate is initially visible in several components.
        missing = np.flatnonzero(assigned == 0)
        order = sorted(missing.tolist(), key=lambda i: (-areas[i], int(parent_uids[i]), i))
        for component in order:
            uid = self._next_uid
            if uid > np.iinfo(np.int32).max:
                raise OverflowError('visible domain identity range exhausted')
            self._next_uid += 1
            p, parent_uid = int(owners[component]), int(parent_uids[component])
            primary = parent_uid not in self._seen_plate_uids
            self._seen_plate_uids.add(parent_uid)
            record = dict(uid=uid, name=str(names[p]) if primary else f'Fragment {uid:03d}',
                          plate_id=p, parent_plate_uid=parent_uid,
                          area_km2=float(areas[component]), created_myr=float(time_myr),
                          source_domain_uid=int(best_source[component]) or None,
                          primary=bool(primary))
            self._records[uid] = record
            assigned[component] = uid
            created.append(dict(record))
        current = []
        for component, uid in enumerate(assigned):
            record = self._records[int(uid)]
            record['plate_id'] = int(owners[component])
            record['area_km2'] = float(areas[component])
            current.append(dict(record))
        current.sort(key=lambda row: row['uid'])
        domain = assigned[labels]
        self._previous = domain.copy()
        retired = sorted(old_current-set(assigned.tolist()))
        return DomainState(domain, current, created, retired)
