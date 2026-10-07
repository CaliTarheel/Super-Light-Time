"""Continue surface ownership through overlapping buoyant crust.

Area is deposited independently of this rule. Two resolved material sheets can
overlap during shortening; small sampling differences in their deposited areas
do not establish a new geological boundary at every cell. The transported
surface owner remains exposed while its own finite footprint independently
meets the same crust-visibility threshold. A vanished/thin sheet yields to the
dominant arriving deposit. This neither transfers nor deletes hidden material.
"""
import numpy as np


def resolve(transported_owner, dominant_owner, land, owner_occupancy):
    prior = np.asarray(transported_owner)
    result = np.asarray(dominant_owner).copy()
    land = np.asarray(land, bool)
    supported = np.zeros(len(prior), bool)
    for owner, cells in owner_occupancy.items():
        cells = np.asarray(cells)
        supported[cells] |= prior[cells] == owner
    result[supported & land] = prior[supported & land]
    result[~land] = prior[~land]
    return result
