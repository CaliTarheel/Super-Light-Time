"""Conservation identities on accumulators need an accumulating allowance.

A quantity that only ever grows carries rounding that grows with the RUN, not
with its own value. Run SEP21T stopped at 194 Myr when the dense-crust
identity's residual - dead linear at about one ULP per step, 7.69e-14 at
16 Myr and 9.97e-13 at 192 - crossed a fixed 1e-12 floor on 2 cells of 36,955,
at 1.000834e-12. No jump, no leak, and nothing physically wrong: a micron of
disagreement on a column kilometres thick.

Where a cell's own value is small the relative term contributes nothing, so
the absolute floor is the only allowance such a cell has. These tests pin that
both identities tolerate a run's worth of accumulation while still refusing a
residual big enough to mean something.
"""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tests'), str(ROOT)]
import crust_inventory
import dense_crust

# What the live run actually reached, and what a long run would reach.
OBSERVED_AT_192_MYR = 1.000834e-12
PER_STEP = 6e-15


class AccumulatorToleranceTests(unittest.TestCase):
    def dense_state(self, converted, dense, eroded, returned):
        """A one-cell retained-phase state that satisfies every other guard."""
        one = np.ones(1)
        state = {
            'thickness_km': one*30., 'area_factor': one.copy(),
            dense_crust.DENSE: one*dense, dense_crust.CONVERTED: one*converted,
            dense_crust.ERODED: one*eroded, dense_crust.RETURNED: one*returned,
            dense_crust.HEAT: np.zeros(1), dense_crust.HEAT_BASELINE: np.zeros(1),
            dense_crust.HEAT_ADDED: np.zeros(1), dense_crust.HEAT_BATH: np.zeros(1),
            dense_crust.HEAT_ERODED: np.zeros(1), dense_crust.HEAT_RETURNED: np.zeros(1),
        }
        inventory = crust_inventory
        # Small enough to sit inside the retained column the guards check.
        state[inventory.BASELINE] = one.copy()
        state[inventory.ADDED] = np.zeros(1)
        state[inventory.ERODED] = np.zeros(1)
        state[inventory.RETURNED] = np.zeros(1)
        state[inventory.REMAINING] = one.copy()
        return state

    def test_the_residual_that_stopped_the_run_is_now_tolerated(self):
        # Exactly the failing cell: a small value carrying a run's rounding.
        value = 1.257616e-05
        state = self.dense_state(value+OBSERVED_AT_192_MYR, value, 0., 0.)
        dense_crust.validate(state)

    def test_a_long_run_of_further_accumulation_is_still_tolerated(self):
        # Another 100,000 steps at the measured rate.
        value = 1.257616e-05
        state = self.dense_state(value+PER_STEP*100000, value, 0., 0.)
        dense_crust.validate(state)

    def test_a_residual_big_enough_to_matter_is_still_refused(self):
        # A micron is noise; a metre is a leak.
        value = 1.257616e-05
        state = self.dense_state(value+1e-3, value, 0., 0.)
        with self.assertRaises(ValueError):
            dense_crust.validate(state)

    def test_the_dense_identity_keeps_its_relative_term(self):
        # Large values must not be given a free pass proportional to nothing.
        state = self.dense_state(1000.+1e-3, 1000., 0., 0.)
        with self.assertRaises(ValueError):
            dense_crust.validate(state)

    def test_the_crust_inventory_closure_tolerates_the_same_accumulation(self):
        one = np.ones(1)
        state = {'thickness_km': one*30., 'area_factor': one.copy(),
                 crust_inventory.BASELINE: one*(1.257616e-05+OBSERVED_AT_192_MYR),
                 crust_inventory.ADDED: np.zeros(1),
                 crust_inventory.ERODED: np.zeros(1), crust_inventory.RETURNED: np.zeros(1),
                 crust_inventory.REMAINING: one*1.257616e-05}
        crust_inventory.validate(state)
        state[crust_inventory.BASELINE] = one*(1.257616e-05+1e-3)
        with self.assertRaises(ValueError):
            crust_inventory.validate(state)


if __name__ == '__main__':
    unittest.main()
