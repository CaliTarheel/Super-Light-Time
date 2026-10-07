"""Small worker integration checks; no simulation is constructed or advanced."""
from decimal import getcontext, localcontext
import os
import unittest
from unittest.mock import patch
import numpy as np
import budget
import burial_depth
from parallel_runtime import RuntimePool, share_inputs
from tests.test_burial_parallel_partition import _stack

EVIDENCE = []

def worker_context_probe(job):
    shared, caller, modes, items, expected = job
    with localcontext() as previous:
        previous.prec = 37
        rows = burial_depth._partition_batch((shared, caller, modes, items))
        actual = budget.snapshot()
        budget.record('burial_worker_equivalence_probe', partitions=len(rows))
        return dict(pid=os.getpid(), restored=getcontext().prec == 37,
                    inherited=all(actual[k] == expected[k] for k in ('started','deadline','physical_deadline','force_refinement_deadline')),
                    partitions=len(rows), successful=all(row[1] is not None for row in rows))

class WorkerIntegration(unittest.TestCase):
    def test_worker_context_is_restored_and_absolute_budget_inherited(self):
        triangles, upper, jobs = _stack(level=0)
        context = getcontext().copy(); context.prec = 61
        with budget.step_budget(1., seconds_per_myr=100, reserve_seconds=8):
            expected = budget.snapshot()
            with RuntimePool(workers=2,max_workers=2) as pool:
                with share_inputs(dict(triangles=triangles,upper=upper)) as shared:
                    rows = pool.map(worker_context_probe, [(shared,context,np.geterr(),[(i,)+tuple(jobs[i])],expected) for i in range(4)])
                status = pool.status()
            self.assertTrue(status['pool_started'])
            self.assertEqual(status['completed_batches'], 1)
            self.assertTrue(all(r['pid'] != os.getpid() and r['restored'] and r['inherited'] and r['successful'] for r in rows))
            self.assertTrue(pool.status()['pool_closed'])
            self.assertTrue(any(e.get('stage')=='burial_worker_equivalence_probe' for e in budget.evidence()['events']))
            EVIDENCE.append(dict(worker_rows=rows,status=status,closed_status=pool.status()))

    def test_budget_exhaustion_is_not_swallowed_as_serial_fallback(self):
        triangles, upper, jobs = _stack(level=0)
        with share_inputs(dict(triangles=triangles,upper=upper)) as shared:
            with patch.object(burial_depth,'_partition_face_uncached',side_effect=budget.BudgetExhausted('synthetic exhausted budget')):
                with self.assertRaises(budget.BudgetExhausted):
                    burial_depth._partition_batch((shared,getcontext().copy(),np.geterr(),[(0,)+tuple(jobs[0])]))
