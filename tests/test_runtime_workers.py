"""Event-controlled checks of live dispatch changes and batch ownership."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
from threading import Event, Lock, Thread
import time
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'runtime_workers.py'
spec = importlib.util.spec_from_file_location('candidate_runtime_workers', SOURCE)
workers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workers)


def eventually(predicate):
    deadline = time.monotonic() + 5
    tick = Event()
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError('Expected state was not observed.')
        tick.wait(.005)


class RecordingExecutor:
    def __init__(self, executor):
        self.executor = executor
        self.futures = []
        self.lock = Lock()

    def submit(self, fn, item):
        future = self.executor.submit(fn, item)
        with self.lock:
            self.futures.append(future)
        return future

    def snapshot(self):
        with self.lock:
            return list(self.futures)


class Batch:
    def __init__(self, executor, fn, items, policy):
        self.done = Event()
        self.result = self.error = None

        def run():
            try:
                self.result = workers.bounded_ordered_map(executor, fn, items, policy)
            except BaseException as error:
                self.error = error
            finally:
                self.done.set()

        self.thread = Thread(target=run)
        self.thread.start()

    def join(self):
        if not self.done.wait(5):
            raise AssertionError('Batch did not terminate.')
        self.thread.join()


class WorkerPolicyTests(unittest.TestCase):
    def test_default_and_invalid_limits(self):
        policy = workers.WorkerPolicy()
        self.assertEqual(policy.status(), dict(max_workers=6, requested_workers=2,
                                              in_flight=0, revision=0))
        for value in [True, False, 0, -1, 7, 2.0, '2', None]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                policy.set_limit(value)
        self.assertEqual(policy.status()['revision'], 0)
        self.assertEqual(policy.set_limit(4)['revision'], 1)
        self.assertEqual(policy.set_limit(4)['revision'], 1)
        for kwargs in [dict(max_workers=0), dict(max_workers=True),
                       dict(initial_workers=0), dict(max_workers=1, initial_workers=2)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                workers.WorkerPolicy(**kwargs)

    def test_live_six_to_two_throttle(self):
        policy = workers.WorkerPolicy(initial_workers=6)
        entered, gates = [Event() for _ in range(8)], [Event() for _ in range(8)]

        def job(i):
            entered[i].set()
            self.assertTrue(gates[i].wait(5))
            return i * 10

        with ThreadPoolExecutor(max_workers=6) as pool:
            executor = RecordingExecutor(pool)
            batch = Batch(executor, job, range(8), policy)
            try:
                for event in entered[:6]:
                    self.assertTrue(event.wait(5))
                eventually(lambda: policy.status()['in_flight'] == 6)
                self.assertEqual(policy.set_limit(2)['in_flight'], 6)
                for gate in gates[:4]:
                    gate.set()
                eventually(lambda: policy.status()['in_flight'] == 2)
                self.assertEqual(len(executor.snapshot()), 6)
                self.assertFalse(entered[6].is_set())
                gates[4].set()
                self.assertTrue(entered[6].wait(5))
                eventually(lambda: len(executor.snapshot()) == 7)
                self.assertEqual(len(executor.snapshot()), 7)
                self.assertLessEqual(policy.status()['in_flight'], 2)
                self.assertFalse(entered[7].is_set())
                gates[5].set()
                self.assertTrue(entered[7].wait(5))
                self.assertLessEqual(policy.status()['in_flight'], 2)
            finally:
                for gate in gates:
                    gate.set()
                batch.join()
            self.assertIsNone(batch.error)
            self.assertEqual(batch.result, [i * 10 for i in range(8)])
            self.assertEqual(policy.status()['in_flight'], 0)

    def test_live_two_to_four_increase_and_input_order(self):
        policy = workers.WorkerPolicy()
        entered, gates = [Event() for _ in range(4)], [Event() for _ in range(4)]

        def job(i):
            entered[i].set()
            self.assertTrue(gates[i].wait(5))
            return str(i)

        with ThreadPoolExecutor(max_workers=4) as pool:
            executor = RecordingExecutor(pool)
            batch = Batch(executor, job, range(4), policy)
            try:
                self.assertTrue(entered[0].wait(5))
                self.assertTrue(entered[1].wait(5))
                eventually(lambda: len(executor.snapshot()) == 2)
                self.assertEqual(len(executor.snapshot()), 2)
                policy.set_limit(4)
                self.assertTrue(entered[2].wait(5))
                self.assertTrue(entered[3].wait(5))
                eventually(lambda: len(executor.snapshot()) == 4)
                for i in [3, 2, 1, 0]:
                    gates[i].set()
                    eventually(lambda i=i: executor.snapshot()[i].done())
            finally:
                for gate in gates:
                    gate.set()
                batch.join()
            self.assertIsNone(batch.error)
            self.assertEqual(batch.result, ['0', '1', '2', '3'])

    def test_single_batch_ownership_and_executor_is_not_closed(self):
        policy = workers.WorkerPolicy()
        entered, release = Event(), Event()

        def job(i):
            entered.set()
            self.assertTrue(release.wait(5))
            return i

        with ThreadPoolExecutor(max_workers=2) as pool:
            batch = Batch(pool, job, [1], policy)
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaisesRegex(RuntimeError, 'owning batch'):
                    workers.bounded_ordered_map(pool, lambda x: x, [2], policy)
            finally:
                release.set()
                batch.join()
            self.assertEqual(workers.bounded_ordered_map(pool, lambda x: x + 1, [4], policy), [5])
            self.assertEqual(workers.bounded_ordered_map(pool, lambda x: x, [], policy), [])
            self.assertEqual(pool.submit(lambda: 9).result(), 9)

    def test_worker_error_cancels_queue_and_joins_running_work(self):
        policy = workers.WorkerPolicy(initial_workers=6)
        fail, release, second_started = Event(), Event(), Event()

        def job(i):
            if i == 0:
                self.assertTrue(fail.wait(5))
                raise ValueError('worker exploded')
            if i == 1:
                second_started.set()
            self.assertTrue(release.wait(5))
            return i

        with ThreadPoolExecutor(max_workers=2) as pool:
            executor = RecordingExecutor(pool)
            batch = Batch(executor, job, range(8), policy)
            try:
                self.assertTrue(second_started.wait(5))
                eventually(lambda: len(executor.snapshot()) == 6)
                fail.set()
                eventually(lambda: any(f.cancelled() for f in executor.snapshot()))
                self.assertFalse(batch.done.is_set())  # Running job still owns its input.
                self.assertEqual(len(executor.snapshot()), 6)
            finally:
                fail.set()
                release.set()
                batch.join()
            self.assertIsInstance(batch.error, ValueError)
            self.assertEqual(str(batch.error), 'worker exploded')
            self.assertTrue(all(f.done() for f in executor.snapshot()))
            self.assertEqual(policy.status()['in_flight'], 0)
            self.assertEqual(workers.bounded_ordered_map(pool, lambda x: x, [3], policy), [3])

    def test_iterator_and_submit_errors_release_ownership(self):
        policy = workers.WorkerPolicy()

        def broken():
            yield 1
            raise LookupError('input failed')

        with ThreadPoolExecutor(max_workers=2) as pool:
            with self.assertRaisesRegex(LookupError, 'input failed'):
                workers.bounded_ordered_map(pool, lambda x: x, broken(), policy)
            self.assertEqual(policy.status()['in_flight'], 0)
            self.assertEqual(workers.bounded_ordered_map(pool, lambda x: x, [5], policy), [5])
        with self.assertRaises(RuntimeError):
            workers.bounded_ordered_map(pool, lambda x: x, [7], policy)
        self.assertEqual(policy.status()['in_flight'], 0)
        with ThreadPoolExecutor(max_workers=1) as next_pool:
            self.assertEqual(workers.bounded_ordered_map(next_pool, lambda x: x, [8], policy), [8])


if __name__ == '__main__':
    unittest.main()
