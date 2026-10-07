"""A brief Windows destination lock must not destroy an atomic history update."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from server import write_json


class AtomicJsonTests(unittest.TestCase):
    def test_transient_replace_lock_retries_same_complete_temp_and_commits_once(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'manifest.json'
            target.write_text('{"epoch":494}', encoding='utf-8')
            previous = target.read_bytes()
            temporary = target.with_suffix('.json.tmp')
            replace = Path.replace
            write = Path.write_bytes
            attempts = []

            def temporarily_locked(source, destination):
                attempts.append((source, destination))
                self.assertEqual(target.read_bytes(), previous)
                self.assertEqual(json.loads(source.read_bytes()), {'epoch': 496, 'state': 'running'})
                if len(attempts) <= 3:
                    raise PermissionError(13, 'A reader temporarily holds the destination')
                return replace(source, destination)

            with patch.object(Path, 'replace', autospec=True, side_effect=temporarily_locked), \
                    patch.object(Path, 'write_bytes', autospec=True, side_effect=write) as writes, \
                    patch('server.time.sleep') as sleep:
                write_json(target, {'epoch': 496, 'state': 'running'})
            self.assertEqual(writes.call_count, 1)
            self.assertEqual(attempts, [(temporary, target)]*4)
            self.assertEqual(sleep.call_count, 3)
            self.assertEqual(json.loads(target.read_bytes()), {'epoch': 496, 'state': 'running'})
            self.assertFalse(temporary.exists())

    def test_permanent_permission_failure_is_bounded_propagates_and_keeps_old_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'manifest.json'
            target.write_text('{"epoch":494}', encoding='utf-8')
            previous = target.read_bytes()
            locked = PermissionError(13, 'Destination remains locked')
            with patch.object(Path, 'replace', autospec=True, side_effect=locked) as replace, \
                    patch('server.time.sleep') as sleep:
                with self.assertRaises(PermissionError) as raised:
                    write_json(target, {'epoch': 496})
            self.assertIs(raised.exception, locked)
            self.assertGreater(replace.call_count, 1)
            delays = [args.args[0] for args in sleep.call_args_list]
            self.assertEqual(len(delays), replace.call_count-1)
            self.assertEqual(delays, sorted(delays))
            self.assertGreaterEqual(min(delays), .02)
            self.assertLessEqual(sum(delays), 2.)
            self.assertEqual(target.read_bytes(), previous)
            self.assertEqual(json.loads(target.with_suffix('.json.tmp').read_bytes()), {'epoch': 496})

    def test_unrelated_replace_error_is_not_retried_or_swallowed(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'manifest.json'
            target.write_text('{"epoch":494}', encoding='utf-8')
            with patch.object(Path, 'replace', autospec=True, side_effect=FileNotFoundError('Source vanished')) as replace, \
                    patch('server.time.sleep') as sleep:
                with self.assertRaises(FileNotFoundError):
                    write_json(target, {'epoch': 496})
            self.assertEqual(replace.call_count, 1)
            sleep.assert_not_called()
            self.assertEqual(json.loads(target.read_bytes()), {'epoch': 494})


if __name__ == '__main__':
    unittest.main()
