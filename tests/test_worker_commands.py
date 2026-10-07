import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_workers import WorkerPolicy
from worker_commands import WorkerCommands


class CommandTests(unittest.TestCase):
    def test_explicit_count_and_modes(self):
        commands = WorkerCommands(WorkerPolicy(max_workers=6, initial_workers=2))
        self.assertEqual(commands.apply({'workers': 3})['requested_workers'], 3)
        self.assertEqual(commands.apply({'mode': 'maximum'})['requested_workers'], 6)
        self.assertEqual(commands.apply({'mode': 'quiet'})['requested_workers'], 2)

    def test_invalid_command_preserves_limit(self):
        commands = WorkerCommands(WorkerPolicy(max_workers=6, initial_workers=2))
        for body in [{}, [], {'mode':'maximum','workers':3}, {'workers':True},
                     {'workers':0}, {'workers':7}, {'workers':2.5}, {'mode':[]},
                     {'mode':'unknown'}, {'physics_dt':1}]:
            with self.subTest(body=body):
                with self.assertRaises(ValueError):commands.apply(body)
                self.assertEqual(commands.status()['requested_workers'],2)

    def test_single_core_host_quiet_mode(self):
        commands = WorkerCommands(WorkerPolicy(max_workers=1, initial_workers=1))
        self.assertEqual(commands.apply({'mode':'quiet'})['requested_workers'],1)


if __name__ == '__main__':unittest.main()
