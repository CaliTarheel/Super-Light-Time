"""Exercise recipient setup safeguards without installing into a shared runtime."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(os.name == "nt" and sys.version_info[:2] == (3, 12), "Windows Python 3.12 setup")
class PortableSetupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="deep time setup ")
        self.root = Path(self.temporary.name)
        source = Path(__file__).resolve().parents[1]
        for name in ("Setup.ps1", "requirements-portable.txt"):
            shutil.copy2(source / name, self.root / name)

    def tearDown(self):
        self.temporary.cleanup()

    def run_setup(self, *args):
        return subprocess.run(
            ["powershell.exe", "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", str(self.root / "Setup.ps1"), "-PythonPath", sys.executable, *args],
            capture_output=True, text=True, timeout=30,
        )

    def test_check_only_finds_python_without_creating_environment(self):
        result = self.run_setup("-CheckOnly")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("CheckOnly did not create", result.stdout)
        self.assertFalse((self.root / ".venv").exists())

    def test_existing_incomplete_environment_is_never_overwritten(self):
        environment = self.root / ".venv"
        environment.mkdir()
        marker = environment / "keep-my-environment.txt"
        marker.write_text("preserved", encoding="utf-8")
        result = self.run_setup()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("Nothing was overwritten", result.stdout)
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserved")
        self.assertEqual(list(environment.iterdir()), [marker])

    def test_missing_requirements_fails_before_creating_environment(self):
        (self.root / "requirements-portable.txt").unlink()
        result = self.run_setup()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("Extract the entire ZIP", result.stdout)
        self.assertFalse((self.root / ".venv").exists())


if __name__ == "__main__":
    unittest.main()
