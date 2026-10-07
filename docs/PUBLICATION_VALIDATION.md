# Publication validation — 7 October 2026

- All 230 sealed engine and browser files match the running source's prepared SHA-256 inventory byte for byte.
- All 729 copied-source entries in SOURCE_MANIFEST.json were checked against their published hashes. The manifest inventories copied files; publication-authored documentation and Git metadata are separate.
- All 614 included Python source files parsed successfully without importing or executing them.
- Credential-pattern and private-locator scans, plus an independent source/documentation review, found no remaining publication blocker.
- 12 focused tests passed in 2.615 seconds using Python 3.12 and the existing dependencies. The command was:

    python -B -m unittest tests.test_coarse_history tests.test_burial_exact_homogeneous tests.test_candidate_worker_contract -q

These tests cover synthetic mode activation/metadata, exact burial geometry, decimal context restoration, and inherited worker budget behavior. They do not construct or advance a simulation. No second model run or server was launched for publication.

The full inherited test suite was not rerun. Passing these checks is not a claim of 1000 Myr completion, Earth calibration, a guaranteed simulation speed, or portable production startup. The recorded production's external controller and accepted history remain separate from this source repository.
