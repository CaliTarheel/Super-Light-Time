# Recorded production operations

These are source references for the 2026-10-07 burial-speed continuation: preparation, the independent Windows Job deadline controller, shared readers, and receipt-checked native PNG delivery. The actual simulation engine and encoder are at the repository root.

They are **not portable launchers**. The original scripts required exact source hashes, approved bindings, immutable source and acceptance receipts, an existing checkpoint/history, Windows process creation identities, and a specific filesystem layout. Those local records are not distributed. Private machine paths and Drive destination identifiers have been replaced with explicit placeholders; the source manifest records original and published hashes where changed. The shared helper files are collected in shared/ for inspection rather than recreating the original machine layout.

Do not use these scripts to restart a spent attempt, fabricate acceptance, or disable the running simulation's guard. Their presence does not arm a guard for server.py. Generic local app usage is documented in the root README. The archive tests retain their original operational path assumptions; they are reference code and are not advertised as portable tests.

The delivery helper reads accepted files and writes local export receipts; it does not upload through Google Drive itself. No Drive credentials, live folder identifiers, account state, authorization/control bindings, active process records, saved worlds, checkpoints or history are included.
