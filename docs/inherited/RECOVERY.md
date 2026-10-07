# Recovery and authored branches

New simulations save a resumable checkpoint after initialization and then every
10 Myr, at the end of the first completed integration step reaching the interval.
This does not split integration steps or add public history frames. A checkpoint
between saved epochs can therefore be newer than the most recent viewable frame.
Status reports `checkpoint_time_myr` separately from the public frame list.

`checkpoint.npz` is the latest committed state; `checkpoint.previous.npz` retains
the preceding readable generation. A new archive is flushed before atomic
replacement. The prior primary remains available while its backup is committed.
Both use the existing explicitly typed, non-pickle format and preserve the exact
pending output time, random generator, native state and engine/helper hashes.

After a process interruption or error, reload the saved experiment. The loader
fully decodes candidate checkpoint state, verifies source compatibility and its
matching frame index, and selects the newest readable complete generation. It
reports the recovered epoch and reverted frame count. Resume copies later frame
artifacts and the old manifest into `recovery-abandoned/<id>/` before their frame
numbers can be reused. Those artifacts remain available for diagnosing the failed
attempt. A deliberate cancellation or completed run stays terminal.

The recovery guarantee applies to committed checkpoints. It does not imply that
every history frame can resume, nor that an interrupted timestep is recoverable.
Persistent disk failure still stops a run; an incompatible engine is never used
to resume it.

`POST /api/branch` accepts the selected paused `run_id` and a portable
`world_design`, and returns a new paused run. The parent must have an exact
compatible checkpoint. The branch preserves its committed frames byte for byte,
copies and verifies captured sources, and writes a separate checkpoint carrying
the new future design. Resolution, duration and integration settings stay those
of the parent. Interventions that already affected recorded history must remain;
new or changed active interventions must start at or after the branch epoch.

The new manifest records `branch_origin` with parent run, checkpoint epoch and
SHA-256, and design revision. Its engine records an authored branch event at that
epoch. The manifest is published only after the complete branch has been written;
an interrupted partial copy does not appear as a resumable saved experiment.
The parent is never rewritten by branching or by running its child.

Focused checks are in `tests/test_recovery_checkpoints.py`,
`tests/test_checkpoint_branch.py`, and the existing `tests/test_pause_resume.py`.
