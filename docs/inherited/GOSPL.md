# Evolving tectonic history for goSPL

> **Documentation scope:** This retained design document includes historical development results. Refer to [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) and [VALIDATION.md](VALIDATION.md) for the initial GitHub snapshot. References to local `validation/`, `output/`, legacy copies, and handoff documents describe artifacts not included in this repository.

The intended worldbuilding deliverable is detailed terrain at the final chosen epoch, supported by a reviewable tectonic history. Intermediate epochs need approximate regional elevations, plate boundaries, and material records sufficient to carry that history forward. They do not each need a finished high-resolution heightmap.

Deep Time's detailed terrain generation runs only for an explicitly selected epoch. The goSPL adapter uses the recorded regional surfaces to estimate changing tectonic forcing; it does not generate a sequence of procedural 8K surfaces or impose the intermediate reference maps on goSPL. A landscape-evolution run still maintains intermediate numerical elevations because erosion, deposition, and drainage depend on the preceding surface. Those working states can remain approximate while the final landscape is the result to develop and assess.

Final relief should retain the effects of earlier tectonics as well as current boundaries: old collision belts and accumulated deformation matter alongside ongoing uplift, rifting, arc construction, and ocean-floor aging. Current plate boundaries alone are not the complete terrain history.

Back-arc formation enters through the ordinary saved plate motion, elevations and material deformation. Its basin history is review metadata, not a second vertical forcing term. Do not subtract a separate basin-depth field or add cumulative opening to goSPL uplift. The captured engine import closure includes the back-arc process and spherical geometry helpers.

Use **Export to goSPL** beneath the terrain controls. Open a saved experiment, choose the starting and ending epochs, select mesh density, and choose **Build history ZIP**. The exporter reads the saved interval while simulation and review remain available. Its source experiment and times stay fixed if you select another world. Completed downloads remain in **Saved goSPL exports**.

To choose the world's placement, open **Orient the globe** beneath the map and choose **Apply to view & exports** before building. The fixed-axis rotation applies yaw about Z, then pitch about Y, then roll about X, in degrees. The browser remembers this setting per experiment. A goSPL job captures those angles along with its selected history; later view changes do not alter it. Existing saved exports keep their recorded orientation, and their source note identifies a mismatch with the current view. Build again to export the new placement. New detailed terrain builds capture the same applied angles.

The export's elapsed-time display measures active package construction, including validation and ZIP writing. Its approximate time remaining appears after enough progress has been measured and adjusts to recent speed. It is separate from the goSPL solver's runtime. Deep Time also shows active elapsed time for terrain builds and for simulations; simulation resume preserves that total while excluding paused time.

The export starts from the recorded topography at the first selected epoch and supplies a forcing file for every subsequent saved interval. It includes `input.yml` for landscape evolution, `forcing-check.yml` for goSPL's fast inspection mode, a standalone validator, reference crust/boundary/elevation maps on the mesh, and source fingerprints. It uses no cloud service and requires no new Deep Time dependencies.

## Bring the evolved landscape back

Under **Develop the terrain**, choose **goSPL landscape** as the surface source. Browse to the solver's `gospl.xdmf`, or paste its output-folder path, then click **Load results**. Select the required solver epoch (latest by default) and **Render goSPL landscape**. Keep the complete output folder, including its `xmf` and `h5` subfolders. A single partition's HDF5 file requires the corresponding complete epoch descriptor and the other referenced partitions.

The importer reads nodal `/elev`, `/coords` and triangle `/cells` from all declared MPI blocks. It welds duplicate shared nodes/triangles, rejects conflicting copies and incomplete/nonclosed geometry, and samples the reconstructed spherical surface directly. Only the chosen epoch's HDF5 arrays are loaded. Solver time is labeled separately from a tectonic experiment's elapsed time. Existing export metadata is retained when available; the importer does not invent a world-history association for unrelated output.

Terrain detail defaults to **0** to retain the computed landscape. Imported coordinates already include the export's globe rotation, so the interface preserves that orientation. An 8K image provides finer sampling of the solver surface, not additional solved drainage detail. The completed terrain package retains the canonical mesh/elevation, source descriptor/HDF5 fingerprints, reconstruction metadata and the rendering code; raw multi-epoch HDF5 histories are not duplicated into it.

Reading HDF5 requires either local `h5py` or, on this laptop, the existing **DeepTime-goSPL** WSL runtime. The Windows NumPy/Pillow environment does not need a new HDF5 installation. The WSL bridge runs a fixed reader on selected local data; it does not start goSPL. On another computer, install `h5py` in the application's Python environment or restore the documented WSL setup. The renderer itself remains NumPy/Pillow based.

Real existing two-worker outputs have been imported at both 642 and 163,842 unique nodes. These checks establish input reconstruction and rendering, not a new long-duration landscape-evolution validation. See [CODEX_HANDOFF.md](CODEX_HANDOFF.md) for the remaining terrain/climate acceptance work.

## Run the package

On this Windows laptop, **Run-goSPL.cmd** opens a file picker. Select `input.yml` from an extracted export for landscape evolution, or `forcing-check.yml` to inspect the prescribed motion without erosion. You can also drag either YAML file onto the launcher. Keep the YAML and its adjacent `input` folder together: the YAML describes the run and refers to the mesh and forcing NPZ files in that folder. Selecting the ZIP itself or a heightmap PNG is not sufficient.

The launcher uses the dedicated **DeepTime-goSPL** WSL 2 distribution and normal Linux user `gospl`. Its isolated micromamba environment is `/home/gospl/micromamba/envs/gospl`, pinned to goSPL 2026.7.14 and Python 3.12. It uses two MPI workers by default and one numerical-library thread per worker. The pinned environment recipe is `gospl-environment.yml`; it is separate from the Windows Python environment used by Deep Time. This follows the [official Windows/Conda installation route](https://gospl.readthedocs.io/en/latest/getting_started/installConda.html).

`gospl-linux-explicit.txt` records the exact 318 Linux package builds installed on this laptop. The runtime itself lives in a separately installed WSL virtual disk; it is not inside the application ZIP. The launcher selects this distribution explicitly and does not change the laptop's default WSL distribution. Its OpenMPI settings use local shared memory and TCP without changing Windows or Linux security settings.

The model runs in the selected YAML's folder, streams progress to the terminal, and writes the output directory configured in that YAML there. Exported configurations use numbered output folders to preserve earlier results. Press **Ctrl+C** in the run terminal to stop goSPL. Deep Time's pause/resume button controls the tectonic simulation, not this separate landscape process.

From PowerShell, these commands inspect or launch a selected export:

```powershell
.\Run-goSPL.ps1 -InputFile 'C:\Worlds\My world\input.yml' -DryRun
.\Run-goSPL.ps1 -InputFile 'C:\Worlds\My world\input.yml' -Processes 2
```

The preserved historical full-world export is already unpacked under `output/gospl/20260905-101354-679c598d/`; its `input.yml` can be selected directly. It predates the current crustal-column, local-accretion and trench-lifecycle revisions and retains its original source history and forcing. Running that file evolves its full chosen 1,000 Myr history. Build a new export from a new experiment to use the current foundations. The limited installation checks under `output/gospl-install-check/` cover only 2 Myr and are separate from the full world.

The actual 163,842-node mesh passed a two-worker, 20-step landscape test through this launcher in **346.35 seconds**. Its final HDF5/XDMF outputs passed finite-field, mesh-coverage, and time checks. A full billion-year landscape evolution remains a much longer job; this short test does not establish its eventual runtime or convergence. See `VALIDATION.md` and `output/gospl-install-check/validation.json` for the measured scope.

For a different machine with goSPL installed, the equivalent direct commands are below.

Unzip into a new folder. In an environment where goSPL is installed, run from that folder:

```sh
python validate_export.py
gospl -i forcing-check.yml
gospl -i input.yml
```

The files use JSON syntax, a valid subset of YAML. They follow the documented **goSPL v2026.7.14** mesh/forcing contract. The exporter validates geometry, dimensions, units, interval alignment, and archive construction; it does not execute goSPL or install its PETSc/MPI environment. [goSPL installation and execution](https://gospl.readthedocs.io/en/latest/user_guide/running.html)

The normal configuration enables erosion and deposition. The inspection configuration sets `fast: true` to review prescribed motion and vertical changes. Both use `makedir: true`, which creates numbered output directories. Uniform rainfall defaults to 1 m/year; rainfall, stream-power erodibility and diffusion are editable starting assumptions. Sea level starts at the model's zero-metre datum.

## Time, mesh and units

The original timeline remains **elapsed Myr**. Exported goSPL time starts at zero at the selected first epoch and is measured in **years**. For example, exporting 100–120 Myr gives a 0–20,000,000 year goSPL run. `source/frame-index.csv` provides the mapping. Each interval has explicit start/end times, and the final selected epoch is included.

The default goSPL step is 100,000 years, independently of the 2 Myr tectonic integration. The step must be a positive whole number dividing every saved interval exactly; the exporter rejects a misaligned choice because `interp` advection is applied at interval ends. A 100,000 year step divides an ordinary 2 Myr interval into 20 landscape steps.

| Mesh level | Nodes | Approximate average edge spacing |
| --- | ---: | ---: |
| 6 | 40,962 | 120 km |
| 7, default | 163,842 | 60 km |
| 8 | 655,362 | 30 km |
| 9, very large | 2,621,442 | 15 km |

Meshes are closed icospheres with triangles, 5–6 neighbors per vertex, and no duplicated longitude seam or polar fan singularity. Mesh resolution is separate from the 8,192 × 4,096 heightmap. Finer meshes sample the recorded tectonic fields more densely; they do not recover absent fine-scale tectonic history. Export storage scales with both mesh size and the number of intervals, and a high-resolution goSPL run can require substantially more resources than this laptop's tectonic simulation.

`mesh.npz` contains Cartesian `v` coordinates in metres on a sphere of radius 6,371,000 m, zero-based triangular `c` connectivity, and separate `z` elevations in metres. The X axis points to longitude 0°, Y to 90° E, and Z north. Every forcing file has `hdisp` of shape (N,3) in Cartesian m/year and `upsub` of shape (N,) in m/year. [Native mesh inputs](https://gospl.readthedocs.io/en/latest/user_guide/inputfile.html), [tectonic forcing](https://gospl.readthedocs.io/en/latest/user_guide/optfile2.html)

With an applied orientation, the mesh retains these standard axes while the original saved history is sampled at the corresponding inverse-rotated positions. Horizontal forcing vectors rotate into the output coordinates; elevation, vertical forcing, and reference fields follow the same placement. This avoids fitting motion from repeatedly resampled history grids. `export_metadata.json` records the angles, matrix, convention, and orientation-source fingerprint, and `source/orientation.py` preserves that implementation. The original run, its motion estimates, and the tectonic force rules remain unchanged. See the sampling limits in [README.md](README.md); rotating or refining a mesh does not create missing tectonic detail.

## What the vertical forcing means

The adapter estimates each finite plate rotation from matching material markers where possible. For untracked plates, mainly ocean, it uses recorded Euler motion. If a saved interval spans different motions after splitting or welding, the fitting residual is reported and the fit is explicitly classified as approximate. Horizontal forcing uses the chord from each initial vertex to its rotated destination divided by interval years. This matches goSPL's `interp` implementation, which adds the displacement and then remaps onto the fixed mesh.

At each original mesh position `p`, the adapter computes:

```text
geometric change = next elevation at rotated(p) − previous elevation at p
vertical forcing = (geometric change + estimated erosion correction) / years
```

Correcting for motion prevents an arriving continent from being interpreted simply as local uplift at a fixed coordinate. Geometric change also carries arc emergence, crustal exposure, mixtures of overlapping column heights, thermal bathymetry and sampling effects. Current column histories have no separate overlap-height bonus. The approach is informed by the [geometric forcing method in the official goSPL examples](https://github.com/Geodels/goSPL-examples/blob/main/shared_scripts/umeshFcts.py).

The erosion correction uses matched material `trace_erosion_m` counter changes, interpolated on a grid no finer than 256 × 128 and restricted to the same plate and crust class. New crustal-column histories record **net denudation after rebound**: adding that net loss back removes both the model's simple erosion and its associated rebound before goSPL develops the landscape. Gross removal and rebound diagnostics must not be counted again. Unsupported regions use the saved last-step net erosion rate over the interval, an approximation rather than a measured interval total. Legacy histories retain signed relief relaxation and a bounded exponential fallback; a negative legacy correction represents relaxation of negative relief. Ocean cells receive no continental erosion correction. Separate `geometric_rate`, `relaxation_correction`, and `correction_supported` arrays and per-interval support statistics make these choices reviewable. Rift cooling and foreland deflection already contribute to saved elevation and therefore geometric forcing. See [FOUNDATIONS.md](FOUNDATIONS.md) for the height ledger.

For recorded ridge-subduction episodes, the temporary slab-window bulge rises and cools through the geometric change above. New column histories use the saved net erosion rate, which excludes this temporary overlay. For legacy histories, both analytical fallback paths remove the recorded thermal overlay before estimating relief decay, so thermal subsidence is not misidentified as erosion to add back. That overlay is reconstructed on the original source grid and interpolated like elevation. Lasting ridge-related volcanic construction is already included in surface relief and total uplift; its separate history counter is a subset and is not added again. Packages capture `source/ridge_interaction.py` and its fingerprint to preserve this sampling rule. Histories without slab-window records retain their earlier correction behavior.

Continental rift inversion also changes the recorded material relief. Its uplift and subsequent erosion therefore enter the same geometric-forcing and erosion-correction calculation, using the convention recorded by that history. `trace_inversion_uplift_m` identifies the portion of total uplift supplied by inversion; the exporter does not add this contribution again. Captured engine helper sources include `rift_inversion.py` for histories generated with that mechanism. The regional history records the originating rift and the extension-to-inversion sequence.

This reduces applying Deep Time's erosion twice. Sparse material sampling, changing ownership, clipping and mixed deposits mean the correction is approximate. It does **not** turn the surface reconstruction into a complete physical uplift budget. goSPL's own three-neighbor interpolation and subsequent erosion/deposition also change the resulting terrain. The output is intended to preserve the tectonic basis of a worldbuilding history while allowing goSPL to develop its landscape; exact replay of saved maps is not promised.

Each interval retains `reference_z_end`, `reference_crust_end`, and `reference_boundary_end` on the fixed mesh for comparison. Those are review fields, not constraints applied to goSPL's evolved surface. Procedural 8K detail is not inserted into tectonic rates because those fixed-coordinate fine patterns are not material histories.

Histories predating saved material/erosion counters cannot support this corrected evolving export and receive a clear message. The new exporter works with compatible existing histories; curved breakup geometry applies to newly generated histories. ROCKE-3D configuration remains a separate downstream task.
