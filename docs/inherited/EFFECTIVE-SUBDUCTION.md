# Lite · effective subduction

Optional mechanical feedback is documented in LITE-MECHANICAL-FEEDBACK.md.
The historical one-sided default described here remains unchanged unless its
separately versioned carrier-traction and force-breakup policies are explicitly
enabled. The experiment changes future source physics through a sealed durable
boundary; it does not reconstruct omitted slab motion or historical work.

**Lite** replaces the detailed slab source with a prescribed trench
traction. Established, finite trench arcs pull their incoming plates even when
their initial speed is zero. The existing spherical force balance derives plate
motion from that load, ridge forces and represented passive resistance. The
mode is the default for new reviewed worlds in this Lite checkout. Saved
configurations and checkpoints keep their recorded laws; old worlds are not
converted.

Lite simplifies the hidden subduction mechanics, while retaining the app's full
map, coastline, material, adaptive and rift detail and saved-frame cadence.

The intended standard is **plausible tectonic worldbuilding**. A simple, tunable
trench force is sufficient if it yields understandable motion and useful,
coherent geological histories. The force coefficient is a modeling control;
acceptance does not depend on resolving every slab process or calibrating the
world against Earth. Preserve force direction, plate connectivity, passive
resistance and material bookkeeping, then judge the resulting motion and maps.
Look for sustained continental migration explained by connected offshore
trenches, ocean-basin closure with surviving continental material, collision
and breakup that respond to geometry and strength, and stable, readable maps
through successive events. The existing short tests demonstrate the constituent
mechanisms; these visible history outcomes are the practical acceptance target.

The aim is to test a simpler causal chain: declared subduction → force → solved
motion → ocean consumption, collision and possible breakup. Removing slab
inventory and neck evolution reduces the represented state. A short coarse
benchmark measured about 1.7× faster whole steps than the coupled slab lifecycle,
and about 1.08× faster force assembly plus solve. Geometry, contact constraints,
material transport and the joint motion solver still run; these measurements do
not establish a general production speedup.

## Select the law

New worlds in the app select **Force balance · revised physics** and **Lite ·
effective trench force** under **Subduction force law**. This selects declared
starting subduction margins. The pull input uses **10¹² N/m**: `5` is
`5e12 N/m`. This new-run choice replaces detailed slab entry, slab-driven rift
traction and rollback behavior with compatible fixed-trench and immediate
foundering closures. The unused detailed controls are hidden; switching back to
the slab law restores their saved draft choices. The starting artwork, mixed
plate ownership and finite trench selection remain available.

The target share of eligible trench length and selection seed choose whole
connected starting arcs. A target below 1 can make an asymmetric arrangement,
but the realized share can exceed the target because arcs are kept intact.
Initialization fails if there is no resolved eligible two-plate interface.
Inspect the resulting trench geometry and incoming ownership before interpreting
continental motion.

The command line uses the same Lite default when no configuration file is supplied:

```sh
python run_simulation.py --initial my-start.json
```

Lite is the user-facing fork name. Existing `effective_subduction` configuration
keys, command-line flags and checkpoint versions keep their meanings.

Use `--subduction-force-n-per-m 5e12` to set its force explicitly. Supplying that
flag or `--effective-subduction` selects Lite when overriding a saved
configuration. `--no-effective-subduction` selects the detailed fresh laws when
no file is supplied, or disables Lite on a supplied configuration. `--config`
without either override preserves the saved mode and force. An inherited finite
trench selection is used unless the configuration supplies one. These commands create fresh histories;
they do not convert or resume a detailed slab checkpoint under another law.

A compatible configuration is:

```json
{
  "physics_profile": "reviewed_v1",
  "effective_subduction": {"enabled": true, "force_n_per_m": 5e12},
  "primordial_subduction": {
    "enabled": true,
    "target_margin_fraction": 0.35,
    "selection_seed": 7
  },
  "subduction_response": "fixed_trench",
  "retained_phases": "disabled",
  "continental_lifecycle": {"enabled": false},
  "rift_traction": {"enabled": false},
  "trench_persistence": "kinematic",
  "slab_allocation": "uniform"
}
```

`primordial_subduction` supplies the declaration geometry and polarity in this
mode. Its historical `initial_slab_depth_km` field has no force or inventory
effect. No inherited slab mass, artificial elapsed time or startup velocity is
created. The force must be finite and positive; `5e12 N/m` is a chosen experiment
parameter, not a calibrated universal trench strength.

Detailed `continental_lifecycle`, `rift_traction`, thermal retained phases,
attached-slab persistence and located slab allocation are incompatible. The
engine rejects raw incompatible combinations. Explicitly selecting effective
subduction in the form or with a command-line flag supplies compatible disabled
overrides for the next fresh run, so inherited detailed presets such as Highland
can use this alternate law without hand-editing their map. The form preserves
the detailed draft for switching back. Existing checkpoints keep their recorded
laws and are not migrated by this selection.

## Force and motion

For segment `i`, the horizontal force is

```text
F_i = f × L_i × w_i × n_i
torque_i = (R × r_i) cross F_i
```

Here `f` is pull per metre in N/m, `L_i` is actual trench length in metres,
`w_i` is the represented incoming ocean fraction, `n_i` is the tangent-plane
normal pointing from the incoming side toward the trench, and `r_i` is the
unit-sphere position. Forces at their actual spherical locations contribute to
the ordinary plate balance. Their net torque can cancel even with long active
trenches and large individual loads.

The traction is one-sided: it acts on the incoming plate. An omitted slab and
mantle reservoir supplies its reaction and power. The represented resistance
must still oppose motion, but this is not a closed energy or angular-momentum
budget for the whole mantle. No matching artificial force is placed on the
overriding plate, no trench-suction term is added, and no independent rollback
speed is prescribed. The trace follows the existing overriding-plate history;
the heuristic forearc retreat transfer is disabled in this mode.

Continental material and plate ownership remain separate. Offshore trench pull
moves a continent directly only if that incoming ocean floor shares the
continent's plate. Pull on an independent ocean plate does not directly load
its overriding continent. Mixed plate connections must therefore be explicit;
see [Overriding response and connected mixed plates](OVERRIDING-RESPONSE.md).

## Persistence, arrival and rifting

Established traces supply force. With initiation disabled, a new convergent
boundary color does not create a system. The optional
[induced initiation rule](LITE-INITIATION.md) requires separate finite loading
history before establishing a later trench. Declared force does not require prior convergence,
maturity buildup, or a velocity kick; quiet and opening intervals do not trigger
the historical shutdown timers. A temporarily unmatched trace remains a record,
but it supplies no force until it matches a resolved interface with its original
owners. Loss of a plate owner ends that system.

Actual ocean intake still needs closing motion, incoming ocean and available
finite overlap. No minimum intake or forced seafloor supply is imposed. Local
force scales with the incoming ocean fraction and becomes zero where incoming
material is fully buoyant; that material enters the existing collision handling
instead of a slab-entry or neck-breakoff calculation. Other portions of a trench
can continue pulling. This does not resolve terrane-specific resistance,
eduction or mineral phases. Later initiation is available only through the
explicit optional induced-initiation policy.

Lite back-arc loading uses the overriding plate's solved trench-normal motion
away from the declared trench in the stationary mantle frame. It applies only
at declared, force-bearing trenches with incoming ocean and actual closing
motion. Current plate ownership and Euler motion are checked, so an old boundary
color alone cannot activate the rule. This removes the detailed slab-depth and
mass gate without adding independent rollback or a force driving the upper
plate. Existing arc footprints, sector selection, loading memory, rupture gates,
basin geometry and measured opening histories remain. New Lite histories record
`backarc_driver_version: 2`; older histories keep their saved driver.

The same local forces are exported through the shared force ledger used by
the optional `force_limit_rifting` experiment. Opposing forces can therefore
produce internal tensile loading despite small rigid plate motion. Enabling
effective subduction alone does not enable a new rupture criterion: ordinary
rifting keeps its selected law, and force-limit breakup must be selected
explicitly. That separate law still needs a viable cut and material-strength
criterion; force or a stationary symmetric ring does not guarantee breakup.

For the internal-tension investigation, check **Breakup from plate forces** in
the app, or add `"force_limit_rifting": {"enabled": true}` to the compatible
configuration above. The form preserves any saved parameters of that law.
Keep this separate from the basic startup and speed comparison, because the
additional membrane and rupture analysis has its own computational cost.

## Scientific qualification and evidence

This is a computational approximation consistent with the force-balance emphasis
of Scotese's Rules I–II, the plate-normal preference of Rule VI and persistent
subduction in Rule IV. It departs from the literal attached-slab explanation by
prescribing a constant horizontal force instead of deriving it from slab negative
buoyancy, depth, dip and attachment. Persistent declaration handles an assumed
already-established zone. With initiation disabled it does not explain how a
later zone starts; the optional induced-initiation policy adds the separate
observed loading gates documented above. Local
buoyant arrival is a reduced collision cutoff, not a claim that every small
terrane stops an entire real subduction system. For Rules VII–IX, the simple
back-arc rule retains dependence on absolute upper-plate motion and localized
basin geometry. Independent rollback, slab suction and resolved slab-driven
upper-plate forces remain outside this closure.

The chosen behavior lets us isolate geometric torque cancellation, ownership,
startup from rest and represented resistance without solving unresolved slab
details. Its consequences include no age- or depth-dependent strength, no slab
mass conservation or resolved breakoff, no independent rollback or suction,
no spontaneous initiation and no guarantee of self-sustaining ocean supply.
These are declared limitations rather than a scientific disagreement with the
guide. The supplied [Rules of Thumb](RULES-OF-THUMB.md) remain unchanged.

Frames record `effective_subduction_version: 1` and
`effective_subduction_diagnostics`, including force, initial selection, matched
and force-bearing length, integrated incoming force and the omitted-reservoir
closure. Force magnitude alone is not net torque. Review actual plate and
continental displacement, solved-force residuals, passive resisting work,
local buoyant cutoff, and checkpoint agreement. Tests qualify the implemented
equations and bookkeeping; they do not calibrate this approximation against
Earth or establish billion-year predictive accuracy.

The [dated qualification receipt](benchmarks/effective_subduction/QUALIFICATION.md)
and [raw measurements](benchmarks/effective_subduction/measurements-20261002.json)
record a 320-control comparison with seven warmed force-solve samples and three
0.05 Myr whole steps per law. Median whole-step times were 8.163 s for coupled
slab lifecycle, 5.724 s for inventory-only slabs and 4.736 s for effective
traction. The laws produce different forces and evolving trajectories, so this
is a short workload comparison, not an accuracy comparison or long-run guarantee.

The first solve from exactly zero input velocity produced motion without slab
mass, elapsed time or initial ocean consumption. A separate mixed plate test
pulled continental carrier UID 3 through its two offshore incoming trenches:
all 206 original continental markers survived a real 0.02 Myr step and moved
8.91 m on average. Both native startup experiments replayed bitwise through
checkpoints. Mechanical tests cover torque and subdivision, passive resistance,
local buoyant arrest, opposing-force cancellation with internal tension, and
strong-cut arrest versus weak-cut opening. The reproducible experiment is
[experiments/effective_subduction.py](experiments/effective_subduction.py).
