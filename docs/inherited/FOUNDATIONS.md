# Geological foundations for new histories

> **Documentation scope:** This retained design document includes historical development results. Refer to [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) and [VALIDATION.md](VALIDATION.md) for the initial GitHub snapshot. References to local `validation/`, `output/`, legacy copies, and handoff documents describe artifacts not included in this repository.

The model builds a tectonic framework for worldbuilding, a reviewable history, and subsequent landscape development. Four additions strengthen that framework: material crustal columns, continental basin subsidence, local terrane accretion, and persistent subduction histories. They apply to newly generated histories. Earlier saved worlds retain their original fields, equations and provenance.

This document describes the implementation and its assumptions, not a claim of scientific calibration or a completed full-world validation. Validation results belong in [VALIDATION.md](VALIDATION.md). The existing mantle-force equations remain unchanged; these additions do not introduce a three-dimensional mantle or lithosphere solver.

## 1. Crustal thickness, buoyancy and erosion unloading

Every buoyant parcel and representative history marker carries a scalar crustal column. Initial neutral thicknesses are **35 km** for mobile continental crust, **42 km** for cratons and **25 km** for juvenile arcs. Each column is anchored to its actual starting elevation, so choosing a thicker reference column does not impose an extra plateau. These reference values are model assumptions, not inferred rock types.

Local shortening thickens a column; extension thins it. Normal boundary speeds in km/Myr are converted to unresolved strain across a nominal **400 km** belt, with a **0.25** partition, existing craton resistance, and a rate cap of **0.08/Myr**. Thickness stays between **8 and 75 km**. A reciprocal effective-area factor conserves `thickness × effective area` during this mechanical change. The actual transported parcel positions, finite footprints and reference-area weights remain unchanged. This is a local column-volume accounting device, not resolved deformation of three-dimensional crust.

New histories sample those rates directly at moving material positions using finite spherical boundary segments. A cosine-squared profile reaches zero **400 km** from the segment, with smooth endcaps and a cross-belt integral of 400 km. Only participating owners receive the load; volcanic belts are displaced **180 km** toward the overriding plate. Overlapping segments combine by maximum so subdividing a boundary does not multiply its forcing. Rift identity and extension memory follow columns that actually thin throughout the belt. These widths and profiles are reduced-model choices. Boundary detection still uses the tectonic grid, so this removes pixel jumps in column forcing without making the whole simulation independent of raster resolution.

Local Airy compensation turns thickness changes into elevation. With mantle and crust densities of **3,300** and **2,800 kg/m³**, a 10 km thickness increase supplies about **1.52 km** of subaerial elevation. Below the water datum, the buoyancy response includes water density **1,030 kg/m³**. The scientific basis is equal column mass above a compensation depth; water and sediment loads alter that balance. The implementation includes the water term but no sediment layer. [Harmonica's documented Airy equations](https://www.fatiando.org/harmonica/latest/api/generated/harmonica.isostatic_moho_airy.html)

Erosion removes actual rock thickness from exposed positive-elevation columns. Unloading produces instantaneous local rebound, so surface lowering is less than gross rock removal. The existing erosion control sets a simple height-dependent denudation request with 180 Myr and 350 Myr decay scales for mobile material and cratons. It does not calculate rivers, rainfall, sediment transport or erosion of submerged basins.

Volcanic construction adds column material at fixed effective area. Inherited-rift inversion thickens the column with reciprocal area adjustment. Both are limited by the available thickness capacity, and only realized elevation gain enters their counters. The former overlap-based elevation bonus is replaced by column-derived support; overlap is a sampling quantity, not an independent mountain-building term.

## 2. Rifted margins and basin accommodation

Actual thinning also stores bounded thermal support on the moving material. Once extension ends, that support decays with a **63 Myr** time scale while the thinned crust persists. The resulting continued subsidence can leave an old rift or passive continental margin below sea level after its active boundary has moved away. This implements the relationship between stretching, thinning and later cooling described by [McKenzie, 1978](https://www.zetaware.com/public/McKenzie_1978.pdf), DOI [10.1016/0012-821X(78)90071-7](https://doi.org/10.1016/0012-821X(78)90071-7).

The numerical heating rule is a bounded strain-driven ordinary differential equation, with coupling **1.4** and an upper support scale equivalent to approximately **3.2 km** under water. The cooling time and scale are inspired by McKenzie's example. This is not his exact stretching-factor solution, a solved temperature profile, or measured heat flow. An accepted initial structural rift starts with zero thermal anomaly and no invented extension history. Progressive local rifting now supplies gradual interior thinning and heating through this ledger; it does not add a fixed 250 m cut at breakthrough. That requested cut remains a property of the explicit legacy fracture helper. [RIFTING.md](RIFTING.md) explains the local mechanical update and the subtraction of already-realized boundary thinning.

Current continental collisions also load a nearby foreland-depression proxy. The trough follows finite spherical collision segments on the same material owner. It is zero within a **60 km** central peak belt, deepest about **250 km** away, and ends by **440 km**. Its target depth is one quarter of nearby mountain height above 500 m, capped at **1,800 m**. Deflection approaches a growing target over **12 Myr** and relaxes after unloading over **100 Myr**.

Mountain loading can bend lithosphere and create adjacent accommodation; classical flexure depends on rigidity and load geometry. [University at Buffalo's flexure equations and examples](https://www.glyfac.buffalo.edu/mib/class/325/Lecture/14/1401Thermal/thermal.html) describe that relationship. This program uses a compact trough and delayed response instead of solving those equations. It does not resolve a forebulge, flexural rigidity, sediment loading or the full lingering load after a boundary disappears. The inherited deflection's slow relaxation supplies only a bounded memory of that load.

These fields create **space that could receive sediment**. They do not fill it. A continental shelf is not guaranteed along every coast, and a low surface remains continental crust if its material has continental origin. Fine shelf breaks, deltas and depositional wedges belong to later landscape and sediment modelling.

## 3. Local, coherent terrane accretion

Accretion now depends on sustained contact between connected buoyant material bodies. The history clock follows the actual nearby patch sets on both sides as a contact advances. Both sets must overlap the earlier local contact for its loading to continue; matching just a plate name or one side is insufficient. When local contacts merge, the strongest existing loading survives without adding their debts together. A remote island sharing the incoming plate does not inherit another terrane's collision clock.

After sufficient convergent loading, the smaller connected terrane can join its receiver. Complete material patches and protected original cratons remain indivisible. Unresolved juvenile particles cannot bridge an otherwise empty ocean to annex remote crust. The receiver's existing motion averaging uses the transferred terrane's reference area; the source retains its remote material and oceanic support. Contact anchors are remapped when juvenile parcels coalesce. In fresh reviewed native worlds, overlapping sheets of one plate (a welded thrust stack, more than 1 km2 of overlap) count as one terrane, so a docked terrane stays with its host's stack until a later rift or fault separates it, and loading built toward a plate the rock has left in an accretion transfer is cleared. A whole stack can dock only once its farthest face has matured inside the 400 km process zone: roughly 30 Myr for a stack reaching 185 km behind the contact, about 95 Myr at 285 km, and effectively never beyond about 300 km.

Transferred columns keep their thickness, cooling, basin and rift histories. An accreted arc changes its current classification, while its recorded origin remains juvenile. The relief coordinate is adjusted to cancel the class's 100 m datum difference; classification itself must not raise the full material surface. Independent marker identities survive these changes.

Before accretion, overlapping buoyant sheets can remain distinct material bodies. Surface ownership follows the transported interface while that owner's own deposited footprint independently meets the 36% visibility threshold. When its footprint becomes insufficient, the dominant current deposit becomes exposed. This avoids turning slight differences between overlapping parcel samples into a checkerboard of false plate boundaries. It is an exposure convention, not extra accretion or a transfer of hidden material. All deposits still contribute to the same conservative area and height averages.

Natural accretionary orogens combine magmatic growth, buoyant terrane collisions, shortening and crustal reworking. [Cawood et al., 2009, USGS publication record](https://www.usgs.gov/publications/accretionary-orogens-through-earth-history) provides the geological basis. The model represents a coherent local transfer with heuristic contact thresholds. It does not resolve thrust sheets, underplating, subduction erosion or a detailed terrane-detachment fault system.

## 4. Subduction systems with persistent local histories

Connected trench segments receive persistent identities, incoming and overriding plate UIDs, polarity, approximate spherical geometry, accumulated convergence and lifecycle records. Geometry follows its overriding plate between observations and is matched to current contacts. Separate trenches between the same plate pair can retain separate histories.

New systems begin with zero maturity. Arc supply develops as both active duration and cumulative shortening accumulate, reaching the model's mature state only after at least **10 Myr** and **100 km** of convergence. Quiet intervals retain memory; sustained buoyant collision, extension, disappearance or loss of an owner can end an episode. Default delays are **6 Myr** for buoyant blockage, **10 Myr** for sustained extension and **20 Myr** for other inactivity. A recently terminated trench can record a new episode within a **100 Myr** local memory window.

A temporary gap in an existing trench does not immediately create a new geological story. A matched branch must remain spatially separate for **16 Myr** before receiving a separate identity; this is a numerical persistence filter, not an inferred physical separation time. Throughout that interval, only actual converging segments supply arcs or retreat, with no production across the gap. Sustained separation and local host transfer inherit the developed slab's maturity. When tracked pieces rejoin, the history records a **joined continuation** and its successor identity, rather than reporting a geological shutdown. Back-arc loading follows that continuing identity without being reset. Initially distant trenches and actual polarity reversals still retain separate histories.

These are history and production rules, not simulated slab depth. The distinction between initiation, episodic restart, polarity reversal and simple separation follows the [Crameri et al., 2020 subduction-initiation compilation](https://pmc.ncbi.nlm.nih.gov/articles/PMC7385650/). That study documents diverse initiation settings and uncertainty; it does not establish this program's universal maturation or shutdown times. Persistent polarity and mature arc supply do not demonstrate that a self-consistent subduction force balance has been solved.

## State, saved fields and height accounting

The checkpoint dictionaries `structure` and `trace_structure` contain plain arrays. They move with their material identity; scalar values require no special pole treatment. Juvenile coalescence preserves reference-area-weighted surface height and effective column volume. The common finite-footprint deposition kernel projects geological scalars onto the map, so extra fields use the same spherical sampling as the crust.

| Internal field | Unit and meaning |
| --- | --- |
| `thickness_km` | Current column thickness, km. |
| `reference_thickness_km` | Neutral reference thickness, km. |
| `reference_elevation_m` | Neutral elevation anchor, metres; may be reanchored during juvenile mixing. |
| `area_factor` | Dimensionless effective area relative to parcel reference area. |
| `rift_heat_m` | Air-equivalent thermal support, metres; not a temperature. |
| `rift_age_myr` | Time since most recent modeled thinning; −1 means no recorded thinning. |
| `foreland_m` | Current downward deflection, actual metres. |

Optional saved grids include `crustal_thickness_km`, `crustal_root_km`, `rift_thermal_support_m`, `rift_cooling_age_myr`, `foreland_deflection_m` and `erosion_rate_m_myr`, plus the local `trench` display index and `trench_systems` records. The root field is **positive thickness excess over the neutral reference**, not absolute Moho depth. Ocean cells use a constant 7 km crustal-thickness reference. Cooling age is reported only where at least half the deposited material has a recorded thinning age. Saved thermal support is the actual elevation effect, which differs from the internal air-equivalent value under water.

For histories with `structure_version: 1`, the legacy marker ledger remains:

```text
relief = birth_relief + uplift − tectonic_lowering − net_erosion + adjustment
net_erosion = gross_denudation − rebound
```

`trace_extension_m` retains its file name but now records all positive tectonic lowering: mechanical extension, loss of rift thermal support and increasing foreland deflection. `trace_rift_extension_m` is the mechanical inherited-rift subset. Positive thermal support and foreland unloading enter total uplift. Inversion and ridge-subduction volcanic counters are subsets of total uplift. Gross denudation, rebound and cumulative basin subsidence are explanatory diagnostics; adding them again would double-count the height changes. Arc replenishment and class-datum corrections remain explicit adjustments.

The goSPL adapter derives geometric elevation change along saved plate motion, then adds back the recorded **net** erosion loss. This removes the model's simplified denudation and associated rebound together, allowing goSPL to develop its own landscape. Matched material histories supply the preferred correction. Unsupported locations use the saved last-step net erosion rate as a bounded temporal approximation, with coverage reported. They must not use the former relief-decay formula for a new column history. Older histories retain their original signed-relaxation convention. See [GOSPL.md](GOSPL.md).

Rift thermal support and foreland deflection are already in saved elevation. They are not extra goSPL uplift or subsidence inputs. The separate temporary ridge-subduction thermal overlay also remains distinct from permanent volcanic construction. Fine procedural terrain belongs to the selected final stage; it is not a tracked uplift history for every detailed pixel.

## Practical limits

The reference-area inventory is not rock mass in kilograms. Effective column volume does not make fixed parcel footprints a deforming finite-element mesh. Numerical deposition, crust visibility thresholds, overlapping material, rigid plate motion and source resolution still affect mapped geography. Fixed densities and category-based resistance omit compositional buoyancy, density evolution, fault mechanics and temperature-dependent rheology. These columns change elevation without replacing the existing mantle or plate-driving equations.

No lithology, mineral phases, sediment stratigraphy, drainage routing, compaction, glaciation or water-volume-driven sea level is added here. goSPL supplies a later landscape stage; ROCKE-3D still needs separate climate boundary-condition preparation. Review the coarse tectonic and material histories to understand broad terrain ancestry, and treat fine output detail as worldbuilding terrain synthesis rather than newly resolved geological evidence.
