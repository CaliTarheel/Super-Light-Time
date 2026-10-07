# The spherical mesh edition

> **Documentation scope:** This retained design document includes historical development results. Refer to [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) and [VALIDATION.md](VALIDATION.md) for the initial GitHub snapshot. References to local `validation/`, `output/`, legacy copies, and handoff documents describe artifacts not included in this repository.

This edition makes spherical geometry authoritative during evolution. The browser still shows a familiar rectangular map, but that map is an output projection. It is not the grid on which ocean transport, material coverage, plate connectivity or native boundary geometry are calculated.

There are two connected representations: a fixed, closed triangular control mesh for ocean lithosphere and boundary calculations, and adaptive moving triangular surfaces for continental, cratonic and island-arc material. The material layer combines rigid cratons/interiors with deforming regions around active boundaries.

## What carries the world

The control mesh is a subdivided icosahedron. Vertices use unit-sphere Cartesian coordinates. Each face has its actual spherical area, three neighbors and shared-edge geometry. Edges have great-circle length and a tangent normal pointing from one adjacent face toward the other. There is no longitude seam or special polar row in this topology. Plate ownership, ocean age, structural history and fractional plate support live on native cells.

Buoyant crust consists of actual indexed spherical triangles. Neighbors share vertices within their owner and material region. An ownership change duplicates vertices where the two sides must move independently. Each material face retains a stable identity, reference area, crust category, owner and geological history. Connected cratons retain protected identities. An array row is not a permanent rock identity.

Rigid plate motion rotates shared material vertices once using a finite Euler rotation. Deforming regions add bounded relative motion while rigid anchors keep their exact rotation. Point queries use spherical triangle containment and explicitly return missing coverage or multiple overlapping sheets. They do not extrapolate a nearest face across a material hole. Deterministic edge ties give consistent categorical sampling at shared boundaries.

Physical coverage is measured by **spherical polygon intersection** between material and control triangles. A small island contributes its actual intersected area, rather than an entire coarse cell. Overlapping material sheets remain separate coverage records: reference material is not deleted to hide collision overlap. The engine selects exposed material for mapped properties. Approximations remain in assigning exposed ownership and boundary types on the finite control mesh.

Latitude–longitude artwork is imported once. The initial outline, cratons and curved partition use the earlier initialization algorithm at an import resolution determined by the greater of `mesh_level` and `coast_geometry_level`. The latter is independently selectable (2–6, default 4). Extra conforming refinement concentrates around transitions while rigid interiors and the ocean control mesh stay simpler. Subsequent evolution does not use the earlier raster transport. A different geometry level can change initial sampling and the resulting history. Independence from **review-map size** is not independence from **tectonic resolution**.

Initial material geometry now follows shared contours through triangle interiors. Joint owner, crust-type and craton-identity indicators are sampled from the starting artwork; conforming refinement adds smaller triangles near transitions. The complete initial land/ocean partition is checked for closed shared edges and spherical area closure before retaining its buoyant material. This avoids snapping coastlines and cratons to whole control triangles. It adds material faces while keeping the ocean control-cell budget unchanged. It is initialization refinement, not adaptive remeshing during evolution.

Connected material heights are reconstructed through area-weighted values on actual shared vertices of the same owner. This removes artificial constant-height triangular facets; separate sheets and plate owners are never averaged together. Stored per-face columns and cumulative geological budgets remain unchanged. The reconstruction is bounded and continuous across shared edges, but is not a conservative redistribution of column volume. New epochs record `surface_reconstruction_version=1`; older epochs retain their original sampling.

New epochs also record `continental_margin_version=1` and the saved `continental_margin_parameters`. A spherical distance profile inside the true continental footprint produces an ocean-depth toe, slope, submerged shelf and coastal transition. Its nominal width is 150 km and shelf depth 180 m; widths are bounded for small disconnected fragments. It does not extend material across a real gap or add/remove column mass. This is a versioned surface reconstruction, not a flexural or sedimentary margin model. Highest-contained continental exposure avoids selecting a low buried sheet under a higher one. Older histories keep their original interpretation unless explicitly upgraded for a derived terrain product.

Physical boundary directions and lengths now come from dominant-pair equal-support contours in the native mesh. Third-owner scores clip triple junctions. Existing contact keys still connect the two adjacent control faces; reconstructed lengths are apportioned among those keys and unresolved contacts retain an explicit fallback. The same corrected geometry feeds relative-motion classification and the inherited force equations. The browser draws saved spherical contour segments directly, including subduction teeth, instead of tracing review-pixel edges. This removes mesh-axis bias without adding a new driving force. Contours remain piecewise curves at finite resolution; this is not a fully deformable fault network.

New epochs also retain compact native vertex ownership scores (`mesh_owner_slots`, `mesh_vertex_support`, `owner_reconstruction_version=1`). Ocean plate colours and native goSPL queries use the same interpolated ownership. Domain names come from nearby actual connected domains; where no local winning cell exists the display records domain 0 and an unresolved-domain diagnostic rather than borrowing a remote parent name. Connected material islands retain their exact material domains.

## Resolution and cost

The default level is 4. There are `20 × 4^level` control faces and `10 × 4^level + 2` vertices. The scale below is the square root of mean face area, not a minimum feature size or the length of every triangular edge.

| UI choice | Level | Control faces | Vertices | Mean area scale |
| --- | ---: | ---: | ---: | ---: |
| Geometry check | 2 | 320 | 162 | about 1,263 km |
| Quick draft | 3 | 1,280 | 642 | about 631 km |
| Balanced | 4 | 5,120 | 2,562 | about 316 km |
| Detailed | 5 | 20,480 | 10,242 | about 158 km |
| Advanced | 6 | 81,920 | 40,962 | about 79 km |

These budgets are separate from continental mechanics-node detail, the review raster and the detailed terrain image. Tiny material triangles can survive at coarse control resolution, but that resolution still limits their dynamically resolved neighboring boundaries. Increasing mechanics nodes cannot create finer control edges.

On September 6, 2026, exact coverage was measured on this laptop using a rigidly rotated material subset covering roughly 40% of the initial mesh. Conservative Cartesian candidate bins and vectorized spherical clipping gave:

| Control faces | Moving material faces in fixture | Nonzero intersections | One coverage calculation |
| ---: | ---: | ---: | ---: |
| 1,280 | 516 | 2,628 | 0.22 s |
| 5,120 | 2,038 | 10,300 | 0.77 s |
| 20,480 | 8,176 | 41,116 | 3.43 s |

These are helper timings, not whole-step or billion-year runtime estimates. More arcs, overlapping sheets, boundaries and active plates add work. Cached areas can be reused while only column properties change; moving vertices requires new coverage. At level 5, a separate benchmark built control geometry in 0.071 s, built its point locator in 0.354 s and located 18,432 points in 0.472 s. Other laptop activity affects timings.

Search and clipping use bounded temporary buffers and sparse candidate pairs instead of an all-material-by-all-control matrix. Storage still grows with native resolution, material history and saved-frame count. A 1,000 Myr history saved every 2 Myr has 501 scheduled epochs. More powerful hardware permits higher budgets; the application does not claim that every setting is equally practical on this laptop. Use its measured elapsed time and ETA for the actual experiment.

## Ocean motion, creation and breakup

The primordial ocean begins as one plate with ordinary Euler motion. Its identity no longer exempts it from breakup. Pure ocean does not need continental parcels before it can qualify for an oceanic fracture.

Ocean fields move through conservative finite-volume fluxes across native edges. The integrated rigid-motion flux comes from edge endpoints; the fluxes telescope around a closed triangle, preserving a uniform field. Internal CFL subdivisions keep upwind transport bounded without changing the requested 2 Myr history cadence. Fractional support permits movement smaller than a cell.

Thermal age, primordial fraction, damage, weakness and tensile-exposure history travel as support-weighted material quantities. Damage is not a permanent timer on a stationary geographic edge. Mixed cells carry averaged properties; they are not an exact catalogue of every oceanic rock's ancestry.

Existing relative-motion and finite half-stage spreading rules now use native geometry. An active ridge reconstructs a shared fractional front and adds paired strips from measured opening and segment length. True third-owner neighborhoods block inappropriate reconstruction at junctions. Both admitted sides remain paired when a junction clips available area. Only the produced fraction receives juvenile age and loses inherited ocean damage/primordial provenance. A stationary ridge axis can still separate its two plates.

Local mature subduction preferentially removes incoming overlapping support. Consumption is distinct from unresolved fractional overlap and contact normalization. Initial or quiet contact does not automatically inherit a mature slab. Finite-volume transport is monotone and conservative before these creation/contact operations, but its first-order upwind form diffuses sharp ocean-property transitions.

Ocean breakup uses incompatible boundary loading and the retained sparse mechanical response. Cooling age affects relative strength; inherited weakness and damage can localize deformation. **Age, size or elapsed time alone supplies no fracture load.** Positive resolved extension can accumulate damage; a failure corridor must still divide viable, connected daughters. Ocean cuts are checked against actual buoyant triangles, including small islands hidden by coarse classification. Ocean labels cannot license a cut through protected continental material.

Daughter rotations are fitted from the resolved response and inherit the parent's mantle state. There is no newly added random opening kick or automatic rule that chops a large ocean into a fixed number of plates. Without suitable loading, a primordial plate may survive for a long time.

## Geological processes retained

The mantle-force implementation and coefficients are inherited. Slab pull, ridge push, collision resistance and deep-craton drag still determine the reduced motion response alongside that retained mantle contribution. Native areas and boundary geometry change the inputs, so preserving equations does not make a new run retrace an old raster history.

Continental shortening and extension evolve the retained crustal columns. Continuous spherical belts apply forcing at material locations; a pixel blur does not define their width. Positive areal strain controls thinning, so tensile shear alone is not charged as area expansion. Cratons, sutures, thermal support and accumulated damage affect progressive rifting. Corridors may stall, cool, reactivate or break through; a developing basin does not automatically become a new plate.

The column model retains cooling subsidence, foreland deflection, erosion with rebound, magmatic construction and rift inversion. Native column strain now comes from the actual change in spherical triangle area. Stretching thins the column; shortening thickens it; a rigid stage or area-preserving shear adds no geometric thinning. Geometry is limited before moving if it would invert triangles or violate column thickness limits. Rift inversion identifies compression of an inherited basin without applying a second independent thickening. Opening a fault changes connectivity; its subsequent physical opening supplies thinning.

Trench chains use shared vertices of native boundary edges. Persistent local identities carry initiation, maturity, quieting, shutdown and restart. Subduction troughs, 180 km arc offsets, trench retreat, ridge-subduction/slab-window episodes and existing arc-production rates remain in the process path, retaining the earlier parameterized assumptions.

Juvenile additions create connected oval patches with a core, flanks, and a submerged apron. Their shared triangles support deformation and adaptive refinement. Lateral growth conserves old physical column volume and mixes an explicit juvenile magma source, rather than copying a thick mountain root. Stable arc identities and material charts survive growth and refinement. Budgets below 0.001 km², and currently inadmissible growth, remain owner-attached until representable. See [ARC_REPAIR.md](ARC_REPAIR.md) for contact-aware loading, surface exposure, source versions, and validation.

Local accretion follows connected material contact. Back-arc slivers use spherical trench geometry and native adjacency to require a connected child and viable remaining parent. Cratons and actual buoyant material retain the relevant fracture protections. Existing collision events, sutures, thermal episodes and material counters continue into saved history.

## Review, orientation and restart

Each mesh epoch includes control vertices/faces, native cell fields, boundary edges/polarity, and moving material vertices/faces. Material records preserve face IDs, owner, type, reference area, columns, rift memory and available domain identity. Ocean records include age, damage, weakness and primordial fraction. Review rasters and representative markers remain available to the browser and exports.

Native arrays have independent shapes. `mesh_faces` indexes `mesh_vertices`; `material_faces` indexes `material_vertices`. Match unchanged faces using `material_face_id`, not array rows. Under subdivision/coarsening, `material_root_id`, `material_parent_id`, and homogeneous `material_reference_corners` preserve material ancestry. Native boundary edges index control vertices. JSON supplies time and process records. Do not reshape native geometry to displayed map dimensions.

Whole-globe orientation rotates coordinates and corresponding spatial/Euler records. Triangle indices, scalar ages and accumulated counters stay the same. This repositions the recorded world; it does not introduce another tectonic force.

Pause/resume retains the application's checkpoint workflow: finish the integration step, store tagged arrays and random-generator state, then continue with matching engine/helper sources and NumPy version. An old raster checkpoint belongs to its original raster engine. Opening or exporting its recorded history does not convert it into a mesh simulation.

The goSPL interface retains evolving-history forcing and numerical elevation exports. Versioned material transport follows exact reference-chart correspondence through deformed and subdivided triangles. Its vertical forcing compares material column heights and cumulative net erosion, so changing mesh resolution alone cannot become tectonic uplift. The continuous terrain reconstruction is used for terrain, with its difference from raw column changes reported separately. Ocean transport uses recorded Euler motions. Older histories retain their original adapter path. Native geometry does not supply missing landscape physics or a ready-made ROCKE-3D climate setup.

## Remaining work and validation limits

The material layer now combines **rigid blocks and deforming belts**. Every vertex incident to a craton and every interior vertex outside the active belts follows its plate's exact Euler rotation. Owner-matched boundary motion supplies a finite-width residual velocity field, smoothed on the connected spherical material graph. Each shared vertex has one position; separated faults retain independent sides. Independent active regions use bounded integration and geometric backtracking. The existing plate-force equations, including mantle contribution, are retained.

This takes inspiration from the rigid blocks and triangulated deforming networks described in the [GPlates deformation primer](https://www.gplates.org/docs/pygplates/pygplates_primer#deformation). It is a bounded kinematic closure driven by this model's plate motions, not GPlates reconstruction data or a full nonlinear continuum stress solver. Interior damage still controls fracture localization. A future extension can couple the interior mechanical response directly into the deformation field, beyond the current boundary belts.

Adaptive material refinement concentrates triangles around deformation and damaged continental regions, under a chosen subdivision limit and face budget. Shared-edge refinement is conforming; cratons may receive boundary subdivisions without changing their rigid geometry. Coarsening only restores registered quiet sibling families whose shape and geological histories can be preserved. It deliberately refuses to flatten a deformed boundary or average away differing histories. Connected juvenile arc patches participate in refinement; legacy private triangles retain their earlier representation. The ocean control mesh remains at the selected tectonic level; its finite-volume CFL subdivisions and the deformation substeps adapt integration in time.

Control mesh resolution still limits boundary detection and ocean properties; one exposed owner must be chosen for control-cell classification. Exact area coverage treats tiny islands proportionately but does not resolve every subcell contact or eliminate overlap ambiguity. Ocean diffusion and finite-resolution boundary detection remain numerical limitations. Refinement preserves geometry rather than inventing a smoother coastline. More triangle detail cannot itself establish more accurate physics.

An 8K or 16K procedural heightmap adds visual detail conditioned by its source epoch. It does not create finer tectonic physics, river routing, sediment transport or a tracked history for each fine ridge. Use the recorded tectonic timeline when asking why a feature formed.

Focused tests cover manifold topology, exact global area, pole/seam rotation, true gaps/overlaps, conservative transport, paired ridge birth, juvenile area, native trench/back-arc connectivity and coverage conservation. Success establishes those tested behaviors, not validation of every process through 1,000 Myr at every level. **[VALIDATION.md](VALIDATION.md)** is the record for completed integrated runs and remaining checks.

The raster application and earlier histories remain separate. Historical test totals and long-run timings in inherited documents describe their original source versions; they are not evidence for this edition's complete behavior.
