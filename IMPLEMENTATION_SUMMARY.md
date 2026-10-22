# Feature-Aware Quad-Dominant Retopology — Implementation Summary

A detailed record of what has been built, why, and what was learned along the way. See
`README.md` for a terser status summary and `/Users/bhargavreddy/.claude/plans/users-bhargavreddy-downloads-3d-researc-cheerful-lemur.md`
for the original and follow-up implementation plans.

## 1. Goal and architecture

Convert an arbitrary, unstructured triangle `.glb` (e.g. a raw AI-generated or scanned asset)
into a **feature-aware, quad-dominant `.glb`** — clean edge loops that lock onto real creases,
with quad density that adapts to local detail (small quads on complex regions, large quads on
flat ones), matching the quality bar of commercial tools like Tripo3D's "AI Quad Remesher" /
Meshy's "Remesh" (mesh-to-mesh retopology — **not** Tripo's "Smart Mesh" native generation,
which synthesizes geometry from an image/text prompt with no input mesh at all).

The pipeline follows a 6-stage architecture:

1. **Ingestion & sanitization** — load, weld, repair, decimate, build a high-poly BVH reference.
2. **Feature/crease graph extraction** — locate hard creases to lock during remeshing.
3. **Adaptive sizing & pre-conditioning** — grade target quad size by local curvature.
4. **Cross-field computation** (QuadWild) — a smooth 4-RoSy orientation field respecting creases.
5. **Bi-MDF patch quantization** (libSatsuma) — the actual global quad extraction, plus a local
   triangle-insertion pass for quads too distorted to keep.
6. **Relaxation, projection, export** — snap the result back onto the true surface and write `.glb`.

**The core architectural decision, made early and validated throughout**: use the real
QuadWild + Bi-MDF (libSatsuma) global solver as the quad-extraction engine, not a local-field
method like Instant Meshes. Prior prototyping in this project (see `docs/reference/`) always
silently fell back to Instant Meshes because QuadWild was never actually built/installed —
which defeats the point, since Instant Meshes is exactly the kind of local optimizer that
produces 200+ singularities and swirling artifacts on flat regions. This project's
`src/retopo/crossfield_backend.py` has exactly **one** registered backend and raises
`BackendUnavailableError` loudly if QuadWild is missing; it must never silently substitute a
different solver.

## 2. Getting the core solver working (Milestone 0)

QuadWild + Bi-MDF is a C++ research tool, not a pip package. The single highest-risk step was
getting it to actually run on this machine, before investing in any surrounding Python.

- Found that `cgg-bern/quadwild-bimdf` (the fork the project's design doc cites via
  `mlstr0m/requad`) ships **prebuilt macOS universal binaries (arm64+x86_64)** as a GitHub
  release asset (`v0.0.2`, `macos-binaries.zip`, ~5 MB) — verified directly on GitHub before
  committing to this path, avoiding a risky from-source C++ build.
- Downloaded, confirmed they're legitimate Mach-O universal binaries with no quarantine flag,
  and smoke-tested the exact CLI contract:
  ```
  quadwild path/to/mesh.obj 2 config/prep_config/basic_setup.txt        # -> mesh_rem_p0.obj + feature/patch files
  quad_from_patches path/to/mesh_rem_p0.obj 123 config/main_config/flow_noalign_lemon.txt   # -> quantized quad mesh
  ```
  This uses the Bi-MDF/lemon solver by default — fully open-source, no Gurobi dependency,
  matching the project's GPL-3.0 constraint.
- Binaries + provenance live in `third_party/quadwild_bimdf/` (see its `PROVENANCE.md`), kept
  entirely outside the Python venv and invoked via `subprocess`.

**First real finding**: feeding the raw koala test asset straight to QuadWild produced a badly
cracked quadrangulation (~30% boundary/crack edges) even though QuadWild itself worked
correctly. Root cause: the koala's raw `.glb` has duplicate, unwelded vertices across glTF
primitive boundaries (each primitive ships its own private vertex buffer), which trimesh's
*default* `merge_vertices()` doesn't fix because it refuses to merge vertices with different
UVs/normals even at identical positions. Fixed in `sanitize.weld_vertices` by passing
`merge_tex=True, merge_norm=True` — this collapsed what looked like 851 disconnected "bodies"
down to the true ~3 physically separate pieces. This is the single most important lesson from
this phase: **sanitization isn't optional polish, it's a hard prerequisite for QuadWild to
produce a correct result.**

## 3. Module-by-module implementation (`src/retopo/`)

| Module | Responsibility |
|---|---|
| `config.py` | Single source of truth for every threshold. `PipelineConfig` + `CategoryProfile` (ORGANIC/UTENSIL/HARD_SURFACE presets for crease angle, sizing ratio, Lipschitz gradient, triangle budget). |
| `sanitize.py` | Load, weld (UV/normal-insensitive), remove degenerate faces, discard debris components, repair non-manifold geometry (pymeshlab), QEM-decimate oversized inputs, build a `HighPolyReference` BVH for later projection. |
| `curvature.py` | `bilateral_normal_filter` (vectorized face-normal denoising, suppresses triangulation-jitter false-positive creases) + `filtered_dihedral_angles_deg` + `estimate_vertex_curvature` (libigl `principal_curvature` wrapper — compiled, not a from-scratch per-vertex Python loop). |
| `feature_graph.py` | Standalone dihedral crease detection for QA/diagnostics (the actual enforcement happens inside the QuadWild binary via the generated prep config). |
| `defeature.py` | Detects isolated feature regions too small for even the finest sizing tier to resolve (true sub-resolution noise) and cuts them out, re-triangulating the hole flush with pymeshlab. Tracks each removed feature's true apex (`_feature_peak_point`) for optional recovery. |
| `sizing_field.py` | The adaptive density engine — see §5 below. `compute_relative_sizing_field` (curvature-driven, Lipschitz-smoothed, dimensionless) + `precondition_mesh_graded` (tiered selective remeshing via pymeshlab). |
| `crossfield_backend.py` | `QuadWildBackend` — subprocess wrapper for the `quadwild` binary (Stage 4). Exactly one backend, fails loudly if missing. |
| `quantization.py` | Subprocess wrapper for `quad_from_patches` (Stage 5 Bi-MDF quantization), parses the resulting quad mesh. |
| `triangle_transitions.py` | Per-quad quality gate: quads too distorted for strict all-quad conformity (acute corners, slivers, near-zero area) get converted to a triangle pair, up to a per-category budget, with genuinely degenerate quads (Scaled Jacobian ≤ 0) always converted regardless of budget. |
| `relax_project.py` | `project_to_surface` (BVH nearest-point snap onto the sanitized high-poly surface) + `restore_feature_peaks` (optional: snap the nearest coarse vertex onto a defeatured feature's true peak). |
| `metrics.py` | `weld_quad_mesh`, `vertex_valence`/`singularity_count`, `scaled_jacobian_quad`, `chamfer_distance` — used by both the pipeline and the test suite. |
| `io_glb.py` | GLB export using the quad-as-two-triangles convention (shared diagonal) so DCC tools can losslessly recover quads via Tris-to-Quads. |
| `pipeline.py` | `run_pipeline()` — the **only** place stage ordering lives. Orchestrates sanitize → defeature → sizing field → calibration loop → QuadWild/Bi-MDF → weld/project → triangle transitions → export. |

## 4. Category-aware feature locking (Stage 2)

Fixed a real bug where the bundled `basic_setup_Organic.txt` QuadWild prep config disabled
sharp-edge detection entirely (`sharp_feature_thr -1`), silently dropping all feature locking
for organic assets regardless of what the project's own `CategoryProfile.sharp_dihedral_deg`
said. `config.write_prep_config` now generates the prep config file dynamically from the
active profile every run, so the crease threshold actually used always matches what the config
claims.

## 5. Real, controllable adaptive density (Stage 3) — the biggest piece of work

### 5.1 What was wrong

The first working version of Stage 3 used pymeshlab's `meshing_isotropic_explicit_remeshing`
with `adaptive=True` and a single scalar `targetlen`. This looked reasonable, but direct
measurement (and later, live introspection of the pymeshlab filter itself) showed:

- `adaptive=True` is a **fixed, un-tunable on/off toggle** into VCGlib's internal curvature
  heuristic — confirmed via `pymeshlab.print_filter_parameter_list` and `strings` on the
  compiled filter plugin binary. There is no exposed parameter for how aggressively it should
  grade size by curvature.
- Four `CategoryProfile` fields that looked like they should control this —`sizing_ratio`,
  `lipschitz_k`, `max_anisotropy`, `ridge_percentile` — were **100% dead code**: defined with
  different per-category values, never read anywhere in `src/retopo/` or `tests/`. Leftovers
  from an earlier, abandoned from-scratch sizing-field design.
- Measured on real output: a genuine but weak (~6x) and **locally chaotic** curvature-to-area
  correlation — quad areas spanning 3 orders of magnitude within a single small feature with
  no smooth grading, while flat regions stayed nearly uniform instead of grading up to the big
  flat quads a Tripo-style result shows.

### 5.2 The fix

- **`curvature.estimate_vertex_curvature`**: a thin, vectorized wrapper over
  `igl.principal_curvature` (compiled C++, not a Python per-vertex loop).
- **`sizing_field.compute_relative_sizing_field`**: a real, dimensionless (mean ≈ 1) per-vertex
  field —
  1. Chordal/sagitta ideal size from curvature: `iota(v) = sqrt(8·eps_tol / (curv_mag + eps))`.
  2. Clamp to `[h_min, h_max] = [mean_h/sqrt(ratio), mean_h·sqrt(ratio)]` using the now-**revived**
     `CategoryProfile.sizing_ratio`.
  3. Lipschitz-smooth via a **vectorized multi-pass edge relaxation** (`h = min(h, h_neighbor +
     lipschitz_k · edge_length)`, iterated to convergence) — not the Dijkstra-based approach from
     the archived design, which would be both slow at scale and seeded from an arbitrary
     percentile of vertices.
- **`sizing_field.precondition_mesh_graded`**: discretizes `[h_min, h_max]` into
  `config.sizing_tiers` (default 4) geometrically-spaced tiers and, **finest tier first**,
  selects the matching vertices (`pymeshlab.compute_selection_by_scalar_per_vertex`, fed by a
  per-vertex quality array set via `v_scalar_array`), dilates the selection by one ring to blend
  tier boundaries, and remeshes *only that selection* to its own explicit target length
  (`selectedonly=True`). Verified empirically (not just assumed from docs) that pymeshlab
  correctly carries quality values through to newly-created vertices after a selective remesh
  pass (0.9998 correlation in a controlled test), so curvature only needs to be computed once
  and feeds every tier.
- **`pipeline.py`'s calibration loop** now scales one dimensionless multiplier (`lambda_scale`)
  uniformly across the whole pre-shaped field to hit `target_faces`, rather than one flat edge
  length — the field's *shape* (how much bigger flat quads are than detail quads) is fixed per
  category and never touched by calibration, only its absolute size is.

### 5.3 Measurement, not eyeballing

Built `tools/measure_sizing_contrast.py` to quantify this directly instead of relying on
wireframe screenshots: Spearman correlation between curvature and log-area, a flat-vs-detail
decile area ratio, and a **local coherence** metric (each quad's area ratio to its immediate
neighbors' median — the old approach's chaos vs. the new approach's smoothness, numerically).

Results on the wooden-rack asset: Spearman ρ=-0.78, flat-vs-detail contrast **14.3x**, local
coherence **1.003** (vs. the old approach's locally chaotic 27.5 — i.e. now smooth, not jagged).

### 5.4 A hypothesis tested and rejected

Hypothesized that QuadWild/Bi-MDF's `regularityNonQuadrilateralsWeight=0.9` (identical across
all 12 bundled main configs) might homogenize patch sizes and mute the sizing field's contrast
in the final quad mesh. Created a `flow_noalign_lemon_lowreg.txt` variant (weight 0.3) and
measured it directly: contrast got **worse** (3.76x → 1.80x on the koala), not better. Kept the
default config; the lowreg file remains only as a documented negative result.

### 5.5 A real pymeshlab instability, found and fixed

On the wooden rack (dense feature-locking from real CAD-like creases), certain absolute tier
target lengths made pymeshlab's selective remesh land in a reproducible "stuck" state —
triangle count going *up* despite requesting a *coarser* target, with many different `lambda`
values producing the exact same anomalous output (e.g. several different scales all landing on
precisely 50,550 faces). Ruled out `checksurfdist`, `featuredeg` (feature-locking), and
iteration count as causes via direct controlled experiments on the actual asset; root cause is
internal to pymeshlab's remesher. The koala (organic, lighter feature-locking) never showed this
at all — it's specific to heavily-creased meshes.

Fix: `pipeline.py`'s calibration loop now detects this symptom (triangle count rising despite a
larger requested scale) *before* the expensive QuadWild call, and retries with a small random
perturbation rather than trusting an anomalous attempt to drive the next correction.

### 5.6 The actual knob fix: `sizing_ratio` was miscalibrated

Even after all of the above, a wooden-rack drawer knob (visually pointed out by the user) still
wasn't resolving as a clean round shape. The precise cause, computed directly rather than
guessed: at `HARD_SURFACE.sizing_ratio=8` (the design-doc-inherited default), the *finest* tier
the field could produce (`h_min ≈ 0.031` model units) was barely the same size as the knob
itself (`≈0.033`) — physically, no more than ~1 quad could ever fit across it, regardless of how
smooth the surrounding grading was. Raising `HARD_SURFACE.sizing_ratio` to **80** (verified both
visually — the knob now renders as a clean multi-quad dome with visibly larger flat-panel quads
around it — and via the measurement tool) fixed this directly. `ORGANIC`/`UTENSIL` were raised
proportionally less (**30**/**25**) since organic characters generally don't have as extreme a
tiny-detail-on-a-huge-body scale gap. The diagnostic for re-tuning this on a future asset is
documented in `config.py`'s `CategoryProfile` docstring.

## 6. Defeaturing saga (small-feature handling)

Three distinct phases, each with a real bug found and fixed:

1. **First attempt**: restricted Laplacian smoothing of a crease loop's vertices plus a small
   ring buffer. Measurably failed — crease-edge count was unchanged after smoothing — because a
   small buffer doesn't reach far enough to cover a bump's whole raised disc, only its rim.
2. **Second attempt**: cut the feature's face region out entirely and patch the hole
   (`pymeshlab.meshing_close_holes`) — this worked for removing the feature, but `restore_feature_peaks`
   (snapping the nearest coarse vertex onto the feature's recorded peak) initially picked the
   *wrong* peak. Two bugs here, both confirmed on the wooden-rack asset and regression-tested in
   `tests/test_defeature.py`:
   - Picked the vertex with the largest **raw 3D distance** from the boundary ring's centroid,
     which can select a lateral point on a wider base flange instead of the actual summit —
     fixed by measuring distance **perpendicular to the ring's own fitted plane** instead.
   - The boundary was computed against the **global** combined removal mask (all small features
     at once), which is wrong whenever a feature is itself surrounded by *other* small features
     rather than the main body directly — a knob's raised dome turned out to be topologically
     separated from the main panel by an intermediate *ring* component (a second, smaller crease
     loop around the dome's base), so against the global mask the dome's boundary came back
     empty. Fixed by computing the boundary **per-component**.
3. **Final state, after the sizing-ratio fix (§5.6)**: with real per-category `sizing_ratio`
   values, `find_small_feature_faces`'s threshold changed from "smaller than the old single
   global edge length" to **"smaller than `defeature_max_extent_factor × h_min`"** — i.e. only
   cuts out features too small for even the finest tier to resolve. This means genuinely
   important small details (like the knob, once `sizing_ratio` was fixed) are now resolved
   directly by the sizing field and never need to be cut out and spike-recovered at all.
   `restore_defeatured_peaks` now defaults to **off**: since defeaturing only catches true noise
   now, marking its location with a visible spike would reintroduce exactly the artifact being
   suppressed.

## 7. Stage 5 triangle transitions

Implements 2 of the design doc's 3 documented triangle-insertion triggers ("conical/tapered
extremities", "acute feature-curve corners") as a per-quad quality gate: quads with bad corner
angles, sliver edges, or near-zero area get split into a triangle pair along their shorter
diagonal, up to `profile.triangle_fraction_max`, with genuinely degenerate quads (Scaled
Jacobian ≤ 0) always converted regardless of budget. The third trigger ("rapid 2:1 density
step-down", which would require restructuring a whole T-mesh patch's subdivision) isn't
implemented — in practice it's no longer needed now that the graded sizing field resolves small
features directly instead of relying on a post-hoc local fix.

Verified on the wooden rack: eliminated a fully degenerate quad (min Scaled Jacobian 0.0 → a
healthy positive value) and raised mean Scaled Jacobian.

## 8. Viewer (backend + frontend)

A FastAPI server (`backend/app.py`) + Three.js single-page app (`frontend/`) wired directly to
`retopo.pipeline.run_pipeline` — no mock/fallback remesher. Endpoints:

- `GET /api/samples` — lists preset `.glb` files.
- `POST /api/preview` — sanitize-only load, for the left ("Input Mesh") viewport.
- `POST /api/remesh` — runs the real pipeline, returns both meshes (as flat vertex/quad/triangle
  arrays plus per-quad Scaled Jacobian and per-vertex valence) and a quality report.
- `GET /api/download/{run_id}` — serves the result as `.glb` or `.obj`.

Frontend shading modes: Clay Studio, Face Classifier (quad vs. triangle, color-coded), Scaled
Jacobian heatmap, and Valence/Singularities (color-coded by vertex valence vs. the regular
value of 4).

**Controls added this session** (to let the sizing-field/defeature work above actually be
tested interactively, rather than only via Python scripts):
- **Sizing Ratio slider** — directly exercises `CategoryProfile.sizing_ratio` per-run, with
  profile-aware defaults (30/25/80) that auto-update when the Category Profile dropdown changes,
  mirroring `config.py`'s actual values.
- **Defeature checkbox** — toggles `PipelineConfig.defeature` per-run.
- **"Features Defeatured" KPI card** — shows the count of features removed as sub-resolution
  noise, plus the sizing ratio actually used for that run.

Backend changes: `/api/remesh` gained `sizing_ratio: float | None` (overrides the selected
profile's default when provided) and `defeature: bool` form parameters, both wired into the
`PipelineConfig` construction via `dataclasses.replace`; the report now includes `sizing_ratio`,
`defeature_enabled`, and `n_features_defeatured`.

Verified end-to-end in the browser (not just assumed from code review): selecting "Hard
Surface" auto-set the Sizing Ratio slider to 80 and crease angle to 25°; running the wooden-rack
sample with `target_faces=3000` produced 22,086 quads / 3,840 triangles, mean Scaled Jacobian
0.831, with the KPI panel correctly reporting "Sizing Ratio: 80" and "32" features defeatured.

A `.claude/launch.json` was added so the backend can be started via the standard dev-server
tooling (`uvicorn backend.app:app --port 8420`).

## 9. Testing

9 tests across 5 files, all passing:

- `tests/test_sanitize.py` — koala sanitizes to a small number of real components (regression
  test for the UV/normal-insensitive welding fix in §2).
- `tests/test_pipeline_e2e.py` — full koala GLB→GLB round trip, asserts singularity rate stays
  well under what a local-field solver would produce, mean Scaled Jacobian, Chamfer distance.
- `tests/test_triangle_transitions.py` — good quads left untouched, degenerate quads always
  converted even over budget, budget is respected otherwise.
- `tests/test_defeature.py` — regression test for the peak-detection plane-fit bug (§6).
- `tests/test_sizing_field.py` — on a synthetic bumpy sphere: the field is smaller near the
  bump than far from it, has no discontinuous jumps between adjacent vertices, and
  `precondition_mesh_graded` produces measurably smaller faces near the bump.

## 10. Known limitations (honest, not hidden)

- **Hitting an exact `target_faces` count has a practical floor**, independent of all the sizing
  work above. On both test assets, final output runs well above the requested target (e.g.
  ~18-22k quads for a 3,000-quad request on the wooden rack). This is driven by patch *count*
  from cross-field singularities and feature complexity, not patch size — the sizing field
  doesn't directly control how many patches QuadWild's cross-field tracing produces.
  `pipeline.py`'s plateau-detection stops calibration early rather than thrashing once further
  coarsening stops reducing quad count.
- Full design-doc Stage 6 (explicit corner/crease/interior vertex classification with
  crease-tangent-locked relaxation) is simplified to plain nearest-point BVH projection.
- The third triangle-insertion trigger ("rapid 2:1 density step-down", a global T-mesh patch
  restructuring operation) is not implemented, per §7.
- `backend/app.py`'s `/api/remesh` lets you choose a profile and override its sizing ratio, but
  there's no automatic category *detection* from the uploaded asset — the user has to pick the
  right profile themselves.

## 11. File map

```
third_party/quadwild_bimdf/   # prebuilt QuadWild+Bi-MDF binaries (GPL-3.0), see PROVENANCE.md
docs/reference/                 # archived prior prototyping, kept for reference, never imported
src/retopo/                     # the pipeline itself (see §3 table above)
backend/app.py                  # FastAPI viewer server
frontend/                       # Three.js viewer UI
tools/                          # asset export, QuadWild smoke test, visual QA renders,
                                 # measure_sizing_contrast.py
tests/                          # pytest suite (§9)
.claude/launch.json             # dev-server config for the backend
```
