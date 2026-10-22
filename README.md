# Feature-Aware Quad-Dominant Retopology Pipeline

Converts an arbitrary triangle `.glb` into a feature-aware quad-dominant `.glb`,
built around the real QuadWild + Bi-MDF (libSatsuma) global solver -- not a
local-field fallback like Instant Meshes. See
`/Users/bhargavreddy/.claude/plans/users-bhargavreddy-downloads-3d-researc-cheerful-lemur.md`
for the full implementation plan and rationale.

## Status

- **Working end-to-end**: Stage 1 (sanitize) -> defeature (sub-h_min noise
  only) -> Stage 3 (graded, curvature-adaptive pre-conditioning via a tiered
  sizing field, calibrated toward `target_faces`) -> Stage 4/5 (QuadWild +
  Bi-MDF quantization, real binary, not a fallback, `do_remesh=0` since
  Stage 3 already sized the mesh) -> Stage 6-lite (weld + BVH surface
  projection) -> GLB export.
- **Stage 2** (feature/crease locking) is real and wired per-category: fixed a
  bug where the bundled `basic_setup_Organic.txt` config disabled sharp-edge
  detection entirely (`sharp_feature_thr -1`); now generates the prep config
  dynamically from `PipelineConfig.profile.sharp_dihedral_deg`
  (`config.write_prep_config`).
- **Stage 3 (sizing) -- real, controllable adaptive density, replacing an
  earlier version that relied on pymeshlab's opaque `adaptive=True` flag.**
  That flag turned out to be a fixed, un-tunable heuristic (confirmed via
  live filter introspection and `strings` on the compiled plugin) producing
  weak, locally chaotic contrast (measured: ~6x global but up to 27x within
  a single small feature, with slivers). The current implementation
  (`curvature.estimate_vertex_curvature` via `igl.principal_curvature` +
  `sizing_field.compute_relative_sizing_field`, a real chordal/sagitta
  sizing field Lipschitz-smoothed via vectorized edge relaxation, fed to
  pymeshlab through `precondition_mesh_graded`'s tiered `selectedonly`
  remeshing) gives genuine, measured control:
  - `tools/measure_sizing_contrast.py` quantifies this on real output (quad
    area vs. curvature, via Spearman correlation + flat/detail decile ratio
    + local area-ratio-to-neighbors coherence). Wooden rack: Spearman
    rho=-0.78, flat-vs-detail contrast 14.3x, local coherence 1.003 (vs. the
    old approach's 27.5 -- i.e. now smooth, not chaotic).
  - **`CategoryProfile.sizing_ratio` had to be recalibrated** after
    discovering the design-doc-inherited defaults (6/5/8) were far too
    conservative for assets with a large scale gap between the whole object
    and its smallest important feature: a wooden-rack drawer knob
    (~0.033 model units) vs. `h_min` at ratio=8 (~0.031) meant the *finest*
    tier the field could produce was barely the same size as the knob
    itself -- it could never be resolved as more than a single-quad sliver,
    no matter how smooth the grading was. Raising `HARD_SURFACE.sizing_ratio`
    to 80 (verified visually and via the measurement tool) resolves the same
    knob as a clean multi-quad dome with visibly larger flat-panel quads
    around it. `ORGANIC`/`UTENSIL` were raised proportionally less (30/25)
    since organic characters generally don't have as extreme a tiny-detail-
    on-a-huge-body gap. See `config.py`'s `CategoryProfile` docstring for the
    diagnostic to use if a future asset needs this re-tuned again.
  - A **lower QuadWild `regularityNonQuadrilateralsWeight`** was tried as a
    hypothesis for improving contrast further (reasoning: a high regularity
    weight could homogenize patch sizes) -- measured to make contrast
    *worse* (3.76x -> 1.80x on the koala), not better. The default QuadWild
    main_config is correct; don't switch to the lowreg variant
    (`flow_noalign_lemon_lowreg.txt`, kept only as a documented negative
    result).
  - **A real, reproducible pymeshlab instability was found and fixed**: on
    meshes with dense feature-locking (the wooden rack), certain absolute
    tier target lengths make pymeshlab's selective remesh land in a "stuck"
    state -- triangle count going *up* despite requesting a *coarser* target,
    with many different input scales producing the exact same anomalous
    output. Ruled out `checksurfdist`, `featuredeg`, and iteration count as
    causes via direct experiment; root cause is internal to pymeshlab's
    remesher. `pipeline.py`'s calibration loop now detects this (triangle
    count rising despite a larger requested scale) and retries with a small
    perturbation -- cheap, before the expensive QuadWild call -- rather than
    trusting an anomalous attempt to drive the next correction.
- **Known limitation**: hitting an *exact* `target_faces` count has a
  practical floor, independent of the sizing-field work above -- on both
  test assets, final output runs well above the requested target (e.g.
  ~18-20k quads for a 3,000-quad request on the wooden rack). This is driven
  by patch *count* from cross-field singularities and feature complexity,
  not patch size, which the sizing field doesn't directly control.
  `pipeline.py`'s plateau-detection stops calibration early rather than
  thrashing once further coarsening stops reducing quad count.
- **Stage 5** (triangle step-down transitions) is a per-quad quality gate
  (`triangle_transitions.py`): quads distorted past recovery by strict
  all-quad conformity (acute corners, slivers, near-zero area) are converted
  to a triangle pair along their shorter diagonal, up to
  `profile.triangle_fraction_max`, with genuinely degenerate quads (Scaled
  Jacobian <= 0) always converted regardless of budget. This covers 2 of the
  design doc's 3 documented triangle-insertion triggers ("conical/tapered
  extremities", "acute feature-curve corners"); the third ("rapid 2:1
  density step-down") would require restructuring a whole T-mesh patch's
  subdivision and is NOT implemented -- not needed in practice now that the
  graded sizing field (above) resolves small features directly instead of
  relying on this post-hoc pass.
- **`defeature.py`**: cuts out isolated feature regions too small for even
  the *finest* sizing tier to resolve (`defeature_max_extent_factor * h_min`,
  not a single global edge length) -- i.e. genuine sub-resolution noise, not
  important detail (which the graded sizing field now resolves directly, so
  it no longer needs to be cut out and spike-recovered). `_feature_peak_point`
  (locates a removed feature's true apex) had two bugs found and fixed via
  the wooden-rack asset, both regression-tested in `tests/test_defeature.py`:
  (a) picked the vertex with the largest raw 3D distance from the boundary
  ring's centroid, which can pick a lateral point on a wider base flange
  instead of the actual summit -- fixed by measuring distance perpendicular
  to the ring's own fitted plane; (b) the boundary was computed against the
  *global* combined removal mask, wrong whenever a feature is itself
  surrounded by other small features (a knob's dome is topologically
  separated from the main panel by an intermediate ring component) -- fixed
  by computing the boundary per-component. `restore_defeatured_peaks`
  (visually marking a defeatured spot with a snapped vertex) now defaults to
  **off**: since defeaturing only catches true noise now, marking its
  location with a spike would reintroduce the artifact being suppressed.
- Full design-doc Stage 6 (corner/crease/interior vertex classification,
  crease-tangent-locked relaxation) is simplified to plain nearest-point
  projection for now.

## Layout

```
third_party/quadwild_bimdf/   # prebuilt QuadWild+Bi-MDF binaries (GPL-3.0, see PROVENANCE.md)
docs/reference/                # prior prototyping code kept as reference, never imported
src/retopo/                    # the pipeline itself (see each module's docstring)
backend/app.py                 # FastAPI viewer server, calls retopo.pipeline.run_pipeline directly
frontend/                      # Three.js viewer (Clay/Face-Classifier/Scaled-Jacobian/Valence shading)
tools/                         # one-off scripts (asset export, smoke test, visual QA render)
tests/                         # pytest suite
```

Note: `backend/app.py`'s `/api/remesh` endpoint does not expose a category
`profile` selector -- it always runs `ORGANIC` (`sizing_ratio=30`,
`triangle_fraction_max=0.05`, `sharp_dihedral_deg` overridable but the rest of
the profile fixed), regardless of what asset is loaded. This matters a lot
more now that `sizing_ratio` is a real, load-bearing parameter (see Status):
for hard-surface/furniture assets, `HARD_SURFACE`'s `sizing_ratio=80` can be
the difference between a small feature resolving as a clean dome vs. a sliver.
Passing `profile=HARD_SURFACE` (or exposing a profile selector in the API)
is worth doing before relying on this server for non-organic assets.

## Running

```bash
./.venv/bin/python3 -m pytest tests/ -v
./.venv/bin/python3 tools/render_wireframe.py   # visual QA render to tools/render_work/
./.venv/bin/python3 tools/measure_sizing_contrast.py <input.glb> [target_faces] [profile]
```

```python
from retopo.pipeline import run_pipeline
from retopo.config import PipelineConfig

run_pipeline("input.glb", "output.glb", PipelineConfig(target_faces=5650))
```

## Key lesson baked into this codebase

Prior prototyping (see `docs/reference/`) always silently fell back to Instant
Meshes for the core solver step because QuadWild was never actually installed.
`src/retopo/crossfield_backend.py` has exactly one backend and raises
`BackendUnavailableError` loudly if QuadWild is missing -- it must never
silently substitute a different solver.
