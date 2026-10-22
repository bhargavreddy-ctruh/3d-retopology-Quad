"""Stage 3: curvature-adaptive sizing & pre-conditioning.

History of this module's approach (kept for context, not current behavior):
v1 used pymeshlab's `meshing_isotropic_explicit_remeshing(adaptive=True)`
with a single global `targetlen`, reasoning that QuadWild's own
`alpha`/`scaleFact` prep-config knobs had been found to have zero effect on
output density (still true -- see `precondition_mesh`'s docstring below for
that specific, separate finding), and that pymeshlab's built-in curvature
heuristic would give "good enough" adaptive sizing for free.

That assumption was wrong, confirmed by direct measurement on a real asset:
`adaptive=True` is a FIXED, un-tunable on/off toggle (confirmed via
`pymeshlab.print_filter_parameter_list` and `strings` on the compiled filter
plugin -- there is no exposed curvature-sensitivity parameter). Measured
output showed real but weak (~6x) and locally INCOHERENT contrast (quad
areas spanning 3 orders of magnitude within a single small feature, with no
smooth grading), while flat regions stayed nearly uniform instead of being
graded up to genuinely large quads. There was simply no dial for "how much
bigger should flat-area quads be than detail-area quads" -- which is exactly
what a user comparing our output to a Tripo reference (dense quads on
detail, big flat quads on plain regions) needs controlled.

v2 (current): compute a real per-vertex relative sizing field from curvature
(`compute_relative_sizing_field`, see its docstring), then feed it to
pymeshlab via TIERED SELECTIVE remeshing (`precondition_mesh_graded`):
discretize the field into a handful of geometrically-spaced tiers and
remesh each tier's vertex selection to its own explicit target length via
`selectedonly=True`. This was verified (not just read from docs) to work:
pymeshlab's per-vertex "quality" scalar (stored via `v_scalar_array`) is
correctly carried through to newly-created vertices after a selective
remesh pass (confirmed empirically: 0.9998 correlation between a known
field and the carried-through quality after a test selective remesh), which
is what lets one curvature computation drive several successive tier passes
without recomputing it between them.
"""
from __future__ import annotations

import numpy as np
import pymeshlab
import trimesh

from .config import PipelineConfig
from .curvature import estimate_vertex_curvature


def estimate_initial_edge_length(mesh: trimesh.Trimesh, target_faces: int,
                                  quads_per_tri_ratio: float = 1.6) -> float:
    """Mean/baseline edge length for the given target face count.

    Empirically, the final quad count tends to run ~1.3x-2x the pre-remesh
    triangle count (patch quantization can both merge and subdivide), so we
    aim the pre-remesh triangle count at `target_faces / quads_per_tri_ratio`
    and convert that to an edge length via the equilateral-triangle area
    relation: area = sqrt(3)/4 * edge^2.
    """
    target_tris = max(50, target_faces / quads_per_tri_ratio)
    return float(np.sqrt(4 * mesh.area / (np.sqrt(3) * target_tris)))


def _lipschitz_smooth(mesh: trimesh.Trimesh, h: np.ndarray, lipschitz_k: float,
                       max_iters: int) -> np.ndarray:
    """Vectorized multi-pass edge relaxation enforcing
    `h(u) <= h(v) + lipschitz_k * edge_length(u, v)` for every mesh edge, so
    the sizing field can't jump faster than `lipschitz_k` between neighbors
    (no degenerate sliver polygons at a small-feature/flat-region boundary).

    This replaces the archived prototype's multi-source
    `scipy.sparse.csgraph.dijkstra` propagation (seeded from an arbitrary
    percentile of "most detailed" vertices): that approach is both slow at
    scale (effectively many single-source searches) and conceptually
    arbitrary (why the 15th percentile specifically). Iterative relaxation
    converges to the same kind of Lipschitz-constrained lower envelope,
    fully vectorized per mesh-wide numpy ops, capped at `max_iters` passes."""
    edges = mesh.edges_unique
    edge_lens = mesh.edges_unique_length
    i_idx, j_idx = edges[:, 0], edges[:, 1]
    h = h.copy()

    for _ in range(max_iters):
        candidate_for_i = h[j_idx] + lipschitz_k * edge_lens
        candidate_for_j = h[i_idx] + lipschitz_k * edge_lens
        new_h = h.copy()
        np.minimum.at(new_h, i_idx, candidate_for_i)
        np.minimum.at(new_h, j_idx, candidate_for_j)
        if np.allclose(new_h, h, rtol=1e-6, atol=1e-12):
            h = new_h
            break
        h = new_h

    return h


def compute_relative_sizing_field(mesh: trimesh.Trimesh, config: PipelineConfig) -> np.ndarray:
    """Dimensionless (mean ~= 1) per-vertex sizing field: small values near
    high curvature (small quads wanted), large values on flat regions (large
    quads wanted). Dimensionless so `precondition_mesh_graded`'s caller can
    scale the whole field uniformly (to hit `target_faces`) without
    recomputing curvature -- see pipeline.py's calibration loop.

    Formula matches the archived docs/reference/blueprint_modules_reference.py
    (`compute_lipschitz_sizing_field`)'s chordal/sagitta sizing math, reused
    for its formula only -- not its code or its per-vertex Python loop in the
    curvature step (replaced by the vectorized `curvature.estimate_vertex_curvature`)
    nor its multi-source Dijkstra smoothing (replaced by `_lipschitz_smooth`
    above)."""
    curv_mag = estimate_vertex_curvature(mesh)
    bbox_diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    eps_tol = max(config.sizing_eps_tol_factor * bbox_diag, 1e-6)
    eps_reg = 1e-4 / bbox_diag

    # Ideal chordal sizing: iota(v) = sqrt(8 * eps_tol / (curv_mag + eps))
    iota = np.sqrt((8.0 * eps_tol) / (curv_mag + eps_reg))

    mean_h = estimate_initial_edge_length(mesh, config.target_faces)
    ratio = config.profile.sizing_ratio
    h_min = mean_h / np.sqrt(ratio)
    h_max = mean_h * np.sqrt(ratio)

    h = np.clip(iota, h_min, h_max)
    h = _lipschitz_smooth(mesh, h, config.profile.lipschitz_k, config.lipschitz_smoothing_max_iters)
    h = np.clip(h, h_min, h_max)

    return h / np.mean(h)


def precondition_mesh_graded(mesh: trimesh.Trimesh, relative_field: np.ndarray,
                              lambda_scale: float, config: PipelineConfig,
                              iterations: int = 8) -> trimesh.Trimesh:
    """Curvature-adaptive, feature-locked isotropic remeshing, graded across
    `config.sizing_tiers` discrete target-length tiers (finest to coarsest)
    via pymeshlab's `selectedonly` remeshing -- see this module's docstring
    for why a single uniform `targetlen` (the old `precondition_mesh`,
    removed) can't deliver real, controllable size contrast.

    `lambda_scale` is the single scalar pipeline.py's calibration loop
    adjusts to hit `target_faces`; it scales the whole pre-shaped field
    uniformly, so the field's *shape* (how much bigger flat quads are than
    detail quads, i.e. `config.profile.sizing_ratio`) never changes during
    calibration -- only its absolute size does.
    """
    mean_h = estimate_initial_edge_length(mesh, config.target_faces)
    absolute_target = lambda_scale * mean_h * relative_field

    ratio = config.profile.sizing_ratio
    h_min = lambda_scale * mean_h / np.sqrt(ratio)
    h_max = lambda_scale * mean_h * np.sqrt(ratio)

    tiers = config.sizing_tiers
    tier_edges = h_min * (h_max / h_min) ** (np.arange(tiers + 1) / tiers)
    tier_edges[-1] *= 1.001  # include anything landing exactly at h_max

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(
        vertex_matrix=mesh.vertices, face_matrix=mesh.faces,
        v_scalar_array=absolute_target,
    ))

    for i in range(tiers):  # finest tier first
        lo, hi = float(tier_edges[i]), float(tier_edges[i + 1])
        tier_target = float(np.sqrt(lo * hi))  # geometric mean of the tier's bounds

        ms.compute_selection_by_scalar_per_vertex(minq=lo, maxq=hi)
        for _ in range(config.sizing_tier_overlap_rings):
            ms.apply_selection_dilatation()

        ms.meshing_isotropic_explicit_remeshing(
            iterations=iterations,
            adaptive=False,
            selectedonly=True,
            targetlen=pymeshlab.PureValue(tier_target),
            featuredeg=config.profile.sharp_dihedral_deg,
            checksurfdist=True,
            maxsurfdist=pymeshlab.PureValue(tier_target * 0.2),
        )

    m = ms.current_mesh()
    return trimesh.Trimesh(vertices=m.vertex_matrix(), faces=m.face_matrix(), process=False)
