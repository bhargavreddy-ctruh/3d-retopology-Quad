"""End-to-end orchestrator. This is the ONLY place stage ordering lives.

Stage 1 (sanitize) -> defeature (sub-h_min noise only) -> Stage 3 (graded,
curvature-adaptive pre-conditioning via a tiered sizing field, calibrated to
hit target_faces) -> Stage 4+5 (QuadWild + Bi-MDF, do_remesh=0 since Stage 3
already sized the mesh) -> Stage 6-lite (weld + BVH project) -> GLB export.

The calibration loop adjusts one scalar, `lambda_scale`, which uniformly
scales the whole pre-shaped sizing field; it never changes the field's shape
(how much bigger flat-region quads are than detail-region quads, fixed by
`config.profile.sizing_ratio`) -- only its absolute size, to hit
`target_faces`. This exists because the pre-remesh-scale -> final-quad-count
relationship is not perfectly linear (Bi-MDF's integer arc quantization
introduces its own rounding); see sizing_field.py's module docstring for the
empirical investigation behind both this design and the tiered sizing field
itself.
"""
from __future__ import annotations

import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import io_glb, metrics
from .config import PipelineConfig, write_prep_config
from .crossfield_backend import QuadWildBackend
from .defeature import defeature_small_bumps
from .quantization import QuadMeshResult, run_bimdf_quantization
from .relax_project import project_to_surface, restore_feature_peaks
from .sanitize import load_and_sanitize
from .sizing_field import (
    compute_relative_sizing_field,
    estimate_initial_edge_length,
    precondition_mesh_graded,
)
from .triangle_transitions import apply_triangle_transitions


@dataclass
class PipelineResult:
    output_path: str
    n_vertices: int
    n_quads: int
    n_triangles: int
    singularity_stats: dict
    mean_scaled_jacobian: float
    min_scaled_jacobian: float
    chamfer_mean: float
    stats: dict = field(default_factory=dict)
    vertices: object = None    # np.ndarray, (N, 3) -- exposed for callers that
    quads: object = None       # np.ndarray, (Q, 4)    need the raw mesh (e.g.
    triangles: object = None   # np.ndarray, (T, 3)    a UI), not just the glb


def _run_quadwild_once(obj_path: Path, workdir: Path, config: PipelineConfig,
                        do_remesh: bool) -> QuadMeshResult:
    backend = QuadWildBackend(config)
    prep_config = write_prep_config(
        workdir / "prep_config.txt",
        do_remesh=do_remesh,
        sharp_feature_thr=config.profile.sharp_dihedral_deg,
        alpha=config.quadwild_alpha,
    )
    stage4 = backend.run(obj_path, workdir, prep_config=prep_config)
    return run_bimdf_quantization(stage4, config, workdir)


def run_pipeline(input_glb: str, output_glb: str, config: PipelineConfig | None = None
                  ) -> PipelineResult:
    config = config or PipelineConfig()

    # Stage 1
    mesh, highpoly = load_and_sanitize(input_glb, config)

    # Pre-Stage-3 defeaturing: suppress decorative details too small for even
    # the finest sizing tier to resolve (genuine noise, not important detail
    # -- see config.py's defeature docstring and defeature.py's module
    # docstring for why this still has to happen before Stage 3 rather than
    # relying on the graded sizing field alone: locking a sub-h_min loop as a
    # hard crease still forces Bi-MDF's uniform per-patch grid density up
    # across the whole enclosing patch, regardless of how well-graded the
    # surrounding triangle mesh is).
    n_features_removed = 0
    feature_peaks = None
    if config.defeature:
        mesh, n_features_removed, feature_peaks = defeature_small_bumps(
            mesh, config, max_extent_factor=config.defeature_max_extent_factor)

    # Stage 3 sizing field: computed once, on the final pre-Stage-3 topology
    # (after defeaturing, which changes vertex/face counts). Dimensionless
    # (mean ~= 1) -- the calibration loop below scales it uniformly via
    # `lambda_scale` to hit target_faces, without ever recomputing curvature
    # or changing the field's *shape* (how much bigger flat-region quads are
    # than detail-region quads, fixed by config.profile.sizing_ratio).
    relative_field = compute_relative_sizing_field(mesh, config)

    calibration_log = []
    quad_result: QuadMeshResult | None = None
    mean_h = estimate_initial_edge_length(mesh, config.target_faces)
    final_edge_length = mean_h

    if config.precondition:
        lambda_scale = 1.0
        best = None
        prev_lambda = None
        prev_pre_tris = None
        for attempt in range(config.calibration_max_iters):
            preconditioned = precondition_mesh_graded(mesh, relative_field, lambda_scale, config)
            pre_tris = len(preconditioned.faces)

            # pymeshlab's tiered selective remesh can land in a "stuck" state
            # for certain absolute target lengths on complex/heavily-creased
            # meshes: triangle count goes UP despite requesting a coarser
            # (larger) target length -- confirmed empirically on the
            # wooden-rack asset (NOT caused by feature-locking or the
            # surface-deviation check; both ruled out by direct testing).
            # Root cause is internal to pymeshlab's remesher, not fixable
            # from here. Retry with a small perturbation (cheap, pymeshlab
            # only, before the expensive QuadWild call) rather than trusting
            # an anomalous attempt to drive the next lambda correction.
            retry_budget = 3
            while (prev_lambda is not None and lambda_scale > prev_lambda
                   and pre_tris > prev_pre_tris and retry_budget > 0):
                lambda_scale *= random.uniform(1.05, 1.25)
                preconditioned = precondition_mesh_graded(mesh, relative_field, lambda_scale, config)
                pre_tris = len(preconditioned.faces)
                retry_budget -= 1

            with tempfile.TemporaryDirectory(prefix="retopo_") as tmp:
                workdir = Path(tmp)
                obj_path = workdir / "input.obj"
                preconditioned.export(obj_path)
                quad_result = _run_quadwild_once(obj_path, workdir, config, do_remesh=False)

            n_quads = len(quad_result.quads)
            calibration_log.append({"attempt": attempt, "lambda_scale": lambda_scale,
                                     "pre_tris": pre_tris, "n_quads": n_quads})
            if best is None or abs(n_quads - config.target_faces) < abs(best[0] - config.target_faces):
                best = (n_quads, quad_result, lambda_scale)
            prev_lambda, prev_pre_tris = lambda_scale, pre_tris

            deviation = abs(n_quads - config.target_faces) / config.target_faces
            if deviation <= config.calibration_tolerance:
                break

            prev_n_quads = calibration_log[-2]["n_quads"] if len(calibration_log) > 1 else None
            if prev_n_quads is not None and abs(n_quads - prev_n_quads) / max(1, prev_n_quads) < 0.05:
                # Plateaued: further coarsening isn't reducing quad count, which
                # happens when output size is dominated by patch COUNT (driven by
                # cross-field singularities/feature complexity) rather than the
                # pre-remesh edge length. Stop wasting iterations chasing a floor
                # we can't move with this knob; use the best attempt so far.
                calibration_log.append({"note": "plateaued, stopping calibration early"})
                break
            if attempt == config.calibration_max_iters - 1:
                break
            # quad count scales ~ 1/edge_length^2; adjust proportionally
            lambda_scale *= (n_quads / config.target_faces) ** 0.5
        n_quads, quad_result, lambda_scale = best
        final_edge_length = lambda_scale * mean_h
    else:
        with tempfile.TemporaryDirectory(prefix="retopo_") as tmp:
            workdir = Path(tmp)
            obj_path = workdir / "input.obj"
            mesh.export(obj_path)
            quad_result = _run_quadwild_once(obj_path, workdir, config, do_remesh=True)

    assert quad_result is not None

    # Stage 6 (lite): weld + project back onto the sanitized high-poly surface
    vertices, quads = metrics.weld_quad_mesh(quad_result.vertices, quad_result.quads)
    vertices = project_to_surface(vertices, highpoly)

    # Optional bump recovery (default off): with the graded sizing field,
    # anything still defeatured should be true sub-h_min noise, so marking
    # its location with a visible spike would reintroduce the exact artifact
    # being suppressed -- see config.py's restore_defeatured_peaks docstring.
    if config.restore_defeatured_peaks and feature_peaks is not None and len(feature_peaks):
        vertices = restore_feature_peaks(vertices, feature_peaks, max_snap_dist=final_edge_length)

    # Singularity count reflects the cross-field/Bi-MDF solution quality
    # (Stage 4/5), so it's measured BEFORE the triangle-transition post-pass:
    # converting a quad to 2 triangles reduces its corner vertices' quad-
    # valence by construction, which would otherwise look like a regression
    # in solver quality when it's actually a deliberate, separate step.
    sing = metrics.singularity_count(quads, len(vertices))

    # Stage 5: strategic triangle placement -- convert genuinely distorted/
    # degenerate quads (acute corners, slivers, conical pinches) into triangles
    # rather than leaving them in the output.
    quads, triangles = apply_triangle_transitions(
        vertices, quads, config, min_scaled_jacobian=config.triangle_min_scaled_jacobian)

    io_glb.export_quad_glb(vertices, quads, output_glb, triangles=triangles)

    mean_sj, min_sj = metrics.mean_min_scaled_jacobian(vertices, quads) if len(quads) else (1.0, 1.0)
    chamfer = metrics.chamfer_distance(vertices, highpoly.vertices, highpoly.faces)

    return PipelineResult(
        output_path=output_glb,
        n_vertices=len(vertices),
        n_quads=len(quads),
        n_triangles=len(triangles),
        singularity_stats=sing,
        mean_scaled_jacobian=mean_sj,
        min_scaled_jacobian=min_sj,
        chamfer_mean=chamfer,
        stats={"source_highpoly_faces": len(highpoly.faces), "calibration_log": calibration_log,
               "n_features_defeatured": n_features_removed},
        vertices=vertices,
        quads=quads,
        triangles=triangles,
    )
