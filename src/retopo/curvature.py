"""Bilateral normal filtering for feature detection (Stage 2 prerequisite).

Raw per-face dihedral angles on an AI-generated/scanned mesh are noisy:
marching-cubes-style triangulation jitter produces many small, locally
inconsistent face-normal perturbations that individually exceed a sharp-angle
threshold even though the underlying surface is smooth there. The design doc
calls this out explicitly ("triangulation jitter triggers false-positive sharp
edges") and specifies bilateral normal filtering as the fix before any
dihedral-angle crease detection runs.

Confirmed empirically on the wooden-rack asset: raw (unfiltered) dihedral
detection at 25 degrees produces several 300-700-vertex tangled crease
networks with 20+ branch points sprawled across what should be flat drawer
panels -- not plausible real CAD features, a clear noise signature. This
module exists to fix that at the source, rather than trying to filter the
resulting messy feature graph after the fact.
"""
from __future__ import annotations

import numpy as np
import trimesh


def bilateral_normal_filter(mesh: trimesh.Trimesh, iters: int = 3,
                             sigma_spatial: float | None = None,
                             sigma_range: float = 0.35) -> np.ndarray:
    """Returns a (F, 3) array of filtered face normals. Vectorized across all
    face-adjacency pairs per iteration (no per-face Python loop), since
    sanitized inputs can have 10^5 faces."""
    faces_adj = mesh.face_adjacency
    centers = mesh.triangles_center
    normals = mesh.face_normals.copy()
    i_idx, j_idx = faces_adj[:, 0], faces_adj[:, 1]
    n_faces = len(mesh.faces)

    if sigma_spatial is None:
        sigma_spatial = float(np.mean(mesh.edges_unique_length) * 2.5)
    sigma_s2 = 2.0 * sigma_spatial ** 2
    sigma_r2 = 2.0 * sigma_range ** 2

    d_spatial_sq = np.sum((centers[i_idx] - centers[j_idx]) ** 2, axis=1)
    spatial_w = np.exp(-d_spatial_sq / sigma_s2)  # fixed across iterations

    for _ in range(iters):
        d_range_sq = np.sum((normals[i_idx] - normals[j_idx]) ** 2, axis=1)
        w = spatial_w * np.exp(-d_range_sq / sigma_r2)

        sum_n = np.zeros((n_faces, 3))
        sum_w = np.zeros(n_faces)
        np.add.at(sum_n, i_idx, w[:, None] * normals[j_idx])
        np.add.at(sum_w, i_idx, w)
        np.add.at(sum_n, j_idx, w[:, None] * normals[i_idx])
        np.add.at(sum_w, j_idx, w)

        mask = sum_w > 1e-12
        new_normals = normals.copy()
        new_normals[mask] = sum_n[mask] / sum_w[mask, None]
        norms = np.linalg.norm(new_normals, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        normals = new_normals / norms

    return normals


def filtered_dihedral_angles_deg(mesh: trimesh.Trimesh, filtered_normals: np.ndarray) -> np.ndarray:
    """Dihedral angle (degrees, unsigned) between each adjacent face pair,
    recomputed from filtered normals instead of trimesh's raw
    face_adjacency_angles."""
    i_idx, j_idx = mesh.face_adjacency[:, 0], mesh.face_adjacency[:, 1]
    cos_a = np.sum(filtered_normals[i_idx] * filtered_normals[j_idx], axis=1)
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def estimate_vertex_curvature(mesh: trimesh.Trimesh, radius: int = 5) -> np.ndarray:
    """Per-vertex curvature magnitude, via libigl's compiled (vectorized)
    principal-curvature estimator -- not a from-scratch per-vertex quadric
    fit (the archived docs/reference/blueprint_modules_reference.py's
    compute_discrete_curvature_tensor does that with a Python
    `for i in range(num_v)` loop, which this codebase's convention avoids;
    see bilateral_normal_filter's docstring).

    Returns kmax_abs = max(|PV1|, |PV2|) per vertex: the larger-magnitude
    principal curvature, used as the drive signal for Stage 3's sizing field
    (small values -> flat -> large quads; large values -> detailed -> small
    quads)."""
    import igl

    pd1, pd2, pv1, pv2, bad = igl.principal_curvature(
        np.ascontiguousarray(mesh.vertices, dtype=np.float64),
        np.ascontiguousarray(mesh.faces, dtype=np.int64),
        radius=radius,
    )
    kmax_abs = np.maximum(np.abs(pv1), np.abs(pv2))
    if bad:
        # "Bad" vertices (degenerate neighborhoods) get no reliable curvature
        # signal; fall back to the mesh-wide median rather than 0 (which
        # would misread as "perfectly flat" and request oversized quads).
        kmax_abs[bad] = np.median(np.delete(kmax_abs, bad)) if len(bad) < len(kmax_abs) else 0.0
    return kmax_abs
