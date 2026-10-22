"""Stage 5: strategic triangle placement in an otherwise quad-dominant mesh.

Implements the per-quad quality-gate mechanism from the design doc's
"Strategic Triangle Placement" table: a quad whose shape has been distorted
past recovery by strict all-quad conformity (an acute/obtuse corner, a sliver
edge, a near-degenerate area) is deterministically split into two triangles
along its shorter diagonal, rather than left in the mesh as a bad element.

This directly targets two of the table's three documented triggers:
- "Conical & Tapered Extremities": quads collapsing toward a point (caught by
  the min-edge-ratio and corner-angle checks below)
- "Acute Feature-Curve Corners": quads pinched at a sharp corner (caught by
  the corner-angle checks)

The third trigger ("Rapid 2:1 Density Step-Down / T-Junction Resolution")
requires re-structuring a whole patch's subdivision -- a global operation on
the Bi-MDF quantization, not a local per-quad fix -- and is NOT implemented
here; see the module-level note in pipeline.py for why a specific over-
tessellation symptom (small decorative features forcing a locked crease loop,
which cross-field alignment then spirals quads toward) is a Stage 2/4
interaction, not something this local pass can resolve.
"""
from __future__ import annotations

import numpy as np

from .config import PipelineConfig
from .metrics import scaled_jacobian_quad


def _quad_corner_angles_deg(verts: np.ndarray, quad: np.ndarray) -> list[float]:
    p = verts[quad]
    angles = []
    for i in range(4):
        prev_v = p[i - 1] - p[i]
        next_v = p[(i + 1) % 4] - p[i]
        denom = np.linalg.norm(prev_v) * np.linalg.norm(next_v)
        cos_a = np.dot(prev_v, next_v) / denom if denom > 1e-12 else 1.0
        angles.append(float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))))
    return angles


def _quad_edge_lengths(verts: np.ndarray, quad: np.ndarray) -> list[float]:
    p = verts[quad]
    return [float(np.linalg.norm(p[i] - p[(i + 1) % 4])) for i in range(4)]


def _badness_score(verts: np.ndarray, quad: np.ndarray,
                    min_angle_deg: float, max_angle_deg: float,
                    min_edge_ratio: float, min_scaled_jacobian: float) -> float:
    """0 = perfectly fine quad; higher = worse. Used to prioritize which
    quads get converted first if a triangle-fraction budget is exhausted."""
    angles = _quad_corner_angles_deg(verts, quad)
    edges = _quad_edge_lengths(verts, quad)
    mean_edge = sum(edges) / 4.0
    min_edge = min(edges)
    sj = scaled_jacobian_quad(verts, quad)

    score = 0.0
    if min(angles) < min_angle_deg:
        score += (min_angle_deg - min(angles))
    if max(angles) > max_angle_deg:
        score += (max(angles) - max_angle_deg)
    if mean_edge > 1e-12 and (min_edge / mean_edge) < min_edge_ratio:
        score += (min_edge_ratio - min_edge / mean_edge) * 100
    if sj < min_scaled_jacobian:
        score += (min_scaled_jacobian - sj) * 100
    return score


def _split_along_shorter_diagonal(quad: np.ndarray, verts: np.ndarray) -> np.ndarray:
    a, b, c, d = quad
    d02 = np.linalg.norm(verts[a] - verts[c])
    d13 = np.linalg.norm(verts[b] - verts[d])
    if d02 <= d13:
        return np.array([[a, b, c], [a, c, d]])
    return np.array([[b, c, d], [b, d, a]])


def apply_triangle_transitions(
    vertices: np.ndarray, quads: np.ndarray, config: PipelineConfig,
    min_angle_deg: float = 20.0, max_angle_deg: float = 160.0,
    min_edge_ratio: float = 0.15, min_scaled_jacobian: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (remaining_quads, new_triangles). Converts the worst-quality
    quads into triangle pairs, up to config.profile.triangle_fraction_max of
    the mesh (prioritizing the worst first), always converting quads that are
    truly degenerate (near-zero Scaled Jacobian) regardless of the budget --
    a genuinely broken element should never be left in the output."""
    n = len(quads)
    budget = max(1, int(n * config.profile.triangle_fraction_max))

    scored = []
    for i, q in enumerate(quads):
        score = _badness_score(vertices, q, min_angle_deg, max_angle_deg,
                                min_edge_ratio, min_scaled_jacobian)
        sj = scaled_jacobian_quad(vertices, q)
        if score > 0:
            scored.append((score, i, sj))

    # Always convert truly degenerate quads (sj <= 0), even over budget.
    forced = [i for (score, i, sj) in scored if sj <= 1e-6]
    scored.sort(key=lambda t: -t[0])
    chosen = set(forced)
    for score, i, sj in scored:
        if len(chosen) >= budget:
            break
        chosen.add(i)

    remaining_quads = []
    triangles = []
    for i, q in enumerate(quads):
        if i in chosen:
            triangles.extend(_split_along_shorter_diagonal(q, vertices).tolist())
        else:
            remaining_quads.append(q.tolist())

    return (np.array(remaining_quads, dtype=np.int64) if remaining_quads else np.zeros((0, 4), dtype=np.int64),
            np.array(triangles, dtype=np.int64) if triangles else np.zeros((0, 3), dtype=np.int64))
