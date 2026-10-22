"""Quality metrics used both by tests and for visual-QA reporting."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def weld_quad_mesh(vertices: np.ndarray, quads: np.ndarray, tol: float = 1e-7
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Collapse exactly-coincident vertex duplicates (rounding-based; QuadWild's
    own output is already well-welded on sanitized input -- this is a cheap
    safety net, not a crack-closer for unsanitized input)."""
    keys: dict[tuple[float, float, float], int] = {}
    remap = np.empty(len(vertices), dtype=np.int64)
    kept: list[np.ndarray] = []
    inv_tol = 1.0 / tol
    for i, v in enumerate(vertices):
        key = (round(v[0] * inv_tol), round(v[1] * inv_tol), round(v[2] * inv_tol))
        idx = keys.get(key)
        if idx is None:
            idx = len(kept)
            keys[key] = idx
            kept.append(v)
        remap[i] = idx
    new_vertices = np.array(kept, dtype=vertices.dtype)
    new_quads = remap[quads]
    return new_vertices, new_quads


def vertex_valence(quads: np.ndarray, n_vertices: int) -> np.ndarray:
    valence = np.zeros(n_vertices, dtype=np.int64)
    for q in quads:
        for v in set(q.tolist()):
            valence[v] += 1
    return valence


def singularity_count(quads: np.ndarray, n_vertices: int) -> dict:
    valence = vertex_valence(quads, n_vertices)
    hist: dict[int, int] = defaultdict(int)
    for v in valence:
        hist[int(v)] += 1
    regular = hist.get(4, 0)
    irregular = n_vertices - regular
    return {
        "histogram": dict(sorted(hist.items())),
        "regular": regular,
        "irregular": irregular,
        "irregular_frac": irregular / max(1, n_vertices),
    }


def scaled_jacobian_quad(verts: np.ndarray, quad: np.ndarray) -> float:
    """Minimum Scaled Jacobian over the quad's 4 corners (standard quad-quality
    metric, range roughly [-1, 1]; 1 = perfect square, <=0 = degenerate)."""
    p = verts[quad]
    sjs = []
    n = 4
    for i in range(n):
        prev_v = p[i] - p[(i - 1) % n]
        next_v = p[(i + 1) % n] - p[i]
        ln = np.linalg.norm(prev_v) * np.linalg.norm(next_v)
        if ln < 1e-12:
            sjs.append(0.0)
            continue
        cross = np.cross(prev_v, next_v)
        sjs.append(float(np.linalg.norm(cross) / ln) * np.sign(np.dot(cross, cross) + 1))
    return float(min(sjs))


def mean_min_scaled_jacobian(verts: np.ndarray, quads: np.ndarray) -> tuple[float, float]:
    sjs = [scaled_jacobian_quad(verts, q) for q in quads]
    return float(np.mean(sjs)), float(np.min(sjs))


def chamfer_distance(points: np.ndarray, b_vertices: np.ndarray, b_faces: np.ndarray) -> float:
    """One-directional mean distance from a set of query `points` (e.g. sampled
    from mesh A's surface) to the surface (b_vertices, b_faces)."""
    import igl

    sqd, _, _ = igl.point_mesh_squared_distance(
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(b_vertices, dtype=np.float64),
        np.ascontiguousarray(b_faces, dtype=np.int64),
    )
    return float(np.sqrt(sqd).mean())
