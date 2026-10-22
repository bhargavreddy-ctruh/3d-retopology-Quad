"""Quantify how well output quad area correlates with surface curvature --
the measurable version of "does flat get big quads and detail get small
quads, smoothly" -- rather than eyeballing wireframe renders.

Usage: measure_sizing_contrast.py <input.glb> [target_faces] [profile]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import networkx as nx
import numpy as np
from scipy.stats import spearmanr

from retopo.config import ORGANIC, HARD_SURFACE, UTENSIL, PipelineConfig
from retopo.curvature import estimate_vertex_curvature
from retopo.pipeline import run_pipeline
from retopo.sanitize import load_and_sanitize

PROFILES = {"organic": ORGANIC, "hard_surface": HARD_SURFACE, "utensil": UTENSIL}


def quad_area(verts: np.ndarray, quad: np.ndarray) -> float:
    p = verts[quad]
    a1 = 0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]))
    a2 = 0.5 * np.linalg.norm(np.cross(p[2] - p[0], p[3] - p[0]))
    return float(a1 + a2)


def quad_adjacency_graph(quads: np.ndarray) -> nx.Graph:
    edge_to_quads: dict[tuple[int, int], list[int]] = {}
    for qi, q in enumerate(quads):
        for k in range(4):
            a, b = int(q[k]), int(q[(k + 1) % 4])
            key = (min(a, b), max(a, b))
            edge_to_quads.setdefault(key, []).append(qi)
    G = nx.Graph()
    G.add_nodes_from(range(len(quads)))
    for qis in edge_to_quads.values():
        for i in range(len(qis)):
            for j in range(i + 1, len(qis)):
                G.add_edge(qis[i], qis[j])
    return G


def measure(input_glb: str, target_faces: int = 5650, profile_name: str = "organic") -> dict:
    config = PipelineConfig(target_faces=target_faces, profile=PROFILES[profile_name])
    mesh, highpoly = load_and_sanitize(input_glb, config)
    highpoly_curv = estimate_vertex_curvature(mesh)  # same mesh load_and_sanitize uses pre-decimation

    out_path = str(Path(__file__).resolve().parent / "render_work" / "_measure_tmp.glb")
    result = run_pipeline(input_glb, out_path, config)

    verts, quads = result.vertices, result.quads
    areas = np.array([quad_area(verts, q) for q in quads])
    centroids = verts[quads].mean(axis=1)

    import igl
    _, face_idx, _ = igl.point_mesh_squared_distance(
        np.ascontiguousarray(centroids, dtype=np.float64),
        np.ascontiguousarray(mesh.vertices, dtype=np.float64),
        np.ascontiguousarray(mesh.faces, dtype=np.int64),
    )
    quad_curv = highpoly_curv[mesh.faces[face_idx]].mean(axis=1)

    rho, pval = spearmanr(quad_curv, np.log(areas + 1e-12))

    order = np.argsort(quad_curv)  # ascending: order[0] = flattest, order[-1] = most detailed
    n = len(order)
    flattest_decile_areas = areas[order[: n // 10]]
    most_detailed_decile_areas = areas[order[-(n // 10):]]
    contrast_ratio = float(np.median(flattest_decile_areas) / np.median(most_detailed_decile_areas))

    G = quad_adjacency_graph(quads)
    local_ratios = []
    for qi in G.nodes:
        nbrs = list(G.neighbors(qi))
        if not nbrs:
            continue
        nbr_median = np.median(areas[nbrs])
        if nbr_median > 1e-12:
            local_ratios.append(areas[qi] / nbr_median)
    local_ratios = np.array(local_ratios)
    local_coherence = float(np.median(np.maximum(local_ratios, 1 / local_ratios)))

    return {
        "n_quads": len(quads),
        "spearman_rho": float(rho),
        "spearman_pval": float(pval),
        "flat_vs_detail_contrast": contrast_ratio,
        "local_coherence_max_over_median": local_coherence,
        "mean_scaled_jacobian": result.mean_scaled_jacobian,
        "min_scaled_jacobian": result.min_scaled_jacobian,
    }


if __name__ == "__main__":
    input_glb = sys.argv[1]
    target_faces = int(sys.argv[2]) if len(sys.argv) > 2 else 5650
    profile_name = sys.argv[3] if len(sys.argv) > 3 else "organic"
    stats = measure(input_glb, target_faces, profile_name)
    import json
    print(json.dumps(stats, indent=2))
