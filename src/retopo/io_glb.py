"""GLB/OBJ I/O helpers, including the quad-as-two-triangles export convention
used by commercial tools (Tripo/Meshy) so DCC software can losslessly recover
quads via Tris-to-Quads (Alt+J in Blender)."""
from __future__ import annotations

import numpy as np
import trimesh


def export_obj_debug(vertices: np.ndarray, quads: np.ndarray, path: str) -> None:
    """Write a plain quad OBJ (1-indexed), useful for debugging/inspection
    with tools that understand n-gon faces directly."""
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for q in quads:
            f.write("f " + " ".join(str(i + 1) for i in q) + "\n")


def quads_to_paired_triangles(quads: np.ndarray) -> np.ndarray:
    """Split each quad [a,b,c,d] into two triangles [a,b,c],[a,c,d] along a
    consistent diagonal, so downstream Tris-to-Quads reconstruction is exact."""
    tris = np.empty((len(quads) * 2, 3), dtype=quads.dtype)
    tris[0::2] = quads[:, [0, 1, 2]]
    tris[1::2] = quads[:, [0, 2, 3]]
    return tris


def export_quad_glb(vertices: np.ndarray, quads: np.ndarray, path: str,
                     triangles: np.ndarray | None = None) -> None:
    """Export a quad-dominant mesh as .glb. Quads are encoded as paired
    triangles sharing a diagonal (the standard convention); any pure
    triangles from Stage 5's triangle-transition pass are appended as-is."""
    tri_from_quads = quads_to_paired_triangles(quads) if len(quads) else np.zeros((0, 3), dtype=int)
    if triangles is not None and len(triangles):
        faces = np.vstack([tri_from_quads, triangles])
    else:
        faces = tri_from_quads
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(path)
