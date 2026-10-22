"""Stage 1: ingestion & sanitization.

Raw AI-generated .glb meshes are frequently NOT a single clean manifold surface:
each glTF primitive typically ships its own private vertex buffer, so vertices
that are supposed to coincide across primitive boundaries are numerically
distinct until welded. Left unwelded, a mesh like this can look like hundreds
of disconnected "bodies" to naive connected-component analysis, and -- as
confirmed empirically against the koala test asset during the M0 spike --
feeding such a mesh straight into QuadWild produces a badly cracked
quadrangulation (~30% boundary edges) even though QuadWild itself is working
correctly. Welding + debris removal is therefore a hard prerequisite, not an
optional cleanup pass.
"""
from __future__ import annotations

import numpy as np
import pymeshlab
import trimesh

from .config import PipelineConfig


def weld_vertices(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Merge coincident vertices by position, ignoring UV/normal differences.

    This is the single most important sanitization step for raw AI-generated
    .glb files: each glTF primitive ships its own private vertex buffer, so
    vertices that are meant to coincide across primitive boundaries carry
    distinct (but identical-position) entries, frequently with slightly
    different UVs or normals. trimesh's default merge_vertices() treats a UV
    or normal difference as a reason NOT to merge, which leaves these
    boundaries un-welded (confirmed: on the koala asset, default settings left
    851 fake "bodies" merged down to only 142, whereas merge_tex=True,
    merge_norm=True collapses it to the true ~3 physically-disconnected
    components). Leaving this unwelded produces a badly cracked QuadWild
    quantization (~30% boundary edges) even though QuadWild itself works
    correctly -- confirmed empirically during the M0 spike.
    """
    mesh = mesh.copy()
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    return mesh


def remove_degenerate_faces(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def discard_debris(mesh: trimesh.Trimesh, area_frac: float) -> trimesh.Trimesh:
    """Split into connected components and drop any component whose surface
    area is below `area_frac` of the total -- floating specks, stray debris
    from AI generation, etc."""
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh
    areas = np.array([c.area for c in components])
    total = areas.sum()
    kept = [c for c, a in zip(components, areas) if a > area_frac * total]
    if not kept:
        kept = [components[int(np.argmax(areas))]]
    return trimesh.util.concatenate(kept)


def repair_non_manifold(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Repair non-manifold edges/vertices via pymeshlab, since trimesh itself
    has no equivalent repair primitive."""
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))
    try:
        ms.meshing_repair_non_manifold_edges()
    except Exception:
        pass
    try:
        ms.meshing_repair_non_manifold_vertices()
    except Exception:
        pass
    try:
        ms.meshing_remove_duplicate_vertices()
    except Exception:
        pass
    m = ms.current_mesh()
    return trimesh.Trimesh(vertices=m.vertex_matrix(), faces=m.face_matrix(), process=False)


def decimate_qem(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Quadric-error-metric decimation of oversized inputs, preserving an
    unmodified high-poly copy for Stage 6 projection (caller's responsibility
    to keep the original around)."""
    if len(mesh.faces) <= target_faces:
        return mesh
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_faces,
        preserveboundary=True,
        preservenormal=True,
    )
    m = ms.current_mesh()
    return trimesh.Trimesh(vertices=m.vertex_matrix(), faces=m.face_matrix(), process=False)


class HighPolyReference:
    """Holds the original (post-sanitization, pre-decimation) high-poly mesh
    for Stage 6 BVH shrinkwrap projection. Uses igl.point_mesh_squared_distance
    directly rather than a persistent igl.AABB object -- at the scale of a
    single-asset retopology run this is simple and fast enough (KISS)."""

    def __init__(self, mesh: trimesh.Trimesh):
        self.vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
        self.faces = np.ascontiguousarray(mesh.faces, dtype=np.int64)

    def closest_points(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (squared_distances, face_indices, closest_points) for each
        row of `query`."""
        import igl

        query = np.ascontiguousarray(query, dtype=np.float64)
        sqd, face_idx, closest = igl.point_mesh_squared_distance(
            query, self.vertices, self.faces
        )
        return sqd, face_idx, closest


def load_and_sanitize(path: str, config: PipelineConfig) -> tuple[trimesh.Trimesh, HighPolyReference]:
    """Full Stage 1 pipeline: load -> weld -> remove degenerate -> discard
    debris -> repair non-manifold -> (optional decimate). Returns the
    sanitized working mesh plus a HighPolyReference snapshot taken *before*
    decimation, for later shrinkwrap projection."""
    scene = trimesh.load(path, force="scene")
    geoms = [g for g in scene.geometry.values() if hasattr(g, "vertices") and len(g.faces) > 0]
    if not geoms:
        raise ValueError(f"No geometry found in {path}")
    mesh = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]

    mesh = weld_vertices(mesh)
    mesh = remove_degenerate_faces(mesh)
    mesh = discard_debris(mesh, config.debris_area_frac)
    mesh = repair_non_manifold(mesh)
    mesh = weld_vertices(mesh)  # repair can reintroduce duplicates

    highpoly = HighPolyReference(mesh)

    if len(mesh.faces) > config.max_tris_before_decimate:
        mesh = decimate_qem(mesh, config.decimate_target_tris)

    return mesh, highpoly
