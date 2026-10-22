"""Pre-Stage-3 defeaturing: suppress decorative details too small to matter
at the target mesh resolution -- the "shirt button" scenario from the design
doc, generalized.

Root cause confirmed empirically on the wooden-rack asset: a small closed
crease loop (e.g. a drawer-pull knob's rim) sitting in the interior of what
would otherwise be one large flat patch forces Bi-MDF's per-patch UNIFORM
grid density up across the *entire* enclosing patch, not just locally around
the loop -- QuadWild/Bi-MDF quantizes each T-mesh patch as a single regular
M x N grid; it does not locally grade density within a patch. Post-hoc local
fixes (Stage 5's triangle transitions) can't undo this: by the time Stage 5
sees the quad mesh, the whole patch has already been laid out at the fine
resolution the loop demanded.

Implementation approach (matches docs/reference's parts.py segmentation
pattern -- connected components of the FACE dual graph split at creases,
rather than vertex-ring smoothing): a first attempt at this used restricted
Laplacian smoothing of the crease-loop vertices plus a small ring buffer, but
that measurably failed to reduce the dihedral signal (confirmed empirically:
crease-edge count was unchanged after smoothing) because a small buffer does
not reach far enough to cover the knob's whole raised disc, only its rim.
Instead: cut the small isolated face region out entirely and re-triangulate
the resulting hole flush with the surrounding surface via pymeshlab -- this
removes the crease loop and the bump geometry together, rather than trying to
locally flatten a feature whose true extent is a surface *region*, not just
its boundary curve.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pymeshlab
import trimesh

from .config import PipelineConfig
from .curvature import bilateral_normal_filter, filtered_dihedral_angles_deg
from .sizing_field import estimate_initial_edge_length


def find_small_feature_faces(
    mesh: trimesh.Trimesh, config: PipelineConfig, max_extent_factor: float = 2.5,
) -> list[np.ndarray]:
    """Returns a list of face-index arrays, one per small isolated feature
    region (e.g. a decorative knob) -- connected components of the mesh's
    face dual graph after cutting along bilateral-filtered dihedral creases,
    excluding whichever component is largest (the main body)."""
    filt_normals = bilateral_normal_filter(mesh, iters=3)
    angles = filtered_dihedral_angles_deg(mesh, filt_normals)
    is_crease = angles > config.profile.sharp_dihedral_deg

    face_adj = mesh.face_adjacency[~is_crease]
    G = nx.Graph()
    G.add_nodes_from(range(len(mesh.faces)))
    G.add_edges_from(map(tuple, face_adj))
    components = list(nx.connected_components(G))
    if len(components) <= 1:
        return []

    components.sort(key=len, reverse=True)
    main_body = components[0]
    # Threshold relative to the sizing field's FINEST tier (h_min), not the
    # old single global edge length -- a feature only gets defeatured if even
    # the finest tier can't resolve it with a handful of quads, i.e. it's
    # genuine noise, not an important small detail the graded sizing field
    # should otherwise handle directly (see config.py's defeature docstring).
    mean_h = estimate_initial_edge_length(mesh, config.target_faces)
    h_min = mean_h / np.sqrt(config.profile.sizing_ratio)
    max_extent = max_extent_factor * h_min

    small_feature_faces = []
    for comp in components[1:]:
        face_idx = np.array(list(comp), dtype=np.int64)
        verts_idx = np.unique(mesh.faces[face_idx])
        pts = mesh.vertices[verts_idx]
        extent = float(np.linalg.norm(pts.max(0) - pts.min(0)))
        if extent < max_extent:
            small_feature_faces.append(face_idx)

    return small_feature_faces


def _feature_peak_point(mesh: trimesh.Trimesh, face_idx: np.ndarray) -> np.ndarray:
    """The feature's own apex: the vertex (on the ORIGINAL, pre-removal mesh)
    farthest OUT OF THE LOCAL BASE PLANE formed by the feature's boundary
    ring -- i.e. the point that sticks out the most from where the bump
    attaches to the surrounding surface. Used later to re-insert a vertex at
    the true peak after the coarse mesh has otherwise smoothed/flattened it
    away.

    NOTE on two bugs found and fixed here:
    1. An earlier version picked the vertex with the largest raw 3D Euclidean
       distance from the boundary ring's centroid. That is wrong whenever a
       feature's footprint isn't perfectly round (e.g. a wider base flange):
       a lateral point on the flange can be farther from the centroid than
       the dome's actual apex. Fixed by measuring distance perpendicular to
       the ring's own fitted plane instead.
    2. The boundary was computed against the GLOBAL combined `remove_mask`
       (all small features at once), which is wrong whenever a feature is
       itself topologically surrounded by OTHER small features rather than
       directly by the main body -- confirmed empirically: a knob's raised
       dome is its own connected component, separated from the main body by
       an intermediate RING component (a second, smaller crease loop around
       the dome's base), so against the global mask the dome's boundary was
       empty and the code fell back to treating the whole dome blob as its
       own "boundary", which skews the fitted plane enough to misidentify
       the apex. Fixed by computing the boundary per-component: any vertex
       shared with a face outside THIS component's own face_idx (whether
       that neighboring face belongs to the main body or another small
       feature) correctly finds the real surrounding ring.
    """
    removed_verts = np.unique(mesh.faces[face_idx])
    this_component_mask = np.zeros(len(mesh.faces), dtype=bool)
    this_component_mask[face_idx] = True
    verts_used_elsewhere = np.unique(mesh.faces[~this_component_mask])
    boundary = np.intersect1d(removed_verts, verts_used_elsewhere, assume_unique=False)
    if len(boundary) < 3:
        boundary = removed_verts  # fallback: shouldn't happen for a true isolated bump

    boundary_pts = mesh.vertices[boundary]
    centroid = boundary_pts.mean(axis=0)
    # Fit the boundary ring's local plane via SVD; its normal is the
    # smallest-variance direction.
    _, _, Vt = np.linalg.svd(boundary_pts - centroid)
    normal = Vt[2]

    removed_pts = mesh.vertices[removed_verts]
    out_of_plane = np.abs((removed_pts - centroid) @ normal)
    peak_vertex = removed_verts[np.argmax(out_of_plane)]
    return mesh.vertices[peak_vertex].copy()


def defeature_small_bumps(
    mesh: trimesh.Trimesh, config: PipelineConfig, max_extent_factor: float = 2.5,
) -> tuple[trimesh.Trimesh, int, np.ndarray]:
    """Cuts out small isolated feature regions (decorative bumps/knobs below
    the target resolution) and re-triangulates the resulting holes flush with
    the surrounding surface. Returns (patched_mesh, n_features_removed,
    peak_points) where peak_points is (n_features_removed, 3) -- each
    feature's own apex point, for Stage 6's bump-recovery step (see
    relax_project.restore_feature_peaks)."""
    small_faces = find_small_feature_faces(mesh, config, max_extent_factor)
    if not small_faces:
        return mesh, 0, np.zeros((0, 3))

    remove_mask = np.zeros(len(mesh.faces), dtype=bool)
    for face_idx in small_faces:
        remove_mask[face_idx] = True

    peak_points = np.array(
        [_feature_peak_point(mesh, face_idx) for face_idx in small_faces]
    )

    kept = mesh.copy()
    kept.update_faces(~remove_mask)
    kept.remove_unreferenced_vertices()

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=kept.vertices, face_matrix=kept.faces))
    # Hole size is bounded by how many boundary edges a removed region leaves
    # behind; a generous cap keeps this robust without needing to size it
    # exactly per-feature.
    ms.meshing_close_holes(maxholesize=500, refinehole=True)
    m = ms.current_mesh()
    patched = trimesh.Trimesh(vertices=m.vertex_matrix(), faces=m.face_matrix(), process=False)
    return patched, len(small_faces), peak_points
