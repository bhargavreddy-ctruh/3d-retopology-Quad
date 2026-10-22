"""Stage 2: feature/crease detection.

The actual feature-locking enforcement in v1 happens inside the QuadWild
binary itself (it performs its own dihedral-angle crease detection and
cross-field alignment during Stage 4, using the `sharp_feature_thr` value we
now generate per-CategoryProfile via config.write_prep_config -- see
crossfield_backend.py). This module exists to (a) make that threshold
decision inspectable/testable independent of the binary, and (b) support
visual QA by overlaying detected creases on the sanitized mesh.

A from-scratch multi-modal feature graph (crest-line curvature ridges, UV/
material seam detection, spur-pruning graph cleanup -- as in the design doc
and docs/reference/blueprint_pipeline_reference.py) is deferred: QuadWild's
own dihedral-based detection, driven by our per-category threshold, already
locks the same visually-dominant creases (collar/cuff/lapel-equivalent edges)
that matter most for the Tripo-quality bar.
"""
from __future__ import annotations

import numpy as np
import trimesh


def dihedral_crease_edges(mesh: trimesh.Trimesh, threshold_deg: float) -> np.ndarray:
    """Returns the (E, 2) vertex-index array of edges whose dihedral angle
    exceeds `threshold_deg`, using trimesh's face-adjacency angle computation."""
    angles_deg = np.degrees(mesh.face_adjacency_angles)
    mask = angles_deg > threshold_deg
    return mesh.face_adjacency_edges[mask]


def crease_fraction(mesh: trimesh.Trimesh, threshold_deg: float) -> float:
    """Fraction of all interior (non-boundary) edges flagged as creases --
    used by CategoryProfile auto-detection heuristics and sanity checks (a
    sane organic character should lock a small minority of edges, not most
    of the mesh)."""
    n_interior_edges = len(mesh.face_adjacency_edges)
    if n_interior_edges == 0:
        return 0.0
    return len(dihedral_crease_edges(mesh, threshold_deg)) / n_interior_edges
