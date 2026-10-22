"""Stage 6 (initial version): BVH shrinkwrap projection onto the original
high-poly surface.

This is a simplified first cut: every output vertex is snapped to its nearest
point on the sanitized high-poly reference mesh via igl's point-mesh distance
query. The full design-doc treatment (classify corner/crease/interior
vertices, relax creases along their 1D feature curve, relax interior vertices
in-plane before projecting) is deferred -- see docs/reference and the design
doc's Stage 6 -- but even nearest-point projection meaningfully restores
surface fidelity lost during QuadWild's internal remeshing/decimation, and is
cheap to validate.
"""
from __future__ import annotations

import numpy as np

from .sanitize import HighPolyReference


def project_to_surface(vertices: np.ndarray, highpoly: HighPolyReference) -> np.ndarray:
    _, _, closest = highpoly.closest_points(vertices)
    return closest


def restore_feature_peaks(vertices: np.ndarray, peak_points: np.ndarray,
                           max_snap_dist: float) -> np.ndarray:
    """Bump recovery for defeatured small features (see defeature.py): a
    coarse mesh generally has no vertex deliberately targeting a removed
    feature's apex, so plain nearest-surface projection lets the bump
    disappear entirely rather than survive as a soft undulation. For each
    feature's peak point, snap the single nearest existing output vertex
    (within `max_snap_dist`) onto it -- a lightweight approximation of
    reinserting the bump: it shows up as a visible protrusion near its true
    location without any topology changes or re-triangulation.

    If no output vertex is close enough, the feature is simply too far from
    any coarse-mesh vertex to recover this way and is left flattened (rather
    than snapping something unrelated across a large distance).
    """
    if len(peak_points) == 0:
        return vertices
    vertices = vertices.copy()
    for peak in peak_points:
        dists = np.linalg.norm(vertices - peak, axis=1)
        nearest = int(np.argmin(dists))
        if dists[nearest] <= max_snap_dist:
            vertices[nearest] = peak
    return vertices
