import numpy as np
import trimesh

from retopo.config import PipelineConfig, ORGANIC
from retopo.sizing_field import compute_relative_sizing_field, precondition_mesh_graded


def _bumpy_sphere():
    """An icosphere with a small, raised bump near the north pole -- high
    local curvature there, low (uniform) curvature everywhere else, same
    shape as the real wooden-rack-knob scenario this whole sizing field was
    built to fix (a small high-curvature feature embedded in an otherwise
    smooth/flat surface). Uses a smooth (cosine-falloff) radial profile
    rather than a hard cutoff, so the bump's geometry itself doesn't have an
    artificial sharp fold at some arbitrary mask radius -- any abruptness
    measured afterward is from the sizing field's own behavior, not a
    testing artifact."""
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    verts = mesh.vertices.copy()
    north = np.array([0.0, 0.0, 1.0])
    dist_to_north = np.linalg.norm(verts - north, axis=1)
    bump_radius = 0.3
    bump_mask = dist_to_north < bump_radius
    falloff = np.zeros(len(verts))
    falloff[bump_mask] = 0.5 * (1 + np.cos(np.pi * dist_to_north[bump_mask] / bump_radius))
    radial = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    verts += radial * (falloff * 0.3)[:, None]
    return trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False), north, bump_mask


def test_sizing_field_is_smaller_near_the_bump():
    mesh, north, bump_mask = _bumpy_sphere()
    config = PipelineConfig(target_faces=1000, profile=ORGANIC)
    field = compute_relative_sizing_field(mesh, config)

    near_bump = field[bump_mask]
    far_from_bump = field[~bump_mask]
    assert near_bump.mean() < far_from_bump.mean(), (
        "sizing field should request smaller quads (lower field value) near "
        "the high-curvature bump than on the smooth rest of the sphere"
    )


def test_sizing_field_has_no_discontinuous_jumps():
    """Not an exact numeric check of the Lipschitz constant (the field is
    normalized to mean ~= 1 after smoothing, which rescales -- not breaks --
    the bound), but a sanity check that smoothing actually did something: no
    pair of mesh-adjacent vertices should differ by anywhere near the field's
    full min-max range, which an un-smoothed raw chordal-sizing field (jumping
    straight from the bump's small value to the sphere's large value with no
    gradient) would violate."""
    mesh, _, _ = _bumpy_sphere()
    config = PipelineConfig(target_faces=1000, profile=ORGANIC)
    field = compute_relative_sizing_field(mesh, config)

    edges = mesh.edges_unique
    diffs = np.abs(field[edges[:, 0]] - field[edges[:, 1]])
    field_range = field.max() - field.min()
    assert diffs.max() < 0.5 * field_range, (
        "adjacent vertices should never span a large fraction of the field's "
        "whole range -- smoothing should grade size changes over several "
        "rings of vertices, not in a single discontinuous step"
    )


def test_precondition_mesh_graded_produces_smaller_faces_near_bump():
    mesh, north, bump_mask = _bumpy_sphere()
    config = PipelineConfig(target_faces=1000, profile=ORGANIC)
    field = compute_relative_sizing_field(mesh, config)
    preconditioned = precondition_mesh_graded(mesh, field, lambda_scale=1.0, config=config)

    face_centers = preconditioned.triangles_center
    dist_to_north = np.linalg.norm(face_centers - north, axis=1)
    near_bump_faces = preconditioned.area_faces[dist_to_north < 0.25]
    far_faces = preconditioned.area_faces[dist_to_north >= 0.5]

    assert len(near_bump_faces) > 0 and len(far_faces) > 0
    assert np.median(near_bump_faces) < np.median(far_faces), (
        "faces near the bump should be measurably smaller than faces far from it"
    )
