import numpy as np

from retopo.defeature import _feature_peak_point


def test_peak_point_picks_true_apex_not_a_lateral_flange_point():
    """Regression test for the exact bug found on the wooden-rack asset:
    picking the vertex farthest (by raw 3D distance) from the boundary
    ring's centroid can pick a lateral point on a wide base flange instead
    of the feature's actual raised summit, if the flange extends farther
    sideways than the bump rises vertically. Out-of-plane distance from the
    ring's own fitted plane must pick the true apex instead."""
    # A hexagonal boundary ring in the z=0 plane (radius 1), plus:
    #  - a "flange" vertex at radius 1.8 in-plane (z=0) -- farther from the
    #    ring centroid than the true apex, by raw distance, but NOT raised.
    #  - the true apex, close to the ring's center in (x, y) but raised in z.
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ring = np.stack([np.cos(angles), np.sin(angles), np.zeros(6)], axis=1)
    flange = np.array([[1.8, 0.0, 0.0]])
    apex = np.array([[0.05, 0.05, 1.2]])
    vertices = np.vstack([ring, flange, apex])  # indices: ring=0-5, flange=6, apex=7

    faces = []
    # Fan the ring + flange into triangles around a shared edge, plus
    # triangles connecting the ring to the apex, so everything is one
    # connected component sharing vertices with "outside" (none here --
    # single component, boundary == all verts, which is the intended
    # fallback path for this focused unit test).
    for i in range(6):
        j = (i + 1) % 6
        faces.append([i, j, 7])       # ring -> apex fan
    faces.append([0, 1, 6])           # a triangle touching the lateral flange

    class FakeMesh:
        def __init__(self, vertices, faces):
            self.vertices = vertices
            self.faces = np.array(faces)

    mesh = FakeMesh(vertices, faces)
    face_idx = np.arange(len(faces))

    peak = _feature_peak_point(mesh, face_idx)

    assert np.allclose(peak, apex[0]), (
        f"Expected the true raised apex {apex[0]}, got {peak} -- "
        "this means out-of-plane distance regressed back to raw centroid distance, "
        "which would pick the lateral flange vertex instead."
    )
