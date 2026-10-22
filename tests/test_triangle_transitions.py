import numpy as np

from retopo.config import PipelineConfig, ORGANIC
from retopo.metrics import scaled_jacobian_quad
from retopo.triangle_transitions import apply_triangle_transitions


def _grid_mesh(nx=4, ny=4):
    xs, ys = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    vertices = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    quads = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            b = a + 1
            c = a + nx + 1
            d = a + nx
            quads.append([a, b, c, d])
    return vertices, np.array(quads, dtype=np.int64)


def test_good_quad_grid_is_left_untouched():
    vertices, quads = _grid_mesh()
    config = PipelineConfig(profile=ORGANIC)
    remaining, triangles = apply_triangle_transitions(vertices, quads, config)
    assert len(remaining) == len(quads)
    assert len(triangles) == 0


def test_degenerate_quad_is_always_converted_even_over_budget():
    vertices, quads = _grid_mesh(nx=10, ny=10)
    # Collapse one quad's two corners onto the same point -> zero area / degenerate.
    # (This necessarily distorts its grid neighbors too, since they share the
    # moved vertices -- so more than just this one quad may get converted.)
    bad_quad = tuple(quads[0].tolist())
    vertices = vertices.copy()
    vertices[bad_quad[2]] = vertices[bad_quad[0]]
    vertices[bad_quad[3]] = vertices[bad_quad[0]]

    config = PipelineConfig(profile=ORGANIC)
    remaining, triangles = apply_triangle_transitions(vertices, quads, config)

    remaining_set = {tuple(q.tolist()) for q in remaining}
    assert bad_quad not in remaining_set, "the degenerate quad must always be converted"
    assert len(remaining) < len(quads)
    assert len(triangles) == 2 * (len(quads) - len(remaining))
    for q in remaining:
        assert scaled_jacobian_quad(vertices, q) > -1.0  # all finite, no crash


def test_respects_triangle_fraction_budget():
    vertices, quads = _grid_mesh(nx=20, ny=20)
    rng = np.random.default_rng(0)
    # Distort many quads' corner angles so they'd all be flagged if unbudgeted.
    vertices = vertices.copy()
    for idx in rng.choice(len(quads), size=100, replace=False):
        q = quads[idx]
        vertices[q[1]] += rng.normal(scale=0.3, size=3)

    config = PipelineConfig(profile=ORGANIC)
    remaining, triangles = apply_triangle_transitions(vertices, quads, config)
    budget = max(1, int(len(quads) * config.profile.triangle_fraction_max))
    n_converted = len(quads) - len(remaining)
    assert n_converted <= budget + 5  # small slack for forced degenerate conversions
