from retopo.config import PipelineConfig
from retopo.sanitize import load_and_sanitize


def test_koala_sanitizes_to_few_components(koala_glb_path):
    mesh, highpoly = load_and_sanitize(koala_glb_path, PipelineConfig())
    # Raw asset looks like 851 "bodies" due to unwelded per-primitive vertex
    # buffers; after position-only welding (ignoring UV/normal) it should
    # collapse to a small number of genuinely separate physical pieces.
    assert mesh.body_count <= 10, (
        f"Expected sanitized koala to have few real components, got {mesh.body_count}. "
        "This usually means weld_vertices regressed to not merging across UV/normal "
        "boundaries (see sanitize.weld_vertices docstring)."
    )
    assert len(mesh.faces) > 5000
    assert len(highpoly.faces) == len(mesh.faces)
