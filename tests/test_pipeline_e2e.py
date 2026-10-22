from retopo.config import PipelineConfig
from retopo.pipeline import run_pipeline


def test_koala_end_to_end(koala_glb_path, tmp_path):
    out = str(tmp_path / "koala_output.glb")
    result = run_pipeline(koala_glb_path, out, PipelineConfig())

    assert result.n_quads > 0
    assert result.singularity_stats["irregular_frac"] < 0.10, (
        "Singularity rate regressed above 10% -- the whole point of using "
        "QuadWild+Bi-MDF over a local solver (Instant Meshes) is a low "
        "singularity count; this threshold should stay well under what a "
        "local field optimizer would produce (200+ on comparable meshes)."
    )
    assert result.mean_scaled_jacobian > 0.7
    assert result.chamfer_mean < 1e-3  # projected directly onto the high-poly surface
