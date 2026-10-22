"""Single source of truth for every pipeline threshold.

Every other module takes a PipelineConfig instance and never hardcodes a magic
number. This exists specifically to avoid the copy-paste drift found in prior
prototyping (tripo_grade_quad_pipeline.py / test_blueprint_*.py each had their
own slightly-different copies of category thresholds).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
QUADWILD_DIR = REPO_ROOT / "third_party" / "quadwild_bimdf"
QUADWILD_BIN = QUADWILD_DIR / "quadwild"
QUAD_FROM_PATCHES_BIN = QUADWILD_DIR / "quad_from_patches"


@dataclass(frozen=True)
class CategoryProfile:
    """Per-category defaults, per the design doc's parameter-tuning matrix.

    `sizing_ratio` and `lipschitz_k` are genuinely wired into
    `sizing_field.compute_relative_sizing_field` (see that module) -- this
    replaces an earlier version of this dataclass where both fields (plus
    `max_anisotropy` and `ridge_percentile`) were defined with different
    per-category values but never actually read anywhere in the pipeline,
    confirmed by a full-repo grep. `max_anisotropy` (a directional metric-
    tensor parameter) was removed outright rather than left dead: nothing in
    the current pipeline (pymeshlab's *isotropic* remeshing) can consume a
    directional size target -- re-add it, with real wiring, if anisotropic
    quad elongation becomes an explicit goal. `ridge_percentile` was dropped
    for the same reason; curvature-ridge-based force-clamping is a plausible
    future refinement of the sizing field, not implemented here.

    `sizing_ratio` values (h_max/h_min) were recalibrated after discovering
    the initial values (6/5/8) were far too conservative: h_min
    (= mean_h/sqrt(ratio)) was barely the same size as a genuinely important
    small feature (a wooden-rack drawer knob, ~0.033 model units diameter vs.
    h_min=0.031 at ratio=8), meaning the finest tier the sizing field could
    produce still couldn't resolve the feature with more than ~1 quad across
    it -- it showed up as a sliver/ripple, never a recognizable round bump,
    no matter how smooth the overall grading was. HARD_SURFACE=80 was
    verified (visually and via tools/measure_sizing_contrast.py) to resolve
    the same knob as a clean, multi-quad dome with visibly larger flat-panel
    quads around it. ORGANIC/UTENSIL were raised proportionally less (30/25)
    since organic characters generally don't have as extreme a tiny-decal-
    on-a-huge-body scale gap as furniture hardware does; if a future organic
    asset has a similarly extreme small feature, raise ORGANIC's ratio using
    the same diagnostic: compare the feature's real diameter against
    `estimate_initial_edge_length(mesh, target_faces) / sqrt(sizing_ratio)`.
    """

    name: str
    sharp_dihedral_deg: float      # hard crease threshold
    sizing_ratio: float            # h_max / h_min for the sizing field's clamp range
    lipschitz_k: float             # sizing field's max allowed gradient per unit edge length
    triangle_fraction_max: float   # permitted triangle fraction in output
    quadwild_prep_config: str      # which bundled prep config to use


ORGANIC = CategoryProfile(
    name="organic",
    sharp_dihedral_deg=42.0,
    sizing_ratio=30.0,
    lipschitz_k=0.35,
    triangle_fraction_max=0.05,
    quadwild_prep_config="basic_setup_Organic.txt",
)

UTENSIL = CategoryProfile(
    name="utensil",
    sharp_dihedral_deg=30.0,
    sizing_ratio=25.0,
    lipschitz_k=0.25,
    triangle_fraction_max=0.02,
    quadwild_prep_config="basic_setup.txt",
)

HARD_SURFACE = CategoryProfile(
    name="hard_surface",
    sharp_dihedral_deg=25.0,
    sizing_ratio=80.0,
    lipschitz_k=0.50,
    triangle_fraction_max=0.08,
    quadwild_prep_config="basic_setup_Mechanical.txt",
)

PROFILES = {p.name: p for p in (ORGANIC, UTENSIL, HARD_SURFACE)}


@dataclass
class PipelineConfig:
    """Top-level configuration for a single run of the pipeline."""

    target_faces: int = 5650
    profile: CategoryProfile = field(default_factory=lambda: ORGANIC)

    # Stage 1 (sanitization)
    weld_tolerance_frac: float = 1e-4   # fraction of bbox diagonal
    debris_area_frac: float = 1e-3      # components smaller than this fraction of
                                         # total surface area are discarded as debris
    max_tris_before_decimate: int = 250_000
    decimate_target_tris: int = 100_000

    # Pre-Stage-3 defeaturing (isolated bumps too small for even the finest
    # sizing tier to resolve with a handful of quads -- i.e. genuine noise,
    # not important detail). The threshold is now relative to the sizing
    # field's finest tier (h_min), not a single global edge length, so a
    # feature only gets defeatured if the new graded sizing field genuinely
    # can't resolve it -- see defeature.py's find_small_feature_faces.
    # Default ON: still useful for true noise-level blemishes (scan
    # artifacts, sub-h_min bumps). `restore_defeatured_peaks` defaults OFF:
    # with the graded sizing field, anything still being defeatured should be
    # true noise, so marking its location with a visible spike
    # (relax_project.restore_feature_peaks) would be reintroducing the exact
    # artifact being suppressed, not a useful recovery. Kept available as an
    # opt-in for cases where a visual marker is wanted anyway.
    defeature: bool = True
    defeature_max_extent_factor: float = 2.5
    restore_defeatured_peaks: bool = False

    # Stage 3 (sizing field / pre-conditioning)
    precondition: bool = True           # pre-remesh to target_faces before QuadWild
    calibration_max_iters: int = 3      # bisection-style retries to hit target_faces
    calibration_tolerance: float = 0.20 # acceptable fractional deviation from target_faces
    sizing_tiers: int = 4                     # discrete target-length tiers, finest to coarsest
    sizing_tier_overlap_rings: int = 1        # ring-dilation per tier selection, blends tier boundaries
    sizing_eps_tol_factor: float = 1e-3       # chordal/sagitta tolerance, as a fraction of bbox diagonal
    lipschitz_smoothing_max_iters: int = 40   # cap on the sizing field's edge-relaxation passes

    # Stage 4-5 (QuadWild backend)
    quadwild_main_config: str = "flow_noalign_lemon.txt"
    quadwild_run_id: int = 123
    quadwild_alpha: float = 0.02

    # Stage 6 (relax/project)
    relax_iterations: int = 8

    # Stage 5 (triangle transitions) -- quads whose Scaled Jacobian falls
    # below this are candidates for conversion to a triangle pair; see
    # triangle_transitions.apply_triangle_transitions's own docstring.
    triangle_min_scaled_jacobian: float = 0.15

    def quadwild_prep_config_path(self) -> Path:
        return QUADWILD_DIR / "config" / "prep_config" / self.profile.quadwild_prep_config

    def quadwild_main_config_path(self) -> Path:
        return QUADWILD_DIR / "config" / "main_config" / self.quadwild_main_config


def write_prep_config(path: Path, do_remesh: bool, sharp_feature_thr: float,
                       alpha: float = 0.02, scale_fact: float = 1.0) -> Path:
    """Write a QuadWild prep_config file. Used when Stage 3 has already
    pre-conditioned the mesh (do_remesh=False, so QuadWild doesn't undo our
    curvature-adaptive sizing with its own fixed internal remesh), with
    sharp_feature_thr tied to our CategoryProfile rather than the bundled
    static config's hardcoded value (which, for the bundled Organic profile,
    is -1 / disabled -- silently dropping all feature locking)."""
    path.write_text(
        f"do_remesh {1 if do_remesh else 0}\n"
        f"sharp_feature_thr {sharp_feature_thr}\n"
        f"alpha {alpha}\n"
        f"scaleFact {scale_fact}\n"
    )
    return path
