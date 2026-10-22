"""Stage 4: 4-RoSy cross-field computation via the QuadWild binary.

This is the architecture's highest-risk integration point. Prior prototyping
(see docs/reference/) always silently fell back to Instant Meshes here when
QuadWild wasn't available, which defeats the entire point of this architecture
(Instant Meshes' local field optimizer is exactly the failure mode QuadWild's
global Bi-MDF quantization is meant to replace). There is deliberately only
ONE backend in v1, and it fails loudly -- never silently -- if unavailable.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import PipelineConfig, QUADWILD_BIN


class BackendUnavailableError(RuntimeError):
    """Raised when the QuadWild binary is missing or fails to run. Callers
    must not catch this and silently substitute a different solver."""


@dataclass
class QuadWildStage4Result:
    rem_obj: Path       # <stem>_rem_p0.obj -- the pre-remeshed mesh ready for quantization
    rosy: Path          # cross-field file
    sharp: Path         # feature constraint file
    patch: Path         # traced T-mesh patch layout
    corners: Path
    feature: Path
    c_feature: Path


class QuadWildBackend:
    """Wraps the `quadwild` binary: feature detection, cross-field smoothing,
    and separatrix tracing into a T-mesh patch layout, all in one call (that's
    how the upstream tool is structured -- it is not split into finer-grained
    CLI steps)."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        if not QUADWILD_BIN.exists():
            raise BackendUnavailableError(
                f"QuadWild binary not found at {QUADWILD_BIN}. "
                "See third_party/quadwild_bimdf/PROVENANCE.md for how to fetch it. "
                "Refusing to silently substitute another solver."
            )
        if not shutil.os.access(QUADWILD_BIN, shutil.os.X_OK):
            raise BackendUnavailableError(f"QuadWild binary at {QUADWILD_BIN} is not executable.")

    def run(self, obj_path: Path, workdir: Path, prep_config: Path | None = None
            ) -> QuadWildStage4Result:
        workdir.mkdir(parents=True, exist_ok=True)
        local_obj = workdir / obj_path.name
        if local_obj != obj_path:
            shutil.copy(obj_path, local_obj)

        prep_config = prep_config or self.config.quadwild_prep_config_path()
        if not prep_config.exists():
            raise BackendUnavailableError(f"QuadWild prep config not found: {prep_config}")

        cmd = [str(QUADWILD_BIN), str(local_obj), "2", str(prep_config)]
        proc = subprocess.run(
            cmd, cwd=str(QUADWILD_BIN.parent), capture_output=True, text=True
        )
        stem = local_obj.stem
        rem_obj = workdir / f"{stem}_rem_p0.obj"
        if proc.returncode != 0 or not rem_obj.exists():
            raise BackendUnavailableError(
                f"quadwild failed (exit {proc.returncode}) or did not produce {rem_obj}.\n"
                f"stdout tail:\n{proc.stdout[-2000:]}\nstderr tail:\n{proc.stderr[-2000:]}"
            )

        def _req(suffix: str) -> Path:
            p = workdir / f"{stem}_rem_p0{suffix}"
            if not p.exists():
                raise BackendUnavailableError(f"Expected QuadWild output missing: {p}")
            return p

        return QuadWildStage4Result(
            rem_obj=rem_obj,
            rosy=workdir / f"{stem}_rem.rosy",
            sharp=workdir / f"{stem}_rem.sharp",
            patch=_req(".patch"),
            corners=_req(".corners"),
            feature=_req(".feature"),
            c_feature=_req(".c_feature"),
        )
