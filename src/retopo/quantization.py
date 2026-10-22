"""Stage 5: Bi-MDF patch quantization via the quad_from_patches binary.

Takes the T-mesh patch layout traced by QuadWildBackend (Stage 4) and solves
the integer arc-subdivision assignment (Bi-MDF via libSatsuma), extracting the
final quantized quad mesh.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import PipelineConfig, QUAD_FROM_PATCHES_BIN
from .crossfield_backend import BackendUnavailableError, QuadWildStage4Result


@dataclass
class QuadMeshResult:
    vertices: np.ndarray   # (N, 3) float64, NOT welded -- see sanitize-like post-step
    quads: np.ndarray      # (Q, 4) int, 0-indexed
    obj_path: Path


def _parse_obj_faces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                parts = line.split()[1:]
                faces.append([int(p.split("/")[0]) - 1 for p in parts])
    vertices = np.array(verts, dtype=np.float64)
    if not all(len(f) == 4 for f in faces):
        raise ValueError(
            f"{path} contains non-quad faces ({set(len(f) for f in faces)}); "
            "expected a pure quad mesh from quad_from_patches."
        )
    quads = np.array(faces, dtype=np.int64)
    return vertices, quads


def run_bimdf_quantization(
    stage4: QuadWildStage4Result, config: PipelineConfig, workdir: Path
) -> QuadMeshResult:
    if not QUAD_FROM_PATCHES_BIN.exists():
        raise BackendUnavailableError(
            f"quad_from_patches binary not found at {QUAD_FROM_PATCHES_BIN}."
        )

    main_config = config.quadwild_main_config_path()
    if not main_config.exists():
        raise BackendUnavailableError(f"QuadWild main config not found: {main_config}")

    cmd = [
        str(QUAD_FROM_PATCHES_BIN),
        str(stage4.rem_obj),
        str(config.quadwild_run_id),
        str(main_config),
    ]
    proc = subprocess.run(
        cmd, cwd=str(QUAD_FROM_PATCHES_BIN.parent), capture_output=True, text=True
    )

    stem = stage4.rem_obj.stem  # "<name>_rem_p0"
    out_obj = workdir / f"{stem}_{config.quadwild_run_id}_quadrangulation_smooth.obj"
    if not out_obj.exists():
        # fall back to the non-smoothed variant if present
        alt = workdir / f"{stem}_{config.quadwild_run_id}_quadrangulation.obj"
        if alt.exists():
            out_obj = alt
        else:
            raise BackendUnavailableError(
                f"quad_from_patches (exit {proc.returncode}) did not produce {out_obj}.\n"
                f"stdout tail:\n{proc.stdout[-2000:]}\nstderr tail:\n{proc.stderr[-2000:]}"
            )

    vertices, quads = _parse_obj_faces(out_obj)
    return QuadMeshResult(vertices=vertices, quads=quads, obj_path=out_obj)
