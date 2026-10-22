"""FastAPI backend for frontend/ (the "Quad-Dominant Studio" UI), wired to
the real retopo.pipeline.run_pipeline -- no fallback/fake remesher.

Endpoints match what frontend/app.js already expects:
  GET  /api/samples              -> list of preset .glb files to try
  POST /api/preview               -> load+sanitize a mesh for the left viewport
  POST /api/remesh                -> run the real pipeline, return both meshes + report
  GET  /api/download/{run_id}     -> download the result as .glb or .obj
"""
from __future__ import annotations

import sys
import time
import uuid
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import trimesh
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from retopo import io_glb, metrics  # noqa: E402
from retopo.config import PROFILES, PipelineConfig  # noqa: E402
from retopo.pipeline import run_pipeline  # noqa: E402
from retopo.sanitize import load_and_sanitize  # noqa: E402

RUNS_DIR = Path(__file__).resolve().parent / "_runs"
RUNS_DIR.mkdir(exist_ok=True)

# Where preset sample .glb files live on this machine (same assets the old
# prototype under /Users/bhargavreddy/Downloads/3D/web used).
SAMPLE_DIRS = [
    Path("/Users/bhargavreddy/Downloads/3D/glb"),
    Path("/Users/bhargavreddy/Downloads/3D/obj"),
]

app = FastAPI(title="Quad-Dominant Studio API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RUNS: dict[str, dict] = {}


def _find_sample(name: str) -> Path:
    for d in SAMPLE_DIRS:
        p = d / name
        if p.exists():
            return p
    raise HTTPException(404, f"sample '{name}' not found")


def _mesh_payload(vertices: np.ndarray, quads: np.ndarray | None = None,
                   triangles: np.ndarray | None = None) -> dict:
    """Flat arrays in the shape frontend/app.js's MeshViewer.setMesh expects."""
    quads = quads if quads is not None else np.zeros((0, 4), dtype=np.int64)
    triangles = triangles if triangles is not None else np.zeros((0, 3), dtype=np.int64)

    quad_sjs = [metrics.scaled_jacobian_quad(vertices, q) for q in quads] if len(quads) else []
    valences = metrics.vertex_valence(quads, len(vertices)).tolist() if len(quads) else [4] * len(vertices)

    return {
        "vertices": vertices.astype(np.float64).ravel().tolist(),
        "quads": quads.astype(np.int64).ravel().tolist(),
        "tris": triangles.astype(np.int64).ravel().tolist(),
        "quad_sjs": quad_sjs,
        "valences": valences,
        "num_quads": int(len(quads)),
        "num_tris": int(len(triangles)),
    }


def _boundary_edge_count(quads: np.ndarray, triangles: np.ndarray) -> int:
    counts: Counter[tuple[int, int]] = Counter()
    for q in quads:
        for i in range(4):
            a, b = int(q[i]), int(q[(i + 1) % 4])
            counts[(min(a, b), max(a, b))] += 1
    for t in triangles:
        for i in range(3):
            a, b = int(t[i]), int(t[(i + 1) % 3])
            counts[(min(a, b), max(a, b))] += 1
    return sum(1 for c in counts.values() if c == 1)


def _homeometry_variance(vertices: np.ndarray, quads: np.ndarray) -> float:
    """Coefficient of variation of quad edge lengths -- 0 = perfectly
    uniform element sizing, higher = more size variance across the mesh."""
    if not len(quads):
        return 0.0
    lengths = []
    for q in quads:
        p = vertices[q]
        for i in range(4):
            lengths.append(float(np.linalg.norm(p[i] - p[(i + 1) % 4])))
    lengths = np.array(lengths)
    mean = lengths.mean()
    return float(lengths.std() / mean) if mean > 1e-12 else 0.0


@app.get("/api/samples")
def list_samples():
    samples = []
    seen = set()
    for d in SAMPLE_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.glb")):
            if p.name in seen:
                continue
            seen.add(p.name)
            size = p.stat().st_size
            size_str = f"{size / 1024 / 1024:.1f} MB" if size > 1024 * 1024 else f"{size / 1024:.0f} KB"
            samples.append({"name": p.name, "size_str": size_str})
    return {"samples": samples}


@app.post("/api/preview")
async def preview(file: UploadFile | None = File(None), sample_name: str | None = Form(None)):
    if file is not None:
        run_dir = RUNS_DIR / uuid.uuid4().hex[:12]
        run_dir.mkdir(parents=True)
        input_path = run_dir / "input.glb"
        input_path.write_bytes(await file.read())
    elif sample_name:
        input_path = _find_sample(sample_name)
    else:
        raise HTTPException(400, "provide either 'file' or 'sample_name'")

    mesh, _ = load_and_sanitize(str(input_path), PipelineConfig())
    payload = _mesh_payload(np.asarray(mesh.vertices), triangles=np.asarray(mesh.faces))
    return {"mesh": payload, "face_count": int(len(mesh.faces))}


@app.post("/api/remesh")
async def remesh(
    file: UploadFile | None = File(None),
    sample_name: str | None = Form(None),
    input_mode: str = Form("auto"),
    engine: str = Form("tripo_production"),
    profile: str = Form("organic"),
    target_faces: int = Form(5650),
    crease_angle: float = Form(30.0),
    min_sj: float = Form(0.15),
    smooth_iters: int = Form(6),
    sizing_ratio: float | None = Form(None),
    defeature: bool = Form(True),
):
    if profile not in PROFILES:
        raise HTTPException(400, f"unknown profile '{profile}', expected one of {list(PROFILES)}")
    run_id = uuid.uuid4().hex[:8]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True)

    if file is not None:
        display_name = file.filename or "upload.glb"
        input_path = run_dir / "input.glb"
        input_path.write_bytes(await file.read())
    elif sample_name:
        display_name = sample_name
        src = _find_sample(sample_name)
        input_path = run_dir / "input.glb"
        input_path.write_bytes(src.read_bytes())
    else:
        raise HTTPException(400, "provide either 'file' or 'sample_name'")

    profile_obj = replace(PROFILES[profile], sharp_dihedral_deg=crease_angle)
    if sizing_ratio is not None:
        profile_obj = replace(profile_obj, sizing_ratio=sizing_ratio)
    config = replace(
        PipelineConfig(),
        target_faces=target_faces,
        relax_iterations=smooth_iters,
        triangle_min_scaled_jacobian=min_sj,
        defeature=defeature,
        profile=profile_obj,
    )

    # Input preview (sanitized, pre-remesh) mesh for the left viewport.
    sanitized_mesh, _ = load_and_sanitize(str(input_path), config)
    input_payload = _mesh_payload(np.asarray(sanitized_mesh.vertices), triangles=np.asarray(sanitized_mesh.faces))

    output_glb = run_dir / "output.glb"
    t0 = time.perf_counter()
    try:
        result = run_pipeline(str(input_path), str(output_glb), config)
    except Exception as exc:
        raise HTTPException(500, f"pipeline failed: {exc}") from exc
    elapsed = time.perf_counter() - t0

    output_payload = _mesh_payload(result.vertices, quads=result.quads, triangles=result.triangles)

    output_faces = result.n_quads + result.n_triangles
    quad_ratio_pct = 100.0 * result.n_quads / output_faces if output_faces else 0.0
    boundary_edges = _boundary_edge_count(result.quads, result.triangles)
    homeometry_variance = _homeometry_variance(result.vertices, result.quads)
    sing = result.singularity_stats

    obj_path = run_dir / "output.obj"
    quad_tris = io_glb.quads_to_paired_triangles(result.quads) if len(result.quads) else np.zeros((0, 3), dtype=np.int64)
    trimesh.Trimesh(
        vertices=result.vertices,
        faces=np.vstack([quad_tris, result.triangles]),
        process=False,
    ).export(obj_path)

    RUNS[run_id] = {"glb": str(output_glb), "obj": str(obj_path), "display_name": display_name}

    report = {
        "run_id": run_id,
        "display_name": display_name,
        "input_faces": int(len(sanitized_mesh.faces)),
        "output_faces": int(output_faces),
        "output_vertices": int(result.n_vertices),
        "output_quad_ratio_pct": round(quad_ratio_pct, 2),
        "is_watertight": boundary_edges == 0,
        "boundary_edges": int(boundary_edges),
        "mean_scaled_jacobian": result.mean_scaled_jacobian,
        "min_scaled_jacobian": result.min_scaled_jacobian,
        "regular_valence4_pct": round(100.0 * sing["regular"] / max(1, result.n_vertices), 2),
        "homeometry_variance": homeometry_variance,
        "elapsed_seconds": elapsed,
        "detected_category": config.profile.name,
        "detected_symmetry": False,
        "chamfer_mean": result.chamfer_mean,
        "sizing_ratio": config.profile.sizing_ratio,
        "defeature_enabled": config.defeature,
        "n_features_defeatured": result.stats.get("n_features_defeatured", 0),
    }

    return {"input_mesh": input_payload, "output_mesh": output_payload, "report": report}


@app.get("/api/download/{run_id}")
def download(run_id: str, format: str = "glb"):
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "unknown run id")
    if format == "obj":
        return FileResponse(run["obj"], media_type="text/plain", filename=f"{run_id}_{run['display_name']}.obj")
    return FileResponse(run["glb"], media_type="model/gltf-binary", filename=f"{run_id}_{run['display_name']}.glb")


FRONTEND_DIR = REPO_ROOT / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
