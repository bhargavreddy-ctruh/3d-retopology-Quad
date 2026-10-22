"""Quick visual QA: run the full pipeline (via retopo.pipeline.run_pipeline)
and render a matplotlib wireframe of the final quad mesh from a few angles."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from retopo.config import PipelineConfig
from retopo.io_glb import quads_to_paired_triangles  # noqa: F401 (not used, kept for reference)
from retopo.pipeline import run_pipeline
import trimesh
import numpy as np

KOALA = ("/Users/bhargavreddy/Downloads/3D/obj/"
         "pngtree-cute-cartoon-animal-character-clipart-colorful-koala-png-image_9200765_raw_1790k.glb.glb")

workdir = Path(__file__).resolve().parent / "render_work"
workdir.mkdir(exist_ok=True)
out_glb = str(workdir / "koala_output.glb")

result = run_pipeline(KOALA, out_glb, PipelineConfig(target_faces=5650))
print("Pipeline result:", result.n_quads, "quads,",
      f"{result.singularity_stats['irregular_frac']:.1%} irregular,",
      f"mean SJ {result.mean_scaled_jacobian:.3f}")
print("Calibration log:", result.stats.get("calibration_log"))

# Re-derive the quad faces from the paired-triangle GLB export: every pair of
# consecutive triangles sharing the (0,2) diagonal is one original quad.
mesh = trimesh.load(out_glb, force="mesh")
vertices = np.asarray(mesh.vertices)
tris = np.asarray(mesh.faces)
quads = []
i = 0
while i + 1 < len(tris):
    t0, t1 = tris[i], tris[i + 1]
    if t0[0] == t1[0] and t0[2] == t1[1]:
        quads.append([t0[0], t0[1], t0[2], t1[2]])
        i += 2
    else:
        i += 1
quads = np.array(quads)
print(f"Recovered {len(quads)} quads from {len(tris)} exported triangles")

edges = set()
for q in quads:
    for k in range(4):
        a, b = int(q[k]), int(q[(k + 1) % 4])
        edges.add((min(a, b), max(a, b)))
segments = [(vertices[a], vertices[b]) for a, b in edges]

fig = plt.figure(figsize=(16, 8))
views = [(90, -90, "Front"), (0, 0, "Side"), (0, 90, "Other side")]
for i, (elev, azim, title) in enumerate(views):
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    lc = Line3DCollection(segments, linewidths=0.3, colors="black")
    ax.add_collection3d(lc)
    mins, maxs = vertices.min(0), vertices.max(0)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect([maxs[0]-mins[0], maxs[1]-mins[1], maxs[2]-mins[2]])
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"{title} ({len(quads)} quads)")
    ax.set_axis_off()

plt.tight_layout()
out_path = str(workdir / "koala_wireframe_v2.png")
plt.savefig(out_path, dpi=150)
print(f"Saved {out_path}")
