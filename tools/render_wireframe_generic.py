"""Visual QA: render a wireframe of an already-produced quad GLB (paired-
triangle convention) from a few angles. Usage: render_wireframe_generic.py
<glb_path> <out_png_path>"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Line3DCollection

glb_path = sys.argv[1]
out_png = sys.argv[2]

mesh = trimesh.load(glb_path, force="mesh")
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
views = [(90, -90, "Front"), (0, 0, "Side"), (20, -60, "3/4")]
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
plt.savefig(out_png, dpi=150)
print(f"Saved {out_png}")
