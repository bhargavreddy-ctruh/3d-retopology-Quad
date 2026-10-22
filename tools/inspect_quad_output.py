"""M0 smoke-test verification: load the QuadWild+Bi-MDF output and report face-type
and singularity stats, without using trimesh's triangulating OBJ loader (which would
hide quad structure) -- parse faces directly.
"""
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else (
    "/Users/bhargavreddy/Downloads/3D/new_approach/tools/smoke_work/"
    "koala_rem_p0_123_quadrangulation_smooth.obj"
)

verts = []
faces = []
with open(path) as f:
    for line in f:
        if line.startswith("v "):
            parts = line.split()
            verts.append(tuple(float(x) for x in parts[1:4]))
        elif line.startswith("f "):
            parts = line.split()[1:]
            idx = [int(p.split("/")[0]) - 1 for p in parts]
            faces.append(idx)

# Weld coincident vertices by rounded position -- the raw OBJ may have duplicate
# vertex entries at shared quad corners (common QuadWild/libigl export behavior),
# which would otherwise make every shared corner look like multiple low-valence
# vertices instead of one properly-connected one.
weld_map = {}
weld_id = {}
for i, p in enumerate(verts):
    key = (round(p[0], 5), round(p[1], 5), round(p[2], 5))
    if key not in weld_id:
        weld_id[key] = len(weld_id)
    weld_map[i] = weld_id[key]

faces = [[weld_map[v] for v in f] for f in faces]
n_welded_verts = len(weld_id)
print(f"Welded {len(verts)} raw vertices -> {n_welded_verts} unique positions")

n_tri = sum(1 for f in faces if len(f) == 3)
n_quad = sum(1 for f in faces if len(f) == 4)
n_other = sum(1 for f in faces if len(f) not in (3, 4))

valence = defaultdict(int)
for f in faces:
    for v in f:
        valence[v] += 1

# boundary-ish detection skipped (closed organic mesh expected); report raw valence histogram
hist = defaultdict(int)
for v, c in valence.items():
    hist[c] += 1

regular = hist.get(4, 0)
singular = sum(c for v, c in hist.items() if v != 4)

print(f"File: {path}")
print(f"Vertices (welded): {n_welded_verts}  Faces: {len(faces)}")
print(f"  triangles: {n_tri}  quads: {n_quad}  other-ngon: {n_other}")
print(f"  quad fraction: {n_quad / max(1, len(faces)):.1%}")
print(f"Vertex valence histogram (valence -> count): {dict(sorted(hist.items()))}")
print(f"Regular (valence-4) vertices: {regular}")
print(f"Irregular (singular) vertices: {singular}  "
      f"({singular / max(1, n_welded_verts):.1%} of all vertices)")
