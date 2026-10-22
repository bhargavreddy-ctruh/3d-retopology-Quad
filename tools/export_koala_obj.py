"""One-off: export the koala GLB sample asset to a plain triangle OBJ for the QuadWild smoke test."""
import trimesh

SRC = "/Users/bhargavreddy/Downloads/3D/obj/pngtree-cute-cartoon-animal-character-clipart-colorful-koala-png-image_9200765_raw_1790k.glb.glb"
DST = "/Users/bhargavreddy/Downloads/3D/new_approach/tools/koala.obj"

scene = trimesh.load(SRC, force="scene")
mesh = trimesh.util.concatenate(
    [g for g in scene.geometry.values() if hasattr(g, "vertices")]
)
print(f"Loaded: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
mesh.export(DST)
print(f"Exported to {DST}")
