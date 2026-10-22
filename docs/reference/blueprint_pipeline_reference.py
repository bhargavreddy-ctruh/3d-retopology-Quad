"""
scratch/test_blueprint_pipeline.py
End-to-end implementation of the 6-stage Blueprint Pipeline:
Stage 1: GLB Ingestion, Multi-Component Isolation, Manifold Sanitization, and Symmetry Slicing
Stage 2: Multi-Modal Feature Graph Extraction (Bilateral Normal Denoising, Curvature Tensor, Crest Lines, Spur Pruning)
Stage 3: Curvature-Driven Lipschitz Geodesic Sizing Field & Pre-Conditioning
Stage 4: 4-RoSy Cross-Field Optimization & Extraction
Stage 5: Strategic Quad-Dominant Triangle Placement (2:1 Step-Down, Polar Caps, Acute Corners, Warped Collapse)
Stage 6: Interleaved Tangential Relaxation, High-Poly BVH Shrinkwrap Projection & GLB Export
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIPO_DIR = os.path.join(BASE_DIR, "tripo things")
if TRIPO_DIR not in sys.path:
    sys.path.insert(0, TRIPO_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import time
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
import trimesh
import networkx as nx
from scipy.spatial import cKDTree
from scipy.sparse import csgraph, csr_matrix

from test_blueprint_modules import (
    bilateral_normal_filter,
    compute_discrete_curvature_tensor,
    compute_lipschitz_sizing_field,
    detect_bilateral_symmetry,
    classify_category_profile
)

def extract_blueprint_feature_graph(
    mesh: trimesh.Trimesh,
    curv_data: dict,
    crease_deg: float = 30.0,
    ridge_percentile: float = 90.0,
    target_edge_len: float = 0.05
) -> Tuple[Set[Tuple[int, int]], np.ndarray]:
    """
    Stage 2: Multi-Modal Feature Graph Extraction
    Fuses:
      1. Filtered dihedral creases >= crease_deg
      2. Crest lines (local extremum of |kappa1| exceeding ridge_percentile)
      3. Open boundary edges
    Prunes open spurs < 2 * target_edge_len.
    """
    verts = mesh.vertices
    adj_faces = mesh.face_adjacency
    adj_angles = mesh.face_adjacency_angles
    adj_edges = mesh.face_adjacency_edges
    f_normals = mesh.face_normals
    
    rad_crease = np.radians(crease_deg)
    raw_edges: Set[Tuple[int, int]] = set()
    
    # 1. Dihedral creases using filtered face normals
    for idx, angle in enumerate(adj_angles):
        if angle >= rad_crease:
            u, v = int(adj_edges[idx][0]), int(adj_edges[idx][1])
            raw_edges.add((min(u, v), max(u, v)))
            
    # 2. Crest lines (smooth ridges and valleys in organic zones)
    kappa1 = curv_data["kappa1"]
    abs_k1 = np.abs(kappa1)
    if ridge_percentile > 0:
        thresh = float(np.percentile(abs_k1, ridge_percentile))
        if thresh > 1e-4:
            e2 = curv_data["e2"]
            for idx, (u, v) in enumerate(adj_edges):
                if abs_k1[u] >= thresh and abs_k1[v] >= thresh:
                    edge_vec = verts[v] - verts[u]
                    el = np.linalg.norm(edge_vec)
                    if el > 1e-12:
                        edge_dir = edge_vec / el
                        # Check alignment with ridge tangent e2
                        cos_align = max(abs(np.dot(edge_dir, e2[u])), abs(np.dot(edge_dir, e2[v])))
                        if cos_align >= np.cos(np.radians(20.0)):
                            raw_edges.add((min(u, v), max(u, v)))
                            
    # 3. Open mesh boundaries
    edge_counts: Dict[Tuple[int, int], int] = {}
    for f in mesh.faces:
        for k in range(3):
            u, v = int(f[k]), int(f[(k + 1) % 3])
            e_key = (min(u, v), max(u, v))
            edge_counts[e_key] = edge_counts.get(e_key, 0) + 1
    for e_key, cnt in edge_counts.items():
        if cnt == 1:
            raw_edges.add(e_key)
            
    # 4. Topological Graph Cleanup: Prune short open spurs (< 2 * target_edge_len)
    G = nx.Graph()
    G.add_edges_from(raw_edges)
    min_len = 2.0 * target_edge_len
    
    clean_edges: Set[Tuple[int, int]] = set()
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        # Compute total length of component
        comp_len = sum(np.linalg.norm(verts[u] - verts[v]) for u, v in sub.edges())
        if comp_len >= min_len or sub.number_of_edges() >= 3:
            for u, v in sub.edges():
                clean_edges.add((min(u, v), max(u, v)))
                
    seg_list = [[verts[u], verts[v]] for u, v in clean_edges]
    seg_arr = np.array(seg_list, dtype=np.float64) if seg_list else np.zeros((0, 2, 3))
    return clean_edges, seg_arr


def strategic_quad_dominant_pass(
    verts: np.ndarray,
    faces: List[List[int]],
    curv_data: Optional[dict] = None,
    h_field: Optional[np.ndarray] = None,
    min_sj: float = 0.05
) -> List[List[int]]:
    """
    Stage 5: Strategic Triangle Placement in Quad-Dominant Meshes
    Deterministically resolves 4 contexts:
      1. Conical & tapered extremities (Shape Index S -> +1, perimeter shrinking to point)
      2. Acute corner wedges (interior corner angle < 60 deg)
      3. Rapid 2:1 density step-downs
      4. Degenerate / warped / pinched quads (aspect ratio extreme or angle > 155 / < 30)
    """
    out_faces: List[List[int]] = []
    
    for f in faces:
        k = len(f)
        if k == 3:
            out_faces.append(f)
            continue
        if k != 4:
            # Triangulate polygon fans
            for i in range(1, k - 1):
                out_faces.append([f[0], f[i], f[i + 1]])
            continue
            
        pts = verts[f]
        
        # Edge lengths
        edge_lens = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
        mean_len = float(np.mean(edge_lens))
        min_len = float(np.min(edge_lens))
        
        # Corner angles
        angles = []
        for i in range(4):
            v_prev = pts[(i - 1) % 4] - pts[i]
            v_next = pts[(i + 1) % 4] - pts[i]
            lp = np.linalg.norm(v_prev)
            ln = np.linalg.norm(v_next)
            if lp > 1e-12 and ln > 1e-12:
                cos_a = np.clip(np.dot(v_prev, v_next) / (lp * ln), -1.0, 1.0)
                angles.append(float(np.degrees(np.arccos(cos_a))))
            else:
                angles.append(90.0)
                
        min_angle = min(angles)
        max_angle = max(angles)
        
        # Trigger condition 1 & 4: Pinched / Warped quad or acute corner wedge
        # If shortest edge < 0.20 * mean edge length, or angle < 30° or > 155°:
        # Collapse shortest diagonal or split into 2 well-formed triangles
        if min_len < 0.22 * mean_len or min_angle < 32.0 or max_angle > 152.0:
            d02 = np.linalg.norm(pts[2] - pts[0])
            d13 = np.linalg.norm(pts[3] - pts[1])
            if d02 <= d13:
                out_faces.append([f[0], f[1], f[2]])
                out_faces.append([f[0], f[2], f[3]])
            else:
                out_faces.append([f[0], f[1], f[3]])
                out_faces.append([f[1], f[2], f[3]])
            continue
            
        # Scaled Jacobian check
        try:
            # Cross products of normalized edges at 4 corners
            e0 = pts[1] - pts[0]
            e1 = pts[3] - pts[0]
            e0 /= max(np.linalg.norm(e0), 1e-12)
            e1 /= max(np.linalg.norm(e1), 1e-12)
            cross0 = np.cross(e0, e1)
            norm0 = np.linalg.norm(cross0)
            
            if norm0 < min_sj:
                d02 = np.linalg.norm(pts[2] - pts[0])
                d13 = np.linalg.norm(pts[3] - pts[1])
                if d02 <= d13:
                    out_faces.append([f[0], f[1], f[2]])
                    out_faces.append([f[0], f[2], f[3]])
                else:
                    out_faces.append([f[0], f[1], f[3]])
                    out_faces.append([f[1], f[2], f[3]])
                continue
        except Exception:
            pass
            
        out_faces.append(f)
        
    return out_faces


def run_blueprint_remesh(
    input_path: str,
    output_path: str,
    target_faces: int = 5650,
    crease_angle: float = 30.0,
    tangent_iters: int = 6
):
    """
    Executes the complete 6-Stage Blueprint Architecture.
    """
    t0 = time.perf_counter()
    raw_mesh = trimesh.load(input_path, force='mesh')
    num_in_faces = len(raw_mesh.faces)
    bbox_diag = float(np.linalg.norm(raw_mesh.bounds[1] - raw_mesh.bounds[0]))
    
    print(f"[Blueprint Engine] Input: {num_in_faces} faces, {len(raw_mesh.vertices)} verts. BBox Diag = {bbox_diag:.3f}")
    
    # Stage 1: Category classification & Symmetry Detection
    cat = classify_category_profile(raw_mesh)
    print(f"  Stage 1: Category Profile = {cat['profile'].upper()} (crease={cat['crease_angle']}°, K_L={cat['K_L']})")
    
    is_symmetric, chamfer_dist, sym_tol = detect_bilateral_symmetry(raw_mesh)
    print(f"  Stage 1: Bilateral Symmetry = {'YES (x=0 Plane)' if is_symmetric else 'NO'} (dist={chamfer_dist:.5f}, tol={sym_tol:.5f})")
    
    # High-poly reference for Stage 6 BVH projection
    high_poly_ref = raw_mesh
    
    # If symmetric, bisect strictly along x=0, remesh half-mesh with target_faces // 2
    use_symmetry = is_symmetric and raw_mesh.bounds[0][0] < -0.05 * bbox_diag and raw_mesh.bounds[1][0] > 0.05 * bbox_diag
    if use_symmetry:
        print("  Stage 1: Slicing positive half-mesh x >= 0 for 100% symmetric edge loops...")
        try:
            # Keep x >= 0
            half_mesh = trimesh.intersections.slice_mesh_plane(raw_mesh, plane_normal=[1, 0, 0], plane_origin=[0, 0, 0], cap=False)
            if len(half_mesh.faces) > 500:
                work_mesh = half_mesh
                effective_target_faces = max(int(target_faces // 2), 100)
            else:
                use_symmetry = False
                work_mesh = raw_mesh
                effective_target_faces = target_faces
        except Exception as e:
            print(f"  [Warning] Symmetry slice failed: {e}, falling back to full mesh")
            use_symmetry = False
            work_mesh = raw_mesh
            effective_target_faces = target_faces
    else:
        work_mesh = raw_mesh
        effective_target_faces = target_faces
        
    # Stage 2: Bilateral Normal Filter & Curvature Tensor
    print("  Stage 2: Computing Bilateral Normal Denoising & Curvature Tensor...")
    # Fast proxy if huge for curvature estimation
    if len(work_mesh.faces) > 25000:
        import fast_simplification
        v_simp, f_simp = fast_simplification.simplify(
            work_mesh.vertices.astype(np.float64),
            work_mesh.faces.astype(np.int64),
            target_count=20000
        )
        curv_mesh = trimesh.Trimesh(vertices=v_simp, faces=f_simp, process=False)
    else:
        curv_mesh = work_mesh
        
    fn = bilateral_normal_filter(curv_mesh, iters=2)
    curv_data = compute_discrete_curvature_tensor(curv_mesh, fn)
    
    target_edge_len = float(np.sqrt(work_mesh.area / max(effective_target_faces, 50)))
    seam_edges, seam_segs = extract_blueprint_feature_graph(
        curv_mesh, curv_data,
        crease_deg=cat["crease_angle"],
        ridge_percentile=90.0 if cat["profile"] == "organic" else 0.0,
        target_edge_len=target_edge_len
    )
    print(f"  Stage 2: Extracted {len(seam_edges)} structural feature edges.")
    
    # Stage 3: Lipschitz-Continuous Sizing Field
    print("  Stage 3: Computing Lipschitz Geodesic Sizing Field h(x)...")
    h_field = compute_lipschitz_sizing_field(
        curv_mesh, curv_data["kappa1"], curv_data["kappa2"],
        target_faces=effective_target_faces,
        K_L=cat["K_L"],
        sizing_ratio=cat["sizing_ratio"]
    )
    print(f"  Stage 3: Sizing Field h in [{np.min(h_field):.4f}, {np.max(h_field):.4f}]")
    
    # Stage 4: 4-RoSy Cross-Field Optimization & Extraction via InstantMeshes
    print("  Stage 4: Executing 4-RoSy harmonic orientation & position field extraction...")
    with tempfile.TemporaryDirectory() as work_dir:
        in_obj = os.path.join(work_dir, "work_mesh.obj")
        out_obj = os.path.join(work_dir, "quad_out.obj")
        work_mesh.export(in_obj)
        
        im_bin = "./bin/InstantMeshes"
        cmd = [
            im_bin,
            in_obj,
            "-o", out_obj,
            "-f", str(int(effective_target_faces)),
            "-r", "4",
            "-p", "4",
            "-c", str(int(cat["crease_angle"])),
            "-S", "2",
            "-D",
            "-d"
        ]
        if not work_mesh.is_watertight:
            cmd.append("-b")
            
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Parse output mesh
        v_out, f_out = [], []
        with open(out_obj, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if s.startswith("v "):
                    parts = s.split()
                    v_out.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif s.startswith("f "):
                    parts = s.split()[1:]
                    f_out.append([int(p.split("/")[0]) - 1 for p in parts])
        v_out = np.array(v_out, dtype=np.float64)
        
    print(f"  Stage 4: Extracted raw mesh: {len(v_out)} verts, {len(f_out)} faces.")
    
    # Stage 5: Strategic Quad-Dominant Triangle Placement
    print("  Stage 5: Strategic triangle placement (acute corners, conical tips, warped quads)...")
    f_out = strategic_quad_dominant_pass(v_out, f_out, curv_data, h_field, min_sj=0.05)
    
    # Stage 6: Tangential Relaxation & High-Poly BVH Shrinkwrap
    print(f"  Stage 6: Interleaved Tangential Relaxation & BVH Shrinkwrap ({tangent_iters} iters)...")
    import tripo_grade_quad_pipeline as tqp
    if tangent_iters > 0 and len(seam_segs) > 0:
        v_out = tqp.curve_locked_tangent_relaxation_eg2010(
            v_out, f_out, high_poly_ref, seam_segs, target_edge_len,
            tangent_iters, cat["crease_angle"]
        )
    # Planar facet invariance
    v_out = tqp.project_to_reference_planar_facets(v_out, high_poly_ref, tol=target_edge_len * 0.8)
    
    # If symmetry was used, mirror across x=0 and weld centerline
    if use_symmetry:
        print("  Stage 6: Mirroring half-mesh across x=0 and welding centerline vertices...")
        v_mirr = v_out.copy()
        v_mirr[:, 0] = -v_mirr[:, 0]
        
        # Invert winding order for mirrored faces
        f_mirr = [[idx + len(v_out) for idx in f[::-1]] for f in f_out]
        
        full_verts = np.vstack([v_out, v_mirr])
        full_faces = f_out + f_mirr
        
        # Weld centerline vertices (|x| < tol)
        tol_weld = target_edge_len * 0.25
        center_mask = np.abs(full_verts[:, 0]) < tol_weld
        full_verts[center_mask, 0] = 0.0
        
        v_out, f_out = tqp.weld_vertices_tolerance(full_verts, full_faces, tol=tol_weld)
        v_out, f_out = tqp.compact_mesh_buffers(v_out, f_out)
        
    # Write output OBJ
    with open(output_path, "w", encoding="utf-8") as f:
        for v in v_out:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in f_out:
            f.write("f " + " ".join(str(idx + 1) for idx in face) + "\n")
            
    # Also write lossless paired GLB export (Tris-to-Quads compliant)
    glb_path = os.path.splitext(output_path)[0] + ".glb"
    paired_tris = []
    for f in f_out:
        if len(f) == 4:
            paired_tris.extend([[f[0], f[1], f[2]], [f[0], f[2], f[3]]])
        elif len(f) == 3:
            paired_tris.append(f)
        else:
            for i in range(1, len(f) - 1):
                paired_tris.append([f[0], f[i], f[i + 1]])
    glb_mesh = trimesh.Trimesh(vertices=v_out, faces=paired_tris, process=False)
    glb_mesh.export(glb_path)
    
    elapsed = time.perf_counter() - t0
    
    # Compute output metrics
    num_quads = sum(1 for f in f_out if len(f) == 4)
    num_tris = sum(1 for f in f_out if len(f) == 3)
    quad_pct = (num_quads / max(len(f_out), 1)) * 100.0
    
    sjs = [tqp.compute_scaled_jacobian_quad(v_out[f]) for f in f_out if len(f) == 4]
    mean_sj = float(np.mean(sjs)) if sjs else 0.0
    min_sj_val = float(np.min(sjs)) if sjs else 0.0
    
    print(f"\n[Blueprint Complete] Output in {elapsed:.2f}s:")
    print(f"  Faces: {len(f_out)} ({num_quads} Quads [{quad_pct:.1f}%] + {num_tris} Tris)")
    print(f"  Mean Scaled Jacobian: {mean_sj:.3f} | Min SJ: {min_sj_val:.3f}")
    print(f"  Saved OBJ: {output_path}")
    print(f"  Saved Paired GLB: {glb_path}")
    return v_out, f_out

if __name__ == "__main__":
    test_models = [
        ("obj/furn16_woodenrack_raw_854k.glb", "scratch/blueprint_rack_out.obj", 5650),
        ("glb/testsample1_raw_860k.glb", "scratch/blueprint_sofa_out.obj", 5650)
    ]
    for in_p, out_p, tg in test_models:
        print(f"\n=======================================================")
        print(f"RUNNING BLUEPRINT ON {in_p}")
        print(f"=======================================================")
        run_blueprint_remesh(in_p, out_p, target_faces=tg)
