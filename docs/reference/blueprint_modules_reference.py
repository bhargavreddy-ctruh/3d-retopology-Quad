"""
scratch/test_blueprint_modules.py
Unit test and validation for the Mathematical Foundations of the Blueprint:
1. Category-Agnostic Profile Auto-Classification
2. Bilateral Normal Denoising & Discrete Differential Curvature Tensor (Rusinkiewicz / Shape Operator)
3. Lipschitz-Continuous Geodesic Adaptive Sizing Field h(x) via SciPy Dijkstra
4. Bilateral Reflection Symmetry Detection & Bisection
"""

import os
import sys
import time
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.sparse import csgraph, csr_matrix

def bilateral_normal_filter(mesh: trimesh.Trimesh, iters: int = 3, sigma_spatial: float = None, sigma_range: float = 0.35) -> np.ndarray:
    """
    Bilateral normal denoising:
    Smooths high-frequency tessellation noise (e.g. Marching Cubes jitter)
    while strictly preserving sharp dihedral jumps across true geometric edges.
    """
    f_normals = mesh.face_normals.copy()
    f_centers = mesh.triangles_center
    adj_faces = mesh.face_adjacency
    
    if sigma_spatial is None:
        # 2x mean edge length
        sigma_spatial = float(np.mean(mesh.edges_unique_length) * 2.5)
    
    sigma_s2 = 2.0 * (sigma_spatial ** 2)
    sigma_r2 = 2.0 * (sigma_range ** 2)
    
    # Build face adjacency dictionary
    f_nbrs = [[] for _ in range(len(mesh.faces))]
    for f0, f1 in adj_faces:
        f_nbrs[f0].append(f1)
        f_nbrs[f1].append(f0)
        
    for _ in range(iters):
        new_normals = f_normals.copy()
        for i in range(len(mesh.faces)):
            nbrs = f_nbrs[i]
            if not nbrs:
                continue
            ci = f_centers[i]
            ni = f_normals[i]
            
            c_nbrs = f_centers[nbrs]
            n_nbrs = f_normals[nbrs]
            
            # Spatial distance
            d_spatial_sq = np.sum((c_nbrs - ci) ** 2, axis=1)
            # Normal angular distance (1 - dot)
            d_range_sq = np.sum((n_nbrs - ni) ** 2, axis=1)
            
            weights = np.exp(-d_spatial_sq / sigma_s2) * np.exp(-d_range_sq / sigma_r2)
            w_sum = np.sum(weights)
            if w_sum > 1e-12:
                filtered_n = np.sum(n_nbrs * weights[:, None], axis=0)
                norm = np.linalg.norm(filtered_n)
                if norm > 1e-12:
                    new_normals[i] = filtered_n / norm
        f_normals = new_normals
    return f_normals


def compute_discrete_curvature_tensor(mesh: trimesh.Trimesh, filtered_f_normals: np.ndarray = None):
    """
    Computes principal curvatures (kappa1, kappa2), principal directions (e1, e2),
    Mean Curvature H, Gaussian Curvature K, Deviation from Flatness D, and Shape Index S
    using vertex 1-ring neighborhood differential geometry.
    """
    num_v = len(mesh.vertices)
    verts = mesh.vertices
    faces = mesh.faces
    
    if filtered_f_normals is None:
        filtered_f_normals = mesh.face_normals
        
    # Vertex normals from filtered face normals weighted by face area
    v_normals = np.zeros((num_v, 3), dtype=np.float64)
    f_areas = mesh.area_faces
    for i, f in enumerate(faces):
        fn = filtered_f_normals[i] * f_areas[i]
        for v in f:
            v_normals[v] += fn
    norms = np.linalg.norm(v_normals, axis=1, keepdims=True)
    v_normals = np.divide(v_normals, np.maximum(norms, 1e-12))
    
    # Vertex adjacency list
    adj_v = [set() for _ in range(num_v)]
    for f in faces:
        for i in range(3):
            u, v = f[i], f[(i + 1) % 3]
            adj_v[u].add(v)
            adj_v[v].add(u)
            
    kappa1 = np.zeros(num_v, dtype=np.float64)
    kappa2 = np.zeros(num_v, dtype=np.float64)
    e1 = np.zeros((num_v, 3), dtype=np.float64)
    e2 = np.zeros((num_v, 3), dtype=np.float64)
    
    for i in range(num_v):
        nbrs = list(adj_v[i])
        if len(nbrs) < 3:
            continue
        n_i = v_normals[i]
        
        # Construct local orthonormal tangent frame (t1, t2)
        if abs(n_i[0]) > 0.9:
            up = np.array([0.0, 1.0, 0.0])
        else:
            up = np.array([1.0, 0.0, 0.0])
        t1 = np.cross(n_i, up)
        t1 /= np.maximum(np.linalg.norm(t1), 1e-12)
        t2 = np.cross(n_i, t1)
        
        # Fit local Weingarten map: S * [du, dv]^T = [d_normal_u, d_normal_v]
        # or local quadric z = 0.5 * (a*u^2 + 2*b*u*v + c*v^2)
        nbr_pts = verts[nbrs] - verts[i]
        u_coords = np.dot(nbr_pts, t1)
        v_coords = np.dot(nbr_pts, t2)
        w_coords = np.dot(nbr_pts, n_i)
        
        # Least squares system: [0.5*u^2, u*v, 0.5*v^2] * [a, 2b, c]^T = w
        A = np.column_stack([0.5 * u_coords**2, u_coords * v_coords, 0.5 * v_coords**2])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, w_coords, rcond=None)
            a, b, c = coeffs[0], coeffs[1], coeffs[2]
            
            # Fundamental curvature matrix II = [[a, b], [b, c]]
            II = np.array([[a, b], [b, c]])
            eigvals, eigvecs = np.linalg.eigh(II)
            
            # Sort such that |k1| >= |k2|
            idx_sort = np.argsort(np.abs(eigvals))[::-1]
            k1 = float(eigvals[idx_sort[0]])
            k2 = float(eigvals[idx_sort[1]])
            v1_2d = eigvecs[:, idx_sort[0]]
            v2_2d = eigvecs[:, idx_sort[1]]
            
            kappa1[i] = k1
            kappa2[i] = k2
            e1[i] = v1_2d[0] * t1 + v1_2d[1] * t2
            e2[i] = v2_2d[0] * t1 + v2_2d[1] * t2
        except Exception:
            pass
            
    # Differential Invariants
    H = (kappa1 + kappa2) / 2.0
    K = kappa1 * kappa2
    D = np.sqrt(kappa1**2 + kappa2**2)
    # Shape index in [-1, 1]
    denom = np.maximum(np.abs(kappa1 - kappa2), 1e-12)
    S = -(2.0 / np.pi) * np.arctan((kappa1 + kappa2) / denom)
    
    return {
        "kappa1": kappa1,
        "kappa2": kappa2,
        "e1": e1,
        "e2": e2,
        "H": H,
        "K": K,
        "D": D,
        "S": S,
        "v_normals": v_normals
    }


def compute_lipschitz_sizing_field(
    mesh: trimesh.Trimesh,
    kappa1: np.ndarray,
    kappa2: np.ndarray,
    target_faces: int = 5650,
    K_L: float = 0.35,
    sizing_ratio: float = 6.0,
    eps_tol_factor: float = 0.001
) -> np.ndarray:
    """
    Computes a Lipschitz-continuous adaptive sizing field h(x) over the mesh geodesic metric:
    h(x) = min_{y} (clamp(iota(y), h_min, h_max) + K_L * d_M(x, y))
    normalized to strictly honor target_faces.
    """
    num_v = len(mesh.vertices)
    bbox_diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    eps_tol = max(eps_tol_factor * bbox_diag, 1e-6)
    
    # Ideal chordal sizing: iota(v) = sqrt(8 * eps_tol / (|k1| + eps))
    curv_mag = np.abs(kappa1)
    eps_reg = 1e-4 / bbox_diag
    iota = np.sqrt((8.0 * eps_tol) / (curv_mag + eps_reg))
    
    # Estimate baseline mean edge length for target faces
    mean_h = float(np.sqrt(mesh.area / max(target_faces, 50)))
    h_min = mean_h / np.sqrt(sizing_ratio)
    h_max = mean_h * np.sqrt(sizing_ratio)
    
    clamped_iota = np.clip(iota, h_min, h_max)
    
    # Multi-source Dijkstra geodesic propagation
    # Build sparse edge weight matrix
    edges = mesh.edges_unique
    edge_lens = mesh.edges_unique_length
    
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([edge_lens, edge_lens])
    graph = csr_matrix((data, (row, col)), shape=(num_v, num_v))
    
    # Lipschitz bound propagation:
    # We can run Dijkstra from local curvature extrema sources
    # To be extremely fast, pick the most detailed vertices as sources (bottom 10% of iota)
    source_mask = clamped_iota <= np.percentile(clamped_iota, 15)
    source_indices = np.where(source_mask)[0]
    if len(source_indices) == 0:
        source_indices = np.array([np.argmin(clamped_iota)])
        
    dist_matrix = csgraph.dijkstra(graph, directed=False, indices=source_indices)
    
    # h(v) = min_{source s} (clamped_iota[s] + K_L * dist(s, v))
    lipschitz_terms = clamped_iota[source_indices, None] + (K_L * dist_matrix)
    h_field = np.min(lipschitz_terms, axis=0)
    h_field = np.minimum(h_field, clamped_iota)
    h_field = np.clip(h_field, h_min, h_max)
    
    # Global normalization factor lambda:
    # F_target = sum_t ( Area(t) / (lambda * h(c_t))^2 )
    # lambda = sqrt( (1 / F_target) * sum_t ( Area(t) / h(c_t)^2 ) )
    tri_h = np.mean(h_field[mesh.faces], axis=1)
    sum_area_h2 = np.sum(mesh.area_faces / (tri_h ** 2))
    lam = float(np.sqrt(sum_area_h2 / max(target_faces, 50)))
    
    normalized_h = lam * h_field
    return normalized_h


def detect_bilateral_symmetry(mesh: trimesh.Trimesh, tol_pct: float = 1.5) -> bool:
    """
    Evaluates reflection symmetry across x=0 plane via Chamfer distance.
    If Chamfer distance < 1.5% of bbox diagonal, returns True.
    """
    bbox_diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    tol = (tol_pct / 100.0) * bbox_diag
    
    # Sample 1000 points on surface
    sample_pts, _ = trimesh.sample.sample_surface(mesh, 2000)
    # Reflect across x=0
    refl_pts = sample_pts.copy()
    refl_pts[:, 0] = -refl_pts[:, 0]
    
    tree = cKDTree(sample_pts)
    dists, _ = tree.query(refl_pts)
    chamfer_dist = float(np.mean(dists))
    
    is_symmetric = chamfer_dist < tol
    return is_symmetric, chamfer_dist, tol


def classify_category_profile(mesh: trimesh.Trimesh, filtered_dihedrals: np.ndarray = None):
    """
    Category-Agnostic Profile Classification:
    Inspects distribution of filtered dihedral angles and principal curvature anisotropy.
    Returns: 'organic', 'utensil', or 'hard_surface'.
    """
    if filtered_dihedrals is None:
        filtered_dihedrals = mesh.face_adjacency_angles
        
    sharp_deg = 30.0
    rad_sharp = np.radians(sharp_deg)
    num_adj = len(filtered_dihedrals)
    if num_adj == 0:
        return "organic"
        
    sharp_ratio = float(np.sum(filtered_dihedrals > rad_sharp) / num_adj)
    
    if sharp_ratio < 0.02:
        profile = "organic"
        crease_angle = 42.0
        sizing_ratio = 6.0
        K_L = 0.35
        alpha_max = 2.5
    elif sharp_ratio <= 0.08:
        profile = "utensil"
        crease_angle = 30.0
        sizing_ratio = 5.0
        K_L = 0.25
        alpha_max = 4.0
    else:
        profile = "hard_surface"
        crease_angle = 25.0
        sizing_ratio = 8.0
        K_L = 0.50
        alpha_max = 5.0
        
    return {
        "profile": profile,
        "sharp_ratio": sharp_ratio,
        "crease_angle": crease_angle,
        "sizing_ratio": sizing_ratio,
        "K_L": K_L,
        "alpha_max": alpha_max
    }

if __name__ == "__main__":
    test_files = [
        "glb/testsample1_raw_860k.glb",
        "obj/furn16_woodenrack_raw_854k.glb",
        "obj/pngtree-cute-cartoon-animal-character-clipart-colorful-koala-png-image_9200765_raw_1790k.glb.glb"
    ]
    
    for fpath in test_files:
        if not os.path.exists(fpath):
            continue
        print(f"\n==========================================")
        print(f"Testing model: {fpath}")
        mesh = trimesh.load(fpath, force='mesh')
        print(f"Faces: {len(mesh.faces)}, Vertices: {len(mesh.vertices)}")
        
        # Test 1: Category classification
        cat = classify_category_profile(mesh)
        print(f"Profile: {cat['profile'].upper()} (Sharp Edge Ratio: {cat['sharp_ratio']*100:.2f}%)")
        print(f"  Crease: {cat['crease_angle']}°, Sizing Ratio: {cat['sizing_ratio']}, K_L: {cat['K_L']}")
        
        # Test 2: Symmetry detection
        is_sym, c_dist, tol = detect_bilateral_symmetry(mesh)
        print(f"Symmetry: {'YES' if is_sym else 'NO'} (Chamfer Dist: {c_dist:.5f}, Tol: {tol:.5f})")
        
        # Downsample for quick curvature test if huge
        if len(mesh.faces) > 20000:
            import fast_simplification
            v_simp, f_simp = fast_simplification.simplify(
                mesh.vertices.astype(np.float64),
                mesh.faces.astype(np.int64),
                target_count=15000
            )
            test_m = trimesh.Trimesh(vertices=v_simp, faces=f_simp, process=False)
        else:
            test_m = mesh
            
        t0 = time.perf_counter()
        fn = bilateral_normal_filter(test_m, iters=2)
        curv = compute_discrete_curvature_tensor(test_m, fn)
        h_field = compute_lipschitz_sizing_field(test_m, curv['kappa1'], curv['kappa2'], target_faces=5000, K_L=cat['K_L'], sizing_ratio=cat['sizing_ratio'])
        t_el = time.perf_counter() - t0
        print(f"Curvature & Lipschitz Sizing computed in {t_el:.3f}s")
        print(f"  h_min = {np.min(h_field):.5f}, h_max = {np.max(h_field):.5f}, ratio = {np.max(h_field)/np.min(h_field):.2f}")
        print(f"  Mean Curvature H range: [{np.min(curv['H']):.2f}, {np.max(curv['H']):.2f}]")
        print(f"  Shape Index S range: [{np.min(curv['S']):.2f}, {np.max(curv['S']):.2f}]")
