"""
Bilateral Normal Integration (BiNI) — PS-type
Prefers p/q from Photometric Stereo; reconstructs normals from p,q if available.

CONVENTIONS (consistent everywhere in this script):
- Image axes: x → right, y → down, z → toward camera
- Normal map encoding (PNG & in-memory): RGB = (nx, ny, nz) in [-1, 1]
  -> Blue channel encodes n_z (perpendicular to camera)
- Gradients (orthographic): p = ∂z/∂x = -nx/nz,  q = ∂z/∂y = -ny/nz
"""

__author__ = "Xu Cao (orig) + PS adapter"
__version__ = "2.0-ps"

from scipy.sparse import spdiags, csr_matrix, vstack
from scipy.sparse.linalg import cg
import numpy as np
from tqdm.auto import tqdm
import time, os, cv2, pyvista as pv
import warnings
warnings.filterwarnings('ignore')

# -------------------- helpers --------------------

def move_left(mask): return np.pad(mask,((0,0),(0,1)),'constant',constant_values=0)[:,1:]
def move_right(mask): return np.pad(mask,((0,0),(1,0)),'constant',constant_values=0)[:,:-1]
def move_top(mask): return np.pad(mask,((0,1),(0,0)),'constant',constant_values=0)[1:,:]
def move_bottom(mask): return np.pad(mask,((1,0),(0,0)),'constant',constant_values=0)[:-1,:]
def move_top_left(mask): return np.pad(mask,((0,1),(0,1)),'constant',constant_values=0)[1:,1:]
def move_top_right(mask): return np.pad(mask,((0,1),(1,0)),'constant',constant_values=0)[1:,:-1]
def move_bottom_left(mask): return np.pad(mask,((1,0),(0,1)),'constant',constant_values=0)[:-1,1:]
def move_bottom_right(mask): return np.pad(mask,((1,0),(1,0)),'constant',constant_values=0)[:-1,:-1]

def generate_dx_dy(mask, nz_horizontal, nz_vertical, step_size=1):
    num_pixel = np.sum(mask)
    pixel_idx = np.zeros_like(mask, dtype=int)
    pixel_idx[mask] = np.arange(num_pixel)

    has_left_mask   = np.logical_and(move_right(mask), mask)
    has_right_mask  = np.logical_and(move_left(mask),  mask)
    has_bottom_mask = np.logical_and(move_top(mask),   mask)
    has_top_mask    = np.logical_and(move_bottom(mask), mask)

    nz_left   = nz_horizontal[has_left_mask[mask]]
    nz_right  = nz_horizontal[has_right_mask[mask]]
    nz_top    = nz_vertical[has_top_mask[mask]]
    nz_bottom = nz_vertical[has_bottom_mask[mask]]

    data = np.stack([-nz_left/step_size, nz_left/step_size], -1).flatten()
    indices = np.stack((pixel_idx[move_left(has_left_mask)], pixel_idx[has_left_mask]), -1).flatten()
    indptr = np.concatenate([np.array([0]), np.cumsum(has_left_mask[mask].astype(int) * 2)])
    D_horizontal_neg = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = np.stack([-nz_right/step_size, nz_right/step_size], -1).flatten()
    indices = np.stack((pixel_idx[has_right_mask], pixel_idx[move_right(has_right_mask)]), -1).flatten()
    indptr = np.concatenate([np.array([0]), np.cumsum(has_right_mask[mask].astype(int) * 2)])
    D_horizontal_pos = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = np.stack([-nz_top/step_size, nz_top/step_size], -1).flatten()
    indices = np.stack((pixel_idx[has_top_mask], pixel_idx[move_top(has_top_mask)]), -1).flatten()
    indptr = np.concatenate([np.array([0]), np.cumsum(has_top_mask[mask].astype(int) * 2)])
    D_vertical_pos = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = np.stack([-nz_bottom/step_size, nz_bottom/step_size], -1).flatten()
    indices = np.stack((pixel_idx[move_bottom(has_bottom_mask)], pixel_idx[has_bottom_mask]), -1).flatten()
    indptr = np.concatenate([np.array([0]), np.cumsum(has_bottom_mask[mask].astype(int) * 2)])
    D_vertical_neg = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    return D_horizontal_pos, D_horizontal_neg, D_vertical_pos, D_vertical_neg

def construct_facets_from(mask):
    idx = np.zeros_like(mask, dtype=int)
    idx[mask] = np.arange(np.sum(mask))

    facet_move_top_mask = move_top(mask)
    facet_move_left_mask = move_left(mask)
    facet_move_top_left_mask = move_top_left(mask)

    facet_top_left_mask = np.logical_and.reduce(
        (facet_move_top_mask, facet_move_left_mask, facet_move_top_left_mask, mask)
    )

    facet_top_right_mask   = move_right(facet_top_left_mask)
    facet_bottom_left_mask = move_bottom(facet_top_left_mask)
    facet_bottom_right_mask= move_bottom_right(facet_top_left_mask)

    return np.stack((4*np.ones(np.sum(facet_top_left_mask)),
                     idx[facet_top_left_mask],
                     idx[facet_bottom_left_mask],
                     idx[facet_bottom_right_mask],
                     idx[facet_top_right_mask]), axis=-1).astype(int)

def map_depth_map_to_point_clouds(depth_map, mask, K=None, step_size=1):
    H, W = mask.shape
    yy, xx = np.meshgrid(range(W), range(H))
    xx = np.flip(xx, axis=0)
    if K is None:
        vertices = np.zeros((H, W, 3))
        vertices[..., 0] = xx * step_size
        vertices[..., 1] = yy * step_size
        vertices[..., 2] = depth_map
        return vertices[mask]
    else:
        u = np.zeros((H, W, 3))
        u[...,0]=xx; u[...,1]=yy; u[...,2]=1
        u = u[mask].T
        return (np.linalg.inv(K) @ u).T * depth_map[mask, np.newaxis]

def sigmoid(x, k=1): return 1/(1+np.exp(-k*x))

# -------------------- PS adapter --------------------

def normals_from_pq(p, q):
    # p = -nx/nz, q = -ny/nz  =>  nx = -p*nz, ny = -q*nz
    denom = np.sqrt(1.0 + p*p + q*q)
    nz = 1.0 / np.maximum(denom, 1e-8)
    nx = -p * nz
    ny = -q * nz
    return nx, ny, nz

def load_ps_inputs(path):
    """
    Prefer p.npy/q.npy; fall back to normal_map.png.

    Returns:
        normal_map : float32 (H,W,3) in [-1,1], **RGB=(nx,ny,nz)**
        mask       : bool (H,W)
        used_pq    : bool
    """
    p_path = os.path.join(path, "p.npy")
    q_path = os.path.join(path, "q.npy")

    # mask
    m = cv2.imread(os.path.join(path, "mask.png"), cv2.IMREAD_GRAYSCALE)
    mask = (m > 0) if m is not None else None

    if os.path.isfile(p_path) and os.path.isfile(q_path):
        print("[INFO] Using p.npy/q.npy to reconstruct normals.")
        p = np.load(p_path).astype(np.float32)
        q = np.load(q_path).astype(np.float32)

        if mask is not None:
            p = np.where(mask, p, 0.0)
            q = np.where(mask, q, 0.0)

        nx, ny, nz = normals_from_pq(p, q)
        normal_map = np.stack([nx, ny, nz], axis=-1).astype(np.float32)  # RGB=(nx,ny,nz)

        # Sanity logs
        with np.errstate(invalid='ignore'):
            nz_mean = float(nz[mask].mean()) if mask is not None else float(np.nanmean(nz))
        print(f"[DEBUG] Reconstructed normals from p/q: "
              f"mean(nz)={nz_mean:.6f}, "
              f"nx∈[{np.nanmin(nx):.3f},{np.nanmax(nx):.3f}], "
              f"ny∈[{np.nanmin(ny):.3f},{np.nanmax(ny):.3f}], "
              f"nz∈[{np.nanmin(nz):.3f},{np.nanmax(nz):.3f}]")

        if mask is None:
            mask = np.ones(p.shape, bool)
        return normal_map, mask, True

    # --- Fallback: read normal_map.png encoded as RGB=(nx,ny,nz) in [0,255]/[0,65535] ---
    print("[INFO] p/q not found. Using normal_map.png")
    nm_path = os.path.join(path, "normal_map.png")
    img = cv2.imread(nm_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read {nm_path}")

    # drop alpha if present, ensure 3-ch
    if img.ndim == 2:
        raise ValueError(f"{nm_path} is grayscale; expected 3 channels.")
    if img.shape[2] == 4:
        img = img[:, :, :3]

    # BGR -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # to [0,1]
    if img_rgb.dtype == np.uint16:
        nm01 = img_rgb.astype(np.float32) / 65535.0
    else:
        nm01 = img_rgb.astype(np.float32) / 255.0

    normal_map = (nm01 * 2.0 - 1.0).astype(np.float32)  # RGB=(nx,ny,nz)

    if mask is None:
        mask = np.ones(normal_map.shape[:2], bool)

    # Sanity logs
    nxs, nys, nzs = [normal_map[...,i] for i in range(3)]
    with np.errstate(invalid='ignore'):
        nz_mean = float(nzs[mask].mean()) if mask is not None else float(np.nanmean(nzs))
    print(f"[DEBUG] Loaded normals from PNG: "
          f"mean(nz)={nz_mean:.6f}, "
          f"nx∈[{np.nanmin(nxs):.3f},{np.nanmax(nxs):.3f}], "
          f"ny∈[{np.nanmin(nys):.3f},{np.nanmax(nys):.3f}], "
          f"nz∈[{np.nanmin(nzs):.3f},{np.nanmax(nzs):.3f}]")
    return normal_map, mask, False

# -------------------- BiNI core --------------------

def bilateral_normal_integration(normal_map, normal_mask, k=2, depth_map=None, depth_mask=None,
                                 lambda1=0, K=None, step_size=1, max_iter=150, tol=1e-4,
                                 cg_max_iter=5000, cg_tol=1e-3, save_path=None):
    num_normals = np.sum(normal_mask)
    projection = "orthographic" if K is None else "perspective"
    print(f"Running bilateral normal integration with k={k} in the {projection} case.\n"
          f"The number of normal vectors is {num_normals}.")

    # Transform from normal-map encoding -> camera normals (RGB = nx, ny, nz)
    nx = normal_map[normal_mask, 0]
    ny = normal_map[normal_mask, 1]
    nz = normal_map[normal_mask, 2]

    # Sanity logs on input normals to BiNI
    def _rng(x): 
        with np.errstate(invalid='ignore'):
            return float(np.nanmin(x)), float(np.nanmax(x))
    with np.errstate(invalid='ignore'):
        nz_mean = float(nz.mean()) if nz.size else float('nan')
        frac_back = float((nz < 0).mean())
    print(f"[DEBUG] BiNI input normals: mean(nz)={nz_mean:.6f}, "
          f"nx∈[{_rng(nx)[0]:.3f},{_rng(nx)[1]:.3f}], "
          f"ny∈[{_rng(ny)[0]:.3f},{_rng(ny)[1]:.3f}], "
          f"nz∈[{_rng(nz)[0]:.3f},{_rng(nz)[1]:.3f}], "
          f"fraction(nz<0)={frac_back:.4f}")

    if K is not None:
        img_height, img_width = normal_mask.shape[:2]
        yy, xx = np.meshgrid(range(img_width), range(img_height))
        xx = np.flip(xx, axis=0)
        cx, cy = K[0, 2], K[1, 2]
        fx, fy = K[0, 0], K[1, 1]
        uu = xx[normal_mask] - cx
        vv = yy[normal_mask] - cy
        nz_u = uu * nx + vv * ny + fx * nz
        nz_v = uu * nx + vv * ny + fy * nz
        del xx, yy, uu, vv
    else:
        nz_u = nz.copy()
        nz_v = nz.copy()

    # derivative matrices
    A3, A4, A1, A2 = generate_dx_dy(normal_mask, nz_horizontal=nz_v, nz_vertical=nz_u, step_size=step_size)

    # linear system
    A = vstack((A1, A2, A3, A4))
    # NOTE: with p = -nx/nz, q = -ny/nz, the correct RHS uses +nx, +ny here.
    b = np.concatenate((nx, nx, ny, ny))

    # optimize
    W = spdiags(0.5 * np.ones(4*num_normals), 0, 4*num_normals, 4*num_normals, format="csr")
    z = np.zeros(np.sum(normal_mask), dtype=np.float32)
    energy = (A @ z - b).T @ W @ (A @ z - b)

    tic = time.time()
    energy_list = []
    if depth_map is not None:
        m = depth_mask[normal_mask].astype(int)
        M = spdiags(m, 0, num_normals, num_normals, format="csr")
        z_prior = np.log(depth_map)[normal_mask] if K is not None else depth_map[normal_mask]

    pbar = tqdm(range(max_iter))
    for i in pbar:
        A_mat = A.T @ W @ A
        b_vec = A.T @ W @ b
        if depth_map is not None:
            depth_diff = M @ (z_prior - z)
            depth_diff[depth_diff==0] = np.nan
            offset = np.nanmean(depth_diff)
            z = z + offset
            A_mat += lambda1 * M
            b_vec += lambda1 * M @ z_prior

        D = spdiags(1/np.clip(A_mat.diagonal(), 1e-5, None), 0, num_normals, num_normals, format="csr")
        z, _ = cg(A_mat, b_vec, x0=z, M=D, maxiter=cg_max_iter, tol=cg_tol)

        wu = sigmoid((A2 @ z) ** 2 - (A1 @ z) ** 2, k)
        wv = sigmoid((A4 @ z) ** 2 - (A3 @ z) ** 2, k)
        W = spdiags(np.concatenate((wu, 1-wu, wv, 1-wv)), 0, 4*num_normals, 4*num_normals, format="csr")

        energy_old = energy
        energy = (A @ z - b).T @ W @ (A @ z - b)
        energy_list.append(energy)
        relative_energy = np.abs(energy - energy_old) / max(abs(energy_old), 1e-12)
        pbar.set_description(f"step {i + 1}/{max_iter} energy: {energy:.3f} relative energy: {relative_energy:.3e}")
        if relative_energy < tol:
            break
    toc = time.time()
    print(f"Total time: {toc - tic:.3f} sec")

    depth_map_out = np.ones_like(normal_mask, float) * np.nan
    depth_map_out[normal_mask] = z

    # Debug range of z before saving
    z_valid = z[np.isfinite(z)]
    if z_valid.size:
        print(f"[DEBUG] z (pix-units) range before save: [{float(z_valid.min()):.2f}, {float(z_valid.max()):.2f}]")

    # save z_pix
    if save_path is not None:
        np.save(os.path.join(save_path, "z_pix.npy"), depth_map_out.astype(np.float32))
        z_norm = cv2.normalize(depth_map_out, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(os.path.join(save_path, "z_pix.png"), z_norm.astype(np.uint8))
        print(f"[OK] Saved z_pix.npy and z_pix.png to {save_path}")

        # also save a small normals preview for sanity (blue = nz)
        nm = np.zeros((*normal_mask.shape, 3), dtype=np.float32)
        nm[normal_mask, 0] = nx
        nm[normal_mask, 1] = ny
        nm[normal_mask, 2] = nz
        nm_vis = np.clip((nm * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)  # RGB
        nm_vis_bgr = nm_vis[..., ::-1]  # RGB->BGR for cv2
        cv2.imwrite(os.path.join(save_path, "normals_preview.png"), nm_vis_bgr)
        print(f"[OK] Saved normals_preview.png (RGB=(nx,ny,nz); blue=nz)")

    # surface
    if K is not None:
        depth_for_pc = np.exp(depth_map_out)
        vertices = map_depth_map_to_point_clouds(depth_for_pc, normal_mask, K=K)
    else:
        vertices = map_depth_map_to_point_clouds(depth_map_out, normal_mask, K=None, step_size=step_size)

    facets = construct_facets_from(normal_mask)
    # NOTE: with RGB=(nx,ny,nz) and nz>0 on average, this should NOT trigger.
    if normal_map[:, :, -1].mean() < 0:
        print("[HINT] mean(nz) < 0 on normal_map; flipping facet order for correct orientation.")
        facets = facets[:, [0, 1, 4, 3, 2]]
    surface = pv.PolyData(vertices, facets)

    wu_map = np.ones_like(normal_mask) * np.nan
    wv_map = np.ones_like(normal_mask) * np.nan
    wu_map[normal_mask] = sigmoid((A2 @ z) ** 2 - (A1 @ z) ** 2, k)
    wv_map[normal_mask] = sigmoid((A4 @ z) ** 2 - (A3 @ z) ** 2, k)

    return depth_map_out, surface, wu_map, wv_map, energy_list

# -------------------- CLI --------------------

if __name__ == '__main__':
    import argparse
    def dir_path(string):
        if os.path.isdir(string): return string
        raise FileNotFoundError(string)

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=dir_path, required=True)
    parser.add_argument('-k', type=float, default=2)
    parser.add_argument('-i', '--iter', type=int, default=150)
    parser.add_argument('-t', '--tol', type=float, default=1e-4)
    args = parser.parse_args()

    # Load inputs (prefer p/q)
    normal_map, mask, used_pq = load_ps_inputs(args.path)

    # Camera intrinsics?
    if os.path.exists(os.path.join(args.path, "K.txt")):
        K = np.loadtxt(os.path.join(args.path, "K.txt"))
    else:
        K = None

    # Quick top-level sanity on normal_map before passing to BiNI
    nz_mean_all = float(normal_map[...,2][mask].mean())
    frac_nz_neg = float((normal_map[...,2][mask] < 0).mean())
    print(f"[SUMMARY] normal_map: mean(nz within mask)={nz_mean_all:.6f}, fraction(nz<0)={frac_nz_neg:.4f}, used_pq={used_pq}")

    # Run BiNI
    depth_map, surface, wu_map, wv_map, energy_list = bilateral_normal_integration(
        normal_map=normal_map, normal_mask=mask, k=args.k, K=K,
        max_iter=args.i, tol=args.t, save_path=args.path
    )

    # Save artifacts
    np.save(os.path.join(args.path, "energy.npy"), np.array(energy_list))
    surface.save(os.path.join(args.path, f"mesh_k_{args.k}.ply"), binary=False)

    wu_vis = cv2.applyColorMap((255 * np.nan_to_num(wu_map, nan=1.0)).astype(np.uint8), cv2.COLORMAP_JET)
    wv_vis = cv2.applyColorMap((255 * np.nan_to_num(wv_map, nan=1.0)).astype(np.uint8), cv2.COLORMAP_JET)
    wu_vis[~mask] = 255; wv_vis[~mask] = 255
    cv2.imwrite(os.path.join(args.path, f"wu_k_{args.k}.png"), wu_vis)
    cv2.imwrite(os.path.join(args.path, f"wv_k_{args.k}.png"), wv_vis)

    print(f"[OK] saved {args.path} (used_pq={used_pq})")