import os
import argparse
import numpy as np
import cv2

# import your BiNI-PS utilities
from bilateral_normal_integration_numpy_ps import (
    load_ps_inputs,
    bilateral_normal_integration,
    normals_from_pq,  # for sanity logs only
)

def _rng(a, m=None):
    if m is not None:
        a = a[m]
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.min()), float(a.max())

def _mean(a, m=None):
    if m is not None:
        a = a[m]
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")

def _corr(a, b, m):
    a = a[m]; b = b[m]
    a = a[np.isfinite(a) & np.isfinite(b)]
    b = b[np.isfinite(a) & np.isfinite(b)]
    if a.size < 10:
        return float("nan")
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

def frankot_chellappa(p, q, mask):
    """Integrate gradients (p=dz/dx, q=dz/dy) to depth using Frankot–Chellappa."""
    H, W = p.shape
    # zero outside mask
    P = np.zeros_like(p, dtype=np.float32); P[mask] = p[mask]
    Q = np.zeros_like(q, dtype=np.float32); Q[mask] = q[mask]

    # frequency coords
    wx = np.fft.fftfreq(W)*2*np.pi
    wy = np.fft.fftfreq(H)*2*np.pi
    kx, ky = np.meshgrid(wx, wy)

    # FFTs
    P_hat = np.fft.fft2(P)
    Q_hat = np.fft.fft2(Q)

    denom = (kx**2 + ky**2)
    denom[0, 0] = 1.0  # avoid /0 at DC
    Z_hat = (-1j*kx*P_hat - 1j*ky*Q_hat) / denom
    Z_hat[0, 0] = 0.0  # remove arbitrary offset

    z = np.real(np.fft.ifft2(Z_hat)).astype(np.float32)
    z[~mask] = np.nan
    return z

def main():
    ap = argparse.ArgumentParser(description="Run BiNI with FC prior from p/q.")
    ap.add_argument("--path", required=True, help="Folder containing p.npy, q.npy, mask.png, etc.")
    ap.add_argument("--k", type=float, default=4.0, help="BiNI k (edge sensitivity).")
    ap.add_argument("--iter", type=int, default=300, help="BiNI max iterations.")
    ap.add_argument("--tol", type=float, default=1e-6, help="BiNI stopping tolerance.")
    ap.add_argument("--lambda1", type=float, default=0.1, help="Weight for FC depth prior.")
    args = ap.parse_args()

    root = args.path
    assert os.path.isdir(root), f"Path not found: {root}"

    # 1) Load p, q, mask
    p_path = os.path.join(root, "p.npy")
    q_path = os.path.join(root, "q.npy")
    m_path = os.path.join(root, "mask.png")
    assert os.path.isfile(p_path) and os.path.isfile(q_path), "Missing p.npy or q.npy"
    assert os.path.isfile(m_path), "Missing mask.png"

    p = np.load(p_path).astype(np.float32)
    q = np.load(q_path).astype(np.float32)
    mask = (cv2.imread(m_path, cv2.IMREAD_GRAYSCALE) > 0)

    # ---- Logs: p/q stats
    print("[LOG] p/q stats:")
    print(f"      p mean={_mean(p, mask):.6f}, range={_rng(p, mask)}")
    print(f"      q mean={_mean(q, mask):.6f}, range={_rng(q, mask)}")

    # ---- Reconstruct normals from p/q for sanity (expects p=-nx/nz, q=-ny/nz)
    nx_pq, ny_pq, nz_pq = normals_from_pq(p, q)
    frac_nz_neg_pq = float((nz_pq[mask] < 0).mean())
    print("[LOG] normals from p/q (sanity):")
    print(f"      mean(nz)={_mean(nz_pq, mask):.6f}, frac(nz<0)={frac_nz_neg_pq:.4f}")
    print(f"      nx range={_rng(nx_pq, mask)}, ny range={_rng(ny_pq, mask)}, nz range={_rng(nz_pq, mask)}")

    # 2) FC prior
    print("[INFO] Computing FC prior from p/q...")
    z_fc = frankot_chellappa(p, q, mask)
    np.save(os.path.join(root, "z_fc.npy"), z_fc)
    z_fc_vis = cv2.normalize(z_fc, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(root, "z_fc.png"), z_fc_vis.astype(np.uint8))
    zfc_rng = _rng(z_fc, mask)
    print(f"[OK] Saved z_fc.npy and z_fc.png  |  z_fc range (masked) = {zfc_rng}")

    # 3) Load normals/mask the same way your BiNI-PS script does
    normal_map, mask_bini, used_pq = load_ps_inputs(root)

    # ---- Logs: normal_map stats
    nx_png = normal_map[..., 0]
    ny_png = normal_map[..., 1]
    nz_png = normal_map[..., 2]
    print("[LOG] normal_map (loaded) stats:")
    print(f"      mean(nz)={_mean(nz_png, mask_bini):.6f}, frac(nz<0)={float((nz_png[mask_bini] < 0).mean()):.4f}")
    print(f"      nx range={_rng(nx_png, mask_bini)}, ny range={_rng(ny_png, mask_bini)}, nz range={_rng(nz_png, mask_bini)}")

    # ---- Compare normals-from-pq vs normal_map (when both available)
    if used_pq:
        # Build a full image of normals-from-pq for correlation on the mask
        nx_img = np.zeros_like(nx_png); ny_img = np.zeros_like(ny_png); nz_img = np.zeros_like(nz_png)
        nx_img[mask] = nx_pq[mask]
        ny_img[mask] = ny_pq[mask]
        nz_img[mask] = nz_pq[mask]

        corr_nx = _corr(nx_img, nx_png, mask_bini)
        corr_ny = _corr(ny_img, ny_png, mask_bini)
        corr_nz = _corr(nz_img, nz_png, mask_bini)
        print("[LOG] correlation normals(pq) vs normal_map.png on mask:")
        print(f"      corr(nx)={corr_nx:.4f}, corr(ny)={corr_ny:.4f}, corr(nz)={corr_nz:.4f}")

    # Optionally load intrinsics if present
    K_path = os.path.join(root, "K.txt")
    K = np.loadtxt(K_path) if os.path.exists(K_path) else None

    # 4) Run BiNI with the prior
    print(f"[INFO] Running BiNI with prior (lambda1={args.lambda1})...")
    depth_map, surface, wu_map, wv_map, energy_list = bilateral_normal_integration(
        normal_map=normal_map,
        normal_mask=mask_bini,
        k=args.k,
        depth_map=z_fc,          # prior (in pixel units)
        depth_mask=mask,         # valid prior region
        lambda1=args.lambda1,    # weight for the prior
        K=K,
        max_iter=args.iter,
        tol=args.tol,
        save_path=root           # saves z_pix.npy and z_pix.png
    )

    # 5) Post BiNI logs and saves
    z = depth_map[mask_bini]
    print("[LOG] BiNI output z (pixel units):")
    print(f"      mean={_mean(z):.6f}, range=({_rng(z)[0]:.2f}, {_rng(z)[1]:.2f})")
    if np.isfinite(z).any():
        q25, q50, q75 = np.nanpercentile(z, [25, 50, 75])
        print(f"      quartiles: 25%={q25:.2f}, 50%={q50:.2f}, 75%={q75:.2f}")

    # Save extras (mesh, energies, wu/wv visualizations)
    np.save(os.path.join(root, "energy_with_prior.npy"), np.array(energy_list, dtype=np.float64))
    surface.save(os.path.join(root, f"mesh_k_{args.k}_prior.ply"), binary=False)

    wu_vis = cv2.applyColorMap((255 * np.nan_to_num(wu_map, nan=1.0)).astype(np.uint8), cv2.COLORMAP_JET)
    wv_vis = cv2.applyColorMap((255 * np.nan_to_num(wv_map, nan=1.0)).astype(np.uint8), cv2.COLORMAP_JET)
    wu_vis[~mask_bini] = 255
    wv_vis[~mask_bini] = 255
    cv2.imwrite(os.path.join(root, f"wu_k_{args.k}_prior.png"), wu_vis)
    cv2.imwrite(os.path.join(root, f"wv_k_{args.k}_prior.png"), wv_vis)

    print(f"[DONE] Saved all outputs to: {root}")

if __name__ == "__main__":
    main()
