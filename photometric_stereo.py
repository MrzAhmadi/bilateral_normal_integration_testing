# photometric_stereo.py  (PS + robust shadow handling + min-lights + optional FC prior)
import os, glob, argparse, shutil, re
import numpy as np
import cv2

# ------------------------- utils -------------------------

def _natural_key(s):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', os.path.basename(s))]

def load_images(images_dir):
    paths = sorted(
        glob.glob(os.path.join(images_dir, "*.png")) +
        glob.glob(os.path.join(images_dir, "*.jpg")) +
        glob.glob(os.path.join(images_dir, "*.jpeg")) +
        glob.glob(os.path.join(images_dir, "*.bmp"))
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {images_dir}")
    paths = sorted(paths, key=_natural_key)
    imgs = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise FileNotFoundError(f"Could not read image {p}")
        imgs.append(im.astype(np.float32) / 255.0)
    I = np.stack(imgs, axis=2)  # H x W x N
    print(f"[INFO] Loaded {len(paths)} images from {images_dir} -> shape {I.shape}")
    return I, paths

def load_lights(lights_path, expected_N=None):
    L = np.loadtxt(lights_path).astype(np.float32)  # N x 3
    if L.ndim != 2 or L.shape[1] != 3:
        raise ValueError("lights.txt must be Nx3 (one light direction per image).")
    L /= (np.linalg.norm(L, axis=1, keepdims=True) + 1e-8)
    if expected_N is not None and L.shape[0] != expected_N:
        raise ValueError(f"Number of lights ({L.shape[0]}) != number of images ({expected_N}).")
    print(f"[INFO] Loaded lights: {L.shape[0]} directions from {lights_path}")
    return L

def load_mask(mask_path, shape_hw):
    if not mask_path or not os.path.isfile(mask_path):
        print("[WARN] No mask provided; using all-ones mask.")
        return np.ones(shape_hw, dtype=bool)
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None or m.shape != shape_hw:
        raise ValueError(f"Mask not found or wrong size: expected {shape_hw}, got {None if m is None else m.shape}")
    valid = int((m > 0).sum())
    print(f"[INFO] Loaded mask {mask_path} with {valid} valid pixels")
    return (m > 0)

def load_shadow_masks(shadows_dir, shape_hw, N, image_paths_sorted):
    if not shadows_dir or not os.path.isdir(shadows_dir):
        print("[INFO] No shadows_dir; proceeding without per-image shadow masks.")
        return None
    W = np.ones((shape_hw[0], shape_hw[1], N), dtype=np.float32)
    used = 0
    for i, p in enumerate(image_paths_sorted):
        fname = os.path.basename(p)
        cand = os.path.join(shadows_dir, fname)
        if os.path.isfile(cand):
            m = cv2.imread(cand, cv2.IMREAD_GRAYSCALE)
            if m is not None and m.shape == shape_hw:
                W[:, :, i] = (m > 0).astype(np.float32)
                used += 1
    print(f"[INFO] Loaded {used}/{N} shadow masks from {shadows_dir}")
    return W

def maybe_fix_shadow_polarity(Wshadow):
    if Wshadow is None:
        return None, False
    frac_valid = float(Wshadow.mean())
    print(f"[DEBUG] Shadow masks fraction 'valid' (mean): {frac_valid:.4f}")
    flipped = False
    if frac_valid < 0.10:
        Wshadow = 1.0 - Wshadow
        flipped = True
        print("[INFO] Shadow masks looked inverted (white=shadow). Inverted so white=valid.")
    return Wshadow, flipped

def save_normals_png_dataset(normals, out_path_png):
    n = np.clip(normals, -1.0, 1.0)

    # R -> n_x, G -> n_y, B -> n_z
    r = ((n[..., 0] + 1.0) * 0.5 * 255.0).astype(np.uint8)
    g = ((n[..., 1] + 1.0) * 0.5 * 255.0).astype(np.uint8)
    b = ((n[..., 2] + 1.0) * 0.5 * 255.0).astype(np.uint8)

    n_img = np.stack([r, g, b], axis=-1)  # RGB order
    cv2.imwrite(out_path_png, cv2.cvtColor(n_img, cv2.COLOR_RGB2BGR))

def viz_percentile(img, mask, lo=2, hi=98):
    v = img.copy().astype(np.float32)
    v[~mask] = np.nan
    vv = v[np.isfinite(v)]
    if vv.size == 0:
        return np.zeros_like(v, dtype=np.uint8)
    a = np.percentile(vv, lo); b = np.percentile(vv, hi)
    if b <= a: b = a + 1e-6
    u = (np.clip((v - a) / (b - a), 0, 1) * 255).astype(np.uint8)
    u[~mask] = 0
    return u

def frankot_chellappa(p, q, mask):
    """Integrate gradients (p=dz/dx, q=dz/dy) to depth using Frankot–Chellappa."""
    H, W = p.shape
    P = np.zeros_like(p, dtype=np.float32); P[mask] = p[mask]
    Q = np.zeros_like(q, dtype=np.float32); Q[mask] = q[mask]
    wx = np.fft.fftfreq(W)*2*np.pi
    wy = np.fft.fftfreq(H)*2*np.pi
    kx, ky = np.meshgrid(wx, wy)
    P_hat = np.fft.fft2(P); Q_hat = np.fft.fft2(Q)
    denom = (kx**2 + ky**2); denom[0,0] = 1.0
    Z_hat = (-1j*kx*P_hat - 1j*ky*Q_hat) / denom
    Z_hat[0,0] = 0.0
    z = np.real(np.fft.ifft2(Z_hat)).astype(np.float32)
    z[~mask] = np.nan
    return z

def pct_clip(arr, mask, pct=99.5):
    """Clip arr inside mask to percentile range [100-pct, pct]. pct=0 disables."""
    if pct <= 0:
        return arr
    vv = arr[mask]
    lo, hi = np.nanpercentile(vv, [100 - pct, pct])
    out = arr.copy()
    out[mask] = np.clip(vv, lo, hi)
    return out

# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser(description="Calibrated Lambertian Photometric Stereo")
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--lights", required=True)
    ap.add_argument("--mask", default="")
    ap.add_argument("--shadows_dir", default="")

    # shadow controls
    ap.add_argument("--shadow_thresh", type=float, default=0.03,
                    help="Fixed threshold if --shadow_mode=fixed")
    ap.add_argument("--shadow_mode", choices=["fixed","per_image","per_pixel"], default="fixed",
                    help="Shadow weighting mode")
    ap.add_argument("--shadow_pct", type=float, default=20.0,
                    help="Percentile for per_image mode (e.g., 20 means bottom 20%% are shadow)")
    ap.add_argument("--shadow_scale", type=float, default=0.25,
                    help="Scale for per_pixel mode: I >= scale*max(I) -> lit")
    ap.add_argument("--save_shadow_debug", action="store_true",
                    help="Save debug images for shadow weighting")

    # stability / outputs
    ap.add_argument("--min_lights", type=int, default=25,
                    help="Require at least this many valid lights per pixel; others are ignored.")
    ap.add_argument("--clip_pq_pct", type=float, default=0.0,
                    help="If >0, clip p,q inside valid region to these percentiles (e.g., 99.5).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--copy_from", default="data/Fig8_wallrelief")
    ap.add_argument("--save_fc_prior", action="store_true",
                    help="Also save Frankot–Chellappa prior z_fc from computed p,q")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Load images and lights
    I, img_paths = load_images(args.images_dir)
    H, W, N = I.shape
    L = load_lights(args.lights, expected_N=N)

    # 2) Mask(s)
    mask = load_mask(args.mask, (H, W))
    Wshadow = load_shadow_masks(args.shadows_dir, (H, W), N, img_paths)
    Wshadow, _ = maybe_fix_shadow_polarity(Wshadow)

    # --- Exposure normalization per image (robust; ignore shadows) ---
    valid_for_expo = (I > args.shadow_thresh) & mask[..., None]
    if Wshadow is not None:
        valid_for_expo &= (Wshadow > 0.5)
    I_valid = np.where(valid_for_expo, I, np.nan)
    med = np.nanmedian(I_valid.reshape(-1, N), axis=0)
    if not np.isfinite(med).all():
        gmed = np.nanmedian(med)
        med = np.where(np.isfinite(med), med, gmed)
    med = np.clip(med, 1e-3, None)
    scale = med / np.nanmedian(med)
    I = I / scale
    np.save(os.path.join(args.out_dir, "ps_exposure_scale.npy"), scale.astype(np.float32))
    print("[DEBUG] Exposure normalization done. scale stats: "
          f"min={scale.min():.3g}, med={np.nanmedian(scale):.3g}, max={scale.max():.3g}")

    # 3) Shadow/Lit weights (Wobs)
    if args.shadow_mode == "fixed":
        Wobs = (I > args.shadow_thresh).astype(np.float32)

    elif args.shadow_mode == "per_image":
        Wobs = np.zeros_like(I, dtype=np.float32)
        for n in range(N):
            roi = mask.copy()
            if Wshadow is not None:
                roi &= (Wshadow[..., n] > 0.5)
            vals = I[..., n][roi]
            thr = args.shadow_thresh if vals.size == 0 else np.percentile(vals, args.shadow_pct)
            Wobs[..., n] = (I[..., n] >= thr).astype(np.float32)

    else:  # per_pixel
        if Wshadow is not None:
            I_eff = np.where(Wshadow > 0.5, I, np.nan)
            ref = np.nanmax(I_eff, axis=2, keepdims=True)
            ref = np.where(np.isfinite(ref), ref, np.nanmax(I, axis=2, keepdims=True))
        else:
            ref = np.max(I, axis=2, keepdims=True)
        ref = np.clip(ref, 1e-6, None)
        Wobs = (I >= (args.shadow_scale * ref)).astype(np.float32)

    # combine with provided Wshadow and mask
    if Wshadow is not None:
        Wobs *= Wshadow
    Wobs *= mask[..., None].astype(np.float32)
    print(f"[INFO] Valid obs fraction: {Wobs.sum() / (H*W*N):.4f}")

    if args.save_shadow_debug:
        meanW = np.nanmean(Wobs, axis=2)
        cv2.imwrite(os.path.join(args.out_dir, "shadow_mean_valid.png"),
                    (255*meanW).astype(np.uint8))
        # np.save(os.path.join(args.out_dir, "Wobs.npy"), Wobs.astype(np.float32))
        if args.shadow_mode == "per_image":
            thr_list = []
            for n in range(N):
                roi = mask.copy()
                if Wshadow is not None:
                    roi &= (Wshadow[..., n] > 0.5)
                vals = I[..., n][roi]
                thr = args.shadow_thresh if vals.size == 0 else np.percentile(vals, args.shadow_pct)
                thr_list.append(thr)
            np.save(os.path.join(args.out_dir, "per_image_thresholds.npy"),
                    np.array(thr_list, dtype=np.float32))

    # 4) Build normal equations
    A = np.einsum('hwn,nc,nd->hwcd', Wobs, L, L)   # (H,W,3,3)
    b = np.einsum('hwn,nc,hwn->hwc', Wobs, L, I)   # (H,W,3)

    # count valid lights per pixel
    n_valid = Wobs.sum(axis=2)  # (H,W)
    good_px = (n_valid >= args.min_lights) & mask
    print(f"[INFO] Pixels with >= {args.min_lights} valid lights: {good_px.mean():.3f}")

    # 5) Solve only on well-constrained pixels
    A_ = A.reshape(-1, 3, 3)
    b_ = b.reshape(-1, 3)
    reg = 1e-6 * np.eye(3, dtype=np.float32)

    g_ = np.zeros_like(b_, dtype=np.float32)
    valid_lin = good_px.reshape(-1)
    if valid_lin.any():
        A_reg = A_[valid_lin] + reg
        try:
            g_[valid_lin] = np.linalg.solve(A_reg, b_[valid_lin])
        except np.linalg.LinAlgError:
            g_[valid_lin], *_ = np.linalg.lstsq(A_reg, b_[valid_lin], rcond=None)

    G = g_.reshape(H, W, 3)

    # 6) Normals + albedo
    albedo = np.linalg.norm(G, axis=2, keepdims=True) + 1e-8
    Nrm = G / albedo
    Nrm[~mask] = 0.0
    albedo[~mask] = 0.0

    nz_masked = Nrm[..., 2][mask]
    print(f"[DEBUG] mean(nz) before flip: {np.nanmean(nz_masked):.6f}")
    if nz_masked.size and np.nanmean(nz_masked) < 0:
        Nrm = -Nrm
        print("[INFO] Flipped normals globally")
    print(f"[DEBUG] mean(nz) after  flip: {np.nanmean(Nrm[..., 2][mask]):.6f}")

    # 7) Normals → gradients (BiNI convention): p=dz/dx (right +), q=dz/dy (down +)
    nx, ny, nz = Nrm[..., 0], Nrm[..., 1], Nrm[..., 2]
    valid_grad = (nz > 1e-3) & mask
    p = np.zeros_like(nx, dtype=np.float32)
    q = np.zeros_like(ny, dtype=np.float32)
    p[valid_grad] = -nx[valid_grad] / nz[valid_grad]
    q[valid_grad] =  ny[valid_grad] / nz[valid_grad]

    # optional tail clipping for safety when writing p/q (does not affect solve)
    if args.clip_pq_pct > 0:
        p = pct_clip(p, valid_grad, args.clip_pq_pct)
        q = pct_clip(q, valid_grad, args.clip_pq_pct)
        print(f"[DEBUG] Applied percentile clip to p,q @ {args.clip_pq_pct}%")

    # sanity logs
    pm = float(np.nanmean(p[valid_grad])); qm = float(np.nanmean(q[valid_grad]))
    print(f"[HINT] Expect p rightwards positive, q downwards positive (your means: p={pm:.3f}, q={qm:.3f})")
    print(f"[DEBUG] p range (masked): {np.nanmin(p[valid_grad]):.3f} .. {np.nanmax(p[valid_grad]):.3f}")
    print(f"[DEBUG] q range (masked): {np.nanmin(q[valid_grad]):.3f} .. {np.nanmax(q[valid_grad]):.3f}")
    sample_idx = (H//2, W//2)
    print(f"[DEBUG] Sample center pixel p,q = {p[sample_idx]:.3f}, {q[sample_idx]:.3f}")

    # 8) Save results
    np.save(os.path.join(args.out_dir, "normals_unit.npy"), Nrm.astype(np.float32))
    np.save(os.path.join(args.out_dir, "albedo.npy"), albedo.squeeze(-1).astype(np.float32))
    np.save(os.path.join(args.out_dir, "p.npy"), p.astype(np.float32))
    np.save(os.path.join(args.out_dir, "q.npy"), q.astype(np.float32))
    save_normals_png_dataset(Nrm, os.path.join(args.out_dir, "normal_map.png"))

    # previews
    alb = albedo.squeeze(-1)
    s = np.percentile(alb[mask], 99) if mask.any() else 1.0
    alb_vis = np.clip(alb / max(s, 1e-6), 0, 1)
    cv2.imwrite(os.path.join(args.out_dir, "albedo_preview.png"), (alb_vis*255).astype(np.uint8))
    cv2.imwrite(os.path.join(args.out_dir, "p_preview.png"), viz_percentile(p, mask))
    cv2.imwrite(os.path.join(args.out_dir, "q_preview.png"), viz_percentile(q, mask))

    print("[OK] Saved normals, albedo, p, q, and previews")

    # Optional FC prior (use the stricter good_px mask to avoid noisy regions)
    if args.save_fc_prior:
        print("[INFO] Computing FC prior from p/q...")
        z_fc = frankot_chellappa(p, q, good_px)
        np.save(os.path.join(args.out_dir, "z_fc.npy"), z_fc.astype(np.float32))
        z_fc_vis = cv2.normalize(z_fc, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(os.path.join(args.out_dir, "z_fc.png"), z_fc_vis.astype(np.uint8))
        print("[OK] Saved z_fc.npy and z_fc.png")

    # copy mask for downstream
    mask_src = os.path.join(args.copy_from, "mask.png")
    if os.path.isfile(mask_src):
        shutil.copy(mask_src, os.path.join(args.out_dir, "mask.png"))
        print(f"[OK] Copied mask from {args.copy_from}/mask.png")

if __name__ == "__main__":
    main()
