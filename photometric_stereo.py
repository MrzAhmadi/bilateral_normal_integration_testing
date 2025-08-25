# photometric_stereo.py (with extra logs)
import os, glob, argparse, shutil
import numpy as np
import cv2
import re


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
    r = ((n[..., 1] + 1.0) * 0.5 * 255.0).astype(np.uint8)
    g = ((n[..., 0] + 1.0) * 0.5 * 255.0).astype(np.uint8)
    b = ((-n[..., 2] + 1.0) * 0.5 * 255.0).astype(np.uint8)
    n_img = np.stack([r, g, b], axis=-1)
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

def main():
    ap = argparse.ArgumentParser(description="Calibrated Lambertian Photometric Stereo")
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--lights", required=True)
    ap.add_argument("--mask", default="")
    ap.add_argument("--shadows_dir", default="")
    ap.add_argument("--shadow_thresh", type=float, default=0.03)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--copy_from", default="data/Fig8_wallrelief")
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

    # 3) Weights
    Wobs = (I > args.shadow_thresh).astype(np.float32)
    if Wshadow is not None:
        Wobs *= Wshadow
    Wobs *= mask[..., None].astype(np.float32)
    print(f"[INFO] Valid obs fraction: {Wobs.sum() / (H*W*N):.4f}")

    # 4) Build normal equations
    A = np.einsum('hwn,nc,nd->hwcd', Wobs, L, L)
    b = np.einsum('hwn,nc,hwn->hwc', Wobs, L, I)

    # 5) Solve
    A_ = A.reshape(-1, 3, 3)
    b_ = b.reshape(-1, 3)
    reg = 1e-6 * np.eye(3, dtype=np.float32)[None, :, :]
    try:
        g_ = np.linalg.solve(A_ + reg, b_)
    except np.linalg.LinAlgError:
        g_, *_ = np.linalg.lstsq((A_ + reg), b_, rcond=None)
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

    # 7) Normals → gradients  (use the same convention as BiNI expects)
    nx, ny, nz = Nrm[..., 0], Nrm[..., 1], Nrm[..., 2]
    valid = (nz > 1e-3) & mask

    p = np.zeros_like(nx, dtype=np.float32)
    q = np.zeros_like(ny, dtype=np.float32)
    p[valid] = -nx[valid] / nz[valid]   # ∂z/∂x (rightwards)
    q[valid] =  ny[valid] / nz[valid]   # ∂z/∂y (downwards)  <-- flipped sign vs before

    # sanity logs AFTER filling p,q
    pm = float(np.nanmean(p[valid])); qm = float(np.nanmean(q[valid]))
    print(f"[HINT] Expect p rightwards positive, q downwards positive (your means: p={pm:.3f}, q={qm:.3f})")
    print(f"[DEBUG] p range (masked): {np.nanmin(p[valid]):.3f} .. {np.nanmax(p[valid]):.3f}")
    print(f"[DEBUG] q range (masked): {np.nanmin(q[valid]):.3f} .. {np.nanmax(q[valid]):.3f}")

    # === DEBUG LOGS ===
    print(f"[DEBUG] p range (masked): {np.nanmin(p[valid]):.3f} .. {np.nanmax(p[valid]):.3f}")
    print(f"[DEBUG] q range (masked): {np.nanmin(q[valid]):.3f} .. {np.nanmax(q[valid]):.3f}")
    print(f"[DEBUG] p mean: {np.nanmean(p[valid]):.3f}, q mean: {np.nanmean(q[valid]):.3f}")
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

    mask_src = os.path.join(args.copy_from, "mask.png")
    if os.path.isfile(mask_src):
        shutil.copy(mask_src, os.path.join(args.out_dir, "mask.png"))
        print(f"[OK] Copied mask from {mask_src}")

if __name__ == "__main__":
    main()