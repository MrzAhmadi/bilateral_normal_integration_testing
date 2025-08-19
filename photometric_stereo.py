# photometric_stereo.py
import os, glob, argparse, shutil
import numpy as np
import cv2

def load_images(images_dir):
    paths = sorted(
        glob.glob(os.path.join(images_dir, "*.png")) +
        glob.glob(os.path.join(images_dir, "*.jpg")) +
        glob.glob(os.path.join(images_dir, "*.jpeg")) +
        glob.glob(os.path.join(images_dir, "*.bmp"))
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {images_dir}")
    imgs = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise FileNotFoundError(f"Could not read image {p}")
        imgs.append(im.astype(np.float32) / 255.0)
    I = np.stack(imgs, axis=2)  # H x W x N
    return I, paths

def load_lights(lights_path, expected_N=None):
    L = np.loadtxt(lights_path).astype(np.float32)  # N x 3
    if L.ndim != 2 or L.shape[1] != 3:
        raise ValueError("lights.txt must be Nx3 (one light direction per image).")
    L /= (np.linalg.norm(L, axis=1, keepdims=True) + 1e-8)
    if expected_N is not None and L.shape[0] != expected_N:
        raise ValueError(f"Number of lights ({L.shape[0]}) != number of images ({expected_N}).")
    return L

def load_mask(mask_path, shape_hw):
    if not mask_path or not os.path.isfile(mask_path):
        return np.ones(shape_hw, dtype=bool)
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None or m.shape != shape_hw:
        raise ValueError(f"Mask not found or wrong size: expected {shape_hw}, got {None if m is None else m.shape}")
    return (m > 0)

def load_shadow_masks(shadows_dir, shape_hw, N, image_paths_sorted):
    if not shadows_dir or not os.path.isdir(shadows_dir):
        return None
    W = np.ones((shape_hw[0], shape_hw[1], N), dtype=np.float32)
    for i, p in enumerate(image_paths_sorted):
        fname = os.path.basename(p)
        cand = os.path.join(shadows_dir, fname)
        if os.path.isfile(cand):
            m = cv2.imread(cand, cv2.IMREAD_GRAYSCALE)
            if m is not None and m.shape == shape_hw:
                W[:, :, i] = (m > 0).astype(np.float32)
    return W

def save_normals_png_dataset(normals, out_path_png):
    """
    Save normals in the dataset encoding expected by BiNI/evaluator:
      R = ny, G = nx, B = -nz   (each in [-1,1] mapped to [0,255])
    Input 'normals' must be in camera coordinates (nx, ny, nz) with unit length.
    """
    n = np.clip(normals, -1.0, 1.0)
    r = ((n[..., 1] + 1.0) * 0.5 * 255.0).astype(np.uint8)   # ny -> R
    g = ((n[..., 0] + 1.0) * 0.5 * 255.0).astype(np.uint8)   # nx -> G
    b = ((-n[..., 2] + 1.0) * 0.5 * 255.0).astype(np.uint8)  # -nz -> B
    n_img = np.stack([r, g, b], axis=-1)
    cv2.imwrite(out_path_png, cv2.cvtColor(n_img, cv2.COLOR_RGB2BGR))

def main():
    ap = argparse.ArgumentParser(description="Calibrated Lambertian Photometric Stereo")
    ap.add_argument("--images_dir", required=True, help="Folder with per-light images")
    ap.add_argument("--lights", required=True, help="Path to lights.txt (Nx3)")
    ap.add_argument("--mask", default="", help="Optional mask.png (white=valid)")
    ap.add_argument("--shadows_dir", default="", help="Optional folder of per-image shadow masks (white=valid)")
    ap.add_argument("--shadow_thresh", type=float, default=0.03, help="Ignore pixels darker than this [0..1]")
    ap.add_argument("--out_dir", required=True, help="Output folder")
    ap.add_argument("--copy_from", default="data/Fig8_wallrelief",
                    help="Folder to copy mask.png and ground_truth/depth_map.png from")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Load images and lights
    I, img_paths = load_images(args.images_dir)   # H x W x N
    H, W, N = I.shape
    L = load_lights(args.lights, expected_N=N)    # N x 3

    # 2) Mask(s)
    mask = load_mask(args.mask, (H, W))           # H x W (bool)
    Wshadow = load_shadow_masks(args.shadows_dir, (H, W), N, img_paths)  # H x W x N or None

    # 3) Per-observation weights
    Wobs = (I > args.shadow_thresh).astype(np.float32)  # H x W x N
    if Wshadow is not None:
        Wobs *= Wshadow

    # 4) Build normal equations A = Σ w L L^T, b = Σ w I L  (per pixel)
    A = np.einsum('hwn,nc,nd->hwcd', Wobs, L, L)   # H x W x 3 x 3
    b = np.einsum('hwn,nc,hwn->hwc', Wobs, L, I)   # H x W x 3

    # 5) Solve for G (albedo-scaled normals)
    A_ = A.reshape(-1, 3, 3)
    b_ = b.reshape(-1, 3)
    reg = 1e-6 * np.eye(3, dtype=np.float32)[None, :, :]
    g_ = np.linalg.solve(A_ + reg, b_)             # (H*W) x 3
    G = g_.reshape(H, W, 3)

    # 6) Unit normals & albedo (camera coords: nx, ny, nz)
    albedo = np.linalg.norm(G, axis=2, keepdims=True) + 1e-8
    Nrm = G / albedo                               # H x W x 3
    # DO NOT globally flip axes; encoding handled on save.
    Nrm[~mask] = 0.0
    albedo[~mask] = 0.0

    # 7) Save results
    np.save(os.path.join(args.out_dir, "normals_unit.npy"), Nrm.astype(np.float32))
    np.save(os.path.join(args.out_dir, "albedo.npy"), albedo.squeeze(-1).astype(np.float32))
    save_normals_png_dataset(Nrm, os.path.join(args.out_dir, "normal_map.png"))

    # Albedo preview
    alb = albedo.squeeze(-1)
    if np.any(mask):
        s = np.percentile(alb[mask], 99)
        if s <= 1e-8: s = 1.0
        alb_vis = np.clip(alb / s, 0, 1)
    else:
        alb_vis = np.clip(alb, 0, 1)
    cv2.imwrite(os.path.join(args.out_dir, "albedo_preview.png"), (alb_vis * 255).astype(np.uint8))

    print(f"[OK] Saved: {os.path.join(args.out_dir, 'normal_map.png')}")

    # 8) Copy mask & GT depth for downstream scripts
    mask_src = os.path.join(args.copy_from, "mask.png")
    depth_src = os.path.join(args.copy_from, "ground_truth", "depth_map.png")
    if os.path.isfile(mask_src):
        shutil.copy(mask_src, os.path.join(args.out_dir, "mask.png"))
        print(f"[OK] Copied mask from {mask_src}")
    gt_dest_dir = os.path.join(args.out_dir, "ground_truth")
    os.makedirs(gt_dest_dir, exist_ok=True)
    if os.path.isfile(depth_src):
        shutil.copy(depth_src, os.path.join(gt_dest_dir, "depth_map.png"))
        print(f"[OK] Copied GT depth from {depth_src}")

if __name__ == "__main__":
    main()