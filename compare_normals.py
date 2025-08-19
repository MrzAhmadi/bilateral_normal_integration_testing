# compare_normals.py
import os, cv2, numpy as np
import matplotlib.pyplot as plt

def decode_dataset_normals(path):
    """
    Dataset encoding on disk (RGB in [0..255]):
      R = ny, G = nx, B = -nz     (each mapped from [-1,1] to [0,255])
    We decode to camera coordinates (nx, ny, nz) with unit length.
    """
    rgb = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if rgb is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32)
    nraw = rgb / 255.0 * 2.0 - 1.0  # back to [-1,1] per channel (R,G,B)
    nx = nraw[..., 1]
    ny = nraw[..., 0]
    nz = -nraw[..., 2]
    n = np.stack([nx, ny, nz], axis=-1)
    n /= (np.linalg.norm(n, axis=2, keepdims=True) + 1e-8)
    return n, rgb.astype(np.uint8)

def angular_error_deg(n1, n2, mask=None):
    dot = np.sum(n1 * n2, axis=2)
    dot = np.clip(dot, -1.0, 1.0)
    ang = np.degrees(np.arccos(dot))
    if mask is not None:
        ang = np.where(mask, ang, np.nan)
    return ang

def main():
    gt_png = "data/Fig8_wallrelief/normal_map.png"       # GT normals (encoded)
    ps_png = "data/Fig8_wallrelief_ps/normal_map.png"    # PS normals (encoded)
    mask_p = "data/Fig8_wallrelief/mask.png"
    outdir = "data/Fig8_wallrelief_ps/normals_compare"
    os.makedirs(outdir, exist_ok=True)

    gt_n, gt_rgb = decode_dataset_normals(gt_png)
    ps_n, ps_rgb = decode_dataset_normals(ps_png)

    if gt_n.shape != ps_n.shape:
        raise ValueError(f"Shape mismatch: {gt_n.shape} vs {ps_n.shape}")

    m = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
    mask = (m > 0) if m is not None and m.shape[:2] == gt_n.shape[:2] else np.ones(gt_n.shape[:2], bool)

    # Try both orientations for PS (global sign ambiguity)
    ang_as_is = angular_error_deg(gt_n, ps_n, mask=mask)
    ang_flipped = angular_error_deg(gt_n, -ps_n, mask=mask)

    mean_as_is = np.nanmean(ang_as_is); med_as_is = np.nanmedian(ang_as_is)
    mean_flip  = np.nanmean(ang_flipped); med_flip  = np.nanmedian(ang_flipped)

    if mean_flip < mean_as_is:
        ang = ang_flipped
        chosen = "global flip applied"
        chosen_mean, chosen_median = mean_flip, med_flip
    else:
        ang = ang_as_is
        chosen = "no flip"
        chosen_mean, chosen_median = mean_as_is, med_as_is

    # Save side-by-side as provided (encoded form) for human inspection
    cv2.imwrite(os.path.join(outdir, "gt_normal_rgb.png"), cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(outdir, "ps_normal_rgb.png"), cv2.cvtColor(ps_rgb, cv2.COLOR_RGB2BGR))

    # Heatmap with colorbar
    plt.figure()
    ang_vis = np.nan_to_num(ang, nan=0.0)
    im = plt.imshow(ang_vis, vmin=0, vmax=np.nanpercentile(ang_vis, 99))
    plt.title(f"Angular error (deg) [{chosen}] | mean={chosen_mean:.2f}, median={chosen_median:.2f}")
    plt.axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04, label="deg")
    plt.savefig(os.path.join(outdir, "angular_error_deg.png"), bbox_inches='tight', dpi=150)
    plt.close()

    print(f"[OK] Saved to {outdir}")
    print(f"Mean angular error: {chosen_mean:.2f} deg | Median: {chosen_median:.2f} deg | ({chosen})")

if __name__ == "__main__":
    main()