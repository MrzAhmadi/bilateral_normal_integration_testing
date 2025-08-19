import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

MM_PER_GT_UNIT = 5000.0 / 65535.0   # GT 16-bit unit -> mm
MM_PER_PIXEL   = 4000.0 / 512.0     # pixel unit -> mm


def find_gt_path(root):
    candidates = [
        os.path.join(root, "ground_truth", "depth_map.png"),
        os.path.join(root, "ground_truth", "depth.png"),
        os.path.join(root, "depth.png"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "Ground-truth depth not found. Tried:\n" + "\n".join(candidates)
    )


def load_gt_mm(root):
    gt_path = find_gt_path(root)
    gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    if gt is None:
        raise FileNotFoundError(f"Unable to read GT image at {gt_path}")
    if gt.dtype != np.uint16:
        raise TypeError(f"GT must be 16-bit (uint16). Got {gt.dtype} at {gt_path}")
    gt_mm = gt.astype(np.float32) * MM_PER_GT_UNIT
    return gt_mm, gt_path


def load_est_mm(root):
    z_path = os.path.join(root, "z_pix.npy")
    if not os.path.isfile(z_path):
        raise FileNotFoundError(f"{z_path} not found. Run the integrator to produce z_pix.npy.")
    z = np.load(z_path).astype(np.float32)
    return z * MM_PER_PIXEL, z_path


def load_mask(root, shape):
    m = cv2.imread(os.path.join(root, "mask.png"), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.ones(shape, dtype=bool)
    if m.shape != shape:
        raise ValueError(f"Mask shape {m.shape} does not match GT shape {shape}.")
    return m > 0


def fit_affine_est_to_gt(est_mm, gt_mm, mask):
    """
    Solve gt ~ a*est + b in least squares sense on masked pixels.
    Returns: est_aligned, a, b
    """
    est_v = est_mm[mask].reshape(-1, 1)
    gt_v  = gt_mm[mask].reshape(-1, 1)
    X = np.hstack([est_v, np.ones_like(est_v)])  # [est, 1]
    coef = np.linalg.lstsq(X, gt_v, rcond=None)[0].ravel()
    a, b = float(coef[0]), float(coef[1])
    est_aligned = a * est_mm + b
    return est_aligned, a, b


def norm01_for_viz(x, mask, p_lo=2, p_hi=98, lo_hi=None):
    x = x.copy()
    x[~mask] = np.nan
    v = x[np.isfinite(x)]
    if v.size == 0:
        return np.zeros_like(x, dtype=np.uint8)

    if lo_hi is None:
        lo, hi = np.percentile(v, p_lo), np.percentile(v, p_hi)
    else:
        lo, hi = lo_hi

    if hi <= lo:
        hi = lo + 1e-6
    y = (x - lo) / (hi - lo)
    y = np.clip(np.nan_to_num(y, nan=0.0), 0, 1)
    return (y * 255).astype(np.uint8)


def save_with_colorbar(path, img_mm, mask, vmin, vmax, cmap="viridis", title=""):
    """Save a depth map with a shared colormap and a colorbar."""
    plt.figure()
    mimg = img_mm.copy().astype(float)
    mimg[~mask] = np.nan
    im = plt.imshow(mimg, vmin=vmin, vmax=vmax, cmap=cmap)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04, label="mm")
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()


def save_artifacts(out_dir, gt_mm, est_mm_aligned, mask, viz_cmap="viridis", p_lo=2, p_hi=98):
    os.makedirs(out_dir, exist_ok=True)

    # Absolute error (in mm)
    abs_err = np.full_like(gt_mm, np.nan, dtype=np.float32)
    abs_err[mask] = np.abs(est_mm_aligned - gt_mm)[mask]

    # Error heatmap (normalize to 95th percentile)
    valid = abs_err[np.isfinite(abs_err)]
    scale = max(np.percentile(valid, 95), 1e-6) if valid.size > 0 else 1.0
    vis = np.clip(np.nan_to_num(abs_err, nan=0.0) / scale, 0, 1)
    err_u8 = (vis * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, "error_mm_abs.png"), err_u8)

    # Shared display range from GT (percentiles)
    v_gt = gt_mm[mask]
    lo = np.percentile(v_gt, p_lo)
    hi = np.percentile(v_gt, p_hi)
    lo_hi = (lo, hi)

    # Save grayscale normalized (legacy panel, consistent lo/hi)
    gt_u8  = norm01_for_viz(gt_mm,          mask, lo_hi=lo_hi)
    est_u8 = norm01_for_viz(est_mm_aligned, mask, lo_hi=lo_hi)
    cv2.imwrite(os.path.join(out_dir, "gt_mm_norm.png"), gt_u8)
    cv2.imwrite(os.path.join(out_dir, "est_mm_aligned_norm.png"), est_u8)

    # Side-by-side grayscale panel
    h = gt_u8.shape[0]
    bar = np.full((h, 8), 255, np.uint8)
    panel = np.hstack([gt_u8, bar, est_u8, bar, err_u8])
    cv2.imwrite(os.path.join(out_dir, "compare_gt_est_err.png"), panel)

    save_with_colorbar(
        os.path.join(out_dir, "gt_mm_norm_cb.png"),
        gt_mm, mask, vmin=lo, vmax=hi, cmap=viz_cmap, title="GT depth (mm)"
    )
    save_with_colorbar(
        os.path.join(out_dir, "est_mm_aligned_norm_cb.png"),
        est_mm_aligned, mask, vmin=lo, vmax=hi, cmap=viz_cmap, title="Estimated depth aligned (mm)"
    )


def main():
    ap = argparse.ArgumentParser(description="Evaluate integrated depth (z_pix.npy) against GT depth.png")
    ap.add_argument("--path", "-p", required=True, help="Dataset folder, e.g. data/Fig8_wallrelief")
    ap.add_argument("--viz_cmap", default="viridis", help="Matplotlib colormap name (e.g., viridis, viridis_r, plasma)")
    ap.add_argument("--p_lo", type=float, default=2.0, help="Lower percentile for display range (shared)")
    ap.add_argument("--p_hi", type=float, default=98.0, help="Upper percentile for display range (shared)")
    args = ap.parse_args()

    # Output dir
    out_dir = os.path.join(args.path, "eval_results")
    os.makedirs(out_dir, exist_ok=True)

    gt_mm, gt_path   = load_gt_mm(args.path)
    est_mm, z_path   = load_est_mm(args.path)
    if gt_mm.shape != est_mm.shape:
        raise ValueError(f"Shape mismatch: GT {gt_mm.shape} vs est {est_mm.shape}")

    mask = load_mask(args.path, gt_mm.shape)
    valid = mask & np.isfinite(gt_mm) & np.isfinite(est_mm)

    # Fit affine transform (fixes sign, scale, average)
    est_mm_aligned, a_scale, b_offset = fit_affine_est_to_gt(est_mm, gt_mm, valid)
    flipped = (a_scale < 0)

    # Metrics
    diff    = (est_mm_aligned - gt_mm)[valid]
    mae_mm  = float(np.mean(np.abs(diff)))
    rmse_mm = float(np.sqrt(np.mean(diff**2)))
    mae_px  = mae_mm / MM_PER_PIXEL
    rmse_px = rmse_mm / MM_PER_PIXEL

    # Console summary
    print("=== Evaluation (GT vs Integrated Depth) ===")
    print(f"GT path              : {gt_path}")
    print(f"Estimate path        : {z_path}")
    print(f"GT unit -> mm        : {MM_PER_GT_UNIT:.9f} mm / GT-unit")
    print(f"Pixel -> mm          : {MM_PER_PIXEL:.9f} mm / pixel")
    print(f"Affine scale (a)     : {a_scale:.6f} ({'flip' if flipped else 'no flip'})")
    print(f"Affine offset (b,mm) : {b_offset:.6f}")
    print(f"MAE                  : {mae_mm:.6f} mm   ({mae_px:.6f} px)")
    print(f"RMSE                 : {rmse_mm:.6f} mm  ({rmse_px:.6f} px)")
    print(f"Valid pixels         : {int(valid.sum())} / {gt_mm.size}")

    # Save images (grayscale panel + colorbar images)
    save_artifacts(out_dir, gt_mm, est_mm_aligned, valid, viz_cmap=args.viz_cmap, p_lo=args.p_lo, p_hi=args.p_hi)

    # Save CSV metrics
    csv_path = os.path.join(out_dir, "depth_eval_metrics.csv")
    with open(csv_path, "w") as f:
        f.write("gt_path,z_path,mm_per_gt_unit,mm_per_pixel,a_scale,b_offset,mae_mm,rmse_mm,mae_px,rmse_px,valid_px,total_px\n")
        f.write(f"{gt_path},{z_path},{MM_PER_GT_UNIT},{MM_PER_PIXEL},{a_scale:.6f},{b_offset:.6f},{mae_mm:.6f},{rmse_mm:.6f},{mae_px:.6f},{rmse_px:.6f},{int(valid.sum())},{gt_mm.size}\n")

    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()