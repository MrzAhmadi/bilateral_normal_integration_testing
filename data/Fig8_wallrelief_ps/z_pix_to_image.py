import numpy as np
import cv2
import os

path = "data/Fig8_wallrelief_ps"
z = np.load(os.path.join(path, "z_pix.npy"))

# Normalize to 0–255 for visualization
z_norm = (z - np.nanmin(z)) / (np.nanmax(z) - np.nanmin(z) + 1e-8)
z_uint8 = (z_norm * 255).astype(np.uint8)

# Apply a colormap (viridis)
z_color = cv2.applyColorMap(z_uint8, cv2.COLORMAP_VIRIDIS)

# Save as PNG
out_path = os.path.join(path, "z_pix.png")
cv2.imwrite(out_path, z_color)
print(f"Saved visualization to {out_path}")