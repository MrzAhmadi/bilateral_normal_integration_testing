from PIL import Image
from pathlib import Path

script_dir = Path(__file__).resolve().parent
img = Image.open(script_dir / "normal_map.png")
white_img = Image.new(img.mode, img.size, "white")
white_img.save(script_dir / "mask.png")