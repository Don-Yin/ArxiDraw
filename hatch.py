import hatched
from pathlib import Path

image_path = Path("public", "vlad.jpg")
hatched.hatch(str(image_path), hatch_pitch=5, levels=(20, 100, 180), blur_radius=1, save_svg=True)
