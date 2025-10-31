import sys
import os
from pathlib import Path
import torch
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
vggt_path = parent_dir / "vggt"

if vggt_path.exists():
    sys.path.insert(0, str(vggt_path))
else:
    print(f"ERROR: vggt path not found at {vggt_path}")
    print("Make sure the vggt repository is cloned in the correct location")
    sys.exit(1)


import subprocess
from pathlib import Path

import os
import shutil
import random
from pathlib import Path

def split_large_image_folder(src_folder, max_images: int = 40):
    src = Path(src_folder)
    img_dir = src / "images"
    
    if not img_dir.is_dir():
        raise ValueError(f"'images' subfolder not found in {src}")
    
    image_files = [f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}]
    
    if len(image_files) <= max_images:
        return None  # No split needed
    
    # Choose random subset
    selected = random.sample(image_files, max_images)
    
    # Create new folder at same level
    new_folder = src.parent / f"{src.name}_vggt"
    new_img_dir = new_folder / "images"
    new_img_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy selected images
    for img in selected:
        shutil.copy2(img, new_img_dir / img.name)
    
    print(f"Created subset with {max_images} images: {new_folder}")
    return new_folder


if __name__ == "__main__":
    # ---- Your arguments in a dict ----
    args = {
        "scene_dir": "data/gerrard-hall/",
        "use_ba": False,
        # add/remove any others as needed
    }

    new_folder = split_large_image_folder(args["scene_dir"], max_images=40)
    if new_folder is None:
        new_folder = args["scene_dir"]

    # ---- Build CLI string ----
    cli = [f"--{k.replace('_', '-')} {v}" if not isinstance(v, bool) else f"--{k.replace('_', '-')}" if v else ""
        for k, v in args.items()]
    cli = " ".join([x for x in cli if x])  # drop empty

    #cmd = f"python ../vggt/demo_colmap.py --scene-dir {args['scene_dir']}"

    cmd = f"python ../vggt/demo_colmap.py --scene_dir={new_folder}"
    # ---- Run ----
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)