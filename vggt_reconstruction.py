import sys
import os
from pathlib import Path
import torch
import numpy as np
import random
import warnings
import shutil
from utils import split_large_image_folder
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

def vggt_reconstruction(dataset_path, output_dir):
    args = {
        "scene_dir": f"./{dataset_path}/",
        "use_ba": False,
        
    }
    images_dir = os.path.join(dataset_path, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        for f in os.listdir(dataset_path):
            p = os.path.join(dataset_path, f)
            if os.path.isfile(p):
                shutil.move(p, os.path.join(images_dir, f))

    #new_folder = split_large_image_folder(args["scene_dir"], "vggt", max_images=40)
    #if new_folder is None:
    #    new_folder = args["scene_dir"]


    #cmd = f"python ../vggt/demo_colmap.py --scene-dir {args['scene_dir']}"

    cmd = f"python ../vggt/demo_colmap.py --scene_dir={dataset_path} --max_query_pts=1024 --query_frame_num=5"
    # ---- Run ----
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)




if __name__ == "__main__":
    vggt_reconstruction("data/gerrard-hall/")
    