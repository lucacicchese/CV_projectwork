import sys
import os
from pathlib import Path
import subprocess
import shutil
import random
import warnings
from utils import split_large_image_folder

warnings.filterwarnings("ignore", category=FutureWarning)

# Set up paths
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
mast3r_path = parent_dir / "mast3r"

if not mast3r_path.exists():
    print(f"ERROR: mast3r path not found at {mast3r_path}")
    print("Make sure the mast3r repository is cloned in the correct location")
    sys.exit(1)

sys.path.insert(0, str(mast3r_path))



if __name__ == "__main__":
    args = {
        "scene_dir": "data/gerrard-hall/",
        "max_images": 40,
        "model_name": "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
        "retrieval_model": "../mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth",
        "scene_graph": "retrieval-30-5",
        "output_dir": "reconstruction_output/",
    }

    # Step 1: Split large folder
    new_folder = split_large_image_folder(args["scene_dir"], "mast3r", max_images=args["max_images"])
    if new_folder is None:
        subset_dir = Path(args["scene_dir"]) / "images"
        print(f"No split needed. Using: {subset_dir}")
    else:
        subset_dir = new_folder / "images"
        print(f"Using subset: {subset_dir}")

    image_dir_str = str(subset_dir) + "/"

    # Step 2: Clean output directory and remove old database
    output_path = Path(args["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)

    colmap_db_path = output_path / "database.db"
    if colmap_db_path.exists():
        print(f"Removing existing COLMAP database: {colmap_db_path}")
        colmap_db_path.unlink()  # Delete it

    # Step 3: Make pairs
    pairs_file = output_path / "pairs.txt"  # Save in output dir for clarity

    make_pairs = [
        "python", str(mast3r_path / "make_pairs.py"),
        "--dir", image_dir_str,
        "--output", str(pairs_file),
        "--model_name", args["model_name"],
        "--retrieval_model", args["retrieval_model"],
        "--scene_graph", args["scene_graph"]
    ]

    print(f"\nRunning pair generation...")
    print(" ".join(make_pairs))
    result1 = subprocess.run(make_pairs, check=True, cwd=current_dir)
    if result1.returncode != 0:
        print("Failed to generate pairs.")
        sys.exit(1)

    # Step 4: Command 2 - Run MAST3R mapping
    mapping = [
        "python", str(mast3r_path / "kapture_mast3r_mapping.py"),
        "--dir", image_dir_str,
        "--pairsfile_path", str(pairs_file),
        "--model_name", args["model_name"],
        "-o", str(output_path) + "/"
    ]

    print(f"\nRunning MAST3R mapping...")
    print(" ".join(mapping))
    result2 = subprocess.run(mapping, check=True, cwd=current_dir)
    if result2.returncode != 0:
        print("MAST3R mapping failed.")
        sys.exit(1)

    print(f"\nSuccess! Reconstruction saved to: {output_path}")