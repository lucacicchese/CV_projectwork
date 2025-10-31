import sys
import os
from pathlib import Path
import subprocess
import shutil
import random
import warnings

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


def split_large_image_folder(src_folder, max_images: int = 40):
    src = Path(src_folder)
    img_dir = src / "images"

    if not img_dir.is_dir():
        raise ValueError(f"'images' subfolder not found in {src}")

    image_files = [
        f for f in img_dir.iterdir()
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    ]

    if len(image_files) <= max_images:
        return None

    selected = random.sample(image_files, max_images)
    new_folder = src.parent / f"{src.name}_subset"
    new_img_dir = new_folder / "images"
    new_img_dir.mkdir(parents=True, exist_ok=True)

    for img in selected:
        shutil.copy2(img, new_img_dir / img.name)

    print(f"Created subset with {max_images} images: {new_folder}")
    return new_folder


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
    new_folder = split_large_image_folder(args["scene_dir"], max_images=args["max_images"])
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

    # Step 3: Command 1 - Make pairs
    pairs_file = output_path / "pairs.txt"  # Save in output dir for clarity

    cmd1 = [
        "python", str(mast3r_path / "make_pairs.py"),
        "--dir", image_dir_str,
        "--output", str(pairs_file),
        "--model_name", args["model_name"],
        "--retrieval_model", args["retrieval_model"],
        "--scene_graph", args["scene_graph"]
    ]

    print(f"\nRunning pair generation...")
    print(" ".join(cmd1))
    result1 = subprocess.run(cmd1, check=True, cwd=current_dir)
    if result1.returncode != 0:
        print("Failed to generate pairs.")
        sys.exit(1)

    # Step 4: Command 2 - Run MAST3R mapping
    cmd2 = [
        "python", str(mast3r_path / "kapture_mast3r_mapping.py"),
        "--dir", image_dir_str,
        "--pairsfile_path", str(pairs_file),
        "--model_name", args["model_name"],
        "-o", str(output_path) + "/"
    ]

    print(f"\nRunning MAST3R mapping...")
    print(" ".join(cmd2))
    result2 = subprocess.run(cmd2, check=True, cwd=current_dir)
    if result2.returncode != 0:
        print("MAST3R mapping failed.")
        sys.exit(1)

    print(f"\nSuccess! Reconstruction saved to: {output_path}")