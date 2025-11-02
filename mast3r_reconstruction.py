import sys
import os
from pathlib import Path
import subprocess
import shutil
import random
import warnings
from utils import split_dataset

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

def mast3r_reconstruction(dataset_path="data/gerrard-hall", output_dir="reconstruction_output/"):
    args = {
        "scene_dir": dataset_path,
        "max_images": 40,
        "model_name": "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
        "retrieval_model": "../mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth",
        "scene_graph": "retrieval-30-5",
        "output_dir": output_dir,
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
    

import os
import subprocess
import shutil
from pathlib import Path
import tempfile

def mast3r_full_dataset(
    input_dataset_path,
    output_base_dir="full_reconstruction",
    dataset_name=None,
    split_size=40,
    min_overlap=5,
    seed=None,
    keep_temp_splits=False
):
    """
    Full MAST3R pipeline:
    1. Split input dataset into overlapping splits using split_dataset()
    2. Run mast3r_reconstruction() on each split
    3. Merge all COLMAP databases using colmap database_merger
    
    Args:
        input_dataset_path (str or Path): Path to the input image dataset
        output_base_dir (str): Base output directory
        dataset_name (str, optional): Name used in split_dataset
        split_size (int): Number of images per split
        min_overlap (int): Minimum overlap between consecutive splits
        seed (int, optional): Random seed for reproducibility
        keep_temp_splits (bool): If False, delete temporary split folders after use
    
    Returns:
        dict: Paths to merged database and individual reconstructions
    """
    input_path = Path(input_dataset_path)
    output_base = Path(output_base_dir)
    output_base.mkdir(exist_ok=True)
    
    if not input_path.exists():
        raise ValueError(f"Input dataset path does not exist: {input_path}")
    
    # Step 1: Create temporary directory for splits
    temp_splits_dir = output_base / "temp_splits"
    if temp_splits_dir.exists():
        shutil.rmtree(temp_splits_dir)
    temp_splits_dir.mkdir(exist_ok=True)
    
    print(f"Splitting dataset: {input_path}")
    print(f"   split_size={split_size}, min_overlap={min_overlap}, seed={seed}")
    
    # Call your split_dataset function
    split_dataset(
        input_folder=str(input_path),
        dataset_name=dataset_name,
        split_size=split_size,
        min_overlap=min_overlap,
        seed=seed
    )
    
    # The split_dataset function should create folders like:
    # temp_splits/split_1, temp_splits/split_2, ...
    # But we need to know where it outputs them.
    # ASSUMPTION: split_dataset creates splits inside the input_folder or a subfolder.
    # We'll search for split_* directories in input_path or input_path parent.
    
    possible_split_locations = [
        input_path,  # splits created inside input_folder
        input_path.parent,  # splits created next to input_folder
    ]
    
    split_dirs = []
    for loc in possible_split_locations:
        candidates = sorted([d for d in loc.iterdir() 
                           if d.is_dir() and d.name.startswith("split_")])
        if candidates:
            split_dirs = candidates
            print(f"Found {len(split_dirs)} splits in: {loc}")
            break
    
    if not split_dirs:
        raise RuntimeError("split_dataset did not create any 'split_*' folders in expected locations.")
    
    # Move splits to temp directory for clean organization (optional but clean)
    for split_dir in split_dirs:
        dest = temp_splits_dir / split_dir.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(split_dir), str(dest))
    
    split_dirs = sorted(temp_splits_dir.iterdir())
    print(f"Processing {len(split_dirs)} splits: {[d.name for d in split_dirs]}")
    
    # Step 2: Run reconstruction on each split
    db_paths = []
    recon_dirs = []
    
    per_split_output = output_base / "per_split_reconstructions"
    per_split_output.mkdir(exist_ok=True)
    
    for i, split_dir in enumerate(split_dirs):
        print(f"\nReconstructing {split_dir.name} ({i+1}/{len(split_dirs)})...")
        
        split_output_dir = per_split_output / split_dir.name
        split_output_dir.mkdir(exist_ok=True)
        
        # Run MAST3R reconstruction
        mast3r_reconstruction(
            dataset_path=str(split_dir),
            output_dir=str(split_output_dir)
        )
        
        db_path = split_output_dir / "database.db"
        recon_dir = split_output_dir / "reconstruction" / "0"
        
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        if not recon_dir.exists():
            raise FileNotFoundError(f"Reconstruction not found: {recon_dir}")
        
        db_paths.append(db_path)
        recon_dirs.append(recon_dir)
    
    # Step 3: Merge databases
    print(f"\nMerging {len(db_paths)} databases...")
    merged_db_path = output_base / "merged_database.db"
    
    cmd = ["colmap", "database_merger"]
    cmd += [f"--database_path{i+1}={db}" for i, db in enumerate(db_paths)]
    cmd += [f"--merged_database_path={merged_db_path}"]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("COLMAP merge failed:")
        print(result.stderr)
        raise RuntimeError(f"Database merging failed: {result.stderr}")
    
    print(f"Merged database: {merged_db_path}")
    
    # Step 4: Copy individual reconstructions to final merged folder
    merged_recon_dir = output_base / "merged_reconstruction"
    merged_recon_dir.mkdir(exist_ok=True)
    
    for recon_dir in recon_dirs:
        split_name = recon_dir.parents[1].name  # e.g., split_1
        dest = merged_recon_dir / split_name
        dest.mkdir(exist_ok=True)
        
        for bin_file in recon_dir.glob("*.bin"):
            shutil.copy(bin_file, dest / bin_file.name)
    
    print(f"Individual reconstructions saved in: {merged_recon_dir}")
    
    # Step 5: Cleanup (optional)
    if not keep_temp_splits:
        print("Cleaning up temporary split directories...")
        shutil.rmtree(temp_splits_dir)
    else:
        print(f"Temporary splits preserved at: {temp_splits_dir}")
    
    return {
        "merged_database": str(merged_db_path),
        "individual_reconstructions": [str(d) for d in recon_dirs],
        "merged_reconstruction_dir": str(merged_recon_dir),
        "temp_splits_dir": str(temp_splits_dir) if keep_temp_splits else None
    }

if __name__ == "__main__":
    #mast3r_reconstruction()
    result = mast3r_full_dataset(
    input_dataset_path="data/gerrard-hall",
    output_base_dir="mast3r_full_recon_gerrard",
    split_size=40,
    min_overlap=10,
    seed=42,
    keep_temp_splits=False
)

print(result["merged_database"])
