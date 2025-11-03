import sys
import os
from pathlib import Path
import subprocess
import shutil
import random
import warnings
from utils import split_dataset, split_dataset_no_overlap

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

def mast3r_reconstruction(dataset_path="data/gerrard-hall/images/", output_dir="reconstruction_output/"):
    args = {
        "scene_dir": dataset_path,
        "max_images": 40,
        "model_name": "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
        "retrieval_model": "../mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth",
        "scene_graph": "retrieval-30-5",
        "output_dir": output_dir,
    }

 
    image_dir_str = dataset_path

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

import subprocess
from pathlib import Path
from typing import List

def merge_models_incrementally(
    model_dirs: List[Path],
    output_dir: Path,
    temp_dir: Path,
    keep_intermediates: bool = False,
) -> Path:
    """
    Incrementally merge a list of COLMAP sparse models using `model_merger`.

    Parameters
    ----------
    model_dirs : list[Path]
        List of folders that each contain a valid COLMAP sparse model
        (cameras.bin, images.bin, points3D.bin).
    output_dir : Path
        Where the final merged model will be written.
    temp_dir : Path
        Scratch folder for intermediate merged models.
    keep_intermediates : bool
        If True, keep every intermediate merge (useful for debugging).

    Returns
    -------
    Path
        Path to the final merged model folder.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Start with the first model – just copy it to the first temp slot
    # ------------------------------------------------------------------
    merged_model = temp_dir / "merge_01"
    merged_model.mkdir(exist_ok=True)

    # copy the first reconstruction verbatim
    for f in model_dirs[0].glob("*.bin"):
        (merged_model / f.name).write_bytes(f.read_bytes())

    temp_files = [merged_model]          # keep track of intermediates
    print(f"Initialized merge with model 1: {model_dirs[0].name}")

    # ------------------------------------------------------------------
    # 2. Iteratively merge the remaining models
    # ------------------------------------------------------------------
    for i, current_dir in enumerate(model_dirs[1:], start=2):
        temp_output = temp_dir / f"merge_{i:02d}"
        temp_output.mkdir(exist_ok=True)
        temp_files.append(temp_output)

        cmd = [
            "colmap", "model_merger",
            "--input_path1", str(merged_model),
            "--input_path2", str(current_dir),
            "--output_path", str(temp_output),
            # optional flags (tune if you know the models are already aligned)
            # "--min_common_images", "3",
            # "--align_cameras", "1",
        ]

        print(
            f"Merging model {i}/{len(model_dirs)}: "
            f"{merged_model.name} + {current_dir.name} → {temp_output.name}"
        )

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error_msg = result.stderr.strip()
            print(f"COLMAP model_merger ERROR:\n{error_msg}")
            raise RuntimeError(
                f"Model merging failed at step {i}: {error_msg}"
            )

        # success → the next iteration uses this merged folder
        merged_model = temp_output

        # (optional) print a short summary
        summary = "\n".join(
            line for line in result.stdout.splitlines()
            if "Merged images" in line or "Merged points" in line
        )
        if summary:
            print(summary)

    # ------------------------------------------------------------------
    # 3. Move the final result to the requested output folder
    # ------------------------------------------------------------------
    final_merged = output_dir / "merged_model"
    final_merged.mkdir(exist_ok=True)

    for f in merged_model.glob("*.bin"):
        (final_merged / f.name).write_bytes(f.read_bytes())

    # ------------------------------------------------------------------
    # 4. Clean-up intermediates (keep only the final one)
    # ------------------------------------------------------------------
    if not keep_intermediates:
        for tmp in temp_files:
            if tmp.exists() and tmp != final_merged:
                for bin_file in tmp.glob("*.bin"):
                    bin_file.unlink()
                tmp.rmdir()

    print(f"Successfully merged all {len(model_dirs)} models into: {final_merged}")
    return final_merged


# ----------------------------------------------------------------------
# Example usage (replace with your actual paths)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Suppose you have 5 incremental reconstructions:
    recon_dirs = [
        Path("mast3r_full_recon_gerrard/merged_reconstruction/split_1"),
        Path("mast3r_full_recon_gerrard/merged_reconstruction/split_2"),
        Path("mast3r_full_recon_gerrard/merged_reconstruction/split_3"),
        Path("mast3r_full_recon_gerrard/merged_reconstruction/split_4"),
        Path("mast3r_full_recon_gerrard/merged_reconstruction/split_5"),
    ]

    final = merge_models_incrementally(
        model_dirs=recon_dirs,
        output_dir=Path("mast3r_full_recon_gerrard/merged_reconstruction/final"),
        temp_dir=Path("mast3r_full_recon_gerrard/merged_reconstruction/.temp_merge"),
        keep_intermediates=False,
    )

def mast3r_full_dataset(
    input_dataset_path,
    output_base_dir="full_reconstruction",
    dataset_name=None,
    split_size=40,
    min_overlap=0,
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
        out_folder=str(temp_splits_dir),
        dataset_name=dataset_name,
        split_size=split_size,
        seed=seed
    )
    


    split_dirs = []
    for item in os.listdir(temp_splits_dir):
        item_path = os.path.join(temp_splits_dir, item)
        if os.path.isdir(item_path):
            split_dirs.append(item)

    
    if not split_dirs:
        raise RuntimeError("split_dataset did not create any 'split_*' folders in expected locations.")
    

    
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
        print(f"mast3r_reconstruction(dataset_path={str(split_dir)}, output_dir={str(split_output_dir)})...")
        mast3r_reconstruction(
            dataset_path=str(split_dir),
            output_dir=str(split_output_dir)
        )
        
        db_path = split_output_dir / "colmap.db"
        recon_dir = split_output_dir / "reconstruction" / "0"
        
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        if not recon_dir.exists():
            raise FileNotFoundError(f"Reconstruction not found: {recon_dir}")
        
        db_paths.append(db_path)
        recon_dirs.append(recon_dir)

    subdirs = []

    for name in os.listdir(split_dir):
        if os.path.isdir(os.path.join(split_dir, name)):
            subdirs.append(name)

    merge_models_incrementally(model_dirs=subdirs, output_dir=f"{output_base_dir}/merged/", temp_dir=temp_splits_dir)

    # Step 3: Merge databases
    if len(db_paths) < 2:
        raise ValueError("At least 2 databases are required for merging.")

    merged_db = db_paths[0]
    temp_dir = Path(output_base_dir).parent
    temp_files = []

    print(f"Starting incremental merge of {len(db_paths)} databases...")

    # Start merging from the second database (index 1)
    for i, current_db in enumerate(db_paths[1:], start=2):
        temp_output = temp_dir / f"temp_merge_up_to_{i}.db"
        temp_files.append(temp_output)

        # CRITICAL FIX: Delete existing merged database if it exists
        if temp_output.exists():
            print(f"Removing existing file: {temp_output}")
            temp_output.unlink()  # Safe delete

        cmd = [
            "colmap", "database_merger",
            "--database_path1", str(merged_db),
            "--database_path2", str(current_db),
            "--merged_database_path", str(temp_output)
        ]

        print(f"Merging database {i}/{len(db_paths)}: {Path(merged_db).name} + {Path(current_db).name} → {temp_output.name}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error_msg = result.stderr.strip()
            print(f"COLMAP Error: {error_msg}")
            raise RuntimeError(f"Database merging failed at step {i}: {error_msg}")

        # Update merged_db to be the new merged result for next iteration
        merged_db = temp_output

    # Optional: Clean up intermediate temp files (keep only the final one)
    final_merged_db = temp_files[-1] if temp_files else merged_db
    for temp_file in temp_files[:-1]:  # Keep only the last one
        if temp_file.exists():
            temp_file.unlink()

    print(f"Successfully merged all databases into: {final_merged_db}")
        

    
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

    return
    


if __name__ == "__main__":
    #mast3r_reconstruction()
    mast3r_full_dataset(
    input_dataset_path="data/gerrard-hall/images/",
    output_base_dir="mast3r_full_recon_gerrard",
    split_size=40,
    min_overlap=10,
    seed=42,
    keep_temp_splits=False
)


