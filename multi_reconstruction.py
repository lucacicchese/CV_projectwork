import os
import subprocess
from model_merger import merge_all_colmap_models
from utils import split_dataset
from mast3r_reconstruction import mast3r_reconstruction
from vggt_reconstruction import vggt_reconstruction
import shutil


def multi_stage_reconstruction(dataset_folder, output_folder, model_name):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Step 1: SPLIT DATASET WITH OVERLAP
    splits_dir = f"{output_folder}/splits"
    split_dataset(
        input_folder=dataset_folder,
        out_folder=splits_dir
    )

    # Step 2: RUN RECONSTRUCTION
    if not os.path.exists(splits_dir):
        raise RuntimeError(f"Splits directory not found: {splits_dir}")
    
    split_paths = sorted([
        os.path.join(splits_dir, d) 
        for d in os.listdir(splits_dir) 
        if os.path.isdir(os.path.join(splits_dir, d))
    ])
    
    if len(split_paths) == 0:
        raise RuntimeError(f"No split directories found in {splits_dir}")
    
    reconstructions_dir = f"{output_folder}/reconstructions"
    os.makedirs(reconstructions_dir, exist_ok=True)
    
    for i, split_path in enumerate(split_paths):
        i += 1
        if not os.path.exists(split_path):
            raise RuntimeError(f"Split path does not exist: {split_path}")
        
        split_number = i
        recon_output = f"{reconstructions_dir}/{split_number}"
        os.makedirs(recon_output, exist_ok=True)
        
        if model_name == "mast3r":
            mast3r_reconstruction(
                dataset_path=split_path,
                output_dir=recon_output
            )
        elif model_name == "vggt":
            vggt_reconstruction(
                split_path,
                output_dir=recon_output
            )
            for name in os.listdir(splits_dir):
                src = os.path.join(splits_dir, name, "sparse")
                dst = os.path.join(reconstructions_dir, name)
                if os.path.isdir(src):
                    os.makedirs(dst, exist_ok=True)
                    for f in os.listdir(src):
                        shutil.move(os.path.join(src, f), os.path.join(dst, f))

    # Step 3: MERGE RECONSTRUCTIONS
    merge_all_colmap_models(
        root_folder=reconstructions_dir,
        final_output_dir=f"{output_folder}/final_reconstruction")

if __name__ == "__main__":
    model_name = "mast3r"  # or "vggt"
    dataset_folder = "data/gerrard-hall/images/"
    output_folder = f"data/gerrard-hall/{model_name}_multi"

    
    multi_stage_reconstruction(dataset_folder, output_folder, model_name)