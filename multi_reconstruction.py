import os
import subprocess
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
    
    # Step 2: RUN CORRECT RECONSTRUCTION
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

    
    # Step 3 & 4: COMPUTE COMMON IMAGES AND MERGE MODELS INCREMENTALLY
    if len(split_paths) < 2:
        # Only one split: just copy it to final
        final_recon_path = f"{output_folder}/final_reconstruction"
        os.makedirs(final_recon_path, exist_ok=True)
        src = f"{reconstructions_dir}/0"
        if os.path.exists(final_recon_path):
            shutil.rmtree(final_recon_path)
        if os.path.exists(src):
            shutil.copytree(src, final_recon_path)
        return
    
    # Start with first reconstruction
    current_recon_path = f"{reconstructions_dir}/0"
    if not os.path.exists(current_recon_path):
        raise RuntimeError(f"Reconstruction not found: {current_recon_path}")
    
    current_split_images = set(os.listdir(split_paths[0]))
    
    for i in range(1, len(split_paths)):
        next_split_path = split_paths[i]
        next_recon_path = f"{reconstructions_dir}/{i}"
        next_split_images = set(os.listdir(next_split_path))
        
        if not os.path.exists(next_recon_path):
            raise RuntimeError(f"Reconstruction not found: {next_recon_path}")
        
        # Compute common images
        common_files = sorted(current_split_images.intersection(next_split_images))
        
        common_txt_path = f"{output_folder}/common_images_{i}.txt"
        with open(common_txt_path, 'w') as f:
            for fname in common_files:
                f.write(f"{fname}\n")
        
        # Merge
        merged_recon_path = f"{output_folder}/merged_reconstruction_{i}"
        os.makedirs(merged_recon_path, exist_ok=True)
        
        cmd = [
            "colmap", "models_merger",
            "--input_path1", current_recon_path,
            "--input_path2", next_recon_path,
            "--output_path", merged_recon_path
        ]
        subprocess.run(cmd, check=True)
        
        # Update current state
        current_recon_path = merged_recon_path
        current_split_images = current_split_images.union(next_split_images)
    
    # Final output
    final_recon_path = f"{output_folder}/final_reconstruction"
    os.makedirs(final_recon_path, exist_ok=True)
    if os.path.exists(final_recon_path):
        shutil.rmtree(final_recon_path)
    if os.path.exists(current_recon_path):
        shutil.copytree(current_recon_path, final_recon_path)

if __name__ == "__main__":
    model_name = "mast3r"  # or "vggt"
    dataset_folder = "data/gerrard-hall/images/"
    output_folder = f"data/gerrard-hall/{model_name}_multi"

    
    multi_stage_reconstruction(dataset_folder, output_folder, model_name)