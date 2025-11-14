import metrics
from colmap_reconstruction import colmap_reconstruction
from mast3r_reconstruction import mast3r_reconstruction
from vggt_reconstruction import vggt_reconstruction
from utils import copy_folders_to_combined
import os
from multi_reconstruction import multi_stage_reconstruction
import shutil
from horn_loss import compare_colmap_reconstructions_horn
from icp_loss import compare_colmap_reconstructions_icp

if __name__ == "__main__":

    output_dir = "All_Reconstructions"
    dataset_name = "gerrard-hall"

    dataset_folder = f"data/{dataset_name}/images/"
   

    # Extract features using COLMAP, MAST3R, and VGGT
    print("Checking if reconstructions already exist...")
    if not (os.path.exists(output_dir) and os.path.isdir(output_dir)):
        os.makedirs(output_dir, exist_ok=True)
        print("Starting reconstructions...")

        colmap_reconstruction(image_folder=dataset_folder, database_path="data/reconstruction/colmap.db")
        shutil.copytree("data/reconstruction/0/", f"{output_dir} /colmap", dirs_exist_ok=True)
        multi_stage_reconstruction(dataset_folder, f"{output_dir}/mast3r/", "mast3r")
        multi_stage_reconstruction(dataset_folder, f"{output_dir}/vggt/", "vggt")

        print("All reconstructions completed.")

    print(f"All reconstructions available in: {output_dir}")

    print("Evaluating reconstructions...")
    colmap_reconstruction_path = f"{output_dir}/colmap/reconstruction/"
    mast3r_reconstruction_path = f"{output_dir}/mast3r/reconstructions/1/reconstruction/0/"
    vggt_reconstruction_path = f"{output_dir}/vggt/reconstructions/1/"


    horn_loss_mast3r, horn_R_mast3r, horn_t_mast3r, horn_scale_mast3r = compare_colmap_reconstructions_horn(colmap_reconstruction_path, mast3r_reconstruction_path)
    icp_loss_mast3r, icp_R_mast3r, icp_t_mast3r, icp_scale_mast3r = compare_colmap_reconstructions_icp(colmap_reconstruction_path, mast3r_reconstruction_path)

    horn_loss_vggt, horn_R_vggt, horn_t_vggt, horn_scale_vggt = compare_colmap_reconstructions_horn(colmap_reconstruction_path, vggt_reconstruction_path)
    icp_loss_vggt, icp_R_vggt, icp_t_vggt, icp_scale_vggt = compare_colmap_reconstructions_icp(colmap_reconstruction_path, vggt_reconstruction_path)
    print("Evaluation completed.")