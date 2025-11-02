import metrics
from colmap_reconstruction import colmap_reconstruction
from mast3r_reconstruction import mast3r_reconstruction
from vggt_reconstruction import vggt_reconstruction
from utils import copy_folders_to_combined
import os

if __name__ == "__main__":

    output_dir = "All_Reconstructions"
    dataset_name = "gerrard-hall"
    mast3r_output_dir = "data/mast3r_reconstruction_output/"

    # Extract features using COLMAP, MAST3R, and VGGT
    print("Checking if reconstructions already exist...")
    if not (os.path.exists(output_dir) and os.path.isdir(output_dir)):
        print("Starting reconstructions...")
        colmap_reconstruction(image_folder=f"data/{dataset_name}/images/", database_path="data/colmap.db")
        mast3r_reconstruction(dataset_name, mast3r_output_dir)
        vggt_reconstruction(dataset_name)

        print("All reconstructions completed.")
        copy_folders_to_combined(
            "data/reconstruction/0/",
            f"{mast3r_output_dir}/reconstruction/0/",
            f"data/{dataset_name}_vggt/sparse/",
            "All_Reconstructions/"
        )

    print(f"All reconstructions available in: {output_dir}")


    print("Evaluating reconstructions...")
