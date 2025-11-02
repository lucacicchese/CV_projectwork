import metrics
from colmap_reconstruction import colmap_reconstruction
from mast3r_reconstruction import mast3r_reconstruction
from vggt_reconstruction import vggt_reconstruction
from utils import copy_folders_to_combined

if __name__ == "__main__":

    # Extract features using COLMAP, MAST3R, and VGGT
    print("Starting reconstructions...")
    colmap_reconstruction()
    mast3r_reconstruction()
    vggt_reconstruction()

    print("All reconstructions completed.")
    copy_folders_to_combined(
        "data/reconstruction/0/",
        "reconstruction_output/reconstruction/0/",
        "data/gerrard-hall_vggt/sparse/",
        "All_Reconstructions/"
    )
