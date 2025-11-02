import metrics
from colmap_reconstruction import colmap_reconstruction
from mast3r_reconstruction import mast3r_reconstruction
from vggt_reconstruction import vggt_reconstruction

if __name__ == "__main__":

    # Extract features using COLMAP, MAST3R, and VGGT
    print("Starting reconstructions...")
    colmap_reconstruction()
    mast3r_reconstruction()
    vggt_reconstruction()

    print("All reconstructions completed.")