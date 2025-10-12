import metrics
import colmap
import mast3r_extract
import vggt_extract
import torch

if __name__ == "__main__":



    # Extract features using COLMAP, MAST3R, and VGGT
    colmap_results = colmap.extract_features(image_folder="data/gerrard-hall/images/",database_path="data/colmap.db")
    mast3r_results = mast3r_extract.extract_features_mast3r(
        image_folder="data/gerrard-hall/images/",
        output_folder="data/mast3r_reconstruction",
        model_name="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
        scene_graph='swin',  # Use 'swin' instead of 'complete' for lower memory
        image_size=512,
        max_images=50  # Limit number of images, increase if you have more RAM
    )
    vggt_results = vggt_extract.extract_features_vggt(
        image_folder="data/gerrard-hall/images/",
        output_folder="data/vggt_reconstruction",
        model_name="facebook/VGGT-1B",  # or "facebook/VGGT-1B-Commercial" for commercial use
        device='cuda' if torch.cuda.is_available() else 'cpu',
        max_images=50,  # Limit number of images if needed
        use_point_map=False  # Use depth-based reconstruction (more accurate)
    )

    # Compare poses
    print("Comparing poses")
    horn_loss_mast3r, horn_R_mast3r, horn_t_mast3r = metrics.horn_loss(colmap_results, mast3r_results, use_camera_centers=True)
    icp_loss_mast3r, icp_R_mast3r, icp_t_mast3r = metrics.icp_loss(colmap_results, mast3r_results)

    print(f"Horn Loss (COLMAP vs MAST3R): {horn_loss_mast3r}")
    print(f"ICP Loss (COLMAP vs MAST3R): {icp_loss_mast3r}")

    horn_loss_vggt, horn_R_vggt, horn_t_vggt = metrics.horn_loss(colmap_results, vggt_results, use_camera_centers=True)
    icp_loss_vggt, icp_R_vggt, icp_t_vggt = metrics.icp_loss(colmap_results, vggt_results)

    print(f"Horn Loss (COLMAP vs VGGT): {horn_loss_vggt}")
    print(f"ICP Loss (COLMAP vs VGGT): {icp_loss_vggt}")

   

    # Gaussian splatting
    #splatted_image_mast3r = gaussian_splatting(points_3d_mast3r, mast3r_poses)
    #splatted_image_vggt = gaussian_splatting(points_3d_vggt, vggt_poses)
