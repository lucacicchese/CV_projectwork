import extract
import metrics

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract features from images using COLMAP.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("output_path", type=str, help="Path to save the extracted features.")
    args = parser.parse_args()

    # Extract features using COLMAP, MAST3R, and VGGT
    colmap_poses, points_3d_colmap = extract.extract_colmap_features(args.image_path, args.output_path)
    mast3r_poses, points_3d_mast3r = extract.extract_mast3r_features(args.image_path, args.output_path)
    vggt_poses, points_3d_vggt = extract.extract_vggt_features(args.image_path, args.output_path)

    # Compare poses
    print("Comparing poses")
    horn_loss_mast3r_poses = metrics.horn_loss(colmap_poses, mast3r_poses)
    icp_loss_mast3r_poses = metrics.icp_loss(colmap_poses, mast3r_poses)

    print(f"Horn Loss (COLMAP vs MAST3R): {horn_loss_mast3r_poses}")
    print(f"ICP Loss (COLMAP vs MAST3R): {icp_loss_mast3r_poses}")

    horn_loss_vggt_poses = metrics.horn_loss(colmap_poses, vggt_poses)
    icp_loss_vggt_poses = metrics.icp_loss(colmap_poses, vggt_poses)

    print(f"Horn Loss (COLMAP vs VGGT): {horn_loss_vggt_poses}")
    print(f"ICP Loss (COLMAP vs VGGT): {icp_loss_vggt_poses}")

    # Compare 3D points
    print("Comparing 3D points")
    horn_loss_mast3r_points = metrics.horn_loss(points_3d_colmap, points_3d_mast3r)
    icp_loss_mast3r_points = metrics.icp_loss(points_3d_colmap, points_3d_mast3r)

    print(f"Horn Loss (COLMAP vs MAST3R 3D Points): {horn_loss_mast3r_points}")
    print(f"ICP Loss (COLMAP vs MAST3R 3D Points): {icp_loss_mast3r_points}")

    horn_loss_vggt_points = metrics.horn_loss(points_3d_colmap, points_3d_vggt)
    icp_loss_vggt_points = metrics.icp_loss(points_3d_colmap, points_3d_vggt)

    # Gaussian splatting
    splatted_image_mast3r = extract.gaussian_splatting(points_3d_mast3r, mast3r_poses)
    splatted_image_vggt = extract.gaussian_splatting(points_3d_vggt, vggt_poses)
