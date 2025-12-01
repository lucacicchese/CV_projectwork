import pycolmap
import numpy as np
import csv
from pathlib import Path
from horn_loss_downloaded import read_csv


def colmap_to_csv(images_bin_path, output_csv_path, dataset_name='dataset1', scene_name='scene1', allowed_images=None):
    print(f"Reading {images_bin_path}...")
    reconstruction = pycolmap.Reconstruction(images_bin_path.replace('images.bin', ''))
    images = reconstruction.images

    if allowed_images is not None:
        images = {i: img for i, img in images.items() if img.name in allowed_images}

    print(f"Writing {len(images)} images to {output_csv_path}...")

    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['dataset', 'scene', 'image', 'rotation_matrix', 'translation_vector'])

        for img_id, img in images.items():
            R = img.cam_from_world.rotation.matrix()
            t = img.cam_from_world.translation

            R_str = ';'.join([str(x) for x in R.flatten()])
            t_str = ';'.join([str(x) for x in t.flatten()])

            writer.writerow([
                dataset_name,
                scene_name,
                img.name,
                R_str,
                t_str
            ])

    print(f"Conversion completed! {len(images)} images saved.")


def create_thresholds_csv(output_path, dataset_name='dataset1', scene_name='scene1', 
                          thresholds=None):
    if thresholds is None:
        thresholds = [0.001, 0.005, 0.1, 0.2, 0.5, 1]
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['dataset', 'scene', 'thresholds'])
        
        th_str = ';'.join([str(x) for x in thresholds])
        writer.writerow([dataset_name, scene_name, th_str])
    
    print(f"Thresholds file created: {output_path}")


if __name__ == "__main__":
    output_dir = Path("evaluate")
    output_dir.mkdir(exist_ok=True)
    
    gt_images_bin = "All_Reconstructions/colmap/reconstruction/images.bin"
    user_images_bin_mast3r = "All_Reconstructions/mast3r/reconstructions/1/reconstruction/0/images.bin"
    user_images_bin_vggt = "All_Reconstructions/vggt/reconstructions/1/images.bin"

    gt_reconstruction = pycolmap.Reconstruction(gt_images_bin.replace('images.bin', ''))
    mast3r_reconstruction = pycolmap.Reconstruction(user_images_bin_mast3r.replace('images.bin', ''))
    vggt_reconstruction = pycolmap.Reconstruction(user_images_bin_vggt.replace('images.bin', ''))

    gt = gt_reconstruction.images
    mast3r = mast3r_reconstruction.images
    vggt = vggt_reconstruction.images

    common_names_mast3r = set(img.name for img in gt.values()) & \
                          set(img.name for img in mast3r.values())
    
    common_names_vggt = set(img.name for img in gt.values()) & \
                        set(img.name for img in vggt.values())
    
    common_names_mast3rvggt = set(img.name for img in mast3r.values()) & \
                        set(img.name for img in vggt.values())


    colmap_to_csv(
        gt_images_bin,
        output_dir / "gt_poses_mast3r.csv",
        dataset_name="gerrard-hall",
        scene_name="my_scene",
        allowed_images=common_names_mast3r
    )

    colmap_to_csv(
        user_images_bin_mast3r,
        output_dir / "user_poses_mast3r.csv",
        dataset_name="gerrard-hall",
        scene_name="my_scene",
        allowed_images=common_names_mast3r
    )

    colmap_to_csv(
        gt_images_bin,
        output_dir / "gt_poses_vggt.csv",
        dataset_name="gerrard-hall",
        scene_name="my_scene",
        allowed_images=common_names_vggt
    )

    colmap_to_csv(
        user_images_bin_vggt,
        output_dir / "user_poses_vggt.csv",
        dataset_name="gerrard-hall",
        scene_name="my_scene",
        allowed_images=common_names_vggt
    )

    colmap_to_csv(
        user_images_bin_mast3r,
        output_dir / "gt_mast3r_poses_vggt.csv",
        dataset_name="gerrard-hall",
        scene_name="my_scene",
        allowed_images=common_names_mast3rvggt
    )

    colmap_to_csv(
        user_images_bin_vggt,
        output_dir / "user_mast3r_poses_vggt.csv",
        dataset_name="gerrard-hall",
        scene_name="my_scene",
        allowed_images=common_names_mast3rvggt
    )

    create_thresholds_csv(
        output_dir / "thresholds.csv",
        dataset_name="gerrard-hall",
        scene_name="my_scene"
    )
    
    print("Files successfully created!")