"""
Converte ricostruzioni COLMAP (formato binario) in CSV per IMC 3D error metric
Richiede: pip install numpy
"""

import struct
import numpy as np
import csv
from pathlib import Path
from downloaded_horn import read_csv


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = struct.unpack(endian_character + format_char_sequence, fid.read(num_bytes))
    return data


def read_cameras_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = 4
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = {
                'model': model_id,
                'width': width,
                'height': height,
                'params': params
            }
    return cameras


def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]

            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]

            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            fid.read(24 * num_points2D)

            R = qvec2rotmat(qvec)

            images[image_id] = {
                'name': image_name,
                'camera_id': camera_id,
                'R': R,
                't': tvec,
                'qvec': qvec
            }
    return images


def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
    ])


def colmap_to_csv(images_bin_path, output_csv_path, dataset_name='dataset1', scene_name='scene1', allowed_images=None):
    print(f"Reading {images_bin_path}...")
    images = read_images_binary(images_bin_path)

    if allowed_images is not None:
        images = {i: d for i, d in images.items() if d['name'] in allowed_images}

    print(f"Writing {len(images)} images to {output_csv_path}...")

    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['dataset', 'scene', 'image', 'rotation_matrix', 'translation_vector'])

        for img_id, img_data in images.items():
            R = img_data['R']
            t = img_data['t']

            R_str = ';'.join([str(x) for x in R.flatten()])
            t_str = ';'.join([str(x) for x in t.flatten()])

            writer.writerow([
                dataset_name,
                scene_name,
                img_data['name'],
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
    gt_images_bin = "All_Reconstructions//colmap/reconstruction/images.bin"
    user_images_bin_mast3r = "All_Reconstructions/mast3r/reconstructions/1/reconstruction/0/images.bin"
    user_images_bin_vggt = "All_Reconstructions/vggt/reconstructions/1/images.bin"

    gt = read_images_binary(gt_images_bin)
    mast3r = read_images_binary(user_images_bin_mast3r)
    vggt = read_images_binary(user_images_bin_vggt)

    common_names_mast3r = set(d['name'] for d in gt.values()) & \
                   set(d['name'] for d in mast3r.values())
    
    common_names_vggt = set(d['name'] for d in gt.values()) & \
                   set(d['name'] for d in vggt.values())

    colmap_to_csv(
        gt_images_bin,
        "gt_poses_mast3r.csv",
        dataset_name="gerrard-hall",
        scene_name="my_scene",
        allowed_images=common_names_mast3r
    )

    colmap_to_csv(
        user_images_bin_mast3r,
        "user_poses_mast3r.csv",
        dataset_name="gerrard-hall",
        scene_name="my_scene",
        allowed_images=common_names_mast3r
    )

    colmap_to_csv(
        gt_images_bin,
        "gt_poses_vggt.csv",
        dataset_name="gerrard-hall",
        scene_name="my_scene",
        allowed_images=common_names_vggt
    )

    colmap_to_csv(
        user_images_bin_vggt,
        "user_poses_vggt.csv",
        dataset_name="gerrard-hall",
        scene_name="my_scene",
        allowed_images=common_names_vggt
    )

    create_thresholds_csv(
        "thresholds.csv",
        dataset_name="gerrard-hall",
        scene_name="my_scene"
    )
    
    gt_data = read_csv("gt_poses_mast3r.csv")
    user_data = read_csv("user_poses_mast3r.csv")

    for img in list(gt_data['gerrard-hall']['my_scene'].keys())[:3]:
        print(f"\n{img}:")
        print(f"GT center: {gt_data['gerrard-hall']['my_scene'][img]['c']}")
        print(f"User center: {user_data['gerrard-hall']['my_scene'][img]['c']}")

