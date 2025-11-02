import os
import subprocess
import pycolmap as colmap
from PIL import Image
import numpy as np


def extract_features(image_folder, database_path):
    input_folder = image_folder
    resized_folder = f'{image_folder.rstrip("/")}_resized'
    output_path = "data/reconstruction"
    

    os.makedirs(resized_folder, exist_ok=True)
    

    need_resize = True
    if os.path.exists(resized_folder):
        original_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
        resized_files = [f for f in os.listdir(resized_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
        if len(original_files) == len(resized_files):
            need_resize = False
            print("Images already resized, skipping resize step...")
    
    if need_resize:
        print("Resizing images...")
        max_size = 1024
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                img_path = os.path.join(input_folder, filename)
                img = Image.open(img_path)
                img.thumbnail((max_size, max_size))
                img.save(os.path.join(resized_folder, filename))
    
    # Check if database already has features
    run_feature_extraction = True
    if os.path.exists(database_path):
        try:
            db = colmap.Database(database_path)
            if len(db.read_all_images()) > 0 and len(db.read_all_keypoints()) > 0:
                print("Features already extracted, skipping feature extraction...")
                run_feature_extraction = False
            db.close()
        except:
            pass
    
    if run_feature_extraction:
        print("Extracting features using COLMAP...")
        subprocess.run([
            "colmap", "feature_extractor",
            "--database_path", database_path,
            "--image_path", resized_folder
        ], check=True)
        
        print("Matching features using COLMAP...")
        subprocess.run([
            "colmap", "exhaustive_matcher",
            "--database_path", database_path
        ], check=True)
    
    print("Running incremental mapping using pycolmap...")
    os.makedirs(output_path, exist_ok=True)
    
    reconstructions = colmap.incremental_mapping(
        database_path=database_path,
        image_path=resized_folder,
        output_path=output_path
    )

    #reconstructions[0].write(output_path)
    colmap.Reconstruction.write_text(reconstructions[0], output_path)
    
    
    reconstruction = reconstructions[0]

    
    camera_poses = {}
    for image_id, image in reconstruction.images.items():
        
        cam_from_world = image.cam_from_world()
        rotation = cam_from_world.rotation.matrix()  
        translation = cam_from_world.translation
        
        camera_poses[image.name] = {
            'image_id': image_id,
            'rotation_matrix': rotation.tolist(),
            'translation': translation.tolist(),
            'camera_center': (-rotation.T @ translation).tolist()
        }
    
    point_cloud = []
    for point_id, point in reconstruction.points3D.items():
        point_cloud.append({
            'point_id': point_id,
            'xyz': point.xyz.tolist(),
            'color': point.color.tolist(),  
            'error': point.error
        })
    
    camera_pose_list = []
    for pose in camera_poses.values():
        R = np.array(pose['rotation_matrix'])
        t = np.array(pose['translation']).reshape(3, 1)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3:] = t
        camera_pose_list.append(T)

    points3d_array = np.array([p['xyz'] for p in point_cloud])

    rotations = []
    translations = []
    for pose in camera_poses.values():
        R = np.array(pose['rotation_matrix'])
        t = np.array(pose['translation'])
        rotations.append(R)
        translations.append(t)

    camera_poses_array = np.stack([
        np.block([
            [R, t.reshape(3, 1)],
            [np.zeros((1, 3)), np.ones((1, 1))]
        ]) for R, t in zip(rotations, translations)
    ])

    standard_output = {
        "camera_poses": camera_poses_array,
        "points3d": points3d_array,
        "rotations": np.stack(rotations),
        "translations": np.stack(translations),
        "image_paths": list(camera_poses.keys())
    }
    return standard_output

if __name__ == "__main__":
    results = extract_features(
        image_folder="data/gerrard-hall/images/",
        database_path="data/colmap.db"
    )
    
    print(f"Extracted {len(results['camera_poses'])} camera poses")
    print(f"Extracted {len(results['points3d'])} 3D points")