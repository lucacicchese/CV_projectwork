import os
import subprocess
import pycolmap as colmap
from PIL import Image
import numpy as np


def extract_features(image_folder, database_path):
    input_folder = image_folder
    resized_folder = f'{image_folder.rstrip("/")}_resized'
    output_path = "data/reconstruction"
    
    # Check if reconstruction already exists
    if os.path.exists(output_path) and os.listdir(output_path):
        print("Reconstruction already exists, loading from cache...")
        try:
            reconstruction = colmap.Reconstruction(output_path)
            # Skip to extracting poses and point cloud
            camera_poses = {}
            for image_id, image in reconstruction.images.items():
                # Fix: cam_from_world is a method, call it to get the transform
                cam_from_world = image.cam_from_world()
                rotation = cam_from_world.rotation_matrix()
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
                    'color': point.rgb.tolist(),
                    'error': point.error
                })
            
            return camera_poses, point_cloud
        except Exception as e:
            print(f"Failed to load cached reconstruction: {e}")
            print("Proceeding with fresh extraction...")
    
    # Proceed with fresh extraction
    os.makedirs(resized_folder, exist_ok=True)
    
    # Check if images are already resized
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
    
    if not reconstructions:
        raise RuntimeError("Incremental mapping failed - no reconstruction created")
    
    reconstruction = reconstructions[0]
    
    camera_poses = {}
    for image_id, image in reconstruction.images.items():
        # Fix: cam_from_world returns a Rigid3d object
        cam_from_world = image.cam_from_world()
        rotation = cam_from_world.rotation.matrix()  # Use .rotation.matrix()
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
            'color': point.color.tolist(),  # Use .color instead of .rgb
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

    standard_output = {
        "camera_poses": np.stack(camera_pose_list),
        "points3d": points3d_array,
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