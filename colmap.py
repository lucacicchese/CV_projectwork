import os
import pycolmap as colmap
from PIL import Image

def extract_features(image_folder, database_path):

    input_folder = image_folder
    resized_folder = f'{image_folder}_resized'
    os.makedirs(resized_folder, exist_ok=True)

    max_size = 1024

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            img.thumbnail((max_size, max_size))
            img.save(os.path.join(resized_folder, filename))

     # 1. Extract features
    print("Extracting features...")
    features = colmap.extract_features(
        database_path=database_path,
        image_path=image_folder
        )
    
    # 2. Match features
    print("Matching features...")
    matches =colmap.match_features(
        database_path=database_path
        )
    
    # 3. Incremental mapping
    print("Running incremental mapping...")
    reconstruction = colmap.incremental_mapping(
        database_path=database_path,
        image_folder=image_folder
        )
    
    camera_poses = {}

    for image_id, image in reconstruction.images.items():
        camera_poses[image.name] = {
            'image_id': image_id,
            'rotation_matrix': image.rotation_matrix().tolist(),   
            'translation_vector': image.tvec.tolist(),
            'camera_center': image.camera_center().tolist()
        }

    point_cloud = []

    for point_id, point in reconstruction.points3D.items():
        point_data = {
            'point_id': point_id,
            'xyz': point.xyz.tolist(),              # 3D coordinates
            'color': point.rgb.tolist(),            # RGB color
            'error': point.error,                   # Reprojection error
            'track_length': len(point.track.elements)  # Number of observations
        }
        point_cloud.append(point_data)

    return camera_poses, point_cloud


# Example usage
if __name__ == "__main__":

    poses, point_cloud = extract_features(
        image_folder="data/gerrard-hall/images/",
        database_path="data/colmap.db"
        )
    
    print("Camera Poses:", poses)
    print("Point Cloud:", point_cloud)
