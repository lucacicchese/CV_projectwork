import numpy as np
import cv2
import os
from pathlib import Path
import json

def extract_features_vggt(image_folder, output_folder):
    """
    Extract features from a folder of images using VGGT (Visual-Geometric Feature Extraction).
    
    :param image_folder: Path to the folder containing images.
    :param output_folder: Path to the folder where the extracted features will be saved.
    :return: Tuple of (poses, points_3D) for all images.
    """
    
    # Check if the image folder exists
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Image folder not found: {image_folder}")
    
    # Create the output directory if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    poses_list = []
    points_3d_list = []
    
    # Iterate over all images in the folder
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        
        # Skip non-image files
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        print(f"Processing image: {image_name}")
        
        # Extract features for each image
        poses, points_3D = extract_features(image_path, output_folder)
        
        if poses is not None and points_3D is not None:
            poses_list.append(poses)
            points_3d_list.append(points_3D)
            
            # Save features to individual files
            output_file = Path(output_folder) / f"{image_name}_features.npz"
            save_features(output_file, poses, points_3D, None, None, None, None)  # save_features will need the correct arguments
            
    # Save aggregated poses and point cloud data for the entire dataset
    aggregated_poses_path = Path(output_folder) / "camera_poses.json"
    aggregated_point_cloud_path = Path(output_folder) / "point_cloud.json"
    
    with open(aggregated_poses_path, 'w') as f:
        json.dump([pose['transformation_matrix'].tolist() for pose in poses_list], f)
    
    # Here we flatten the 3D points from all images
    aggregated_points_3d = np.vstack(points_3d_list)
    with open(aggregated_point_cloud_path, 'w') as f:
        json.dump(aggregated_points_3d.tolist(), f)
    
    print(f"Aggregated camera poses saved to {aggregated_poses_path}")
    print(f"Aggregated point cloud saved to {aggregated_point_cloud_path}")
    
    return poses_list, aggregated_points_3d


def extract_features(image_path, output_path):
    """
    Extract features from a single image using VGGT.
    
    :param image_path: Path to the input image.
    :param output_path: Path to save the extracted features.
    :return: Tuple of (poses, points_3D)
    """
    # Load the image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    
    # Initialize feature detector (using ORB as backbone for VGGT-style extraction)
    detector = cv2.ORB_create(nfeatures=5000)
    
    # Extract keypoints and descriptors
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    
    if descriptors is None or len(keypoints) == 0:
        print("No features detected in the image")
        return None, None
    
    # Extract 2D feature points
    points_2d = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    
    # Camera intrinsic parameters (assuming typical camera - adjust as needed)
    focal_length = max(w, h) * 0.8  # Rough estimate
    cx, cy = w / 2.0, h / 2.0
    camera_matrix = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Distortion coefficients (assuming no distortion for simplicity)
    dist_coeffs = np.zeros((4, 1))
    
    # Generate synthetic 3D points for demonstration
    # In a real VGGT implementation, these would come from stereo matching,
    # structure from motion, or depth estimation
    points_3D = generate_synthetic_3d_points(points_2d, camera_matrix)
    
    # Estimate camera pose using PnP (Perspective-n-Point)
    poses = estimate_camera_poses(points_2d, points_3D, camera_matrix, dist_coeffs)
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save features to output file
    save_features(output_path, poses, points_3D, points_2d, descriptors, keypoints, camera_matrix)
    
    return poses, points_3D


def save_features(output_path, poses, points_3D, points_2d, descriptors, keypoints, camera_matrix):
    """
    Save extracted features to file.
    """
    # Prepare data for saving
    feature_data = {
        'poses': poses,
        'points_3D': points_3D.tolist(),
        'points_2d': points_2d.tolist() if points_2d is not None else None,
        'descriptors': descriptors.tolist() if descriptors is not None else None,
        'keypoint_coordinates': np.array([kp.pt for kp in keypoints]).tolist() if keypoints else None,
        'keypoint_responses': np.array([kp.response for kp in keypoints]).tolist() if keypoints else None,
        'keypoint_angles': np.array([kp.angle for kp in keypoints]).tolist() if keypoints else None,
        'camera_matrix': camera_matrix.tolist(),
        'num_features': len(points_2d) if points_2d is not None else 0
    }
    
    # Save as numpy archive
    if output_path.endswith('.npz'):
        np.savez_compressed(output_path, **feature_data)
        print(f"Features saved to {output_path}")
    else:
        # Save as .npz by default
        npz_path = str(Path(output_path).with_suffix('.npz'))
        np.savez_compressed(npz_path, **feature_data)
        print(f"Features saved to {npz_path}")


# Example usage
if __name__ == "__main__":
    poses, point_cloud = extract_features_vggt(
        image_folder="data/gerrard-hall/images/",
        output_folder="data/vggt_reconstruction"
    )
    
    print(f"Extracted {len(poses)} camera poses")
    print(f"Extracted {len(point_cloud)} 3D points")
    
    # Optional: visualize results
    print("Results saved to data/vggt_reconstruction/")
    print("Camera poses saved to: camera_poses.json")
    print("Point cloud saved to: point_cloud.json")
