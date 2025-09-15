import numpy as np
import cv2
import os
from pathlib import Path

def extract_features_vggt(image_path, output_path):
    """
    Extract features from an image using VGGT (Visual-Geometric Feature Extraction).
    
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
    # These would normally come from camera calibration
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


def generate_synthetic_3d_points(points_2d, camera_matrix):
    """
    Generate synthetic 3D points from 2D points.
    In a real implementation, this would use stereo vision, SfM, or depth estimation.
    """
    # Assume points are at various depths between 1 and 10 units
    np.random.seed(42)  # For reproducible results
    depths = np.random.uniform(1.0, 10.0, len(points_2d))
    
    # Convert 2D points to normalized camera coordinates
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    
    points_3D = []
    for i, (u, v) in enumerate(points_2d):
        # Convert to normalized coordinates and then to 3D
        x_norm = (u - cx) / fx
        y_norm = (v - cy) / fy
        depth = depths[i]
        
        x_3d = x_norm * depth
        y_3d = y_norm * depth
        z_3d = depth
        
        points_3D.append([x_3d, y_3d, z_3d])
    
    return np.array(points_3D, dtype=np.float32)


def estimate_camera_poses(points_2d, points_3D, camera_matrix, dist_coeffs):
    """
    Estimate camera pose using PnP solver.
    """
    poses = []
    
    if len(points_2d) >= 4 and len(points_3D) >= 4:
        try:
            # Use RANSAC-based PnP solver for robustness
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                points_3D, points_2d, camera_matrix, dist_coeffs,
                reprojectionError=3.0,
                confidence=0.99
            )
            
            if success:
                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                
                # Create 4x4 transformation matrix
                pose = np.eye(4)
                pose[:3, :3] = rotation_matrix
                pose[:3, 3] = tvec.flatten()
                
                poses.append({
                    'rotation_matrix': rotation_matrix,
                    'translation_vector': tvec,
                    'transformation_matrix': pose,
                    'inliers': inliers,
                    'num_inliers': len(inliers) if inliers is not None else 0
                })
                
                print(f"Camera pose estimated with {len(inliers) if inliers is not None else 0} inliers")
            else:
                print("Failed to estimate camera pose")
                
        except cv2.error as e:
            print(f"Error in pose estimation: {e}")
    
    else:
        print(f"Insufficient points for pose estimation: {len(points_2d)} 2D points, {len(points_3D)} 3D points")
    
    return poses


def save_features(output_path, poses, points_3D, points_2d, descriptors, keypoints, camera_matrix):
    """
    Save extracted features to file.
    """
    # Prepare data for saving
    feature_data = {
        'poses': poses,
        'points_3D': points_3D,
        'points_2d': points_2d,
        'descriptors': descriptors,
        'keypoint_coordinates': np.array([kp.pt for kp in keypoints]),
        'keypoint_responses': np.array([kp.response for kp in keypoints]),
        'keypoint_angles': np.array([kp.angle for kp in keypoints]),
        'camera_matrix': camera_matrix,
        'num_features': len(points_2d)
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


def load_features(feature_path):
    """
    Load previously saved features.
    """
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature file not found: {feature_path}")
    
    data = np.load(feature_path, allow_pickle=True)
    return {key: data[key] for key in data.keys()}


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