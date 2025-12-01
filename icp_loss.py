import numpy as np
import csv
from pathlib import Path


def read_csv(filename):
    """Read camera poses from CSV file."""
    poses = {}
    
    with open(filename, newline='\n') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image = row['image']
            R = np.array([float(x) for x in row['rotation_matrix'].split(';')]).reshape(3, 3)
            t = np.array([float(x) for x in row['translation_vector'].split(';')]).reshape(3)
            c = -R.T @ t  # camera center
            poses[image] = c
    
    return poses


def icp_alignment(source, target, max_iter=500, tol=1e-6):
    """Align source points to target points using ICP."""
    src = source.copy()
    R_total = np.eye(3)
    t_total = np.zeros(3)
    prev_error = float('inf')
    
    for _ in range(max_iter):
        # Find nearest neighbors
        distances = np.sum((src[:, :, None] - target[:, None, :])**2, axis=0)
        indices = np.argmin(distances, axis=1)
        matched = target[:, indices]
        
        # Compute centroids
        src_mean = src.mean(axis=1)
        tgt_mean = matched.mean(axis=1)
        
        # Center the points
        src_centered = src - src_mean[:, None]
        tgt_centered = matched - tgt_mean[:, None]
        
        # Compute rotation using SVD
        H = src_centered @ tgt_centered.T
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Compute translation
        t = tgt_mean - R @ src_mean
        
        # Update cumulative transformation
        R_total = R @ R_total
        t_total = R @ t_total + t
        
        # Apply transformation
        src = R @ src + t[:, None]
        
        # Check convergence
        error = np.mean(np.sum((src - matched)**2, axis=0))
        if abs(prev_error - error) < tol:
            break
        prev_error = error
    
    return R_total, t_total


def compute_icp_metrics(gt_csv, user_csv):
    """Compute ICP alignment error between ground truth and user poses."""
    gt_poses = read_csv(gt_csv)
    user_poses = read_csv(user_csv)
    
    common_images = sorted(set(gt_poses.keys()) & set(user_poses.keys()))
    
    if len(common_images) < 3:
        print(f"Not enough common images: {len(common_images)}")
        return
    
 
    n = len(common_images)
    user_centers = np.zeros((3, n))
    gt_centers = np.zeros((3, n))
    
    for i, img in enumerate(common_images):
        user_centers[:, i] = user_poses[img]
        gt_centers[:, i] = gt_poses[img]
    
    R, t = icp_alignment(user_centers, gt_centers)
    
    aligned = R @ user_centers + t[:, None]
    errors = np.linalg.norm(aligned - gt_centers, axis=0)
    
    print(f"Number of images: {n}")
    print(f"RMS error: {np.sqrt(np.mean(errors**2)):.6f}")
    print(f"Mean error: {np.mean(errors):.6f}")
    print(f"Median error: {np.median(errors):.6f}")
    print(f"Std error: {np.std(errors):.6f}")
    print(f"Min error: {np.min(errors):.6f}")
    print(f"Max error: {np.max(errors):.6f}")


if __name__ == "__main__":
    output_dir = Path("evaluate")
    
    print("Evaluating MASt3R reconstruction...")
    compute_icp_metrics(
        output_dir / "gt_poses_mast3r.csv",
        output_dir / "user_poses_mast3r.csv"
    )
    
    print("\nEvaluating VGGT reconstruction...")
    compute_icp_metrics(
        output_dir / "gt_poses_vggt.csv",
        output_dir / "user_poses_vggt.csv"
    )