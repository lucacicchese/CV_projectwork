import numpy as np
import csv
from pathlib import Path
from scipy.spatial.transform import Rotation


def read_csv(filename, header=True):
    data = {}
    label_idx = {}    
    
    with open(filename, newline='\n') as csvfile:    
        csv_lines = csv.reader(csvfile, delimiter=',')
        for row in csv_lines:
            if header:
                header = False
                for i, name in enumerate(row): label_idx[name] = i
                continue
            dataset = row[label_idx['dataset']]
            scene = row[label_idx['scene']]
            image = row[label_idx['image']]
            R = np.array([float(x) for x in (row[label_idx['rotation_matrix']].split(';'))]).reshape(3,3)
            t = np.array([float(x) for x in (row[label_idx['translation_vector']].split(';'))]).reshape(3)
            c = -R.T @ t

            if not (dataset in data):
                data[dataset] = {}            
            if not (scene in data[dataset]):
                data[dataset][scene] = {}
            data[dataset][scene][image] = {'R': R, 't': t, 'c': c}
    return data


# MODIFIED: ICP implementation instead of Horn's method
def icp_alignment(source_points, target_points, max_iterations=50, tolerance=1e-6):
    """
    Align source points to target points using Iterative Closest Point (ICP).
    Returns transformation matrix (4x4) and final error.
    source_points, target_points: shape (3, N)
    """
    src = source_points.copy()
    tgt = target_points.copy()
    
    prev_error = 0
    R_final = np.eye(3)
    t_final = np.zeros((3, 1))
    
    for i in range(max_iterations):
        distances = np.sum((src[:, :, None] - tgt[:, None, :])**2, axis=0)
        indices = np.argmin(distances, axis=1)
        matched_tgt = tgt[:, indices]
        
        src_mean = src.mean(axis=1, keepdims=True)
        tgt_mean = matched_tgt.mean(axis=1, keepdims=True)
        
        src_centered = src - src_mean
        tgt_centered = matched_tgt - tgt_mean
        
        H = src_centered @ tgt_centered.T
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        t = tgt_mean - R @ src_mean
        
        R_final = R @ R_final
        t_final = R @ t_final + t
        
        src = R @ src + t
        
        error = np.mean(np.sum((src - matched_tgt)**2, axis=0))
        
        if abs(prev_error - error) < tolerance:
            break
        prev_error = error
    
    T = np.eye(4)
    T[:3, :3] = R_final
    T[:3, 3] = t_final.squeeze()
    
    return T, np.sqrt(error)


# MODIFIED: Compute alignment and errors using ICP instead of Horn
def evaluate_with_icp(gt_data, user_data, dataset_name, scene_name):
    """
    Evaluate camera poses using ICP alignment.
    Returns per-image errors after ICP alignment.
    """
    gt_scene = gt_data[dataset_name][scene_name]
    user_scene = user_data[dataset_name][scene_name]
    
    common_images = set(gt_scene.keys()) & set(user_scene.keys())
    
    if len(common_images) < 3:
        print(f"Not enough common images for {dataset_name}/{scene_name}")
        return {}
    
    n = len(common_images)
    gt_centers = np.zeros((3, n))
    user_centers = np.zeros((3, n))
    
    image_list = list(common_images)
    for i, img in enumerate(image_list):
        gt_centers[:, i] = gt_scene[img]['c']
        user_centers[:, i] = user_scene[img]['c']
    
    # MODIFIED: Use ICP instead of Horn for alignment
    transform, mean_error = icp_alignment(user_centers, gt_centers)
    
    errors = {}
    for i, img in enumerate(image_list):
        user_center_hom = np.append(user_centers[:, i], 1)
        transformed_center = (transform @ user_center_hom)[:3]
        error = np.linalg.norm(transformed_center - gt_centers[:, i])
        errors[img] = error
    
    return errors, transform


# MODIFIED: Main evaluation function using ICP
def compute_icp_metrics(gt_csv, user_csv, output_txt):
    """
    Compute evaluation metrics using ICP alignment.
    """
    gt_data = read_csv(gt_csv)
    user_data = read_csv(user_csv)
    
    results = []
    
    for dataset in gt_data.keys():
        for scene in gt_data[dataset].keys():
            if scene not in user_data[dataset]:
                continue
            
            errors, transform = evaluate_with_icp(gt_data, user_data, dataset, scene)
            
            if not errors:
                continue
            
            error_values = list(errors.values())
            mean_error = np.mean(error_values)
            median_error = np.median(error_values)
            max_error = np.max(error_values)
            min_error = np.min(error_values)
            std_error = np.std(error_values)
            
            results.append({
                'dataset': dataset,
                'scene': scene,
                'num_images': len(errors),
                'mean_error': mean_error,
                'median_error': median_error,
                'std_error': std_error,
                'min_error': min_error,
                'max_error': max_error
            })
            
            print(f"{dataset}/{scene}:")
            print(f"  Images: {len(errors)}")
            print(f"  Mean error: {mean_error:.6f}")
            print(f"  Median error: {median_error:.6f}")
            print(f"  Std error: {std_error:.6f}")
            print(f"  Min/Max: {min_error:.6f} / {max_error:.6f}")
            print()
    
    with open(output_txt, 'w') as f:
        f.write("ICP-based Camera Pose Evaluation Results\n")
        f.write("=" * 80 + "\n\n")
        
        for result in results:
            f.write(f"Dataset: {result['dataset']}\n")
            f.write(f"Scene: {result['scene']}\n")
            f.write(f"Number of images: {result['num_images']}\n")
            f.write(f"Mean error: {result['mean_error']:.6f}\n")
            f.write(f"Median error: {result['median_error']:.6f}\n")
            f.write(f"Std deviation: {result['std_error']:.6f}\n")
            f.write(f"Min error: {result['min_error']:.6f}\n")
            f.write(f"Max error: {result['max_error']:.6f}\n")
            f.write("-" * 80 + "\n\n")
        
        if results:
            overall_mean = np.mean([r['mean_error'] for r in results])
            overall_median = np.mean([r['median_error'] for r in results])
            f.write(f"Overall average mean error: {overall_mean:.6f}\n")
            f.write(f"Overall average median error: {overall_median:.6f}\n")
    
    print(f"Results saved to {output_txt}")


if __name__ == "__main__":
    output_dir = Path("evaluate")
    output_dir.mkdir(exist_ok=True)
    
    print("Evaluating MASt3R reconstruction...")
    compute_icp_metrics(
        output_dir / "gt_poses_mast3r.csv",
        output_dir / "user_poses_mast3r.csv",
        output_dir / "icp_results_mast3r.txt"
    )
    
    print("\nEvaluating VGGT reconstruction...")
    compute_icp_metrics(
        output_dir / "gt_poses_vggt.csv",
        output_dir / "user_poses_vggt.csv",
        output_dir / "icp_results_vggt.txt"
    )