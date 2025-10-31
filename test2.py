import numpy as np
import pycolmap
from pathlib import Path
import torch
from PIL import Image
import os
import sys
import glob
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# Add mast3r to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
mast3r_path = parent_dir / "mast3r"
sys.path.insert(0, str(mast3r_path))

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.utils.image import load_images


def cosine_schedule(t, lr_start, lr_end):
    """Cosine learning rate schedule"""
    return lr_end + (lr_start - lr_end) * (1 + np.cos(np.pi * t)) / 2


def extract_features_mast3r_single_batch(image_files, output_folder, model, device='cuda', 
                                         batch_size=1, lr=0.01, niter=300, 
                                         image_size=512, scene_graph='swin'):
    """
    Extract features and perform SfM for a single batch of images.
    
    Args:
        image_files (list): List of image file paths
        output_folder (str): Path to folder where outputs will be saved
        model: Loaded MASt3R model
        device (str): Device to run on
        batch_size (int): Batch size for inference
        lr (float): Learning rate for optimization
        niter (int): Number of iterations for optimization
        image_size (int): Image size for processing
        scene_graph (str): Scene graph type
    
    Returns:
        dict: Dictionary containing scene reconstruction results
    """
    schedule_fn = cosine_schedule
    
    # Load images
    print(f"  Loading {len(image_files)} images...")
    images = load_images(image_files, size=image_size)
    
    # Create pairs for processing
    pairs = make_pairs(images, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    print(f"  Created {len(pairs)} image pairs")
    
    # Run inference
    print("  Running inference...")
    output = inference(pairs, model, device, batch_size=batch_size)
    
    torch.cuda.empty_cache() if device == 'cuda' else None
    
    # Build imgs_dict mapping idx to image file path
    imgs_dict = {}
    for i, img in enumerate(images):
        idx = img['idx']
        imgs_dict[idx] = image_files[i]
    
    imgs_list = [imgs_dict[i] for i in sorted(imgs_dict.keys())]
    
    # Perform sparse global alignment
    scene = sparse_global_alignment(
        imgs_list,
        pairs,
        output_folder,
        model,
        lr1=lr,
        niter1=niter,
        schedule=schedule_fn,
        device=device
    )
    
    # Extract results
    poses = scene.get_im_poses().cpu().numpy()
    pts3d = scene.get_pts3d().cpu().numpy() if hasattr(scene, 'get_pts3d') else None
    focals = scene.get_focals().cpu().numpy()
    
    results = {
        'camera_poses': poses,
        'points3d': pts3d,
        'rotations': poses[:, :3, :3],
        'translations': poses[:, :3, 3],
        'focals': focals,
        'image_paths': image_files
    }
    
    return results


def find_correspondences_icp(points1, points2, max_distance=0.5):
    """
    Find point correspondences between two point clouds using nearest neighbor matching.
    
    Args:
        points1: First point cloud [N1, 3]
        points2: Second point cloud [N2, 3]
        max_distance: Maximum distance for valid correspondences
    
    Returns:
        indices1, indices2: Corresponding point indices
    """
    if points1 is None or points2 is None or len(points1) == 0 or len(points2) == 0:
        return np.array([]), np.array([])
    
    # Sample points if too many
    max_points = 10000
    if len(points1) > max_points:
        idx1 = np.random.choice(len(points1), max_points, replace=False)
        points1 = points1[idx1]
    else:
        idx1 = np.arange(len(points1))
    
    if len(points2) > max_points:
        idx2 = np.random.choice(len(points2), max_points, replace=False)
        points2 = points2[idx2]
    else:
        idx2 = np.arange(len(points2))
    
    # Compute pairwise distances
    distances = cdist(points1, points2)
    
    # Find nearest neighbors
    nn_idx = np.argmin(distances, axis=1)
    nn_distances = distances[np.arange(len(points1)), nn_idx]
    
    # Filter by distance threshold
    valid_mask = nn_distances < max_distance
    indices1 = idx1[valid_mask]
    indices2 = idx2[nn_idx[valid_mask]]
    
    return indices1, indices2


def estimate_similarity_transform(points_src, points_dst):
    """
    Estimate similarity transformation (rotation, translation, scale) between two point sets.
    
    Args:
        points_src: Source points [N, 3]
        points_dst: Destination points [N, 3]
    
    Returns:
        R: Rotation matrix [3, 3]
        t: Translation vector [3]
        s: Scale factor
    """
    if len(points_src) < 3:
        return np.eye(3), np.zeros(3), 1.0
    
    # Center the points
    centroid_src = np.mean(points_src, axis=0)
    centroid_dst = np.mean(points_dst, axis=0)
    
    points_src_centered = points_src - centroid_src
    points_dst_centered = points_dst - centroid_dst
    
    # Compute scale
    scale_src = np.sqrt(np.mean(np.sum(points_src_centered**2, axis=1)))
    scale_dst = np.sqrt(np.mean(np.sum(points_dst_centered**2, axis=1)))
    
    if scale_src < 1e-8:
        scale = 1.0
    else:
        scale = scale_dst / scale_src
    
    # Normalize for rotation estimation
    points_src_normalized = points_src_centered / (scale_src + 1e-8)
    points_dst_normalized = points_dst_centered / (scale_dst + 1e-8)
    
    # Compute rotation using SVD
    H = points_src_normalized.T @ points_dst_normalized
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    t = centroid_dst - scale * R @ centroid_src
    
    return R, t, scale


def align_reconstructions(batch_results, overlap_images=5):
    """
    Align multiple batch reconstructions into a single coordinate system.
    
    Args:
        batch_results: List of reconstruction results from each batch
        overlap_images: Number of overlapping images between consecutive batches
    
    Returns:
        dict: Merged reconstruction results
    """
    if len(batch_results) == 1:
        return batch_results[0]
    
    print("\n" + "="*60)
    print("Aligning and merging batch reconstructions")
    print("="*60)
    
    # Start with the first batch as reference
    merged_poses = [batch_results[0]['camera_poses']]
    merged_points = [batch_results[0]['points3d']] if batch_results[0]['points3d'] is not None else []
    merged_focals = [batch_results[0]['focals']]
    merged_image_paths = [batch_results[0]['image_paths']]
    
    # Align each subsequent batch to the merged reconstruction
    for i in range(1, len(batch_results)):
        print(f"\nAligning batch {i+1}/{len(batch_results)}...")
        
        current_batch = batch_results[i]
        
        # Find overlapping images (last images of previous batch with first images of current batch)
        prev_paths = merged_image_paths[-1][-overlap_images:]
        curr_paths = current_batch['image_paths'][:overlap_images]
        
        # Get poses for overlapping images
        prev_poses = merged_poses[-1][-overlap_images:]
        curr_poses = current_batch['camera_poses'][:overlap_images]
        
        # Extract camera centers
        prev_centers = -np.array([R.T @ t for R, t in zip(prev_poses[:, :3, :3], prev_poses[:, :3, 3])])
        curr_centers = -np.array([R.T @ t for R, t in zip(curr_poses[:, :3, :3], curr_poses[:, :3, 3])])
        
        # Estimate transformation using camera centers
        R, t, s = estimate_similarity_transform(curr_centers, prev_centers)
        
        print(f"  Transformation: scale={s:.4f}, translation norm={np.linalg.norm(t):.4f}")
        
        # Transform current batch poses
        transformed_poses = np.zeros_like(current_batch['camera_poses'])
        for j, pose in enumerate(current_batch['camera_poses']):
            R_cam = pose[:3, :3]
            t_cam = pose[:3, 3]
            
            # Apply similarity transformation
            R_new = R @ R_cam
            t_new = s * R @ t_cam + t
            
            transformed_poses[j, :3, :3] = R_new
            transformed_poses[j, :3, 3] = t_new
            transformed_poses[j, 3, 3] = 1.0
        
        # Transform 3D points
        transformed_points = None
        if current_batch['points3d'] is not None:
            transformed_points = (s * (current_batch['points3d'] @ R.T) + t)
        
        # Add non-overlapping poses (skip first overlap_images)
        merged_poses.append(transformed_poses[overlap_images:])
        if transformed_points is not None:
            merged_points.append(transformed_points)
        merged_focals.append(current_batch['focals'][overlap_images:])
        merged_image_paths.append(current_batch['image_paths'][overlap_images:])
    
    # Concatenate all results
    final_poses = np.vstack(merged_poses)
    final_focals = np.concatenate(merged_focals)
    final_image_paths = [path for batch_paths in merged_image_paths for path in batch_paths]
    
    if len(merged_points) > 0:
        final_points = np.vstack([p for p in merged_points if p is not None])
    else:
        final_points = None
    
    print(f"\nMerge complete:")
    print(f"  Total camera poses: {len(final_poses)}")
    print(f"  Total 3D points: {len(final_points) if final_points is not None else 0}")
    print(f"  Total images: {len(final_image_paths)}")
    
    merged_results = {
        'camera_poses': final_poses,
        'points3d': final_points,
        'rotations': final_poses[:, :3, :3],
        'translations': final_poses[:, :3, 3],
        'focals': final_focals,
        'image_paths': final_image_paths
    }
    
    return merged_results


def extract_features_mast3r_batched(image_folder, output_folder, model_name="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
                                    device='cuda', batch_size=1, lr=0.01, niter=300,
                                    image_size=512, max_images=None, scene_graph='swin',
                                    images_per_batch=40, overlap=5):
    """
    Extract features and perform SfM using MASt3R pipeline with batching.
    
    Args:
        image_folder (str): Path to folder containing input images
        output_folder (str): Path to folder where outputs will be saved
        model_name (str): Model checkpoint name
        device (str): Device to run on ('cuda' or 'cpu')
        batch_size (int): Batch size for inference
        lr (float): Learning rate for optimization
        niter (int): Number of iterations for optimization
        image_size (int): Image size for processing
        max_images (int): Maximum number of images to process (None for all)
        scene_graph (str): Scene graph type ('complete', 'swin', 'oneref')
        images_per_batch (int): Maximum images per reconstruction batch
        overlap (int): Number of overlapping images between consecutive batches
    
    Returns:
        dict: Dictionary containing merged scene reconstruction results
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the model once
    print(f"Loading model: {model_name}")
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    
    # Get all image files
    print(f"Loading images from: {image_folder}")
    image_files = sorted(glob.glob(os.path.join(image_folder, '*')))
    image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG'))]
    
    if max_images is not None:
        image_files = image_files[:max_images]
    
    total_images = len(image_files)
    print(f"Found {total_images} images")
    
    # Calculate number of batches
    if total_images <= images_per_batch:
        print("All images fit in one batch, proceeding with standard reconstruction...")
        return extract_features_mast3r(image_folder, output_folder, model_name, device, 
                                      batch_size, lr, niter, image_size, max_images, scene_graph)
    
    # Create batches with overlap
    batches = []
    start_idx = 0
    batch_num = 0
    
    while start_idx < total_images:
        end_idx = min(start_idx + images_per_batch, total_images)
        batch_files = image_files[start_idx:end_idx]
        batches.append(batch_files)
        
        print(f"Batch {batch_num + 1}: images {start_idx} to {end_idx-1} ({len(batch_files)} images)")
        
        # Move start index, accounting for overlap
        if end_idx < total_images:
            start_idx = end_idx - overlap
        else:
            start_idx = end_idx
        
        batch_num += 1
    
    print(f"\nProcessing {len(batches)} batches with {overlap} image overlap")
    print("="*60)
    
    # Process each batch
    batch_results = []
    for i, batch_files in enumerate(batches):
        print(f"\nProcessing batch {i+1}/{len(batches)}...")
        batch_output_folder = os.path.join(output_folder, f"batch_{i}")
        os.makedirs(batch_output_folder, exist_ok=True)
        
        try:
            results = extract_features_mast3r_single_batch(
                batch_files, batch_output_folder, model, device,
                batch_size, lr, niter, image_size, scene_graph
            )
            batch_results.append(results)
            
            # Save individual batch results
            np.savez(os.path.join(batch_output_folder, "batch_reconstruction.npz"),
                    camera_poses=results['camera_poses'],
                    points3d=results['points3d'],
                    focals=results['focals'])
            
            # Clear cache
            torch.cuda.empty_cache() if device == 'cuda' else None
            
        except Exception as e:
            print(f"Error processing batch {i+1}: {e}")
            continue
    
    if len(batch_results) == 0:
        raise RuntimeError("No batches were successfully processed")
    
    # Align and merge all batches
    merged_results = align_reconstructions(batch_results, overlap_images=overlap)
    
    # Save final merged results
    output_path = os.path.join(output_folder, "scene_reconstruction.npz")
    print(f"\nSaving final merged results to: {output_path}")
    
    np.savez(output_path,
            focals=merged_results['focals'],
            poses=merged_results['camera_poses'],
            pts3d=merged_results['points3d'])
    
    # Save image paths
    with open(os.path.join(output_folder, "image_paths.txt"), 'w') as f:
        for path in merged_results['image_paths']:
            f.write(f"{path}\n")
    
    return merged_results


# Keep original function for backward compatibility
def extract_features_mast3r(image_folder, output_folder, model_name="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
                           device='cuda', batch_size=1, lr=0.01, niter=300,
                           image_size=512, max_images=None, scene_graph='swin'):
    """Original single-batch reconstruction function (kept for compatibility)"""
    os.makedirs(output_folder, exist_ok=True)
    schedule_fn = cosine_schedule
    
    print(f"Loading model: {model_name}")
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    
    print(f"Loading images from: {image_folder}")
    images = load_images(image_folder, size=image_size)
    
    if max_images is not None and len(images) > max_images:
        print(f"Limiting to {max_images} images (out of {len(images)})")
        images = images[:max_images]
    
    image_files = sorted(glob.glob(os.path.join(image_folder, '*')))
    image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG'))]
    
    if max_images is not None:
        image_files = image_files[:max_images]
    
    print(f"Processing {len(images)} images")
    
    pairs = make_pairs(images, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    print(f"Created {len(pairs)} image pairs")
    
    print("Running inference...")
    output = inference(pairs, model, device, batch_size=batch_size)
    torch.cuda.empty_cache() if device == 'cuda' else None
    
    imgs_dict = {}
    for i, img in enumerate(images):
        idx = img['idx']
        if i < len(image_files):
            imgs_dict[idx] = image_files[i]
        else:
            imgs_dict[idx] = f"image_{idx}"
    
    imgs_list = [imgs_dict[i] for i in sorted(imgs_dict.keys())]
    
    scene = sparse_global_alignment(
        imgs_list, pairs, output_folder, model,
        lr1=lr, niter1=niter, schedule=schedule_fn, device=device
    )
    
    output_path = os.path.join(output_folder, "scene_reconstruction.npz")
    print(f"Saving results to: {output_path}")
    
    image_paths = []
    for img in images:
        if 'filepath' in img:
            image_paths.append(img['filepath'])
        elif 'img' in img:
            image_paths.append(img['img'])
        elif 'path' in img:
            image_paths.append(img['path'])
        else:
            for v in img.values():
                if isinstance(v, str):
                    image_paths.append(v)
                    break
    
    results = {
        'focals': scene.get_focals().cpu().numpy(),
        'poses': scene.get_im_poses().cpu().numpy(),
        'pts3d': scene.get_pts3d().cpu().numpy() if hasattr(scene, 'get_pts3d') else None,
        'intrinsics': scene.intrinsics.cpu().numpy() if hasattr(scene, 'intrinsics') else None,
        'num_images': len(images),
        'image_paths': image_paths
    }
    
    np.savez(output_path, **{k: v for k, v in results.items() if v is not None and not isinstance(v, list)})
    
    with open(os.path.join(output_folder, "image_paths.txt"), 'w') as f:
        for path in results['image_paths']:
            f.write(f"{path}\n")
    
    poses = results["poses"]
    rotations = poses[:, :3, :3]
    translations = poses[:, :3, 3]
    
    standard_output = {
        "camera_poses": poses,
        "points3d": results["pts3d"],
        "rotations": rotations,
        "translations": translations,
        "image_paths": results["image_paths"]
    }
    return standard_output


if __name__ == "__main__":
    image_folder = "/home/studente/Documenti/luca_cicchese/CV_projectwork/data/gerrard-hall/images"
    mast3r_output_folder = "data/mast3r_reconstruction_batched"
    
    # Use batched reconstruction for large datasets
    results = extract_features_mast3r_batched(
        image_folder, 
        mast3r_output_folder,
        images_per_batch=40,  # Process 40 images at a time
        overlap=5,            # 5 overlapping images between batches
        max_images=None,      # Process all images
        scene_graph='swin'    # Use 'swin' for lower memory
    )
    
    print("\n" + "="*60)
    print("Batched reconstruction complete!")
    print("="*60)
    print(f"Total camera poses: {len(results['camera_poses'])}")
    print(f"Total 3D points: {len(results['points3d']) if results['points3d'] is not None else 0}")
    print(f"Total images: {len(results['image_paths'])}")