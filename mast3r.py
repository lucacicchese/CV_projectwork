import os
import numpy as np
import torch
from PIL import Image
import json
from pathlib import Path

# MASt3R imports - you may need to install from: https://github.com/naver/mast3r
try:
    from mast3r.model import AsymmetricMASt3R
    from dust3r.utils.image import load_images
    from mast3r.cloud_opt import global_alignment_loop
    from mast3r.viz import show_raw_pointcloud
except ImportError:
    print("MASt3R not found. Please install it from: https://github.com/naver/mast3r")
    raise

def extract_features_mast3r(image_folder, output_folder="data/mast3r_reconstruction", model_name='MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric'):
    """
    Extract camera poses and 3D point cloud using MASt3R
    
    Args:
        image_folder: Path to folder containing images
        output_folder: Path to save reconstruction results
        model_name: MASt3R model to use
    """
    
    input_folder = Path(image_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Cache files
    poses_cache = output_folder / "camera_poses.json"
    pointcloud_cache = output_folder / "point_cloud.json"
    
    # Check if reconstruction already exists
    if poses_cache.exists() and pointcloud_cache.exists():
        print("MASt3R reconstruction already exists, loading from cache...")
        try:
            with open(poses_cache, 'r') as f:
                camera_poses = json.load(f)
            with open(pointcloud_cache, 'r') as f:
                point_cloud = json.load(f)
            print(f"Loaded {len(camera_poses)} camera poses and {len(point_cloud)} 3D points from cache")
            return camera_poses, point_cloud
        except Exception as e:
            print(f"Failed to load cached reconstruction: {e}")
            print("Proceeding with fresh extraction...")
    
    # Load and prepare images
    print("Loading images...")
    image_paths = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
    
    for img_path in input_folder.iterdir():
        if img_path.suffix.lower() in valid_extensions:
            image_paths.append(str(img_path))
    
    if len(image_paths) < 2:
        raise ValueError(f"Need at least 2 images, found {len(image_paths)}")
    
    image_paths.sort()  # Ensure consistent ordering
    print(f"Found {len(image_paths)} images")
    
    # Resize images if needed
    resized_folder = input_folder.parent / f"{input_folder.name}_resized"
    resized_folder.mkdir(exist_ok=True)
    
    max_size = 512  # MASt3R typically works well with 512x512
    resized_paths = []
    
    for img_path in image_paths:
        img_name = Path(img_path).name
        resized_path = resized_folder / img_name
        
        if not resized_path.exists():
            print(f"Resizing {img_name}...")
            img = Image.open(img_path)
            # Resize while maintaining aspect ratio
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            img.save(resized_path, quality=95)
        
        resized_paths.append(str(resized_path))
    
    # Load MASt3R model
    print(f"Loading MASt3R model: {model_name}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    
    # Load images for processing
    images = load_images(resized_paths, size=max_size)
    
    # Run MASt3R pairwise matching
    print("Running MASt3R pairwise matching...")
    pairs = []
    outputs = []
    
    # Create all possible pairs
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            pairs.append((i, j))
    
    print(f"Processing {len(pairs)} image pairs...")
    
    # Process pairs
    for i, (idx1, idx2) in enumerate(pairs):
        if i % 10 == 0:
            print(f"Processing pair {i+1}/{len(pairs)}")
        
        img1 = images[idx1]
        img2 = images[idx2]
        
        # Prepare batch
        batch = {
            'view1': {
                'img': img1['img'].unsqueeze(0).to(device),
                'true_shape': img1['true_shape'].unsqueeze(0).to(device),
                'idx': idx1,
                'instance': str(resized_paths[idx1])
            },
            'view2': {
                'img': img2['img'].unsqueeze(0).to(device),
                'true_shape': img2['true_shape'].unsqueeze(0).to(device),
                'idx': idx2,
                'instance': str(resized_paths[idx2])
            }
        }
        
        # Run inference
        with torch.no_grad():
            output = model(batch)
        
        outputs.append({
            'view1': {
                'idx': idx1,
                'instance': str(resized_paths[idx1]),
                'pred': output['view1']['pred'].cpu(),
                'conf': output['view1']['conf'].cpu()
            },
            'view2': {
                'idx': idx2,
                'instance': str(resized_paths[idx2]),
                'pred': output['view2']['pred'].cpu(),
                'conf': output['view2']['conf'].cpu()
            }
        })
    
    # Global alignment
    print("Running global alignment...")
    try:
        scene = global_alignment_loop(
            outputs,
            lr=0.01,
            niter=300,
            schedule='cosine',
            verbose=True
        )
        
        # Extract camera poses
        camera_poses = {}
        for i, img_path in enumerate(resized_paths):
            img_name = Path(img_path).name
            
            # Get camera parameters
            if hasattr(scene, 'get_focals'):
                focal = scene.get_focals()[i].item()
            else:
                focal = 500.0  # Default focal length
            
            if hasattr(scene, 'get_poses'):
                pose = scene.get_poses()[i].cpu().numpy()
            else:
                # Fallback: identity pose
                pose = np.eye(4)
            
            # Extract rotation and translation
            rotation = pose[:3, :3]
            translation = pose[:3, 3]
            
            camera_poses[img_name] = {
                'image_id': i,
                'rotation_matrix': rotation.tolist(),
                'translation': translation.tolist(),
                'camera_center': (-rotation.T @ translation).tolist(),
                'focal_length': focal,
                'pose_matrix': pose.tolist()
            }
        
        # Extract point cloud
        print("Extracting point cloud...")
        point_cloud = []
        
        if hasattr(scene, 'get_pts3d'):
            pts3d = scene.get_pts3d()
            if hasattr(scene, 'get_masks'):
                masks = scene.get_masks()
            else:
                masks = [torch.ones_like(pts[:, :, 0], dtype=torch.bool) for pts in pts3d]
            
            point_id = 0
            for i, (pts, mask) in enumerate(zip(pts3d, masks)):
                pts = pts.cpu().numpy()
                mask = mask.cpu().numpy()
                
                # Get valid points
                valid_pts = pts[mask]
                
                # Add color if available (use image colors)
                img = Image.open(resized_paths[i])
                img_array = np.array(img)
                
                if len(img_array.shape) == 3:
                    colors = img_array[mask] if mask.shape == img_array.shape[:2] else [128, 128, 128]
                else:
                    colors = [128, 128, 128]  # Default gray
                
                for j, pt in enumerate(valid_pts):
                    if len(pt) >= 3 and not np.any(np.isnan(pt[:3])) and not np.any(np.isinf(pt[:3])):
                        color = colors[j] if isinstance(colors, np.ndarray) and j < len(colors) else [128, 128, 128]
                        if not isinstance(color, (list, np.ndarray)) or len(color) < 3:
                            color = [128, 128, 128]
                        
                        point_cloud.append({
                            'point_id': point_id,
                            'xyz': pt[:3].tolist(),
                            'color': color[:3].tolist() if isinstance(color, np.ndarray) else color,
                            'error': 0.0  # MASt3R doesn't provide reprojection error
                        })
                        point_id += 1
        
        print(f"Extracted {len(camera_poses)} camera poses")
        print(f"Extracted {len(point_cloud)} 3D points")
        
        # Save cache
        with open(poses_cache, 'w') as f:
            json.dump(camera_poses, f, indent=2)
        with open(pointcloud_cache, 'w') as f:
            json.dump(point_cloud, f, indent=2)
        
        return camera_poses, point_cloud
        
    except Exception as e:
        print(f"Global alignment failed: {e}")
        print("Trying alternative approach...")
        
        # Fallback: extract poses and points directly from pairwise outputs
        camera_poses = {}
        point_cloud = []
        
        for i, img_path in enumerate(resized_paths):
            img_name = Path(img_path).name
            camera_poses[img_name] = {
                'image_id': i,
                'rotation_matrix': np.eye(3).tolist(),
                'translation': [0.0, 0.0, 0.0],
                'camera_center': [0.0, 0.0, 0.0],
                'focal_length': 500.0,
                'pose_matrix': np.eye(4).tolist()
            }
        
        # Extract points from pairwise predictions
        point_id = 0
        for output in outputs:
            pred1 = output['view1']['pred']
            conf1 = output['view1']['conf']
            
            # Convert predictions to 3D points (simplified)
            if pred1.shape[-1] >= 3:
                valid_mask = conf1 > 0.5  # Confidence threshold
                pts = pred1[valid_mask]
                
                for pt in pts:
                    if len(pt) >= 3 and not torch.any(torch.isnan(pt[:3])) and not torch.any(torch.isinf(pt[:3])):
                        point_cloud.append({
                            'point_id': point_id,
                            'xyz': pt[:3].tolist(),
                            'color': [128, 128, 128],  # Default color
                            'error': 0.0
                        })
                        point_id += 1
        
        # Save fallback results
        with open(poses_cache, 'w') as f:
            json.dump(camera_poses, f, indent=2)
        with open(pointcloud_cache, 'w') as f:
            json.dump(point_cloud, f, indent=2)
        
        print(f"Fallback extraction: {len(camera_poses)} camera poses, {len(point_cloud)} 3D points")
        return camera_poses, point_cloud

if __name__ == "__main__":
    poses, point_cloud = extract_features_mast3r(
        image_folder="data/gerrard-hall/images/",
        output_folder="data/mast3r_reconstruction"
    )
    
    print(f"Extracted {len(poses)} camera poses")
    print(f"Extracted {len(point_cloud)} 3D points")
    
    # Optional: visualize results
    print("Results saved to data/mast3r_reconstruction/")
    print("Camera poses saved to: camera_poses.json")
    print("Point cloud saved to: point_cloud.json")