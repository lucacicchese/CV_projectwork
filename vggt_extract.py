import sys
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import glob

# Add vggt to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
vggt_path = parent_dir / "vggt"

if vggt_path.exists():
    sys.path.insert(0, str(vggt_path))
else:
    print(f"ERROR: vggt path not found at {vggt_path}")
    print("Make sure the vggt repository is cloned in the correct location")
    sys.exit(1)

# Import VGGT
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


def extract_features_vggt(image_folder, output_folder, model_name="facebook/VGGT-1B", 
                          device='cuda', batch_size=1, image_size=512, max_images=None,
                          use_point_map=False, dtype=None):
    """
    Extract features and perform 3D reconstruction using VGGT pipeline.
    
    Args:
        image_folder (str): Path to folder containing input images
        output_folder (str): Path to folder where outputs will be saved
        model_name (str): Model checkpoint name (default: "facebook/VGGT-1B")
        device (str): Device to run on ('cuda' or 'cpu')
        batch_size (int): Batch size (VGGT processes all images at once per scene)
        image_size (int): Image size for processing (default: 512)
        max_images (int): Maximum number of images to process (None for all)
        use_point_map (bool): If True, use point map branch; otherwise use depth-based reconstruction
        dtype (torch.dtype): Data type for computation (None for auto-detection)
    
    Returns:
        dict: Dictionary containing scene reconstruction results
    """
    
    os.makedirs(output_folder, exist_ok=True)
    
    if dtype is None:
        if device == 'cuda' and torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            dtype = torch.float32
    
    # Load the model
    print(f"Loading model: {model_name}")
    model = VGGT.from_pretrained(model_name).to(device)
    model.eval()
    
    # Get image files
    print(f"Loading images from: {image_folder}")
    image_files = sorted(glob.glob(os.path.join(image_folder, '*')))
    image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG', '.bmp', '.tiff'))]

    # Limit number of images if specified
    if max_images is not None and len(image_files) > max_images:
        print(f"Limiting to {max_images} images (out of {len(image_files)})")
        image_files = image_files[:max_images]
    
    print(f"Processing {len(image_files)} images")
    
    images = load_and_preprocess_images(image_files).to(device)
    
    # Run VGGT inference
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype) if device == 'cuda' else torch.amp.autocast('cpu', dtype=dtype):
            # Add batch dimension (VGGT expects [B, N, C, H, W])
            images_batch = images[None]

            aggregated_tokens_list, ps_idx = model.aggregator(images_batch)
            
            # Predict Cameras
            print("  - Predicting cameras...")
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_batch.shape[-2:])
            
            # Predict Depth Maps
            print("  - Predicting depth maps...")
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images_batch, ps_idx)
            
            # Predict Point Maps
            print("  - Predicting point maps...")
            point_map, point_conf = model.point_head(aggregated_tokens_list, images_batch, ps_idx)
            
            # Construct 3D Points from Depth Maps (usually more accurate than point map branch)
            if not use_point_map:
                print("  - Constructing 3D points from depth maps and cameras...")
                point_map_reconstructed = unproject_depth_map_to_point_map(
                    depth_map.squeeze(0),
                    extrinsic.squeeze(0),
                    intrinsic.squeeze(0)
                )
            else:
                point_map_reconstructed = point_map.squeeze(0)

    # Clean up to free memory
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    # Convert to numpy and remove batch dimension
    extrinsic_np = extrinsic.squeeze(0).cpu().numpy()  # [N, 3, 4] or [N, 4, 4]
    intrinsic_np = intrinsic.squeeze(0).cpu().numpy()  # [N, 3, 3]
    depth_map_np = depth_map.squeeze(0).cpu().numpy()  # [N, H, W, 1]
    

    if isinstance(point_map_reconstructed, torch.Tensor):
        point_map_np = point_map_reconstructed.cpu().numpy()  # [N, H, W, 3]
    else:
        point_map_np = point_map_reconstructed  
    
    depth_conf_np = depth_conf.squeeze(0).cpu().numpy() if depth_conf is not None else None
    point_conf_np = point_conf.squeeze(0).cpu().numpy() if point_conf is not None else None
    

    if depth_map_np.shape[-1] == 1:
        depth_map_np = depth_map_np.squeeze(-1)  # [N, H, W]
    
    # Convert extrinsic [R|t] (3x4) to full homogeneous matrix (4x4) if needed
    if extrinsic_np.shape[-2:] == (3, 4):
        N = extrinsic_np.shape[0]
        extrinsic_4x4 = np.zeros((N, 4, 4), dtype=extrinsic_np.dtype)
        extrinsic_4x4[:, :3, :] = extrinsic_np  # Copy [R|t]
        extrinsic_4x4[:, 3, 3] = 1.0  # Set bottom-right to 1
        extrinsic_full = extrinsic_4x4
    else:
        extrinsic_full = extrinsic_np
    
    # Extract camera poses (world from camera, i.e., inverse of extrinsic)
    poses = np.linalg.inv(extrinsic_full)  # [N, 4, 4]
    
    # Extract focal lengths from intrinsic matrices
    focals = intrinsic_np[:, [0, 1], [0, 1]]  # [N, 2] (fx, fy)
    
    # Save results
    print(f"Saving results to: {output_folder}")
    
    # Save camera parameters
    np.savez(
        os.path.join(output_folder, "cameras.npz"),
        extrinsic=extrinsic_np,
        intrinsic=intrinsic_np,
        poses=poses,
        focals=focals
    )
    
    # Save depth maps
    np.savez(
        os.path.join(output_folder, "depth_maps.npz"),
        depth_maps=depth_map_np,
        depth_confidence=depth_conf_np
    )
    
    # Save point maps
    np.savez(
        os.path.join(output_folder, "point_maps.npz"),
        point_maps=point_map_np,
        point_confidence=point_conf_np
    )
    
    # Save image paths
    with open(os.path.join(output_folder, "image_paths.txt"), 'w') as f:
        for path in image_files:
            f.write(f"{path}\n")
    
    # Create a comprehensive results file
    results = {
        'extrinsic': extrinsic_np,  # Original [N, 3, 4] format
        'extrinsic_4x4': extrinsic_full,  # Full [N, 4, 4] format
        'intrinsic': intrinsic_np,
        'poses': poses,
        'focals': focals,
        'depth_maps': depth_map_np,
        'point_maps': point_map_np,
        'depth_confidence': depth_conf_np,
        'point_confidence': point_conf_np,
        'num_images': len(image_files),
        'image_paths': image_files,
        'image_size': images_batch.shape[-2:]
    }

    rotations = results["extrinsic_4x4"][:, :3, :3]
    translations = results["extrinsic_4x4"][:, :3, 3]

    standard_output = {
        "camera_poses": results["poses"],                     # [N, 4, 4]
        "points3d": results["point_maps"].reshape(-1, 3),     # [M, 3]
        "rotations": rotations,                               # [N, 3, 3]
        "translations": translations,                         # [N, 3]
        "image_paths": results["image_paths"]
    }
    return standard_output



if __name__ == "__main__":
    # Example usage
    image_folder = "/home/studente/Documenti/luca_cicchese/CV_projectwork/data/gerrard-hall/images"
    output_folder = "data/vggt_reconstruction"
    
    # Extract features using VGGT
    results = extract_features_vggt(
        image_folder=image_folder,
        output_folder=output_folder,
        model_name="facebook/VGGT-1B",  
        device='cuda' if torch.cuda.is_available() else 'cpu',
        max_images=50,  
        use_point_map=False  
    )
    
