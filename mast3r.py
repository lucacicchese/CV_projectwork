import os
import sys
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from torchvision.io import read_image
from pathlib import Path

# Add mast3r and inner dust3r directories to Python path
parent_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
mast3r_path = parent_dir / "mast3r"
dust3r_path = parent_dir / "dust3r" / "dust3r"  # Point to inner dust3r folder
for path in [mast3r_path, dust3r_path]:
    if str(path) not in sys.path:
        sys.path.append(str(path))

# Debugging: Print sys.path to verify
print("sys.path:", sys.path)
print("dust3r path:", str(dust3r_path))

try:
    from inference import inference_pairs
    from model import AsymmetricCroCoHead
    from image_pairs import make_pairs
    from utils.image import load_images
except ImportError as e:
    raise ImportError(f"Failed to import dust3r modules: {e}. Ensure dust3r/dust3r/ contains inference.py, model.py, image_pairs.py, and utils/image.py.")

def extract_features_mast3r(image_folder, output_folder="data/mast3r_reconstruction", model_name=None):
    """
    Extract 3D points and camera poses from a dataset using MASt3R,
    then convert to COLMAP-compatible formats for comparison.
    
    Args:
        image_folder (str): Path to folder containing images (.jpg, .png, etc.).
        output_folder (str): Path to save outputs (3D points, poses, etc.). Defaults to 'data/mast3r_reconstruction'.
        model_name (str): Name of the MASt3R model checkpoint (e.g., 'MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric').
    
    Returns:
        dict: Paths to output files, including COLMAP txt files.
    """
    # Validate inputs
    image_folder = Path(image_folder)
    output_folder = Path(output_folder)
    if not image_folder.is_dir():
        raise ValueError(f"Image folder {image_folder} does not exist.")
    if not model_name:
        model_name = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    
    # Ensure output directory exists
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Paths to checkpoint (in mast3r/checkpoints/)
    checkpoint_dir = mast3r_path / "checkpoints"
    model_path = checkpoint_dir / f"{model_name}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint {model_path} not found. Please download it to {checkpoint_dir}.")
    
    # Get sorted list of images
    images = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    num_images = len(images)
    if num_images < 2:
        raise ValueError("At least two images are required for pairwise processing.")
    
    # Load MASt3R model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AsymmetricCroCoHead.from_pretrained(str(model_path)).to(device).eval()
    
    # Load images and create pairs
    img_paths = [str(image_folder / img) for img in images]
    imgs = load_images(img_paths, size=512)  # Resize to 512 as per MASt3R default
    pairs = make_pairs(imgs, scene_graph="sequential", prefilter=None, symmetrize=False)
    
    # Run inference
    with torch.no_grad():
        output = inference_pairs(pairs, model, device, batch_size=1)
    
    # Extract poses and points
    poses = []  # Camera-to-world matrices
    points3d_all = []
    intrinsics = []
    
    for (img1_idx, img2_idx), pred in zip(pairs, output):
        # Extract predictions
        pts3d_1 = pred['pts3d'].cpu().numpy()  # [H, W, 3]
        pts3d_2 = pred['pts3d_in_other_view'].cpu().numpy()  # [H, W, 3]
        conf = pred['conf'].cpu().numpy()  # [H, W]
        pose1 = pred['view1']['cam2world'].cpu().numpy()  # [4, 4]
        pose2 = pred['view2']['cam2world'].cpu().numpy()  # [4, 4]
        K1 = pred['view1']['K'].cpu().numpy()  # [3, 3]
        K2 = pred['view2']['K'].cpu().numpy()  # [3, 3]
        
        # Store poses and intrinsics (only for first appearance of each image)
        if len(poses) <= img1_idx:
            poses.append(pose1)
            intrinsics.append({'K': K1.tolist()})
        if len(poses) <= img2_idx:
            poses.append(pose2)
            intrinsics.append({'K': K2.tolist()})
        
        # Filter points by confidence and flatten
        valid = conf > 0.5  # Arbitrary threshold; adjust as needed
        pts3d = np.concatenate([pts3d_1[valid], pts3d_2[valid]], axis=0)
        points3d_all.append(pts3d)
    
    # Merge points (simple concatenation; no deduplication)
    points3d = np.concatenate(points3d_all, axis=0) if points3d_all else np.zeros((0, 3))
    poses = np.array(poses)  # [num_images, 4, 4]
    
    # Save native outputs
    np.save(output_folder / "pts3d.npy", points3d)
    np.save(output_folder / "cam2w.npy", poses)
    with open(output_folder / "intrinsics.json", 'w') as f:
        json.dump(intrinsics, f)
    
    # Convert to COLMAP formats
    # Write cameras.txt
    cameras_path = output_folder / "cameras.txt"
    with open(cameras_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i, img_name in enumerate(images):
            img_path = image_folder / img_name
            img = read_image(str(img_path))
            _, height, width = img.shape
            K = intrinsics[i]['K']
            fx = K[0][0]
            fy = K[1][1]
            cx = K[0][2]
            cy = K[1][2]
            f.write(f"{i+1} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")
    
    # Write images.txt
    images_path = output_folder / "images.txt"
    with open(images_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, img_name in enumerate(images):
            pose = poses[i]
            R_cam2w = pose[:3, :3]
            t = pose[:3, 3]
            R_w2c_mat = R_cam2w.T
            rot = R.from_matrix(R_w2c_mat)
            quat_xyzw = rot.as_quat()  # [x, y, z, w]
            qw, qx, qy, qz = quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]
            tx, ty, tz = t
            f.write(f"{i+1} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {i+1} {img_name}\n")
            f.write("\n")  # No POINTS2D
    
    # Write points3D.txt
    points3d_path = output_folder / "points3D.txt"
    with open(points3d_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for pid, p in enumerate(points3d):
            x, y, z = p
            f.write(f"{pid+1} {x} {y} {z} 128 128 128 0.0\n")
    
    # Collect output paths
    output_files = {
        "pts3d": str(output_folder / "pts3d.npy"),
        "cam2w": str(output_folder / "cam2w.npy"),
        "intrinsics": str(output_folder / "intrinsics.json"),
        "depthmaps": str(output_folder / "depthmaps"),
        "cameras_txt": str(cameras_path),
        "images_txt": str(images_path),
        "points3D_txt": str(points3d_path)
    }
    
    return output_files


if __name__ == "__main__":
    results = extract_features_mast3r(
        image_folder="data/gerrard-hall/images/",
        output_folder="data/mast3r_reconstruction",
        model_name="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    )
    print("Output files:", results)
        