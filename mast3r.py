import os
import sys
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from torchvision.io import read_image
from pathlib import Path
import subprocess

# Add mast3r directory to Python path
parent_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
mast3r_path = parent_dir / "mast3r"
if str(mast3r_path) not in sys.path:
    sys.path.append(str(mast3r_path))

def extract_features_mast3r(image_folder, output_folder="data/mast3r_reconstruction", model_name=None):
    """
    Extract 3D points and camera poses using MASt3R-SfM pipeline via demo.py,
    then convert to COLMAP-compatible formats.
    
    Args:
        image_folder (str): Path to folder containing images (.jpg, .png, etc.).
        output_folder (str): Path to save outputs (3D points, poses, etc.).
        model_name (str): Name of the MASt3R model checkpoint.
    
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
    
    # Check for demo.py
    demo_path = mast3r_path / "demo.py"
    if not demo_path.exists():
        raise FileNotFoundError(f"demo.py not found in {mast3r_path}. Ensure MASt3R repository is complete.")
    
    # Check for checkpoint
    checkpoint_dir = mast3r_path / "checkpoints"
    model_path = checkpoint_dir / f"{model_name}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint {model_path} not found. Please download it to {checkpoint_dir}.")
    
    # Run MASt3R-SfM pipeline via demo.py
    cmd = [
        "python3", "demo.py",
        "--model_name", model_name,
        "--image_dir", str(image_folder),
        "--output_dir", str(output_folder),
        "--top_k", "20"
    ]
    try:
        subprocess.run(cmd, check=True, cwd=str(mast3r_path))
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MASt3R-SfM pipeline failed: {e}")
    
    # Load MASt3R outputs (adjust paths based on demo.py output)
    points3d_path = output_folder / "sparse" / "0" / "points3D.npy"
    poses_path = output_folder / "sparse" / "0" / "cam2w.npy"
    intrinsics_path = output_folder / "sparse" / "0" / "intrinsics.json"
    
    if not points3d_path.exists() or not poses_path.exists() or not intrinsics_path.exists():
        raise FileNotFoundError(f"MASt3R outputs not found in {output_folder / 'sparse' / '0'}. Check demo.py execution.")
    
    points3d = np.load(points3d_path)
    poses = np.load(poses_path)
    with open(intrinsics_path, 'r') as f:
        intrinsics = json.load(f)
    
    # Get sorted list of images
    images = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
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
    points3d_path_txt = output_folder / "points3D.txt"
    with open(points3d_path_txt, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for pid, p in enumerate(points3d):
            x, y, z = p
            f.write(f"{pid+1} {x} {y} {z} 128 128 128 0.0\n")
    
    # Collect output paths
    output_files = {
        "pts3d": str(points3d_path),
        "cam2w": str(poses_path),
        "intrinsics": str(intrinsics_path),
        "depthmaps": str(output_folder / "depthmaps"),
        "cameras_txt": str(cameras_path),
        "images_txt": str(images_path),
        "points3D_txt": str(points3d_path_txt)
    }
    
    return output_files


if __name__ == "__main__":
    results = extract_features_mast3r(
        image_folder="data/gerrard-hall/images/",
        output_folder="data/mast3r_reconstruction",
        model_name="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    )
    print("Output files:", results)
        