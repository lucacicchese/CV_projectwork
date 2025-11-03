import os
import shutil
import random
from pathlib import Path
import sys

def extract_common_images(path1: str, path2: str):
    p1 = Path(path1) / "images.txt"
    p2 = Path(path2) / "images.txt"
    s1 = {l.split()[9] for l in p1.read_text().splitlines()[4:] if l and not l.startswith("#")}
    s2 = {l.split()[9] for l in p2.read_text().splitlines()[4:] if l and not l.startswith("#")}
    open("common_images.txt","w").write("\n".join(sorted(s1 & s2)) + "\n")

def split_large_image_folder(src_folder, model_name, max_images: int = 40):
    src = Path(src_folder)
    img_dir = src / "images"
    
    if not img_dir.is_dir():
        raise ValueError(f"'images' subfolder not found in {src}")
    
    image_files = [f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}]
    
    if len(image_files) <= max_images:
        return None  # No split needed
    
    # Choose random subset
    selected = random.sample(image_files, max_images)
    
    # Create new folder at same level
    new_folder = src.parent / f"{src.name}_{model_name}"
    new_img_dir = new_folder / "images"
    new_img_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy selected images
    for img in selected:
        shutil.copy2(img, new_img_dir / img.name)
    
    print(f"Created subset with {max_images} images: {new_folder}")
    return new_folder


import os
import shutil
from pathlib import Path
from typing import List

def copy_folders_to_combined(path1: str, path2: str, path3: str, destination: str):
    """
    Copies contents of 3 folders into a new folder with 3 subfolders.
    
    Args:
        path1, path2, path3: Paths to the 3 source folders
        destination: Path to the destination folder (will be created)
    """
    # Convert to Path objects for easier handling
    folder1 = Path(path1)
    folder2 = Path(path2)
    folder3 = Path(path3)
    dest = Path(destination)
    
    # Validate source folders exist
    if not folder1.exists():
        raise FileNotFoundError(f"Folder 1 does not exist: {folder1}")
    if not folder2.exists():
        raise FileNotFoundError(f"Folder 2 does not exist: {folder2}")
    if not folder3.exists():
        raise FileNotFoundError(f"Folder 3 does not exist: {folder3}")
    
    # Create destination folder
    dest.mkdir(parents=True, exist_ok=True)
    
    # Copy each folder's contents
    shutil.copytree(folder1, dest / "colmap", dirs_exist_ok=True)
    shutil.copytree(folder2, dest / "mast3r", dirs_exist_ok=True)
    shutil.copytree(folder3, dest / "vggt", dirs_exist_ok=True)

    print(f"Successfully copied folders to: {dest}")


import os
import shutil
import random
from pathlib import Path
import math

def split_dataset(input_folder, out_folder, dataset_name=None, split_size=40, min_overlap=10,  seed=None):
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Convert to Path
    input_path = Path(input_folder)
    
    # Get all image files (common extensions)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
    image_files = [
        f for f in input_path.iterdir()
        if f.suffix.lower() in image_extensions and f.is_file()
    ]
    
    if len(image_files) == 0:
        print("No image files found in the input folder.")
        return
    
    total_images = len(image_files)
    print(f"Found {total_images} images.")
    
    # Use input folder name if dataset_name not provided
    if dataset_name is None:
        dataset_name = input_path.name
    
    # Create output directory
    output_root = Path(out_folder)
    output_root.mkdir(exist_ok=True)
    
    # Shuffle the full list of images
    shuffled_images = image_files.copy()
    random.shuffle(shuffled_images)
    
    # Calculate step size: split_size - overlap
    step = split_size - min_overlap
    
    if step <= 0:
        raise ValueError("split_size must be greater than min_overlap")
    
    splits = []
    start_idx = 0
    
    while start_idx < total_images:
        end_idx = start_idx + split_size
        current_split = shuffled_images[start_idx:end_idx]
        
        # If we don't have enough for a full split, include remaining
        if len(current_split) < split_size and start_idx > 0:
            # Fill with overlap from previous split to reach split_size
            needed = split_size - len(current_split)
            overlap_from_prev = splits[-1][-needed:]  # Take last N from previous
            current_split = overlap_from_prev + current_split
        
        splits.append(current_split)
        start_idx += step
        
        # Break if next start would exceed total
        if start_idx >= total_images:
            break
    
    # Create split folders and copy images
    for i, split_images in enumerate(splits, 1):
        split_folder = output_root / f"{i}"
        split_folder.mkdir(exist_ok=True)
        
        print(f"Creating {split_folder} with {len(split_images)} images...")
        
        for img_path in split_images:
            # Use just the filename to avoid path conflicts
            dest_path = split_folder / img_path.name
            shutil.copy2(img_path, dest_path)
    
    print(f"\nDataset split complete!")
    print(f"Output saved to: {output_root}")
    print(f"Created {len(splits)} splits")
    
    # Verify overlaps
    for i in range(1, len(splits)):
        prev_set = {f.name for f in splits[i-1]}
        curr_set = {f.name for f in splits[i]}
        overlap = prev_set & curr_set
        print(f"Split {i} and {i+1} overlap: {len(overlap)} images")
    
    return output_root

import os
import shutil
import random
from pathlib import Path

def split_dataset_no_overlap(input_folder, out_folder, dataset_name=None, split_size=40, seed=None):
    """
    Splits files from input_folder into non-overlapping subfolders in out_folder.
    
    Args:
        input_folder (str or Path): Path to the folder containing original files.
        out_folder (str or Path): Path to the output directory where splits will be created.
        dataset_name (str, optional): If provided, creates a subfolder with this name.
        split_size (int): Maximum number of files per split folder (default: 40).
        seed (int, optional): Random seed for reproducibility.
    
    Creates:
        out_folder/[dataset_name]/split_1, split_2, ... each with <= split_size files.
        If dataset_name is None, splits are created directly in out_folder.
    """
    input_path = Path(input_folder)
    out_path = Path(out_folder)
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Validate input folder
    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"Input folder does not exist or is not a directory: {input_path}")
    
    # Get all files in the input folder (non-recursive)
    files = [f for f in input_path.iterdir() if f.is_file()]
    
    if not files:
        print("Warning: No files found in input folder.")
        return
    
    # Shuffle files randomly
    random.shuffle(files)
    
    # Determine base output directory
    base_out = out_path / dataset_name if dataset_name else out_path
    base_out.mkdir(parents=True, exist_ok=True)
    
    # Split files into chunks
    split_idx = 1
    for i in range(0, len(files), split_size):
        chunk = files[i:i + split_size]
        
        # Create split folder
        split_folder = base_out / f"split_{split_idx}"
        split_folder.mkdir(exist_ok=True)
        
        # Copy files to split folder
        for file_path in chunk:
            shutil.copy2(file_path, split_folder / file_path.name)
        
        print(f"Created {split_folder} with {len(chunk)} files.")
        split_idx += 1
    
    print(f"\nSplitting complete: {split_idx - 1} split(s) created in {base_out}")


if __name__ == "__main__":
    split_dataset("data/gerrard-hall/images/", dataset_name="gerrard-hall", split_size=40, min_overlap=5, seed=42)