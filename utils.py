import os
import shutil
import random
from pathlib import Path

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

