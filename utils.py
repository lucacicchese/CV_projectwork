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