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



if __name__ == "__main__":
    split_dataset("data/gerrard-hall/images/", dataset_name="gerrard-hall", split_size=40, min_overlap=5, seed=42)