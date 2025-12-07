import os
import random
import shutil
from pathlib import Path

# ----------------------------- CONFIGURATION -----------------------------
# Change these paths according to your setup
SOURCE_FOLDER = "data/santissima/images"   # <-- UPDATE THIS
DESTINATION_FOLDER = "data/santissima/1000"  # <-- UPDATE THIS

# Supported image extensions (add more if needed)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp'}

# Number of random images to select
NUM_IMAGES = 1000
# --------------------------------------------------------------------------------

def get_random_images(source_dir, num_images=1000):
    source_path = Path(source_dir)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source folder not found: {source_path}")
    
    # Get all image files recursively (includes subfolders)
    image_files = [f for f in source_path.rglob('*') if f.suffix.lower() in IMAGE_EXTENSIONS and f.is_file()]
    
    if len(image_files) < num_images:
        print(f"Warning: Only {len(image_files)} images found, but {num_images} requested.")
        num_images = len(image_files)
    
    # Select random images
    selected_images = random.sample(image_files, num_images)
    print(f"Selected {len(selected_images)} random images.")
    return selected_images

def main():
    # Create destination folder if it doesn't exist
    dest_path = Path(DESTINATION_FOLDER)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Source folder: {SOURCE_FOLDER}")
    print(f"Destination folder: {DESTINATION_FOLDER}")
    print(f"Copying {NUM_IMAGES} random images...\n")
    
    try:
        random_images = get_random_images(SOURCE_FOLDER, NUM_IMAGES)
        
        for i, img_path in enumerate(random_images, 1):
            # Create a new filename to avoid conflicts (optional: keep original name)
            new_filename = f"image_{i:04d}{img_path.suffix.lower()}"
            dest_file = dest_path / new_filename
            
            # Copy the file
            shutil.copy2(img_path, dest_file)  # copy2 preserves metadata
            
            if i % 100 == 0 or i == len(random_images):
                print(f"Copied {i}/{len(random_images)} images...")
        
        print(f"\nDone! {len(random_images)} images copied to:\n   {dest_path.resolve()}")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Optional: set a random seed for reproducibility
    # random.seed(42)
    
    main()