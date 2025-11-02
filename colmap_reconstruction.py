import os
import subprocess
import pycolmap as colmap
from PIL import Image
import numpy as np


def colmap_reconstruction(image_folder, database_path):
    input_folder = image_folder
    resized_folder = f'{image_folder.rstrip("/")}_resized'
    output_path = "data/reconstruction"
    

    os.makedirs(resized_folder, exist_ok=True)
    

    need_resize = True
    if os.path.exists(resized_folder):
        original_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
        resized_files = [f for f in os.listdir(resized_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
        if len(original_files) == len(resized_files):
            need_resize = False
            print("Images already resized, skipping resize step...")
    
    if need_resize:
        print("Resizing images...")
        max_size = 1024
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                img_path = os.path.join(input_folder, filename)
                img = Image.open(img_path)
                img.thumbnail((max_size, max_size))
                img.save(os.path.join(resized_folder, filename))
    
    # Check if database already has features
    run_feature_extraction = True
    if os.path.exists(database_path):
        try:
            db = colmap.Database(database_path)
            if len(db.read_all_images()) > 0 and len(db.read_all_keypoints()) > 0:
                print("Features already extracted, skipping feature extraction...")
                run_feature_extraction = False
            db.close()
        except:
            pass
    
    if run_feature_extraction:
        print("Extracting features using COLMAP...")
        subprocess.run([
            "colmap", "feature_extractor",
            "--database_path", database_path,
            "--image_path", resized_folder
        ], check=True)
        
        print("Matching features using COLMAP...")
        subprocess.run([
            "colmap", "exhaustive_matcher",
            "--database_path", database_path
        ], check=True)
    
    print("Running incremental mapping using pycolmap...")
    os.makedirs(output_path, exist_ok=True)
    
    reconstructions = colmap.incremental_mapping(
        database_path=database_path,
        image_path=resized_folder,
        output_path=output_path
    )

    reconstructions[0].write(output_path)
 
    return 

if __name__ == "__main__":
    colmap_reconstruction(
        image_folder="data/gerrard-hall/images/",
        database_path="data/colmap.db"
    )
    
    print(f"Colmap reconstruction completed.")