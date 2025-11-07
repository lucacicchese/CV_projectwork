import os
import numpy as np
import pycolmap
import subprocess

import os
import pycolmap

def reindex_model_ids(model_path, output_path, offset=1000):
    os.makedirs(output_path, exist_ok=True)
    rec = pycolmap.Reconstruction(model_path)
    new_rec = pycolmap.Reconstruction()

    # Copy cameras
    for cam in rec.cameras.values():
        new_rec.add_camera(cam)

    # Reindex images
    id_map = {}
    for img in rec.images.values():
        new_id = img.image_id + offset
        id_map[img.image_id] = new_id
        new_img = pycolmap.Image()
        new_img.name = img.name
        new_img.image_id = new_id
        new_img.camera_id = img.camera_id

        if hasattr(img, "qvec"):
            new_img.qvec = img.qvec
        if hasattr(img, "tvec"):
            new_img.tvec = img.tvec

        new_rec.add_image(new_img)

    # Add 3D points without any track info (safe)
    for p in rec.points3D.values():
        new_rec.add_point3D(p.xyz, pycolmap.Track(), p.color)

    new_rec.write(output_path)
    return output_path




def create_image_list(model1_path, model2_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    m1 = pycolmap.Reconstruction(model1_path)
    m2 = pycolmap.Reconstruction(model2_path)
    names1 = {i.name for i in m1.images.values()}
    names2 = {i.name for i in m2.images.values()}
    common = sorted(names1 & names2)
    path = os.path.join(output_folder, "image_list.txt")
    with open(path, "w") as f:
        f.write("\n".join(common))
    return path

def align_and_merge(model1_path, model2_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image_list = create_image_list(model1_path, model2_path, output_folder)

    reindexed_model2 = os.path.join(output_folder, "model2_reindexed")
    model2_path = reindex_model_ids(model2_path, reindexed_model2, offset=1000)

    aligned_model = os.path.join(output_folder, "aligned")
    merged_model = os.path.join(output_folder, "merged")
    os.makedirs(aligned_model, exist_ok=True)
    os.makedirs(merged_model, exist_ok=True)

    subprocess.run([
        "colmap", "model_aligner",
        "--input_path", model2_path,
        "--output_path", aligned_model,
        "--ref_images_path", image_list,
        "--ref_model_path", model1_path,
        "--alignment_max_error", "5"
    ], check=True)

    subprocess.run([
        "colmap", "model_merger",
        "--input_path1", model1_path,
        "--input_path2", aligned_model,
        "--output_path", merged_model
    ], check=True)

    return merged_model




if __name__ == "__main__":
    #image_list = create_image_list("data/gerrard-hall/mast3r_multi/reconstructions/1/reconstruction/0/", "data/gerrard-hall/mast3r_multi/reconstructions/2/reconstruction/0/", "data/gerrard-hall/mast3r_multi/")
    align_and_merge("data/gerrard-hall/mast3r_multi/reconstructions/1/reconstruction/0/", "data/gerrard-hall/mast3r_multi/reconstructions/2/reconstruction/0/", "data/gerrard-hall/mast3r_multi/")