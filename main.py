from colmap_reconstruction import colmap_reconstruction
from mast3r_reconstruction import mast3r_reconstruction
from vggt_reconstruction import vggt_reconstruction
from utils import split_dataset
import os
from multi_reconstruction import multi_stage_reconstruction
import shutil
from horn_loss_downloaded import score
from icp_loss import compute_icp_metrics
from reconstruction_to_csv import colmap_to_csv, create_thresholds_csv
import pycolmap
from pathlib import Path

if __name__ == "__main__":

    output_dir = "All_Reconstructions"
    dataset_name = "gerrard-hall"
    dataset_folder = f"data/{dataset_name}/images/"


    # FLAGS
    multi = False
    reconstruct = True
    recon_colmap = True
    recon_mast3r = True
    recon_vggt = True
    convert = True
    evaluate = True

   

    # Extract features using COLMAP, MAST3R, and VGGT
    if reconstruct == True:
    
        os.makedirs(output_dir, exist_ok=True)
        print("Starting reconstructions...")

        if recon_colmap == True:
            print("COLMAP RECONSTRUCTION")
            colmap_reconstruction(image_folder=dataset_folder, database_path="data/reconstruction/colmap.db")
            shutil.copytree("data/reconstruction/0/", f"{output_dir}/colmap", dirs_exist_ok=True)
        if multi == True:
            if recon_mast3r == True:
                print("MAST3R RECONSTRUCTION")
                multi_stage_reconstruction(dataset_folder, f"{output_dir}/mast3r/", "mast3r")
            if recon_vggt == True:
                print("VGGT RECONSTRUCTION")
                multi_stage_reconstruction(dataset_folder, f"{output_dir}/vggt/", "vggt")
        else:
            split_name = split_dataset(f"{dataset_folder}/images/",f"{dataset_folder}splits/", "mast3r") 
            if recon_mast3r == True:
                print("MAST3R RECONSTRUCTION")
                mast3r_reconstruction(dataset_path=f"{dataset_folder}splits/1/", output_dir=f"{output_dir}/mast3r/")
            if recon_vggt:
                print("VGGT RECONSTRUCTION")
                vggt_reconstruction(dataset_path=f"{dataset_folder}splits/1/", output_dir=f"{output_dir}/vggt/")
                shutil.copytree(f"{dataset_folder}splits/1/sparse/", f"{output_dir}/vggt/", dirs_exist_ok=True)


        print(f"All reconstructions available in: {output_dir}")

    # Prepare evaluation: convert colmap models to csv files
    eval_dir = Path("evaluate")
    eval_dir.mkdir(exist_ok=True)
    if convert == True:
        eval_dir = Path("evaluate")
        eval_dir.mkdir(exist_ok=True)
        if multi == True:
            colmap_model_path = f"{output_dir}/colmap/reconstruction/"
            mast3r_model_path = f"{output_dir}/mast3r/reconstructions/1/reconstruction/0/"
            vggt_model_path = f"{output_dir}/vggt/reconstructions/1/"
        else:
            colmap_model_path = f"{output_dir}/colmap/"
            mast3r_model_path = f"{output_dir}/mast3r/reconstruction/0/"
            vggt_model_path = f"{output_dir}/vggt/"


        gt_reconstruction = pycolmap.Reconstruction(colmap_model_path)
        mast3r_reconstruction = pycolmap.Reconstruction(mast3r_model_path)
        vggt_reconstruction = pycolmap.Reconstruction(vggt_model_path)

        gt = gt_reconstruction.images
        mast3r = mast3r_reconstruction.images
        vggt = vggt_reconstruction.images

        common_names_mast3r = set(img.name for img in gt.values()) & \
                            set(img.name for img in mast3r.values())
        
        common_names_vggt = set(img.name for img in gt.values()) & \
                            set(img.name for img in vggt.values())
        
        common_names_mast3rvggt = set(img.name for img in mast3r.values()) & \
                            set(img.name for img in vggt.values())


        colmap_to_csv(
            f"{colmap_model_path}/images.bin",
            eval_dir / "gt_poses_mast3r.csv",
            dataset_name="gerrard-hall",
            scene_name="my_scene",
            allowed_images=common_names_mast3r
        )

        colmap_to_csv(
            f"{mast3r_model_path}/images.bin",
            eval_dir / "user_poses_mast3r.csv",
            dataset_name="gerrard-hall",
            scene_name="my_scene",
            allowed_images=common_names_mast3r
        )

        colmap_to_csv(
            f"{colmap_model_path}/images.bin",
            eval_dir / "gt_poses_vggt.csv",
            dataset_name="gerrard-hall",
            scene_name="my_scene",
            allowed_images=common_names_vggt
        )

        colmap_to_csv(
            f"{vggt_model_path}/images.bin",
            eval_dir / "user_poses_vggt.csv",
            dataset_name="gerrard-hall",
            scene_name="my_scene",
            allowed_images=common_names_vggt
        )

        colmap_to_csv(
            f"{mast3r_model_path}/images.bin",
            eval_dir / "gt_mast3r_poses_vggt.csv",
            dataset_name="gerrard-hall",
            scene_name="my_scene",
            allowed_images=common_names_mast3rvggt
        )

        colmap_to_csv(
            f"{vggt_model_path}/images.bin",
            eval_dir / "user_mast3r_poses_vggt.csv",
            dataset_name="gerrard-hall",
            scene_name="my_scene",
            allowed_images=common_names_mast3rvggt
        )

        create_thresholds_csv(
            eval_dir / "thresholds.csv",
            dataset_name="gerrard-hall",
            scene_name="my_scene"
        )
        
        print("Files successfully created!")



    # Evaluation horn and icp losses

    if evaluate == True:

        print("Evaluating models...")

        print("Horn: Mast3r vs COLMAP")
        sc = score(
        gt_csv=f"{eval_dir}/gt_poses_mast3r.csv",
        user_csv=f"{eval_dir}/user_poses_mast3r.csv",
        thresholds_csv=f"{eval_dir}/thresholds.csv",
        skip_top_thresholds=0,  
        to_dec=0  
        )

        print("Horn: VGGT vs COLMAP")
        sc = score(
        gt_csv=f"{eval_dir}/gt_poses_vggt.csv",
        user_csv=f"{eval_dir}/user_poses_vggt.csv",
        thresholds_csv=f"{eval_dir}/thresholds.csv",
        skip_top_thresholds=0,  
        to_dec=0  
        )

        print("Horn: VGGT vs Mast3r")
        sc = score(
        gt_csv=f"{eval_dir}/gt_mast3r_poses_vggt.csv",
        user_csv=f"{eval_dir}/user_mast3r_poses_vggt.csv",
        thresholds_csv=f"{eval_dir}/thresholds.csv",
        skip_top_thresholds=0,  
        to_dec=0  
        )

        print("ICP: Mast3r vs COLMAP")
        compute_icp_metrics(
        f"{eval_dir}/gt_poses_mast3r.csv",
        f"{eval_dir}/user_poses_mast3r.csv",
        f"{eval_dir}/icp_results_mast3r.txt"
        )

        print("ICP: VGGT vs COLMAP")
        compute_icp_metrics(
        f"{eval_dir}/gt_poses_vggt.csv",
        f"{eval_dir}/user_poses_vggt.csv",
        f"{eval_dir}/icp_results_vggt.txt"
        )

        print("ICP: VGGT vs Mast3r")
        compute_icp_metrics(
        f"{eval_dir}/gt_mast3r_poses_vggt.csv",
        f"{eval_dir}/user_mast3r_poses_vggt.csv",
        f"{eval_dir}/icp_results_mast3rvggt.txt"
        )

        print("Evaluation completed.")