import sys
import os
from pathlib import Path
import torch
import numpy as np

# Add mast3r to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
mast3r_path = parent_dir / "mast3r"
sys.path.insert(0, str(mast3r_path))

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.utils.image import load_images


# Define schedule functions directly
def cosine_schedule(t, lr_start, lr_end):
    """Cosine learning rate schedule"""
    return lr_end + (lr_start - lr_end) * (1 + np.cos(np.pi * t)) / 2


def linear_schedule(t, lr_start, lr_end):
    """Linear learning rate schedule"""
    return lr_start + (lr_end - lr_start) * t


def extract_features_mast3r(image_folder, output_folder, model_name="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric", 
                             device='cuda', batch_size=1, schedule='cosine', lr=0.01, niter=300, 
                             image_size=512, max_images=None, scene_graph='swin'):
    """
    Extract features and perform SfM using MASt3R pipeline.
    
    Args:
        image_folder (str): Path to folder containing input images
        output_folder (str): Path to folder where outputs will be saved
        model_name (str): Model checkpoint name
        device (str): Device to run on ('cuda' or 'cpu')
        batch_size (int): Batch size for inference
        schedule (str): Learning rate schedule ('cosine' or 'linear')
        lr (float): Learning rate for optimization
        niter (int): Number of iterations for optimization
        image_size (int): Image size for processing (default: 512)
        max_images (int): Maximum number of images to process (None for all)
        scene_graph (str): Scene graph type ('complete', 'swin', 'oneref'). Use 'swin' or 'oneref' for lower memory
    
    Returns:
        dict: Dictionary containing scene reconstruction results
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Convert schedule string to function
    if schedule == 'cosine':
        schedule_fn = cosine_schedule
    elif schedule == 'linear':
        schedule_fn = linear_schedule
    else:
        raise ValueError(f"Unknown schedule: {schedule}. Use 'cosine' or 'linear'")
    
    # Load the model
    print(f"Loading model: {model_name}")
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    
    # Load images
    print(f"Loading images from: {image_folder}")
    images = load_images(image_folder, size=image_size)
    
    if len(images) == 0:
        raise ValueError(f"No images found in {image_folder}")
    
    # Limit number of images if specified
    if max_images is not None and len(images) > max_images:
        print(f"Limiting to {max_images} images (out of {len(images)})")
        images = images[:max_images]
    
    # Get the actual image file paths for the imgs_dict
    # load_images returns images but we need to track their original paths
    import glob
    image_files = sorted(glob.glob(os.path.join(image_folder, '*')))
    # Filter for common image extensions
    image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG'))]
    
    if max_images is not None:
        image_files = image_files[:max_images]
    
    print(f"Processing {len(images)} images with scene_graph='{scene_graph}'")
    
    # Create pairs for processing - use swin or oneref for lower memory usage
    pairs = make_pairs(images, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    print(f"Created {len(pairs)} image pairs")
    
    # Run inference
    print("Running inference...")
    output = inference(pairs, model, device, batch_size=batch_size)

    # Clean up to free memory
    torch.cuda.empty_cache() if device == 'cuda' else None
    
    # Perform sparse global alignment (SfM)
    print("Performing sparse global alignment...")
    
    # Debug: Check structures
    print("\nDEBUG: Checking structures...")
    print(f"Output type: {type(output)}")
    print(f"Output keys: {list(output.keys()) if isinstance(output, dict) else 'not a dict'}")
    
    # Check the structure of the output predictions
    if 'pred1' in output:
        print(f"pred1 type: {type(output['pred1'])}")
        if isinstance(output['pred1'], dict):
            print(f"pred1 keys (first 5): {list(output['pred1'].keys())[:5]}")
        elif hasattr(output['pred1'], 'shape'):
            print(f"pred1 shape: {output['pred1'].shape}")
    
    if 'view1' in output:
        print(f"view1 type: {type(output['view1'])}")
        if isinstance(output['view1'], dict):
            print(f"view1 keys (first 5): {list(output['view1'].keys())[:5]}")
    
    print(f"Pairs type: {type(pairs)}, length: {len(pairs)}")
    
    # Build imgs_dict mapping idx to the actual image file path
    # Since load_images doesn't preserve paths in the dict, we need to map them manually
    imgs_dict = {}
    for i, img in enumerate(images):
        idx = img['idx']
        # Use the actual file path from our list
        if i < len(image_files):
            imgs_dict[idx] = image_files[i]
        else:
            # Fallback to a string representation if something went wrong
            imgs_dict[idx] = f"image_{idx}"
    
    print(f"imgs_dict sample (first 3): {[(k, os.path.basename(v)) for k, v in list(imgs_dict.items())[:3]]}")
    
    # Check what indices the pairs are using
    if len(pairs) > 0:
        first_pair = pairs[0]
        print(f"First pair[0]['idx']: {first_pair[0]['idx']}, type: {type(first_pair[0]['idx'])}")
        print(f"First pair[1]['idx']: {first_pair[1]['idx']}, type: {type(first_pair[1]['idx'])}")
    
    # For sparse_global_alignment, based on the error, it seems imgs should be a list of strings
    # not a dict. Let's try passing a list of image paths instead
    imgs_list = [imgs_dict[i] for i in sorted(imgs_dict.keys())]
    print(f"imgs_list sample (first 3): {[os.path.basename(p) for p in imgs_list[:3]]}")
    
    # For sparse_global_alignment, we should pass the pairs list directly
    # The inference output is already structured correctly
    print(f"\nPassing {len(pairs)} pairs and {len(imgs_list)} images to sparse_global_alignment")

    # Perform sparse global alignment
    # Pass imgs_list (list of image paths) instead of dict
    scene = sparse_global_alignment(
        imgs_list,  # list of image file paths in order
        pairs,      # original pairs list from make_pairs
        output_folder,  
        model,
        lr1=lr,
        niter1=niter,
        schedule=schedule_fn,  # pass the function, not the string
        device=device
        # removed silent parameter - not supported
    )
    
    # Save results
    output_path = os.path.join(output_folder, "scene_reconstruction.npz")
    print(f"Saving results to: {output_path}")
    
    # Extract image paths for saving
    image_paths = []
    for img in images:
        if 'filepath' in img:
            image_paths.append(img['filepath'])
        elif 'img' in img:
            image_paths.append(img['img'])
        elif 'path' in img:
            image_paths.append(img['path'])
        else:
            for v in img.values():
                if isinstance(v, str):
                    image_paths.append(v)
                    break
    
    # Extract key results
    # SparseGA object has attributes, not getter methods
    results = {
        'focals': scene.get_focals().cpu().numpy(),
        'poses': scene.get_im_poses().cpu().numpy(),
        'pts3d': scene.get_pts3d().cpu().numpy() if hasattr(scene, 'get_pts3d') else None,
        'intrinsics': scene.intrinsics.cpu().numpy() if hasattr(scene, 'intrinsics') else None,
        'num_images': len(images),
        'image_paths': image_paths
    }
    
    # Save to npz file
    np.savez(output_path, **{k: v for k, v in results.items() if v is not None and not isinstance(v, list)})
    
    # Also save image paths separately
    with open(os.path.join(output_folder, "image_paths.txt"), 'w') as f:
        for path in results['image_paths']:
            f.write(f"{path}\n")
    
    print("Feature extraction and SfM complete!")
    print(f"Results saved to: {output_folder}")
    
    return results


if __name__ == "__main__":
    # Corrected path to images
    image_folder = "/home/studente/Documenti/luca_cicchese/CV_projectwork/data/gerrard-hall/images"
    results = extract_features_mast3r(
        image_folder=image_folder,
        output_folder="data/mast3r_reconstruction",
        model_name="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
        scene_graph='swin',  # Use 'swin' instead of 'complete' for lower memory
        image_size=512,
        max_images=20  # Limit number of images, increase if you have more RAM
    )
    print("Output files:", results)