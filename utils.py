import os
import shutil
import random
from pathlib import Path


def split_dataset(
    input_folder,
    out_folder,
    dataset_name=None,
    split_size=40,
    min_overlap=10,
    seed=None
):
    if seed is not None:
        random.seed(seed)

    input_path = Path(input_folder)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
    image_files = [
        f for f in input_path.iterdir()
        if f.suffix.lower() in image_extensions and f.is_file()
    ]

    if len(image_files) == 0:
        print("No image files found.")
        return None

    total_images = len(image_files)
    print(f"Found {total_images} images.")

    if dataset_name is None:
        dataset_name = input_path.name

    if split_size <= min_overlap:
        raise ValueError("split_size must be greater than min_overlap")

    output_root = Path(out_folder)
    output_root.mkdir(parents=True, exist_ok=True)

    # Shuffle all images once
    shuffled = image_files.copy()
    random.shuffle(shuffled)

    persistent_overlap = set()
    splits = []
    used_images = set()  
    idx = 0

    split_idx = 1
    while idx < total_images or persistent_overlap:
        current_split = set()

        # Always include persistent overlap images
        current_split.update(persistent_overlap)

        # Add new unique images until we reach split_size
        needed = split_size - len(current_split)
        new_images = []

        while len(new_images) < needed and idx < total_images:
            candidate = shuffled[idx]
            idx += 1
            if candidate.name not in {p.name for p in current_split}:
                new_images.append(candidate)
                used_images.add(candidate.name)

        current_split.update(new_images)

        if len(current_split) < split_size and idx >= total_images:
            remaining = [f for f in shuffled if f.name not in {p.name for p in persistent_overlap}]
            random.shuffle(remaining)
            for img in remaining:
                if len(current_split) >= split_size:
                    break
                if img.name not in {p.name for p in current_split}:
                    current_split.add(img)

        current_list = list(current_split)
        random.shuffle(current_list)  

        candidates_for_overlap = [img for img in current_list if img.name not in {p.name for p in persistent_overlap}]
        if len(candidates_for_overlap) < min_overlap:
            candidates_for_overlap = current_list

        random.shuffle(candidates_for_overlap)
        new_overlap = candidates_for_overlap[:min_overlap]
        persistent_overlap = set(new_overlap)  # replace old overlap with new selection

        splits.append(current_list)

        print(f"Split {split_idx}: {len(current_list)} images | "
              f"{len(new_overlap)} new overlap images carried forward")

        split_idx += 1

        if idx >= total_images and len(persistent_overlap) <= min_overlap:
            break

    for i, split_images in enumerate(splits, 1):
        split_folder = output_root / f"{i}"
        print(f"Creatingg split folder: {split_folder}")
        split_folder.mkdir(exist_ok=True)

        for img_path in split_images:
            shutil.copy2(img_path, split_folder / img_path.name)

    print(f"\nRandom overlapping split complete!")
    print(f"Created {len(splits)} splits in: {output_root}")

    print("\nOverlap between consecutive splits:")
    for i in range(len(splits) - 1):
        set1 = {p.name for p in splits[i]}
        set2 = {p.name for p in splits[i + 1]}
        overlap = set1 & set2
        print(f"  Part {i+1} and Part {i+2}: {len(overlap)} images (target: {min_overlap})")

    return output_root


if __name__ == "__main__":
    split_dataset(
        input_folder="data/gerrard-hall/images/",
        out_folder="data/gerrard-hall/splits_random_overlap",
        dataset_name="gerrard-hall",
        split_size=40,
        min_overlap=10,
        seed=42
    )