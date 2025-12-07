import sys
from pathlib import Path
import os
import shutil
from typing import List
from visual_merge import merge


def merge_all_colmap_models(root_folder: str, final_output_dir: str = None, tmp_dir: str = "tmp_merge") -> str:
    root = Path(root_folder).resolve()
    final_output_dir = Path(final_output_dir) if final_output_dir else root.parent / "merged_reconstruction"
    tmp = root.parent / tmp_dir
    tmp.mkdir(parents=True, exist_ok=True)

    def is_valid_model_dir(p: Path) -> bool:
        return (
            (p / "cameras.bin").exists()
            and (p / "images.bin").exists()
            and (p / "points3D.bin").exists()
        )

    
    model_dirs: List[Path] = [
        p for p in root.glob("**/reconstruction/0") if is_valid_model_dir(p)
    ]

    
    fallback_dirs = [
        p for p in root.iterdir()
        if p.is_dir() and is_valid_model_dir(p)
    ]

    # Avoid duplicates
    for f in fallback_dirs:
        if f not in model_dirs:
            model_dirs.append(f)

    if not model_dirs:
        raise RuntimeError("No valid COLMAP models found in either structure")

    print(f"Found {len(model_dirs)} COLMAP models to merge.")

    current = model_dirs.copy()
    step = 0

    while len(current) > 1:
        step += 1
        next_round = []
        for i in range(0, len(current), 2):
            if i + 1 == len(current):
                next_round.append(current[i])
                continue

            m1, m2 = current[i], current[i+1]
            out = tmp / f"step{step:02d}_pair{i//2:02d}"
            out.mkdir(parents=True, exist_ok=True)

            merge(path1=str(m1), path2=str(m2), output_path=str(out))

            if not is_valid_model_dir(out):
                raise ValueError(f"Merge failed: missing .bin files in {out}")

            next_round.append(out)

        current = next_round

    final_model = current[0]

    if final_output_dir.exists():
        shutil.rmtree(final_output_dir)
    final_output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copytree(final_model, final_output_dir, dirs_exist_ok=True)
    shutil.rmtree(tmp)

    return str(final_output_dir)


if __name__ == "__main__":
    merge_all_colmap_models("data/reconstructions/mast3r_multi/reconstructions")
