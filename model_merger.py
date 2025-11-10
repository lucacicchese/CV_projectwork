import sys
from pathlib import Path
import os
import shutil
from pathlib import Path
from typing import List

# Set up paths
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
imped_path = parent_dir / "IMPED" / "src"

# Check for IMPED repository
if not imped_path.exists():
    print(f"ERROR: IMPED path not found at {imped_path}")
    print("Make sure the IMPED repository is cloned in the correct location")
    sys.exit(1)

# Add IMPED to Python path
sys.path.insert(0, str(imped_path))

# Import the function
from imped import align_colmap_models


def merge_all_colmap_models(root_folder: str, final_output_dir: str = None, tmp_dir: str = "tmp_merge") -> str:
    root = Path(root_folder).resolve()
    final_output_dir = Path(final_output_dir) if final_output_dir else root.parent / "merged_reconstruction"
    tmp = root.parent / tmp_dir
    tmp.mkdir(parents=True, exist_ok=True)

    model_dirs: List[Path] = [p for p in root.glob("**/reconstruction/0") 
                             if (p / "cameras.bin").exists() and (p / "images.bin").exists() and (p / "points3D.bin").exists()]

    if not model_dirs:
        raise RuntimeError("No valid COLMAP models found")

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

            align_colmap_models(model_path1=str(m1), model_path2=str(m2), output_model=str(out))

            if not ((out / "cameras.bin").exists() and (out / "images.bin").exists() and (out / "points3D.bin").exists()):
                raise ValueError(f"Merge failed: missing .bin files in {out}")

            next_round.append(out)
        current = next_round

    final_model = current[0]
    final_output_dir.mkdir(parents=True, exist_ok=True)
    if final_output_dir.exists():
        shutil.rmtree(final_output_dir)
    shutil.copytree(final_model, final_output_dir)
    shutil.rmtree(tmp)

    return str(final_output_dir)


if __name__ == "__main__":
    merge_all_colmap_models("data/reconstructions/mast3r_multi/reconstructions")
