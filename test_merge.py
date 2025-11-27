import pycolmap
from pathlib import Path
import tempfile
import os
import shutil

def merge_two_reconstructions(
    rec1_path: Path,
    rec2_path: Path,
    output_path: Path,
    min_common_images: int = 5,
    max_reproj_error: float = 8.0,
    min_inlier_ratio: float = 0.1,
):
    rec1_path = Path(rec1_path).resolve()
    rec2_path = Path(rec2_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    rec1 = pycolmap.Reconstruction(rec1_path)
    rec2 = pycolmap.Reconstruction(rec2_path)

    # Find common images by name
    name_to_id1 = {img.name: img_id for img_id, img in rec1.images.items()}
    name_to_id2 = {img.name: img_id for img_id, img in rec2.images.items()}
    common_names = set(name_to_id1.keys()) & set(name_to_id2.keys())

    if len(common_names) < min_common_images:
        raise RuntimeError(f"Only {len(common_names)} common images found")

    print(f"Found {len(common_names)} common images (by name) → remapping IDs in rec2 for merge...")

    # Create temporary directory for remapped rec2
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        rec2_remap_path = temp_path / "rec2_remapped"

        # Clone rec2 to temporary path
        shutil.copytree(rec2_path, rec2_remap_path)

        # Load the copied reconstruction
        rec2_remap = pycolmap.Reconstruction(rec2_remap_path)

        # Remap image IDs in rec2_remap to match rec1's IDs for common images
        id_map = {name_to_id2[name]: name_to_id1[name] for name in common_names}
        image_id_changes = []
        for old_id, new_id in id_map.items():
            if old_id != new_id and old_id in rec2_remap.images:
                image_id_changes.append((old_id, new_id))

        # Update images
        for old_id, new_id in image_id_changes:
            img = rec2_remap.images.pop(old_id)
            img.image_id = new_id
            rec2_remap.images[new_id] = img

        # Update point tracks
        for p3d_id, point3D in list(rec2_remap.points3D.items()):
            updated = False
            for old_id, new_id in image_id_changes:
                if old_id in point3D.track.track_elements:
                    elem = point3D.track.track_elements.pop(old_id)
                    elem.image_id = new_id
                    point3D.track.track_elements[new_id] = elem
                    updated = True
            if updated:
                point3D.track.num_elements = len(point3D.track.track_elements)
            if len(point3D.track.track_elements) == 0:
                del rec2_remap.points3D[p3d_id]

        # Save remapped rec2
        rec2_remap.write(rec2_remap_path)

        # Now align using the remapped rec2 (IDs match for common images)
        sim3 = pycolmap.align_reconstructions_via_reprojections(
            rec1,
            rec2_remap,
            max_reproj_error=max_reproj_error,
            min_inlier_observations=min_inlier_ratio,
        )

        if sim3 is None:
            raise RuntimeError(f"Alignment failed (try increasing max_reproj_error to {max_reproj_error * 2})")

        print(f"Alignment successful! Scale: {sim3.scale:.4f}")

        # Transform the original rec2 using the similarity
        rec2.transform(sim3)

        # Remap IDs in the transformed original rec2 to avoid conflicts during merge
        for old_id, new_id in id_map.items():
            if old_id != new_id and old_id in rec2.images:
                img = rec2.images.pop(old_id)
                img.image_id = new_id
                rec2.images[new_id] = img

                for p3d_id, point3D in list(rec2.points3D.items()):
                    updated = False
                    if old_id in point3D.track.track_elements:
                        elem = point3D.track.track_elements.pop(old_id)
                        elem.image_id = new_id
                        point3D.track.track_elements[new_id] = elem
                        updated = True
                    if updated:
                        point3D.track.num_elements = len(point3D.track.track_elements)
                    if len(point3D.track.track_elements) == 0:
                        del rec2.points3D[p3d_id]

        # Now merge (common IDs match, non-common are unique)
        merged = rec1.merge(rec2, min_common_observations=2)

        # Clean up
        num_removed = merged.normalize(
            max_reprojection_error=4.0,
            min_track_length=3,
            min_tri_angle=1.5,
        )
        print(f"Normalization removed {num_removed} points")

        # Optional bundle adjustment for refinement
        merged.bundle_adjustment()

        # Save
        merged.write(output_path)

        print(f"Merged model saved to {output_path}")
        print(f"   Images: {len(merged.images)} | Points: {len(merged.points3D)}")




if __name__ == "__main__":
    merge_two_reconstructions(
        rec1_path=Path("All_Reconstructions/mast3r/reconstructions/1/reconstruction/0"),
        rec2_path=Path("All_Reconstructions/mast3r/reconstructions/2/reconstruction/0"),
        output_path=Path("All_Reconstructions/mast3r/merged_reconstruction"),
        min_common_images=5
    )