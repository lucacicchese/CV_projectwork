import os
import struct
import subprocess
import numpy as np
from pathlib import Path

def read_images_binary(path):
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("dddd", f.read(32))
            tx, ty, tz = struct.unpack("ddd", f.read(24))
            camera_id = struct.unpack("I", f.read(4))[0]
            name = b""
            while True:
                char = f.read(1)
                if char == b"\x00":
                    break
                name += char
            name = name.decode("utf-8")
            num_points2D = struct.unpack("Q", f.read(8))[0]
            points2D = []
            for _ in range(num_points2D):
                x, y = struct.unpack("dd", f.read(16))
                point3D_id = struct.unpack("Q", f.read(8))[0]
                points2D.append((x, y, point3D_id))
            images[image_id] = {
                "qvec": np.array([qw, qx, qy, qz]),
                "tvec": np.array([tx, ty, tz]),
                "camera_id": camera_id,
                "name": name,
                "points2D": points2D
            }
    return images

def write_images_binary(images, path):
    with open(path, "wb") as f:
        f.write(struct.pack("Q", len(images)))
        for image_id in sorted(images.keys()):
            img = images[image_id]
            f.write(struct.pack("I", image_id))
            f.write(struct.pack("dddd", *img["qvec"]))
            f.write(struct.pack("ddd", *img["tvec"]))
            f.write(struct.pack("I", img["camera_id"]))
            f.write(img["name"].encode("utf-8") + b"\x00")
            f.write(struct.pack("Q", len(img["points2D"])))
            for x, y, point3D_id in img["points2D"]:
                f.write(struct.pack("dd", x, y))
                f.write(struct.pack("Q", point3D_id))

def copy_file(src, dst):
    with open(src, "rb") as f:
        data = f.read()
    with open(dst, "wb") as f:
        f.write(data)

def read_points3D_binary(path):
    points3D = {}
    with open(path, "rb") as f:
        num_points = struct.unpack("Q", f.read(8))[0]
        for _ in range(num_points):
            point3D_id = struct.unpack("Q", f.read(8))[0]
            xyz = struct.unpack("ddd", f.read(24))
            rgb = struct.unpack("BBB", f.read(3))
            error = struct.unpack("d", f.read(8))[0]
            track_length = struct.unpack("Q", f.read(8))[0]
            track = []
            for _ in range(track_length):
                image_id = struct.unpack("I", f.read(4))[0]
                point2D_idx = struct.unpack("I", f.read(4))[0]
                track.append((image_id, point2D_idx))
            points3D[point3D_id] = {
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "track": track
            }
    return points3D

def align_image_ids(model1_path, model2_path, output_path, common_images_txt=None):
    os.makedirs(output_path, exist_ok=True)
    
    images1 = read_images_binary(os.path.join(model1_path, "images.bin"))
    images2 = read_images_binary(os.path.join(model2_path, "images.bin"))
    
    name_to_id1 = {img["name"]: img_id for img_id, img in images1.items()}
    name_to_id2 = {img["name"]: img_id for img_id, img in images2.items()}
    
    if common_images_txt:
        with open(common_images_txt, "r") as f:
            common_names = set(line.strip() for line in f if line.strip())
    else:
        common_names = set(name_to_id1.keys()) & set(name_to_id2.keys())
    
    new_images2 = {}
    for name in images2.values():
        img_name = name["name"]
        if img_name in common_names and img_name in name_to_id1:
            new_id = name_to_id1[img_name]
            old_id = name_to_id2[img_name]
            img_data = images2[old_id].copy()
            
            points1 = images1[new_id]["points2D"]
            points2 = images2[old_id]["points2D"]
            
            num_points1 = len(points1)
            num_points2 = len(points2)
            
            if num_points2 < num_points1:
                aligned_points = list(points2)
                for i in range(num_points1 - num_points2):
                    aligned_points.append((0.0, 0.0, 18446744073709551615))
            elif num_points2 > num_points1:
                aligned_points = points2[:num_points1]
            else:
                aligned_points = points2
            
            img_data["points2D"] = aligned_points
            new_images2[new_id] = img_data
        else:
            old_id = name_to_id2[img_name]
            new_images2[old_id] = images2[old_id]
    
    write_images_binary(new_images2, os.path.join(output_path, "images.bin"))
    
    for filename in ["cameras.bin", "points3D.bin"]:
        src = os.path.join(model2_path, filename)
        dst = os.path.join(output_path, filename)
        if os.path.exists(src):
            copy_file(src, dst)



if __name__ == "__main__":
    os.makedirs("data/gerrard-hall/mast3r_multi/reconstructions/2/reconstruction/aligned/", exist_ok=True)
    model1_path = "data/gerrard-hall/mast3r_multi/reconstructions/1/reconstruction/0/"
    model2_path = "data/gerrard-hall/mast3r_multi/reconstructions/2/reconstruction/0/"
    common_images_txt = None
    output_path = "data/gerrard-hall/mast3r_multi/reconstructions/2/reconstruction/aligned/"
    merged_output_path = "data/gerrard-hall/mast3r_multi"
    
    align_image_ids(model1_path, model2_path, output_path, common_images_txt)
    print(f"Aligned model saved to {output_path}")
    
    os.makedirs(merged_output_path, exist_ok=True)
    
    cmd = [
        "colmap", "model_merger",
        "--input_path1", model1_path,
        "--input_path2", output_path,
        "--output_path", merged_output_path,
        "--max_reproj_error", "8.0"
    ]
    
    print("Running COLMAP model_merger...")
    result = subprocess.run(cmd, check=True)
    print(f"Merged model saved to {merged_output_path}")

