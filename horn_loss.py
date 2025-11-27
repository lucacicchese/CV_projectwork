import numpy as np
import struct
import os
from collections import defaultdict

def read_images_bin(path):
    with open(os.path.join(path, "images.bin"), "rb") as f:
        num_images = struct.unpack("Q", f.read(8))[0]
        images = {}
        for _ in range(num_images):
            image_id = struct.unpack("I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("dddd", f.read(32))
            tx, ty, tz = struct.unpack("ddd", f.read(24))
            camera_id = struct.unpack("I", f.read(4))[0]

            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            name = name.decode("utf-8")

            num_points = struct.unpack("Q", f.read(8))[0]
            points2d = []
            for _ in range(num_points):
                x, y = struct.unpack("dd", f.read(16))
                point3d_id = struct.unpack("Q", f.read(8))[0]
                if point3d_id != -1:
                    points2d.append((x, y, point3d_id))

            images[name] = points2d
    return images

def read_points3d_bin(path):
    with open(os.path.join(path, "points3D.bin"), "rb") as f:
        num_points = struct.unpack("Q", f.read(8))[0]
        points = {}
        for _ in range(num_points):
            point_id = struct.unpack("Q", f.read(8))[0]
            x, y, z = struct.unpack("ddd", f.read(24))
            f.read(3)  # rgb
            f.read(8)  # error
            track_len = struct.unpack("Q", f.read(8))[0]
            f.read(8 * track_len)  # skip track
            points[point_id] = np.array([x, y, z])
    return points

def get_common_points(path1, path2, threshold=2.0):
    images1 = read_images_bin(path1)
    images2 = read_images_bin(path2)
    points1 = read_points3d_bin(path1)
    points2 = read_points3d_bin(path2)

    common_names = set(images1.keys()) & set(images2.keys())

    pid1_to_pid2 = defaultdict(list)  # pid1 → list of possible pid2

    for name in common_names:
        obs1 = images1[name]
        obs2 = images2[name]

        obs2_dict = {pid2: (x, y) for x, y, pid2 in obs2}

        for x1, y1, pid1 in obs1:
            best_pid2 = None
            min_dist = float('inf')
            for pid2, (x2, y2) in obs2_dict.items():
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < min_dist and dist < threshold:
                    min_dist = dist
                    best_pid2 = pid2
            if best_pid2 is not None:
                pid1_to_pid2[pid1].append(best_pid2)

    # Select unique: take mode if multiple
    matches = {}
    for pid1, pid2_list in pid1_to_pid2.items():
        if len(pid2_list) == 0:
            continue
        pid2 = max(set(pid2_list), key=pid2_list.count)
        count = pid2_list.count(pid2)
        if count / len(pid2_list) > 0.5 and pid1 in points1 and pid2 in points2:  # majority vote
            matches[pid1] = pid2

    if len(matches) < 10:
        return None, None

    pts1 = np.array([points1[k] for k in matches.keys()])
    pts2 = np.array([points2[v] for v in matches.values()])

    return pts1, pts2

def align_point_clouds_with_scale(src, dst):
    c1 = np.mean(src, axis=0)
    c2 = np.mean(dst, axis=0)
    src_c = src - c1
    dst_c = dst - c2

    scale = np.sqrt(np.sum(dst_c**2) / np.sum(src_c**2))
    src_c *= scale

    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    t = c2 - scale * R @ c1
    return R, t, scale

def horn_loss(R, t, scale, src, dst):
    transformed = scale * R @ src.T + t[:, np.newaxis]
    transformed = transformed.T
    return np.sum((transformed - dst)**2)

def compare_colmap_reconstructions_horn(path1, path2, threshold=2.0):
    pts1, pts2 = get_common_points(path1, path2, threshold)
    if pts1 is None:
        print("Not enough common points")
        return None

    R, t, scale = align_point_clouds_with_scale(pts1, pts2)
    loss = horn_loss(R, t, scale, pts1, pts2)

    print(f"Matched points: {len(pts1)} | Scale: {scale:.4f} | RMSE: {np.sqrt(loss/len(pts1)):.4f} m")

    return loss, R, t, scale

if __name__ == "__main__":
    path_gt = "data/reconstructions/colmap/reconstruction/"
    path_mast3r = "data/reconstructions/mast3r_multi/reconstructions/1/reconstruction/0/"
    path_vggt = "data/reconstructions/vggt_multi/reconstructions/1/"

    print("GT vs MASt3R")
    compare_colmap_reconstructions_horn(path_gt, path_mast3r)

    print("\nGT vs VGGT")
    compare_colmap_reconstructions_horn(path_gt, path_vggt)

    print("\nMASt3R vs VGGT")
    compare_colmap_reconstructions_horn(path_mast3r, path_vggt)