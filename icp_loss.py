import numpy as np
import struct
from scipy.spatial import KDTree

def read_colmap_images_bin(path):
    images_file = path + "/images.bin"
    images = {}
    
    with open(images_file, 'rb') as f:
        num_images = struct.unpack('Q', f.read(8))[0]
        
        for i in range(num_images):
            image_id = struct.unpack('I', f.read(4))[0]
            qw, qx, qy, qz = struct.unpack('dddd', f.read(32))
            tx, ty, tz = struct.unpack('ddd', f.read(24))
            camera_id = struct.unpack('I', f.read(4))[0]
            
            name_len = 0
            name_bytes = b''
            while True:
                char = f.read(1)
                if char == b'\x00':
                    break
                name_bytes += char
            name = name_bytes.decode('utf-8')
            
            num_points2d = struct.unpack('Q', f.read(8))[0]
            points2d = []
            for j in range(num_points2d):
                x, y = struct.unpack('dd', f.read(16))
                point3d_id = struct.unpack('Q', f.read(8))[0]
                points2d.append((x, y, point3d_id))
            
            images[name] = {
                'id': image_id,
                'points2d': points2d
            }
    
    return images

def read_colmap_points_bin(path):
    points3d_file = path + "/points3D.bin"
    points = {}
    
    with open(points3d_file, 'rb') as f:
        num_points = struct.unpack('Q', f.read(8))[0]
        
        for i in range(num_points):
            point_id = struct.unpack('Q', f.read(8))[0]
            x, y, z = struct.unpack('ddd', f.read(24))
            r, g, b = struct.unpack('BBB', f.read(3))
            error = struct.unpack('d', f.read(8))[0]
            track_length = struct.unpack('Q', f.read(8))[0]
            
            for j in range(track_length):
                image_id = struct.unpack('I', f.read(4))[0]
                point2d_idx = struct.unpack('I', f.read(4))[0]
            
            points[point_id] = np.array([x, y, z])
    
    return points

def get_common_points(path1, path2):
    images1 = read_colmap_images_bin(path1)
    images2 = read_colmap_images_bin(path2)
    points1 = read_colmap_points_bin(path1)
    points2 = read_colmap_points_bin(path2)
    
    common_image_names = set(images1.keys()) & set(images2.keys())
    
    point_matches = {}
    
    for img_name in common_image_names:
        points2d_1 = images1[img_name]['points2d']
        points2d_2 = images2[img_name]['points2d']
        
        for i, (x1, y1, p3d_id1) in enumerate(points2d_1):
            if p3d_id1 == -1:
                continue
            if i >= len(points2d_2):
                continue
            x2, y2, p3d_id2 = points2d_2[i]
            
            if p3d_id2 != -1 and p3d_id1 in points1 and p3d_id2 in points2:
                if p3d_id1 not in point_matches:
                    point_matches[p3d_id1] = []
                point_matches[p3d_id1].append(p3d_id2)
    
    matched_pairs = []
    for p1_id, p2_ids in point_matches.items():
        p2_id = max(set(p2_ids), key=p2_ids.count)
        matched_pairs.append((points1[p1_id], points2[p2_id]))
    
    if len(matched_pairs) == 0:
        return None, None
    
    pts1 = np.array([p[0] for p in matched_pairs])
    pts2 = np.array([p[1] for p in matched_pairs])
    
    return pts1, pts2

def align_point_clouds_with_scale(src, dst):
    centroid_src = np.mean(src, axis=0)
    centroid_dst = np.mean(dst, axis=0)
    
    src_centered = src - centroid_src
    dst_centered = dst - centroid_dst
    
    scale = np.sqrt(np.sum(dst_centered ** 2) / np.sum(src_centered ** 2))
    src_centered *= scale
    
    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    t = centroid_dst - scale * R @ centroid_src
    
    return R, t, scale

def icp_align(src, dst, max_iterations=50, tolerance=1e-6):
    R = np.eye(3)
    t = np.zeros(3)
    scale = 1.0
    
    src_transformed = src.copy()
    
    for iteration in range(max_iterations):
        tree = KDTree(dst)
        distances, indices = tree.query(src_transformed)
        
        matched_dst = dst[indices]
        
        R_iter, t_iter, scale_iter = align_point_clouds_with_scale(src_transformed, matched_dst)
        
        src_transformed = scale_iter * (R_iter @ src_transformed.T).T + t_iter
        
        R = R_iter @ R
        t = scale_iter * R_iter @ t + t_iter
        scale = scale * scale_iter
        
        mean_error = np.mean(distances)
        if iteration > 0 and abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    
    return R, t, scale

def icp_loss(R, t, scale, src, dst):
    transformed = scale * (R @ src.T).T + t
    tree = KDTree(dst)
    distances, indices = tree.query(transformed)
    loss = np.sum(distances ** 2)
    return loss

def compare_colmap_reconstructions_icp(path1, path2, max_iterations=50):
    points1, points2 = get_common_points(path1, path2)
    
    if points1 is None:
        print("No common points found")
        return None
    
    R, t, scale = icp_align(points1, points2, max_iterations=max_iterations)
    loss = icp_loss(R, t, scale, points1, points2)
    
    print(f"Number of matched points: {len(points1)}")
    print(f"Scale factor: {scale}")
    print(f"Mean error per point: {np.sqrt(loss / len(points1))}")
    
    return loss, R, t, scale
if __name__ == "__main__":
    path_first = "data/reconstruction/0"
    path_second = "data/reconstruction_first part_colmap/"
    loss, R, t, scale = compare_colmap_reconstructions_icp(path_first, path_second)
    print(f"Icp loss mast3r: {loss}")

#if __name__ == "__main__":
#    path_gt = "data/reconstructions/colmap/reconstruction/"
#    path_pred_mast3r = "data/reconstructions/mast3r_multi/reconstructions/1/reconstruction/0/"
#    path_pred_vggt = "data/reconstructions/vggt_multi/reconstructions/1/"
#    loss, R, t, scale = compare_colmap_reconstructions_icp(path_gt, path_pred_mast3r)
#    print(f"Icp loss mast3r: {loss}")
#    loss, R, t, scale = compare_colmap_reconstructions_icp(path_gt, path_pred_vggt)
#    print(f"Icp loss vggt: {loss}")
#    loss, R, t, scale = compare_colmap_reconstructions_icp(path_pred_mast3r, path_pred_vggt)
#    print(f"Icp loss vggt vs mast3r: {loss}")