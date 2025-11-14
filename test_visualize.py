import numpy as np
import struct
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def visualize_registration(path1, path2, use_icp=True, max_iterations=50, subsample=100000):
    points1, points2 = get_common_points(path1, path2)
    
    if points1 is None:
        print("No common points found")
        return
    
    if use_icp:
        R, t, scale = icp_align(points1, points2, max_iterations=max_iterations)
    else:
        R, t, scale = align_point_clouds_with_scale(points1, points2)
    
    points1_aligned = scale * (R @ points1.T).T + t
    
    if len(points1) > subsample:
        indices = np.random.choice(len(points1), subsample, replace=False)
        points1_sub = points1[indices]
        points1_aligned_sub = points1_aligned[indices]
        points2_sub = points2[indices]
    else:
        points1_sub = points1
        points1_aligned_sub = points1_aligned
        points2_sub = points2
    
    fig = plt.figure(figsize=(18, 6))
    
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(points1_sub[:, 0], points1_sub[:, 1], points1_sub[:, 2], 
                c='red', s=1, alpha=0.5, label='Model 1 (original)')
    ax1.scatter(points2_sub[:, 0], points2_sub[:, 1], points2_sub[:, 2], 
                c='blue', s=1, alpha=0.5, label='Model 2')
    ax1.set_title('Before Registration')
    ax1.legend()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(points1_aligned_sub[:, 0], points1_aligned_sub[:, 1], points1_aligned_sub[:, 2], 
                c='red', s=1, alpha=0.5, label='Model 1 (aligned)')
    ax2.scatter(points2_sub[:, 0], points2_sub[:, 1], points2_sub[:, 2], 
                c='blue', s=1, alpha=0.5, label='Model 2')
    ax2.set_title('After Registration')
    ax2.legend()
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    distances = np.linalg.norm(points1_aligned - points2, axis=1)
    
    if len(distances) > subsample:
        distances_sub = distances[indices]
    else:
        distances_sub = distances
    
    ax3 = fig.add_subplot(133, projection='3d')
    scatter = ax3.scatter(points1_aligned_sub[:, 0], points1_aligned_sub[:, 1], points1_aligned_sub[:, 2], 
                         c=distances_sub, cmap='jet', s=2, vmin=0, vmax=np.percentile(distances_sub, 95))
    ax3.set_title('Registration Error (color-coded)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    cbar = plt.colorbar(scatter, ax=ax3, shrink=0.5)
    cbar.set_label('Distance Error')
    
    plt.tight_layout()
    plt.savefig('registration_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Number of points: {len(points1)}")
    print(f"Scale factor: {scale}")
    print(f"Mean error: {np.mean(distances):.6f}")
    print(f"Median error: {np.median(distances):.6f}")
    print(f"Max error: {np.max(distances):.6f}")
    print(f"95th percentile error: {np.percentile(distances, 95):.6f}")


colmap_reconstruction_path = f"All_Reconstructions/colmap/reconstruction"
mast3r_reconstruction_path = f"All_Reconstructions/mast3r/reconstructions/1/reconstruction/0"

visualize_registration(colmap_reconstruction_path, mast3r_reconstruction_path, use_icp=True)