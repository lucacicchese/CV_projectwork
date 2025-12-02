import numpy as np
import pycolmap
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation


class COLMAPReconstruction:
    def __init__(self, path):
        self.reconstruction = pycolmap.Reconstruction(path)
        self.cameras = {cam.camera_id: cam for cam in self.reconstruction.cameras.values()}
        self.images = {img.image_id: img for img in self.reconstruction.images.values()}
        self.points3D = {pt_id: pt for pt_id, pt in self.reconstruction.points3D.items()}
        
    def get_image_names(self):
        return {img.name: img_id for img_id, img in self.images.items()}
    
    def get_camera_pose(self, image_id):
        img = self.images[image_id]
        R = img.cam_from_world.rotation.matrix()
        t = img.cam_from_world.translation
        return R, t
    
    def get_camera_center(self, image_id):
        img = self.images[image_id]
        return img.projection_center()
    
    def get_point_cloud(self):
        return np.array([pt.xyz for pt in self.points3D.values()])


def find_shared_cameras(recon1, recon2):
    names1 = recon1.get_image_names()
    names2 = recon2.get_image_names()
    shared_names = set(names1.keys()) & set(names2.keys())
    return [(names1[name], names2[name]) for name in shared_names]


def horn_alignment(src_points, dst_points):
    src_centroid = src_points.mean(axis=0)
    dst_centroid = dst_points.mean(axis=0)
    
    src_centered = src_points - src_centroid
    dst_centered = dst_points - dst_centroid
    
    H = src_centered.T @ dst_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    src_scaled = src_centered
    dst_scaled = dst_centered
    scale = np.sum(dst_scaled * (src_scaled @ R.T)) / np.sum(src_scaled ** 2)
    
    t = dst_centroid - scale * R @ src_centroid
    
    return scale, R, t


def align_reconstructions(recon1, recon2, shared_pairs, alignment_method=horn_alignment):
    src = []
    dst = []
    for img1_id, img2_id in shared_pairs:
        C1 = recon1.get_camera_center(img1_id)
        C2 = recon2.get_camera_center(img2_id)
        dst.append(C1)
        src.append(C2)
    src = np.array(src)
    dst = np.array(dst)
    return alignment_method(src, dst)



def transform_pose(R_cam, t_cam, scale, R_align, t_align):
    R_w = R_cam.T
    t_w = -R_cam.T @ t_cam
    t_w_new = scale * (R_align @ t_w) + t_align
    R_w_new = R_align @ R_w
    R_cam_new = R_w_new.T
    t_cam_new = -R_cam_new @ t_w_new
    return R_cam_new, t_cam_new



def visualize_viewing_directions(recon1, recon2, title, scale2=1.0, R2=None, t2=None):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for img_id, img in recon1.images.items():
        center = recon1.get_camera_center(img_id)
        R_cam = img.cam_from_world.rotation.matrix()
        direction = R_cam.T @ np.array([0.0, 0.0, 1.0])
        direction /= np.linalg.norm(direction)
        ax.quiver(center[0], center[1], center[2],
                  direction[0], direction[1], direction[2],
                  length=0.5, arrow_length_ratio=0.3, color='blue')

    for img_id, img in recon2.images.items():
        center = recon2.get_camera_center(img_id)
        R_cam = img.cam_from_world.rotation.matrix()
        direction = R_cam.T @ np.array([0.0, 0.0, 1.0])
        direction /= np.linalg.norm(direction)

        if R2 is not None:
            center = scale2 * (R2 @ center) + t2
            direction = R2 @ direction
            direction /= np.linalg.norm(direction)

        ax.quiver(center[0], center[1], center[2],
                  direction[0], direction[1], direction[2],
                  length=0.5, arrow_length_ratio=0.3, color='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.tight_layout()



def transform_point(point, scale, R, t):
    return scale * R @ point + t


def visualize_cameras(recon1, recon2, title, scale2=1.0, R2=None, t2=None):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    centers1 = np.array([recon1.get_camera_center(img_id) for img_id in recon1.images.keys()])
    ax.scatter(centers1[:, 0], centers1[:, 1], centers1[:, 2], c='blue', marker='o', s=50, label='Model 1')
    
    centers2 = []
    for img_id in recon2.images.keys():
        center = recon2.get_camera_center(img_id)
        if R2 is not None:
            center = transform_point(center, scale2, R2, t2)
        centers2.append(center)
    centers2 = np.array(centers2)
    ax.scatter(centers2[:, 0], centers2[:, 1], centers2[:, 2], c='red', marker='^', s=50, label='Model 2')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(title)
    plt.tight_layout()



def visualize_point_clouds(recon1, recon2, title, scale2=1.0, R2=None, t2=None):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    pc1 = recon1.get_point_cloud()
    if len(pc1) > 0:
        sample1 = pc1[np.random.choice(len(pc1), min(5000, len(pc1)), replace=False)]
        ax.scatter(sample1[:, 0], sample1[:, 1], sample1[:, 2], c='blue', s=1, alpha=0.3, label='Model 1')
    
    pc2 = recon2.get_point_cloud()
    if len(pc2) > 0:
        if R2 is not None:
            pc2 = np.array([transform_point(p, scale2, R2, t2) for p in pc2])
        sample2 = pc2[np.random.choice(len(pc2), min(5000, len(pc2)), replace=False)]
        ax.scatter(sample2[:, 0], sample2[:, 1], sample2[:, 2], c='red', s=1, alpha=0.3, label='Model 2')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(title)
    plt.tight_layout()


def merge_reconstructions(recon1, recon2, shared_pairs, scale, R, t, output_path):
    merged = pycolmap.Reconstruction()
    
    shared_img2_ids = {img2_id for _, img2_id in shared_pairs}
    
    for cam_id, cam in recon1.cameras.items():
        merged.add_camera(cam)
    
    next_cam_id = max(recon1.cameras.keys()) + 1
    cam2_to_merged = {}
    for cam_id, cam in recon2.cameras.items():
        if cam_id not in merged.cameras:
            new_cam = pycolmap.Camera(
                model=cam.model,
                width=cam.width,
                height=cam.height,
                params=cam.params,
                camera_id=next_cam_id
            )
            merged.add_camera(new_cam)
            cam2_to_merged[cam_id] = next_cam_id
            next_cam_id += 1
        else:
            cam2_to_merged[cam_id] = cam_id
    
    img1_to_merged = {}
    for img_id, img in recon1.images.items():
        cam_from_world = pycolmap.Rigid3d(
            rotation=img.cam_from_world.rotation,
            translation=img.cam_from_world.translation
        )
        new_img = pycolmap.Image(
            name=img.name,
            camera_id=img.camera_id,
            cam_from_world=cam_from_world,
            image_id=img.image_id
        )
        
        num_points2D = len(img.points2D)
        new_img.points2D = [pycolmap.Point2D(xy=img.points2D[i].xy) for i in range(num_points2D)]
        
        merged.add_image(new_img)
        merged.register_image(img.image_id)
        img1_to_merged[img_id] = img.image_id
    
    next_img_id = max(recon1.images.keys()) + 1
    img2_to_merged = {}
    
    for img_id, img in recon2.images.items():
        if img_id in shared_img2_ids:
            continue
        
        R_cam, t_cam = recon2.get_camera_pose(img_id)
        R_new, t_new = transform_pose(R_cam, t_cam, scale, R, t)
        
        cam_from_world = pycolmap.Rigid3d(
            rotation=pycolmap.Rotation3d(R_new),
            translation=t_new
        )
        
        new_img = pycolmap.Image(
            name=img.name,
            camera_id=cam2_to_merged[img.camera_id],
            cam_from_world=cam_from_world,
            image_id=next_img_id
        )
        
        num_points2D = len(img.points2D)
        new_img.points2D = [pycolmap.Point2D(xy=img.points2D[i].xy) for i in range(num_points2D)]
        
        merged.add_image(new_img)
        merged.register_image(next_img_id)
        img2_to_merged[img_id] = next_img_id
        next_img_id += 1
    
    for pt_id, pt in recon1.points3D.items():
        new_track = pycolmap.Track()
        for elem in pt.track.elements:
            if elem.image_id in img1_to_merged:
                new_track.add_element(img1_to_merged[elem.image_id], elem.point2D_idx)
        
        if len(new_track.elements) > 0:
            merged.add_point3D(pt.xyz, new_track, pt.color)
    
    for pt_id, pt in recon2.points3D.items():
        xyz_new = transform_point(pt.xyz, scale, R, t)
        
        new_track = pycolmap.Track()
        for elem in pt.track.elements:
            if elem.image_id in img2_to_merged:
                new_track.add_element(img2_to_merged[elem.image_id], elem.point2D_idx)
        
        if len(new_track.elements) > 0:
            merged.add_point3D(xyz_new, new_track, pt.color)
    
    merged.write(output_path)
    return merged


def merge(path1, path2, output_path):
    recon1 = COLMAPReconstruction(path1)
    recon2 = COLMAPReconstruction(path2)
    
    shared_pairs = find_shared_cameras(recon1, recon2)
    print(f"Found {len(shared_pairs)} shared cameras")
    
    visualize_cameras(recon1, recon2, "Camera Positions - Before Alignment")
    visualize_viewing_directions(recon1, recon2, "Camera Viewing Directions - Before Alignment")
    visualize_point_clouds(recon1, recon2, "Point Clouds - Before Alignment")
    
    scale, R, t = align_reconstructions(recon1, recon2, shared_pairs)
    print(f"Alignment: scale={scale:.4f}")
    
    visualize_cameras(recon1, recon2, "Camera Positions - After Alignment", scale, R, t)
    visualize_viewing_directions(recon1, recon2, "Camera Viewing Directions - After Alignment", scale, R, t)
    visualize_point_clouds(recon1, recon2, "Point Clouds - After Alignment", scale, R, t)
    
    merged = merge_reconstructions(recon1, recon2, shared_pairs, scale, R, t, output_path)
    print(f"Merged reconstruction saved to {output_path}")
    print(f"Total cameras: {len(merged.cameras)}")
    print(f"Total images: {len(merged.images)}")
    print(f"Total 3D points: {len(merged.points3D)}")
    
    plt.show()


if __name__ == "__main__":
    path1 = "data/gerrard-hall/vggt_test/"
    path2 = "data/gerrard-hall/vggt_test/reconstructions/images_part3"
    output_path = "data/gerrard-hall/vggt_test/"
    
    merge(path1, path2, output_path)