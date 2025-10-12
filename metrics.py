import numpy as np
from scipy.spatial import cKDTree


def horn_loss(method_a, method_b, use_camera_centers=False):
    """
    Compute Horn alignment loss between two reconstruction methods.
    Returns loss, R, t.
    """
    # Use 3D points or camera centers
    if use_camera_centers:
        A = method_a["translations"]
        B = method_b["translations"]
    else:
        A = method_a["points3d"]
        B = method_b["points3d"]

    n = min(len(A), len(B))
    if n == 0:
        raise ValueError("No valid points to compare.")
    A, B = A[:n], B[:n]

    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    A_centered = A - centroid_A
    B_centered = B - centroid_B

    H = A_centered.T @ B_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B - R @ centroid_A

    A_aligned = (R @ A.T).T + t
    loss = np.mean(np.sum((A_aligned - B) ** 2, axis=1))

    return loss, R, t

def icp_loss(method_a, method_b, max_iterations=20, tolerance=1e-6, verbose=False):
    A = method_a.get("points3d")
    B = method_b.get("points3d")

    # Fallback: use camera centers if 3D points missing or empty
    if A is None or len(A) == 0 or B is None or len(B) == 0:
        print("[Warning] Missing or empty 'points3d' in one of the methods. Falling back to camera centers.")
        A = np.array([pose[:3, 3] for pose in method_a["camera_poses"]])
        B = np.array([pose[:3, 3] for pose in method_b["camera_poses"]])

    # Sample to comparable size for speed
    n = min(len(A), len(B), 50000)  # cap for efficiency
    A = A[:n]
    B = B[:n]

    # Initialize transformation
    R = np.eye(3)
    t = np.zeros(3)
    prev_loss = np.inf

    # Build KD-tree for nearest neighbor search
    tree_B = cKDTree(B)

    for it in range(max_iterations):
        # Apply current transformation
        A_transformed = (R @ A.T).T + t

        # Find nearest neighbors in B
        distances, indices = tree_B.query(A_transformed, k=1)
        B_matched = B[indices]

        # Compute optimal R, t using Horn’s method
        centroid_A = A_transformed.mean(axis=0)
        centroid_B = B_matched.mean(axis=0)
        H = (A_transformed - centroid_A).T @ (B_matched - centroid_B)
        U, S, Vt = np.linalg.svd(H)
        R_update = Vt.T @ U.T
        if np.linalg.det(R_update) < 0:
            Vt[-1, :] *= -1
            R_update = Vt.T @ U.T
        t_update = centroid_B - R_update @ centroid_A

        # Update cumulative transform
        R = R_update @ R
        t = R_update @ t + t_update

        # Compute mean squared error
        A_aligned = (R @ A.T).T + t
        loss = np.mean(np.sum((A_aligned - B_matched) ** 2, axis=1))

        if verbose:
            print(f"[ICP] Iter {it+1}/{max_iterations} - loss: {loss:.6f}")

        # Check convergence
        if abs(prev_loss - loss) < tolerance:
            if verbose:
                print(f"[ICP] Converged at iteration {it+1}")
            break
        prev_loss = loss

    return loss, R, t
