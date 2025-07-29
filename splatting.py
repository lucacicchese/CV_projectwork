def gaussian_splatting(points, weights, grid_size):
    """
    Perform Gaussian splatting on a set of points with associated weights.

    Args:
        points (np.ndarray): An array of shape (N, 3) representing the 3D coordinates of the points.
        weights (np.ndarray): An array of shape (N,) representing the weights for each point.
        grid_size (tuple): A tuple (H, W) representing the size of the output grid.

    Returns:
        np.ndarray: A 2D array of shape (H, W) representing the splatted image.
    """
    import numpy as np

    H, W = grid_size
    splatted_image = np.zeros((H, W), dtype=np.float32)

    for point, weight in zip(points, weights):
        x, y = int(point[0]), int(point[1])
        if 0 <= x < W and 0 <= y < H:
            splatted_image[y, x] += weight

    return splatted_image