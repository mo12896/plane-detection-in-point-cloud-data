import time

import numpy as np
import open3d as o3d
from time import perf_counter
import functools


def remove_by_indices(points: np.ndarray, indices: list):
    """Remove sub-lists in nested lists by index"""
    final_points = []
    index_set = set(indices)
    for idx, point in enumerate(points):
        if index_set and idx in index_set:
            index_set.remove(idx)
            continue
        final_points.append(point.tolist())
    return np.asarray(final_points)


def display_pointcloud_from_array(points: np.ndarray):
    """Display pointcloud from numpy array"""
    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd_out])


# Taken form http://www.open3d.org/docs/latest/tutorial/Advanced/pointcloud_outlier_removal.html
def display_inlier_outlier(cloud, ind):
    """Visualize selected points and the non-selected points"""
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def timer(func):
    """Timer decorator"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start = perf_counter()
        out = func(*args, **kwargs)
        stop = perf_counter()
        print(f"Elapsed time of function {func.__name__!r}: {stop-start} seconds!")
        return out

    return wrapper_timer
