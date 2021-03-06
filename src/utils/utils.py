import time

import numpy as np
import open3d as o3d
from time import perf_counter
import functools


# own implementation of index removal
# Note that it runs a little slower than the open3D select_by_index
def remove_indices(points, indices):
    final_points = []
    index_set = set(indices)
    for point, coor in enumerate(points):
        if index_set and point in index_set:
            index_set.remove(point)
            continue
        final_points.append(coor)
    return final_points


# Display pointcloud from numpy array
def display_pointcloud_from_array(points: np.ndarray):
    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd_out])


# Visualize selected points and the non-selected points
# Taken form http://www.open3d.org/docs/latest/tutorial/Advanced/pointcloud_outlier_removal.html
def display_inlier_outlier(cloud, ind, verbose=False):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


# Timer decorator
def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start = perf_counter()
        out = func(*args, **kwargs)
        stop = perf_counter()
        print(f"Elapsed time of function {func.__name__!r}: {stop-start} seconds!")
        return out
    return wrapper_timer
