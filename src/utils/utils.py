from time import perf_counter
import functools
from typing import Callable, Any, List, Dict
from multiprocessing import Pool

from pathlib import Path
import yaml
import numpy as np
import open3d as o3d
from open3d.cpu.pybind.geometry import PointCloud


def load_dict_from_yaml(filename: Path) -> Dict[str, Any]:
    """Loads yaml file into python dictionary

    Args:
        folders (Path): Path to load configs from
    """
    try:
        configs: Dict[str, Any] = yaml.safe_load(filename.read_text())
        print("Loaded config file into python dict!")
    except yaml.YAMLError as exc:
        print(exc)
    return configs


def folder_cleanup(folders: List[Path]) -> None:
    """Clean up multiple folders

    Args:
        folders (List[Path]): List of folders to cleanup
    """
    try:
        for folder in folders:
            for f in folder.iterdir():
                filename = folder / f
                filename.unlink()
        print("Cleaned up relveant data folders!")
    except Exception as exc:
        print(exc)


def multi_processing(function: Callable, files: Any) -> None:
    """Utility function for multi-processing

    Args:
        function (Callable): Any function
        files (Any): files on which to call the function
    """
    try:
        pool = Pool()
        pool.map(function, files)
        pool.close()
        pool.join()
    except Exception as exc:
        print(exc)


def remove_by_indices(points: np.ndarray, indices: List[int]) -> np.ndarray:
    """Remove sub-lists in nested lists by index

    Args:
        points (np.ndarray): Input point cloud
        indices (List[int]): Indices of points to remove

    Returns:
        _type_: np.ndarray
    """
    final_points = []
    index_set = set(indices)
    for idx, point in enumerate(points):
        if index_set and idx in index_set:
            index_set.remove(idx)
            continue
        final_points.append(point.tolist())
    return np.asarray(final_points)


def display_pointcloud_from_array(points: np.ndarray) -> None:
    """Display pointcloud from numpy array

    Args:
        points (np.ndarray): points to display
    """
    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd_out])


# Taken form http://www.open3d.org/docs/latest/tutorial/Advanced/pointcloud_outlier_removal.html
def display_inlier_outlier(cloud: PointCloud, ind: List[int]) -> None:
    """Visualize selected points and the non-selected points

    Args:
        cloud (PointCloud): point cloud to display
        ind (List[int]): indicies of selected points
    """
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def timer(func):
    """Timer decorator

    Args:
        func (_type_): Any function to be timed

    Returns:
        _type_: Callable
    """

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start = perf_counter()
        out = func(*args, **kwargs)
        stop = perf_counter()
        print(f"Elapsed time of function {func.__name__!r}: {stop-start} seconds!")
        return out

    return wrapper_timer
