"""PlaneRemoval Interface and concrete implementation for removing all detected planes"""
import pickle
from abc import abstractmethod
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import open3d as o3d
from open3d.cpu.pybind.geometry import PointCloud

from utils.utils import remove_by_indices, timer
from utils.dataloader import DataLoader
from .pointcloud_processor import PointCloudProcessor


class PlaneRemoval(PointCloudProcessor):
    """Abstract Class for removing detected planes"""

    @abstractmethod
    def remove_planes(self, filename: str) -> PointCloud:
        """Remove detected planes"""


class PlaneRemovalAll(PlaneRemoval):
    """
    This the class for removing detected planes, based on the extracted plane equations
    from the original point cloud data!
    """

    def __init__(
        self,
        out_dir: Path,
        eqs_dir: Path,
        dataloader: DataLoader,
        remove_params: Dict[str, float],
        store: bool = True,
    ):

        self.dataloader = dataloader
        self.out_dir = out_dir
        self.eqs_dir = eqs_dir
        self.thresh = remove_params["THRESH"]
        self.store = store
        self.pcd_out: PointCloud = None

    @timer
    def remove_planes(self, filename: str) -> PointCloud:
        """Remove all planes based on heuristics set in the configuration file"""
        print("Remove planes from original point cloud...")
        # Read the point cloud from raw directory
        try:
            cloud: PointCloud = self.dataloader.load_data(filename)
        except Exception as exc:
            print(exc)

        # TODO: separate equation loader method
        # Read the equations as python list
        try:
            eqs = filename.split(".")[0] + "_best_eqs"
            eqs_path = self.eqs_dir / eqs

            with eqs_path.open("rb") as fp:
                best_eqs: List[List[Any]] = pickle.load(fp)
        except Exception as exc:
            print(exc)

        pts = np.asarray(cloud.points)

        # Remove the planes from original point cloud
        for plane_eq in best_eqs:
            dist_pts = (
                plane_eq[0] * pts[:, 0]
                + plane_eq[1] * pts[:, 1]
                + plane_eq[2] * pts[:, 2]
                + plane_eq[3]
            ) / np.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)
            inliers = np.where(np.abs(dist_pts) <= self.thresh)[0].tolist()
            pts = remove_by_indices(pts, inliers)

        self.pcd_out = o3d.geometry.PointCloud()
        self.pcd_out.points = o3d.utility.Vector3dVector(pts)

        # Retain color information for final point cloud
        try:
            dists = np.array(cloud.compute_point_cloud_distance(self.pcd_out))
            ind = np.where(dists < 0.01)[0]
            self.pcd_out = cloud.select_by_index(ind)
        except Exception as exc:
            print(exc)

        # Store intermediate point cloud data
        if self.store:
            self.save_pcs(filename, self.out_dir, self.pcd_out)

        return self.pcd_out
