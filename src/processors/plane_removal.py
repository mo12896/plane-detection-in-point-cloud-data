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

    @abstractmethod
    def _load_plane_eqs(self, filename: str) -> List[List[Any]]:
        """Load a list of detected plane equations"""


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
        # pcd_out=o3d.geometry.PointCloud(),
    ):

        self.dataloader = dataloader
        self.out_dir = out_dir
        self.eqs_dir = eqs_dir
        self.thresh = remove_params["THRESH"]
        self.store = store
        self.pcd_out: PointCloud = None

    @timer
    def remove_planes(self, filename: str) -> PointCloud:
        """Remove all planes based on stored plane equations in pickle file

        Args:
            filename (str): path to raw point cloud file

        Returns:
            PointCloud: raw point cloud without detected planes
        """

        # Load raw point cloud data and the best plane equations
        cloud: PointCloud = self.dataloader.load_data(filename)
        pts = np.asarray(cloud.points)

        best_eqs = self._load_plane_eqs(filename)

        print("Remove planes from original point cloud...")
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

        # Generate point cloud data
        self.pcd_out = o3d.geometry.PointCloud()
        self.pcd_out.points = o3d.utility.Vector3dVector(pts)
        self.pcd_out = self._restore_color(cloud, self.pcd_out)

        # Store intermediate point cloud data
        if self.store:
            self.save_pcs(filename, self.out_dir, self.pcd_out)

        return self.pcd_out

    def _load_plane_eqs(self, filename: str) -> List[List[Any]]:
        """Load plane equations from pickle file

        Args:
            filename (str): filename as blueprint for pickle file

        Returns:
            List[List[Any]]: list of best
        """
        try:
            eqs = filename.split(".")[0] + "_best_eqs"
            eqs_path = self.eqs_dir / eqs

            with eqs_path.open("rb") as fp:
                best_eqs: List[List[Any]] = pickle.load(fp)
        except Exception as exc:
            print(exc)
        return best_eqs

    @staticmethod
    def _restore_color(color_cloud: PointCloud, raw_cloud: PointCloud) -> PointCloud:
        """Restores color of raw point cloud

        Args:
            color_cloud (PointCloud): colorful source point cloud
            raw_cloud (PointCloud): unicolor target point cloud

        Returns:
            PointCloud: colorfied target point cloud
        """
        try:
            dists = np.array(color_cloud.compute_point_cloud_distance(raw_cloud))
            ind = np.where(dists < 0.01)[0]
            raw_cloud = color_cloud.select_by_index(ind)
        except Exception as exc:
            print(exc)
        return raw_cloud
