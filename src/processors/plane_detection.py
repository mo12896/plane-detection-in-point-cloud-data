"""Plane Detection Interface and concrete RANSAC implementation"""
import pickle
from abc import abstractmethod
from pathlib import Path
from typing import Dict, Any, List

import open3d as o3d
from open3d.cpu.pybind.geometry import PointCloud
import numpy as np
import pyransac3d as pyrsc

import system_setup as setup
from utils.utils import timer
from utils.dataloader import DataLoader
from .pointcloud_processor import PointCloudProcessor


class PlaneDetection(PointCloudProcessor):
    """Abstract Class of Plane Detection"""

    @abstractmethod
    def detect_planes(self, filename: str) -> PointCloud:
        """Plane Detection"""

    @abstractmethod
    def _store_best_eqs(self, filename: str) -> None:
        """Store Best Plane Equations"""


class IterativeRANSAC(PlaneDetection):
    """
    Iterative RANSAC algorithm to detect n planes based on minimal plane size,
    set by the user.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        geometry: pyrsc.Plane,
        out_dir: Path,
        ransac_params: Dict[str, float],
        debug: bool = False,
        store: bool = False,
    ):

        self.dataloader = dataloader
        self.out_dir = out_dir
        self.geometry = geometry
        self.plane_size = ransac_params["PLANE_SIZE"]
        self.thresh = ransac_params["THRESH"]
        self.store = store
        self.debug = debug
        self.pcd_out: PointCloud = None
        self.eqs: List[List[Any]] = []
        # For debugging only!
        self.planes: List[PointCloud] = []

    @timer
    def detect_planes(self, filename: str) -> PointCloud:
        """Detect planes using an iterative RANSAC algorithm

        Args:
            filename (str): path to raw point cloud file

        Returns:
            PointCloud: downsampled point cloud without detected planes
        """

        cloud: PointCloud = self.dataloader.load_data(filename)
        points = np.asarray(cloud.points)

        print("Iterative RANSAC...")
        plane_counter = 0
        while True:
            # Find best plane using RANSAC
            best_eq, best_inliers = self.geometry.fit(points, self.thresh)

            # Only remove planes larger than size heuristic
            if len(best_inliers) < self.plane_size:
                break

            plane_counter += 1
            self.eqs.append(best_eq)
            # Remove the best inliers from overall point cloud
            pcd_points = o3d.geometry.PointCloud()
            pcd_points.points = o3d.utility.Vector3dVector(points)
            self.pcd_out = pcd_points.select_by_index(best_inliers, invert=True)

            if self.debug:
                plane = pcd_points.select_by_index(best_inliers)
                self.planes.append(plane)

            points = np.asarray(self.pcd_out.points)

        # Display plane removals during debugging
        if self.debug and self.planes:
            print("Debugging...")
            o3d.visualization.draw_geometries(self.planes)

        # Retain color information for final point cloud
        if not self.pcd_out:
            raise ValueError("No point cloud was generated!")
        self.pcd_out = self._restore_color(cloud, self.pcd_out)

        # Store intermediate point cloud data
        if self.store:
            self.save_pcs(filename, self.out_dir, self.pcd_out)

        # Store best plane equations
        self._store_best_eqs(filename)

        print(f"Identified {plane_counter} plane(s) in point cloud '{filename}'")
        return self.pcd_out

    def _store_best_eqs(self, filename: str) -> None:
        """Saves best plane equations in a pickle file

        Args:
            filename (str): filename as blueprint for pickle file
        """
        try:
            filename = filename.split(".")[0] + "_best_eqs"
            file_path = setup.LOGS_DIR / filename

            if file_path.is_file():
                file_path.unlink()

            with file_path.open("wb") as fp:
                pickle.dump(self.eqs, fp)
        except Exception as exc:
            print(exc)

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
