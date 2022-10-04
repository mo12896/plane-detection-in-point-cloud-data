"""Plane Detection Interface and concrete RANSAC implementation"""
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import open3d as o3d
from open3d.cpu.pybind.geometry import PointCloud
import numpy as np
import pyransac3d as pyrsc

import system_setup as setup
from .utils import timer
from .dataset import DataLoader


class PlaneDetection(ABC):
    """Plane Detection Interface"""

    @abstractmethod
    def detect_planes(self, filename: str) -> PointCloud:
        """Plane Detection"""

    @abstractmethod
    def store_best_eqs(self, filename: str) -> None:
        """Store Best Plane Equations"""

    @abstractmethod
    def display_final_pc(self) -> None:
        """Display Final PointCloud"""

class IterativeRANSAC(PlaneDetection):
    """
    Iterative RANSAC algorithm to detect n planes based on minimal plane size, 
    set by the user.
    """

    pcd_out = None
    eqs = []
    # For debugging only!
    planes = []

    def __init__(self,
                 dataloader: DataLoader,
                 data_dir: Path, 
                 ransac_params: Dict[str, float],
                 debug: bool = False,
                 store: bool = False):

        self.dataloader = dataloader
        self.data_dir = data_dir
        self.plane_size = ransac_params['PLANE_SIZE']
        self.thresh = ransac_params['THRESH']
        self.store = store
        self.debug = debug

    @timer
    def detect_planes(self, filename: str) -> PointCloud:
        """Detect planes using an iterative RANSAC algorithm"""
        try:
            cloud = self.dataloader.load_data(filename)
        except Exception as exc:
            print(f"File {filename} could not be loaded!")
            print(exc)

        print("Iterative RANSAC...")
        points = np.asarray(cloud.points)

        plane_counter = 0
        while True:
            # Find best plane using RANSAC
            plane = pyrsc.Plane()
            best_eq, best_inliers = plane.fit(points, self.thresh)

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
        else:
            raise ValueError("Debugging is not possible!")

        # Retain color information for final point cloud
        if not self.pcd_out:
            raise ValueError("No point cloud was generated!")

        dists = np.array(cloud.compute_point_cloud_distance(self.pcd_out))
        ind = np.where(dists < 0.01)[0]
        self.pcd_out = cloud.select_by_index(ind)

        # Store intermediate point cloud data
        if self.store: self._save_pcs(filename)

        print(f"Identified {plane_counter} plane(s) in point cloud '{filename}'")
        return self.pcd_out

    def _save_pcs(self, filename) -> None:
        """
        Saves point cloud data to a file
        :param filename:
        :return:
        """
        data_path = self.data_dir / filename
        if not data_path.is_file():
            o3d.io.write_point_cloud(str(data_path), self.pcd_out)

    def store_best_eqs(self, filename: str) -> None:
        """
        Saves best plane equations in a pickle file
        :param filename:
        :return:
        """
        try:
            filename = filename.split('.')[0] + "_best_eqs"
            file_path = setup.LOGS_DIR / filename

            if file_path.is_file(): file_path.unlink() 

            with file_path.open('wb') as fp:
                pickle.dump(self.eqs, fp)
        except Exception as exc:
            print("No plane equations were extracted!")
            print(exc)
    
    def display_final_pc(self) -> None:
        """
        Displays the final point cloud
        :return:
        """
        if not self.pcd_out:
            raise ValueError("You try to display an empty point cloud!")
        
        o3d.visualization.draw_geometries([self.pcd_out])



