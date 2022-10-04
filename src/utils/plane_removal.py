"""PlaneRemoval Interface and concrete implementation for removing all detected planes"""
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import open3d as o3d

from .utils import remove_by_indices, timer
from .dataset import DataLoader

class PlaneRemoval(ABC):
    """Interface for removing detected planes"""

    @abstractmethod
    def remove_planes(self, filename: str):
        """Remove detected planes"""

    @abstractmethod
    def display_final_pc(self):
        """Display the final point cloud"""

class PlaneRemovalAll(PlaneRemoval):
    """
    This the class for removing detected planes, based on the extracted plane equations from the
    original point cloud data!
    """
    pcd_out = None

    def __init__(self,
                 out_dir: Path, 
                 eqs_dir: Path, 
                 dataloader: DataLoader, 
                 remove_params: dict() = {},
                 store: bool = True):

        self.dataloader = dataloader
        self.out_dir = out_dir
        self.eqs_dir = eqs_dir
        self.thresh = remove_params['THRESH']
        self.store = store

    @timer
    def remove_planes(self, filename: str):
        """Remove all planes based on heuristics set in the configuration file"""
        print("Remove planes from original point cloud...")
        # Read the point cloud from raw directory
        try:
            cloud = self.dataloader.load_data(filename)
        except Exception as exc:
            print(f"File {filename} could not be loaded!")
            print(exc)

        # Read the equations as python list
        try:
            eqs = filename.split('.')[0] + "_best_eqs"
            eqs_path = self.eqs_dir / eqs
            
            with eqs_path.open('rb') as fp:
               best_eqs = pickle.load(fp)
        except Exception as exc:
            print(f"File {eqs} could not be loaded!")
            print(exc)

        pts = np.asarray(cloud.points)

        # Remove the planes from original point cloud
        for plane_eq in best_eqs:
            dist_pts = (plane_eq[0] * pts[:, 0] + plane_eq[1] * pts[:, 1] + plane_eq[2] * pts[:, 2] + plane_eq[3]
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
            print("No point cloud was generated!")
            print(exc)

        # Store intermediate point cloud data
        if self.store:
            self._save_pcs(filename)

        return self.pcd_out

    def _save_pcs(self, filename: str) -> None:
        """
        Saves point cloud data to a file
        :param filename:
        :return:
        """
        data_path = self.out_dir / filename
        if not data_path.is_file():
            o3d.io.write_point_cloud(str(data_path), self.pcd_out)

    def display_final_pc(self) -> None:
        """
        Displays the final point cloud
        :return:
        """
        if not self.pcd_out:
            raise ValueError("You try to display an empty point cloud!")

        o3d.visualization.draw_geometries([self.pcd_out])
       





