import os
import numpy as np
import open3d as o3d
import pickle
from abc import ABC, abstractmethod

from .utils import remove_by_indices, timer
from .dataset import DataLoader

class PlaneRemoval(ABC):

    @abstractmethod
    def remove_planes(self, file: str):
        pass

    @abstractmethod
    def display_final_pc(self):
        pass


class PlaneRemovalAll(PlaneRemoval):
    """
    This the class for removing detected planes, based on the extracted plane equations from the
    original point cloud data!
    """
    pcd_out = None

    def __init__(self,
                 out_dir: str, 
                 eqs_dir: str, 
                 dataloader: DataLoader, 
                 remove_params: dict() = {},
                 store: bool = True):

        self.dataloader = dataloader
        self.out_dir = out_dir
        self.eqs_dir = eqs_dir
        self.thresh = remove_params['THRESH']
        self.store = store

    @timer
    def remove_planes(self, file: str):
        print("Remove planes from original point cloud...")
        # Read the point cloud from raw directory
        try:
            cloud = self.dataloader.load_data(file)
        except:
            print(f"File {file} could not be loaded!")

        # Read the equations as python list
        try:
            eqs = file.split('.')[0] + "_best_eqs"
            eqs_path = os.path.join(self.eqs_dir, eqs)
            
            with open(eqs_path, 'rb') as fp:
                best_eqs = pickle.load(fp)
        except:
            print(f"File {eqs} could not be loaded!")

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
        except:
            print("No point cloud was generated!")

        # Store intermediate point cloud data
        # TODO: Add functionality in DataLoader and call it DataHandler
        if self.store:
            data_path = os.path.join(self.out_dir, file)
            if not os.path.isfile(data_path):
                o3d.io.write_point_cloud(data_path, self.pcd_out)

        return self.pcd_out

    def display_final_pc(self):
        if not self.pcd_out:
            raise ValueError("You try to display an empty point cloud!")

        o3d.visualization.draw_geometries([self.pcd_out])
       





