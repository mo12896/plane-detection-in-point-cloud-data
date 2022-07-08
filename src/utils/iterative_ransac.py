import open3d as o3d
import numpy as np
import os
from open3d import JVisualizer
import pyransac3d as pyrsc
from functools import lru_cache


from src.utils.utils import timer


class IterativeRANSAC:
    def __init__(self, data_dir: str, n_planes: int, thresh: float, debugging: bool = False):
        self.data_dir = data_dir
        self.n_planes = n_planes
        self.thresh = thresh
        self.debugging = debugging
        self.points = None
        self.pcd_out = None

    @timer
    def remove_planes(self, cloud: o3d.cpu.pybind.geometry.PointCloud, file: str):
        print("Iterative RANSAC...")
        self.points = np.asarray(cloud.points)

        for i in range(self.n_planes):
            # Find best plane using RANSAC
            plane = pyrsc.Plane()
            best_eq, best_inliers = plane.fit(self.points, self.thresh)

            # Remove best inliers from overall pointcloud
            pcd_points = o3d.geometry.PointCloud()
            pcd_points.points = o3d.utility.Vector3dVector(self.points)
            self.pcd_out = pcd_points.select_by_index(best_inliers, invert=True)

            # Display plane removal during debugging:
            if self.debugging:
                o3d.visualization.draw_geometries([self.pcd_out])

            self.points = np.asarray(self.pcd_out.points)

        data_path = os.path.join(self.data_dir, file)
        if not os.path.isfile(data_path):
            o3d.io.write_point_cloud(data_path, self.pcd_out)

        return self.pcd_out

    def display_final_pc(self):
        if self.pcd_out:
            o3d.visualization.draw_geometries([self.pcd_out])
        else:
            raise ValueError("You try to display an empty point cloud!")



