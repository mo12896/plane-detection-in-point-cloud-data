import open3d as o3d
import numpy as np
import os
from open3d import JVisualizer
import pyransac3d as pyrsc
from time import perf_counter

from utils import *


class IterativeRANSAC:
    def __init__(self, cloud, n_planes: int, thresh: float, debugging: bool = False):
        self.cloud = cloud
        self.n_planes = n_planes
        self.thresh = thresh
        self.debugging = debugging
        self.points = None
        self.pcd_out = None

    def remove_planes(self):
        print("Iterative RANSAC...")
        self.points = np.asarray(self.cloud.points)

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

        print("Finished!")
        return self.pcd_out

    def display_final_pc(self):
        if self.pcd_out:
            o3d.visualization.draw_geometries([self.pcd_out])
        else:
            raise ValueError("You try to display an empty point cloud!")



