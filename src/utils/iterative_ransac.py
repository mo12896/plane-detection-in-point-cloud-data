import open3d as o3d
import numpy as np
import os
import pyransac3d as pyrsc
import pickle

import system_setup as setup
from .utils import timer


class IterativeRANSAC:
    def __init__(self, data_dir: str, plane_size: int, thresh: float, debug: bool = False):
        self.data_dir = data_dir
        self.plane_size = plane_size
        self.thresh = thresh
        self.debug = debug
        self.points = None
        self.pcd_out = None
        self.file = None
        self.eqs = []
        # For debugging only!
        self.planes = []

    @timer
    def remove_planes(self, cloud, file: str):
        print("Iterative RANSAC...")
        self.points = np.asarray(cloud.points)
        self.file = file

        plane_counter = 0
        while True:
            # Find best plane using RANSAC
            plane = pyrsc.Plane()
            best_eq, best_inliers = plane.fit(self.points, self.thresh)

            # Only remove planes larger than size heuristic
            if len(best_inliers) > self.plane_size:
                plane_counter += 1
                self.eqs.append(best_eq)
                # Remove the best inliers from overall point cloud
                pcd_points = o3d.geometry.PointCloud()
                pcd_points.points = o3d.utility.Vector3dVector(self.points)
                self.pcd_out = pcd_points.select_by_index(best_inliers, invert=True)

                if self.debug:
                    plane = pcd_points.select_by_index(best_inliers)
                    self.planes.append(plane)

                self.points = np.asarray(self.pcd_out.points)
            else:
                break

        # Display plane removals during debugging
        if self.debug:
            print("Debugging...")
            o3d.visualization.draw_geometries(self.planes)

        # Retain color information for final point cloud
        if self.pcd_out:
            dists = cloud.compute_point_cloud_distance(self.pcd_out)
            dists = np.asarray(dists)
            ind = np.where(dists < 0.01)[0]
            self.pcd_out = cloud.select_by_index(ind)
        else:
            raise ValueError("No point cloud was generated!")

        # Store intermediate point cloud data
        data_path = os.path.join(self.data_dir, file)
        if not os.path.isfile(data_path):
            o3d.io.write_point_cloud(data_path, self.pcd_out)

        print(f"Removed {plane_counter} plane(s) from {file}")
        return self.pcd_out

    def display_final_pc(self):
        if self.pcd_out:
            o3d.visualization.draw_geometries([self.pcd_out])
        else:
            raise ValueError("You try to display an empty point cloud!")

    def store_best_eqs(self):
        if self.eqs:
            file = self.file.split('.')[0] + "_best_eqs"
            file_name = os.path.join(setup.LOGS_DIR, file)

            if os.path.isfile(file_name):
                os.remove(file_name)

            with open(file_name, 'wb') as fp:
                pickle.dump(self.eqs, fp)
        else:
            raise ValueError("No equations were extracted!")



