import os
import numpy as np
import open3d as o3d
import pickle

from .utils import remove_by_indices, timer


class PlaneRemoval:
    def __init__(self, data_dir: str, eqs_path: str, thresh: float, store: bool = True):
        self.data_dir = data_dir
        self.eqs_path = eqs_path
        self.thresh = thresh
        self.store = store
        self.pcd_out = None

    @timer
    def remove_planes(self, cloud, file):
        print("Remove planes from original point cloud...")
        file_name = os.path.join(self.eqs_path, file)

        with open(file_name, 'rb') as fp:
            best_eqs = pickle.load(fp)

        pts = np.asarray(cloud.points)
        for plane_eq in best_eqs:
            dist_pts = (plane_eq[0] * pts[:, 0] + plane_eq[1] * pts[:, 1] + plane_eq[2] * pts[:, 2] + plane_eq[3]
                      ) / np.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)
            inliers = np.where(np.abs(dist_pts) <= self.thresh)[0].tolist()
            pts = remove_by_indices(pts, inliers)

        self.pcd_out = o3d.geometry.PointCloud()
        self.pcd_out.points = o3d.utility.Vector3dVector(pts)

        if self.pcd_out:
            dists = cloud.compute_point_cloud_distance(self.pcd_out)
            dists = np.asarray(dists)
            ind = np.where(dists < 0.01)[0]
            self.pcd_out = cloud.select_by_index(ind)
        else:
            raise ValueError("No point cloud was generated!")

        # Store intermediate point cloud data
        if self.store:
            data_path = os.path.join(self.data_dir, file)
            if not os.path.isfile(data_path):
                o3d.io.write_point_cloud(data_path, self.pcd_out)

        return self.pcd_out

    def display_final_pc(self):
        if self.pcd_out:
            o3d.visualization.draw_geometries([self.pcd_out])
        else:
            raise ValueError("You try to display an empty point cloud!")





