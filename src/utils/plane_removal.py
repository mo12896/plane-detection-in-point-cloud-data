import os
import numpy as np
import open3d as o3d
import pickle


class PlaneRemoval:
    def __init__(self, eqs_path: str, thresh: float):
        self.eqs_path = eqs_path
        self.thresh = thresh

    def remove_planes(self, cloud, file):
        print("Remove planes from original point cloud...")
        file_name = os.path.join(self.eqs_path, file)

        with open(file_name, 'rb') as fp:
            best_eqs = pickle.load(fp)
        print(best_eqs)

        points = np.asarray(cloud.points)
        print(points)
        #TODO: Implement point removal: https://github.com/leomariga/pyRANSAC-3D/blob/master/pyransac3d/plane.py


    def display_final_pc(self):
        if self.pcd_out:
            o3d.visualization.draw_geometries([self.pcd_out])
        else:
            raise ValueError("You try to display an empty point cloud!")





