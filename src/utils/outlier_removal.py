import open3d as o3d
import os
from abc import ABC, abstractmethod

from src.utils.utils import display_inlier_outlier, timer


# Interface as abstract base class
class OutlierRemovalInterface(ABC):
    def __init__(self, *args, **kwargs):
        super(OutlierRemovalInterface, self).__init__(*args, **kwargs)

    @abstractmethod
    def remove_outliers(self, pcd: o3d.cpu.pybind.geometry.PointCloud, file: str):
        pass

    @abstractmethod
    def display_in_out(self):
        pass

    @abstractmethod
    def display_final_pc(self):
        pass


class StatisticalOutlierRemoval(OutlierRemovalInterface):
    def __init__(self, data_dir: str, nb_neighbors: int, std_ratio: float):
        self.data_dir = data_dir
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio
        self.cl = None
        self.ind = None

    @timer
    def remove_outliers(self, pcd: o3d.cpu.pybind.geometry.PointCloud, file: str):
        print("Statistical outlier removal")
        self.cl, self.ind = pcd.remove_statistical_outlier(nb_neighbors=self.nb_neighbors,
                                                           std_ratio=self.std_ratio)

        data_path = os.path.join(self.data_dir, file)
        if not os.path.isfile(data_path):
            o3d.io.write_point_cloud(data_path, self.cl)

        return self.cl, self.ind

    def display_in_out(self):
        display_inlier_outlier(self.cl, self.ind)

    def display_final_pc(self):
        o3d.visualization.draw_geometries([self.cl])


class RadiusOutlierRemoval(OutlierRemovalInterface):
    def __init__(self, data_dir: str, nb_points: int, radius: float):
        self.data_dir = data_dir
        self.nb_points = nb_points
        self.radius = radius
        self.cl = None
        self.ind = None

    @timer
    def remove_outliers(self, pcd: o3d.cpu.pybind.geometry.PointCloud, file: str):
        print("Radius oulier removal")
        self.cl, self.ind = pcd.remove_radius_outlier(nb_points=self.nb_points,
                                                      radius=self.radius)
        data_path = os.path.join(self.data_dir, file)
        if not os.path.isfile(data_path):
            o3d.io.write_point_cloud(data_path, self.cl)

        return self.cl, self.ind

    def display_in_out(self):
        display_inlier_outlier(self.cl, self.ind)

    def display_final_pc(self):
        o3d.visualization.draw_geometries([self.cl])
