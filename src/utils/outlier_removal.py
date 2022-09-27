from __future__ import annotations
from abc import ABC, abstractmethod
import open3d as o3d
import os

from .utils import display_inlier_outlier, timer


class Context:
    """
    Implementation of the strategy pattern to call different algorithms
    for the outlier removal.
    """
    def __init__(self, strategy: OutlierRemoval):
        self._strategy = strategy

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: OutlierRemoval):
        self._strategy = strategy

    def run(self, data_dir, filename, debug: bool = False):
        cl, _ = self._strategy.remove_outliers(data_dir, filename)
        o3d.visualization.draw_geometries([cl])
        # TODO: Bug in implementation (malloc() error)
        #if debug:
        #    display_in_out(cl, id)


# Interface for outlier removal algorithms
class OutlierRemoval(ABC):
    def __init__(self, *args, **kwargs):
        super(OutlierRemoval, self).__init__(*args, **kwargs)

    @abstractmethod
    def remove_outliers(self, data_dir, file: str):
        pass


class StatisticalOutlierRemoval(OutlierRemoval):
    def __init__(self, out_dir: str, nb_neighbors: int, std_ratio: float):
        self.out_dir = out_dir
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio
        self.cl = None
        self.ind = None

    @timer
    def remove_outliers(self, data_dir, file: str):
        print("Statistical outlier removal...")
        try:
            file_path = os.path.join(data_dir, file)
            pcd = o3d.io.read_point_cloud(file_path)
        except:
            print(f"File {file} could not be loaded!")

        self.cl, self.ind = pcd.remove_statistical_outlier(nb_neighbors=self.nb_neighbors,
                                                           std_ratio=self.std_ratio)

        data_path = os.path.join(self.out_dir, file)
        try:
            if not os.path.isfile(data_path):
                o3d.io.write_point_cloud(data_path, self.cl)
        except:
            print(f"File {file} could not be written!")

        return self.cl, self.ind


class RadiusOutlierRemoval(OutlierRemoval):
    def __init__(self, out_dir: str, nb_points: int, radius: float):
        self.out_dir = out_dir
        self.nb_points = nb_points
        self.radius = radius
        self.cl = None
        self.ind = None

    @timer
    def remove_outliers(self, data_dir, file: str):
        print("Radius outlier removal...")
        try:
            file_path = os.path.join(data_dir, file)
            pcd = o3d.io.read_point_cloud(file_path)
        except:
            print(f"File {file} could not be loaded!")

        self.cl, self.ind = pcd.remove_radius_outlier(nb_points=self.nb_points,
                                                      radius=self.radius)

        data_path = os.path.join(self.out_dir, file)
        try:
            if not os.path.isfile(data_path):
                o3d.io.write_point_cloud(data_path, self.cl)
        except:
            print(f"File {file} could not be written!")      

        return self.cl, self.ind

