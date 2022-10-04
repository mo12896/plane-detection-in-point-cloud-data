from __future__ import annotations
from abc import ABC, abstractmethod
import open3d as o3d
import os

from .utils import display_inlier_outlier, timer
from .dataset import DataLoader


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

    def run(self, filename, debug: bool = False):
        cl, _ = self._strategy.remove_outliers(filename)
        if not cl:
            raise ValueError("You try to display an empty point cloud!")

        o3d.visualization.draw_geometries([cl])
        # TODO: Bug in implementation (malloc() error)
        #if debug:
        #    display_in_out(cl, id)


# Interface for outlier removal algorithms
class OutlierRemoval(ABC):
    def __init__(self, *args, **kwargs):
        super(OutlierRemoval, self).__init__(*args, **kwargs)

    @abstractmethod
    def remove_outliers(self, filename: str):
        pass


class StatisticalOutlierRemoval(OutlierRemoval):
    def __init__(self,
                 out_dir: str, 
                 dataloader: DataLoader,
                 out_params: dict() = {}):

        self.out_dir = out_dir
        self.dataloader = dataloader
        self.nb_neighbors = out_params['NB_NEIGHBORS']
        self.std_ratio = out_params['STD_RATIO']

    @timer
    def remove_outliers(self, filename: str):
        print("Statistical outlier removal...")
        try:
            pcd = self.dataloader.load_data(filename)
        except:
            print(f"File {filename} could not be loaded!")

        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=self.nb_neighbors,
                                                           std_ratio=self.std_ratio)

        data_path = os.path.join(self.out_dir, filename)
        try:
            if not os.path.isfile(data_path):
                o3d.io.write_point_cloud(data_path, cl)
        except:
            print(f"File {filename} could not be written!")

        return cl, ind


class RadiusOutlierRemoval(OutlierRemoval):
    def __init__(self,
                 out_dir: str, 
                 dataloader: DataLoader,
                 out_params: dict() = {}):

        self.out_dir = out_dir
        self.dataloader = dataloader
        self.nb_points = out_params['NB_POINTS']
        self.radius = out_params['RADIUS']

    @timer
    def remove_outliers(self, filename: str):
        print("Radius outlier removal...")
        try:
            pcd = self.dataloader.load_data(filename)
        except:
            print(f"File {filename} could not be loaded!")

        cl, ind = pcd.remove_radius_outlier(nb_points=self.nb_points,
                                                      radius=self.radius)

        data_path = os.path.join(self.out_dir, filename)
        try:
            if not os.path.isfile(data_path):
                o3d.io.write_point_cloud(data_path, cl)
        except:
            print(f"File {filename} could not be written!")      

        return cl, ind

