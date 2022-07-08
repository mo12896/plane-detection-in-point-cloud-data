import open3d as o3d
import numpy as np
import os
from abc import ABC, abstractmethod

from open3d import JVisualizer

from utils import *


class OutlierRemovalInterface(ABC):
    def __init__(self, pcd, *args, **kwargs):
        super(OutlierRemovalInterface, self).__init__(pcd, *args, **kwargs)

    @abstractmethod
    def remove_outliers(self):
        pass

    @abstractmethod
    def display_in_out(self):
        pass

    @abstractmethod
    def display_final_pc(self):
        pass


class StatisticalOutlierRemoval(OutlierRemovalInterface):
    def __init__(self, pcd, nb_neighbors: int, std_ratio: float):
        self.pcd = pcd
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio
        self.cl = None
        self.ind = None

    def remove_outliers(self):
        print("Statistical oulier removal")
        self.cl, self.ind = self.pcd.remove_statistical_outlier(nb_neighbors=self.nb_neighbors,
                                                                std_ratio=self.std_ratio)
        print("Finished!")
        return self.cl, self.ind

    def display_in_out(self):
        display_inlier_outlier(self.cl, self.ind)

    def display_final_pc(self):
        o3d.visualization.draw_geometries([self.cl])


class RadiusOutlierRemoval(OutlierRemovalInterface):
    def __init__(self, pcd, nb_points: int, radius: float):
        self.pcd = pcd
        self.nb_points = nb_points
        self.radius = radius
        self.cl = None
        self.ind = None

    def remove_outliers(self):
        print("Radius oulier removal")
        self.cl, self.ind = self.pcd.remove_radius_outlier(nb_points=self.nb_points,
                                                           radius=self.radius)
        print("Finished!")
        return self.cl, self.ind

    def display_in_out(self):
            display_inlier_outlier(self.cl, self.ind)

    def display_final_pc(self):
        o3d.visualization.draw_geometries([self.cl])
