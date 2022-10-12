"""Outlier Removal Stragies"""
from __future__ import annotations
from abc import abstractmethod
from pathlib import Path
from typing import Tuple, List, Dict

import open3d as o3d
from open3d.cpu.pybind.geometry import PointCloud

from utils.utils import timer
from utils.dataloader import DataLoader
from .pointcloud_processor import PointCloudProcessor


class Context:
    """
    Implementation of the strategy pattern to call different algorithms
    for the outlier removal.
    """

    def __init__(self, strategy: OutlierRemoval):
        self._strategy = strategy

    @property
    def strategy(self):
        """Get the strategy"""
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: OutlierRemoval):
        """Set the strategy

        Args:
            strategy (OutlierRemoval): Outlier Removal object
        """
        self._strategy = strategy

    def run(self, filename: str) -> None:
        """Run the outlier removal strategy

        Args:
            filename (str): path to intermediate point cloud file

        Raises:
            ValueError: You try to display an empty point cloud!
        """
        cl, _ = self._strategy.remove_outliers(filename)
        if not cl:
            raise ValueError("You try to display an empty point cloud!")

        self._strategy.display_pointcloud(cl)
        # TODO: Bug in implementation (malloc() error)
        # if debug:
        #    display_in_out(cl, id)


# Interface for outlier removal algorithms
class OutlierRemoval(PointCloudProcessor):
    """Abstract Class of Outlier Removal"""

    @abstractmethod
    def remove_outliers(self, filename: str) -> Tuple[PointCloud, List[int]]:
        """Remove Outlier from a PointCloud"""


class StatisticalOutlierRemoval(OutlierRemoval):
    """Removes outliers using statistical analysis"""

    def __init__(
        self, out_dir: Path, dataloader: DataLoader, out_params: Dict[str, float]
    ):

        self.out_dir = out_dir
        self.dataloader = dataloader
        self.nb_neighbors = out_params["NB_NEIGHBORS"]
        self.std_ratio = out_params["STD_RATIO"]

    @timer
    def remove_outliers(self, filename: str) -> Tuple[PointCloud, List[int]]:
        """Removes outliers based on statistical analysis

        Args:
            filename (str): path to intermediate point cloud file

        Returns:
            Tuple[PointCloud, List[int]]: final point cloud
        """
        print("Statistical outlier removal...")

        pcd: PointCloud = self.dataloader.load_data(filename)

        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=self.nb_neighbors, std_ratio=self.std_ratio
        )

        self.save_pcs(filename, self.out_dir, cl)

        return (cl, ind)


class RadiusOutlierRemoval(OutlierRemoval):
    """Removes outliers within a provided radius"""

    def __init__(
        self, out_dir: Path, dataloader: DataLoader, out_params: Dict[str, float]
    ):

        self.out_dir = out_dir
        self.dataloader = dataloader
        self.nb_points = out_params["NB_POINTS"]
        self.radius = out_params["RADIUS"]

    @timer
    def remove_outliers(self, filename: str) -> Tuple[PointCloud, List[int]]:
        """Removes outliers within a provided radius

        Args:
            filename (str): path to intermediate point cloud file

        Returns:
            Tuple[PointCloud, List[int]]: final point cloud
        """
        print("Radius outlier removal...")

        pcd: PointCloud = self.dataloader.load_data(filename)

        cl, ind = pcd.remove_radius_outlier(
            nb_points=self.nb_points, radius=self.radius
        )

        self.save_pcs(filename, self.out_dir, cl)

        return (cl, ind)
