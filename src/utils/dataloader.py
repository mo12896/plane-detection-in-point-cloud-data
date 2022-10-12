"""Data Loader Interface and two concrete implementations"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import open3d as o3d
from open3d.cpu.pybind.geometry import PointCloud


class DataLoader(ABC):
    """Abstract Class of Data Loader"""

    @abstractmethod
    def load_data(self, filename: str) -> PointCloud:
        """Standard method to load data"""


class DataLoaderSTD(DataLoader):
    """Standard Data Loader"""

    def __init__(self, dir_path: Path):
        self.dir_path = dir_path

    def load_data(self, filename: str) -> PointCloud:
        """Load point cloud data from specified file

        Args:
            filename (str): file of point cloud data

        Returns:
            PointCloud: point cloud
        """
        try:
            file_path = self.dir_path / filename
            pcd = o3d.io.read_point_cloud(str(file_path))
        except Exception as exc:
            print(exc)

        return pcd


class DataLoaderDS(DataLoader):
    """Data Loader including a downsampling method"""

    def __init__(
        self, dir_path: Path, down_params: Dict[str, float], verbose: bool = False
    ):

        self.dir_path = dir_path
        self.large_pc = down_params["LARGE_PC"]
        self.voxel_size = down_params["VOXEL_SIZE"]
        self.voxel_step = down_params["VOXEL_STEP"]
        self.verbose = verbose

    def load_data(self, filename: str) -> PointCloud:
        """Load and downsample point cloud into memory

        Args:
            filename (str): file of raw point cloud data

        Returns:
            PointCloud: donwsampled point cloud
        """
        file_path = self.dir_path / filename
        pcd = o3d.io.read_point_cloud(str(file_path))

        # Downsample large point clouds into user-defined processing scope
        pcd_down = self._downsample_data(pcd, filename)

        if self.verbose:
            o3d.visualization.draw_geometries([pcd_down])
            print(pcd_down)

        return pcd_down

    def _downsample_data(self, cloud: PointCloud, filename: str) -> PointCloud:
        """Down sample point cloud data based on the definition of a large pointcloud
        and the speed of downsampling in the config file!

        Args:
            cloud (PointCloud): input point cloud
            filename (str): file of raw point cloud data

        Returns:
            PointCloud: donwsampled point cloud
        """
        try:
            while len(cloud.points) > self.large_pc and self.voxel_size:
                cloud = cloud.voxel_down_sample(voxel_size=self.voxel_size)
                self.voxel_size += self.voxel_step

                print(
                    f"'{filename}' has {len(cloud.points)} points after downsampling!"
                )
        except Exception as exc:
            print(exc)

        return cloud
