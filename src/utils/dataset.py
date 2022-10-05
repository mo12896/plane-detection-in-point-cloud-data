"""Data Loader Interface and two concrete implementations"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import open3d as o3d
from open3d.cpu.pybind.geometry import PointCloud


class DataLoader(ABC):
    """Data Loader Interface"""

    @abstractmethod
    def load_data(self, filename: str) -> PointCloud:
        """Standard method to load data"""


class DataLoaderSTD(DataLoader):
    """Standard Data Loader"""
    def __init__(self, dir_path: Path):
        self.dir_path = dir_path

    def load_data(self, filename: str) -> PointCloud:
        file_path = self.dir_path / filename
        pcd = o3d.io.read_point_cloud(str(file_path))
        
        return pcd
    

class DataLoaderDS(DataLoader):
    """Data Loader including a downsampling method"""
    def __init__(self,
                 dir_path: Path,
                 down_params: Dict[str, float],
                 verbose: bool = False):
                 
        self.dir_path = dir_path
        self.large_pc = down_params['LARGE_PC']
        self.voxel_size = down_params['VOXEL_SIZE']
        self.voxel_step = down_params['VOXEL_STEP']
        self.verbose = verbose

    def load_data(self, filename: str) -> PointCloud:
        file_path = self.dir_path / filename
        pcd = o3d.io.read_point_cloud(str(file_path))

        # Downsample large point clouds into user-defined processing scope
        pcd_down = self._downsample_data(pcd, filename)

        if self.verbose:
            o3d.visualization.draw_geometries([pcd_down])
            print(pcd_down)

        return pcd_down

    def _downsample_data(self, cloud, filename: str) -> PointCloud:
        """Down sample point cloud data based on the definition of a large pointcloud
        and the speed of downsampling in the config file!"""

        while len(cloud.points) > self.large_pc and self.voxel_size:
            cloud = cloud.voxel_down_sample(voxel_size=self.voxel_size)
            self.voxel_size += self.voxel_step

            print(f"'{filename}' has {len(cloud.points)} points after downsampling!")

        return cloud


