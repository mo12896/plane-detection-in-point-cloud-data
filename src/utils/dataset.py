import open3d as o3d
import os
from abc import ABC, abstractmethod

from .utils import timer


class DataLoader(ABC):

    @abstractmethod
    def load_data(self, file: str):
        pass


class DataLoader_STD(DataLoader):
    def __init__(self, dir_path: str):
        self.dir_path = dir_path
   

    def load_data(self, file: str):
        file_path = os.path.join(self.dir_path, file)
        pcd = o3d.io.read_point_cloud(file_path)

        return pcd
    

class DataLoader_DS(DataLoader):
    def __init__(self, dir_path: str, large_pc: int, voxel_size: float, voxel_step: float, verbose: bool = False):
        self.dir_path = dir_path
        self.large_pc = large_pc
        self.voxel_size = voxel_size
        self.voxel_step = voxel_step
        self.verbose = verbose

    def load_data(self, file: str):
        file_path = os.path.join(self.dir_path, file)
        pcd = o3d.io.read_point_cloud(file_path)

        # Downsample large point clouds into user-defined processing scope
        pcd_down = self.downsample_data(pcd, file)

        if self.verbose:
            o3d.visualization.draw_geometries([pcd_down])
            print(pcd_down)

        return pcd_down

    def downsample_data(self, cloud, file: str):
        """Down sample poin cloud data based on the definition of a large pointcloud
        and the speed of downsampling in the config file!"""

        while len(cloud.points) > self.large_pc and self.voxel_size:
            cloud = cloud.voxel_down_sample(voxel_size=self.voxel_size)
            self.voxel_size = self.voxel_size + self.voxel_step

            print(f"'{file}' has {len(cloud.points)} points after downsampling!")

        return cloud


