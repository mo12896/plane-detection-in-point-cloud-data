import open3d as o3d
import numpy as np
import os
from open3d import JVisualizer
import pyransac3d as pyrsc
from time import perf_counter

from src.system_setup import *


class DataLoader:
    def __init__(self, dir_path: str, voxel_size: float, debug: bool = False):
        self.dir_path = dir_path
        self.voxel_size = voxel_size
        self.debug = debug

    def load_data(self, filename: str):
        file_path = os.path.join(self.dir_path, filename)
        pcd = o3d.io.read_point_cloud(file_path)

        # Downsample large pointclouds, if specified
        if len(pcd.points) > 500000 and self.voxel_size:
            print(f"Downsampling {filename} of size {len(pcd.points)}...")
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            print(f"New size is: {len(pcd.points)}")

        if self.debug:
            o3d.visualization.draw_geometries([pcd])
            print(pcd)

        return pcd
