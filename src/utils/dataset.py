import open3d as o3d
import numpy as np
import os
from open3d import JVisualizer
import pyransac3d as pyrsc
from time import perf_counter

from src.system_setup import *


class DataLoader:
    def __init__(self, dir_path: str, debugging: bool = False):
        self.dir_path = dir_path
        self.debugging = debugging

    def load_data(self, filename: str):
        file_path = os.path.join(self.dir_path, filename)
        pcd = o3d.io.read_point_cloud(file_path)

        if self.debugging:
            o3d.visualization.draw_geometries([pcd])
            print(pcd)

        return pcd
