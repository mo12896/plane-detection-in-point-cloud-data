import open3d as o3d
import numpy as np
import os
from open3d import JVisualizer
import pyransac3d as pyrsc
from time import perf_counter

from src.system_setup import *
from utils import *

# TODO: either check in main or here!
pc_formats = ['xyz', 'xyzn', 'xyzrgb', 'pts', 'ply', 'pcd']


class DataLoader:
    def __init__(self, raw_data_path: str, debugging: bool = False):
        self.raw_data_path = raw_data_path
        self.debugging = debugging

    def load_data(self, filename: str):
        data_path = os.path.join(self.raw_data_path, filename)
        print(data_path)
        pcd = o3d.io.read_point_cloud(data_path)

        if self.debugging:
            o3d.visualization.draw_geometries([pcd])
            print(pcd)

        return pcd
