import open3d as o3d
import numpy as np
import os
from open3d import JVisualizer
import pyransac3d as pyrsc
from time import perf_counter

from utils import *


class DataLoader:
    def __init__(self, cloud_path: str):
        self.cloud_path = cloud_path

    def load_data(self):
        return NotImplementedError