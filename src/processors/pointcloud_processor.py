from abc import ABC, abstractmethod

from pathlib import Path
import open3d as o3d
from open3d.cpu.pybind.geometry import PointCloud


class PointCloudProcessor(ABC):
    """Abstract base class for processing point clouds"""

    def save_pcs(self, filename: str, out_dir: Path, pcd_out: PointCloud) -> None:
        """Saves point cloud data to a file"""
        data_path = out_dir / filename
        if not data_path.is_file():
            o3d.io.write_point_cloud(str(data_path), pcd_out)

    def display_pointcloud(self, pcd_out: PointCloud) -> None:
        """
        Displays the final point cloud
        :return:
        """
        if not pcd_out:
            raise ValueError("You try to display an empty point cloud!")

        o3d.visualization.draw_geometries([pcd_out])
