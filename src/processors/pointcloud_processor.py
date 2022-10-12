from abc import ABC, abstractmethod

from pathlib import Path
import open3d as o3d
from open3d.cpu.pybind.geometry import PointCloud


class PointCloudProcessor(ABC):
    """Abstract base class for processing point clouds"""

    def save_pcs(self, filename: str, out_dir: Path, pcd_out: PointCloud) -> None:
        """Saves point cloud data to a file

        Args:
            filename (str): _description_
            out_dir (Path): _description_
            pcd_out (PointCloud): _description_
        """
        try:
            data_path = out_dir / filename
            if not data_path.is_file():
                o3d.io.write_point_cloud(str(data_path), pcd_out)
        except Exception as exc:
            print(exc)

    def display_pointcloud(self, pcd_out: PointCloud) -> None:
        """Displays the final point cloud

        Args:
            pcd_out (PointCloud): point cloud to display

        Raises:
            ValueError: You try to display an empty point cloud!
        """
        try:
            if not pcd_out:
                raise ValueError("You try to display an empty point cloud!")

            o3d.visualization.draw_geometries([pcd_out])
        except Exception as exc:
            print(exc)
