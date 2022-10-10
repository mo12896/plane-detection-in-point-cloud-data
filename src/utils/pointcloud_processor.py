from abc import ABC

import open3d as o3d


class PointCloudProcessor(ABC):
    """Abstract base class for processing point clouds"""

    def display_pointcloud(self, pcd_out) -> None:
        """
        Displays the final point cloud
        :return:
        """
        if not pcd_out:
            raise ValueError("You try to display an empty point cloud!")

        o3d.visualization.draw_geometries([pcd_out])
