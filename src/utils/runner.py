from typing import Dict, Any
import os

from processors.plane_detection import PlaneDetection
from processors.plane_removal import PlaneRemoval
from processors.outlier_removal import (
    Context,
    StatisticalOutlierRemoval,
    RadiusOutlierRemoval,
)
from .enums import PCFormats


class Runner:
    """Runner class for running the plane detection and removal"""

    def __init__(
        self,
        plane_detector: PlaneDetection,
        plane_remover: PlaneRemoval,
        out_remover: Context,
        pc_formats: PCFormats,
        configs: Dict[str, float],
    ):
        self.plane_detector = plane_detector
        self.plane_remover = plane_remover
        self.out_remover = out_remover
        self.pc_formats = pc_formats
        self.configs = configs

    def detect_plane(self, file: Any):
        """Detect planes in a single point cloud

        Args:
            file (Any): file in raw directory
        """

        filename = os.fsdecode(file)
        if filename.endswith(tuple([enum.name.lower() for enum in self.pc_formats])):
            cloud = self.plane_detector.detect_planes(filename)
            # self.plane_detector.store_best_eqs(filename)
            if self.configs["VERBOSE"]:
                self.plane_detector.display_pointcloud(cloud)

    def remove_plane(self, file: Any):
        """Remove planes from a single point cloud

        Args:
            file (Any): file in raw directory
        """
        filename = os.fsdecode(file)
        if filename.endswith(tuple([enum.name.lower() for enum in self.pc_formats])):
            cloud = self.plane_remover.remove_planes(filename)

            if self.configs["OUT_REMOVAL"]["USE"]:
                if self.configs["VERBOSE"]:
                    self.plane_remover.display_pointcloud(cloud)
                self.out_remover.run(filename)
            else:
                self.plane_remover.display_pointcloud(cloud)
