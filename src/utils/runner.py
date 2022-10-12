from typing import Dict, Any
import os

from pathlib import Path

from .plane_detection import PlaneDetection
from .plane_removal import PlaneRemoval
from .dataloader import DataLoaderSTD
from .enums import PCFormats
from utils.outlier_removal import (
    Context,
    StatisticalOutlierRemoval,
    RadiusOutlierRemoval,
)


class Runner:
    def __init__(
        self,
        plane_detector: PlaneDetection,
        plane_remover: PlaneRemoval,
        int_data: DataLoaderSTD,
        final_data: Path,
        pc_formats: PCFormats,
        configs: Dict[str, float],
    ):
        self.plane_detector = plane_detector
        self.plane_remover = plane_remover
        self.int_data = int_data
        self.final_data = final_data
        self.pc_formats = pc_formats
        self.configs = configs

    def detect_plane(self, file: Any):
        """Detect planes in a single point cloud"""

        filename = os.fsdecode(file)
        if filename.endswith(tuple([enum.name.lower() for enum in self.pc_formats])):
            cloud = self.plane_detector.detect_planes(filename)
            self.plane_detector.store_best_eqs(filename)
            if self.configs["VERBOSE"]:
                self.plane_detector.display_pointcloud(cloud)

    def remove_plane(self, file: Any):
        """Remove planes from a single point cloud"""
        filename = os.fsdecode(file)
        if filename.endswith(tuple([enum.name.lower() for enum in self.pc_formats])):
            cloud = self.plane_remover.remove_planes(filename)

            if self.configs["OUT_REMOVAL"]["USE"]:
                if self.configs["VERBOSE"]:
                    self.plane_remover.display_pointcloud(cloud)
                context = Context(
                    eval(self.configs["OUT_REMOVAL"]["METHOD"])(
                        out_dir=self.final_data,
                        dataloader=self.int_data,
                        out_params=self.configs["OUT_REMOVAL"],
                    )
                )
                context.run(filename)
            else:
                self.plane_remover.display_pointcloud(cloud)
