import open3d as o3d
import os


class DataLoader:
    def __init__(self, dir_path: str, large_pc: int, voxel_size: float, debug: bool = False):
        self.dir_path = dir_path
        self.large_pc = large_pc
        self.voxel_size = voxel_size
        self.debug = debug

    def load_data(self, filename: str):
        file_path = os.path.join(self.dir_path, filename)
        pcd = o3d.io.read_point_cloud(file_path)

        # Downsample large pointclouds into processing scope
        while len(pcd.points) > self.large_pc and self.voxel_size:
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            print(f"{filename} has {len(pcd.points)} points after downsampling!")

        # if self.debug:
        #    o3d.visualization.draw_geometries([pcd])
        #    print(pcd)

        return pcd
