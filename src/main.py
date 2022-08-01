import yaml
import os

from multiprocessing import Pool
from argparse import ArgumentParser
import open3d as o3d

import system_setup as setup
from utils.iterative_ransac import IterativeRANSAC
from utils.dataset import DataLoader
from utils.outlier_removal import StatisticalOutlierRemoval
from utils.plane_removal import PlaneRemoval


# Readable point cloud formats for Open3D
pc_formats = ('xyz', 'xyzn', 'xyzrgb', 'pts', 'ply', 'pcd')

"""Define the config file to be used"""
argparser = ArgumentParser(description="Plane Removal")
argparser.add_argument('--config',
                       type=str,
                       default="config",
                       help="File for the iterative RANSAC configuration. Supported:\n"
                            "- config")
argparser.add_argument('--clean',
                       type=bool,
                       default=True,
                       help="Remove intermediate and final point cloud files from previous calls.")

args = argparser.parse_args()

"""Set up variables"""
config_file = args.config + '.yaml'
config_path = os.path.join(setup.CONFIG_DIR, config_file)

raw_data_dir = setup.RAW_DATA_DIR
int_data_dir = setup.INT_DATA_DIR
final_data_dir = setup.FINAL_DATA_DIR
directory = os.fsencode(raw_data_dir)

"""Parse the config.yaml into a python dict"""
with open(config_path, 'r') as stream:
    try:
        configs = yaml.safe_load(stream)
        print("Loaded config file!")
    except yaml.YAMLError as exc:
        print(exc)

"""Clean up the intermediate and final directories"""
if args.clean:
    for f in os.listdir(int_data_dir):
        os.remove(os.path.join(int_data_dir, f))
    for f in os.listdir(final_data_dir):
        os.remove(os.path.join(final_data_dir, f))

"""Instantiate relevant objects"""
data = DataLoader(raw_data_dir,
                  configs['HEURISTICS']['LARGE_PC'],
                  configs['HEURISTICS']['VOXEL_SIZE'],
                  configs['HEURISTICS']['VOXEL_STEP'],
                  configs['DEBUG'])

ransac = IterativeRANSAC(int_data_dir,
                         configs['HEURISTICS']['PLANE_SIZE'],
                         configs['THRESH'],
                         configs['DEBUG'])

rm_outlier = StatisticalOutlierRemoval(final_data_dir,
                                       configs['OUT_REMOVAL']['STATS']['NB_NEIGHBORS'],
                                       configs['OUT_REMOVAL']['STATS']['STD_RATIO'])


# Function to process a single point cloud data file
def process_single_pc(file):
    """Process single point cloud data file"""

    filename = os.fsdecode(file)
    if filename.endswith(pc_formats):
        pcd = data.load_data(filename)
        pcd_out = ransac.remove_planes(pcd, filename)
        ransac.store_best_eqs()
        if configs['OUT_REMOVAL']['USE']:
            rm_outlier.remove_outliers(pcd_out, filename)
            rm_outlier.display_final_pc()
        else:
            ransac.display_final_pc()


def main():
    # Multiprocessing interface
    pool = Pool()
    pool.map(process_single_pc, os.listdir(directory))
    pool.close()
    pool.join()

    # plane removal in original point cloud data
    # TODO: Implement in multiprocessing manner
    if configs['RAW_REMOVAL']['USE']:
        cloud = PlaneRemoval(
            setup.LOGS_DIR,
            configs['RAW_REMOVAL']['THRESH']
        )

        for file in os.listdir(raw_data_dir):
            filename = os.fsdecode(file)
            file_path = os.path.join(raw_data_dir, filename)
            pcd = o3d.io.read_point_cloud(file_path)
            eqs = filename.split('.')[0] + "_best_eqs"
            #final_cloud = cloud.remove_planes(pcd, eqs)
            cloud.remove_planes(pcd, eqs)


if __name__ == '__main__':
    main()